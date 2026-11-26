#!/usr/bin/env python3
"""Publish unmodified CSM pose and scan stamp as odometry/TF; twist is derived.

The initial pose is retained until a second increasing-stamp observation arrives.
An explicit fixed SE2 alignment converts an authenticated candidate pose source
to odom without changing source stamps, smoothing, or replacing measurements.
The default control velocity remains CSM SE(2). An explicit receive-only CAN
option changes only control_odom vx/w; raw odometry and TF remain measured CSM.
"""
import argparse
import hashlib
from pathlib import Path
import json
import copy
import math
import threading
import time
import rospy
import tf2_ros
from geometry_msgs.msg import PoseStamped, TransformStamped
from nav_msgs.msg import Odometry
from std_msgs.msg import String

EXPECTED_CALLER = '/nso_csm/matcher'
SOURCE_FRAME = 'odom'
CHILD_FRAME = 'base_link'


def planar_sample(msg):
    p, q = msg.pose.position, msg.pose.orientation
    values = (p.x, p.y, p.z, q.x, q.y, q.z, q.w)
    if not all(math.isfinite(v) for v in values) or abs(sum(v*v for v in (q.x,q.y,q.z,q.w))-1.0) > 1e-3:
        raise ValueError('invalid pose')
    yaw = math.atan2(2*(q.w*q.z+q.x*q.y), 1-2*(q.y*q.y+q.z*q.z))
    return p.x, p.y, yaw


def derived_twist(previous, current, dt):
    """Constant body-frame SE(2) twist from two measured poses, m/s and rad/s."""
    if not math.isfinite(dt) or dt <= 0:
        raise ValueError('dt must be finite and positive')
    x0, y0, a0 = previous
    x1, y1, a1 = current
    da = math.atan2(math.sin(a1-a0), math.cos(a1-a0))
    dx, dy = x1-x0, y1-y0
    bx, by = math.cos(a0)*dx+math.sin(a0)*dy, -math.sin(a0)*dx+math.cos(a0)*dy
    if abs(da) < 1e-8:
        return bx/dt, by/dt, da/dt
    s, c = math.sin(da)/da, (1-math.cos(da))/da
    det = s*s+c*c
    return (s*bx+c*by)/det/dt, (-c*bx+s*by)/det/dt, da/dt


def finite_se2(value):
    if not isinstance(value, (list, tuple)) or len(value) != 3 or any(
        not isinstance(v, (int, float)) or isinstance(v, bool) or not math.isfinite(v) for v in value):
        raise ValueError('finite SE2 triple required')
    return tuple(float(v) for v in value)


def transform_sample(transform, sample):
    tx, ty, angle = transform
    x, y, yaw = sample
    c, sn = math.cos(angle), math.sin(angle)
    result = (tx+c*x-sn*y, ty+sn*x+c*y,
              math.atan2(math.sin(yaw+angle), math.cos(yaw+angle)))
    if not all(math.isfinite(v) for v in result):
        raise ValueError('nonfinite transformed pose')
    return result


def load_alignment(path, pose_topic, expected_caller, source_frame):
    if path is None:
        if (pose_topic, expected_caller, source_frame) != ('/nso_csm/pose_stamped', EXPECTED_CALLER, SOURCE_FRAME):
            raise ValueError('explicit alignment required for a different pose source')
        return None, None
    raw = Path(path).read_bytes()
    data = json.loads(raw)
    if not isinstance(data, dict):
        raise ValueError('alignment object required')
    expected = dict(schema='csm_fixed_se2_alignment/1', source_topic=pose_topic,
                    expected_caller=expected_caller, source_frame=source_frame, target_frame='odom')
    for key, value in expected.items():
        if data.get(key) != value:
            raise ValueError('alignment binding mismatch: '+key)
    transform = finite_se2(data['odom_from_source'])
    anchor = data['anchor']
    if not isinstance(anchor, dict):
        raise ValueError('alignment anchor object required')
    stamp = anchor['source_stamp_ns']
    if type(stamp) is not int or stamp <= 0 or type(anchor['odom_stamp_ns']) is not int or anchor['odom_stamp_ns'] != stamp:
        raise ValueError('alignment anchors require identical positive original scan stamps')
    source_pose = finite_se2(anchor['source_pose'])
    target_pose = finite_se2(anchor['odom_pose'])
    mapped = transform_sample(transform, source_pose)
    position_error = math.hypot(mapped[0]-target_pose[0], mapped[1]-target_pose[1])
    angle_error = abs(math.atan2(math.sin(mapped[2]-target_pose[2]), math.cos(mapped[2]-target_pose[2])))
    if position_error > 1e-7 or angle_error > 1e-7:
        raise ValueError('fixed alignment does not map same-stamp anchor poses')
    receipt = dict(**expected, path=str(Path(path).resolve()), sha256=hashlib.sha256(raw).hexdigest(),
                   odom_from_source=list(transform), anchor=copy.deepcopy(anchor),
                   anchor_position_error_m=position_error, anchor_angle_error_rad=angle_error,
                   operation='fixed rigid SE2 coordinates only; no smoothing/fallback/stamp replacement',
                   physical_accuracy_verified=False)
    return transform, receipt



def pose_source_parameters(pose_topic, expected_caller, source_frame, receipt):
    def xmlrpc_value(value, key=''):
        if isinstance(value, dict):
            return {k:xmlrpc_value(v, k) for k,v in value.items()}
        if isinstance(value, (tuple, list)):
            return [xmlrpc_value(v) for v in value]
        if type(value) is int and (key.endswith('_ns') or not -(2**31) <= value < 2**31):
            return str(value)
        return value
    applied = receipt is not None
    return dict(profile='fixed_se2_pose_source_diagnostic_v1' if applied else 'original_csm_pose_source_v1',
                pose_topic=pose_topic, expected_caller=expected_caller, source_frame=source_frame,
                target_frame='odom', fixed_coordinate_transform_applied=applied,
                raw_registration_preserved=True, source_stamp_preserved=True,
                nanosecond_stamp_encoding='decimal_string_in_ROS_parameters_only',
                alignment=xmlrpc_value(receipt) if applied else {})


def aligned_pose(pose, transform):
    result = copy.deepcopy(pose)
    tx, ty, yaw = transform
    c, sn = math.cos(yaw), math.sin(yaw)
    result.position.x = tx+c*pose.position.x-sn*pose.position.y
    result.position.y = ty+sn*pose.position.x+c*pose.position.y
    # Left multiply by a constant world-Z rotation; preserve measured Z/tilt.
    q = pose.orientation
    c, sn = math.cos(yaw/2), math.sin(yaw/2)
    result.orientation.x = c*q.x-sn*q.y
    result.orientation.y = sn*q.x+c*q.y
    result.orientation.z = c*q.z+sn*q.w
    result.orientation.w = c*q.w-sn*q.z
    return result


class CsmOdometryBridge:
    def __init__(self, control_velocity_source='csm_se2', can_feedback=None, *,
                 pose_topic='/nso_csm/pose_stamped', expected_caller=EXPECTED_CALLER,
                 source_frame=SOURCE_FRAME, alignment_json=None):
        if control_velocity_source not in ('csm_se2', 'can_feedback'):
            raise ValueError('unknown control velocity source')
        if control_velocity_source == 'can_feedback' and can_feedback is None:
            raise ValueError('can_feedback source is required for opt-in mode')
        if not pose_topic.startswith('/') or not expected_caller.startswith('/') or not source_frame:
            raise ValueError('explicit absolute source topic/caller and nonempty frame required')
        self.pose_topic, self.expected_caller, self.source_frame = pose_topic, expected_caller, source_frame
        self.alignment, self.alignment_receipt = load_alignment(
            alignment_json, pose_topic, expected_caller, source_frame)
        self.control_velocity_source = control_velocity_source
        self.can_feedback = can_feedback
        self.failed = None
        self.can_control_started = False
        self.lock = threading.Lock()
        self.previous = None
        self.pub = rospy.Publisher('/nso_csm/odom', Odometry, queue_size=10)
        self.control_pub = rospy.Publisher('/nso_csm/control_odom', Odometry, queue_size=10)
        self.provenance_pub = None
        if control_velocity_source == 'can_feedback':
            self.provenance_message_type = String
            self.provenance_pub = rospy.Publisher('/nso_csm/control_velocity_provenance', String, queue_size=100)
        self.pose_source_pub = None
        if self.alignment is not None:
            self.pose_source_pub = rospy.Publisher('/nso_csm/pose_source_provenance', String, queue_size=100)
        self.tf = tf2_ros.TransformBroadcaster()
        self.sub = rospy.Subscriber(self.pose_topic, PoseStamped, self.callback, queue_size=10, tcp_nodelay=True)

    def fail(self, reason):
        # Caller holds the bridge lock: no message can publish after this latch.
        if self.failed is None:
            self.failed = reason
            rospy.logerr('CSM control CAN source failed: %s', reason)
            rospy.signal_shutdown(reason)

    def check_can_health(self):
        if self.control_velocity_source == 'can_feedback':
            with self.lock:
                reason = self.can_feedback.health_reason()
                if reason:
                    self.fail(reason)

    def callback(self, msg):
        if msg._connection_header.get('callerid') != self.expected_caller or msg.header.frame_id != self.source_frame:
            rospy.logwarn_throttle(5, 'Rejecting CSM pose with unexpected caller or frame')
            return
        try:
            raw_sample = planar_sample(msg)
            sample = transform_sample(self.alignment, raw_sample) if self.alignment is not None else raw_sample
            published_pose = aligned_pose(msg.pose, self.alignment) if self.alignment is not None else msg.pose
        except (TypeError, ValueError):
            rospy.logwarn_throttle(5, 'Rejecting invalid CSM pose')
            return
        stamp_ns = msg.header.stamp.to_nsec()
        if stamp_ns <= 0:
            return
        with self.lock:
            if self.failed is not None:
                return
            if self.previous is None:
                self.previous = (stamp_ns, sample)
                return
            old_ns, old = self.previous
            if stamp_ns <= old_ns:
                return
            vx, vy, w = derived_twist(old, sample, (stamp_ns-old_ns)*1e-9)
            if not all(math.isfinite(v) for v in (vx,vy,w)):
                return
            self.previous = (stamp_ns, sample)
            odom = Odometry()
            odom.header.stamp = msg.header.stamp
            odom.header.frame_id = SOURCE_FRAME
            odom.child_frame_id = CHILD_FRAME
            odom.pose.pose = published_pose
            odom.twist.twist.linear.x, odom.twist.twist.linear.y = vx, vy
            odom.twist.twist.angular.z = w
            # All-zero covariance is unspecified, not a measured accuracy claim.
            transform = TransformStamped()
            transform.header.stamp = msg.header.stamp
            transform.header.frame_id = SOURCE_FRAME
            transform.child_frame_id = CHILD_FRAME
            transform.transform.translation.x = published_pose.position.x
            transform.transform.translation.y = published_pose.position.y
            transform.transform.translation.z = published_pose.position.z
            transform.transform.rotation = published_pose.orientation
            # Controller model projection for a nonholonomic base, not a raw measurement.
            control_odom = copy.deepcopy(odom)
            control_odom.twist.twist.linear.y = 0.0
            provenance = None
            if self.control_velocity_source == 'can_feedback':
                try:
                    feedback = self.can_feedback.for_scan(stamp_ns)
                except ValueError as exc:
                    self.fail(str(exc))
                    return
                if feedback is None:
                    if not self.can_control_started and self.can_feedback.in_startup_grace():
                        return
                    self.fail('no fresh CAN velocity at or before CSM scan stamp')
                    return
                control_odom.twist.twist.linear.x = feedback.vx
                control_odom.twist.twist.angular.z = feedback.w
                self.can_control_started = True
                provenance = dict(schema='csm_control_velocity_provenance/1',
                                  control_velocity_source='can_feedback',
                                  csm_source_stamp_ns=stamp_ns,
                                  can_pair=feedback.provenance(),
                                  can_source=self.can_feedback.source_receipt(),
                                  raw_csm_twist=dict(vx=vx, vy=vy, w=w),
                                  control_twist=dict(vx=feedback.vx, vy=0., w=feedback.w),
                                  raw_pose_stamp_tf_unchanged=True)
                if self.alignment is not None:
                    provenance.update(raw_pose_stamp_tf_unchanged=False, source_pose_stamp_unchanged=True,
                                      fixed_coordinate_transform_applied=True, raw_registration_preserved=True,
                                      pose_alignment=self.alignment_receipt, source_topic=self.pose_topic,
                                      source_caller=self.expected_caller, source_frame=self.source_frame)
                provenance = self.provenance_message_type(data=json.dumps(provenance, sort_keys=True))
            self.pub.publish(odom)
            self.control_pub.publish(control_odom)
            self.tf.sendTransform(transform)
            if provenance is not None:
                self.provenance_pub.publish(provenance)
            if self.pose_source_pub is not None:
                self.pose_source_pub.publish(String(data=json.dumps(dict(
                    schema='csm_pose_source_provenance/1', source_topic=self.pose_topic,
                    source_caller=msg._connection_header.get('callerid'), source_frame=msg.header.frame_id,
                    source_stamp_ns=stamp_ns, output_frame='odom', output_stamp_ns=stamp_ns,
                    raw_source_sample=list(raw_sample), aligned_output_sample=list(sample),
                    fixed_coordinate_transform_applied=True, raw_registration_preserved=True,
                    alignment=self.alignment_receipt), sort_keys=True)))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--control-velocity-source', choices=('csm_se2', 'can_feedback'), default='csm_se2')
    parser.add_argument('--can-interface', default='can0')
    parser.add_argument('--pose-topic', default='/nso_csm/pose_stamped')
    parser.add_argument('--expected-caller', default=EXPECTED_CALLER)
    parser.add_argument('--source-frame', default=SOURCE_FRAME)
    parser.add_argument('--alignment-json')
    args = parser.parse_args(rospy.myargv()[1:])
    # Validate immutable alignment before constructing/opening any CAN source.
    load_alignment(args.alignment_json, args.pose_topic, args.expected_caller, args.source_frame)
    rospy.init_node('nso_csm_odometry_bridge')
    rospy.set_param('~control_velocity_source', args.control_velocity_source)
    source = None
    bridge = None
    try:
        if args.control_velocity_source == 'can_feedback':
            from can_control_feedback import CanVelocityFeedback
            source = CanVelocityFeedback(args.can_interface)
            source.start()
        bridge = CsmOdometryBridge(args.control_velocity_source, source,
                                   pose_topic=args.pose_topic, expected_caller=args.expected_caller,
                                   source_frame=args.source_frame, alignment_json=args.alignment_json)
        rospy.set_param('~pose_source', pose_source_parameters(
            args.pose_topic, args.expected_caller, args.source_frame, bridge.alignment_receipt))
        if source is None:
            rospy.spin()
        else:
            # Wall/monotonic watchdog also detects CAN loss without CSM callbacks.
            while not rospy.is_shutdown():
                bridge.check_can_health()
                time.sleep(.01)
    finally:
        if source is not None:
            source.close()
    return 2 if bridge is not None and bridge.failed else 0


if __name__ == '__main__':
    raise SystemExit(main())
