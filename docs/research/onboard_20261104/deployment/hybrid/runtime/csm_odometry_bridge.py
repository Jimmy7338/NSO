#!/usr/bin/env python3
"""Publish unmodified CSM pose and scan stamp as odometry/TF; twist is derived.

The initial pose is retained until a second increasing-stamp observation arrives.
The default control velocity remains CSM SE(2). An explicit receive-only CAN
option changes only control_odom vx/w; raw odometry and TF remain measured CSM.
"""
import argparse
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


class CsmOdometryBridge:
    def __init__(self, control_velocity_source='csm_se2', can_feedback=None):
        if control_velocity_source not in ('csm_se2', 'can_feedback'):
            raise ValueError('unknown control velocity source')
        if control_velocity_source == 'can_feedback' and can_feedback is None:
            raise ValueError('can_feedback source is required for opt-in mode')
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
        self.tf = tf2_ros.TransformBroadcaster()
        self.sub = rospy.Subscriber('/nso_csm/pose_stamped', PoseStamped, self.callback, queue_size=10, tcp_nodelay=True)

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
        if msg._connection_header.get('callerid') != EXPECTED_CALLER or msg.header.frame_id != SOURCE_FRAME:
            rospy.logwarn_throttle(5, 'Rejecting CSM pose with unexpected caller or frame')
            return
        try:
            sample = planar_sample(msg)
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
            odom.pose.pose = msg.pose
            odom.twist.twist.linear.x, odom.twist.twist.linear.y = vx, vy
            odom.twist.twist.angular.z = w
            # All-zero covariance is unspecified, not a measured accuracy claim.
            transform = TransformStamped()
            transform.header.stamp = msg.header.stamp
            transform.header.frame_id = SOURCE_FRAME
            transform.child_frame_id = CHILD_FRAME
            transform.transform.translation.x = msg.pose.position.x
            transform.transform.translation.y = msg.pose.position.y
            transform.transform.translation.z = msg.pose.position.z
            transform.transform.rotation = msg.pose.orientation
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
                provenance = self.provenance_message_type(data=json.dumps(provenance, sort_keys=True))
            self.pub.publish(odom)
            self.control_pub.publish(control_odom)
            self.tf.sendTransform(transform)
            if provenance is not None:
                self.provenance_pub.publish(provenance)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--control-velocity-source', choices=('csm_se2', 'can_feedback'), default='csm_se2')
    parser.add_argument('--can-interface', default='can0')
    args = parser.parse_args(rospy.myargv()[1:])
    rospy.init_node('nso_csm_odometry_bridge')
    rospy.set_param('~control_velocity_source', args.control_velocity_source)
    source = None
    bridge = None
    try:
        if args.control_velocity_source == 'can_feedback':
            from can_control_feedback import CanVelocityFeedback
            source = CanVelocityFeedback(args.can_interface)
            source.start()
        bridge = CsmOdometryBridge(args.control_velocity_source, source)
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
