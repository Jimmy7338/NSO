#!/usr/bin/env python3
"""Publish unmodified CSM pose and scan stamp as odometry/TF; twist is derived.

The initial pose is retained until a second increasing-stamp observation arrives.
This process has no command or CAN interfaces and does not select a TF owner.
"""
import copy
import math
import threading
import rospy
import tf2_ros
from geometry_msgs.msg import PoseStamped, TransformStamped
from nav_msgs.msg import Odometry

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
    def __init__(self):
        self.lock = threading.Lock()
        self.previous = None
        self.pub = rospy.Publisher('/nso_csm/odom', Odometry, queue_size=10)
        self.control_pub = rospy.Publisher('/nso_csm/control_odom', Odometry, queue_size=10)
        self.tf = tf2_ros.TransformBroadcaster()
        self.sub = rospy.Subscriber('/nso_csm/pose_stamped', PoseStamped, self.callback, queue_size=10, tcp_nodelay=True)

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
            self.pub.publish(odom)
            self.control_pub.publish(control_odom)
            self.tf.sendTransform(transform)


if __name__ == '__main__':
    rospy.init_node('nso_csm_odometry_bridge')
    CsmOdometryBridge()
    rospy.spin()
