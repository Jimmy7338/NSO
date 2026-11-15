#!/usr/bin/env python3
"""Hold authenticated exact RTAB epoch correction at real CSM sample stamps.

This process receives no occupancy grids, RGB-D images, graph histories or JSON
snapshots. It uses the same inverse mapToOdom correction and actual odometry
timestamps as EpochAdapter. It never extrapolates or dates a transform with now.
Run the mapping backend with publish_held_tf:=false to retain one TF owner.
"""
import copy
import math


class CorrectionHold:
    def __init__(self, correction_caller='/nso_rtabmap_mapping_backend',
                 odom_caller='/nso_csm_odometry_bridge'):
        self.correction_caller=correction_caller
        self.odom_caller=odom_caller
        self.epoch=None

    def receive_correction(self,msg):
        if (getattr(msg,'_connection_header',{}).get('callerid')!=self.correction_caller
                or msg.header.frame_id!='odom' or msg.child_frame_id!='nso_pair_map'
                or msg.header.stamp.to_nsec()<=0):
            raise ValueError('unexpected correction source/frame/stamp')
        p,q=msg.transform.translation,msg.transform.rotation
        values=(p.x,p.y,p.z,q.x,q.y,q.z,q.w)
        if (not all(math.isfinite(x) for x in values)
                or abs(sum(x*x for x in values[3:])-1.)>=1e-5):
            raise ValueError('invalid epoch correction')
        if self.epoch is not None and msg.header.stamp.to_nsec()<=self.epoch[0]:
            return False
        # One immutable tuple is the only shared state between callback threads.
        self.epoch=(msg.header.stamp.to_nsec(),copy.deepcopy(msg.transform))
        return True

    def odom(self,msg):
        if (getattr(msg,'_connection_header',{}).get('callerid')!=self.odom_caller
                or msg.header.frame_id!='odom' or msg.child_frame_id!='base_link'
                or msg.header.stamp.to_nsec()<=0):
            raise ValueError('unexpected CSM odometry source/frame/stamp')
        epoch=self.epoch
        if epoch is None or msg.header.stamp.to_nsec()<epoch[0]:return None
        return copy.deepcopy(epoch[1]),copy.deepcopy(msg.header),epoch[0]


def main():
    import rospy
    from geometry_msgs.msg import TransformStamped
    from nav_msgs.msg import Odometry
    from tf2_msgs.msg import TFMessage
    rospy.init_node('nso_rtabmap_tf_hold')
    hold=CorrectionHold()
    publisher=rospy.Publisher('/tf',TFMessage,queue_size=100)
    def correction(msg):
        try:hold.receive_correction(msg)
        except ValueError as exc:rospy.logerr_throttle(5.,'RTAB TF hold rejected correction: %s',exc)
    def odom(msg):
        try:
            result=hold.odom(msg)
            if result is None:return
            transform,header,_=result
            t=TransformStamped();t.header=header;t.header.frame_id='odom'
            t.child_frame_id='nso_pair_map';t.transform=transform
            publisher.publish(TFMessage(transforms=[t]))
        except ValueError as exc:rospy.logerr_throttle(5.,'RTAB TF hold rejected odom: %s',exc)
    subscribers=[rospy.Subscriber('/nso_pair/epoch_correction',TransformStamped,correction,queue_size=1),
                 rospy.Subscriber('/nso_csm/odom',Odometry,odom,queue_size=100)]
    rospy.spin()


if __name__=='__main__':main()
