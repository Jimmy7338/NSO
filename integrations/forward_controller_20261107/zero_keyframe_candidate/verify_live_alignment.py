"""Receive-only verification of the explicitly selected aligned pose source."""
import argparse
import json
import math
from pathlib import Path
import threading
import time
import rosgraph
import rospy
from geometry_msgs.msg import PoseStamped
from nav_msgs.msg import Odometry
from std_msgs.msg import String
from tf2_msgs.msg import TFMessage

parser=argparse.ArgumentParser()
parser.add_argument('--alignment',required=True)
parser.add_argument('--output',required=True)
args=parser.parse_args()
alignment=json.loads(Path(args.alignment).read_text())
tx,ty,angle=alignment['odom_from_source']; c,s=math.cos(angle),math.sin(angle)
rospy.init_node('nso_verify_aligned_source',anonymous=True,disable_signals=True)
lock=threading.Lock(); data={k:{} for k in ['source','odom','control','tf','provenance']}
def receive(kind,msg):
    caller=msg._connection_header.get('callerid')
    if kind=='tf':
        values=[(t.header.stamp.to_nsec(),t) for t in msg.transforms
                if t.header.frame_id=='odom' and t.child_frame_id=='base_link']
    elif kind=='provenance':
        obj=json.loads(msg.data); values=[(obj['csm_source_stamp_ns'],obj)]
    else: values=[(msg.header.stamp.to_nsec(),msg)]
    with lock:
        for stamp,obj in values: data[kind][stamp]=(obj,caller)
subs=[rospy.Subscriber(alignment['source_topic'],PoseStamped,lambda m:receive('source',m),queue_size=100),
      rospy.Subscriber('/nso_csm/odom',Odometry,lambda m:receive('odom',m),queue_size=100),
      rospy.Subscriber('/nso_csm/control_odom',Odometry,lambda m:receive('control',m),queue_size=100),
      rospy.Subscriber('/tf',TFMessage,lambda m:receive('tf',m),queue_size=100),
      rospy.Subscriber('/nso_csm/control_velocity_provenance',String,lambda m:receive('provenance',m),queue_size=100)]
try:
    time.sleep(3)
    with lock: snapshots={k:dict(v) for k,v in data.items()}
    common=sorted(set.intersection(*(set(v) for v in snapshots.values())))
    assert len(common)>=15, 'Insufficient aligned raw/control/TF/provenance observations'
    position_errors=[]; orientation_errors=[]
    for stamp in common:
        src,caller=snapshots['source'][stamp]
        assert caller==alignment['expected_caller'] and src.header.frame_id==alignment['source_frame']
        odom,oc=snapshots['odom'][stamp]; ctrl,cc=snapshots['control'][stamp]
        tr,tc=snapshots['tf'][stamp]; prov,pc=snapshots['provenance'][stamp]
        assert all(k=='/nso_csm_odometry_bridge' for k in [oc,cc,tc,pc])
        expected=(tx+c*src.pose.position.x-s*src.pose.position.y,
                  ty+s*src.pose.position.x+c*src.pose.position.y)
        position_errors.append(math.hypot(odom.pose.pose.position.x-expected[0],odom.pose.pose.position.y-expected[1]))
        q=src.pose.orientation; a,b=math.cos(angle/2),math.sin(angle/2)
        expected_q=(a*q.x-b*q.y,b*q.x+a*q.y,a*q.z+b*q.w,a*q.w-b*q.z)
        orientation_errors.extend(abs(getattr(odom.pose.pose.orientation,k)-v) for k,v in zip('xyzw',expected_q))
        assert odom.header.frame_id==ctrl.header.frame_id=='odom'
        assert odom.pose==ctrl.pose
        assert all(getattr(tr.transform.translation,k)==getattr(odom.pose.pose.position,k) for k in 'xyz')
        assert tr.transform.rotation==odom.pose.pose.orientation
        assert ctrl.twist.twist.linear.x==prov['can_pair']['vx'] and ctrl.twist.twist.angular.z==prov['can_pair']['w']
        assert ctrl.twist.twist.linear.y==0
        assert prov['fixed_coordinate_transform_applied'] and not prov['raw_pose_stamp_tf_unchanged']
        assert prov['source_topic']==alignment['source_topic']
    assert max(position_errors)<1e-9 and max(orientation_errors)<1e-9
    age=(time.time_ns()-common[-1])/1e9;assert 0<=age<.5
    publishers=dict(rosgraph.Master('/nso_verify_aligned_source').getSystemState()[0])
    assert publishers['/nso_csm/odom']==['/nso_csm_odometry_bridge']
    assert publishers['/nso_csm/control_odom']==['/nso_csm_odometry_bridge']
    result=dict(matched_samples=len(common),source_topic=alignment['source_topic'],
                maximum_position_transform_error_m=max(position_errors),
                maximum_quaternion_component_error=max(orientation_errors),last_source_age_s=age,
                same_source_stamps_raw_control_TF=True,CAN_velocity_preserved=True,
                sole_odom_control_publisher=publishers['/nso_csm/odom'],
                scope='Live representation/source checks while stopped; no independent localization accuracy claim')
    Path(args.output).write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result))
finally:
    for sub in subs:sub.unregister()
    rospy.signal_shutdown('receive-only verification complete')
