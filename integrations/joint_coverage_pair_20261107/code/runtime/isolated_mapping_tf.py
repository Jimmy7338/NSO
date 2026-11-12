#!/usr/bin/env python3
"""Isolate stock gmapping TF and expose its inverse without changing registration.

Global: map -> odom -> {base_link, nso_pair_map}.
Private: nso_pair_map -> odom -> base_link (global map parent excluded).
No commands, navigation goals, sensor restamping or map processing.
"""
import argparse
import copy
import math
from pathlib import Path
import signal
import subprocess

TRIAL_FRAME = 'nso_pair_map'
TRIAL_CALLER = '/nso_pair/gmapping'
ODOM_CALLER = '/nso_csm_odometry_bridge'
GLOBAL_CALLERS = {ODOM_CALLER, '/robot_state_publisher',
                  '/zed2i/zed2i_state_publisher', '/zed2i/zed_node'}


def frame(value):
    return value.lstrip('/')


def valid_transform(t):
    q=t.transform.rotation; p=t.transform.translation
    values=(q.x,q.y,q.z,q.w,p.x,p.y,p.z)
    return (all(math.isfinite(v) for v in values)
            and abs(sum(v*v for v in values[:4])-1.) < 1e-5
            and bool(frame(t.header.frame_id)) and bool(frame(t.child_frame_id))
            and frame(t.header.frame_id)!=frame(t.child_frame_id))


def global_to_private(transforms, caller):
    """Never relay either map parent or our inverse back into the private tree."""
    result=[]
    for t in transforms:
        parent,child=frame(t.header.frame_id),frame(t.child_frame_id)
        if child=='odom' or TRIAL_FRAME in (parent,child):
            continue
        if caller not in GLOBAL_CALLERS or not valid_transform(t):
            continue
        if child=='base_link' and (parent!='odom' or caller!=ODOM_CALLER):
            continue
        result.append(copy.deepcopy(t))
    return result


def inverse_transform(t):
    if not valid_transform(t):
        raise ValueError('invalid rigid transform')
    result=copy.deepcopy(t)
    result.header.frame_id=t.child_frame_id
    result.child_frame_id=t.header.frame_id
    q=t.transform.rotation;p=t.transform.translation
    norm=math.sqrt(q.x*q.x+q.y*q.y+q.z*q.z+q.w*q.w)
    x,y,z,w=q.x/norm,q.y/norm,q.z/norm,q.w/norm
    # -R^T p; quaternion conjugate. Source header stamp is never replaced.
    r=((1-2*(y*y+z*z),2*(x*y-z*w),2*(x*z+y*w)),
       (2*(x*y+z*w),1-2*(x*x+z*z),2*(y*z-x*w)),
       (2*(x*z-y*w),2*(y*z+x*w),1-2*(x*x+y*y)))
    vec=(p.x,p.y,p.z)
    out=result.transform.translation
    out.x,out.y,out.z=(-sum(r[j][i]*vec[j] for j in range(3)) for i in range(3))
    outq=result.transform.rotation
    outq.x,outq.y,outq.z,outq.w=-x,-y,-z,w
    return result


def private_to_global(transforms, caller):
    if caller!=TRIAL_CALLER:
        return []
    return [inverse_transform(t) for t in transforms
            if frame(t.header.frame_id)==TRIAL_FRAME
            and frame(t.child_frame_id)=='odom' and valid_transform(t)]


def main():
    import rospy
    from tf2_msgs.msg import TFMessage
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--launch-mapper',action='store_true',help='Own a separate paired_mapping.launch process; stop only that child on exit')
    args=parser.parse_args(rospy.myargv()[1:])
    rospy.init_node('nso_pair_tf_adapter',disable_signals=True)
    global_pub=rospy.Publisher('/tf',TFMessage,queue_size=100)
    private_pub=rospy.Publisher('/nso_pair/tf',TFMessage,queue_size=100)
    def receive_global(msg):
        ts=global_to_private(msg.transforms,msg._connection_header.get('callerid'))
        if ts:private_pub.publish(TFMessage(transforms=ts))
    def receive_private(msg):
        ts=private_to_global(msg.transforms,msg._connection_header.get('callerid'))
        if ts:global_pub.publish(TFMessage(transforms=ts))
    subscribers=[rospy.Subscriber('/tf',TFMessage,receive_global,queue_size=100),
                 rospy.Subscriber('/nso_pair/tf',TFMessage,receive_private,queue_size=100)]
    child=None
    def stop(signum,unused):rospy.signal_shutdown('signal '+str(signum))
    signal.signal(signal.SIGINT,stop);signal.signal(signal.SIGTERM,stop)
    try:
        if args.launch_mapper:
            launch=Path(__file__).resolve().parents[1]/'config/paired_mapping.launch'
            child=subprocess.Popen(['/opt/ros/noetic/bin/roslaunch',str(launch)])
            rospy.loginfo('isolated mapper supervisor pid=%s argv=%s',child.pid,child.args)
        while not rospy.is_shutdown():
            if child is not None and child.poll() is not None:
                raise RuntimeError('isolated mapper launch exited: '+str(child.returncode))
            rospy.rostime.wallsleep(.1)
    finally:
        if child is not None and child.poll() is None:
            child.send_signal(signal.SIGINT)
            try:child.wait(timeout=15)
            except subprocess.TimeoutExpired:
                child.terminate();child.wait(timeout=5)
        for subscriber in subscribers:subscriber.unregister()


if __name__=='__main__':main()
