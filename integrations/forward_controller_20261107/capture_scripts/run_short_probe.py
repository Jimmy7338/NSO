"""Bounded standard-ROS forward probe with a few passive diagnostic snapshots.

The existing mission owns actuation/stop verification. This is not an NSO result.
"""
import argparse
from collections import deque
import io
import json
import math
from pathlib import Path
import signal
import subprocess
import threading
import time

import numpy as np
import rospy
from dynamic_reconfigure.client import Client
from nav_msgs.msg import OccupancyGrid, Odometry, Path as NavPath
from move_base_msgs.msg import MoveBaseActionGoal
from geometry_msgs.msg import PoseStamped
from sensor_msgs.msg import PointCloud2, LaserScan
from std_msgs.msg import String

parser=argparse.ArgumentParser()
parser.add_argument('--name',default='short_forward_02')
parser.add_argument('--forward',type=float,default=.2)
parser.add_argument('--left',type=float,default=0.)
parser.add_argument('--yaw-offset',type=float,default=0.)
parser.add_argument('--forward-point-distance',type=float)
parser.add_argument('--capture-scan-ring',action='store_true',help='Retain at most 15 seconds of raw scans/CSM poses around this diagnostic outcome')
args=parser.parse_args()
ROOT=Path(__file__).resolve().parent
OUT=ROOT/args.name
OUT.mkdir(exist_ok=False)
REPORT=Path('/home/ouc/nso_robot/reports/nso-hybrid-20261107')/('controller_'+args.name)
WORK=Path('/home/ouc/nso_robot/workspaces/nso_hybrid_20261029')
rospy.init_node('nso_short_forward_diagnostics',anonymous=True,disable_signals=True)
lock=threading.RLock()
latest={}
scan_ring=deque(maxlen=150)
pose_ring=deque(maxlen=150)
goal_started=None
def receive(name,msg):
    global goal_started
    with lock:
        latest[name]=(time.monotonic(),msg)
        if args.capture_scan_ring and name in ('scan','csm_pose'):
            (scan_ring if name=='scan' else pose_ring).append((time.time_ns(),msg))
        if name=='goal' and goal_started is None:goal_started=time.monotonic()

specs={
 'costmap':('/nso_visual/move_base/local_costmap/costmap',OccupancyGrid),
 'global_plan':('/nso_visual/move_base/NavfnROS/plan',NavPath),
 'local_plan':('/nso_visual/move_base/DWAPlannerROS/local_plan',NavPath),
 'trajectories':('/nso_visual/move_base/DWAPlannerROS/trajectory_cloud',PointCloud2),
 'raw_odom':('/nso_csm/odom',Odometry),
 'control_odom':('/nso_csm/control_odom',Odometry),
 'actuator':('/nso_visual/actuator_state',String),
 'control_provenance':('/nso_csm/control_velocity_provenance',String),
 'goal':('/nso_visual/move_base/goal',MoveBaseActionGoal),
}
if args.capture_scan_ring:
    specs.update(scan=('/scan',LaserScan),csm_pose=('/nso_csm/pose_stamped',PoseStamped))
subs=[rospy.Subscriber(topic,kind,lambda m,n=name:receive(n,m),queue_size=1)
      for name,(topic,kind) in specs.items()]
client=Client('/nso_visual/move_base/DWAPlannerROS',timeout=5)
old=client.get_configuration(timeout=3)
restore={k:old[k] for k in ['xy_goal_tolerance']}
if args.forward_point_distance is not None:
    restore['forward_point_distance']=old['forward_point_distance']
process=None
telemetry=[]
snapshots=[]
can_capture=None
def pose_of(msg):
    p,q,v=msg.pose.pose.position,msg.pose.pose.orientation,msg.twist.twist
    yaw=math.atan2(2*(q.w*q.z+q.x*q.y),1-2*(q.y*q.y+q.z*q.z))
    return dict(stamp_ns=msg.header.stamp.to_nsec(),pose=[p.x,p.y,yaw],twist=[v.linear.x,v.linear.y,v.angular.z])
try:
    if args.capture_scan_ring:
        from bounded_can_capture import BoundedCanCapture
        can_capture=BoundedCanCapture()
    requested=dict(xy_goal_tolerance=.08)
    if args.forward_point_distance is not None:
        requested['forward_point_distance']=args.forward_point_distance
    applied=client.update_configuration(requested)
    o=rospy.wait_for_message('/nso_csm/odom',Odometry,timeout=5)
    initial=pose_of(o);x,y,a=initial['pose']
    target=[x+args.forward*math.cos(a)-args.left*math.sin(a),
            y+args.forward*math.sin(a)+args.left*math.cos(a),a+args.yaw_offset]
    goal_file=OUT/'goal.json'
    goal_file.write_text(json.dumps(dict(target_odom=target,purpose='20cm execution diagnosis; no NSO credit'),indent=2))
    (OUT/'configuration.json').write_text(json.dumps(dict(initial=initial,target=target,old_DWA=old,requested_DWA=requested,applied_DWA=applied,
      control_velocity_parameters=rospy.get_param('/nso_csm_odometry_bridge',{}),
      matcher_parameters=rospy.get_param('/nso_csm',{}),
      scope='CSM use_tf=false; explicit requested_DWA changes are temporary execution diagnostics, restored afterwards; no NSO policy modification or credit; no reverse; optional trajectory cloud recorded only if already published',
      snapshots_scope='latest asynchronously received messages with individual timestamps; not claimed perfectly synchronized'),indent=2))
    log=(OUT/'mission.log').open('xb')
    argv=[str(WORK/'run_hybrid.sh'),'--mode','navigation','--execute','--navigation-goal-json',str(goal_file),'--duration','45','--report',str(REPORT)]
    process=subprocess.Popen(argv,stdout=log,stderr=subprocess.STDOUT)
    (OUT/'process.json').write_text(json.dumps(dict(pid=process.pid,argv=argv),indent=2))
    thresholds=[0.,3.,6.,12.]
    while process.poll() is None:
        now=time.monotonic()
        with lock:current=dict(latest);start=goal_started
        row=dict(receipt_monotonic=now,seconds_since_goal=None if start is None else now-start)
        for name in ('raw_odom','control_odom'):
            if name in current:row[name]=dict(pose_of(current[name][1]),receipt_monotonic=current[name][0])
        if 'actuator' in current:
            d=json.loads(current['actuator'][1].data)
            row['actuator']={k:d.get(k) for k in ['sampled_monotonic','sampled_unix_s','can_velocity','can_status','last_command','fault','armed']}
        if 'control_provenance' in current:
            row['control_provenance']=json.loads(current['control_provenance'][1].data)
        telemetry.append(row)
        if start is not None and thresholds and now-start>=thresholds[0]:
            at=thresholds.pop(0);pack={};meta={}
            for name,(received,msg) in current.items():
                buf=io.BytesIO();msg.serialize(buf);pack[name]=np.frombuffer(buf.getvalue(),dtype=np.uint8)
                h=getattr(msg,'header',None)
                meta[name]=dict(topic=specs[name][0],type=msg._type,receipt_monotonic=received,
                                stamp_ns=None if h is None else h.stamp.to_nsec(),frame=None if h is None else h.frame_id)
            filename='snapshot_%02d.npz'%int(at)
            np.savez_compressed(OUT/filename,**pack)
            snapshots.append(dict(path=filename,scheduled_s=at,actual_s=now-start,messages=meta))
        time.sleep(.1)
    print(json.dumps(dict(returncode=process.returncode,report=str(REPORT),snapshots=len(snapshots))),flush=True)
finally:
    if process is not None and process.poll() is None:
        process.send_signal(signal.SIGINT)
        process.wait(timeout=15)
    client.update_configuration(restore)
    (OUT/'telemetry.json').write_text(json.dumps(telemetry,separators=(',',':')))
    (OUT/'snapshots.json').write_text(json.dumps(snapshots,indent=2))
    if args.capture_scan_ring:
        packed={};meta=[]
        with lock:rings=[('scan',list(scan_ring)),('csm_pose',list(pose_ring))]
        for kind,rows in rings:
            for index,(received,msg) in enumerate(rows):
                key='%s_%03d'%(kind,index);buf=io.BytesIO();msg.serialize(buf)
                packed[key]=np.frombuffer(buf.getvalue(),dtype=np.uint8)
                meta.append(dict(key=key,topic=specs[kind][0],type=msg._type,
                                 source_stamp_ns=msg.header.stamp.to_nsec(),receipt_unix_ns=received,
                                 caller=getattr(msg,'_connection_header',{}).get('callerid')))
        np.savez_compressed(OUT/'failure_scan_ring.npz',**packed)
        (OUT/'failure_scan_ring.json').write_text(json.dumps(dict(messages=meta,
            scope='Bounded last150 source messages per topic for this specific odometry-jump diagnosis; not a complete matcher-state replay'),indent=2))
    if can_capture is not None:
        can_capture.save_and_close(OUT/'failure_can_ring.json')
    (OUT/'restoration.json').write_text(json.dumps(dict(requested=restore,actual=client.get_configuration(timeout=3)),indent=2))
    for sub in subs:sub.unregister()
    rospy.signal_shutdown('bounded probe finished')
