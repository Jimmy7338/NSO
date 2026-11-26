"""Single ordinary ROS setup displacement, separate from NSO policy evidence."""
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time
import rospy
import tf2_ros

B=Path('/home/ouc/nso_robot')
A=B/'analyses/resume_20261115'
W=B/'workspaces/nso_hybrid_20261029'
episode=sys.argv[1]
forward,left=float(sys.argv[2]),float(sys.argv[3])
if not all(math.isfinite(v) for v in (forward,left)):raise ValueError('finite relative setup required')
for d in Path('/proc').iterdir():
    if not d.name.isdigit():continue
    try:
        args=(d/'cmdline').read_bytes().split(b'\0')
        if any(x.endswith(b'/runtime/mission.py') for x in args):raise RuntimeError('another mission is active')
    except FileNotFoundError:pass
out=A/episode
report=B/'reports/nso-hybrid-20261115'/episode
assert not report.exists()
out.mkdir(exist_ok=False)
rospy.init_node('nso_explicit_setup_initializer',anonymous=True,disable_signals=True)
buf=tf2_ros.Buffer();listener=tf2_ros.TransformListener(buf)
time.sleep(1.2)
t=buf.lookup_transform('nso_pair_map','base_link',rospy.Time(0),rospy.Duration(2))
p,q=t.transform.translation,t.transform.rotation
yaw=math.atan2(2*(q.w*q.z+q.x*q.y),1-2*(q.y*q.y+q.z*q.z))
target=[p.x+forward*math.cos(yaw)-left*math.sin(yaw),p.y+forward*math.sin(yaw)+left*math.cos(yaw),yaw]
goal=dict(frame='nso_pair_map',mapping_backend='rtabmap_optimized_replay_v1',target_odom=target,
    actual_start_pose=[p.x,p.y,yaw],actual_start_stamp_ns=t.header.stamp.to_nsec(),
    relative_forward_m=forward,relative_left_m=left,purpose='ROS_setup_clearance',NSO_action=False)
(out/'goal.json').write_text(json.dumps(goal,indent=2)+'\n')
rospy.signal_shutdown('goal frozen; setup child owns execution')
argv=['/usr/bin/taskset','-c','0,1,4,5,6,7','/bin/bash',str(W/'run_hybrid.sh'),
    '--mode','navigation','--navigation-goal-json',str(out/'goal.json'),
    '--local-controller-profile','teb_onsite_062x059_v1','--ros-allow-reverse',
    '--odometry-jump-profile','source_owned_diagnostic_v1','--feedback-speed-profile','measured_020_v1',
    '--sensor-loss-profile','recoverable_sensor_pause_v1','--native-sender-cpu','3',
    '--native-sender-rt-priority','20','--adapter-cpu','2','--execute','--report',str(report)]
with (out/'mission.log').open('x') as log:
    proc=subprocess.Popen(argv,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
receipt=dict(pid=proc.pid,argv=argv,started_unix_s=time.time(),goal=goal,report=str(report),NSO_action=False)
(out/'process.json').write_text(json.dumps(receipt,indent=2)+'\n')
print(json.dumps(receipt))
