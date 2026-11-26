"""Reset only the independent experiment map and navigation; no actuator."""
import argparse
import json
import os
from pathlib import Path
import signal
import subprocess
import time

W=Path('/home/ouc/nso_robot/workspaces/nso_hybrid_20261029')
p=argparse.ArgumentParser(description=__doc__)
p.add_argument('--output',type=Path,required=True)
p.add_argument('--old-helper',type=int,required=True)
p.add_argument('--old-navigation',type=int,required=True)
a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False)
assert os.environ.get('ROS_MASTER_URI')=='http://127.0.0.1:11382'

def command(pid):
    return (Path('/proc')/str(pid)/'cmdline').read_bytes().split(b'\0')[:-1]

def stop_owned(pid,required):
    args=[s.decode() for s in command(pid)]
    if required not in args:raise RuntimeError('PID identity mismatch')
    os.kill(pid,signal.SIGINT)
    end=time.monotonic()+22
    while time.monotonic()<end:
        proc=Path('/proc')/str(pid)
        if not proc.exists() or not (proc/'cmdline').read_bytes():return args
        time.sleep(.1)
    raise RuntimeError('owned process did not stop; inspect instead of starting duplicate')

receipt=dict(scope='fresh per-arm stock gmapping and ROS costmaps; original CSM/display map/TSDF retained',no_motion=True)
receipt['stopped_navigation']=stop_owned(a.old_navigation,str(W/'config/joint_navigation.launch'))
receipt['stopped_helper']=stop_owned(a.old_helper,str(W/'runtime/isolated_mapping_tf.py'))

def start(argv,name):
    with (a.output/(name+'.log')).open('xb') as f:
        child=subprocess.Popen(argv,stdin=subprocess.DEVNULL,stdout=f,stderr=subprocess.STDOUT,
                               start_new_session=True)
    return child,dict(pid=child.pid,argv=argv,started_unix_s=time.time())

mapper,receipt['mapper_helper']=start(['/usr/bin/python3','-s','-B',str(W/'runtime/isolated_mapping_tf.py'),'--launch-mapper'],'mapping')
(a.output/'start_receipt.json').write_text(json.dumps(receipt,indent=2)+'\n')
import rospy
from nav_msgs.msg import OccupancyGrid
rospy.init_node('nso_pair_epoch_start',disable_signals=True)
end=time.monotonic()+25
while True:
    if mapper.poll() is not None:raise RuntimeError('new mapper helper exited')
    m=rospy.wait_for_message('/nso_pair/map',OccupancyGrid,timeout=5)
    if (m.header.stamp.to_sec()>=receipt['mapper_helper']['started_unix_s'] and
        m.header.frame_id=='nso_pair_map' and m._connection_header.get('callerid')=='/nso_pair/gmapping'):
        break
    if time.monotonic()>end:raise RuntimeError('fresh experiment map not received')
receipt['initial_map']=dict(stamp_ns=m.header.stamp.to_nsec(),frame=m.header.frame_id,
    publisher=m._connection_header.get('callerid'),known_cells=sum(v>=0 for v in m.data),
    occupied_cells=sum(v==100 for v in m.data))
nav,receipt['navigation']=start(['/opt/ros/noetic/bin/roslaunch',str(W/'config/joint_navigation.launch'),
    'latch_xy_goal_tolerance:=true','odom_topic:=/nso_csm/control_odom','map_topic:=/nso_pair/map'],'navigation')
(a.output/'start_receipt.json').write_text(json.dumps(receipt,indent=2)+'\n')
rospy.wait_for_service('/nso_visual/move_base/make_plan',timeout=25)
if nav.poll() is not None:raise RuntimeError('new navigation launch exited')
receipt['costmap_sources']={k:rospy.get_param('/nso_visual/move_base/'+k+'/static_layer/map_topic')
                          for k in ('global_costmap','local_costmap')}
assert all(v=='/nso_pair/map' for v in receipt['costmap_sources'].values())
receipt['ready_unix_s']=time.time()
(a.output/'start_receipt.json').write_text(json.dumps(receipt,indent=2)+'\n')
print(json.dumps(receipt))
