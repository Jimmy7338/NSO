#!/usr/bin/env python3
"""Start the selected lidar/gmapping/RGB-D stack on the existing camera master."""
import argparse
import json
import os
from pathlib import Path
import socket
import subprocess
import time

from runtime_environment import bounded_environment
from localization_profiles import DEFAULT_LOCALIZATION_SOURCE, localization_profile, control_velocity_receipt

WORK=Path(__file__).resolve().parents[1]
MASTER='http://127.0.0.1:11382'
def source_stack_present(profile, nodes):
    """CSM must be explicitly assembled; retain RF2O's original startup path."""
    nodes = set(nodes)
    if profile['name'] == 'lidar_csm':
        if '/rf2o_laser_odometry' in nodes:
            raise RuntimeError('RF2O source still present; explicit source-owner switch required')
        missing = set(profile['required_nodes']) - nodes
        if missing:
            raise RuntimeError('CSM source stack unavailable; explicit assembly required: '+repr(sorted(missing)))
        return True
    if nodes.intersection(('/nso_csm/matcher', '/nso_csm_odometry_bridge')):
        raise RuntimeError('CSM source present; explicit source-owner switch required')
    required = set(profile['required_nodes'])
    present = required.intersection(nodes)
    if present and present != required:
        raise RuntimeError('partial laser stack requires inspection: '+repr(sorted(present)))
    return bool(present)


def navigation_command(profile):
    return ['/opt/ros/noetic/bin/roslaunch', str(WORK/'config/hybrid_navigation.launch'),
            'odom_topic:='+profile['control_odom_topic']]


def available():
    try:
        with socket.create_connection(('127.0.0.1',11382),timeout=.5):return True
    except OSError:return False


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--start',action='store_true')
    args=parser.parse_args(argv)
    if os.environ.get('ROS_MASTER_URI')!=MASTER: raise ValueError('use existing master 11382')
    if not args.start:
        print(json.dumps(dict(started=False,no_actuator_started=True,master_available=available(),
                              navigation_source='lidar_rgbd')));return
    if not available(): raise RuntimeError('existing camera master unavailable; no automatic camera replacement')
    import rospy
    from nav_msgs.msg import OccupancyGrid
    from std_msgs.msg import String
    rospy.init_node('nso_hybrid_data_stack_manager',anonymous=True,disable_signals=True)
    profile=localization_profile(rospy.get_param('/nso_visual/localization_source',DEFAULT_LOCALIZATION_SOURCE))
    def graph():
        code,msg,state=rospy.get_master().getSystemState()
        if code!=1:raise RuntimeError(msg)
        return state,{n for group in state for _,callers in group for n in callers}
    state,nodes=graph()
    if '/zed2i/zed_node' not in nodes: raise RuntimeError('existing camera node unavailable')
    if '/nso_visual/rtabmap' in nodes: raise RuntimeError('old RTAB mapper still owns map TF; stop before hybrid startup')
    for key in ('publish_tf','publish_map_tf'):
        if rospy.get_param('/zed2i/zed_node/pos_tracking/'+key,None) is not False:
            raise RuntimeError('ZED dynamic TF must remain disabled')
    for topic,callers in state[0]:
        if topic in ('/nso_radius/guarded_cmd','/nso_visual/actuator_state') and callers:
            raise RuntimeError('actuator present; data-stack reconfiguration refused')
    target=WORK/'evidence/hybrid-data-stack';target.mkdir(parents=True,exist_ok=True)
    children=[]
    def launch(command,name):
        log_path=target/(name+'-'+str(time.time_ns())+'.log')
        with log_path.open('xb') as log:
            child=subprocess.Popen(command,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,
                                   start_new_session=True,env=bounded_environment())
        row=dict(name=name,pid=child.pid,argv=command,log=str(log_path),started_unix_s=time.time(),
                 navigation_source='lidar_rgbd',localization_source=profile['name'],
                 localization_tf_owner=profile['tf_owner'],no_actuator=True)
        (target/(name+'-start.json')).write_text(json.dumps(row,indent=2)+'\n')
        children.append((name,child));return child
    present=source_stack_present(profile,nodes)
    if not present:
        launch(['/opt/ros/noetic/bin/roslaunch',str(WORK/'config/hybrid_sensors.launch')],'sensors')
    if '/nso_visual_sensor_bridge' not in nodes:
        launch(['/usr/bin/python3','-s','-B',str(WORK/'runtime/hybrid_sensor_bridge.py'),'--run',
                '--localization-source',profile['name']],'bridge')
    deadline=time.monotonic()+35.
    while True:
        for name,child in children:
            if child.poll() is not None: raise RuntimeError(name+' launch exited; inspect recorded log')
        try:
            message=rospy.wait_for_message('/nso_visual/health',String,timeout=1.)
            health=json.loads(message.data)
            if (health.get('schema')!='nso_hybrid_sensor_bridge/1'
                or health.get('navigation_source')!='lidar_rgbd'):
                raise RuntimeError('legacy visual bridge still active; inspect before switching')
            if (health.get('localization_source')!=profile['name']
                or health.get('localization_tf_owner')!=profile['tf_owner']
                or health.get('velocity_source')!=profile['velocity_source']):
                raise RuntimeError('existing bridge localization profile differs; explicit restart required')
            if all(health.get(kind,{}).get('valid') for kind in ('odom','depth','laser_source')):break
        except rospy.ROSException: pass
        if time.monotonic()>=deadline:raise RuntimeError('hybrid sources not ready; inspect /nso_visual/health')
    rospy.wait_for_message('/nso_visual/map',OccupancyGrid,timeout=30.)
    _,nodes=graph()
    if not source_stack_present(profile,nodes):
        raise RuntimeError('selected source stack missing after startup')
    if '/nso_visual/move_base' not in nodes:
        launch(navigation_command(profile),'navigation')
    elif rospy.get_param('/nso_visual/move_base/navigation_source',None)!='lidar_rgbd':
        raise RuntimeError('existing navigation does not declare lidar_rgbd; stop before switching')
    rospy.wait_for_service('/nso_visual/move_base/make_plan',timeout=20.)
    receipt=dict(status='hybrid_data_navigation_ready',navigation_source='lidar_rgbd',master=MASTER,
        sampled_unix_s=time.time(),no_actuator_started=True,new_processes=[name for name,_ in children],
        localization_source=profile['name'],velocity_source=profile['velocity_source'],
        odom_source_topic=profile['odom_topic'],odom_expected_publisher=profile['expected_publisher'],
        control_odom_topic=profile['control_odom_topic'],
        tf_ownership=dict(odom_to_base=profile['tf_owner'].lstrip('/'),map_to_odom='slam_gmapping',bridge_dynamic_tf=False),
        physical_extrinsic_calibration_verified=False)
    receipt.update(control_velocity_receipt(profile, rospy.get_param(
        '/nso_csm_odometry_bridge/control_velocity_source', None), rospy.get_param(
        '/nso_csm_odometry_bridge/pose_source', {}) if profile['name']=='lidar_csm' else None))
    (target/'ready.json').write_text(json.dumps(receipt,indent=2)+'\n')
    print(json.dumps(receipt))


if __name__=='__main__':main()
