#!/usr/bin/python3
"""Explicitly restore the CSM data/navigation stack; never starts an actuator.

Run with --start after inspecting the environment. Existing incomplete or
conflicting stacks are refused rather than replaced or switched automatically.
"""
import argparse
import json
import os
from pathlib import Path
import socket
import subprocess
import time

WORK = Path('/home/ouc/nso_robot/workspaces/nso_hybrid_20261029')
OUT = Path(__file__).resolve().parent / 'stack_recovery_logs'
MASTER = 'http://127.0.0.1:11382'
SENSOR_NODES = {'/lslidar_driver_node', '/nso_csm/matcher', '/slam_gmapping'}


def master_available():
    try:
        with socket.create_connection(('127.0.0.1', 11382), timeout=.5):
            return True
    except OSError:
        return False


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--start', action='store_true')
    args = parser.parse_args()
    if not args.start:
        print(json.dumps({'started': False, 'master_available': master_available(), 'source': 'lidar_csm'}))
        return
    if os.environ.get('ROS_MASTER_URI') != MASTER:
        raise RuntimeError('ROS_MASTER_URI must be '+MASTER)
    OUT.mkdir(parents=True, exist_ok=True)
    children = []

    def launch(name, command):
        log = OUT / (name + '-' + str(time.time_ns()) + '.log')
        env = os.environ.copy()
        for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
            env[key] = '1'
        with log.open('xb') as stream:
            child = subprocess.Popen(command, stdin=subprocess.DEVNULL, stdout=stream,
                                     stderr=subprocess.STDOUT, start_new_session=True, env=env)
        children.append((name, child, str(log)))

    def wait_for(predicate, seconds, message):
        deadline = time.monotonic() + seconds
        while time.monotonic() < deadline:
            for name, child, log in children:
                if child.poll() is not None:
                    raise RuntimeError(name+' exited; inspect '+log)
            if predicate():
                return
            time.sleep(.2)
        raise RuntimeError(message)

    if not master_available():
        launch('master', ['/opt/ros/noetic/bin/roscore', '-p', '11382'])
        wait_for(master_available, 20, 'master unavailable')
    import rospy
    from std_msgs.msg import String
    from nav_msgs.msg import OccupancyGrid
    rospy.init_node('nso_verified_csm_stack_recovery', anonymous=True, disable_signals=True)

    def graph():
        code, message, state = rospy.get_master().getSystemState()
        if code != 1:
            raise RuntimeError(message)
        return state, {node for group in state for _, nodes in group for node in nodes}

    state, nodes = graph()
    if '/rf2o_laser_odometry' in nodes or '/nso_visual/rtabmap' in nodes:
        raise RuntimeError('conflicting localization or map owner; explicit inspection required')
    if any(topic in ('/nso_radius/guarded_cmd', '/nso_visual/actuator_state') and callers
           for topic, callers in state[0]):
        raise RuntimeError('actuator present; stack reconfiguration refused')
    rospy.set_param('/nso_visual/localization_source', 'lidar_csm')
    if '/zed2i/zed_node' not in nodes:
        launch('camera', ['/opt/ros/noetic/bin/roslaunch', str(WORK/'config/hybrid_camera_neural.launch')])
    def camera_ready():
        _, current = graph()
        return ('/zed2i/zed_node' in current and
                rospy.get_param('/zed2i/zed_node/pos_tracking/publish_tf', None) is False and
                rospy.get_param('/zed2i/zed_node/pos_tracking/publish_map_tf', None) is False)
    wait_for(camera_ready, 90, 'camera node or disabled dynamic TF parameters unavailable')
    _, nodes = graph()
    present = SENSOR_NODES & nodes
    if present and present != SENSOR_NODES:
        raise RuntimeError('partial CSM laser stack; explicit inspection required: '+repr(sorted(present)))
    if not present:
        launch('csm_sensors', ['/opt/ros/noetic/bin/roslaunch', str(WORK/'config/hybrid_sensors_csm.launch')])
    if '/nso_csm_odometry_bridge' not in nodes:
        launch('csm_odometry_bridge', ['/usr/bin/python3', '-s', '-B', str(WORK/'runtime/csm_odometry_bridge.py')])
    wait_for(lambda: SENSOR_NODES | {'/nso_csm_odometry_bridge'} <= graph()[1], 60, 'CSM stack incomplete')
    _, nodes = graph()
    if '/nso_visual_sensor_bridge' not in nodes:
        launch('hybrid_bridge', ['/usr/bin/python3', '-s', '-B', str(WORK/'runtime/hybrid_sensor_bridge.py'),
                                 '--run', '--localization-source', 'lidar_csm'])
    def healthy():
        try:
            health = json.loads(rospy.wait_for_message('/nso_visual/health', String, timeout=1.).data)
            if (health.get('localization_source') != 'lidar_csm' or
                    health.get('localization_tf_owner') != '/nso_csm_odometry_bridge'):
                raise RuntimeError('existing bridge has conflicting source or TF owner')
            return all(health.get(k, {}).get('valid') for k in ('odom', 'depth', 'laser_source'))
        except rospy.ROSException:
            return False
    wait_for(healthy, 90, 'fresh CSM, depth and laser health unavailable')
    rospy.wait_for_message('/nso_visual/map', OccupancyGrid, timeout=30.)
    _, nodes = graph()
    if '/nso_visual/move_base' not in nodes:
        launch('navigation', ['/opt/ros/noetic/bin/roslaunch', str(WORK/'config/joint_navigation.launch'),
                              'latch_xy_goal_tolerance:=true', 'odom_topic:=/nso_csm/control_odom'])
    elif rospy.get_param('/nso_visual/move_base/DWAPlannerROS/odom_topic', None) != '/nso_csm/control_odom':
        raise RuntimeError('existing navigation does not use CSM control odometry')
    rospy.wait_for_service('/nso_visual/move_base/make_plan', timeout=30.)
    _, nodes = graph()
    if '/nso_live_dashboard' not in nodes:
        launch('dashboard', ['/usr/bin/python3', '-s', '-B', str(WORK/'runtime/live_dashboard.py'),
                             '--output', str(OUT/'dashboard'), '--port', '8765'])
    receipt = {'status': 'ready', 'source': 'lidar_csm', 'no_actuator_started': True,
               'started': [{'name': n, 'pid': c.pid, 'log': l} for n, c, l in children],
               'note': 'Fresh source health checked; no physical stationarity or localization accuracy claim.'}
    (OUT/'last_ready.json').write_text(json.dumps(receipt, indent=2)+'\n')
    print(json.dumps(receipt))


if __name__ == '__main__':
    main()
