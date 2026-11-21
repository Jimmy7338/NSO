"""Restore the last declared sensor/display profile after reboot; no actuator."""
import json
import argparse
import os
from pathlib import Path
import socket
import subprocess
import time

BASE = Path('/home/ouc/nso_robot')
WORK = BASE / 'workspaces/nso_hybrid_20261029'
OUT = Path(__file__).resolve().parent
DISPLAY = BASE / 'experiments/live_dashboard_reboot_20261112'
SEMANTIC = BASE / 'experiments/live_dashboard_semantic_reboot_20261112'
receipt = dict(new_localization_epoch=True, old_grid_reused=False,
               no_actuator=True, started=[], boot_id=Path('/proc/sys/kernel/random/boot_id').read_text().strip())
children = []


def save():
    temporary = OUT / 'processes.json.tmp'
    with temporary.open('w') as stream:
        json.dump(receipt, stream, indent=2)
        stream.write('\n')
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(OUT / 'processes.json')


def start(name, argv):
    existing = next((x for x in receipt['started'] if x['name'] == name), None)
    if existing:
        proc = Path('/proc') / str(existing['pid']) / 'cmdline'
        actual = proc.read_bytes().split(b'\0') if proc.exists() else []
        if actual:
            assert all(a.encode() in actual for a in argv[1:]), 'Owner identity mismatch'
            return
        receipt.setdefault('failed_starts', []).append(existing)
        receipt['started'].remove(existing)
        (OUT / (name + '.log')).rename(OUT / (name + '.failed_' + str(existing['pid']) + '.log'))
    with (OUT / (name + '.log')).open('xb', buffering=0) as stream:
        child = subprocess.Popen(argv, stdin=subprocess.DEVNULL, stdout=stream,
                                 stderr=subprocess.STDOUT, start_new_session=True)
    children.append(child)
    receipt['started'].append(dict(name=name, pid=child.pid, argv=argv, started_unix_s=time.time()))
    save()


def wait_until(predicate, seconds):
    end = time.monotonic() + seconds
    while time.monotonic() < end:
        for child in children:
            if child.poll() is not None:
                raise RuntimeError('Owned data process exited: ' + str(child.pid))
        if predicate():
            return
        time.sleep(.2)
    raise RuntimeError('Data readiness timeout; inspect existing processes, do not duplicate')


def port_open():
    try:
        with socket.create_connection(('127.0.0.1', 11382), timeout=.3):
            return True
    except OSError:
        return False


def main():
    global receipt
    parser = argparse.ArgumentParser()
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args()
    assert os.environ.get('ROS_MASTER_URI') == 'http://127.0.0.1:11382'
    if args.resume:
        previous = json.loads((OUT / 'processes.json').read_text())
        assert previous['boot_id'] == receipt['boot_id'], 'Different boot'
        receipt = previous
    else:
        assert not (OUT / 'processes.json').exists(), 'Already started; inspect receipt'
        assert not port_open(), 'Master already exists; inspect owner'
        assert not DISPLAY.exists() and not SEMANTIC.exists()
    start('master', ['/opt/ros/noetic/bin/roscore', '-p', '11382'])
    wait_until(port_open, 20)
    import rospy
    from nav_msgs.msg import OccupancyGrid
    from std_msgs.msg import String
    rospy.init_node('nso_reboot_data_restore', disable_signals=True)
    rospy.set_param('/nso_visual/localization_source', 'lidar_csm')
    py = ['/usr/bin/python3', '-s', '-B']
    start('camera', ['/opt/ros/noetic/bin/roslaunch', str(WORK/'config/hybrid_camera_neural.launch')])
    start('csm_bridge', py + [str(WORK/'runtime/csm_odometry_bridge.py'), '--control-velocity-source', 'can_feedback', '--can-interface', 'can0'])
    start('sensors', ['/opt/ros/noetic/bin/roslaunch', str(WORK/'config/hybrid_sensors_csm.launch')])
    wait_until(lambda: all(rospy.get_param('/zed2i/zed_node/pos_tracking/'+key, None) is False
                           for key in ('publish_tf', 'publish_map_tf')), 30)
    start('sensor_bridge', py + [str(WORK/'runtime/hybrid_sensor_bridge.py'), '--run', '--localization-source', 'lidar_csm', '--no-self-mask'])

    def healthy():
        try:
            health = json.loads(rospy.wait_for_message('/nso_visual/health', String, timeout=2).data)
            receipt['last_health'] = health
            return all(health.get(k, {}).get('valid') for k in ('odom', 'depth', 'laser_source'))
        except rospy.ROSException:
            return False

    wait_until(healthy, 120)
    start('pair_helper', py + [str(WORK/'runtime/isolated_mapping_tf.py'), '--launch-mapper'])
    m = rospy.wait_for_message('/nso_pair/map', OccupancyGrid, timeout=30)
    assert m.header.frame_id == 'nso_pair_map' and m._connection_header.get('callerid') == '/nso_pair/gmapping'
    receipt['initial_map'] = dict(stamp_ns=m.header.stamp.to_nsec(), frame=m.header.frame_id,
                                  publisher=m._connection_header.get('callerid'))
    start('navigation', ['/opt/ros/noetic/bin/roslaunch', str(WORK/'config/joint_navigation.launch'),
                         'latch_xy_goal_tolerance:=true', 'odom_topic:=/nso_csm/control_odom', 'map_topic:=/nso_pair/map'])
    rospy.wait_for_service('/nso_visual/move_base/make_plan', timeout=30)
    receipt['costmap_sources'] = {k: rospy.get_param('/nso_visual/move_base/'+k+'/static_layer/map_topic')
                                  for k in ('global_costmap', 'local_costmap')}
    assert all(v == '/nso_pair/map' for v in receipt['costmap_sources'].values())
    start('dashboard', py + [str(WORK/'runtime/live_dashboard.py'), '--output', str(DISPLAY), '--port', '8765', '--no-self-mask'])
    start('overlay', py + [str(WORK/'runtime/dashboard_overlay_server.py'), '--port', '8766', '--upstream', 'http://127.0.0.1:8765',
                          '--output', str(SEMANTIC), '--semantic-state', str(SEMANTIC/'semantic_state.json'),
                          '--camera-calibration-receipt', str(BASE/'analyses/sensor_repair_20261108/nov06_restore_trial/deployment/tf_verification.json')])
    start('semantics', [str(BASE/'envs/science312/bin/python'), '-s', '-B', str(WORK/'runtime/dashboard_semantic_worker.py'),
                        '--input', str(DISPLAY/'latest_display_frame.npz'), '--output', str(SEMANTIC/'semantic_state.json'), '--interval', '5'])
    receipt.update(status='data_started', display_output=str(DISPLAY), semantic_output=str(SEMANTIC), completed_unix_s=time.time())
    save()
    print(json.dumps(receipt), flush=True)


if __name__ == '__main__':
    main()
