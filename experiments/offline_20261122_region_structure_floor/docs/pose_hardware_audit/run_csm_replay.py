"""Run deployed CSM on archived simulated LiDAR in a private loopback-only ROS master."""

import argparse
import csv
import json
import math
import os
from pathlib import Path
import signal
import socket
import subprocess
import time
from xmlrpc.client import ServerProxy

import numpy as np

from audit_archive import sha256


ROS = Path('/opt/ros/noetic')
LAUNCH = Path('/home/ouc/nso_robot/workspaces/nso_hybrid_20261029/config/csm_primary.launch')
MATCHER = ROS / 'lib/laser_scan_matcher/laser_scan_matcher_node'
LIDAR_TOPIC = '/nso_pose_audit/scan'
POSE_TOPIC = '/nso_csm/pose_stamped'


def free_port():
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(('127.0.0.1', 0))
        return sock.getsockname()[1]


def wait_for_master(uri, process):
    with ServerProxy(uri) as server:
        for _ in range(100):
            if process.poll() is not None:
                raise RuntimeError('private roscore exited early')
            try:
                if server.getPid('/nso_pose_audit')[0] == 1:
                    return
            except OSError:
                time.sleep(.1)
    raise TimeoutError('private roscore never started')


def stop_group(process):
    if process.poll() is None:
        os.killpg(process.pid, signal.SIGINT)
        try:
            process.wait(5)
        except subprocess.TimeoutExpired:
            os.killpg(process.pid, signal.SIGKILL)
            process.wait(2)


def playback(source_path, output_path, period_s):
    import rospy
    from geometry_msgs.msg import PoseStamped, TransformStamped
    from sensor_msgs.msg import LaserScan
    from tf2_ros import StaticTransformBroadcaster

    rospy.init_node('nso_pose_audit', disable_signals=True)
    with np.load(source_path, allow_pickle=False) as data:
        if set(data.files) != {'stamp_ns', 'ranges_m', 'scan_period_s', 'angle_min_rad', 'angle_increment_rad',
                               'range_max_m', 'laser_height_m'}:
            raise ValueError('Estimator sender accepts calibrated scan data only')
        stamps, ranges = data['stamp_ns'], data['ranges_m']
        scan_period_s = float(data['scan_period_s'])
        angle_min, angle_increment = float(data['angle_min_rad']), float(data['angle_increment_rad'])
        range_max, height = float(data['range_max_m']), float(data['laser_height_m'])
    poses = []

    def pose_received(message):
        pos, q = message.pose.position, message.pose.orientation
        yaw = math.atan2(2 * (q.w * q.z + q.x * q.y), 1 - 2 * (q.y * q.y + q.z * q.z))
        poses.append({'stamp_ns': message.header.stamp.to_nsec(), 'frame_id': message.header.frame_id,
                      'callerid': message._connection_header['callerid'], 'xyyaw': [pos.x, pos.y, yaw],
                      'receipt_monotonic': time.monotonic()})

    subscription = rospy.Subscriber(POSE_TOPIC, PoseStamped, pose_received, queue_size=512)
    publisher = rospy.Publisher(LIDAR_TOPIC, LaserScan, queue_size=16)
    broadcaster = StaticTransformBroadcaster()
    transform = TransformStamped()
    transform.header.frame_id = 'base_link'
    transform.child_frame_id = 'nso_pose_audit_lidar'
    transform.transform.translation.z = height
    transform.transform.rotation.w = 1
    broadcaster.sendTransform(transform)
    for _ in range(150):
        if publisher.get_num_connections():
            break
        time.sleep(.1)
    else:
        raise TimeoutError('CSM never subscribed to isolated scan topic')
    time.sleep(.5)
    scan = LaserScan()
    scan.header.frame_id = transform.child_frame_id
    scan.angle_min = angle_min
    scan.angle_max = angle_min + (ranges.shape[1] - 1) * angle_increment
    scan.angle_increment = angle_increment
    scan.range_min = .02
    scan.range_max = range_max
    scan.scan_time = scan_period_s
    scan.time_increment = 0.0
    for stamp, values in zip(stamps, ranges):
        seconds, nanoseconds = divmod(int(stamp), 10**9)
        scan.header.stamp = rospy.Time(seconds, nanoseconds)
        # The simulator encodes out-of-range as its max-range sentinel.
        scan.ranges = np.where(values >= range_max, np.inf, values).tolist()
        publisher.publish(scan)
        time.sleep(period_s)
    time.sleep(.8)
    snapshot = list(poses)
    result = {'schema': 'nso.csm_simulated_poses/1', 'stamp_ns_joined_by_original_lidar_header': True,
              'ros_master_uri': os.environ['ROS_MASTER_URI'], 'hardware_io': False,
              'input_scan_count': len(stamps), 'pose_count': len(snapshot), 'sensor_trace_sha256': sha256(source_path),
              'ros_poses': snapshot, 'pose_topic': POSE_TOPIC, 'scan_topic': LIDAR_TOPIC,
              'input_field_filter_passed': True, 'scan_out_of_range_conversion': 'native max range sentinel to positive infinity',
              'ros_topics': sorted(tuple(row) for row in rospy.get_published_topics()),
              'csm_ros_parameters': rospy.get_param('/nso_csm/matcher'),
              'deployed_estimator_sha256': sha256(MATCHER), 'deployed_launch_sha256': sha256(LAUNCH),
              'linked_estimator_library_sha256': sha256(ROS / 'lib/liblaser_scan_matcher.so')}
    output_path.write_text(json.dumps(result, indent=2) + '\n')
    with output_path.with_suffix('.csv').open('w', newline='') as stream:
        writer = csv.writer(stream)
        writer.writerow(['stamp_ns', 'frame_id', 'x_m', 'y_m', 'yaw_rad'])
        writer.writerows([row['stamp_ns'], row['frame_id'], *row['xyyaw']] for row in snapshot)
    subscription.unregister()
    rospy.signal_shutdown('offline replay finished')
    if not snapshot:
        raise RuntimeError('CSM produced no timestamped pose on the simulated scan sequence')


def run(source_path, output_path, period_s):
    if output_path.exists():
        raise FileExistsError(output_path)
    port = free_port()
    os.environ.pop('ROS_HOSTNAME', None)
    os.environ['ROS_MASTER_URI'] = uri = f'http://127.0.0.1:{port}'
    os.environ['ROS_IP'] = '127.0.0.1'
    os.environ['ROS_HOME'] = str(output_path.parent / 'ros_home')
    os.environ['ROS_LOG_DIR'] = str(output_path.parent / 'ros_home/log')
    os.environ['ROS_PACKAGE_PATH'] = str(ROS / 'share')
    os.environ['LD_LIBRARY_PATH'] = str(ROS / 'lib')
    with (output_path.parent / 'ros_master.log').open('x') as master_log, \
            (output_path.parent / 'csm_node.log').open('x') as matcher_log:
        core = subprocess.Popen([str(ROS / 'bin/roscore'), '-p', str(port)], stdout=master_log,
                                stderr=subprocess.STDOUT, start_new_session=True)
        try:
            wait_for_master(uri, core)
            matcher = subprocess.Popen([str(ROS / 'bin/roslaunch'), '--wait', str(LAUNCH),
                                        f'scan_topic:={LIDAR_TOPIC}'], stdout=matcher_log,
                                       stderr=subprocess.STDOUT, start_new_session=True)
            try:
                playback(source_path, output_path, period_s)
            finally:
                stop_group(matcher)
        finally:
            stop_group(core)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('out', type=Path)
    parser.add_argument('--wall-s-per-scan', type=float, default=.08)
    args = parser.parse_args()
    if args.wall_s_per_scan <= 0:
        parser.error('wall pacing must be positive')
    out = args.out.resolve()
    run(out / 'native_sensor_input.npz', out / 'csm_stamped_poses.json', args.wall_s_per_scan)
