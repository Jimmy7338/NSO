#!/usr/bin/env python3
"""Focal/Noetic service for the pinned full TARE node; no simulator access.

The native 360 x 24 degree visibility model and five-scan planning cadence are
retained. This is a real node bridge, not evidence of a camera-FOV adaptation.
One observe command publishes each paid point cloud exactly once. Parameters
must be frozen by the calling experiment before performance collection.
"""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import signal
import socket
import subprocess
import sys
import time
import xmlrpc.client

ROOT = Path(__file__).resolve().parents[1]
PINNED_NODE_SHA256 = '59fd9b818f4cc6a2edc3e5bf4a8641e2dad92b19ba1d824f1427624aecfd6f03'
NODE_NAME = '/sensor_coverage_planner/tare_planner_node'
DEFAULT_YAML = ROOT / 'third_party/official_baselines/tare_official/src/tare_planner/config/garage.yaml'


def resource_guard(runtime):
    import shutil
    available = int(next(line.split()[1] for line in Path('/proc/meminfo').read_text().splitlines()
                         if line.startswith('MemAvailable:'))) * 1024
    if available < 2 * 1024**3 or shutil.disk_usage(runtime).free < 512 * 1024**2:
        raise RuntimeError('TARE runtime memory reserve would be violated')
    return available


def reexec_runtime(runtime):
    if os.environ.get('NSO_TARE_FOCAL_SERVICE') == '1':
        return
    config = json.loads((runtime / 'environment.json').read_text())
    environment = dict(os.environ)
    environment.update(config)
    environment.update(NSO_TARE_FOCAL_SERVICE='1', OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1')
    executable = runtime / 'runtime/usr/bin/python3.8'
    os.execve(str(executable), [str(executable), '-B', str(Path(__file__).resolve())] + sys.argv[1:], environment)


class Bridge:
    def __init__(self, args):
        import numpy as np
        import yaml
        import rosmaster.master
        import rospy
        from geometry_msgs.msg import PointStamped, PolygonStamped
        from nav_msgs.msg import Odometry, Path as RosPath
        from sensor_msgs.msg import PointCloud2, PointField
        from std_msgs.msg import Bool, Header
        self.np, self.rospy = np, rospy
        self.types = dict(PointStamped=PointStamped, PolygonStamped=PolygonStamped,
                          Odometry=Odometry, Path=RosPath, PointCloud2=PointCloud2,
                          PointField=PointField, Bool=Bool, Header=Header)
        self.args = args
        self.native = None
        self.master = None
        self.log = None
        self.observations = 0
        self.last_action_id = -1
        self.nonempty_scans = 0
        self.output_counter = 0
        self.latest_waypoint = None
        self.confirmed_waypoint = None
        self.latest_paths = {}
        self.finished = False
        self.minimum_available = resource_guard(args.runtime)
        self.native_binary = args.runtime / 'tare_planner_node'
        if hashlib.sha256(self.native_binary.read_bytes()).hexdigest() != PINNED_NODE_SHA256:
            raise ValueError('pinned complete node binary hash mismatch')
        with socket.socket() as probe:
            if probe.connect_ex(('127.0.0.1', args.port)) == 0:
                raise RuntimeError('requested ROS master port is already occupied; refusing to reuse it')
        os.environ['ROS_MASTER_URI'] = 'http://127.0.0.1:%d' % args.port
        os.environ['ROS_IP'] = '127.0.0.1'
        self.master = rosmaster.master.Master(args.port)
        self.master.start()
        self.rpc = xmlrpc.client.ServerProxy(os.environ['ROS_MASTER_URI'], allow_none=True)
        self.parameters = yaml.safe_load(args.parameters.read_text())
        if args.overrides_json:
            overrides = json.loads(args.overrides_json.read_text())
            if not isinstance(overrides, dict):
                raise ValueError('parameter overrides must be a flat dict')
            self.parameters.update(overrides)
        for key, value in self.parameters.items():
            result = self.rpc.setParam('/nso_tare_bridge_v39', NODE_NAME + '/' + key, value)
            if result[0] != 1:
                raise RuntimeError('ROS parameter load failed')
        logfile = args.runtime / 'logs' / ('node-%d.log' % os.getpid())
        self.log = logfile.open('w')
        self.native = subprocess.Popen([str(self.native_binary), '__ns:=/sensor_coverage_planner',
                                        '__name:=tare_planner_node'], stdout=self.log,
                                       stderr=subprocess.STDOUT, start_new_session=True)
        deadline = time.monotonic() + 30
        self.node_uri = None
        while time.monotonic() < deadline:
            self.minimum_available = min(self.minimum_available, resource_guard(args.runtime))
            if self.native.poll() is not None:
                raise RuntimeError('complete TARE node exited before registration; see ' + str(logfile))
            try:
                result = self.rpc.lookupNode('/nso_tare_bridge_v39', NODE_NAME)
                if result[0] == 1:
                    self.node_uri = result[2]
                    subscriptions = xmlrpc.client.ServerProxy(self.node_uri).getSubscriptions('/nso_tare_bridge_v39')
                    if subscriptions[0] == 1 and '/registered_scan' in dict(subscriptions[2]):
                        break
            except (OSError, xmlrpc.client.Error):
                pass
            time.sleep(.05)
        else:
            raise RuntimeError('complete TARE node did not register required sensor input')
        self.registration = self.rpc.getSystemState('/nso_tare_bridge_v39')
        rospy.init_node('nso_tare_bridge_v39', anonymous=False, disable_signals=True)
        self.pubs = {
            'odometry': rospy.Publisher(self.parameters['sub_state_estimation_topic_'], Odometry, queue_size=1),
            'scan': rospy.Publisher(self.parameters['sub_registered_scan_topic_'], PointCloud2, queue_size=1),
            'terrain': rospy.Publisher(self.parameters['sub_terrain_map_topic_'], PointCloud2, queue_size=1),
            'terrain_ext': rospy.Publisher(self.parameters['sub_terrain_map_ext_topic_'], PointCloud2, queue_size=1),
            'start': rospy.Publisher(self.parameters['sub_start_exploration_topic_'], Bool, queue_size=1, latch=True),
            'boundary': rospy.Publisher(self.parameters['sub_viewpoint_boundary_topic_'], PolygonStamped, queue_size=1, latch=True),
        }
        self.subs = [rospy.Subscriber('/way_point', PointStamped, self.waypoint_callback, queue_size=10),
                     rospy.Subscriber('/sensor_coverage_planner/exploration_finish', Bool, self.finish_callback, queue_size=1)]
        for key in ['global_path', 'local_path', 'exploration_path']:
            self.subs.append(rospy.Subscriber('/sensor_coverage_planner/' + key, RosPath,
                                              lambda message, name=key: self.path_callback(name, message), queue_size=1))
        deadline = time.monotonic() + 10
        while time.monotonic() < deadline:
            if all(self.pubs[x].get_num_connections() >= 1 for x in ['odometry', 'scan', 'start']):
                break
            time.sleep(.05)
        else:
            raise RuntimeError('ROS transport connections not established')
        # Native execute() initializes start_time_ only while no odometry has
        # arrived. Wait for its first 1 Hz initialization tick before exposing
        # ready; otherwise a prompt client can leave that timestamp at zero and
        # immediately trigger the native five-second completion gate.
        self.pubs['start'].publish(self.types['Bool'](data=True))
        deadline = time.monotonic() + 5
        while time.monotonic() < deadline:
            if self.native.poll() is not None:
                raise RuntimeError('native node exited during initial timer handshake')
            if self.latest_waypoint is not None:
                break
            time.sleep(.02)
        else:
            raise RuntimeError('native initial timer handshake did not complete')
        self.native_initial_timer_observed = True
        # This unsensed initial waypoint is not a planned action. It must never
        # be exposed as a fallback when the first keypose produces no target.
        self.latest_waypoint = None

    def waypoint_callback(self, message):
        self.output_counter += 1
        self.latest_waypoint = {'xyz': [message.point.x, message.point.y, message.point.z],
                                'frame_id': message.header.frame_id, 'output_counter': self.output_counter}

    def path_callback(self, key, message):
        self.latest_paths[key] = {'frame_id': message.header.frame_id,
                                 'xyz': [[p.pose.position.x, p.pose.position.y, p.pose.position.z] for p in message.poses]}

    def finish_callback(self, message):
        self.finished = bool(message.data)

    def status(self):
        return {'status': 'ready', 'node_name': NODE_NAME, 'node_pid': self.native.pid,
                'node_sha256': PINNED_NODE_SHA256, 'parameters_sha256': hashlib.sha256(
                    json.dumps(self.parameters, sort_keys=True).encode()).hexdigest(),
                'parameters': self.parameters, 'master_uri': os.environ['ROS_MASTER_URI'],
                'minimum_available_ram_bytes': self.minimum_available,
                'sensor_inputs_published': self.observations,
                'native_scan_messages_per_keypose': 5,
                'native_visibility_degrees': [360, 24],
                'camera_visibility_model_adapted': False,
                'native_initial_timer_observed': self.native_initial_timer_observed,
                'registered_system_state': self.registration[2]}

    def pointcloud(self, points, stamp, seq, intensity=False):
        np = self.np
        values = np.asarray(points, dtype='<f4')
        dimensions = 4 if intensity else 3
        if values.ndim != 2 or values.shape[1] != dimensions or not np.isfinite(values).all():
            raise ValueError('finite measured point cloud required')
        message = self.types['PointCloud2']()
        message.header = self.types['Header'](seq=seq, stamp=stamp, frame_id='map')
        names = ['x', 'y', 'z', 'intensity'][:dimensions]
        message.fields = [self.types['PointField'](name=x, offset=i*4, datatype=7, count=1) for i, x in enumerate(names)]
        message.height, message.width = 1, len(values)
        message.is_bigendian, message.is_dense = False, True
        message.point_step, message.row_step = dimensions*4, len(values)*dimensions*4
        message.data = values.tobytes()
        return message

    def observe(self, request):
        np = self.np
        allowed = {'op', 'packet_path', 'action_id'}
        if set(request) != allowed or type(request['action_id']) is not int:
            raise ValueError('observe requires only op, packet_path and integer action_id')
        action_id = request['action_id']
        if action_id != self.last_action_id + 1:
            raise ValueError('exactly one observation per consecutive paid action is required')
        self.minimum_available = min(self.minimum_available, resource_guard(self.args.runtime))
        if self.native.poll() is not None:
            raise RuntimeError('native node is no longer alive')
        packet_path = Path(request['packet_path'])
        # Deliberately never access metadata, semantic, RGB category labels, GT,
        # episode name, or actual hypothesis. Calibration and paid geometry only.
        with np.load(str(packet_path), allow_pickle=False) as packet:
            depth = packet['frame__depth_m'].astype(float)
            intrinsic = packet['frame__intrinsic'].astype(float)
            transform = packet['frame__world_from_camera'].astype(float)
            ranges = packet['scan__ranges_m'].astype(float)
            laser_transform = packet['scan__world_from_laser'].astype(float)
            angle_min = float(packet['scan__angle_min_rad'])
            angle_increment = float(packet['scan__angle_increment_rad'])
            range_max = float(packet['scan__range_max_m'])
        if depth.ndim != 2 or intrinsic.shape != (3, 3) or transform.shape != (4, 4):
            raise ValueError('invalid paid RGB-D calibration shapes')
        if not np.isfinite(depth).all() or not np.isfinite(intrinsic).all() or not np.isfinite(transform).all() or (depth < 0).any():
            raise ValueError('invalid paid RGB-D values')
        yy, xx = np.nonzero(depth > 0)
        zz = depth[yy, xx]
        optical = np.column_stack(((xx-intrinsic[0, 2])*zz/intrinsic[0, 0],
                                   (yy-intrinsic[1, 2])*zz/intrinsic[1, 1], zz))
        rgbd_points = optical.dot(transform[:3, :3].T) + transform[:3, 3]
        if not np.isfinite(ranges).all() or laser_transform.shape != (4, 4):
            raise ValueError('invalid paid planar scan')
        hits = np.flatnonzero((ranges > 0) & (ranges < range_max - 1e-6))
        angles = angle_min + hits*angle_increment
        local = np.column_stack((ranges[hits]*np.cos(angles), ranges[hits]*np.sin(angles), np.zeros(len(hits))))
        laser_points = local.dot(laser_transform[:3, :3].T) + laser_transform[:3, 3]
        points = np.concatenate((rgbd_points, laser_points), axis=0)
        stamp = self.rospy.Time.from_sec(float(action_id + 1))
        odom = self.types['Odometry']()
        odom.header = self.types['Header'](seq=action_id, stamp=stamp, frame_id='map')
        odom.child_frame_id = 'base_link'
        odom.pose.pose.position.x, odom.pose.pose.position.y, odom.pose.pose.position.z = map(float, transform[:3, 3])
        yaw = math.atan2(transform[1, 2], transform[0, 2])
        odom.pose.pose.orientation.z, odom.pose.pose.orientation.w = math.sin(yaw/2), math.cos(yaw/2)
        self.pubs['odometry'].publish(odom)
        self.pubs['start'].publish(self.types['Bool'](data=True))
        # Separate connections need a transport settling interval before the
        # native scan callback checks initialized_. This creates no sensor data.
        time.sleep(.05)
        terrain = np.column_stack((points, np.abs(points[:, 2] - self.args.public_ground_z)))
        self.pubs['terrain'].publish(self.pointcloud(terrain, stamp, action_id, True))
        self.pubs['terrain_ext'].publish(self.pointcloud(terrain, stamp, action_id, True))
        before = self.output_counter
        self.pubs['scan'].publish(self.pointcloud(points, stamp, action_id))
        self.observations += 1
        self.last_action_id = action_id
        if len(points):
            self.nonempty_scans += 1
        planning_due = bool(len(points) and self.nonempty_scans % 5 == 0)
        deadline = time.monotonic() + (self.args.planning_wait_s if planning_due else .15)
        while time.monotonic() < deadline:
            if self.native.poll() is not None:
                raise RuntimeError('native node exited after paid observation')
            self.minimum_available = min(self.minimum_available, resource_guard(self.args.runtime))
            if planning_due and self.output_counter > before:
                break
            time.sleep(.02)
        # Only a waypoint produced after an actual keypose is actionable. A
        # delayed client can receive additional unsensed initialization ticks.
        if planning_due and self.output_counter > before:
            self.confirmed_waypoint = self.latest_waypoint
        waypoint = self.confirmed_waypoint
        return {'status': 'observed', 'action_id': action_id,
                'rgbd_points': len(rgbd_points), 'planar_hit_points': len(laser_points),
                'published_registered_scans': self.nonempty_scans,
                'unique_paid_observations': self.observations,
                'expected_native_planning_due': planning_due,
                'fresh_waypoint': bool(waypoint and self.output_counter > before),
                'waypoint': waypoint, 'paths': self.latest_paths if waypoint else {},
                'exploration_finished': self.finished,
                'public_flat_ground_z': self.args.public_ground_z,
                'native_visibility_degrees': [360, 24], 'camera_visibility_model_adapted': False}

    def boundary(self, request):
        from geometry_msgs.msg import Point32
        vertices = request.get('xy')
        if set(request) != {'op', 'xy'} or not isinstance(vertices, list) or len(vertices) < 3:
            raise ValueError('boundary needs only op and at least three public xy vertices')
        message = self.types['PolygonStamped']()
        message.header.frame_id = 'map'
        for xy in vertices:
            if len(xy) != 2 or not all(math.isfinite(float(x)) for x in xy):
                raise ValueError('finite public boundary required')
            message.polygon.points.append(Point32(x=float(xy[0]), y=float(xy[1]), z=0.))
        self.pubs['boundary'].publish(message)
        return {'status': 'boundary_published', 'vertices': len(vertices)}

    def close(self):
        if self.native is not None and self.native.poll() is None:
            os.killpg(self.native.pid, signal.SIGINT)
            try:
                self.native.wait(timeout=5)
            except subprocess.TimeoutExpired:
                os.killpg(self.native.pid, signal.SIGKILL)
                self.native.wait()
        if self.master is not None:
            self.master.stop()
        if self.log is not None:
            self.log.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--runtime', type=Path, default=Path('/dev/shm/nso_v39_tare'))
    parser.add_argument('--port', type=int, default=11339)
    parser.add_argument('--parameters', type=Path, default=DEFAULT_YAML)
    parser.add_argument('--overrides-json', type=Path)
    parser.add_argument('--public-ground-z', type=float, default=0.)
    parser.add_argument('--planning-wait-s', type=float, default=5.)
    parser.add_argument('--smoke-only', action='store_true')
    parser.add_argument('--receipt', type=Path)
    args = parser.parse_args()
    reexec_runtime(args.runtime)
    bridge = None
    # Constructor failures must also close a partially started native/master.
    bridge = Bridge.__new__(Bridge)
    bridge.native, bridge.master, bridge.log = None, None, None
    try:
        bridge.__init__(args)
        status = bridge.status()
        if args.receipt:
            args.receipt.write_text(json.dumps(status, indent=2) + '\n')
        print(json.dumps(status), flush=True)
        if args.smoke_only:
            return
        for line in sys.stdin:
            request = json.loads(line)
            op = request.get('op')
            if op == 'observe':
                response = bridge.observe(request)
            elif op == 'boundary':
                response = bridge.boundary(request)
            elif op == 'status':
                response = bridge.status()
            elif op == 'close':
                break
            else:
                raise ValueError('unknown bridge operation')
            print(json.dumps(response, allow_nan=False), flush=True)
    finally:
        bridge.close()


if __name__ == '__main__':
    main()
