#!/usr/bin/env python3
"""Loopback-only bridge for native TARE and acquired simulation observations."""

import argparse
from contextlib import redirect_stdout
import hashlib
import json
import math
import os
from pathlib import Path
import signal
import subprocess
import sys
from threading import Condition
import time
from urllib.parse import urlsplit
import xmlrpc.client


NODE = "/sensor_coverage_planner/tare_planner_node"
SCANS_PER_KEYPOSE = 5


def digest(path):
    with path.open("rb") as stream:
        hasher = hashlib.sha256()
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def acquire_packet(path):
    import numpy as np

    with np.load(str(path), allow_pickle=False) as packet:
        depth = np.asarray(packet["frame__depth_m"], dtype=float)
        camera_matrix = np.asarray(packet["frame__intrinsic"], dtype=float)
        camera = np.asarray(packet["frame__world_from_camera"], dtype=float)
        ranges = np.asarray(packet["scan__ranges_m"], dtype=float)
        laser = np.asarray(packet["scan__world_from_laser"], dtype=float)
        angle_min = float(packet["scan__angle_min_rad"])
        angle_step = float(packet["scan__angle_increment_rad"])
        max_range = float(packet["scan__range_max_m"])
        base = np.asarray(packet["base__world_from_base"], dtype=float) if "base__world_from_base" in packet else None

    if (depth.ndim != 2 or ranges.ndim != 1 or camera_matrix.shape != (3, 3)
            or not np.isfinite(camera_matrix).all()
            or camera_matrix[0, 0] <= 0 or camera_matrix[1, 1] <= 0
            or not all(math.isfinite(x) for x in (angle_min, angle_step, max_range))
            or max_range <= 0 or angle_step <= 0):
        raise ValueError("Invalid depth or scan calibration")
    for pose in (camera, laser) if base is None else (camera, laser, base):
        if pose.shape != (4, 4) or not np.isfinite(pose).all() or not np.allclose(pose[3], (0, 0, 0, 1)):
            raise ValueError("Invalid acquired pose")
    yy, xx = np.nonzero(np.isfinite(depth) & (depth > 0))
    z = depth[yy, xx]
    optical = np.column_stack(((xx-camera_matrix[0, 2])*z/camera_matrix[0, 0],
                               (yy-camera_matrix[1, 2])*z/camera_matrix[1, 1], z))
    rgbd = optical @ camera[:3, :3].T + camera[:3, 3]
    hit_indices = np.flatnonzero(np.isfinite(ranges) & (ranges > 0) & (ranges < max_range - 1e-6))
    angles = angle_min + hit_indices * angle_step
    planar_sensor = np.column_stack((ranges[hit_indices]*np.cos(angles),
                                     ranges[hit_indices]*np.sin(angles), np.zeros(len(hit_indices))))
    planar = planar_sensor @ laser[:3, :3].T + laser[:3, 3]
    return np.concatenate((rgbd, planar)), len(rgbd), len(planar), camera, laser, base


class Worker:
    def __init__(self, args):
        self.args = args
        self.master = None
        self.node = None
        self.rospy = None
        self.native_log = None
        self.activity = Condition()
        self.waypoint = None
        self.paths = {}
        self.finish = False
        self.confirmed_waypoint = None
        self.confirmed_paths = {}
        self.waypoint_count = 0
        self.path_counts = {}
        self.runtime_count = 0
        self.native_runtime_s = None
        self.finish_count = 0
        self.last_action_id = -1
        self.scans = 0

    def start(self):
        import rosmaster.master
        import rospy
        import yaml
        from geometry_msgs.msg import PointStamped
        from nav_msgs.msg import Odometry, Path as RosPath
        from sensor_msgs.msg import PointCloud2, PointField
        from std_msgs.msg import Bool, Float32, Header

        self.rospy = rospy
        self.PointCloud2, self.PointField, self.Odometry, self.Header = PointCloud2, PointField, Odometry, Header
        self.parameters = yaml.safe_load(self.args.parameters.read_text())
        if not isinstance(self.parameters, dict):
            raise ValueError("TARE YAML must contain a parameter mapping")
        if self.args.overrides_json:
            overrides = json.loads(self.args.overrides_json.read_text())
            if not isinstance(overrides, dict):
                raise ValueError("TARE overrides must be a parameter mapping")
            self.parameters.update(overrides)

        os.environ.pop("ROS_HOSTNAME", None)
        os.environ.pop("ROS_IPV6", None)
        os.environ["ROS_IP"] = "127.0.0.1"
        os.environ["ROS_HOME"] = str(self.args.native_log.parent / (self.args.native_log.name + ".ros"))
        os.environ["ROS_LOG_DIR"] = os.environ["ROS_HOME"]
        Path(os.environ["ROS_HOME"]).mkdir(parents=True, exist_ok=False)
        self.master = rosmaster.master.Master(port=0, num_workers=3)
        self.master.start()
        master_uri = self.master.uri
        if urlsplit(master_uri).hostname != "127.0.0.1":
            raise RuntimeError("Private ROS master did not bind loopback")
        os.environ["ROS_MASTER_URI"] = master_uri
        rpc = xmlrpc.client.ServerProxy(master_uri, allow_none=True)
        for key, value in self.parameters.items():
            code, _, _ = rpc.setParam("/nso_tare_worker", NODE + "/" + key, value)
            if code != 1:
                raise RuntimeError(f"Could not set native parameter {key}")

        self.binary_sha = digest(self.args.binary)
        self.native_log = self.args.native_log.open("x", encoding="utf-8")
        self.node = subprocess.Popen([str(self.args.binary), "__ns:=/sensor_coverage_planner",
                                      "__name:=tare_planner_node"], stdout=self.native_log,
                                     stderr=subprocess.STDOUT, start_new_session=True)
        rospy.init_node("nso_tare_worker", disable_signals=True)

        self.odometry_pub = rospy.Publisher(self.parameters["sub_state_estimation_topic_"], Odometry, tcp_nodelay=True)
        self.scan_pub = rospy.Publisher(self.parameters["sub_registered_scan_topic_"], PointCloud2, tcp_nodelay=True)
        self.start_pub = rospy.Publisher(self.parameters["sub_start_exploration_topic_"], Bool,
                                         latch=True, queue_size=1)
        self.subscribers = [
            rospy.Subscriber(self.parameters["pub_waypoint_topic_"], PointStamped, self.on_waypoint),
            rospy.Subscriber(self.output_topic("pub_exploration_finish_topic_"), Bool, self.on_finish),
            rospy.Subscriber(self.output_topic("pub_runtime_topic_"), Float32, self.on_runtime),
        ]
        for name in ("exploration_path", "local_path", "global_path"):
            self.subscribers.append(rospy.Subscriber(
                f"/sensor_coverage_planner/{name}", RosPath, self.on_path, callback_args=name))
        ready_deadline = time.monotonic() + 20
        while not all(pub.get_num_connections() for pub in (self.odometry_pub, self.scan_pub, self.start_pub)):
            self.ensure_running()
            if time.monotonic() >= ready_deadline:
                raise TimeoutError("TARE did not connect to the three isolated input topics")
            time.sleep(.02)
        self.start_pub.publish(Bool(data=True))
        if not self.wait_until(lambda: self.waypoint_count > 0, 5):
            raise TimeoutError("TARE did not publish its initial unsensed timer waypoint")
        return {"status": "ready", "master_uri": master_uri, "pid": self.node.pid,
                "node_sha256": self.binary_sha, "parameters": self.parameters,
                "parameters_sha256": digest(self.args.parameters), "native_log": str(self.args.native_log),
                "scans_per_keypose": SCANS_PER_KEYPOSE, "native_timer_hz": 1,
                "terrain_inputs_published": False, "boundary_inputs_published": False,
                "initial_unsensed_waypoint_ignored": True}

    def output_topic(self, key):
        topic = self.parameters[key]
        return topic if topic.startswith("/") else "/sensor_coverage_planner/" + topic

    def ensure_running(self):
        if self.node.poll() is not None:
            raise RuntimeError(f"Native TARE exited with code {self.node.returncode}; {self.args.native_log}")

    def wait_until(self, predicate, seconds):
        deadline = time.monotonic() + seconds
        with self.activity:
            while not predicate():
                self.ensure_running()
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    return False
                self.activity.wait(min(.1, remaining))
        return True

    def on_waypoint(self, message):
        with self.activity:
            xyz = [message.point.x, message.point.y, message.point.z]
            self.waypoint = {"frame_id": message.header.frame_id,
                             "xyz": [value if math.isfinite(value) else None for value in xyz],
                             "valid": all(math.isfinite(value) for value in xyz)}
            self.waypoint_count += 1
            self.activity.notify_all()

    def on_path(self, message, name):
        points = [[pose.pose.position.x, pose.pose.position.y, pose.pose.position.z] for pose in message.poses]
        with self.activity:
            self.paths[name] = {"frame_id": message.header.frame_id,
                                "xyz": [[value if math.isfinite(value) else None for value in pt] for pt in points]}
            self.path_counts[name] = self.path_counts.get(name, 0) + 1
            self.activity.notify_all()

    def on_finish(self, message):
        with self.activity:
            self.finish = bool(message.data)
            self.finish_count += 1
            self.activity.notify_all()

    def on_runtime(self, message):
        with self.activity:
            self.runtime_count += 1
            self.native_runtime_s = float(message.data)
            self.activity.notify_all()

    def pointcloud(self, points, stamp, seq):
        import numpy as np

        values = np.ascontiguousarray(points, dtype="<f4")
        if not np.isfinite(values).all():
            raise ValueError("Cannot publish non-finite acquired point positions")
        cloud = self.PointCloud2()
        cloud.header = self.Header(seq=seq, stamp=stamp, frame_id="map")
        cloud.height, cloud.width = 1, len(values)
        cloud.is_dense = True
        cloud.fields = [self.PointField(name=field, offset=4*i, datatype=self.PointField.FLOAT32, count=1)
                        for i, field in enumerate(("x", "y", "z"))]
        cloud.point_step = 12
        cloud.row_step = 12 * cloud.width
        cloud.data = values.tobytes()
        return cloud

    def observe(self, request):
        if set(request) != {"op", "packet_path", "action_id"} or type(request["action_id"]) is not int:
            raise ValueError("Expected one action ID and one acquired packet path")
        action_id = request["action_id"]
        if action_id != self.last_action_id + 1:
            raise ValueError("Packets must start at action zero and advance once per genuine action")
        packet = Path(request["packet_path"])
        if not packet.is_absolute() or not packet.is_file():
            raise ValueError("Observation needs an absolute path to its acquired packet")
        points, rgbd_count, planar_count, camera, laser, base = acquire_packet(packet)
        self.ensure_running()
        stamp = self.rospy.Time.from_sec(float(action_id + 1))
        odom = self.Odometry()
        odom.header = self.Header(seq=action_id, stamp=stamp, frame_id="map")
        odom.child_frame_id = "base_link"
        orientation = laser if base is None else base
        x, y, z = orientation[:3, 3]
        if base is None:
            z = camera[2, 3]
        yaw = math.atan2(orientation[1, 0], orientation[0, 0])
        odom.pose.pose.position.x, odom.pose.pose.position.y, odom.pose.pose.position.z = map(float, (x, y, z))
        odom.pose.pose.orientation.z, odom.pose.pose.orientation.w = math.sin(yaw/2), math.cos(yaw/2)
        self.odometry_pub.publish(odom)
        # The stock node subscribes to odometry and scans on distinct transports.
        time.sleep(.05)
        with self.activity:
            before_runtime, before_waypoint = self.runtime_count, self.waypoint_count
            before_finish, before_paths = self.finish_count, self.path_counts.copy()
        self.scan_pub.publish(self.pointcloud(points, stamp, action_id))
        self.last_action_id = action_id
        self.scans += bool(len(points))
        due = bool(len(points) and self.scans % SCANS_PER_KEYPOSE == 0)
        completed = False
        started = time.monotonic()
        if due:
            completed = self.wait_until(lambda: self.runtime_count > before_runtime, self.args.planning_wait_s)
            if completed:
                # TARE publishes all path/point topics before /runtime in the
                # same callback, but TCPROS can deliver them out of order.
                self.wait_until(lambda: self.waypoint_count > before_waypoint, .25)
                time.sleep(.05)
        else:
            # Avoid queue coalescing between consecutive recorded actions.
            time.sleep(.06)
        with self.activity:
            fresh = completed and self.waypoint_count > before_waypoint and self.waypoint["valid"]
            if completed:
                self.confirmed_waypoint = dict(self.waypoint, source_action_id=action_id) if fresh else None
                self.confirmed_paths = {
                    name: dict(path, source_action_id=action_id) for name, path in self.paths.items()
                    if self.path_counts[name] > before_paths.get(name, 0)
                }
            return {"status": "observed", "action_id": action_id, "packet_sha256": digest(packet),
                    "rgbd_points": rgbd_count, "planar_hit_points": planar_count,
                    "published_registered_scans": self.scans, "unique_paid_observations": action_id + 1,
                    "expected_native_planning_due": due, "native_keypose_completed": completed,
                    "native_keypose_completion_timeout": due and not completed,
                    "fresh_waypoint": bool(fresh), "waypoint": self.confirmed_waypoint,
                    "paths": self.confirmed_paths, "exploration_finished": self.finish,
                    "fresh_exploration_finished": self.finish_count > before_finish and self.finish,
                    "native_runtime_s": self.native_runtime_s if completed else None,
                    "wall_wait_s": time.monotonic() - started, "odometry_xyz": [float(x), float(y), float(z)],
                    "odometry_yaw_rad": yaw, "odometry_reference":
                    "laser_xy_yaw_and_camera_height" if base is None else "supplied_base_pose"}

    def close(self):
        if self.node is not None and self.node.poll() is None:
            try:
                os.killpg(self.node.pid, signal.SIGINT)
                self.node.wait(timeout=3)
            except subprocess.TimeoutExpired:
                os.killpg(self.node.pid, signal.SIGKILL)
                self.node.wait(timeout=3)
            except ProcessLookupError:
                pass
        if self.rospy is not None:
            self.rospy.signal_shutdown("simulation worker closed")
        if self.master is not None:
            self.master.stop()
        if self.native_log is not None:
            self.native_log.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--binary", type=Path, required=True)
    parser.add_argument("--parameters", type=Path, required=True)
    parser.add_argument("--native-log", type=Path, required=True)
    parser.add_argument("--overrides-json", type=Path)
    parser.add_argument("--planning-wait-s", type=float, default=3.)
    args = parser.parse_args()
    if not 1.5 <= args.planning_wait_s <= 30:
        parser.error("planning wait must allow a native 1 Hz timer cycle")
    worker = Worker(args)
    reply = sys.stdout
    with redirect_stdout(sys.stderr):
        try:
            print(json.dumps(worker.start(), allow_nan=False), file=reply, flush=True)
            for line in sys.stdin:
                request = json.loads(line)
                if request == {"op": "close"}:
                    break
                if request.get("op") != "observe":
                    raise ValueError("Only observe and close operations are supported")
                result = worker.observe(request)
                print(json.dumps(result, allow_nan=False), file=reply, flush=True)
        finally:
            worker.close()


if __name__ == "__main__":
    main()
