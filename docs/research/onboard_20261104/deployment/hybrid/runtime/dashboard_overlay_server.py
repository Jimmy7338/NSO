#!/usr/bin/env python3
"""Read-only dashboard overlay; no reconstruction worker or motion interface."""
import argparse
from collections import deque
import copy
import hashlib
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import math
from pathlib import Path
import signal
import threading
import time
from urllib.parse import urlsplit
from urllib.request import urlopen

ASSETS = Path(__file__).parent / 'dashboard'


def camera_calibration_metadata(receipt_path, actual_matrix=None):
    """Report loaded calibration only for a source-hashed receipt and matching TF."""
    result = dict(camera_extrinsics_loaded=False, camera_extrinsics_verified=False,
                  camera_calibration_status='not_configured')
    if receipt_path is None:
        return result
    try:
        import numpy as np
        receipt = json.loads(Path(receipt_path).read_text())
        if (receipt.get('schema') != 'nso.camera_mount_tf_verification.v1' or
                receipt.get('parent_frame') != 'base_link' or
                receipt.get('camera_frame') != 'zed2i_left_camera_optical_frame'):
            raise ValueError('unsupported_calibration_receipt_frames_or_schema')
        source = Path(receipt['source_yaml'])
        if hashlib.sha256(source.read_bytes()).hexdigest() != receipt['source_yaml_sha256']:
            raise ValueError('calibration_source_hash_mismatch')
        expected = np.asarray(receipt['expected_base_from_camera_optical'], dtype=float)
        if (expected.shape != (4, 4) or not np.isfinite(expected).all() or
                not np.allclose(expected[3], [0, 0, 0, 1], atol=1e-8, rtol=0) or
                not np.allclose(expected[:3, :3].T @ expected[:3, :3], np.eye(3), atol=1e-6, rtol=0) or
                not np.isclose(np.linalg.det(expected[:3, :3]), 1., atol=1e-6, rtol=0)):
            raise ValueError('invalid_expected_calibration_matrix')
        actual = np.asarray(actual_matrix, dtype=float)
        if actual.shape != (4, 4) or not np.isfinite(actual).all():
            raise ValueError('camera_TF_unavailable')
        matched = bool(np.allclose(actual, expected, atol=1e-6, rtol=0))
        result.update(camera_extrinsics_loaded=matched,
                      camera_calibration_status='loaded_TF_matches' if matched else 'camera_TF_mismatch',
                      camera_calibration_receipt=str(receipt_path),
                      camera_calibration_max_matrix_error=float(np.max(np.abs(actual-expected))))
    except (OSError, ValueError, TypeError, KeyError) as exc:
        result['camera_calibration_status'] = str(exc)
    return result


def rigid_matrix(transform):
    import numpy as np
    q = transform.rotation
    x, y, z, w = q.x, q.y, q.z, q.w
    if not math.isclose(x*x+y*y+z*z+w*w, 1., abs_tol=1e-6):
        raise ValueError('invalid_camera_TF_quaternion')
    result = np.eye(4)
    result[:3, :3] = [[1-2*(y*y+z*z), 2*(x*y-z*w), 2*(x*z+y*w)],
                       [2*(x*y+z*w), 1-2*(x*x+z*z), 2*(y*z-x*w)],
                       [2*(x*z-y*w), 2*(y*z+x*w), 1-2*(x*x+y*y)]]
    t = transform.translation
    result[:3, 3] = [t.x, t.y, t.z]
    return result


def freshness(stamp, now, limit):
    if not isinstance(stamp, (int, float)) or isinstance(stamp, bool) or not math.isfinite(stamp) or stamp <= 0:
        return dict(fresh=False, age_s=None)
    age = now - stamp
    return dict(fresh=0 <= age <= limit, age_s=age)


def point(transform, xy):
    x, y, yaw = transform
    c, s = math.cos(yaw), math.sin(yaw)
    return [x + c * xy[0] - s * xy[1], y + s * xy[0] + c * xy[1]]


def inverse(transform):
    x, y, yaw = transform
    c, s = math.cos(yaw), math.sin(yaw)
    return [-c*x-s*y, s*x-c*y, -yaw]


def finite_xy(xy):
    return isinstance(xy, (list, tuple)) and len(xy) >= 2 and all(
        isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v) for v in xy[:2])


class Overlay:
    def __init__(self, semantic_path, clock=time.time):
        self.semantic_path = Path(semantic_path)
        self.clock = clock
        self.lock = threading.Lock()
        self.trajectory = deque(maxlen=5000)
        self.pose = None
        self.map_from_odom = None
        self.last_stamp = None
        self.errors = deque(maxlen=4)
        self.seed_attempted = False
        self.seed_receipt = dict(status='waiting_for_fresh_TF', source='independent_5Hz_TF')

    def sample(self, pose, transform):
        """Called by a dedicated 5Hz sampler, independently of HTTP requests."""
        if not freshness(pose.get('stamp'), self.clock(), 3)['fresh']:
            raise ValueError('stale_or_future_pose_TF')
        if not finite_xy([pose.get('x'), pose.get('y')]) or not math.isfinite(pose['yaw']):
            raise ValueError('invalid_pose_TF')
        if len(transform) != 3 or not all(math.isfinite(x) for x in transform):
            raise ValueError('invalid_map_from_odom')
        with self.lock:
            if self.last_stamp is not None and pose['stamp'] < self.last_stamp:
                raise ValueError('pose_TF_stamp_regressed')
            self.pose = dict(pose, frame='odom')
            self.map_from_odom = list(transform)
            if pose['stamp'] != self.last_stamp:
                xy = [pose['x'], pose['y']]
                if not self.trajectory or math.dist(xy, self.trajectory[-1]) > .01:
                    self.trajectory.append(xy)
            self.last_stamp = pose['stamp']

    def seed(self, upstream):
        """Accept old map-space history only after current TF/epoch corroboration."""
        with self.lock:
            if self.seed_attempted or self.pose is None:
                return
            self.seed_attempted = True
            receipt = dict(status='rejected', source='upstream_map_trajectory',
                           upstream_stamp=upstream.get('stamp'), checks='fresh source, close TF stamps/pose/yaw and endpoint')
            self.seed_receipt = receipt
            old = upstream.get('pose') or {}
            history = upstream.get('trajectory')
            try:
                if (upstream.get('map') or {}).get('frame', 'map') != 'map':
                    raise ValueError('upstream_map_frame_not_map')
                if not isinstance(old.get('yaw'), (int, float)) or not math.isfinite(old['yaw']):
                    raise ValueError('invalid_old_pose_yaw')
                if old.get('frame') != 'map' or not freshness(old.get('stamp'), self.clock(), 3)['fresh']:
                    raise ValueError('old_pose_missing_stale_future_or_frame')
                if not (upstream.get('freshness', {}).get('pose') or {}).get('fresh'):
                    raise ValueError('upstream_rejected_pose')
                if abs(old['stamp'] - self.pose['stamp']) > .5:
                    raise ValueError('TF_epoch_timestamp_not_corroborated')
                current_map = point(self.map_from_odom, [self.pose['x'], self.pose['y']])
                yaw_error = math.atan2(math.sin(old['yaw']-self.pose['yaw']-self.map_from_odom[2]),
                                       math.cos(old['yaw']-self.pose['yaw']-self.map_from_odom[2]))
                if not finite_xy([old.get('x'), old.get('y')]) or math.dist(current_map, [old['x'], old['y']]) > .1 or abs(yaw_error) > .2:
                    raise ValueError('TF_epoch_pose_not_corroborated')
                if not isinstance(history, list) or not history or not all(finite_xy(x) for x in history):
                    raise ValueError('invalid_old_trajectory')
                if math.dist(history[-1][:2], current_map) > .15:
                    raise ValueError('old_trajectory_endpoint_not_current')
                converted = [point(inverse(self.map_from_odom), x) for x in history[-4999:]]
                current = [self.pose['x'], self.pose['y']]
                self.trajectory = deque(converted + [current], maxlen=5000)
                receipt.update(status='accepted', point_count=len(converted),
                               frame_conversion='current odom<-map', source_pose_stamp=old['stamp'],
                               TF_pose_stamp=self.pose['stamp'], map_from_odom=list(self.map_from_odom))
            except (KeyError, TypeError, ValueError) as exc:
                receipt['reason'] = str(exc)

    def semantic(self, transform, now):
        try:
            source = json.loads(self.semantic_path.read_text())
            result = copy.deepcopy(source)
            if source.get('frame') != 'odom':
                raise ValueError('semantic_source_frame_must_be_odom')
            objects = []
            for obj in source.get('objects', []):
                if not finite_xy([obj.get('x'), obj.get('y')]):
                    raise ValueError('invalid_semantic_position')
                item = copy.deepcopy(obj)
                item['x'], item['y'] = point(transform, [obj['x'], obj['y']])
                polygon = obj.get('polygon', [])
                if not isinstance(polygon, list) or not all(finite_xy(p) for p in polygon):
                    raise ValueError('invalid_semantic_polygon')
                item['polygon'] = [point(transform, p) for p in polygon]
                objects.append(item)
            result.update(frame='map', source_frame='odom', objects=objects,
                          freshness=freshness(source.get('stamp'), now, 10),
                          camera_extrinsics_verified=False)
            return result
        except (OSError, ValueError, TypeError, KeyError) as exc:
            return dict(status='unavailable', reason=str(exc), frame='map', objects=[],
                        freshness=dict(fresh=False, age_s=None), camera_extrinsics_verified=False)

    def snapshot(self, upstream, current_transform=None, camera_calibration=None):
        now = self.clock()
        result = copy.deepcopy(upstream)
        with self.lock:
            pose = copy.deepcopy(self.pose)
            transform = list(current_transform) if current_transform is not None else copy.deepcopy(self.map_from_odom)
            history = list(self.trajectory)
            seed = copy.deepcopy(self.seed_receipt)
            errors = list(self.errors)
        result.setdefault('freshness', {})
        for name, limit in (('camera', 3), ('map', 10), ('reconstruction', 10)):
            age = freshness((result.get(name) or {}).get('stamp'), now, limit)
            if (upstream.get('freshness', {}).get(name) or {}).get('fresh') is False:
                age['fresh'] = False
            result['freshness'][name] = age
        if pose is not None and transform is not None:
            x, y = point(transform, [pose['x'], pose['y']])
            result['pose'] = dict(x=x, y=y, yaw=math.atan2(math.sin(pose['yaw']+transform[2]),
                                                       math.cos(pose['yaw']+transform[2])),
                                  frame='map', stamp=pose['stamp'])
            result['trajectory'] = [point(transform, xy) for xy in history]
            result['freshness']['pose'] = freshness(pose['stamp'], now, 3)
            result['semantics'] = self.semantic(transform, now)
        else:
            result['pose'] = None
            result['trajectory'] = []
            result['freshness']['pose'] = dict(fresh=False, age_s=None)
            result['semantics'] = dict(status='waiting_for_TF', frame='map', objects=[],
                                      freshness=dict(fresh=False, age_s=None))
        result['freshness']['semantics'] = result['semantics']['freshness']
        result['semantics'].update(camera_calibration or camera_calibration_metadata(None))
        result['trajectory_source'] = dict(sampler_hz=5, storage_frame='odom', display_frame='map',
                                         maximum_points=5000, seed=seed)
        result['errors'] = list(result.get('errors', [])) + errors
        result['stamp'] = now
        return result


def tf_planar(transform):
    t, q = transform.translation, transform.rotation
    if abs(q.x) > 1e-4 or abs(q.y) > 1e-4:
        raise ValueError('nonplanar_TF_not_supported')
    return [t.x, t.y, math.atan2(2*(q.w*q.z+q.x*q.y), 1-2*(q.y*q.y+q.z*q.z))]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--port', type=int, default=8766)
    parser.add_argument('--upstream', default='http://127.0.0.1:8765')
    parser.add_argument('--output', required=True)
    parser.add_argument('--semantic-state', required=True)
    parser.add_argument('--camera-calibration-receipt', type=Path)
    args = parser.parse_args()
    if urlsplit(args.upstream).hostname not in ('127.0.0.1', 'localhost'):
        raise ValueError('upstream must be localhost')
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    overlay = Overlay(args.semantic_state)
    stop = threading.Event()

    def fetch(route):
        with urlopen(args.upstream.rstrip('/') + route, timeout=2) as response:
            return response.read(), response.headers.get('Content-Type', 'application/octet-stream')

    import rospy
    import tf2_ros
    rospy.init_node('nso_dashboard_overlay', disable_signals=True)
    buffer = tf2_ros.Buffer(cache_time=rospy.Duration(30))
    listener = tf2_ros.TransformListener(buffer)

    def sample_loop():
        while not stop.is_set():
            started = time.monotonic()
            try:
                base = buffer.lookup_transform('odom', 'base_link', rospy.Time(0), rospy.Duration(.05))
                correction = buffer.lookup_transform('map', 'odom', rospy.Time(0), rospy.Duration(.05))
                x, y, yaw = tf_planar(base.transform)
                overlay.sample(dict(x=x, y=y, yaw=yaw, stamp=base.header.stamp.to_sec()), tf_planar(correction.transform))
                if not overlay.seed_attempted:
                    payload, _ = fetch('/api/state')
                    overlay.seed(json.loads(payload))
                    (output/'trajectory_seed_receipt.json').write_text(json.dumps(overlay.seed_receipt, indent=2)+'\n')
            except Exception as exc:
                with overlay.lock:
                    overlay.errors.append('overlay_TF: '+str(exc))
            stop.wait(max(0, .2-(time.monotonic()-started)))

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            route = urlsplit(self.path).path
            try:
                if route == '/api/state':
                    payload, _ = fetch(route)
                    correction = buffer.lookup_transform('map', 'odom', rospy.Time(0), rospy.Duration(.05))
                    camera_matrix = None
                    if args.camera_calibration_receipt is not None:
                        try:
                            camera_tf = buffer.lookup_transform('base_link', 'zed2i_left_camera_optical_frame', rospy.Time(0), rospy.Duration(.05))
                            camera_matrix = rigid_matrix(camera_tf.transform)
                        except Exception:
                            pass  # Unavailable camera TF must never claim loaded calibration.
                    calibration = camera_calibration_metadata(args.camera_calibration_receipt, camera_matrix)
                    payload = json.dumps(overlay.snapshot(json.loads(payload), tf_planar(correction.transform), calibration), allow_nan=False).encode()
                    kind = 'application/json'
                elif route in ('/api/camera.jpg', '/api/map.png', '/api/cloud.json'):
                    payload, kind = fetch(route)
                elif route in ('/', '/index.html', '/dashboard.js', '/dashboard.css'):
                    name = 'index.html' if route == '/' else route.lstrip('/')
                    payload = (ASSETS/name).read_bytes()
                    kind = {'index.html':'text/html', 'dashboard.js':'application/javascript', 'dashboard.css':'text/css'}[name]
                else:
                    self.send_error(404)
                    return
                self.send_response(200)
                self.send_header('Content-Type', kind)
                self.send_header('Content-Length', str(len(payload)))
                self.send_header('Cache-Control', 'no-store')
                self.end_headers()
                self.wfile.write(payload)
            except (BrokenPipeError, ConnectionResetError):
                pass
            except Exception:
                self.send_error(503)

        def log_message(self, *_):
            pass

    server = ThreadingHTTPServer(('127.0.0.1', args.port), Handler)
    worker = threading.Thread(target=sample_loop, daemon=True)
    worker.start()

    def shutdown(*_):
        stop.set()
        threading.Thread(target=server.shutdown, daemon=True).start()

    signal.signal(signal.SIGINT, shutdown)
    signal.signal(signal.SIGTERM, shutdown)
    print(json.dumps(dict(url='http://127.0.0.1:'+str(args.port), upstream=args.upstream,
                          read_only=True, reconstruction_worker_started=False)), flush=True)
    try:
        server.serve_forever(poll_interval=.2)
    finally:
        stop.set()
        worker.join(timeout=3)
        server.server_close()
        rospy.signal_shutdown('overlay ended')


if __name__ == '__main__':
    main()
