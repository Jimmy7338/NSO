#!/usr/bin/env python3
"""Display-only YOLOE observations from one atomic RGB-D packet; no ROS APIs."""
import os
# Set before importing NumPy/Torch; never inherit an unbounded BLAS pool.
for _key in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS',
             'NUMEXPR_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS', 'BLIS_NUM_THREADS'):
    os.environ[_key] = '1'
os.environ['YOLO_OFFLINE'] = 'true'
os.environ['YOLO_AUTOINSTALL'] = 'false'

import argparse
import copy
import json
from pathlib import Path
import sys
import tempfile
import time
import numpy as np

WORKSPACE = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = Path('/home/ouc/nso_robot/experiments/live_dashboard_csm_no_tf_20261107/latest_display_frame.npz')
LABEL_ZH = {'shelves': '置物架', 'bookcase': '书架', 'closed equipment cabinet': '封闭设备柜',
            'cabinet': '柜体', 'table': '桌子', 'chair': '椅子', 'display counter': '展示柜台'}


def validate_geometry(depth, intrinsic, world_from_camera):
    depth = np.asarray(depth)
    intrinsic = np.asarray(intrinsic, dtype=float)
    transform = np.asarray(world_from_camera, dtype=float)
    if depth.ndim != 2 or not np.issubdtype(depth.dtype, np.number):
        raise ValueError('measured depth_m must be a numeric HxW array')
    if intrinsic.shape != (3, 3) or not np.isfinite(intrinsic).all():
        raise ValueError('finite 3x3 intrinsic required')
    if intrinsic[0, 0] <= 0 or intrinsic[1, 1] <= 0 or not np.allclose(intrinsic[2], [0, 0, 1]):
        raise ValueError('invalid optical intrinsic')
    if transform.shape != (4, 4) or not np.isfinite(transform).all():
        raise ValueError('finite 4x4 odom_from_camera required')
    rotation = transform[:3, :3]
    if (not np.allclose(transform[3], [0, 0, 0, 1]) or
            not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-3) or
            not np.isclose(np.linalg.det(rotation), 1., atol=1e-3)):
        raise ValueError('invalid rigid odom_from_camera')
    return depth, np.linalg.inv(intrinsic), transform


def convex_hull_xy(points, max_points=2048, max_vertices=32):
    """Finite sampled visible-depth hull, never a box or hidden-object outline."""
    points = np.asarray(points, dtype=float)
    if len(points) > max_points:
        points = points[np.linspace(0, len(points)-1, max_points, dtype=int)]
    points = sorted(set(map(tuple, points[np.isfinite(points).all(axis=1)])))
    if len(points) < 3:
        return []
    def cross(o, a, b):
        return (a[0]-o[0])*(b[1]-o[1])-(a[1]-o[1])*(b[0]-o[0])
    lower, upper = [], []
    for p in points:
        while len(lower) >= 2 and cross(lower[-2], lower[-1], p) <= 0:
            lower.pop()
        lower.append(p)
    for p in reversed(points):
        while len(upper) >= 2 and cross(upper[-2], upper[-1], p) <= 0:
            upper.pop()
        upper.append(p)
    hull = lower[:-1] + upper[:-1]
    if len(hull) < 3:
        return []
    if len(hull) > max_vertices:
        hull = [hull[i] for i in np.linspace(0, len(hull)-1, max_vertices, dtype=int)]
    return [[float(x), float(y)] for x, y in hull]


def project_instances(instances, depth_m, intrinsic, world_from_camera, stamp_ns):
    depth, inverse_k, transform = validate_geometry(depth_m, intrinsic, world_from_camera)
    objects = []
    for number, instance in enumerate(instances):
        label = str(instance.get('source_label', '')).strip()
        if not label or label in ('floor', 'wall'):
            continue
        # Empty assigned ownership stays empty: do not reclaim overlapped pixels.
        mask = instance.get('assigned_mask')
        mask_source = 'assigned_mask'
        if mask is None:
            mask, mask_source = instance.get('valid_depth_mask'), 'valid_depth_mask'
        if mask is None:
            continue  # Never use a box, polygon annotation, or guessed class mask.
        mask = np.asarray(mask)
        if mask.dtype != bool or mask.shape != depth.shape:
            raise ValueError('instance must provide an aligned actual boolean depth mask')
        valid = mask & np.isfinite(depth) & (depth > 0)
        vv, uu = np.nonzero(valid)
        if not len(vv):
            continue
        rays = inverse_k @ np.stack([uu, vv, np.ones(len(vv))])
        camera = rays * (depth[vv, uu] / rays[2])
        world = transform[:3, :3] @ camera + transform[:3, 3, None]
        xy = world[:2].T
        finite = np.isfinite(xy).all(axis=1)
        xy = xy[finite]
        if not len(xy):
            continue
        score = float(instance.get('detector_score', float('nan')))
        if not np.isfinite(score) or not 0 <= score <= 1:
            raise ValueError('finite actual detector confidence required')
        center = np.median(xy, axis=0)
        stamp = int(stamp_ns) / 1e9
        index = int(instance.get('instance_index', number))
        objects.append(dict(id=f'{int(stamp_ns)}:{index}', label=label,
            label_zh=LABEL_ZH.get(label, label), confidence=score,
            x=float(center[0]), y=float(center[1]), polygon=convex_hull_xy(xy),
            last_seen_stamp=stamp, measured_support_pixels=int(len(xy)),
            mask_source=mask_source, center_kind='median_measured_visible_surface_xy',
            polygon_kind='sampled_visible_surface_xy_convex_hull_approximation_not_complete_object_outline',
            polygon_max_input_points=2048, polygon_max_vertices=32))
    return objects


def packet_from_npz(path):
    with np.load(path, allow_pickle=False) as packet:
        fields = {k: np.array(packet[k], copy=True) for k in
                  ('rgb', 'depth_m', 'intrinsic', 'world_from_camera', 'stamp_ns')}
        if 'frame' in packet and str(np.asarray(packet['frame']).item()) != 'odom':
            raise ValueError('display packet must be expressed in odom')
        if 'frame_id' in packet:
            fields['frame_id'] = str(np.asarray(packet['frame_id']).item())
    source_stamp = fields['stamp_ns']
    if source_stamp.size != 1 or not np.issubdtype(source_stamp.dtype, np.integer) or int(source_stamp.item()) <= 0:
        raise ValueError('positive original integer stamp_ns required')
    fields['stamp_ns'] = int(source_stamp.item())
    rgb = fields['rgb']
    if rgb.dtype != np.uint8 or rgb.ndim != 3 or rgb.shape[2] != 3 or rgb.shape[:2] != fields['depth_m'].shape:
        raise ValueError('aligned HWC uint8 RGB and measured depth required')
    return fields


def base_payload(stamp_ns, status, inference_seconds=0.):
    return dict(frame='odom', stamp=None if stamp_ns is None else int(stamp_ns)/1e9,
        objects=[], status=status, model='YOLOE', inference_seconds=float(inference_seconds),
        camera_extrinsics_verified=False, display_only=True,
        label_semantics='actual YOLOE prompt predictions; identities and confidences not independently calibrated',
        projection_semantics='measured visible depth pixels projected using supplied odom_from_camera; not complete object extents')


def infer_packet(frontend, packet, *, now=None, max_age=15.):
    stamp = packet['stamp_ns']
    age = (time.time() if now is None else now) - stamp/1e9
    if age < 0:
        result = base_payload(stamp, 'error')
        result['error'] = 'source stamp is in the future'
        return result
    if age > max_age:
        result = base_payload(stamp, 'stale')
        result['source_age_seconds'] = age
        return result
    started = time.perf_counter()
    prediction = frontend.predict(packet['rgb'], packet['depth_m'], frame_id=packet.get('frame_id',str(stamp)))
    elapsed = time.perf_counter() - started
    if now is None and time.time() - stamp/1e9 > max_age:
        return base_payload(stamp, 'stale', elapsed)
    result = base_payload(stamp, 'ok', elapsed)
    result['objects'] = project_instances(prediction['instances'], packet['depth_m'],
        packet['intrinsic'], packet['world_from_camera'], stamp)
    if now is None and time.time() - stamp/1e9 > max_age:
        return base_payload(stamp, 'stale', elapsed)
    result['detected_instance_count'] = len(prediction['instances'])
    return result


def atomic_json(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode='w', dir=path.parent, prefix=path.name+'.',
                                         suffix='.tmp', delete=False, encoding='utf-8') as stream:
            temporary = stream.name
            json.dump(payload, stream, ensure_ascii=False, allow_nan=False, separators=(',', ':'))
            stream.write('\n')
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if temporary is not None and os.path.exists(temporary):
            os.unlink(temporary)


def make_frontend(config_path):
    # Deliberately lazy: geometry/tests have no torch or Ultralytics dependency.
    sys.path.insert(0, str(WORKSPACE/'nso_bridge'))
    import torch
    torch.set_num_threads(1)
    if torch.get_num_interop_threads() != 1:
        torch.set_num_interop_threads(1)
    from nso_bridge.real_semantic_frontend import RealSemanticFrontend
    config = json.loads(Path(config_path).read_text())
    return RealSemanticFrontend(**config)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, default=DEFAULT_INPUT)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--config', type=Path, default=WORKSPACE/'config/real_semantic_frontend.json')
    parser.add_argument('--interval', type=float, default=5.)
    parser.add_argument('--max-age', type=float, default=15.)
    parser.add_argument('--once', action='store_true')
    args = parser.parse_args(argv)
    if args.interval <= 0 or args.max_age <= 0 or not np.isfinite([args.interval,args.max_age]).all():
        parser.error('interval and max-age must be finite and positive')
    frontend, cached, last_stamp = None, None, None
    while True:
        started = time.monotonic()
        try:
            packet = packet_from_npz(args.input)
            last_stamp = packet['stamp_ns']
            age = time.time() - last_stamp/1e9
            if age > args.max_age or age < 0:
                payload = infer_packet(None, packet, max_age=args.max_age)
            elif cached is not None and cached['stamp'] == last_stamp/1e9:
                payload = copy.deepcopy(cached)
            else:
                if frontend is None:
                    frontend = make_frontend(args.config)
                payload = infer_packet(frontend, packet, max_age=args.max_age)
                if payload['status'] == 'ok':
                    cached = copy.deepcopy(payload)
        except Exception as exc:
            payload = base_payload(last_stamp, 'error')
            payload['error'] = type(exc).__name__ + ': ' + str(exc)
        atomic_json(args.output, payload)
        if args.once:
            return 0 if payload['status'] == 'ok' else 1
        time.sleep(max(0., args.interval-(time.monotonic()-started)))


if __name__ == '__main__':
    raise SystemExit(main())
