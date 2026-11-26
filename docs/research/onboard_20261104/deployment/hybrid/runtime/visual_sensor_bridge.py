#!/usr/bin/env python3
"""Shared depth, approved self-mask and pose helpers for the hybrid bridge.

This module has no ROS runner or visual odometry estimator. The legacy module
name and interface constants remain for the RF2O/RGB-D bridge. Depth scans
contain measured obstacle endpoints or NaN and must remain marking-only.
"""

import hashlib
import io
import json
import math
from pathlib import Path
import numpy as np


NODE = '/nso_visual_sensor_bridge'
MASTER = 'http://127.0.0.1:11382'
CAMERA_CALLER = '/zed2i/zed_node'
OPTICAL = 'zed2i_left_camera_optical_frame'
BASE_FROM_LEFT = np.array([[1., 0., 0., .061], [0., 1., 0., .0591],
                           [0., 0., 1., .303], [0., 0., 0., 1.]])
BASE_FROM_OPTICAL = BASE_FROM_LEFT @ np.array(
    [[0., 0., 1., 0.], [-1., 0., 0., 0.], [0., -1., 0., 0.], [0., 0., 0., 1.]])
ASSETS = Path(__file__).resolve().parents[1] / 'config/assets/fixed_imu_left_curve_20261031'
# Reviewed fixed foreground upper face and visible side, using pixel-area
# coverage. Original files remain unchanged; excluded depth is missing.
MASK_SHA256 = '119af867a47bb7092ede51a3838cae386788081e0bc551411f1156ee8387b32b'
MASK_RAW_SHA256 = '22ea770d293bf0134b29bc36c1541e7661863106cc5a69a9aee0ca4711c2c90b'
MASK_PIXELS = 46499


def require(ok, reason):
    if not ok:
        raise ValueError(reason)


def finite(*values):
    return all(math.isfinite(float(x)) for x in values)


def wrap(angle):
    return math.atan2(math.sin(angle), math.cos(angle))


def pose_matrix(xyz, q_xyzw):
    p, q = np.asarray(xyz, dtype=float), np.asarray(q_xyzw, dtype=float)
    require(p.shape == (3,) and q.shape == (4,) and np.isfinite(p).all()
            and np.isfinite(q).all(), 'nonfinite_or_wrong_pose_shape')
    require(abs(float(q @ q)-1.) < 1e-3, 'quaternion_not_unit')
    x, y, z, w = q / np.linalg.norm(q)
    out = np.eye(4)
    out[:3, :3] = [[1-2*(y*y+z*z), 2*(x*y-z*w), 2*(x*z+y*w)],
                  [2*(x*y+z*w), 1-2*(x*x+z*z), 2*(y*z-x*w)],
                  [2*(x*z-y*w), 2*(y*z+x*w), 1-2*(x*x+y*y)]]
    out[:3, 3] = p
    return out


def fresh(stamp_ns, receipt, now, wall, ttl):
    return (isinstance(stamp_ns, (int, np.integer)) and stamp_ns > 0
            and finite(receipt, now, wall) and 0 <= now-receipt <= ttl
            and -.02 <= wall-stamp_ns/1e9 <= ttl)


def load_approved_mask(mask_path=ASSETS/'fixed-imu-mask.npy',
                       provenance_path=ASSETS/'fixed-imu-mask-provenance.json'):
    raw = Path(mask_path).read_bytes()
    provenance = json.loads(Path(provenance_path).read_text())
    require(hashlib.sha256(raw).hexdigest() == MASK_SHA256
            == provenance.get('npy_sha256'), 'approved_self_mask_file_hash_mismatch')
    mask = np.load(io.BytesIO(raw), allow_pickle=False)
    require(mask.dtype == np.bool_ and mask.shape == (720, 1280)
            and int(mask.sum()) == MASK_PIXELS, 'approved_self_mask_geometry_mismatch')
    require(hashlib.sha256(mask.tobytes()).hexdigest() == MASK_RAW_SHA256
            == provenance.get('raw_bool_sha256')
            and provenance.get('application_authorized') is True,
            'approved_self_mask_provenance_mismatch')
    return mask.copy(), dict(file_sha256=MASK_SHA256,
                             raw_bool_sha256=MASK_RAW_SHA256, pixels=MASK_PIXELS,
                             asset_path=str(Path(mask_path).resolve()),
                             provenance_path=str(Path(provenance_path).resolve()),
                             excluded_is_free=False, physical_calibration=False)


def depth_array(data, width, height, step, encoding, is_bigendian=False):
    """Decode ROS depth rows, preserving padding, byte order and unknowns."""
    require(encoding in ('32FC1', '16UC1', 'mono16'), 'unsupported_depth_encoding')
    itemsize = 4 if encoding == '32FC1' else 2
    require(width > 0 and height > 0 and step >= width*itemsize
            and len(data) == step*height, 'invalid_depth_buffer_geometry')
    dtype = np.dtype(('>' if is_bigendian else '<')+('f4' if itemsize == 4 else 'u2'))
    view = np.ndarray((height, width), dtype=dtype, buffer=data,
                      strides=(step, itemsize)).astype(np.float32)
    if itemsize == 2:
        view *= .001
    view[~np.isfinite(view) | (view <= 0)] = np.nan
    return view


class DepthProjector:
    """Project actual depth into a virtual base-frame obstacle-only scan.

    The exact approved fixed IMU image mask becomes unknown. No footprint
    deletion, ground fill, neighbourhood dilation or infinity is introduced.
    Camera rays originate at the camera, so scan clearing MUST remain false.
    """

    def __init__(self, self_mask=None, stride=4, range_min=.12, range_max=4.,
                 obstacle_height_min=.05, obstacle_height_max=1.5,
                 minimum_valid_points=128, minimum_support_rows=4,
                 minimum_support_columns=4):
        self.self_mask = self_mask
        self.stride, self.range_min, self.range_max = stride, range_min, range_max
        self.height_min, self.height_max = obstacle_height_min, obstacle_height_max
        # Source health requires real, spatially supported measurements. The
        # unknown fraction of a corridor/wall/ground image is diagnostic only;
        # it is never a whole-image completeness or free-space requirement.
        self.minimum_valid_points = minimum_valid_points
        self.minimum_support_rows = minimum_support_rows
        self.minimum_support_columns = minimum_support_columns
        self.angle_min, self.angle_max, self.bins = -math.pi/2, math.pi/2, 181
        self.angle_increment = (self.angle_max-self.angle_min)/(self.bins-1)
        self.calibration = None

    def project(self, depth, intrinsic, frame_id=OPTICAL):
        depth = np.asarray(depth, dtype=np.float32)
        k = np.asarray(intrinsic, dtype=float).reshape(3, 3)
        require(depth.ndim == 2 and frame_id == OPTICAL, 'depth_shape_or_frame_invalid')
        require(np.isfinite(k).all() and k[0, 0] > 0 and k[1, 1] > 0
                and np.array_equal(k[2], [0., 0., 1.]), 'depth_intrinsics_invalid')
        require(abs(k[0, 1])+abs(k[1, 0]) < 1e-8, 'unsupported_skewed_intrinsics')
        calibration = (depth.shape, tuple(k.reshape(-1)))
        if self.calibration is None:
            self.calibration = calibration
        require(calibration == self.calibration, 'depth_calibration_changed')
        filtered = depth.copy()
        filtered[~np.isfinite(filtered) | (filtered <= 0)] = np.nan
        if self.self_mask is not None:
            require(self.self_mask.shape == depth.shape, 'self_mask_resolution_changed')
            filtered[self.self_mask] = np.nan
        h, w = depth.shape
        rows, cols = np.mgrid[0:h:self.stride, 0:w:self.stride]
        zz = filtered[::self.stride, ::self.stride]
        valid = np.isfinite(zz) & (zz >= self.range_min) & (zz <= self.range_max)
        available = np.ones(valid.shape, dtype=bool) if self.self_mask is None else (
            ~self.self_mask[::self.stride, ::self.stride])
        fraction = float(valid.sum())/max(1, int(available.sum()))
        support_rows = int(np.unique(rows[valid]).size)
        support_columns = int(np.unique(cols[valid]).size)
        ranges = np.full(self.bins, np.nan, dtype=float)
        stats = dict(valid_fraction=fraction, valid_pixels=int(valid.sum()),
                     sampled_pixels=int(available.sum()), obstacle_points=0,
                     supported_rows=support_rows, supported_columns=support_columns,
                     minimum_valid_points=self.minimum_valid_points,
                     minimum_support_rows=self.minimum_support_rows,
                     minimum_support_columns=self.minimum_support_columns,
                     whole_image_completeness_required=False,
                     unknown_is_free=False, clearing_allowed=False)
        if int(valid.sum()) < self.minimum_valid_points:
            return dict(valid=False, reason='insufficient_finite_depth', ranges=ranges,
                        filtered_depth=filtered, statistics=stats)
        if (support_rows < self.minimum_support_rows
                or support_columns < self.minimum_support_columns):
            return dict(valid=False, reason='insufficient_depth_spatial_support', ranges=ranges,
                        filtered_depth=filtered, statistics=stats)
        z = zz[valid].astype(float)
        optical = np.stack(((cols[valid]-k[0, 2])*z/k[0, 0],
                            (rows[valid]-k[1, 2])*z/k[1, 1], z), axis=1)
        base = optical @ BASE_FROM_OPTICAL[:3, :3].T + BASE_FROM_OPTICAL[:3, 3]
        distance = np.hypot(base[:, 0], base[:, 1])
        angle = np.arctan2(base[:, 1], base[:, 0])
        obstacle = ((base[:, 2] >= self.height_min) & (base[:, 2] <= self.height_max)
                    & (distance >= self.range_min) & (distance <= self.range_max)
                    & (angle >= self.angle_min) & (angle <= self.angle_max))
        indices = np.rint((angle[obstacle]-self.angle_min)/self.angle_increment).astype(int)
        minima = np.full(self.bins, np.inf)
        np.minimum.at(minima, indices, distance[obstacle])
        ranges[np.isfinite(minima)] = minima[np.isfinite(minima)]
        stats['obstacle_points'] = int(obstacle.sum())
        return dict(valid=True, reason='ok', ranges=ranges,
                    filtered_depth=filtered, statistics=stats)
