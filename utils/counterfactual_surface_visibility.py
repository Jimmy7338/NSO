"""Evaluator-only visibility and unique physical-area accounting.

Inputs are a fixed reference sampled from the unique exterior of the ground
truth solid union. This module must not be imported by a planning policy.
Visibility is geometric first-hit visibility with nearest-pixel depth validity;
it is not reconstruction correctness or a test of estimated depth agreement.
"""
from collections.abc import Iterable

import numpy as np
import open3d as o3d

from utils.rgbd_contract import RGBDFrame


# Keep the simulator's optical-depth near plane and the frozen evaluator's
# dimensionless tolerance for rays whose direction is (reference - origin).
MIN_AXIAL_DEPTH_M = .15
FIRST_HIT_PARAMETER_TOLERANCE = 1e-4


def _physical_pose(value):
    pose = np.asarray(value, dtype=float)
    if pose.shape != (4, 4) or not np.isfinite(pose).all():
        raise ValueError('physical camera pose must be a finite 4x4 matrix')
    rotation = pose[:3, :3]
    if (not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-5, rtol=0)
            or not np.isclose(np.linalg.det(rotation), 1., atol=1e-5, rtol=0)
            or not np.allclose(pose[3], [0, 0, 0, 1], atol=1e-8, rtol=0)):
        raise ValueError('physical camera pose must be a rigid homogeneous transform')
    return pose


def reference_visible(points, frame: RGBDFrame, truth_ray,
                      physical_world_from_camera=None, max_depth_m=None):
    """Return an N-element visibility mask for a fixed world-frame reference.

    The evaluator should supply the independent *physical* camera pose. Omitting
    it uses ``frame.world_from_camera`` and is appropriate only when the caller
    has established zero pose error. Passing an explicit pose never consults the
    estimated frame pose. RGB, semantics and object identities are not read.

    ``max_depth_m`` is the sensor's optical-axis z limit, matching VirtualWorld
    sensing, not Euclidean distance. None applies no upper range limit; callers
    evaluating the simulator must pass its configured max_depth_m explicitly.
    The near limit is the simulator's strict z > 0.15m. Intrinsics define pixel
    support [-0.5, width-0.5) x [-0.5, height-0.5); the nearest pixel must contain
    finite, positive, in-range depth. Invalid depth pixels give no observation.

    A reference point is geometrically visible only when the truth mesh's first
    intersection has abs(t_hit - 1) < 1e-4 for the unnormalised reference ray.
    This retains the existing evaluator's dimensionless, distance-dependent
    tolerance. Nearest-pixel validity plus continuous reference rays remains a
    finite-image approximation, especially at thin boundaries; no pixel-area
    reconstruction guarantee or photometric/depth-consistency claim is made.
    """
    points = np.asarray(points, dtype=float)
    if points.ndim != 2 or points.shape[1] != 3 or not np.isfinite(points).all():
        raise ValueError('reference points must be a finite Nx3 array')
    depth = np.asarray(frame.depth_m)
    if depth.ndim != 2 or not all(depth.shape) or depth.dtype.kind not in 'fiu':
        raise ValueError('depth must be a nonempty numeric HxW image')
    intrinsic = np.asarray(frame.intrinsic, dtype=float)
    if (intrinsic.shape != (3, 3) or not np.isfinite(intrinsic).all()
            or intrinsic[0, 0] <= 0 or intrinsic[1, 1] <= 0
            or not np.allclose(intrinsic[2], [0, 0, 1], atol=1e-8, rtol=0)):
        raise ValueError('intrinsic must be a finite optical-camera calibration')
    pose = _physical_pose(frame.world_from_camera if physical_world_from_camera is None
                          else physical_world_from_camera)
    if max_depth_m is not None:
        if not np.isscalar(max_depth_m) or not np.isfinite(max_depth_m) or max_depth_m <= MIN_AXIAL_DEPTH_M:
            raise ValueError('max_depth_m must be finite and above the near plane')
        max_depth_m = float(max_depth_m)

    visible = np.zeros(len(points), dtype=bool)
    if not len(points):
        return visible
    local = (points - pose[:3, 3]) @ pose[:3, :3]
    z = local[:, 2]
    possible = z > MIN_AXIAL_DEPTH_M
    if max_depth_m is not None:
        possible &= z <= max_depth_m
    ids = np.flatnonzero(possible)
    if not len(ids):
        return visible
    projected = local[ids] @ intrinsic.T
    u, v = projected[:, 0] / z[ids], projected[:, 1] / z[ids]
    height, width = depth.shape
    in_image = ((u >= -.5) & (u < width - .5)
                & (v >= -.5) & (v < height - .5))
    ids, u, v = ids[in_image], u[in_image], v[in_image]
    if not len(ids):
        return visible
    columns, rows = np.floor(u + .5).astype(int), np.floor(v + .5).astype(int)
    measured = depth[rows, columns]
    valid = np.isfinite(measured) & (measured > MIN_AXIAL_DEPTH_M)
    if max_depth_m is not None:
        valid &= measured <= max_depth_m
    ids = ids[valid]
    if not len(ids):
        return visible
    rays = np.column_stack([np.tile(pose[:3, 3], (len(ids), 1)),
                            points[ids] - pose[:3, 3]]).astype(np.float32)
    hit = truth_ray.cast_rays(o3d.core.Tensor(rays), nthreads=1)['t_hit'].numpy()
    visible[ids] = np.isfinite(hit) & (np.abs(hit - 1.) < FIRST_HIT_PARAMETER_TOLERANCE)
    return visible


def _mask(value, name):
    result = np.asarray(value)
    if result.ndim != 1 or result.dtype != np.bool_:
        raise ValueError(f'{name} must be a one-dimensional boolean mask')
    return result


def _weights(weights, size):
    result = np.asarray(weights, dtype=float)
    if result.ndim == 0:
        if not np.isfinite(result) or result < 0:
            raise ValueError('weights must be finite and nonnegative')
        result = np.full(size, float(result))
    if result.shape != (size,) or not np.isfinite(result).all() or np.any(result < 0):
        raise ValueError('weights must be finite nonnegative scalar or N-vector')
    return result


def surface_increment(prefix_seen, future_seen, weights):
    """Count each newly visible fixed reference at most once, in square metres.

    ``weights`` is either one nonnegative physical-area weight per reference, or
    a shared scalar such as union_area / *initial* sample_count. Do not divide by
    the smaller count left after observability filtering. No semantic weighting
    or re-normalisation is performed here. Inputs are not mutated.
    """
    prefix = _mask(prefix_seen, 'prefix_seen')
    future = _mask(future_seen, 'future_seen')
    if prefix.shape != future.shape:
        raise ValueError('prefix and future must index the same fixed reference')
    area = _weights(weights, len(prefix))
    return float(area[future & ~prefix].sum())


def surface_stage_increments(prefix_seen, stage_masks: Iterable, weights):
    """Deduplicate successive outbound, endpoint and return unions.

    Each stage mask must already be the union of that stage's frame visibility.
    Returns ``{'increments': list[float], 'total': float, 'union_seen': bool[N]}``.
    A stage only receives area not covered by the prefix or an earlier stage.
    Stage order affects attribution but never the total. Inputs are not mutated.
    """
    prefix = _mask(prefix_seen, 'prefix_seen')
    area = _weights(weights, len(prefix))
    seen = prefix.copy()
    increments = []
    for value in stage_masks:
        stage = _mask(value, 'stage mask')
        if stage.shape != prefix.shape:
            raise ValueError('all stages must index the same fixed reference')
        increments.append(float(area[stage & ~seen].sum()))
        seen |= stage
    return dict(increments=increments, total=float(area[seen & ~prefix].sum()),
                union_seen=seen)
