"""Shared approach views around the two ends of an observed face.

This proposes a safe staging pose, never claims to see an unobserved rear
surface. Labels, hidden dimensions, rewards, and completed geometry are absent.
"""
import numpy as np
from utils.grid_geometry import supercover_line


def observed_support_fraction(points, row, mapper):
    """Frustum plus conservative 2-D visibility of actual measured points.

    Unknown cells block support rays. The measured endpoint cell itself may
    be occupied. This is a proposal filter, not a future RGB-D certificate.
    """
    if not len(points):
        return 0.
    config = mapper.config
    local = (points - row['camera'][:3, 3]) @ row['camera'][:3, :3]
    z = local[:, 2]
    tangent = np.tan(np.deg2rad(config.fov_deg / 2))
    keep = ((z > .15) & (z <= config.max_depth_m)
            & (np.abs(local[:, 0]) < z * tangent)
            & (np.abs(local[:, 1]) < z * tangent * config.height_px / config.width_px))
    start = tuple(row['pose'][:2])
    count = 0
    for point in points[keep]:
        end = (mapper.shape[0] - 1 - int(np.floor(point[1] / config.resolution_m)),
               int(np.floor(point[0] / config.resolution_m)))
        if not all(0 <= x < limit for x, limit in zip(end, mapper.shape)):
            continue
        if mapper.belief[end] == -1:
            continue
        ray = supercover_line(end[0] - start[0], end[1] - start[1])
        if all(mapper.belief[start[0] + dr, start[1] + dc] == 0 for dr, dc in ray[:-1]):
            count += 1
    return float(count / len(points))


def select_corner_view(asset, side_sign, pool, used, current, mapper, *, error_limit_m=.35):
    if side_sign not in (-1, 1):
        raise ValueError('corner side must be -1 or 1')
    clearance = max(.4, mapper.config.robot_radius_m + mapper.config.resolution_m)
    desired = (asset['aabb_center'][:2]
               + asset['front_axis'] * (asset['measured_depth_m'] / 2. + clearance)
               + side_sign * asset['side_axis'] * (asset['measured_width_m'] / 2. + clearance))
    audit = dict(desired_xy=desired.tolist(), side_sign=side_sign, clearance_m=float(clearance),
        error_limit_m=error_limit_m, nearest_error_m=None, measured_grid_visible_fraction=None,
        front_corner_only=True, hidden_surface_visibility_claimed=False,
        unknown_cells_block_support_rays=True, reason='no_unused_safe_translated_corner_pose')
    options = []
    for row in pool:
        if row['state'] in used or row['pose'][:2] == current[:2]:
            continue
        error = float(np.linalg.norm(row['camera'][:2, 3] - desired))
        if error > error_limit_m:
            continue
        audit['reason'] = 'no_observed_visible_face_from_feasible_corner_pose'
        support = observed_support_fraction(asset['points'], row, mapper)
        if support > 0:
            options.append((row, error, support))
    if not options:
        return None, audit
    row, error, support = min(options, key=lambda x: (x[1], -x[2], x[0]['cost'], x[0]['pose']))
    audit.update(reason='selected', nearest_error_m=error, measured_grid_visible_fraction=support)
    return row, audit
