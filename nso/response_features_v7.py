"""Measured-surface direction descriptors with fixed physical scaling.

This module never constructs completion geometry. A frustum descriptor is not
an assertion that a hidden surface will be visible. Category votes affect only
conditional feature columns; all target responses are supplied offline.
"""
from dataclasses import dataclass, asdict
import numpy as np
from scipy.ndimage import binary_closing, label
from env.grid_exploration import GridConfig
from env.virtual3d import camera_pose
from nso.camera_joint_planner_v2 import CameraJointPlannerV2
from utils.grid_geometry import visible_mask, inflated_obstacles
from utils.reconstruction_metrics import ray_scene


@dataclass(frozen=True)
class ResponseFeatureConfig:
    quality_voxel_m: float = .15
    min_support_points: int = 8
    max_horizontal_extent_m: float = 2.2
    min_vertical_extent_m: float = .25
    max_descriptor_points: int = 128
    unmarked_objectness: float = .25
    normal_fallback_threshold: float = .2


FEATURE_NAMES = ('radar_area_scaled_rate', 'camera_area_scaled_rate',
                 'observed_quality_rate', 'inverse_cost',
                 'front_support_rate', 'side_support_rate',
                 'back_support_rate', 'new_direction_support_rate',
                 'conditional_front_rate', 'conditional_side_rate',
                 'conditional_back_rate', 'conditional_new_direction_rate')


def observing_camera(mapper, cloud):
    """Choose the retained real view with most depth-consistent support."""
    records = []
    for i, frame in enumerate(mapper.keyframes):
        local = (cloud - frame.world_from_camera[:3, 3]) @ frame.world_from_camera[:3, :3]
        z = local[:, 2]
        u = np.rint(local[:, 0] / np.maximum(z, .01) * frame.intrinsic[0, 0] + frame.intrinsic[0, 2]).astype(int)
        v = np.rint(local[:, 1] / np.maximum(z, .01) * frame.intrinsic[1, 1] + frame.intrinsic[1, 2]).astype(int)
        valid = (z > .15) & (u >= 0) & (u < frame.depth_m.shape[1]) & (v >= 0) & (v < frame.depth_m.shape[0])
        ids = np.flatnonzero(valid)
        depth = frame.depth_m[v[ids], u[ids]]
        count = int(np.count_nonzero((depth > 0) & (np.abs(depth - z[ids]) <= mapper.config.truncation_m)))
        records.append((count, -i, frame.world_from_camera[:3, 3]))
    if not records:
        raise ValueError('measured patches require retained sensor history')
    best = max(records, key=lambda row: row[:2])
    return best[2], best[0]


def observed_patches(mapper, config=ResponseFeatureConfig()):
    """Geometry defines clusters; visible labels only annotate accepted ones."""
    q = mapper.quality_evidence(max_points=10000)
    if q is None:
        return []
    if not np.isclose(mapper.config.resolution_m, .2):
        raise ValueError('current measured ledger requires a 0.2 m grid')
    points = q['point']
    cells = np.column_stack([mapper.shape[0] - 1 - np.floor(points[:, 1] / .2),
                             np.floor(points[:, 0] / .2)]).astype(int)
    inside = ((cells >= 0) & (cells < np.array(mapper.shape))).all(axis=1)
    occupied = np.zeros(mapper.shape, bool)
    occupied[tuple(cells[inside].T)] = True
    groups, _ = label(binary_closing(occupied, structure=np.ones((3, 3))))
    group_ids = np.zeros(len(points), int)
    group_ids[inside] = groups[tuple(cells[inside].T)]
    patches = []
    for group in sorted(set(group_ids) - {0}):
        ids = np.flatnonzero(group_ids == group)
        cloud = points[ids]
        lo, hi = np.quantile(cloud, [.05, .95], axis=0)
        span = hi - lo
        if (len(ids) < config.min_support_points or
                max(span[:2]) > config.max_horizontal_extent_m or
                span[2] < config.min_vertical_extent_m):
            continue
        center = np.median(cloud, axis=0)
        # Use an actually observed camera to orient a measured normal; no box
        # thickness, hidden center, class-dependent axis or shape prior.
        initial_camera, normal_support = observing_camera(mapper, cloud)
        normal = np.mean(q['normal'][ids, :2], axis=0)
        fallback = bool(np.linalg.norm(normal) < config.normal_fallback_threshold)
        if fallback:
            normal = initial_camera[:2] - center[:2]
        elif normal @ (initial_camera[:2] - center[:2]) < 0:
            normal = -normal
        norm = np.linalg.norm(normal)
        if norm < 1e-10:
            normal = np.array([1., 0.])
        else:
            normal = normal / norm
        labels = q['label'][ids]
        known = labels[(labels == 2) | (labels == 3)]
        vote = float(np.mean(2 * (known == 3) - 1)) if len(known) else 0.
        sampled = np.linspace(0, len(ids) - 1, min(len(ids), config.max_descriptor_points), dtype=int)
        patches.append({'group': int(group), 'center': center, 'normal_out_xy': normal,
            'points': cloud[sampled], 'bits': q['bits'][ids][sampled],
            'support_points': len(ids), 'support_m2_proxy': len(ids) * config.quality_voxel_m**2,
            'marked_points': len(known), 'class_vote': vote,
            'normal_fallback': fallback, 'normal_sign_depth_support': normal_support,
            'orientation_available': normal_support > 0,
            'observed_span': span})
    return patches


def state_patch_descriptor(patch, pose, config):
    """Four per-point bounded descriptors; directions use each point's origin."""
    points = patch['points']
    local = (points - pose[:3, 3]) @ pose[:3, :3]
    z = local[:, 2]
    tangent = np.tan(np.deg2rad(config.fov_deg / 2))
    fov = (z > .15) & (z <= config.max_depth_m)
    fov &= np.abs(local[:, 0]) < z * tangent
    fov &= np.abs(local[:, 1]) < z * tangent * config.height_px / config.width_px
    delta = pose[:3, 3] - points
    xy_norm = np.linalg.norm(delta[:, :2], axis=1)
    cosine = np.clip(delta[:, :2] @ patch['normal_out_xy'] / np.maximum(xy_norm, 1e-10), -1, 1)
    distance = np.linalg.norm(delta, axis=1)
    proximity = 1 / (1 + (distance / config.max_depth_m)**2)
    angle = np.arctan2(delta[:, 1], delta[:, 0])
    sector = np.floor((angle + np.pi) / (2 * np.pi) * 8).astype(int) % 8
    novelty = (patch['bits'] & (1 << sector)) == 0
    result = np.column_stack([np.maximum(cosine, 0), np.sqrt(np.maximum(0, 1 - cosine**2)),
                              np.maximum(-cosine, 0), novelty.astype(float)])
    if not patch.get('orientation_available', True):
        result[:, :3] = 0.
    result *= (fov * proximity)[:, None]
    if not np.isfinite(result).all() or (result < 0).any() or (result > 1 + 1e-12).any():
        raise ValueError('invalid bounded geometric descriptor')
    return result


def response_features(mapper, routes, config=ResponseFeatureConfig()):
    """Matrices have common4 + object4 + interaction4, with explicit zeros.

    G uses geometry-only objects; O/S share marker objectness. S/X use raw
    visible category vote / its swap, not a completion-model posterior. N has
    only the common route features. G_capacity adds fixed quadratic geometric
    descriptors instead of class interactions. M explicitly uses absent input
    and restores G; the model bank separately enforces that fallback.
    """
    c = mapper.config
    patches = observed_patches(mapper, config)
    grid = GridConfig(resolution_m=c.resolution_m, robot_radius_m=c.robot_radius_m,
                      sensor_range_m=c.max_depth_m, sensor_fov_deg=360, max_steps=c.max_steps)
    planner = CameraJointPlannerV2(grid, c, semantic=False, route=False,
                                  quality_weight=1., semantic_weight=0.)
    planner.set_mapping(mapper)
    planner.quality = mapper.quality_evidence()
    mesh = mapper.mesh()
    planner.ray = ray_scene(mesh) if len(mesh.triangles) else None
    npoints = max(1, len(planner.quality['point']) if planner.quality is not None else 0)
    radar_unknown = (mapper.belief == -1) & ~inflated_obstacles(mapper.belief == 1,
                                               c.robot_radius_m / c.resolution_m)
    camera_unknown = ~mapper.camera_seen & (mapper.belief != 1)
    states = sorted({tuple(s) for route in routes for s in route['states'][1:]})
    cache = {}
    for state in states:
        cell, heading = state[:2], state[2]
        pose = camera_pose(cell, heading, c, mapper.shape[0])
        radar = visible_mask(mapper.belief == 1, cell, heading,
                             int(c.max_depth_m / c.resolution_m), 360) & radar_unknown
        camera = visible_mask(mapper.belief == 1, cell, heading,
                              int(c.max_depth_m / c.resolution_m), c.fov_deg) & camera_unknown
        quality, _, _ = planner._quality_gain(cell, heading)
        descriptors = [state_patch_descriptor(p, pose, c) for p in patches]
        cache[state] = (radar, camera, quality, descriptors)
    matrices = {name: [] for name in ('G', 'O', 'S', 'X', 'M', 'N', 'G_capacity')}
    rows = []
    area_scale = c.max_depth_m**2
    for route in routes:
        cost = route['cost']
        if cost < 1 or cost != len(route['states']) - 1:
            raise ValueError('all feature routes require exact paid action counts')
        radar = np.zeros(mapper.shape, bool)
        camera = np.zeros(mapper.shape, bool)
        quality = np.zeros(0 if planner.quality is None else len(planner.quality['point']))
        maxima = [np.zeros((len(p['points']), 4)) for p in patches]
        for raw in route['states'][1:]:
            r, cam, q, descriptors = cache[tuple(raw)]
            radar |= r; camera |= cam; quality = np.maximum(quality, q)
            maxima = [np.maximum(a, b) for a, b in zip(maxima, descriptors)]
        common = np.array([radar.sum() * c.resolution_m**2 / area_scale,
                           camera.sum() * c.resolution_m**2 / area_scale,
                           quality.sum() / npoints, 1.]) / cost
        descriptor = np.array([p['support_m2_proxy'] * m.mean(axis=0) / area_scale
                               for p, m in zip(patches, maxima)]).reshape(-1, 4)
        geometrical = descriptor.sum(axis=0) / cost
        weight = np.array([1. if p['marked_points'] else config.unmarked_objectness for p in patches])
        votes = np.array([p['class_vote'] for p in patches])
        objects = (descriptor * weight[:, None]).sum(axis=0) / cost
        conditional = (descriptor * (weight * votes)[:, None]).sum(axis=0) / cost
        capacity = (descriptor**2).sum(axis=0) / cost
        zero = np.zeros(4)
        matrices['G'].append(np.r_[common, geometrical, zero])
        matrices['O'].append(np.r_[common, objects, zero])
        matrices['S'].append(np.r_[common, objects, conditional])
        matrices['X'].append(np.r_[common, objects, -conditional])
        matrices['M'].append(np.r_[common, geometrical, zero])
        matrices['N'].append(np.r_[common, zero, zero])
        matrices['G_capacity'].append(np.r_[common, geometrical, capacity])
        rows.append({'candidate_id': route['candidate_id'], 'cost': cost,
                     'unscaled_patch_descriptors': (descriptor * area_scale).tolist()})
    matrices = {name: np.asarray(values).reshape(-1, 12) for name, values in matrices.items()}
    if any(not np.isfinite(x).all() for x in matrices.values()):
        raise ValueError('nonfinite sensor features')
    return matrices, {'feature_config': asdict(config), 'feature_names': list(FEATURE_NAMES),
        'physical_area_scale_m2': area_scale, 'raw_category_vote_not_calibrated_probability': True,
        'hidden_geometry_model_used': False, 'frustum_is_occlusion_certificate': False,
        'point_directions_use_individual_surface_points': True,
        'patches': [{key: value.tolist() if isinstance(value, np.ndarray) else value
                    for key, value in p.items() if key not in ('points', 'bits')} for p in patches],
        'routes': rows}
