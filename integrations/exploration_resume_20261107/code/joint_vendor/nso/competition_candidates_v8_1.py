"""V8.1 routes: reconcile observed point-cloud bounds with measured occupancy cells.

The generic industrial asset prior is a separate scoring module. This module
does not read labels, object identities, world geometry, or future outcomes.
"""
import hashlib
import numpy as np
from scipy.sparse.csgraph import dijkstra
from env.virtual3d import camera_pose
from nso.response_features_v7 import observed_patches
from nso.response_candidates_v7 import _path, _actions
from nso.route_coverage_v2 import orientation_graph
from utils.grid_geometry import inflated_obstacles, visible_mask

XY_DIRECTIONS = np.array([[0., 1.], [1., 0.], [0., -1.], [-1., 0.]])


def measured_assets(mapper):
    """Snap the direction of the first real camera to a measured AABB axis.

    Bounds are empirical 5th/95th percentiles, not completed dimensions. The
    inferred far side is a geometric hypothesis, not a visibility certificate.
    Raw marker votes annotate these same geometry-defined clusters afterwards.
    """
    first_camera = mapper.keyframes[0].world_from_camera[:2, 3]
    result = []
    for patch in observed_patches(mapper):
        points = patch['points']
        lo, hi = np.quantile(points, [.05, .95], axis=0)
        center = (lo + hi) / 2
        front_index = int(np.argmax(XY_DIRECTIONS @ (first_camera - center[:2])))
        front = XY_DIRECTIONS[front_index]
        back = -front
        side = np.array([-back[1], back[0]])
        width = float((hi[:2] - lo[:2]) @ np.abs(side))
        depth = float((hi[:2] - lo[:2]) @ np.abs(back))
        height = float(hi[2] - lo[2])
        rear = center[:2] + back * depth / 2
        result.append({**patch, 'observed_low': lo, 'observed_high': hi,
                       'aabb_center': center, 'front_axis': front, 'back_axis': back,
                       'side_axis': side, 'rear_boundary_xy': rear,
                       'measured_width_m': width, 'measured_depth_m': depth,
                       'measured_height_m': height})
    return sorted(result, key=lambda a: (a['aabb_center'][0], a['aabb_center'][1], a['group']))


def aperture_support(asset, pose, config):
    """Bounded hypothetical far-plane image support; no hidden ray casting."""
    xy = pose[:2, 3]
    if (xy - asset['rear_boundary_xy']) @ asset['back_axis'] <= .15:
        return np.zeros(25)
    horizontal = np.linspace(-.5, .5, 5) * asset['measured_width_m']
    vertical = np.linspace(asset['observed_low'][2], asset['observed_high'][2], 5)
    offsets, heights = np.meshgrid(horizontal, vertical)
    points = np.column_stack([
        asset['rear_boundary_xy'][0] + offsets.ravel() * asset['side_axis'][0],
        asset['rear_boundary_xy'][1] + offsets.ravel() * asset['side_axis'][1],
        heights.ravel()])
    local = (points - pose[:3, 3]) @ pose[:3, :3]
    z = local[:, 2]
    tangent = np.tan(np.deg2rad(config.fov_deg / 2))
    visible = ((z > .15) & (z <= config.max_depth_m)
               & (np.abs(local[:, 0]) < z * tangent)
               & (np.abs(local[:, 1]) < z * tangent * config.height_px / config.width_px))
    distance = np.linalg.norm(points - pose[:3, 3], axis=1)
    return visible.astype(float) / (1 + (distance / config.max_depth_m) ** 2)


def measured_fov_support(assets, pose, config):
    points = np.concatenate([a['points'] for a in assets])
    local = (points - pose[:3, 3]) @ pose[:3, :3]
    z = local[:, 2]; tangent = np.tan(np.deg2rad(config.fov_deg / 2))
    return float(np.mean((z > .15) & (z <= config.max_depth_m)
                        & (np.abs(local[:, 0]) < z * tangent)
                        & (np.abs(local[:, 1]) < z * tangent * config.height_px / config.width_px)))



def navigation_rear_boundary(mapper, asset):
    """Conservative boundary of measured occupied cells near this geometry group.

    Cells have nonzero area. Taking only a point-cloud quantile can place a
    requested standoff inside the navigation map's inflated safety envelope.
    No unknown cell, true solid, label, or completed asset dimension is read.
    """
    r = mapper.config.resolution_m
    cells = np.argwhere(mapper.belief == 1)
    centers = np.column_stack([(cells[:, 1] + .5) * r,
                               (mapper.shape[0] - cells[:, 0] - .5) * r])
    lo, hi = asset['observed_low'][:2], asset['observed_high'][:2]
    local = ((centers >= lo - r) & (centers <= hi + r)).all(axis=1)
    rear = asset['rear_boundary_xy'].copy()
    if local.any():
        axis = asset['back_axis']
        bound = float(np.max(centers[local] @ axis) + r * .5 * np.sum(np.abs(axis)))
        rear += axis * max(0., bound - rear @ axis)
    return rear


def candidate_routes(mapper, obs, max_actions=48):
    if max_actions != 48:
        raise ValueError('V8 positive control fixes B=48')
    config = mapper.config
    safe = ~inflated_obstacles(mapper.belief != 0, config.robot_radius_m / config.resolution_m)
    audit = {'semantics_used_for_selection': False, 'truth_used_for_selection': False,
             'future_outcome_used': False, 'max_actions': max_actions,
             'safe_sha256': hashlib.sha256(safe.tobytes()).hexdigest()}
    assets = measured_assets(mapper)
    audit['measured_assets'] = [{k: v.tolist() if isinstance(v, np.ndarray) else v
                                for k, v in a.items() if k not in ('points', 'bits')}
                               for a in assets]
    if len(assets) != 2 or not safe[obs.position]:
        return [], {**audit, 'status': 'requires_two_measured_assets_and_safe_anchor'}
    graph, cells, ids = orientation_graph(safe)
    start_pose = (*map(int, obs.position), int(obs.heading))
    start = int(ids[obs.position]) * 4 + obs.heading
    outward, prev = dijkstra(graph, directed=True, indices=start, return_predecessors=True)
    backward, backprev = dijkstra(graph.T.tocsr(), directed=True, indices=start, return_predecessors=True)
    costs = outward + backward
    feasible = np.flatnonzero(np.isfinite(costs) & (costs <= max_actions) & (outward > 0))
    unknown = (mapper.belief == -1) & ~inflated_obstacles(mapper.belief == 1,
                                                         config.robot_radius_m / config.resolution_m)
    camera_unknown = ~mapper.camera_seen & (mapper.belief != 1)
    pool = []
    for state in feasible:
        cell = tuple(map(int, cells[state // 4])); heading = int(state % 4)
        pose = camera_pose(cell, heading, config, mapper.shape[0])
        radar = visible_mask(mapper.belief == 1, cell, heading,
                             int(config.max_depth_m / config.resolution_m), 360) & unknown
        camera = visible_mask(mapper.belief == 1, cell, heading,
                              int(config.max_depth_m / config.resolution_m), config.fov_deg) & camera_unknown
        pool.append({'state': int(state), 'pose': (*cell, heading), 'camera': pose,
                     'cost': int(costs[state]), 'unknown_cells': int(radar.sum() + camera.sum())})
    if not pool:
        return [], {**audit, 'status': 'no_safe_paid_route'}
    chosen = []; used = set(); missing = []

    def add(row, role, asset_index=None):
        if row is None:
            missing.append(role)
        else:
            chosen.append((row, role, asset_index)); used.add(row['state'])

    add(min(pool, key=lambda r: (-r['unknown_cells'] / r['cost'], r['cost'], r['pose'])), 'coverage_anchor')
    rotations = [r for r in pool if r['pose'][:2] == start_pose[:2] and r['state'] not in used
                 and measured_fov_support(assets, r['camera'], config) > 0]
    add(min(rotations, key=lambda r: (-measured_fov_support(assets, r['camera'], config),
                                      r['cost'], r['pose'])) if rotations else None, 'old_surface_rotation')
    target_audit = []
    navigation_rears = [navigation_rear_boundary(mapper, a) for a in assets]
    entry_depths = {}
    # Balanced quotas are fixed before any label scoring or actual evaluation.
    for distance, suffix in ((.4, 'entry'), (.8, 'deep')):
        for ai, asset in enumerate(assets):
            role = ('west' if ai == 0 else 'east') + '_' + suffix
            desired = navigation_rears[ai] + distance * asset['back_axis']
            heading = int(np.argmax(XY_DIRECTIONS @ asset['front_axis']))
            options = [r for r in pool if r['state'] not in used and r['pose'][2] == heading
                       and (r['camera'][:2, 3] - asset['rear_boundary_xy']) @ asset['back_axis'] > .15
                       and aperture_support(asset, r['camera'], config).max() > 0]
            if suffix == 'deep':
                options = [r for r in options if ai in entry_depths and
                           (r['camera'][:2, 3] - asset['rear_boundary_xy']) @ asset['back_axis']
                           >= entry_depths[ai] + .2 - 1e-12]
            selected = min(options, key=lambda r: (np.linalg.norm(r['camera'][:2, 3] - desired),
                                                    r['cost'], r['pose'])) if options else None
            # An unreachable target is a failed role, never silently replaced
            # with an unrelated cheap view labelled as a deep inspection.
            error = None if selected is None else float(np.linalg.norm(selected['camera'][:2, 3] - desired))
            if error is not None and error > .35:
                selected = None
            actual_depth = None if selected is None else float(
                (selected['camera'][:2, 3] - asset['rear_boundary_xy']) @ asset['back_axis'])
            if suffix == 'entry' and selected is not None:
                entry_depths[ai] = actual_depth
            add(selected, role, ai)
            target_audit.append({'role': role, 'desired_xy': desired.tolist(), 'nearest_error_m': error,
                                 'navigation_rear_boundary_xy': navigation_rears[ai].tolist(),
                                 'measured_point_rear_boundary_xy': asset['rear_boundary_xy'].tolist(),
                                 'actual_back_axis_offset_m': actual_depth,
                                 'aperture_support_fraction': None if selected is None else float(
                                     aperture_support(asset, selected['camera'], config).mean())})
    routes = []
    for cid, (row, role, ai) in enumerate(chosen):
        out = _path(prev, start, row['state'], cells)
        back = _path(backprev, start, row['state'], cells, reverse=True)
        states = out + back[1:]; actions = _actions(states)
        if states[0] != states[-1] or len(actions) != row['cost']:
            raise RuntimeError('directed round-trip accounting failed')
        routes.append({'candidate_id': cid, 'group': role, 'asset_index': ai,
                       'pose': list(row['pose']), 'states': [list(s) for s in states],
                       'actions': actions, 'cost': len(actions), 'arrival_action': len(out) - 1,
                       'is_coverage_anchor': role == 'coverage_anchor',
                       'endpoint_measured_fov_support': measured_fov_support(assets, row['camera'], config)})
    return routes, {**audit, 'status': 'complete' if not missing else 'missing_required_roles',
                    'missing_roles': missing, 'target_audit': target_audit,
                    'feasible_states': len(pool), 'all_costs_include_return_heading': True}
