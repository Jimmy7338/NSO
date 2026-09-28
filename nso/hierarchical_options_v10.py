"""Observed-geometry options with a fixed task return anchor and live budget.

Only the outbound part is committed by the runtime. The full states/actions
also contain a *planned* return; no returned sensor support is credited here.
Object count and available option count are variable. Labels never rank or
filter options. Existing measured-patch geometry currently uses a0.2m grid.
"""
import hashlib
from numbers import Integral

import numpy as np
from scipy.sparse.csgraph import dijkstra

from env.virtual3d import camera_pose
from nso.competition_candidates_v8_1 import (
    XY_DIRECTIONS, aperture_support, measured_assets, navigation_rear_boundary,
)
from nso.response_candidates_v7 import _actions, _path
from nso.route_coverage_v2 import orientation_graph
from utils.grid_geometry import inflated_obstacles, visible_mask


# Existing physical geometry rules, carried forward without effect-based tuning.
VIEW_OFFSETS_M = (('entry', .4), ('deep', .8))
TARGET_ERROR_LIMIT_M = .35
MIN_DEEP_SEPARATION_M = .2


def _integer(value, name, minimum=0):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Integral) or value < minimum:
        raise ValueError(f'{name} must be an integer >= {minimum}')
    return int(value)


def _pose(position, heading, name):
    if len(position) != 2:
        raise ValueError(f'{name} position must contain row and column')
    row, col = (_integer(value, name + ' position') for value in position)
    heading = _integer(heading, name + ' heading')
    if heading > 3:
        raise ValueError(f'{name} heading must be0,1,2,3')
    return row, col, heading


def _fov_support(points, pose, config):
    if not len(points):
        return 0.
    local = (points - pose[:3, 3]) @ pose[:3, :3]
    z = local[:, 2]
    tangent = np.tan(np.deg2rad(config.fov_deg / 2))
    visible = ((z > .15) & (z <= config.max_depth_m)
               & (np.abs(local[:, 0]) < z * tangent)
               & (np.abs(local[:, 1]) < z * tangent * config.height_px / config.width_px))
    return float(np.mean(visible))


def _geometry_assets(mapper, q):
    """Keep measured geometry only; class annotations cannot affect selection."""
    if q is None or not len(q['point']):
        return []
    if not mapper.keyframes:
        raise ValueError('measured object geometry requires actual retained camera history')
    fields = ('group', 'observed_low', 'observed_high', 'aabb_center', 'front_axis',
              'back_axis', 'side_axis', 'rear_boundary_xy', 'measured_width_m',
              'measured_depth_m', 'measured_height_m', 'points', 'support_points')
    return [{name: row[name] for name in fields} for row in measured_assets(mapper)]


def generate_options(mapper, position, heading, remaining_budget, return_anchor,
                     max_candidates=12, *, coverage_strategy='rate_single',
                     coverage_slots=4, attempted_camera_mask=None):
    """Return full paid option plans and an observed-input audit.

    ``states/actions``: current -> target -> fixed return anchor, including its
    heading. ``outbound_actions`` and ``return_actions`` are action lists.
    ``outbound_cost + return_cost == cost <= remaining_budget``. At least one
    outbound action is required; same-cell targets with a different heading
    are valid. Runtime executes outbound actions only and checks the latest
    map before every action. Missing object views are recorded, never filled
    with falsely labelled poses. No world, true objects, or rewards are read.
    """
    budget = _integer(remaining_budget, 'remaining_budget')
    limit = _integer(max_candidates, 'max_candidates', 1)
    if coverage_strategy not in ('rate_single', 'total_diverse'):
        raise ValueError('unknown coverage candidate strategy')
    coverage_slots = _integer(coverage_slots, 'coverage_slots', 1)
    current = _pose(position, heading, 'current')
    if len(return_anchor) != 3:
        raise ValueError('return_anchor must contain row, column and heading')
    anchor = _pose(return_anchor[:2], return_anchor[2], 'return anchor')
    belief = np.asarray(mapper.belief)
    if belief.ndim != 2 or tuple(belief.shape) != tuple(mapper.shape) or not np.isin(belief, [-1, 0, 1]).all():
        raise ValueError('expected a matching unknown/free/occupied observed grid')
    seen = np.asarray(mapper.camera_seen)
    if seen.shape != belief.shape or not np.isin(seen, [0, 1]).all():
        raise ValueError('camera_seen must be an observed boolean map of the same shape')
    attempted = np.zeros_like(seen, dtype=bool) if attempted_camera_mask is None else np.asarray(attempted_camera_mask)
    if attempted.shape != belief.shape or attempted.dtype != np.bool_:
        raise ValueError('attempted_camera_mask must be a boolean map of the same shape')
    config = mapper.config
    resolution, radius = float(config.resolution_m), float(config.robot_radius_m)
    if not np.isfinite([resolution, radius]).all() or resolution <= 0 or radius < 0:
        raise ValueError('invalid robot/map scale')
    if not np.isfinite(config.max_depth_m) or config.max_depth_m <= 0:
        raise ValueError('invalid camera/radar range')
    safe = ~inflated_obstacles(belief != 0, radius / resolution)
    audit = {
        'schema_version': 'hierarchical_options_v10/1',
        'remaining_budget': budget, 'max_candidates': limit,
        'current_pose': list(current), 'return_anchor': list(anchor),
        'safe_sha256': hashlib.sha256(safe.tobytes()).hexdigest(),
        'belief_sha256': hashlib.sha256(belief.tobytes()).hexdigest(),
        'camera_seen_sha256': hashlib.sha256(seen.tobytes()).hexdigest(),
        'labels_used_for_selection': False, 'truth_used_for_selection': False,
        'future_outcome_used': False, 'all_costs_include_return_heading': True,
        'coverage_strategy': coverage_strategy, 'coverage_slots': coverage_slots,
        'attempted_camera_sha256': hashlib.sha256(attempted.tobytes()).hexdigest(),
        'runtime_commits_only_outbound': True, 'planned_return_is_observed_evidence': False,
        'missing_roles': [], 'omitted_by_cap': [], 'target_audit': [], 'common_role_audit': [],
        'measured_assets': [], 'observed_asset_count': 0,
        'feasible_states': 0, 'minimum_current_return_cost': None,
    }

    def stop(reason):
        return [], {**audit, 'status': reason, 'available_roles': []}

    def inside(pose):
        return 0 <= pose[0] < safe.shape[0] and 0 <= pose[1] < safe.shape[1]

    if not inside(current) or not safe[current[:2]]:
        return stop('current_footprint_not_known_safe')
    if not inside(anchor) or not safe[anchor[:2]]:
        return stop('anchor_footprint_not_known_safe')
    graph, cells, ids = orientation_graph(safe)
    start = int(ids[current[:2]]) * 4 + current[2]
    home = int(ids[anchor[:2]]) * 4 + anchor[2]
    outward, predecessor = dijkstra(graph, directed=True, indices=start, return_predecessors=True)
    to_anchor, return_predecessor = dijkstra(graph.T.tocsr(), directed=True, indices=home,
                                           return_predecessors=True)
    if not np.isfinite(to_anchor[start]):
        return stop('anchor_disconnected_in_observed_map')
    audit['minimum_current_return_cost'] = int(round(to_anchor[start]))
    if to_anchor[start] > budget:
        return stop('current_return_exceeds_remaining_budget')
    if budget == 0:
        return stop('no_paid_action_budget')
    costs = outward + to_anchor
    feasible = np.flatnonzero(np.isfinite(costs) & (outward > 0) & (costs <= budget))
    if not len(feasible):
        return stop('no_paid_option_with_reserved_return')
    audit['feasible_states'] = len(feasible)
    q = mapper.quality_evidence(max_points=10000)
    points = np.empty((0, 3)) if q is None else np.asarray(q['point'])
    if points.ndim != 2 or points.shape[1] != 3 or not np.isfinite(points).all():
        raise ValueError('invalid measured support points')
    assets = _geometry_assets(mapper, q)
    audit['observed_asset_count'] = len(assets)
    audit['measured_assets'] = [{name: value.tolist() if isinstance(value, np.ndarray) else value
                                 for name, value in asset.items() if name != 'points'} for asset in assets]
    unknown = (belief == -1) & ~inflated_obstacles(belief == 1, radius / resolution)
    camera_unknown = ~seen.astype(bool) & (belief != 1) & ~attempted
    occupied = belief == 1
    range_cells = int(config.max_depth_m / resolution)
    has_unknown = bool(unknown.any()) and range_cells > 0
    has_camera_unknown = bool(camera_unknown.any()) and range_cells > 0
    pool = []
    for node in feasible:
        cell = tuple(map(int, cells[node // 4])); direction = int(node % 4)
        pose = camera_pose(cell, direction, config, belief.shape[0])
        radar_count = int(np.count_nonzero(visible_mask(occupied, cell, direction, range_cells, 360) & unknown)) if has_unknown else 0
        camera_count = int(np.count_nonzero(visible_mask(occupied, cell, direction, range_cells, config.fov_deg) & camera_unknown)) if has_camera_unknown else 0
        pool.append({'state': int(node), 'pose': (*cell, direction), 'camera': pose,
                     'cost': int(round(costs[node])), 'unknown_cells': radar_count + camera_count,
                     'outbound_cost': int(round(outward[node])),
                     'radar_unknown_cells': radar_count, 'camera_unknown_cells': camera_count})
    chosen, used = [], set()

    def add(row, role, asset_index=None):
        if len(chosen) >= limit:
            audit['omitted_by_cap'].append(role)
            return False
        if row is None:
            audit['missing_roles'].append(role)
            return False
        chosen.append((row, role, asset_index)); used.add(row['state'])
        return True

    if coverage_strategy == 'rate_single':
        coverage_rows = [min(pool, key=lambda row: (-row['unknown_cells'] / row['cost'], row['cost'], row['pose']))]
        coverage_reason = 'highest_shared_unknown_proxy_per_reserved_cost'
    else:
        # Candidate generation must not optimize a different rate objective
        # before the global total-gain scorer sees the alternatives. Retain
        # several distinct target cells, ordered by raw observable support.
        ranked = sorted(pool, key=lambda row: (-row['unknown_cells'], row['outbound_cost'],
                                               row['cost'], row['pose']))
        coverage_rows, target_cells = [], set()
        for row in ranked:
            if row['pose'][:2] in target_cells:
                continue
            coverage_rows.append(row); target_cells.add(row['pose'][:2])
            if len(coverage_rows) >= min(coverage_slots, limit):
                break
        if not coverage_rows:
            coverage_rows = [min(pool, key=lambda row: (row['cost'], row['pose']))]
        coverage_reason = 'top_total_observable_support_with_distinct_target_cells'
    for index, coverage in enumerate(coverage_rows):
        role = 'coverage_anchor' if coverage_strategy == 'rate_single' else f'coverage_total_{index}'
        added = add(coverage, role)
        audit['common_role_audit'].append({'role': role, 'reason': coverage_reason if added else 'candidate_cap_reached',
                                         'pose': list(coverage['pose']) if added else None,
                                         'planned_paid_cost': coverage['cost'],
                                         'outbound_cost': coverage['outbound_cost'],
                                         'radar_unknown_cells': coverage['radar_unknown_cells'],
                                         'camera_unknown_cells': coverage['camera_unknown_cells'],
                                         'radar_camera_proxies_may_overlap': True,
                                         'positive_unknown_proxy': coverage['unknown_cells'] > 0})
    rotations = [row for row in pool if row['pose'][:2] == current[:2] and row['state'] not in used
                 and _fov_support(points, row['camera'], config) > 0]
    rotation = min(rotations, key=lambda row: (-_fov_support(points, row['camera'], config),
                                               row['cost'], row['pose'])) if rotations else None
    rotation_added = add(rotation, 'old_surface_rotation')
    audit['common_role_audit'].append({'role': 'old_surface_rotation',
                                     'reason': 'candidate_cap_reached' if 'old_surface_rotation' in audit['omitted_by_cap']
                                     else 'selected' if rotation_added else 'no_unused_paid_rotation_with_measured_support',
                                     'pose': list(rotation['pose']) if rotation_added else None,
                                     'measured_fov_support': _fov_support(points, rotation['camera'], config) if rotation_added else None})
    navigation_rears = [navigation_rear_boundary(mapper, asset) for asset in assets]
    entry_depths = {}
    # First offer one entry per observed geometry group, then one deeper view.
    # Group order is geometric and identical under all label interpretations.
    for suffix, offset in VIEW_OFFSETS_M:
        for ai, asset in enumerate(assets):
            role = f'asset_{ai}_{suffix}'
            desired = navigation_rears[ai] + offset * asset['back_axis']
            record = {'role': role, 'asset_index': ai, 'desired_xy': desired.tolist(),
                      'navigation_rear_boundary_xy': navigation_rears[ai].tolist(),
                      'measured_point_rear_boundary_xy': asset['rear_boundary_xy'].tolist(),
                      'nearest_error_m': None, 'actual_back_axis_offset_m': None,
                      'aperture_support_fraction': None}
            if len(chosen) >= limit:
                audit['omitted_by_cap'].append(role)
                audit['target_audit'].append({**record, 'reason': 'candidate_cap_reached'})
                continue
            if suffix == 'deep' and ai not in entry_depths:
                audit['missing_roles'].append(role)
                audit['target_audit'].append({**record, 'reason': 'deep_requires_available_entry'})
                continue
            wanted_heading = int(np.argmax(XY_DIRECTIONS @ asset['front_axis']))
            options = []
            for row in pool:
                if row['state'] in used or row['pose'][2] != wanted_heading:
                    continue
                depth = float((row['camera'][:2, 3] - asset['rear_boundary_xy']) @ asset['back_axis'])
                if depth <= .15 or aperture_support(asset, row['camera'], config).max() <= 0:
                    continue
                if suffix == 'deep' and depth < entry_depths[ai] + MIN_DEEP_SEPARATION_M - 1e-12:
                    continue
                options.append(row)
            selected = min(options, key=lambda row: (np.linalg.norm(row['camera'][:2, 3] - desired),
                                                       row['cost'], row['pose'])) if options else None
            reason = 'no_unused_feasible_aperture_pose'
            if selected is not None:
                error = float(np.linalg.norm(selected['camera'][:2, 3] - desired))
                record['nearest_error_m'] = error
                reason = 'selected' if error <= TARGET_ERROR_LIMIT_M else 'target_projection_error_exceeds_limit'
                if error > TARGET_ERROR_LIMIT_M:
                    selected = None
            if selected is not None:
                depth = float((selected['camera'][:2, 3] - asset['rear_boundary_xy']) @ asset['back_axis'])
                record['actual_back_axis_offset_m'] = depth
                record['aperture_support_fraction'] = float(aperture_support(asset, selected['camera'], config).mean())
                if suffix == 'entry': entry_depths[ai] = depth
            add(selected, role, ai)
            audit['target_audit'].append({**record, 'reason': reason})
    routes = []
    for cid, (row, role, ai) in enumerate(chosen):
        outbound = _path(predecessor, start, row['state'], cells)
        returning = _path(return_predecessor, home, row['state'], cells, reverse=True)
        states = outbound + returning[1:]
        out_actions, return_actions = _actions(outbound), _actions(returning)
        actions = out_actions + return_actions
        if (states[0] != current or states[-1] != anchor or returning[0] != outbound[-1]
                or not out_actions or len(actions) != row['cost'] or len(actions) > budget):
            raise RuntimeError('directed option / fixed-anchor return accounting failed')
        routes.append({'candidate_id': cid, 'group': role, 'asset_index': ai,
                       'pose': list(row['pose']), 'states': [list(state) for state in states],
                       'actions': actions, 'outbound_actions': out_actions,
                       'return_actions': return_actions, 'return_states': [list(state) for state in returning],
                       'outbound_states': [list(state) for state in outbound],
                       'cost': len(actions), 'outbound_cost': len(out_actions),
                       'return_cost': len(return_actions), 'arrival_action': len(out_actions),
                       'return_anchor': list(anchor), 'is_coverage_anchor': role == 'coverage_anchor',
                       'endpoint_measured_fov_support': _fov_support(points, row['camera'], config)})
    return routes, {**audit, 'status': 'options_available',
                    'available_roles': [route['group'] for route in routes],
                    'candidate_count': len(routes), 'all_target_poses_unique': len(used) == len(routes)}
