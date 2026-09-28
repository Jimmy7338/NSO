"""Small common route pool from measured geometry and paid directed paths.

Every feasible grid pose is considered before selecting geometric roles. No
shape completion, semantic vote, world object center or outcome is consulted.
"""
import hashlib
import numpy as np
from scipy.sparse.csgraph import dijkstra
from env.virtual3d import camera_pose
from nso.response_features_v7 import observed_patches
from nso.route_coverage_v2 import orientation_graph
from utils.grid_geometry import DIRECTIONS, inflated_obstacles, visible_mask


def _path(previous, start, target, cells, reverse=False):
    chain = [int(target)]
    while chain[-1] != start:
        parent = int(previous[chain[-1]])
        if parent < 0 or len(chain) > len(previous):
            raise ValueError('invalid directed route')
        chain.append(parent)
    if not reverse:
        chain.reverse()
    return [(*map(int, cells[node // 4]), node % 4) for node in chain]


def _actions(states):
    actions = []
    for a, b in zip(states, states[1:]):
        if a[:2] == b[:2]:
            turn = (b[2] - a[2]) % 4
            if turn not in (1, 3):
                raise ValueError('route contains an unpaid or invalid rotation')
            actions.append('right' if turn == 1 else 'left')
        else:
            dr, dc = DIRECTIONS[a[2]]
            if b != (a[0] + dr, a[1] + dc, a[2]):
                raise ValueError('route contains invalid translation')
            actions.append('forward')
    return actions


def candidate_routes(mapper, obs, max_actions=48, max_candidates=6):
    if max_candidates != 6 or max_actions != 48:
        raise ValueError('V7 development contract fixes six candidates and 48 paid actions')
    c = mapper.config
    safe = ~inflated_obstacles(mapper.belief != 0, c.robot_radius_m / c.resolution_m)
    # The observed starting footprint must already be safe; do not clear it.
    if not safe[obs.position]:
        return [], {'status': 'initial_footprint_unknown_or_blocked', 'available_roles': []}
    graph, cells, ids = orientation_graph(safe)
    start_pose = (*map(int, obs.position), int(obs.heading))
    start = int(ids[obs.position]) * 4 + obs.heading
    outward, prev = dijkstra(graph, directed=True, indices=start, return_predecessors=True)
    backward, backprev = dijkstra(graph.T.tocsr(), directed=True, indices=start, return_predecessors=True)
    costs = outward + backward
    patches = observed_patches(mapper)
    # Clusters are geometrically ordered, never prioritized by marker presence.
    patch_order = sorted(range(len(patches)), key=lambda i: (-patches[i]['support_points'], patches[i]['group']))
    unknown = (mapper.belief == -1) & ~inflated_obstacles(mapper.belief == 1,
                                                       c.robot_radius_m / c.resolution_m)
    camera_unknown = ~mapper.camera_seen & (mapper.belief != 1)
    pool = []
    for state in np.flatnonzero(np.isfinite(costs) & (costs <= max_actions) & (outward >= 1)):
        cell = tuple(map(int, cells[state // 4])); heading = int(state % 4)
        pose = camera_pose(cell, heading, c, mapper.shape[0])
        radar = visible_mask(mapper.belief == 1, cell, heading, int(c.max_depth_m / c.resolution_m), 360) & unknown
        camera = visible_mask(mapper.belief == 1, cell, heading, int(c.max_depth_m / c.resolution_m), c.fov_deg) & camera_unknown
        memberships = []
        for pi in patch_order:
            patch = patches[pi]
            local = (patch['points'] - pose[:3, 3]) @ pose[:3, :3]
            z = local[:, 2]; tangent = np.tan(np.deg2rad(c.fov_deg / 2))
            fov = (z > .15) & (z <= c.max_depth_m) & (np.abs(local[:, 0]) < z * tangent)
            fov &= np.abs(local[:, 1]) < z * tangent * c.height_px / c.width_px
            if not fov.any():
                continue
            delta = pose[:2, 3] - patch['center'][:2]
            n = patch['normal_out_xy']
            angle = float(np.arctan2(n[0] * delta[1] - n[1] * delta[0], n @ delta))
            memberships.append({'patch': pi, 'angle': angle, 'support_fraction': float(fov.mean()),
                                'distance_m': float(np.linalg.norm(delta))})
        pool.append({'state': int(state), 'pose': (*cell, heading), 'cost': int(costs[state]),
                     'radar_cells': int(radar.sum()), 'camera_cells': int(camera.sum()), 'memberships': memberships})
    if not pool:
        return [], {'status': 'no_safe_paid_round_trip', 'available_roles': []}
    chosen = []; used = set(); role_audit = []

    def add(row, role, patch=None):
        if row is None or row['state'] in used or len(chosen) >= max_candidates:
            return
        chosen.append((row, role)); used.add(row['state'])
        role_audit.append({'role': role, 'patch': patch, 'pose': list(row['pose'])})

    anchor = min(pool, key=lambda r: (-(r['radar_cells'] + r['camera_cells']) / r['cost'], r['cost'], r['pose']))
    add(anchor, 'coverage_anchor')
    near = [r for r in pool if r['pose'][:2] == start_pose[:2] and r['memberships'] and r['state'] not in used]
    if near:
        add(min(near, key=lambda r: (-max(m['support_fraction'] for m in r['memberships']), r['cost'], r['pose'])),
            'old_surface_rotation')
    # A role cannot be filled by a view farther than 45 degrees from its target.
    roles = [('left', np.pi / 2), ('right', -np.pi / 2), ('back', np.pi), ('front', 0.)]
    missing = []
    missing_roles = [] if near else ['old_surface_rotation']
    allocated = {pi: 0 for pi in patch_order}
    for role, target_angle in roles:
        selected = None; selected_patch = None
        for pi in sorted(patch_order, key=lambda i: (allocated[i], -patches[i]['support_points'], patches[i]['group'])):
            if not patches[pi]['orientation_available']:
                continue
            candidates = []
            for row in pool:
                if row['state'] in used:
                    continue
                for member in row['memberships']:
                    if member['patch'] != pi:
                        continue
                    error = abs(np.arctan2(np.sin(member['angle'] - target_angle), np.cos(member['angle'] - target_angle)))
                    if error <= np.pi / 4 + 1e-12:
                        candidates.append((error, -member['support_fraction'], abs(member['distance_m'] - 1.4),
                                           row['cost'], row['pose'], row))
            if candidates:
                selected = min(candidates, key=lambda x: x[:-1])[-1]; selected_patch = pi
                break
        if selected is None:
            missing.append(role)
        else:
            allocated[selected_patch] += 1
        add(selected, role, selected_patch)
    # Preserve failed role counts; deterministic geometric fill is not relabeled
    # as a missing side/back observation.
    for row in sorted(pool, key=lambda r: (-r['camera_cells'], -r['radar_cells'], r['cost'], r['pose'])):
        add(row, 'geometric_fill')
        if len(chosen) == max_candidates:
            break
    routes = []
    for cid, (row, role) in enumerate(chosen):
        out = _path(prev, start, row['state'], cells)
        back = _path(backprev, start, row['state'], cells, reverse=True)
        states = out + back[1:]; actions = _actions(states)
        assert states[0] == states[-1] == start_pose and len(actions) == row['cost']
        routes.append({'candidate_id': cid, 'group': role, 'pose': list(row['pose']),
            'states': [list(s) for s in states], 'actions': actions, 'cost': len(actions),
            'arrival_action': len(out) - 1, 'is_coverage_anchor': role == 'coverage_anchor'})
    return routes, {'status': 'complete', 'safe_hash': hashlib.sha256(safe.tobytes()).hexdigest(),
                   'max_actions': max_actions, 'candidate_limit': max_candidates,
                   'feasible_poses_enumerated': len(pool), 'patch_count': len(patches),
                   'available_roles': role_audit, 'missing_direction_roles': missing,
                   'missing_roles': missing_roles + missing,
                   'direction_role_counts_by_patch': allocated,
                   'semantics_used_for_selection': False, 'outcome_used_for_selection': False,
                   'measured_frustum_is_true_visibility': False}
