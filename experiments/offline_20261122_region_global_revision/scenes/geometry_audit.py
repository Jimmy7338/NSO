#!/usr/bin/env python3
"""Audit source-only budgets, measured aperture witnesses and scene novelty."""

from collections import deque
import json

import numpy as np

from scene import ANALYSES, BUDGET, HERE, PRIOR_SCENES, RESOLUTION_M, SPECS, load_module, make_world


prior = load_module('regional_revision_geometry_audit_contract', PRIOR_SCENES/'geometry_audit.py')
read, sha = prior.read, prior.sha


def interior_visit_witness(edge_valid, reachable, rooms, anchor):
    """Find a budgeted swept-motion tour at least 0.4 m into every room."""
    entry_bits = np.zeros(reachable.shape, dtype=np.int16)
    for index, inside in enumerate(rooms.values()):
        entry_bits[inside & reachable] |= 1 << index
    rows, cols = reachable.shape
    poses = 4 * rows * cols
    full = (1 << len(rooms)) - 1

    def vertex(row, col, heading):
        return 4 * (row * cols + col) + heading

    def pose(index):
        cell, heading = divmod(index, 4)
        row, col = divmod(cell, cols)
        return row, col, heading

    neighbors = [()] * poses
    for row, col in np.argwhere(reachable):
        row, col = int(row), int(col)
        for head, (dr, dc) in enumerate(prior.DIRECTIONS):
            adjacency = [vertex(row, col, (head-1) % 4), vertex(row, col, (head+1) % 4)]
            if edge_valid[row, col, head]:
                adjacency.append(vertex(row+dr, col+dc, head))
            neighbors[vertex(row, col, head)] = adjacency

    anchor_pose = vertex(*anchor)
    origin = int(entry_bits[anchor[:2]]) * poses + anchor_pose
    terminal = full * poses + anchor_pose
    parents = np.full(poses*(full+1), -1, dtype=np.int32)
    parents[origin] = origin
    queue = deque([origin])
    for paid in range(BUDGET+1):
        for _ in range(len(queue)):
            state = queue.popleft()
            if state == terminal:
                walk = [pose(anchor_pose)]
                while state != origin:
                    state = int(parents[state])
                    walk.append(pose(state % poses))
                return {'fits_cap': True, 'actions': paid, 'poses_rc_heading': list(reversed(walk))}
            if paid == BUDGET:
                continue
            mask, state_pose = divmod(state, poses)
            for candidate in neighbors[state_pose]:
                row, col, _ = pose(candidate)
                admitted = (mask | int(entry_bits[row, col])) * poses + candidate
                if parents[admitted] == -1:
                    parents[admitted] = state
                    queue.append(admitted)
    return {'fits_cap': False, 'actions': None, 'searched_to_actions': BUDGET}


def scan_aperture(scan, doorway):
    """Count actual native laser rays travelling through a door and hitting each jamb."""
    axis = 0 if doorway['axis']=='vertical' else 1
    along = 1-axis
    line = doorway['wall_line_m']
    low, high = doorway['clear_interval_m']
    transform = scan.world_from_laser
    angles = scan.angle_min_rad + np.arange(len(scan.ranges_m))*scan.angle_increment_rad
    directions = np.column_stack((np.cos(angles), np.sin(angles), np.zeros_like(angles))) @ transform[:3,:3].T
    origin = transform[:3,3]
    sign = 1 if line > origin[axis] else -1
    toward = sign*directions[:,axis] > 1e-9
    crossing = (line-origin[axis]) / np.where(toward, directions[:,axis], np.inf)
    crossing_along = origin[along] + crossing * directions[:,along]
    recorded = scan.ranges_m < scan.range_max_m-1e-3
    hit = origin[None,:] + scan.ranges_m[:,None]*directions
    face = toward & recorded & (abs(hit[:,axis]-(line-.1*sign)) <= .04)
    through = toward & (crossing+.30 < scan.ranges_m) & (crossing_along > low+.12) & (crossing_along < high-.12)
    return {
        'rays_into_opening': int(through.sum()),
        'rays_on_lower_jamb_wall': int((face & (hit[:,along] >= low-.85) & (hit[:,along] <= low-.04)).sum()),
        'rays_on_upper_jamb_wall': int((face & (hit[:,along] <= high+.85) & (hit[:,along] >= high+.04)).sum()),
    }


def count_seed_archives():
    counts = 0
    used = {spec['seed'] for spec in SPECS.values()}
    for path in sorted(ANALYSES.rglob('initial_state.json')):
        if HERE.parent in path.parents:
            continue
        seed = read(path).get('seed')
        if seed is not None:
            counts += 1
            if seed in used:
                raise ValueError(f'Seed duplicated in archived cold fixture {path}')
    return counts


def main():
    archive, counts = prior.known_archive_signatures()
    seen = {}
    official = []
    for name, spec in SPECS.items():
        row = prior.check_geometry(name, archive, counts, seen)
        with np.load(HERE/'evaluator_references'/name/'evaluator_truth.npz', allow_pickle=False) as truth:
            regions, interiors, _, _ = prior.map_rooms(spec, truth['reachable'].shape)
            row['nearest_interior_all_room_tour'] = interior_visit_witness(
                truth['edge_valid'], truth['reachable'], interiors, tuple(row['start_rc_heading']))
            if not row['nearest_interior_all_room_tour']['fits_cap']:
                raise ValueError(f'No all-room interior visit and return within the release cap: {name}')
            if 'corridor' in name:
                world = make_world(name)
                cold = world.scan()
                for door in row['doors']:
                    center = sum(door['clear_interval_m'])/2
                    r = int(round(truth['reachable'].shape[0]-.5-center/RESOLUTION_M))
                    c = row['start_rc_heading'][1]
                    if not truth['reachable'][r,c] or list(spec['rooms'])[regions[r,c]] != 'hall':
                        raise ValueError(f'Door witness has no safe hall position: {door}')
                    world.position = (r,c)
                    entrance_scan = scan_aperture(world.scan(), door)
                    if min(entrance_scan.values()) < 1:
                        raise ValueError(f'Native scan cannot distinguish the open corridor portal: {door}')
                    door['scan_geometry_witness'] = {
                        'source_only_test_pose_rc_heading': [r,c,world.heading],
                        'aperture_returns': entrance_scan,
                        'returns_at_cold_start': scan_aperture(cold, door),
                    }
                assert world.step_count == 0
        official.append(row)
    record = {
        'status': 'geometry_and_visibility_verified_without_actor_execution',
        'policy_calls': 0,
        'archive_seed_records_checked': count_seed_archives(),
        'witness_clearance_m': .4,
        'source_only_no_truth_leak': True,
        'layout_read_order': list(SPECS),
        'layouts': official,
    }
    with (HERE/'formal_geometry_audit.json').open('x') as stream:
        json.dump(record, stream, indent=2, allow_nan=False)
        stream.write('\n')
    for row in official:
        print(row['layout'], row['nearest_interior_all_room_tour']['actions'], len(row['doors']), flush=True)


if __name__ == '__main__':
    main()
