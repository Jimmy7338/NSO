#!/usr/bin/env python3
"""Reproject the saved west-screened motion and inspect the observed return cut."""

import argparse
from dataclasses import asdict
import math
from pathlib import Path
import sys

import numpy as np

HERE = Path(__file__).resolve().parent
OLD = HERE.parents[2] / 'region_structure_floor_20261122' / 'local'
sys.path.insert(0, str(OLD))

from run import PROTOCOL, SCENES, native, protocol, raw_sensor_hashes, read, sha, write

LAYOUT = 'west_screened'
RUN = OLD / 'runs' / LAYOUT / 'G'
ARM_SHARE = ('G', 'S', 'X')
SENSOR_FIELDS = ('depth_values_sha256', 'rgb_values_sha256', 'scan_ranges_sha256')
TRACE_FIELDS = ('step', 'action', 'position', 'heading', 'packet_sha256', *SENSOR_FIELDS)


def sensor_support(mapper, packet, cells):
    frame, scan = packet.frame, packet.scan
    rows, cols = np.mgrid[0:frame.depth_m.shape[0]:2, 0:frame.depth_m.shape[1]:2]
    valid = frame.depth_m[::2, ::2] > 0
    pixels = list(zip(rows[valid].tolist(), cols[valid].tolist(), strict=True))
    points, _ = frame.points(stride=2)
    support = {cell: {'RGB_D': [], 'laser': []} for cell in cells}
    for point, pixel in zip(points, pixels, strict=True):
        if .15 < point[2] < 1.2 and (cell := mapper.grid_cell(point)) in support:
            support[cell]['RGB_D'].append({'uv': [pixel[1], pixel[0]], 'xyz_m': point.tolist()})
    for index, distance in enumerate(scan.ranges_m):
        if not (math.isfinite(distance) and 0 < distance < scan.range_max_m - .001):
            continue
        angle = scan.angle_min_rad + index * scan.angle_increment_rad
        local = np.array([math.cos(angle) * distance, math.sin(angle) * distance, 0.])
        point = scan.world_from_laser[:3, :3] @ local + scan.world_from_laser[:3, 3]
        if (cell := mapper.grid_cell(point)) in support:
            support[cell]['laser'].append({'index': index, 'xyz_m': point.tolist(), 'range_m': float(distance)})
    return support


def observation_evidence(world, mapper, packet, cells):
    support = sensor_support(mapper, packet, cells)
    frame = packet.frame
    # Source IDs, true footprints and exact geometry are inspected offline
    # after receiving the original sensor data and are never input to the mapper.
    rendered = world._renderer.render(
        frame.intrinsic, frame.world_from_camera, frame.depth_m.shape[1], frame.depth_m.shape[0],
        max_depth_m=world.config.max_depth_m, min_depth_m=.15)
    assert np.array_equal(rendered['depth_m'], frame.depth_m)
    scene, _, _ = world.source_scene_for_evaluator()
    from shapely.geometry import Point
    from shapely import distance
    by_cell = []
    for cell, evidence in sorted(support.items()):
        for sample in evidence['RGB_D']:
            u, v = sample['uv']
            geometry_id = int(rendered['reference_geometry_id'][v, u])
            sample['oracle_only_source_node'] = world._node_for_gid[geometry_id]
        for name in sorted({item['oracle_only_source_node'] for item in evidence['RGB_D']}):
            transform, geom = scene.graph[name]
            mesh = scene.geometry[geom].copy()
            mesh.apply_transform(transform)
            ev = {'source_world_bounds_min_max_m': mesh.bounds.tolist()}
            evidence.setdefault('oracle_only_source_geometry', {})[name] = ev
        center = np.asarray(world.xy(cell))
        radius = world.config.resolution_m / 2
        point = Point(center)
        tree_index = world._geometry.tree.nearest(point)
        clearance = float(distance(world._geometry.obstacles[tree_index], point))
        by_cell.append({'cell_rc': list(cell), 'center_xy_m': center.tolist(),
                        'cell_bounds_xy_m': [list(center-radius), list(center+radius)],
                        'source_occupancy_center': bool(world._occupancy[cell]),
                        'source_robot_disk_center_is_valid': bool(world._robot_valid[cell]),
                        'source_continuous_center_to_obstacle_m': clearance,
                        **evidence})
    return by_cell


def planned_path_geometry(world, position, heading, actions, anchor):
    from shapely.geometry import LineString, Point
    from shapely import distance
    from utils.grid_geometry import DIRECTIONS
    geometry = world._geometry
    path = []
    for action in actions:
        before = position
        if action == 'forward':
            vector = DIRECTIONS[heading]
            position = tuple(p+v for p, v in zip(position, vector, strict=True))
        else:
            heading = (heading + {'left': -1, 'right': 1}[action]) % 4
        p, q = world.xy(before), world.xy(position)
        line = Point(p) if before == position else LineString([p, q])
        nearest = geometry.tree.nearest(line)
        path.append({'from_rc': list(before), 'to_rc': list(position), 'action': action,
                     'oracle_swept_disk_valid': bool(geometry.valid(p, q, radius=world.config.robot_radius_m)),
                     'oracle_centerline_to_real_obstacles_m': float(distance(geometry.obstacles[nearest], line))})
    assert [*position, heading] == anchor
    return path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', default=HERE / 'west_screened_passive', type=Path)
    destination = parser.parse_args().out
    destination.mkdir(exist_ok=False)
    trace = read(RUN / 'steps.json')
    equal_arms = {}
    for arm in ARM_SHARE:
        path = RUN.parent / arm
        other = read(path / 'steps.json')
        assert len(trace) == len(other) == 61
        assert all(all(a[k] == b[k] for k in TRACE_FIELDS) for a, b in zip(trace, other, strict=True))
        assert sha(RUN / 'observed_belief_trace.npz') == sha(path / 'observed_belief_trace.npz')
        equal_arms[arm] = {'original_manifest_sha256': sha(path / 'manifest.json'),
                           'all_motion_sensor_and_belief_bytes_match': True}

    with np.load(RUN / 'observed_belief_trace.npz', allow_pickle=False) as original:
        belief = original['belief']
    local = read(PROTOCOL)
    spec = protocol(local['placements'], local)
    native.HOLDOUT = SCENES
    native.ARM_MODES = {'G': ('G', 'G')}
    old, (world, transform, first, components, _, _, _), initial, _, _ = native.setup_runtime(
        'holdout', LAYOUT, 'G', spec)
    mapper = components._cpu_backend.scenes[0]['mapper']
    from nso.execution_guard_v8 import ObservedExecutionGuard
    guard = ObservedExecutionGuard(world.config.resolution_m, world.config.robot_radius_m)
    anchor = initial['anchor']
    assert np.array_equal(mapper.belief, belief[0])
    old_start = read(RUN / 'initial_raw_rgbd_scan_sha256.json')
    assert raw_sensor_hashes(first, native.array_sha) == old_start
    then = []
    evidence = {}
    first_local = read(RUN / 'local_choices.json')[0]
    rejected_command = first_local['selected_outbound_actions'][24]
    rejection_step = trace[23]
    rejection = guard.assess(belief[24], rejection_step['position'], rejection_step['heading'], rejected_command, 168)
    assert rejection.reason == 'next_footprint_not_known_safe'
    first_rejected_center_unsafe_paid = next(i for i,b in enumerate(belief[:25]) if not guard.safe_grid(b)[rejection.target])
    assert first_rejected_center_unsafe_paid > 0
    for row in trace:
        step = row['step']
        frame, collision, done = world.step(row['action'])
        packet = old.packet(world, transform, frame, LAYOUT, row['action'], collision, done)
        raw = raw_sensor_hashes(packet, native.array_sha)
        assert all(raw[k] == row[k] for k in ('packet_sha256', *SENSOR_FIELDS))
        assert list(packet.position) == row['position'] and packet.heading == row['heading']
        assert not collision
        mapper.update(packet.frame, packet.scan)
        assert np.array_equal(mapper.belief, belief[step])
        returning = guard.return_plan(mapper.belief, packet.position, packet.heading, anchor, 192-step)
        then.append({'paid': step, 'position': list(packet.position), 'heading': packet.heading,
                     'return_available': returning.available, 'reason': returning.reason,
                     'reserved_current_return_paid': returning.paid_cost})
        if step not in (first_rejected_center_unsafe_paid, 24, 60, 61):
            continue
        changed = [tuple(map(int, c)) for c in np.argwhere(belief[step-1] != belief[step])]
        safe = guard.safe_grid(mapper.belief)
        prior_safe = guard.safe_grid(belief[step-1])
        newly_unsafe = [list(map(int, cell)) for cell in np.argwhere(prior_safe & ~safe)]
        info = {'packet_sha256': packet.sha256(), 'action': row['action'], 'pose': [*packet.position, packet.heading],
                'safe_footprint_at_current': bool(safe[packet.position]),
                'safe_footprint_at_anchor': bool(safe[tuple(anchor[:2])]),
                'newly_unsafe_footprint_centers_rc': newly_unsafe,
                'grid_change_counts': {f'{before}_to_{after}':
                                       int(sum(belief[step-1][c] == before and belief[step][c] == after for c in changed))
                                       for before in (-1, 0, 1) for after in (-1, 0, 1) if before != after},
                'original_return_after_update': asdict(returning)}
        if step == 60:
            assert returning.available
            info['frozen_42_step_return_in_continuous_source_diagnostic_only'] = planned_path_geometry(
                world, packet.position, packet.heading, returning.actions, anchor)
        elif step == 24:
            info['first_selected_route_rejection'] = asdict(rejection) | {'command': rejected_command}
            info['first_selected_route_rejection_footprint_already_unsafe_since_paid'] = first_rejected_center_unsafe_paid
            info['rejected_target_source_robot_disk_valid'] = bool(world._robot_valid[rejection.target])
        elif step == first_rejected_center_unsafe_paid:
            blockers = [c for c in changed if max(abs(a-b) for a,b in zip(rejection.target, c, strict=True))<=1
                        and belief[step][c] == 1]
            info['occupied_update_samples'] = observation_evidence(world, mapper, packet, blockers)
        else:
            assert not returning.available
            assert changed == [(23, 16)] and newly_unsafe == [[24, 15]]
            info['occupied_update_samples'] = observation_evidence(world, mapper, packet, changed)
        evidence[str(step)] = info

    assert all(check['return_available'] for check in then[:-1])
    assert then[-1]['reason'] == 'anchor_disconnected_in_latest_map'
    write(destination / 'receipt.json', {
        'scope': 'Fixed archived simulation actions, no planning override, no robot access. Geometry and source labels are offline diagnostics.',
        'frozen_native_guard_modified': False,
        'all_packet_rgbd_lidar_and_observed_belief_bytevalues_equal_to_archive': True,
        'original_belief_trace_sha256': sha(RUN / 'observed_belief_trace.npz'),
        'common_G_S_X': equal_arms,
        'per_paid_guard': then,
        'first_local_rejection_and_anchor_disconnection': evidence,
    })


if __name__ == '__main__':
    main()
