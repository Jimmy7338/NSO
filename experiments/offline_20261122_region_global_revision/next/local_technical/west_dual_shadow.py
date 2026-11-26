#!/usr/bin/env python3
"""Normalize both obstacle endpoint projectors on each west sensor stream."""

import argparse
from dataclasses import asdict
from pathlib import Path

import numpy as np

from precision_patch import bind, normalized_cell
from west_replay import HERE, LAYOUT, PROTOCOL, RUN, SCENES, SENSOR_FIELDS, native
from west_replay import observation_evidence, planned_path_geometry, protocol, raw_sensor_hashes, read, sha, write


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stream', choices=('archived', 'rgb_only'))
    parser.add_argument('--details',action='store_true')
    args = parser.parse_args()
    rgb = args.stream == 'rgb_only'
    folder = HERE / 'west_screened_dual_endpoint_shadow'
    folder.mkdir(exist_ok=True)
    destination = folder / f'{args.stream}{"_detail" if args.details else ""}.json'
    assert not destination.exists()
    run = HERE / 'west_screened_G_precision_native' / 'native' if rgb else RUN
    steps = read(run / 'steps.json')
    assert len(steps) == (51 if rgb else 61)
    with np.load(run / 'observed_belief_trace.npz', allow_pickle=False) as archive:
        expected = archive['belief']
    config = read(PROTOCOL)
    native.HOLDOUT = SCENES
    native.ARM_MODES = {'G': ('G','G')}
    old, (world, transform, first, components, _, _, _), initial, _, _ = native.setup_runtime(
        'holdout', LAYOUT, 'G', protocol(config['placements'], config))
    anchor = initial['anchor']
    mapper_type = type(components._cpu_backend.scenes[0]['mapper'])
    binding = bind(include_laser=True)
    baseline, dual = [mapper_type(world.shape, world.config, world.config.truncation_m) for _ in range(2)]
    baseline._simulator_camera_boundary_shadow = rgb
    dual._simulator_camera_boundary_shadow = dual._simulator_lidar_boundary_shadow = True
    from nso.execution_guard_v8 import ObservedExecutionGuard
    guard = ObservedExecutionGuard(world.config.resolution_m, world.config.robot_radius_m)
    guard_file = Path(__import__('nso.execution_guard_v8', fromlist=['ObservedExecutionGuard']).__file__)
    history = []
    first_direct_witness = {}

    def capture(packet):
        paid = packet.action_id
        for mapper in (baseline, dual):
            mapper.update(packet.frame, packet.scan)
        assert np.array_equal(baseline.belief, expected[paid])
        a, b = baseline.belief, dual.belief
        a_safe, b_safe = guard.safe_grid(a), guard.safe_grid(b)
        old_plan, dual_plan = [
            guard.return_plan(m, packet.position, packet.heading, anchor, 192-paid) for m in (a,b)
        ]
        lost_real = (a==1)&(b!=1)&world._occupancy
        false_safe = ~a_safe & b_safe & ~world._robot_valid
        row = {
            'paid':paid,
            'packet_sha256':packet.sha256(),
            'changed_belief_cells':int(np.count_nonzero(a!=b)),
            'old_occupied_real_centers_not_occupied_after_dual':int(lost_real.sum()),
            'newly_admitted_physical_colliding_centers':int(false_safe.sum()),
            'matching_real_occupied_centers':int(((a==1)&(b==1)&world._occupancy).sum()),
            'baseline_return_available':old_plan.available, 'baseline_return_paid':old_plan.paid_cost,
            'dual_return_available':dual_plan.available, 'dual_return_paid':dual_plan.paid_cost,
        }
        if args.details:
            row['real_obstacle_centers_lost_rc'] = np.argwhere(lost_real).tolist()
            row['invalid_physical_centers_newly_passed_by_guard_rc'] = np.argwhere(false_safe).tolist()
            if lost_real.any() and not first_direct_witness:
                evidence = observation_evidence(world,baseline,packet,[tuple(map(int,p)) for p in np.argwhere(lost_real)])
                for cell in evidence:
                    for sample in cell['laser']:
                        sample['normalized_hit_cell_rc'] = list(normalized_cell(dual,sample['xyz_m']))
                first_direct_witness.update(paid=paid,pose=[*packet.position,packet.heading],
                                            first_lost_occupied_sensor_endpoints=evidence)
        if paid == len(steps):
            assert old_plan.reason == 'anchor_disconnected_in_latest_map'
            row['current_pose'] = [*packet.position,packet.heading]
            row['baseline_return'] = asdict(old_plan)
            row['dual_return'] = asdict(dual_plan)
            row['cells_before_after'] = {f'{r},{c}':[int(a[r,c]),int(b[r,c])] for r,c in (
                (22,16),(22,17),(23,16),(23,17),(24,15))}
            if dual_plan.available:
                sweeps = planned_path_geometry(world, packet.position, packet.heading, dual_plan.actions, anchor)
                row['dual_return_source_swept_valid_all'] = all(s['oracle_swept_disk_valid'] for s in sweeps)
                row['dual_return_swept_min_clearance_m'] = min(s['oracle_centerline_to_real_obstacles_m'] for s in sweeps)
        history.append(row)

    assert raw_sensor_hashes(first, native.array_sha) == read(run/'initial_raw_rgbd_scan_sha256.json')
    capture(first)
    for step in steps:
        frame, hit, done = world.step(step['action'])
        packet = old.packet(world, transform, frame, LAYOUT, step['action'], hit, done)
        raw = raw_sensor_hashes(packet, native.array_sha)
        assert not hit and all(raw[k]==step[k] for k in ('packet_sha256',*SENSOR_FIELDS))
        capture(packet)
    write(destination, {
        'scope':'Passive replay of archived simulator packets. Guard code and laser free-space rays are unchanged.',
        'stream':args.stream,'baseline_manifest_sha256':sha(run/'manifest.json'),
        'guard_source_sha256':sha(guard_file),'binding':binding,'baseline_all_packets_and_beliefs_exact':True,
        'first_normalized_grid_difference_paid':next((r['paid'] for r in history if r['changed_belief_cells']),None),
        'newly_admitted_physical_colliding_centers_max':max(r['newly_admitted_physical_colliding_centers'] for r in history),
        'old_occupied_real_centers_not_occupied_after_dual_max':max(r['old_occupied_real_centers_not_occupied_after_dual'] for r in history),
        'first_direct_witness':first_direct_witness,
        'history':history,
    })


if __name__ == '__main__':
    main()
