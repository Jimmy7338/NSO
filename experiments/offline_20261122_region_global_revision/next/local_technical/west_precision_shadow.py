#!/usr/bin/env python3
"""Paired native and camera-precision mapping from identical frozen west packets."""

from dataclasses import asdict
from pathlib import Path

import numpy as np

from precision_patch import bind, boundary_index, camera_cell
from west_replay import LAYOUT, RUN, PROTOCOL, SCENES, SENSOR_FIELDS, planned_path_geometry, sensor_support
from west_replay import native, protocol, raw_sensor_hashes, read, sha, write, HERE


def main():
    destination = HERE / 'west_screened_precision_shadow'
    destination.mkdir(exist_ok=False)
    assert (boundary_index(3.399999821, .2), boundary_index(3.39999, .2),
            boundary_index(3.4000001, .2), boundary_index(3.40001, .2)) == (17,16,17,17)
    trace = read(RUN / 'steps.json')
    assert len(trace) == 61
    with np.load(RUN / 'observed_belief_trace.npz', allow_pickle=False) as original:
        archive = original['belief']
    local = read(PROTOCOL)
    native.HOLDOUT = SCENES
    native.ARM_MODES = {'G': ('G', 'G')}
    old, (world, transform, initial, components, _, _, _), cold, _, _ = native.setup_runtime(
        'holdout', LAYOUT, 'G', protocol(local['placements'], local))
    baseline = components._cpu_backend.scenes[0]['mapper']
    from nso.execution_guard_v8 import ObservedExecutionGuard
    guard = ObservedExecutionGuard(world.config.resolution_m, world.config.robot_radius_m)
    anchor = cold['anchor']
    guard_source = Path(__import__('nso.execution_guard_v8', fromlist=['ObservedExecutionGuard']).__file__)
    binding = bind()
    corrected = type(baseline)(world.shape, world.config, world.config.truncation_m)
    corrected._simulator_camera_boundary_shadow = True
    corrected.update(initial.frame, initial.scan)
    assert np.array_equal(baseline.belief, archive[0])
    old_start = read(RUN / 'initial_raw_rgbd_scan_sha256.json')
    assert raw_sensor_hashes(initial, native.array_sha) == old_start

    history_rows = []
    precise_belief = [corrected.belief.copy()]

    def measure(step, packet):
        a, b = baseline.belief, corrected.belief
        a_safe, b_safe = guard.safe_grid(a), guard.safe_grid(b)
        return_chances = [guard.return_plan(mapper.belief, packet.position, packet.heading, anchor, 192-step)
                          for mapper in (baseline, corrected)]
        lost_true_centers = (a == 1) & (b != 1) & world._occupancy
        newly_unsafe_allowed = ~a_safe & b_safe & ~world._robot_valid
        row = {
            'paid': step, 'packet_sha256': packet.sha256(),
            'changed_belief_cells': int(np.count_nonzero(a != b)),
            'old_occupied_physical_centers_not_labeled_new': int(np.count_nonzero(lost_true_centers)),
            'new_guard_admitted_centers_with_invalid_physical_footprint': int(np.count_nonzero(newly_unsafe_allowed)),
            'baseline_guard_return': asdict(return_chances[0]) | {'actions':len(return_chances[0].actions)},
            'normalized_guard_return': asdict(return_chances[1]) | {'actions':len(return_chances[1].actions)},
        }
        if step == 61:
            row['critical_cell_values_baseline_normalized'] = {
                f'{r},{c}': [int(a[r,c]), int(b[r,c])] for r,c in ((23,16),(23,17),(24,15))}
            raw = sensor_support(corrected, packet, [(23,16)])[(23,16)]
            moved = {}
            for sample in raw['RGB_D']:
                cell = camera_cell(corrected, sample['xyz_m'])
                moved[str(cell)] = moved.get(str(cell),0)+1
            assert sum(moved.values()) == len(raw['RGB_D']) == 44
            row['all_44_original_depth_hits_still_projected'] = moved
            if return_chances[1].available:
                edges = planned_path_geometry(world, packet.position, packet.heading, return_chances[1].actions, anchor)
                row['normalized_unchanged_guard_proposed_return_reference_only'] = edges
        history_rows.append(row)

    measure(0, initial)
    for record in trace:
        step = record['step']
        frame, collision, done = world.step(record['action'])
        packet = old.packet(world, transform, frame, LAYOUT, record['action'], collision, done)
        assert not collision
        raw = raw_sensor_hashes(packet, native.array_sha)
        assert all(raw[k] == record[k] for k in ('packet_sha256', *SENSOR_FIELDS))
        baseline.update(packet.frame, packet.scan)
        corrected.update(packet.frame, packet.scan)
        assert np.array_equal(baseline.belief, archive[step])
        precise_belief.append(corrected.belief.copy())
        measure(step, packet)
    with (destination / 'normalized_belief_trace.npz').open('xb') as result:
        np.savez_compressed(result, paid=np.arange(len(precise_belief)), belief=np.array(precise_belief))
    write(destination / 'receipt.json', {
        'scope':'same archived RGB-D and laser projection with identical original actions and budget. Independent true geometry is annotation only.',
        'binding':binding, 'guard_sha256':sha(guard_source),
        'original_belief_trace_sha256':sha(RUN / 'observed_belief_trace.npz'),
        'baseline_all_62_observation_belief_states_exact':True,
        'first_grid_difference_paid':next((r['paid'] for r in history_rows if r['changed_belief_cells']),None),
        'original_invalid_to_new_true_physical_footprint_max':max(r['new_guard_admitted_centers_with_invalid_physical_footprint'] for r in history_rows),
        'original_true_occupied_centers_no_longer_occupied_max':max(r['old_occupied_physical_centers_not_labeled_new'] for r in history_rows),
        'history':history_rows,
    })


if __name__ == '__main__':
    main()
