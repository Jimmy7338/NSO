#!/usr/bin/env python3
"""Identify the sensor contributions to the normalized closed-loop map cut."""

from dataclasses import asdict

import numpy as np

from precision_patch import bind, camera_cell
from west_replay import HERE, LAYOUT, PROTOCOL, SCENES, SENSOR_FIELDS, native, protocol
from west_replay import observation_evidence, planned_path_geometry, raw_sensor_hashes, read, sha, write


def main():
    run = HERE / 'west_screened_G_precision_native' / 'native'
    domain = HERE / 'west_screened_G_precision_native' / 'receipt.json'
    result = read(domain)
    last = result['normalized']['paid_actions']
    trace = read(run / 'steps.json')
    with np.load(run / 'observed_belief_trace.npz', allow_pickle=False) as dataset:
        recorded = dataset['belief']
    config = read(PROTOCOL)
    native.HOLDOUT = SCENES
    native.ARM_MODES = {'G': ('G','G')}
    old, (world, transform, packet, components, _, _, _), initial, _, _ = native.setup_runtime(
        'holdout', LAYOUT, 'G', protocol(config['placements'], config))
    original = components._cpu_backend.scenes[0]['mapper']
    assert bind() == result['binding']
    mapper = type(original)(world.shape, world.config, world.config.truncation_m)
    mapper._simulator_camera_boundary_shadow = True
    mapper.update(packet.frame, packet.scan)
    from nso.execution_guard_v8 import ObservedExecutionGuard
    guard = ObservedExecutionGuard(world.config.resolution_m, world.config.robot_radius_m)
    anchor = initial['anchor']
    assert np.array_equal(mapper.belief, recorded[0])
    prior = None
    for step in trace:
        frame, collision, done = world.step(step['action'])
        packet = old.packet(world, transform, frame, LAYOUT, step['action'], collision, done)
        raw = raw_sensor_hashes(packet, native.array_sha)
        assert not collision and all(raw[k] == step[k] for k in ('packet_sha256', *SENSOR_FIELDS))
        mapper.update(packet.frame, packet.scan)
        assert np.array_equal(mapper.belief, recorded[step['step']])
        current = guard.return_plan(mapper.belief, packet.position, packet.heading, anchor, 192-step['step'])
        if step['step'] == last-1:
            assert current.available
            prior = asdict(current)
            prior.update(source_swept_geometry_only=planned_path_geometry(
                world,packet.position,packet.heading,current.actions,anchor))
    assert not current.available and current.reason == result['normalized']['termination']['reason']
    nearby = []
    before, after = recorded[-2:]
    newly_unsafe = guard.safe_grid(before) & ~guard.safe_grid(after)
    near_unsafe = np.array(np.argwhere(newly_unsafe))
    for c in np.argwhere((before != 1) & (after == 1)):
        if np.any(np.max(np.abs(near_unsafe-c),axis=1)<=1):
            nearby.append(tuple(map(int,c)))
    evidence = observation_evidence(world, mapper, packet, nearby)
    for cell in evidence:
        for point in cell['RGB_D']:
            point['camera_normalized_target_rc'] = list(camera_cell(mapper,point['xyz_m']))
    destination = HERE / 'west_screened_G_precision_native' / 'failure_observation_provenance.json'
    assert not destination.exists()
    write(destination, {
        'same_recorded_sensor_values_and_all_belief_states_exact':True,
        'normalized_closed_loop_manifest_sha256':sha(run/'manifest.json'),
        'paid':last,
        'pose':[*packet.position,packet.heading],
        'action':trace[-1]['action'],
        'raw_sensor_sha256':raw,
        'previous_guard':prior,
        'current_guard':asdict(current),
        'newly_unsafe_centers':near_unsafe.tolist(),
        'nearby_blocking_obstacle_endpoints':evidence,
        'geometry_oracle_used_by_actor':False,
    })


if __name__ == '__main__':
    main()
