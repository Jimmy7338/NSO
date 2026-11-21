#!/usr/bin/env python3
"""Replay the historical observed prefix and intervene on one E1 selection."""

import argparse
from copy import deepcopy
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import time
import traceback

for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[key] = '1'

HERE = Path(__file__).resolve().parent
OLD = HERE.parents[1] / 'four_room_completion_20261119'
SEED = 20261120
PREFIX = 198
BUDGET = 512
ORIGINAL = OLD / f'matrix_study/scenes/{SEED}/runs/S/attempt_001'
PREPARED = ORIGINAL.parents[2] / 'prepared'
REFERENCE = OLD / f'surface_evaluation/matrix/{SEED}/S'
MATRIX_RUNNER = OLD / 'matrix_study/run.py'


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read(path):
    return json.loads(path.read_text())


def load_matrix():
    spec = importlib.util.spec_from_file_location('e1_historical_matrix', MATRIX_RUNNER)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.dependencies()
    return module


def source_files():
    result = [Path(__file__), HERE / 'protocol.json', HERE / 'evaluate_e1.py', MATRIX_RUNNER,
              MATRIX_RUNNER.parent / 'protocol.json', PREPARED / 'freeze.json',
              PREPARED / 'initial_state.json', PREPARED / 'initial_sensor.npz',
              PREPARED / 'evaluator_truth.npz', ORIGINAL / 'artifact_manifest.json',
              REFERENCE / 'fixed_reference.npz', REFERENCE / 'result.json',
              OLD / 'surface_evaluation/exterior_reference.py', OLD / 'surface_evaluation/protocol.json']
    result.extend(ORIGINAL / rel for rel in read(ORIGINAL / 'artifact_manifest.json')['files_sha256'])
    return result


def lineage(matrix):
    initial = matrix.verify(SEED)
    manifest = read(ORIGINAL / 'artifact_manifest.json')
    for rel, expected in manifest['files_sha256'].items():
        assert sha(ORIGINAL / rel) == expected, rel
    historical = read(REFERENCE / 'result.json')
    assert historical['source_mesh_sha256'] == sha(ORIGINAL / 'final_tsdf_mesh.ply')
    assert historical['reference_npz_sha256'] == sha(REFERENCE / 'fixed_reference.npz')
    choices = read(ORIGINAL / 'choices.json')
    at = choices[5]
    assert at['outputs']['plan_number'] == 6 and at['inputs']['remaining_budget'] == 314
    assert len(at['outputs']['candidates']) == 12
    candidates = {c['candidate_id']: c for c in at['outputs']['candidates']}
    assert candidates[3]['outbound_cost'] == 78 and candidates[3]['return_cost'] == 40
    assert candidates[5]['outbound_cost'] == 79 and candidates[5]['return_cost'] == 89
    assert at['outputs']['selected']['candidate_id'] == 5
    return initial, choices, read(ORIGINAL / 'steps.json')


def freeze(matrix):
    lineage(matrix)
    with (HERE / 'freeze.json').open('x') as stream:
        json.dump(dict(protocol_sha256=sha(HERE / 'protocol.json'),
                       source_sha256={str(p): sha(p) for p in source_files()},
                       actor_actions_before_freeze=0, created_unix_s=time.time()), stream, indent=2)
        stream.write('\n')


def verify_freeze():
    frozen = read(HERE / 'freeze.json')
    assert set(frozen['source_sha256']) == {str(p) for p in source_files()}
    for path, expected in frozen['source_sha256'].items():
        assert sha(Path(path)) == expected, path


def choices_of(backend):
    return [{key: call[key] for key in ('inputs', 'outputs')}
            for call in backend.calls if call['method'] == 'select_topo_target']


def exact_original_prefix(got, expected, count):
    assert got == expected[:count], 'Historical selection evidence differed'


def write_mesh(out, mapper, name):
    import open3d as o3d

    assert o3d.io.write_triangle_mesh(str(out / name), mapper.mesh(), write_ascii=False)


def frontier_at_robot(belief, position):
    row, col = position
    return bool(belief[row, col] == 0 and any(
        0 <= r < belief.shape[0] and 0 <= c < belief.shape[1] and belief[r, c] == -1
        for r, c in ((row - 1, col), (row + 1, col), (row, col - 1), (row, col + 1))))


def run(matrix, arm):
    import numpy as np
    from nso.cpu_sensor_contract_v10 import digest, json_value

    verify_freeze()
    initial, source_choices, source_steps = lineage(matrix)
    out = HERE / f'candidate_{arm}' / 'attempt_001'
    out.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    matrix.write(out / 'attempt.json', dict(arm=arm, seed=SEED, budget=BUDGET,
                 expected_prefix_actions=PREFIX, source_attempt=str(ORIGINAL),
                 freeze_sha256=sha(HERE / 'freeze.json'), started_unix_s=time.time()))
    world, transform, packet, components, runtime, effective = matrix.bootstrap(SEED, 'S')
    assert packet.sha256() == initial['initial_packet_sha256']
    backend = components._cpu_backend
    mapper = runtime.states[0]['mapper']
    state = backend.scenes[0]
    with np.load(PREPARED / 'evaluator_truth.npz', allow_pickle=False) as data:
        reachable = data['reachable']
    assert np.array_equal(reachable, world.reachable)
    original_select = backend.select_target
    forced = None
    prefix_packet_hashes = [packet.sha256()]
    route_receipt = None
    rows = []

    def select_once(scene_idx):
        nonlocal forced
        target = original_select(scene_idx)
        if forced is not None or world.step_count != PREFIX:
            return target
        assert scene_idx == 0 and state['plans'] == 6
        assert [event['outputs']['plan_number'] for event in backend.calls
                if event['method'] == 'select_topo_target'] == list(range(1, 7))
        exact_original_prefix(choices_of(backend), source_choices, 6)
        record = backend.calls[-1]
        assert record['method'] == 'select_topo_target'
        source = record['outputs']
        chosen = deepcopy(next(c for c in source['candidates'] if c['candidate_id'] == arm))
        assert chosen['outbound_states'][0] == [*world.position, world.heading]
        for key in ('option_id', 'selection_call_id', 'selected_map_version',
                    'selected_feedback_version', 'parent_region'):
            chosen[key] = source['selected'][key]
        if arm == 3:
            output = {**source, 'selected': chosen}
            state['selected'] = deepcopy(chosen)
            state['last_selection'] = json_value(output)
            record['outputs'] = output
            record['output_sha256'] = digest(output)
        else:
            assert chosen == source['selected']
        forced = dict(original_selected_candidate=5, counterfactual_candidate=arm,
                      choice_overridden=(arm != 5), original_mode='S', continuation_mode='S',
                      paid_actions=PREFIX, remaining_budget=314, selection_call_id=record['call_id'],
                      historical_all_twelve_candidates_match=True,
                      historical_all_scores_match=True, full_prefix_packets_match=len(prefix_packet_hashes) == 199,
                      prefix_packet_sequence_sha256=digest(prefix_packet_hashes),
                      prefix_belief_sha256=digest(mapper.belief), prefix_known_mask_sha256=digest(mapper.belief != -1),
                      option=chosen, options=deepcopy(source_choices[5]['outputs']['candidates']),
                      original_scores=source_choices[5]['outputs']['scores'])
        matrix.write(out / 'intervention.json', forced)
        return np.asarray(chosen['pose'][:2], dtype=int)

    backend.select_target = select_once
    try:
        with (out / 'steps.jsonl').open('x') as stream:
            while not runtime.states[0]['closed']:
                if world.step_count == PREFIX and forced is None:
                    assert runtime.states[0]['active_actions'] == []
                    exact_original_prefix(choices_of(backend), source_choices, 5)
                    write_mesh(out, mapper, 'prefix_mesh.ply')
                    np.savez_compressed(out / 'prefix_belief.npz', belief=mapper.belief)
                action = runtime.next_local_action(0)
                if action is None:
                    break
                runtime_state = runtime.states[0]
                option = runtime_state['option']
                phase = runtime_state['phase']
                raw, collision, done = world.step(action)
                packet = matrix.packet(world, transform, raw, SEED, action, collision, done)
                runtime.observe(0, world.step_count, None, None, None, sensor_packet=packet)
                pos = [*world.position, world.heading]
                row = dict(step=world.step_count, action=action, position=list(world.position),
                           heading=world.heading, collision=bool(collision), path_length_m=world.moves*world.config.resolution_m,
                           packet_sha256=packet.sha256(), phase=phase,
                           option_id=option['option_id'] if option else None,
                           option_candidate_id=option['candidate_id'] if option else None,
                           option_group=option['group'] if option else None,
                           option_arrived=option is not None and option['pose'] == pos,
                           observed_frontier_at_robot=frontier_at_robot(mapper.belief, world.position),
                           **matrix.evaluate_belief(mapper.belief, reachable))
                if world.step_count <= PREFIX or arm == 5:
                    prior = source_steps[world.step_count - 1]
                    for key in ('step', 'action', 'position', 'heading', 'collision', 'path_length_m',
                                'packet_sha256', 'coverage_fraction', 'free_correct_fraction'):
                        assert row[key] == prior[key], (key, world.step_count)
                if world.step_count <= PREFIX:
                    prefix_packet_hashes.append(row['packet_sha256'])
                rows.append(row)
                stream.write(json.dumps(json_value(row), allow_nan=False) + '\n')
                stream.flush()
                if world.step_count % 10 == 0:
                    matrix.write(out / 'heartbeat.json', dict(step=world.step_count,
                                 elapsed_s=time.monotonic()-started, coverage=row['free_correct_fraction']))
                if forced is not None and route_receipt is None:
                    fixed_id = forced['option']['option_id']
                    arrived = row['option_id'] == fixed_id and row['option_arrived']
                    exited = row['option_id'] != fixed_id
                    if arrived or exited:
                        boundary = row if arrived else rows[-2]
                        # Save the nominal route-end map on an exact arrival. A
                        # guard-triggered switch is preserved as a censored route.
                        route_receipt = dict(paid_at_route_end=boundary['step'],
                            arrived=arrived, observed_free_fraction=boundary['free_correct_fraction'],
                            path_length_m=boundary['path_length_m'],
                            route_end_mesh_exported=arrived)
                        if arrived:
                            write_mesh(out, mapper, 'route_end_mesh.ply')
                        matrix.write(out / 'route_end.json', route_receipt)
        assert forced is not None
        if arm == 5:
            assert len(rows) == len(source_steps)
            exact_original_prefix(choices_of(backend), source_choices, len(source_choices))
        summary = runtime.sensor_episode_summary(0)
        matrix.export_geometry(out, mapper, world)
        matrix.write(out / 'choices.json', choices_of(backend))
        matrix.write(out / 'runtime_audit.json', runtime.audit)
        matrix.write(out / 'episode_summary.json', summary)
        matrix.write(out / 'effective_configuration.json', effective)
        if arm == 5:
            assert sha(out / 'final_tsdf_mesh.ply') == sha(ORIGINAL / 'final_tsdf_mesh.ply')
        final = matrix.evaluate_belief(mapper.belief, reachable)
        assert len(rows) <= BUDGET
        assert all(r['step'] == i for i, r in enumerate(rows, 1))
        result = dict(status='completed_execution', seed=SEED, layout='E1', candidate_id=arm,
                      budget=BUDGET, paid_actions=len(rows), first_changed_plan=6, route=route_receipt,
                      returned_to_anchor=summary['termination']['returned_to_anchor'],
                      termination=summary['termination'], collisions=world.collisions,
                      final_pose=[*world.position, world.heading],
                      path_length_m=world.moves*world.config.resolution_m, final_evaluation=final,
                      actual_prefix_and_historical_candidates_verified=True,
                      historical_tail_exactly_verified=arm == 5, elapsed_s=time.monotonic()-started)
        verify_freeze()
        matrix.write(out / 'result.json', result)
        matrix.write(out / 'artifact_manifest.json', dict(status='completed_execution',
                     files_sha256={p.name: sha(p) for p in out.iterdir() if p.is_file()}))
        print(json.dumps(json_value(result), ensure_ascii=False), flush=True)
    except BaseException as exc:
        matrix.write(out / 'failure.json', dict(error=repr(exc), traceback=traceback.format_exc(),
                     paid_actions=len(rows), elapsed_s=time.monotonic()-started))
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=('freeze', 'run'))
    parser.add_argument('--candidate', type=int, choices=(3, 5))
    args = parser.parse_args()
    matrix = load_matrix()
    if args.command == 'freeze':
        freeze(matrix)
    elif args.candidate is None:
        parser.error('--candidate is required with run')
    else:
        run(matrix, args.candidate)


if __name__ == '__main__':
    main()
