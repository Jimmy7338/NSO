#!/usr/bin/env python3
"""Recover one native candidate branch without evaluator inputs."""

import argparse
from copy import deepcopy
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys
import time
import traceback

for variable in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[variable] = '1'
sys.dont_write_bytecode = True

import numpy as np

HERE = Path(__file__).resolve().parent
PROTOCOL = HERE.parent / 'PROTOCOL.json'
CONTROLS = ('option_id', 'selection_call_id', 'selected_map_version',
            'selected_feedback_version', 'parent_region')
TRACE = ('step', 'action', 'position', 'heading', 'collision', 'done', 'path_length_m',
         'packet_sha256', 'nonsemantic_packet_sha256', 'owner_sha256', 'active_option_id',
         'selected_target_pose', 'phase_before_action', 'matched_target_pose_after_step')


def read(path):
    return json.loads(path.read_text())


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def array_sha(data):
    return hashlib.sha256(np.ascontiguousarray(data).tobytes()).hexdigest()


def chain_hash(previous, packet):
    return hashlib.sha256(bytes.fromhex(previous) + bytes.fromhex(packet)).hexdigest()


def source_module(protocol):
    path = Path(protocol['previous_local_archive']) / 'mechanism/runner/run.py'
    sys.path.insert(0, str(path.parent))
    spec = importlib.util.spec_from_file_location('nso_preserved_mechanism_actor', path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def source_records(mechanism, layout):
    directory = mechanism.HERE / 'layouts' / layout / 'runs/G/attempt_001'
    manifest = read(directory / 'artifact_manifest.json')
    assert manifest['status'] == 'completed_execution'
    assert manifest['freeze_sha256'] == sha(mechanism.HERE / 'freeze.json')
    files = ('choices.json', 'steps.json', 'intervention.json')
    hashes = {name: sha(directory / name) for name in files}
    assert all(hashes[name] == manifest['files_sha256'][name] for name in files)
    choices, steps, intervention = (read(directory / name) for name in files)
    assert intervention['status'] == 'triggered'
    assert intervention['intervention']['paid_before'] == 0
    return choices, steps, intervention['intervention'], hashes


def capture(out, name, mapper, packet, packet_chain, requested_paid, outcome, step=None, copy_from=None):
    import open3d as o3d
    from runtime_binding import write

    path = out / name
    path.mkdir()
    if copy_from is None:
        mesh = mapper.mesh()
        assert o3d.io.write_triangle_mesh(str(path / 'native_mesh.ply'), mesh, write_ascii=False)
        np.savez_compressed(path / 'belief.npz', belief=mapper.belief,
                            position=np.asarray(packet.position), heading=np.asarray(packet.heading))
        mesh_fields = {f'{field}_sha256': array_sha(getattr(mesh, field))
                       for field in ('vertices', 'triangles')}
    else:
        for filename in ('native_mesh.ply', 'belief.npz'):
            os.link(out / copy_from / filename, path / filename)
        mesh_fields = read(out / copy_from / 'checkpoint.json')['native_mesh_fields']
    receipt = dict(status='captured', actual_paid=int(packet.action_id), requested_paid=requested_paid,
                   packet_sha256=packet.sha256(), packet_chain_sha256=packet_chain,
                   pose=[*packet.position, packet.heading], belief_sha256=array_sha(mapper.belief),
                   native_mesh_fields=mesh_fields, outcome=outcome,
                   active_option_id=None if step is None else step['active_option_id'],
                   active_candidate_id=None if step is None else step['active_candidate_id'],
                   phase=None if step is None else step['phase_before_action'],
                   files_sha256={name: sha(path / name) for name in ('native_mesh.ply', 'belief.npz')})
    write(path / 'checkpoint.json', receipt)
    return receipt


def missing(out, name, requested_paid, actual_paid, outcome):
    from runtime_binding import write

    path = out / name
    path.mkdir()
    receipt = dict(status='unavailable', actual_paid=actual_paid, requested_paid=requested_paid,
                   outcome=outcome, files_sha256={})
    write(path / 'checkpoint.json', receipt)
    return receipt


def execute(layout, candidate_id, protocol, out):
    mechanism = source_module(protocol)
    initial = mechanism.verify(layout)
    choices, history, archived_gate, hashes = source_records(mechanism, layout)
    preserved_g_id = protocol['original_candidate'][layout]
    common_paid = protocol['common_paid_checkpoints_by_parent'][layout.split('_')[0]]
    assert archived_gate['chosen_candidate_id'] == preserved_g_id
    assert initial['stop_budget'] == protocol['native_budget_including_return'] == 256
    assert archived_gate['packet_sha256'] == initial['initial_packet_sha256']
    old, gate = mechanism.install()
    from nso.cpu_sensor_contract_v10 import digest, json_value
    from runtime_binding import write

    world, transform, packet, components, runtime, effective, owner = old.bootstrap(layout, 'G')
    assert packet.sha256() == initial['initial_packet_sha256']
    assert [*packet.position, packet.heading] == initial['anchor']
    assert owner['owner_sha256'] == initial['initial_owner_sha256']
    backend = components._cpu_backend
    mapper = runtime.states[0]['mapper']
    assert array_sha(mapper.belief) == archived_gate['belief_sha256']
    original_select = backend.select_target
    first_anchor = None
    steps = []
    initial_packet_hash = packet.sha256()
    packet_chain = chain_hash('0' * 64, initial_packet_hash)
    checkpoints = {'start': capture(out, 'start', mapper, packet, packet_chain, 0, 'restored_prefix')}
    started = time.monotonic()

    def select_once(scene_idx):
        nonlocal first_anchor
        target = original_select(scene_idx)
        if first_anchor is not None:
            assert backend.scenes[scene_idx]['last_selection']['mode'] == 'G'
            return target
        assert scene_idx == 0 and world.step_count == 0
        assert gate.trigger == archived_gate
        record = backend.calls[-1]
        assert record['method'] == 'select_topo_target'
        current = {key: record[key] for key in ('inputs', 'outputs')}
        assert current == choices[0], 'Native initial planning record differs from archive'
        source_selection = deepcopy(current['outputs']['selected'])
        options = deepcopy(current['outputs']['candidates'])
        assert [option['candidate_id'] for option in options] == protocol['candidates_in_order']
        assert [[option['outbound_cost'], option['return_cost']] for option in options] == (
            protocol['actual_annotated_outbound_return_costs'][layout.split('_')[0]])
        selected = deepcopy(options[candidate_id])
        assert selected['outbound_states'][0] == initial['anchor']
        assert selected['outbound_cost'] == len(selected['outbound_actions']) <= common_paid
        assert len(selected['outbound_states']) == selected['outbound_cost'] + 1
        selected.update({key: source_selection[key] for key in CONTROLS})
        if candidate_id != preserved_g_id:
            updated = {**record['outputs'], 'selected': selected}
            backend.scenes[0]['selected'] = deepcopy(selected)
            backend.scenes[0]['last_selection'] = json_value(updated)
            record['outputs'] = updated
            record['output_sha256'] = digest(updated)
            target = np.asarray(selected['pose'][:2], dtype=int)
        else:
            assert selected == source_selection
        first_anchor = dict(layout=layout, native_options=options, executed_option=selected,
                            paid_actions=0, source_selected_option=source_selection,
                            first_record_sha256=mechanism.canonical(choices[0]),
                            candidate_set_sha256=archived_gate['candidate_set_sha256'],
                            score_set_sha256=archived_gate['score_set_sha256'],
                            source_artifact_sha256=hashes, packet_sha256=initial_packet_hash,
                            belief_sha256=archived_gate['belief_sha256'],
                            scores=deepcopy(choices[0]['outputs']['scores']),
                            candidate_summaries=archived_gate['candidate_summaries'],
                            changed_initial_selection=candidate_id != preserved_g_id,
                            initial_frame_sha256=old.method._frame_hash(packet.frame),
                            initial_nonsemantic_packet_sha256=old.base.nonsemantic_packet_sha(packet),
                            initial_owner_sha256=owner['owner_sha256'])
        write(out / 'first_anchor.json', first_anchor)
        return target

    backend.select_target = select_once
    arrival_outcome = None
    arrival_break_paid = None
    horizon_name = f'paid_{common_paid}'
    historical_choices = []
    for _ in range(common_paid):
        action = runtime.next_local_action(0)
        if action is None:
            break
        state = runtime.states[0]
        active = state['option']
        phase = state['phase']
        route = first_anchor['executed_option']
        bound = active is not None and active['option_id'] == route['option_id']
        target = None if active is None else list(active['pose'])
        raw, collision, done = world.step(action)
        packet = old.packet(world, transform, raw, layout, action, collision, done)
        owner = old.front.associate(mapper, raw, packet.frame_id)
        old.method.observe(mapper, packet.frame)
        runtime.observe(0, world.step_count, None, None, None, sensor_packet=packet)
        paid = world.step_count
        pose = [*world.position, world.heading]
        value = packet.sha256()
        packet_chain = chain_hash(packet_chain, value)
        step = dict(step=paid, action=action, position=list(world.position), heading=world.heading,
                    collision=bool(collision), done=bool(done),
                    path_length_m=world.moves * world.config.resolution_m,
                    packet_sha256=value, nonsemantic_packet_sha256=old.base.nonsemantic_packet_sha(packet),
                    frame_sha256=old.method._frame_hash(raw), owner_sha256=owner['owner_sha256'],
                    packet_chain_sha256=packet_chain, active_option_id=None if active is None else active['option_id'],
                    active_candidate_id=None if active is None else active['candidate_id'],
                    selected_target_pose=target, phase_before_action=phase,
                    matched_target_pose_after_step=target == pose if target is not None else None)
        if candidate_id == preserved_g_id:
            actual = {key: step[key] for key in TRACE}
            expected = {key: history[paid - 1][key] for key in TRACE}
            assert actual == expected, (layout, paid, [key for key in TRACE if actual[key] != expected[key]])
        steps.append(step)
        if arrival_outcome is None:
            if collision:
                arrival_outcome = 'collision_before_bound_target'
            elif not bound:
                arrival_outcome = 'bound_option_replaced_before_target'
            elif paid > route['outbound_cost'] or (
                    action != route['outbound_actions'][paid - 1] or pose != route['outbound_states'][paid]):
                arrival_outcome = 'initial_trace_diverged_before_bound_target'
            elif pose == route['pose']:
                assert paid == route['outbound_cost']
                arrival_outcome = 'reached_bound_target'
                checkpoints['arrival'] = capture(out, 'arrival', mapper, packet, packet_chain,
                                                  route['outbound_cost'], arrival_outcome, step)
            if arrival_outcome is not None:
                arrival_break_paid = paid
        if paid == common_paid:
            copy_from = 'arrival' if checkpoints.get('arrival', {}).get('actual_paid') == paid else None
            checkpoints[horizon_name] = capture(out, horizon_name, mapper, packet, packet_chain, common_paid,
                                                'matched_paid_checkpoint', step, copy_from)
        if paid % 10 == 0:
            write(out / 'heartbeat.json', dict(actual_paid=paid, requested_paid=common_paid,
                                               arrival_outcome=arrival_outcome,
                                               elapsed_s=time.monotonic() - started))
        if runtime.states[0]['closed']:
            break

    assert first_anchor is not None
    route = first_anchor['executed_option']
    final_paid = packet.action_id
    runtime_summary = runtime.sensor_episode_summary(0)
    termination = runtime_summary['termination']
    if arrival_outcome is None:
        arrival_outcome = 'stopped_before_bound_target' if termination else 'bound_target_not_reached'
        arrival_break_paid = final_paid
    if 'arrival' not in checkpoints:
        checkpoints['arrival'] = missing(out, 'arrival', route['outbound_cost'], arrival_break_paid, arrival_outcome)
    if horizon_name not in checkpoints:
        checkpoints[horizon_name] = missing(out, horizon_name, common_paid, final_paid, 'ended_before_matched_paid')
    if candidate_id == preserved_g_id:
        for event in backend.calls:
            if event['method'] == 'select_topo_target':
                historical_choices.append({key: event[key] for key in ('inputs', 'outputs')})
        assert historical_choices == choices[:len(historical_choices)]
        archived_arrival = next((s['step'] for s in history if s['active_option_id'] == route['option_id']
                                 and s['matched_target_pose_after_step']), None)
        assert checkpoints['arrival']['actual_paid'] == archived_arrival
    write(out / 'steps.json', steps)
    write(out / 'audit.json', runtime.audit)
    write(out / 'runtime_summary.json', runtime_summary)
    receipt = dict(status='completed_execution', layout=layout, candidate_id=candidate_id,
                   evaluator_inputs_read=False, source_commit=protocol['source_commit'],
                   source_protocol_sha256=sha(PROTOCOL), runner_sha256=sha(Path(__file__)),
                   neutral_g_continuation=True, native_budget=initial['stop_budget'],
                   original_archived_option=candidate_id == preserved_g_id,
                   source_matched_prefix_actions=len(steps) if candidate_id == preserved_g_id else None,
                   source_matched_choices=len(historical_choices) if candidate_id == preserved_g_id else None,
                   initial_packet_sha256=initial_packet_hash, final_packet_sha256=packet.sha256(),
                   final_packet_chain_sha256=packet_chain, final_pose=[*packet.position, packet.heading],
                   arrival=arrival_outcome, paid_at_arrival_or_interruption=arrival_break_paid,
                   planned_outbound_cost=route['outbound_cost'], planned_return_cost=route['return_cost'],
                   paid_actions=final_paid, collisions=sum(step['collision'] for step in steps),
                   observed_native_termination=termination,
                   stop='short_window_limit' if final_paid == common_paid else 'native_early_stop',
                   checkpoints=checkpoints, elapsed_s=time.monotonic() - started,
                   files_sha256={str(p.relative_to(out)): sha(p) for p in sorted(out.rglob('*')) if p.is_file()})
    write(out / 'manifest.json', receipt)
    print(json.dumps({key: receipt[key] for key in ('layout', 'candidate_id', 'arrival', 'paid_actions', 'elapsed_s')}),
          flush=True)


def main():
    protocol = read(PROTOCOL)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('layout', choices=protocol['layouts_in_order'])
    parser.add_argument('candidate_id', type=int, choices=protocol['candidates_in_order'])
    args = parser.parse_args()
    frozen = read(HERE / 'freeze.json')
    assert all(sha(HERE / pathname) == expected for pathname, expected in frozen['files_sha256'].items())
    out = HERE / 'runs' / args.layout / f'candidate_{args.candidate_id}'
    out.mkdir(parents=True, exist_ok=False)
    try:
        execute(args.layout, args.candidate_id, protocol, out)
    except BaseException as error:
        failure = dict(status='failed', layout=args.layout, candidate_id=args.candidate_id,
                       error=repr(error), traceback=traceback.format_exc())
        (out / 'failure.json').write_text(json.dumps(failure, indent=2) + '\n')
        raise


if __name__ == '__main__':
    main()
