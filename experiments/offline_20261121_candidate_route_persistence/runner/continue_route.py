#!/usr/bin/env python3
"""Recover a short native route, then continue it with the preserved neutral policy."""

import argparse
from copy import deepcopy
import json
from pathlib import Path
import sys
import time
import traceback

HERE = Path(__file__).resolve().parent
PROTOCOL = HERE.parent / 'PROTOCOL.json'
PRIOR = HERE.parents[1] / 'candidate_route_yield_20261121'
PRIOR_RUNNER = PRIOR / 'runner'
sys.dont_write_bytecode = True
sys.path.insert(0, str(PRIOR_RUNNER))
from execute import (CONTROLS, array_sha, capture, chain_hash, missing, np, read, sha,
                     source_module, source_records)

SCHEDULE = (('warehouse_left', 3), ('gallery_left', 2), ('gallery_left', 3),
            ('gallery_right', 2), ('gallery_right', 3))
PRIOR_HEAD = '7fda64686ecd9b2adbc5aa35bc3e277eb720d00f'
NATIVE_CAP = 256
MIDPOINT = 128
DIAGNOSTIC_PAID = (96, MIDPOINT, 160, 192, 224)


def short_run(layout, candidate):
    return PRIOR_RUNNER / 'runs' / layout / f'candidate_{candidate}'


def recover_inputs(layout, candidate):
    short = short_run(layout, candidate)
    manifest = read(short / 'manifest.json')
    assert manifest['status'] == 'completed_execution' and manifest['arrival'] == 'reached_bound_target'
    assert manifest['source_protocol_sha256'] == sha(PRIOR / 'PROTOCOL.json')
    assert not manifest['original_archived_option'] and manifest['native_budget'] == NATIVE_CAP
    common = 30 if layout.startswith('warehouse_') else 58
    assert manifest['paid_actions'] == common
    paths = {}
    for member in ('first_anchor.json', 'steps.json'):
        paths[member] = sha(short / member)
        assert paths[member] == manifest['files_sha256'][member]
    receipts = {}
    for name in ('start', 'arrival', f'paid_{common}'):
        path = short / name / 'checkpoint.json'
        checkpoint = read(path)
        assert checkpoint == manifest['checkpoints'][name] and checkpoint['status'] == 'captured'
        paths[f'{name}/checkpoint.json'] = sha(path)
        assert paths[f'{name}/checkpoint.json'] == manifest['files_sha256'][f'{name}/checkpoint.json']
        for filename, digest in checkpoint['files_sha256'].items():
            member = f'{name}/{filename}'
            paths[member] = sha(short / member)
            assert paths[member] == manifest['files_sha256'][member] == digest
        receipts[name] = checkpoint
    assert receipts['start']['actual_paid'] == 0
    assert receipts['arrival']['actual_paid'] == (30 if common == 30 else 38)
    assert receipts[f'paid_{common}']['actual_paid'] == common
    trace = read(short / 'steps.json')
    assert len(trace) == common and [s['step'] for s in trace] == list(range(1, common + 1))
    return dict(directory=short, files=paths, trace=trace, common=common, checkpoints=receipts,
                anchor=read(short / 'first_anchor.json'))


def measured_short_snapshot(name, original, mapper, packet, chain):
    checkpoint = original['checkpoints'][name]
    assert packet.action_id == checkpoint['actual_paid']
    assert packet.sha256() == checkpoint['packet_sha256'] and chain == checkpoint['packet_chain_sha256']
    assert [*packet.position, packet.heading] == checkpoint['pose']
    assert array_sha(mapper.belief) == checkpoint['belief_sha256']
    mesh = mapper.mesh()
    native = {f'{field}_sha256': array_sha(getattr(mesh, field)) for field in ('vertices', 'triangles')}
    assert native == checkpoint['native_mesh_fields']
    directory = original['directory'] / name
    return dict(status='restored_exact', actual_paid=packet.action_id, packet_sha256=packet.sha256(),
                packet_chain_sha256=chain, pose=checkpoint['pose'], belief_sha256=checkpoint['belief_sha256'],
                native_mesh_fields=native, prior_checkpoint_path=str(directory / 'checkpoint.json'),
                prior_checkpoint_sha256=sha(directory / 'checkpoint.json'),
                prior_files={filename: dict(path=str(directory / filename), sha256=digest)
                             for filename, digest in checkpoint['files_sha256'].items()})


def run(layout, candidate, out):
    old_protocol = read(PRIOR / 'PROTOCOL.json')
    original = recover_inputs(layout, candidate)
    mechanism = source_module(old_protocol)
    source_initial = mechanism.verify(layout)
    choices, _, archived_gate, _ = source_records(mechanism, layout)
    old, gate = mechanism.install()
    from nso.cpu_sensor_contract_v10 import digest, json_value
    from runtime_binding import write

    assert old_protocol['native_budget_including_return'] == source_initial['stop_budget'] == NATIVE_CAP
    assert candidate != old_protocol['original_candidate'][layout]
    world, transform, packet, components, runtime, _effective, owner = old.bootstrap(layout, 'G')
    first = original['anchor']
    assert packet.sha256() == first['packet_sha256'] and [*packet.position, packet.heading] == source_initial['anchor']
    assert owner['owner_sha256'] == first['initial_owner_sha256']
    backend = components._cpu_backend
    mapper = runtime.states[0]['mapper']
    assert array_sha(mapper.belief) == first['belief_sha256']
    original_select = backend.select_target
    selected_once = False
    common = original['common']

    def select_once(scene_index):
        nonlocal selected_once
        original_target = original_select(scene_index)
        if selected_once:
            assert backend.scenes[scene_index]['last_selection']['mode'] == 'G'
            return original_target
        assert scene_index == 0 and world.step_count == 0 and gate.trigger == archived_gate
        record = backend.calls[-1]
        assert record['method'] == 'select_topo_target'
        assert {key: record[key] for key in ('inputs', 'outputs')} == choices[0]
        native = record['outputs']
        option = deepcopy(native['candidates'][candidate])
        option.update({key: native['selected'][key] for key in CONTROLS})
        assert option == first['executed_option']
        updated = {**native, 'selected': option}
        backend.scenes[0]['selected'] = deepcopy(option)
        backend.scenes[0]['last_selection'] = json_value(updated)
        record['outputs'] = updated
        record['output_sha256'] = digest(updated)
        selected_once = True
        return np.asarray(option['pose'][:2], dtype=int)

    backend.select_target = select_once
    write(out / 'first_anchor.json', first)
    start_s = time.monotonic()
    chain = chain_hash('0' * 64, packet.sha256())
    recovered = {'start': measured_short_snapshot('start', original, mapper, packet, chain)}
    checkpoints = {}
    trace, belief_trace = [], [mapper.belief.copy()]
    poses = {(*packet.position, packet.heading)}
    cells = {tuple(packet.position)}
    while not runtime.states[0]['closed']:
        decision_start = time.monotonic()
        action = runtime.next_local_action(0)
        decision_s = time.monotonic() - decision_start
        if action is None:
            break
        state = runtime.states[0]
        active = state['option']
        phase = state['phase']
        target = None if active is None else list(active['pose'])
        before_belief = belief_trace[-1]
        raw, collision, done = world.step(action)
        packet = old.packet(world, transform, raw, layout, action, collision, done)
        owner = old.front.associate(mapper, raw, packet.frame_id)
        old.method.observe(mapper, packet.frame)
        runtime.observe(0, world.step_count, None, None, None, sensor_packet=packet)
        paid = world.step_count
        pose = (*world.position, world.heading)
        packet_sha = packet.sha256()
        chain = chain_hash(chain, packet_sha)
        new_known = int(np.count_nonzero((before_belief == -1) & (mapper.belief != -1)))
        belief_trace.append(mapper.belief.copy())
        row = dict(step=paid, action=action, position=list(world.position), heading=world.heading,
                   collision=bool(collision), done=bool(done), path_length_m=world.moves * world.config.resolution_m,
                   packet_sha256=packet_sha, nonsemantic_packet_sha256=old.base.nonsemantic_packet_sha(packet),
                   frame_sha256=old.method._frame_hash(raw), owner_sha256=owner['owner_sha256'],
                   packet_chain_sha256=chain, active_option_id=None if active is None else active['option_id'],
                   active_candidate_id=None if active is None else active['candidate_id'],
                   selected_target_pose=target, phase_before_action=phase,
                   matched_target_pose_after_step=target == list(pose) if target is not None else None,
                   decision_wall_s=decision_s, recovery=paid <= common,
                   observed_known_cells=int(np.count_nonzero(mapper.belief != -1)),
                   observed_free_cells=int(np.count_nonzero(mapper.belief == 0)), newly_known_cells=new_known,
                   repeat_without_new_known_cells=(new_known == 0),
                   changed_belief_cells=int(np.count_nonzero(before_belief != mapper.belief)),
                   revisited_position=tuple(world.position) in cells, revisited_pose=pose in poses)
        trace.append(row)
        poses.add(pose)
        cells.add(tuple(world.position))
        if paid <= common:
            recorded = original['trace'][paid - 1]
            differing = [key for key, value in recorded.items() if row[key] != value]
            assert not differing, (layout, candidate, paid, differing)
            for name, checkpoint in original['checkpoints'].items():
                if name != 'start' and checkpoint['actual_paid'] == paid:
                    recovered[name] = measured_short_snapshot(name, original, mapper, packet, chain)
                    if name == 'arrival':
                        assert row['active_candidate_id'] == candidate and row['matched_target_pose_after_step']
            if paid == common:
                assert set(recovered) == set(original['checkpoints'])
                write(out / 'checkpoint_recovery.json', dict(status='recovered_exact_through_common_paid',
                    before_first_new_action=True, matched_short_prefix_actions=common, snapshot_references=recovered,
                    prior_manifest_sha256=sha(original['directory'] / 'manifest.json'),
                    prior_steps_sha256=original['files']['steps.json']))
        if paid in DIAGNOSTIC_PAID:
            name = f'paid_{paid}'
            checkpoints[name] = capture(out, name, mapper, packet, chain, paid,
                                        'native_midpoint' if paid == MIDPOINT else 'diagnostic_paid_checkpoint', row)
        if paid % 10 == 0:
            write(out / 'heartbeat.json', dict(actual_paid=paid, native_budget=NATIVE_CAP,
                                               checkpoint_verified=paid >= common,
                                               elapsed_s=time.monotonic() - start_s))

    assert selected_once and set(recovered) == set(original['checkpoints'])
    summary = runtime.sensor_episode_summary(0)
    termination = summary['termination']
    assert summary['closed'] and termination is not None
    assert len(trace) == packet.action_id == termination['final_action_id'] <= NATIVE_CAP
    checkpoints['terminal'] = capture(out, 'terminal', mapper, packet, chain, NATIVE_CAP,
                                      'actual_native_terminal', trace[-1],
                                      copy_from=f'paid_{len(trace)}' if len(trace) in DIAGNOSTIC_PAID else None)
    for paid in DIAGNOSTIC_PAID:
        name = f'paid_{paid}'
        if name not in checkpoints:
            checkpoints[name] = missing(out, name, paid, len(trace), 'native_stop_before_checkpoint')
    choices_record = [{key: event[key] for key in ('inputs', 'outputs')}
                      for event in backend.calls if event['method'] == 'select_topo_target']
    assert choices_record and all(row['outputs']['mode'] == 'G' for row in choices_record)
    assert choices_record[0]['outputs']['selected'] == first['executed_option']
    write(out / 'steps.json', trace)
    write(out / 'choices.json', choices_record)
    write(out / 'audit.json', runtime.audit)
    write(out / 'runtime_summary.json', summary)
    np.savez_compressed(out / 'observed_belief_trace.npz', belief=np.asarray(belief_trace, dtype=np.int8),
                        paid=np.arange(len(belief_trace), dtype=np.int16))
    receipt = dict(status='completed_execution', layout=layout, candidate_id=candidate,
                   source_commit=PRIOR_HEAD, source_protocol_sha256=sha(PROTOCOL),
                   runner_freeze_sha256=sha(HERE / 'freeze.json'), evaluator_inputs_read_by_actor=False,
                   neutral_g_continuation=True, reused_exact_archive=False,
                   matched_short_prefix_actions=common, actual_new_actions=len(trace) - common,
                   native_budget=NATIVE_CAP, paid_actions=len(trace), returned=termination['returned_to_anchor'],
                   collisions=sum(row['collision'] for row in trace), arrival='reached_bound_target',
                   paid_at_arrival=original['checkpoints']['arrival']['actual_paid'],
                   path_length_m=trace[-1]['path_length_m'], elapsed_s=time.monotonic() - start_s,
                   decision_wall_s_total=sum(row['decision_wall_s'] for row in trace),
                   termination=termination, observed_native_termination=termination,
                   final_packet_chain_sha256=chain, checkpoints=checkpoints,
                   checkpoint_recovery_file_sha256=sha(out / 'checkpoint_recovery.json'),
                   global_planning_calls=len(choices_record),
                   files_sha256={str(path.relative_to(out)): sha(path)
                                 for path in sorted(out.rglob('*')) if path.is_file()})
    write(out / 'manifest.json', receipt)
    print(json.dumps({key: receipt[key] for key in ('layout', 'candidate_id', 'paid_actions', 'returned', 'collisions')}), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('layout')
    parser.add_argument('candidate', type=int)
    args = parser.parse_args()
    assert (args.layout, args.candidate) in SCHEDULE
    frozen = read(HERE / 'freeze.json')
    assert all(sha(Path(path)) == digest for path, digest in frozen['files_sha256'].items())
    out = HERE / 'runs' / args.layout / f'candidate_{args.candidate}'
    out.mkdir(parents=True, exist_ok=False)
    try:
        run(args.layout, args.candidate, out)
    except BaseException as error:
        (out / 'failure.json').write_text(json.dumps(dict(status='failed', layout=args.layout,
            candidate_id=args.candidate, error=repr(error), traceback=traceback.format_exc()), indent=2) + '\n')
        raise


if __name__ == '__main__':
    main()
