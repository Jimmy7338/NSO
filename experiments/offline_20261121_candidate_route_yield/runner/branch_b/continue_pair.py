#!/usr/bin/env python3
"""Verify the archived original and continue only the selected alternative."""

import argparse
from copy import deepcopy
import json
from pathlib import Path
import shutil
import sys
import time
import traceback

HERE = Path(__file__).resolve().parent
RUNNER = HERE.parent
sys.path.insert(0, str(RUNNER))
from execute import (CONTROLS, PROTOCOL, TRACE, array_sha, capture, chain_hash, np,
                     read, sha, source_module, source_records)

DECISION = PROTOCOL.with_name('decision.json')
LAYOUT = 'warehouse_right'
CANDIDATES = (2, 3)
COMMON_PAID = 30
NATIVE_CAP = 256


def short_run(candidate):
    return RUNNER / 'runs' / LAYOUT / f'candidate_{candidate}'


def path_for(candidate):
    return HERE / 'runs' / LAYOUT / f'candidate_{candidate}'


def csv_free_selection_check(decision):
    assert decision['branch'] == 'B'
    selected = decision['continuation_pair']
    assert selected['layout'] == LAYOUT and tuple(selected['candidate_ids']) == CANDIDATES


def freeze():
    csv_free_selection_check(read(DECISION))
    paths = [PROTOCOL, DECISION, Path(__file__), RUNNER / 'execute.py', RUNNER / 'freeze.json',
             RUNNER / 'supervisor_receipt.json']
    for candidate in CANDIDATES:
        short = short_run(candidate)
        paths.extend(short / name for name in ('manifest.json', 'first_anchor.json', 'steps.json',
                                              f'paid_{COMMON_PAID}/checkpoint.json'))
    receipt = dict(status='frozen_after_short_branch_decision_before_continuation', layout=LAYOUT,
                   candidates=list(CANDIDATES), original_archived_candidate=2, continued_candidate=3,
                   common_paid=COMMON_PAID, total_native_budget=NATIVE_CAP, neutral_g_continuation=True,
                   files_sha256={str(p): sha(p) for p in paths})
    with (HERE / 'freeze.json').open('x') as stream:
        json.dump(receipt, stream, indent=2)
        stream.write('\n')


def read_short(candidate):
    out = short_run(candidate)
    short = read(out / 'manifest.json')
    assert short['status'] == 'completed_execution' and short['arrival'] == 'reached_bound_target'
    assert short['paid_actions'] == COMMON_PAID and short['source_commit'] == read(PROTOCOL)['source_commit']
    for path in ('first_anchor.json', 'steps.json', f'paid_{COMMON_PAID}/checkpoint.json'):
        assert sha(out / path) == short['files_sha256'][path]
    return read(out / 'first_anchor.json'), read(out / 'steps.json'), read(out / f'paid_{COMMON_PAID}/checkpoint.json')


def write_receipt(out, receipt):
    from runtime_binding import write

    receipt['files_sha256'] = {str(p.relative_to(out)): sha(p) for p in sorted(out.rglob('*')) if p.is_file()}
    receipt['branch_freeze_sha256'] = sha(HERE / 'freeze.json')
    write(out / 'manifest.json', receipt)


def reuse_original(protocol, mechanism, out):
    from runtime_binding import write

    archived = mechanism.HERE / 'layouts' / LAYOUT / 'runs/G/attempt_001'
    source_manifest = read(archived / 'artifact_manifest.json')
    assert source_manifest['status'] == 'completed_execution'
    assert source_manifest['freeze_sha256'] == sha(mechanism.HERE / 'freeze.json')
    for name in ('result.json', 'steps.json', 'choices.json', 'intervention.json', 'runtime_audit.json',
                 'episode_summary.json', 'final_tsdf_mesh.ply', 'final_belief.npz'):
        assert sha(archived / name) == source_manifest['files_sha256'][name]
    anchor, recovered, short_checkpoint = read_short(2)
    historical_steps = read(archived / 'steps.json')
    assert read(archived / 'intervention.json')['intervention']['belief_sha256'] == anchor['belief_sha256']
    assert anchor['executed_option'] == anchor['source_selected_option']
    assert short_checkpoint['active_candidate_id'] == 2 and short_checkpoint['actual_paid'] == COMMON_PAID
    assert all(all(a[key] == b[key] for key in TRACE)
               for a, b in zip(recovered, historical_steps[:COMMON_PAID], strict=True))
    write(out / 'checkpoint_recovery.json', dict(paid=COMMON_PAID,
        short_checkpoint_file_sha256=sha(short_run(2) / f'paid_{COMMON_PAID}' / 'checkpoint.json'),
        packet_chain_sha256=short_checkpoint['packet_chain_sha256'],
        belief_sha256=short_checkpoint['belief_sha256'],
        native_mesh_fields=short_checkpoint['native_mesh_fields'],
        short_snapshot_files_sha256=short_checkpoint['files_sha256'],
        original_trace_prefix_fields_identical=True))
    result = read(archived / 'result.json')
    assert result['paid_action_budget'] == NATIVE_CAP
    assert result['initial_packet_sha256'] == anchor['packet_sha256']
    assert result['paid_actions'] == len(historical_steps)
    assert result['effective_args']['cpu_score_mode'] == 'G'
    assert read(archived / 'episode_summary.json')['termination'] == result['termination']
    chain = chain_hash('0' * 64, anchor['packet_sha256'])
    for step in historical_steps:
        chain = chain_hash(chain, step['packet_sha256'])
        step['packet_chain_sha256'] = chain
    assert historical_steps[COMMON_PAID - 1]['packet_chain_sha256'] == short_checkpoint['packet_chain_sha256']
    terminal = out / 'terminal'
    terminal.mkdir()
    shutil.copyfile(archived / 'final_tsdf_mesh.ply', terminal / 'native_mesh.ply')
    shutil.copyfile(archived / 'final_belief.npz', terminal / 'belief.npz')
    with np.load(terminal / 'belief.npz', allow_pickle=False) as saved:
        belief_sha = array_sha(saved['belief'])
    snapshot = dict(status='captured', requested_paid=NATIVE_CAP, actual_paid=result['paid_actions'],
                    packet_sha256=historical_steps[-1]['packet_sha256'], packet_chain_sha256=chain,
                    pose=result['final_pose'], belief_sha256=belief_sha, outcome='archived_actual_terminal',
                    files_sha256={name: sha(terminal / name) for name in ('native_mesh.ply', 'belief.npz')})
    write(terminal / 'checkpoint.json', snapshot)
    write(out / 'steps.json', historical_steps)
    write(out / 'first_anchor.json', anchor)
    shutil.copyfile(archived / 'runtime_audit.json', out / 'audit.json')
    shutil.copyfile(archived / 'episode_summary.json', out / 'runtime_summary.json')
    receipt = dict(status='completed_execution', layout=LAYOUT, candidate_id=2, reused_exact_archive=True,
                   source_original_archive=str(archived), archive_manifest_sha256=sha(archived / 'artifact_manifest.json'),
                   source_commit=protocol['source_commit'], source_protocol_sha256=sha(PROTOCOL),
                   evaluator_inputs_read_by_actor=False,
                   matched_short_prefix_actions=COMMON_PAID, actual_new_actions=0, native_budget=NATIVE_CAP,
                   paid_actions=result['paid_actions'], returned=result['termination']['returned_to_anchor'],
                   collisions=result['collisions'], arrival='reached_bound_target', paid_at_arrival=COMMON_PAID,
                   path_length_m=result['path_length_m'], elapsed_s=result['elapsed_s'],
                   decision_wall_s_total=result['decision_wall_s_total'], termination=result['termination'],
                   observed_native_termination=result['termination'], final_packet_chain_sha256=chain, checkpoints={'terminal': snapshot})
    write_receipt(out, receipt)


def extend_alternate(protocol, mechanism, out):
    old, gate = mechanism.install()
    from nso.cpu_sensor_contract_v10 import digest, json_value
    from runtime_binding import write

    source_initial = mechanism.verify(LAYOUT)
    choices, _, archived_gate, _ = source_records(mechanism, LAYOUT)
    anchor, recovered, short_checkpoint = read_short(3)
    world, transform, packet, components, runtime, effective, owner = old.bootstrap(LAYOUT, 'G')
    assert source_initial['stop_budget'] == NATIVE_CAP
    assert packet.sha256() == anchor['packet_sha256'] and [*packet.position, packet.heading] == source_initial['anchor']
    assert owner['owner_sha256'] == anchor['initial_owner_sha256']
    backend = components._cpu_backend
    mapper = runtime.states[0]['mapper']
    assert array_sha(mapper.belief) == anchor['belief_sha256']
    original_select = backend.select_target
    selected_once = False

    def select_once(scene):
        nonlocal selected_once
        target = original_select(scene)
        if selected_once:
            assert backend.scenes[scene]['last_selection']['mode'] == 'G'
            return target
        assert scene == 0 and world.step_count == 0 and gate.trigger == archived_gate
        record = backend.calls[-1]
        assert {key: record[key] for key in ('inputs', 'outputs')} == choices[0]
        native = record['outputs']
        selected = deepcopy(native['candidates'][3])
        selected.update({key: native['selected'][key] for key in CONTROLS})
        assert selected == anchor['executed_option']
        updated = {**native, 'selected': selected}
        backend.scenes[0]['selected'] = deepcopy(selected)
        backend.scenes[0]['last_selection'] = json_value(updated)
        record['outputs'] = updated
        record['output_sha256'] = digest(updated)
        selected_once = True
        return np.asarray(selected['pose'][:2], dtype=int)

    backend.select_target = select_once
    write(out / 'first_anchor.json', anchor)
    started = time.monotonic()
    chain = chain_hash('0' * 64, packet.sha256())
    steps = []
    checkpoint_verified = False
    while not runtime.states[0]['closed']:
        before_decision = time.monotonic()
        action = runtime.next_local_action(0)
        decision_s = time.monotonic() - before_decision
        if action is None:
            break
        state = runtime.states[0]
        active = state['option']
        phase = state['phase']
        target = None if active is None else list(active['pose'])
        before_belief = mapper.belief.copy()
        raw, collision, done = world.step(action)
        packet = old.packet(world, transform, raw, LAYOUT, action, collision, done)
        owner = old.front.associate(mapper, raw, packet.frame_id)
        old.method.observe(mapper, packet.frame)
        runtime.observe(0, world.step_count, None, None, None, sensor_packet=packet)
        paid = world.step_count
        pose = [*world.position, world.heading]
        value = packet.sha256()
        chain = chain_hash(chain, value)
        step = dict(step=paid, action=action, position=list(world.position), heading=world.heading,
                    collision=bool(collision), done=bool(done), path_length_m=world.moves * world.config.resolution_m,
                    packet_sha256=value, nonsemantic_packet_sha256=old.base.nonsemantic_packet_sha(packet),
                    frame_sha256=old.method._frame_hash(raw), owner_sha256=owner['owner_sha256'],
                    packet_chain_sha256=chain, active_option_id=None if active is None else active['option_id'],
                    active_candidate_id=None if active is None else active['candidate_id'],
                    selected_target_pose=target, phase_before_action=phase,
                    matched_target_pose_after_step=target == pose if target is not None else None,
                    decision_wall_s=decision_s, recovery=paid <= COMMON_PAID,
                    observed_known_cells=int(np.count_nonzero(mapper.belief != -1)),
                    observed_free_cells=int(np.count_nonzero(mapper.belief == 0)),
                    newly_known_cells=int(np.count_nonzero((before_belief == -1) & (mapper.belief != -1))),
                    repeat_without_new_known_cells=not bool(np.any((before_belief == -1) & (mapper.belief != -1))),
                    changed_belief_cells=int(np.count_nonzero(before_belief != mapper.belief)))
        if paid <= COMMON_PAID:
            prior = recovered[paid - 1]
            assert all(step[key] == val for key, val in prior.items()), (paid, [key for key in prior if step[key] != prior[key]])
        if paid == COMMON_PAID:
            assert step['active_candidate_id'] == 3 and step['matched_target_pose_after_step'] is True
            assert chain == short_checkpoint['packet_chain_sha256'] and pose == short_checkpoint['pose']
            assert array_sha(mapper.belief) == short_checkpoint['belief_sha256']
            current_mesh = mapper.mesh()
            assert all(array_sha(getattr(current_mesh, field)) == short_checkpoint['native_mesh_fields'][f'{field}_sha256']
                       for field in ('vertices', 'triangles'))
            checkpoint_verified = True
            write(out / 'checkpoint_recovery.json', dict(paid=paid, packet_chain_sha256=chain,
                short_checkpoint_file_sha256=sha(short_run(3) / f'paid_{COMMON_PAID}' / 'checkpoint.json'),
                all_step_fields_identical=True, native_mesh_fields_matched=True, belief_matched=True,
                belief_sha256=short_checkpoint['belief_sha256'],
                native_mesh_fields=short_checkpoint['native_mesh_fields'],
                short_snapshot_files_sha256=short_checkpoint['files_sha256']))
        steps.append(step)
        if paid % 10 == 0:
            write(out / 'heartbeat.json', dict(actual_paid=paid, native_budget=NATIVE_CAP,
                                               checkpoint_verified=checkpoint_verified,
                                               elapsed_s=time.monotonic() - started))
    assert selected_once and checkpoint_verified
    summary = runtime.sensor_episode_summary(0)
    termination = summary['termination']
    assert summary['closed'] and termination is not None
    assert len(steps) == packet.action_id == termination['final_action_id'] <= NATIVE_CAP
    terminal = capture(out, 'terminal', mapper, packet, chain, NATIVE_CAP, 'actual_native_terminal', steps[-1])
    write(out / 'steps.json', steps)
    write(out / 'audit.json', runtime.audit)
    write(out / 'runtime_summary.json', summary)
    receipt = dict(status='completed_execution', layout=LAYOUT, candidate_id=3, reused_exact_archive=False,
                   source_commit=protocol['source_commit'], source_protocol_sha256=sha(PROTOCOL),
                   evaluator_inputs_read_by_actor=False,
                   matched_short_prefix_actions=COMMON_PAID, actual_new_actions=len(steps) - COMMON_PAID,
                   native_budget=NATIVE_CAP, paid_actions=len(steps), returned=termination['returned_to_anchor'],
                   collisions=sum(s['collision'] for s in steps), arrival='reached_bound_target',
                   paid_at_arrival=COMMON_PAID, path_length_m=steps[-1]['path_length_m'],
                   elapsed_s=time.monotonic() - started,
                   decision_wall_s_total=sum(s['decision_wall_s'] for s in steps), termination=termination,
                   observed_native_termination=termination,
                   final_packet_chain_sha256=chain, checkpoints={'terminal': terminal})
    write_receipt(out, receipt)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--freeze', action='store_true')
    args = parser.parse_args()
    if args.freeze:
        freeze()
        return
    frozen = read(HERE / 'freeze.json')
    assert all(sha(Path(path)) == expected for path, expected in frozen['files_sha256'].items())
    csv_free_selection_check(read(DECISION))
    protocol = read(PROTOCOL)
    mechanism = source_module(protocol)
    branch_attempts = ((2, reuse_original), (3, extend_alternate))
    for candidate, work in branch_attempts:
        out = path_for(candidate)
        out.mkdir(parents=True, exist_ok=False)
        try:
            work(protocol, mechanism, out)
            print(f'Branch B candidate {candidate} provenance and native terminal saved', flush=True)
        except BaseException as error:
            (out / 'failure.json').write_text(json.dumps(dict(status='failed', candidate_id=candidate,
                error=repr(error), traceback=traceback.format_exc()), indent=2) + '\n')
            raise


if __name__ == '__main__':
    main()
