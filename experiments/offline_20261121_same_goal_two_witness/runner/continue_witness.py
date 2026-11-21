#!/usr/bin/env python3
"""Restore a published observation prefix and let one expanded local ranking act."""

import argparse
from copy import copy, deepcopy
import json
from pathlib import Path
import sys
import time
import traceback

HERE = Path(__file__).resolve().parent
STUDY = HERE.parent
ANALYSES = HERE.parents[1]
PROTOCOL = STUDY / 'PROTOCOL.json'
AUDIT_STUDY = ANALYSES / 'same_goal_pool_audit_20261121'
AUDIT_REPLAY = AUDIT_STUDY / 'replay'
HEAD = '3855d5d710e659adf665aa61dbc66eafb217c467'
MODES = ('G', 'S')
TARGETS = {
    'gallery_left': {'paid': 56, 'call_id': 230, 'state_file': '01_paid_056.json', 'comparison_paid': 104},
    'warehouse_right': {'paid': 142, 'call_id': 578, 'state_file': '04_paid_142.json', 'comparison_paid': 206},
}
GLOBAL_FIELDS = ('call_id', 'action_id', 'frame_id', 'input_sha256', 'output_sha256', 'inputs', 'outputs')
GUARD_FIELDS = ('call_id', 'action_id', 'frame_id', 'method', 'executed_option_id', 'selection_call_id', 'inputs', 'outputs')
sys.dont_write_bytecode = True
sys.path.insert(0, str(AUDIT_REPLAY))
from replay import (SOURCE as DEVELOPER, array_sha, chain_hash, json_sha, load_source, read,
                    read_archive, setup_runtime, sha)
from full_run import capture, check_local_sidecar, local_route_adapter, np
from execute import missing


def witness_inputs(layout):
    target = TARGETS[layout]
    publication = AUDIT_STUDY / 'publication/receipt.json'
    archive = AUDIT_REPLAY / 'runs' / layout
    path = archive / 'states' / target['state_file']
    assert read(publication)['status'] == 'verified_published' and read(publication)['head'] == HEAD
    supervised = read(AUDIT_REPLAY / 'supervisor_receipt.json')
    assert supervised['status'] == 'completed' and supervised['replay_freeze_sha256'] == sha(AUDIT_REPLAY / 'freeze.json')
    receipt = read(archive / 'manifest.json')
    assert receipt['status'] == 'completed_exact_archived_replay' and receipt['counterfactual_actions_executed'] == 0
    assert sha(path) == receipt['files_sha256'][str(path.relative_to(archive))]
    witness = read(path)
    assert (witness['layout'], witness['paid_before'], witness['global_call_id']) == (layout, target['paid'], target['call_id'])
    assert not witness['used_for_action_selection'] and witness['counterfactual_actions_executed'] == 0
    assert witness['expanded']['summary']['expanded_winners_split_by_prior']
    assert witness['expanded']['original_parity']['archived_full_audit_exact']
    archives, baseline_index = read_archive(layout)
    index = next(row for row in baseline_index if row['global_call_id'] == target['call_id'])
    assert all(witness[key] == val for key, val in index.items())
    return path, witness, archives


def guard_event(event):
    return {key: event[key] for key in GUARD_FIELDS}


def observed_prior_replan(backend, layout, witness):
    events = [guard_event(event) for event in backend.calls if event['method'] == 'assess_local_action'
              and event['action_id'] == witness['paid_before'] and event['call_id'] < witness['global_call_id']]
    if layout == 'warehouse_right':
        assert events and events[-1]['outputs']['allowed'] is False
        assert events[-1]['outputs']['reason'] == 'next_footprint_not_known_safe'
        assert events[-1]['executed_option_id'] != witness['original_native_option']['option_id']
    else:
        assert not events
    return events


def choose_expanded(state, native, mode, route_union_area, config, both, witness, consumer):
    """Run the published chooser with only the deduplicated cap made exhaustive."""
    shared = state['mapper']._dual_template_state
    original_cache = shared['cache']
    detached = dict(original_cache, gains={name: dict(gains) for name, gains in original_cache['gains'].items()})
    detached['models'] = {name: copy(model) for name, model in original_cache['models'].items()}
    for model in detached['models'].values():
        model.audit = deepcopy(model.audit)
        if hasattr(model, '_occlusion_repair_summary'):
            model._occlusion_repair_summary = deepcopy(model._occlusion_repair_summary)
        model._b11_cache = detached
    shared['cache'] = detached
    try:
        parity = consumer.enlarge_pool(state, deepcopy(native), route_union_area, config, both)
        assert parity == witness['expanded'], 'The complete live expanded pool or scores differ from the published witness'
        chooser, _ = consumer.frozen_chooser()
        count = both['G']['choice_audit']['proposal_diagnostics']['distinct_alternate_traces']
        full_config = dict(config, max_alternates=max(config['max_alternates'], count + 1))
        selected, audit = chooser.choose_route(state, deepcopy(native), mode, route_union_area, full_config)
        summary = parity['summary']
        assert audit['candidate_pool_sha256'] == summary['expanded_candidate_pool_sha256']
        assert audit['score_table_sha256'] == summary['expanded_score_table_sha256']
        assert audit['selected_route_id'] == summary['expanded_hypothetical_winners'][mode]
        assert audit['hypothetical_selected_route_id'] == summary['expanded_hypothetical_winners']
        candidate = next(row for row in audit['candidates'] if row['route_id'] == audit['selected_route_id'])
        assert candidate['outbound_actions'] == selected['outbound_actions']
        assert candidate['outbound_states'] == selected['outbound_states']
        return selected, audit, full_config['max_alternates']
    finally:
        shared['cache'] = original_cache


def snapshot_fields(packet, mapper, chain):
    return dict(packet_sha256=packet.sha256(), belief_sha256=array_sha(mapper.belief),
                packet_chain_sha256=chain, pose=[*packet.position, packet.heading])


def run(layout, mode, protocol, frozen, out):
    witness_path, witness, archives = witness_inputs(layout)
    consumer = load_source('same_goal_published_exhaustive_selector', AUDIT_STUDY / 'exhaustive.py')
    old_protocol = read(DEVELOPER / 'PROTOCOL.json')
    old, values, initial, global_mode, local_mode = setup_runtime('development', layout, 'newG', old_protocol)
    from nso.cpu_sensor_contract_v10 import digest
    from runtime_binding import write

    world, transform, packet, components, runtime, effective, owner = values
    mapper = runtime.states[0]['mapper']
    backend = components._cpu_backend
    source = archives['newG']
    config = old_protocol['local_route']
    target = TARGETS[layout]
    prefix = target['paid']
    pair = protocol['pairs'][layout]
    assert protocol['total_paid_action_budget'] == old_protocol['development']['paid_action_budget']
    budget = old_protocol['development']['paid_action_budget']
    # These values were frozen by the supervisor against the study protocol.
    schedule = frozen['capture_paid'][layout]
    assert global_mode == local_mode == 'G' and budget == 256
    assert world.seed == 20261121
    rng_identity = dict(scene_seed=int(world.seed),
       seed_components_when_depth_noise_enabled=['scene_seed', 'paid_frame_id', 739],
       deterministic_pose_geometry=True, depth_noise_sigma_m=float(world.config.depth_sigma_m),
       depth_dropout_fraction=float(world.config.dropout),
       stateful_generator_retained_by_simulator=False,
       source_packet_chain_required_through_paid=prefix)
    assert packet.sha256() == initial['initial_packet_sha256'] == source['manifest']['initial_packet_sha256']
    assert owner['owner_sha256'] == source['attempt']['initial_owner_sha256']
    assert array_sha(mapper.belief) == source['manifest']['initial_belief_sha256']
    neutral_events = local_route_adapter(components, old, 'G', config)
    neutral_get = components.get_selected_option
    local_events = []
    trace = []
    belief_trace = [mapper.belief.copy()]
    intervention = None
    global_index = 0
    chain = chain_hash('0' * 64, packet.sha256())
    start = capture(out, 'start', mapper, packet, chain, 0, 'restored_initial')
    assert start == source['manifest']['checkpoints']['start'] or (
        start['files_sha256'] == source['manifest']['checkpoints']['start']['files_sha256']
        and start['belief_sha256'] == source['manifest']['initial_belief_sha256'])
    checkpoints = {'start': start}
    started = time.monotonic()

    def witness_get(scene_index):
        nonlocal global_index, intervention
        base = neutral_get(scene_index)
        current = backend.calls[-1]
        assert current['method'] == 'select_topo_target'
        paid, call_id = current['action_id'], current['call_id']
        in_recovery = intervention is None
        if in_recovery:
            assert {key: current[key] for key in GLOBAL_FIELDS} == source['global_events'][global_index]
            global_index += 1
        if base is None:
            return None
        recorded_g = neutral_events[-1]
        if in_recovery:
            assert recorded_g == source['local_events'][len(local_events)]
        if paid != prefix or call_id != target['call_id']:
            if in_recovery:
                assert paid <= prefix and call_id < target['call_id']
            local_events.append(recorded_g | {'intervention': False})
            return base
        assert in_recovery and len(trace) == prefix and 'prefix' in checkpoints
        assert len(local_events) == witness['source_decision_ordinal']
        state = backend.scenes[scene_index]
        native = state['selected']
        assert native == witness['original_native_option']
        assert current['outputs']['selected'] == native and current['outputs']['mode'] == 'G'
        live = snapshot_fields(state['packet'], mapper, chain)
        assert live['packet_sha256'] == witness['packet_sha256'] and live['belief_sha256'] == witness['belief_sha256']
        assert checkpoints['prefix']['packet_chain_sha256'] == chain == source['trace'][prefix - 1]['packet_chain_sha256']
        assert live['pose'] == witness['actual_pose']
        before_guard = observed_prior_replan(backend, layout, witness)
        source_s = archives['newS']['local_events'][witness['source_decision_ordinal']]
        assert json_sha(recorded_g) == witness['original_source_G_SHA'] and json_sha(source_s) == witness['original_source_S_SHA']
        before = digest(native)
        selected, choice, cap = choose_expanded(state, native, mode, old.method.route_union_area, config,
                                                 {'G': recorded_g, 'S': source_s}, witness, consumer)
        assert before == digest(state['selected']) == witness['global_native_sha256']
        check_local_sidecar(native, selected, state)
        assert choice['selected_route_id'] == pair['archived_expected_winners_for_verification_only'][mode]
        assert selected['outbound_cost'] == pair['archived_nominal_outbound_for_verification_only'][mode]
        assert selected['return_cost'] == pair['native_reserved_return']
        assert snapshot_fields(state['packet'], mapper, chain) == live
        local_events.append(recorded_g | dict(local_mode=mode, choice_audit=choice, intervention=True,
            selected_outbound_actions=selected['outbound_actions'], selected_outbound_states=selected['outbound_states'],
            active_local_route_id=choice['selected_route_id'], active_outbound_sha256=choice['chosen_outbound_sha256'],
            rebound_return_actions=selected['return_actions'], rebound_return_states=selected['return_states'],
            differs_from_native=selected['outbound_actions'] != native['outbound_actions']))
        intervention = dict(status='expanded_winner_bound_once', layout=layout, mode=mode,
            paid_before=paid, global_call_id=call_id, native_option_id=native['option_id'], native_target=native['pose'],
            native_global_mode='G', source_commit=HEAD, witness_file=str(witness_path), witness_sha256=sha(witness_path),
            prior_source_state_id=witness['state_id'], source_G_SHA=witness['original_source_G_SHA'],
            source_S_SHA=witness['original_source_S_SHA'], prior_prefix_reproduced_before_selection=True,
            expanded_pool_and_scores_equal_to_offline_source=True, native_selected_option_unchanged=True,
            exact_deduplicated_proposal_count=choice['proposal_diagnostics']['distinct_alternate_traces'],
            actual_enlarged_alternate_cap=cap, original_neutral_alternate_cap=config['max_alternates'],
            original_return_actions=native['return_actions'], actual_return_actions=selected['return_actions'],
            native_outbound_actions=native['outbound_actions'], selected_outbound_actions=selected['outbound_actions'],
            selected_outbound_states=selected['outbound_states'], chosen_route_id=choice['selected_route_id'],
            expanded_candidate_pool_sha256=choice['candidate_pool_sha256'], expanded_score_table_sha256=choice['score_table_sha256'],
            pre_call_native_guard=before_guard, measured_prefix=live, restored_rng_identity=rng_identity,
            forced_goal=False, forced_route=False,
            next_local_decisions='original_neutral_G_cap4', post_call_native_guard=None)
        write(out / 'intervention.json', intervention)
        return selected

    components.get_selected_option = witness_get
    active_counts = {}
    arrivals = []
    while not runtime.states[0]['closed']:
        if len(trace) == prefix and 'prefix' not in checkpoints:
            checkpoints['prefix'] = capture(out, 'prefix', mapper, packet, chain, prefix, 'exact_common_witness_prefix', trace[-1])
            assert checkpoints['prefix']['packet_sha256'] == witness['packet_sha256']
            assert checkpoints['prefix']['belief_sha256'] == witness['belief_sha256']
        start_decision = time.monotonic()
        shadow_action = runtime.next_local_action(0)
        decision_s = time.monotonic() - start_decision
        if shadow_action is None:
            break
        state = runtime.states[0]
        pending = state['pending']
        assert pending['allowed']
        option = state['option']
        phase = state['phase']
        option_id = None if option is None else option['option_id']
        chosen_event = next((event for event in reversed(local_events) if event['option_id'] == option_id), None)
        choice = None if chosen_event is None else chosen_event['choice_audit']
        expected = source['trace'][len(trace)] if len(trace) < prefix else None
        if expected is None:
            assert intervention is not None, 'No released action is allowed without the one prescribed selection'
        else:
            assert shadow_action == expected['action']
            assert pending['allowed'] == expected['guard_allowed'] and pending['reason'] == expected['guard_reason']
            assert pending['reserved_return_cost'] == expected['reserved_return_cost']
            assert phase == expected['phase_before_action'] and option_id == expected['active_option_id']
        if len(trace) == prefix:
            guard_after = [guard_event(event) for event in backend.calls if event['method'] == 'assess_local_action'
                           and event['action_id'] == prefix and event['call_id'] > target['call_id']]
            assert guard_after and guard_after[-1]['outputs']['allowed'] and guard_after[-1]['outputs']['action'] == shadow_action
            intervention['post_call_native_guard'] = guard_after
            intervention['first_charged_action_after_witness'] = shadow_action
            latest_runtime = [event for event in runtime.audit if event.get('frame_id') == f'frame-{prefix}'
                              and event['event'] == 'exploration_joint_replan_after_rejection']
            if layout == 'warehouse_right':
                assert latest_runtime
                assert latest_runtime[0]['rejected_option_id'] == intervention['pre_call_native_guard'][-1]['executed_option_id']
                assert latest_runtime[0]['selected_option_id'] == intervention['native_option_id']
            intervention['runtime_same_frame_replan_events'] = latest_runtime
            write(out / 'intervention.json', intervention)
        index = None
        if option is not None:
            assert chosen_event is not None
            index = active_counts.get(option_id, 0)
            assert option['outbound_actions'][index] == shadow_action
            active_counts[option_id] = index + 1
        action = expected['action'] if expected is not None else shadow_action
        raw, collision, done = world.step(action)
        packet = old.packet(world, transform, raw, layout, action, collision, done)
        owner = old.front.associate(mapper, raw, packet.frame_id)
        old.method.observe(mapper, packet.frame)
        runtime.observe(0, world.step_count, None, None, None, sensor_packet=packet)
        paid = packet.action_id
        pose = [*packet.position, packet.heading]
        packet_sha = packet.sha256()
        chain = chain_hash(chain, packet_sha)
        before_map = belief_trace[-1]
        belief_trace.append(mapper.belief.copy())
        target_pose = None if option is None else option['pose']
        row = dict(step=paid, action=action, position=list(packet.position), heading=packet.heading,
            collision=bool(collision), done=bool(done), packet_sha256=packet_sha,
            frame_sha256=old.method._frame_hash(raw), nonsemantic_packet_sha256=old.base.nonsemantic_packet_sha(packet),
            owner_sha256=owner['owner_sha256'], packet_chain_sha256=chain,
            depth_values_sha256=array_sha(packet.frame.depth_m), rgb_values_sha256=array_sha(packet.frame.color_rgb),
            scan_ranges_sha256=array_sha(packet.scan.ranges_m),
            active_option_id=option_id, active_candidate_id=None if option is None else option['candidate_id'],
            selection_call_id=None if option is None else option['selection_call_id'],
            route_action_index_zero_based=index, local_route_id=None if choice is None else choice['selected_route_id'],
            active_outbound_sha256=None if choice is None else choice['chosen_outbound_sha256'],
            selected_target_pose=target_pose, phase_before_action=phase,
            matched_target_pose_after_step=(pose == target_pose) if target_pose is not None else None,
            guard_allowed=True, guard_reason=pending['reason'], reserved_return_cost=pending['reserved_return_cost'],
            path_length_m=world.moves * world.config.resolution_m, decision_wall_s=decision_s,
            observed_known_cells=int(np.count_nonzero(mapper.belief != -1)),
            observed_free_cells=int(np.count_nonzero(mapper.belief == 0)),
            newly_known_cells=int(np.count_nonzero((before_map == -1) & (mapper.belief != -1))),
            charged_new_continuation_action=expected is None,
            bound_witness_option=bool(intervention and option_id == intervention['native_option_id']))
        if expected is not None:
            differing = [key for key, value in expected.items() if key != 'decision_wall_s' and row[key] != value]
            assert not differing, (layout, paid, differing)
        trace.append(row)
        if row['bound_witness_option'] and row['matched_target_pose_after_step'] and not row['collision']:
            arrivals.append(paid)
            if 'arrival' not in checkpoints:
                checkpoints['arrival'] = capture(out, 'arrival', mapper, packet, chain, paid, 'bound_witness_goal_arrival', row)
        if paid in schedule and paid != prefix:
            name = f'paid_{paid}'
            checkpoints[name] = capture(out, name, mapper, packet, chain, paid, 'independent_scheduled_checkpoint', row,
                                        copy_from='arrival' if checkpoints.get('arrival', {}).get('actual_paid') == paid else None)
        if paid % 10 == 0:
            write(out / 'heartbeat.json', dict(actual_paid=paid, prefix_paid=prefix, native_budget=budget,
                intervention_bound=intervention is not None, elapsed_s=time.monotonic()-started))

    assert intervention is not None and 'prefix' in checkpoints
    assert sum(row['intervention'] for row in local_events) == 1
    assert all(row['choice_audit']['selected_mode'] == ('G' if not row['intervention'] else mode) for row in local_events)
    summary = runtime.sensor_episode_summary(0)
    termination = summary['termination']
    assert summary['closed'] and termination is not None
    paid = len(trace)
    assert paid == packet.action_id == termination['final_action_id'] <= budget
    copies = [name for name, result in checkpoints.items() if result['actual_paid'] == paid]
    checkpoints['terminal'] = capture(out, 'terminal', mapper, packet, chain, budget, 'actual_native_terminal', trace[-1],
                                      copy_from=copies[0] if copies else None)
    for step in schedule:
        name = f'paid_{step}'
        if step != prefix and name not in checkpoints:
            checkpoints[name] = missing(out, name, step, paid, 'ended_before_observation_checkpoint')
    if 'arrival' not in checkpoints:
        checkpoints['arrival'] = missing(out, 'arrival', prefix + len(intervention['selected_outbound_actions']), paid,
                                         'bound_witness_option_did_not_arrive')
    global_events = [{key: event[key] for key in GLOBAL_FIELDS} for event in backend.calls if event['method'] == 'select_topo_target']
    assert all(event['outputs']['mode'] == 'G' for event in global_events)
    intervention['actual_option_arrival_paid'] = arrivals[0] if arrivals else None
    intervention['charged_witness_option_actions'] = sum(row['bound_witness_option'] for row in trace)
    write(out / 'intervention.json', intervention)
    for name, data in [('steps.json', trace), ('global_choices.json', global_events), ('local_choices.json', local_events),
                       ('runtime_summary.json', summary), ('audit.json', runtime.audit), ('effective_runtime.json', effective)]:
        write(out / name, data)
    np.savez_compressed(out / 'observed_belief_trace.npz', belief=np.asarray(belief_trace, dtype=np.int8),
                        paid=np.arange(len(belief_trace), dtype=np.int16))
    receipt = dict(status='completed_execution', layout=layout, arm=mode, source_commit=HEAD,
        source_protocol_sha256=sha(PROTOCOL), runner_freeze_sha256=sha(HERE / 'freeze.json'),
        actual_closed_loop=True, evaluator_inputs_read_by_actor=False, restored_rng_identity=rng_identity,
        native_budget=budget,
        paid_actions=paid, replayed_archived_prefix_actions=prefix, actual_new_actions=paid-prefix,
        initial_packet_sha256=initial['initial_packet_sha256'], initial_pose=initial['anchor'],
        final_pose=[*packet.position, packet.heading], final_packet_chain_sha256=chain,
        single_intervention_call_id=target['call_id'], single_intervention_paid_before=prefix,
        actual_witness_arrival_paid=arrivals[0] if arrivals else None,
        global_mode='G', subsequent_local_mode='G', subsequent_local_alternate_cap=config['max_alternates'],
        forced_goal=False, forced_route=False, returned=termination['returned_to_anchor'],
        collisions=sum(row['collision'] for row in trace), observed_native_termination=termination,
        path_length_m=trace[-1]['path_length_m'], elapsed_s=time.monotonic()-started, checkpoints=checkpoints,
        files_sha256={str(path.relative_to(out)): sha(path) for path in sorted(out.rglob('*')) if path.is_file()})
    write(out / 'manifest.json', receipt)
    print(json.dumps({key: receipt[key] for key in ('layout','arm','paid_actions','returned','collisions','actual_witness_arrival_paid')}),flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('layout', choices=tuple(TARGETS))
    parser.add_argument('mode', choices=MODES)
    args = parser.parse_args()
    frozen = read(HERE / 'freeze.json')
    assert all(sha(Path(path)) == check for path,check in frozen['files_sha256'].items())
    out = HERE / 'runs' / args.layout / args.mode
    out.mkdir(parents=True, exist_ok=False)
    try:
        run(args.layout,args.mode,read(PROTOCOL),frozen,out)
    except BaseException as error:
        (out/'failure.json').write_text(json.dumps(dict(status='failed_witness_run', layout=args.layout, arm=args.mode,
           error=repr(error),traceback=traceback.format_exc()),indent=2)+'\n')
        raise


if __name__ == '__main__':
    main()
