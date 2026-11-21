#!/usr/bin/env python3
"""Run a whole native exploration episode with an optional same-goal route chooser."""

import argparse
from copy import deepcopy
import json
from pathlib import Path
import sys
import time
import traceback
from types import SimpleNamespace

HERE = Path(__file__).resolve().parent
STUDY = HERE.parent
ROOT = HERE.parents[1]
MECHANISM = ROOT / 'semantic_large_scene_20261121/mechanism/runner'
YIELD = ROOT / 'candidate_route_yield_20261121/runner'
B11 = ROOT / 'paired_unseen_gain_20261120/execution'
STRICT = ROOT / 'paired_unseen_followup_20261120/frontier_strict'
PROTOCOL = STUDY / 'PROTOCOL.json'
METHOD = STUDY / 'route_choice.py'
HOLDOUT = STUDY / 'holdout'
sys.dont_write_bytecode = True
sys.path.insert(0, str(YIELD))
from execute import array_sha, capture, chain_hash, np, read, sha, source_module

ARM_MODES = {
    'newG': ('G', 'G'),
    'newS': ('G', 'S'),
    'originalS': ('S', None),
    'F_strict': ('G', None),
}


def load_source(name, path):
    import importlib.util

    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def setup_runtime(stage, layout, arm, study_protocol):
    if stage == 'development':
        archived = source_module(read(YIELD.parent / 'PROTOCOL.json'))
        initial = archived.verify(layout)
        binding = archived.binding
        definition = archived.SCENES / 'mechanism_world.py'
        scenes = load_source('nso_same_goal_development_world', definition)
        cohort = archived.LAYOUTS
        config = archived.execution_protocol()
        seed = archived.PROTOCOL['seed']
        make_world = lambda name: scenes.make_world(name, seed=seed)
    else:
        sys.path.insert(0, str(B11))
        import runtime_binding as binding

        initial = read(HOLDOUT / 'prepared' / layout / 'initial_state.json')
        scenes = load_source('nso_same_goal_holdout_world', HOLDOUT / 'scene.py')
        cohort = tuple(study_protocol['heldout']['layouts'])
        config = read(B11 / 'protocol.json')
        make_world = scenes.make_world
        config['layouts'] = {name: {'scene_id': f'same-goal-holdout-{name}'} for name in cohort}
    old = binding.install()
    from nso.cpu_sensor_contract_v10 import GridTransform

    assert layout in cohort
    group = study_protocol['development' if stage == 'development' else 'heldout']
    native_budget = int(group['paid_action_budget'])
    config['policy']['paid_action_budget'] = native_budget
    config['arms'] = {mode: dict(score_mode=mode, predictor='visibility') for mode in ('G', 'S')}
    binding.ARMS = config['arms']
    old.method.ARMS = tuple(config['arms'])
    old.method.MODE = {mode: mode for mode in config['arms']}
    old.protocol = lambda: config

    def world_and_packet(name):
        world = make_world(name)
        assert world.config.max_steps >= native_budget
        transform = GridTransform(tuple(world.shape), world.config.resolution_m)
        return world, transform, old.packet(world, transform, world.sense(), name)

    old.world_and_packet = world_and_packet
    global_mode, local_mode = ARM_MODES[arm]
    values = old.bootstrap(layout, global_mode)
    world, transform, packet, components, runtime, effective, owner = values
    assert world.step_count == 0
    assert packet.sha256() == initial['initial_packet_sha256']
    assert [*packet.position, packet.heading] == initial['anchor']
    assert owner['owner_sha256'] == initial['initial_owner_sha256']
    assert components._cpu_backend.scenes[0]['ledger'].remaining_budget == native_budget
    return old, values, initial, global_mode, local_mode


def check_local_sidecar(origin, selected, state):
    permitted = {'states', 'actions', 'outbound_states', 'outbound_actions', 'outbound_cost', 'cost', 'arrival_action'}
    assert set(origin) == set(selected)
    assert all(origin[key] == selected[key] for key in origin if key not in permitted)
    assert selected['outbound_states'][0] == [*state['packet'].position, state['packet'].heading]
    assert selected['outbound_states'][-1] == selected['pose'] == origin['pose']
    assert selected['outbound_cost'] == len(selected['outbound_actions']) == selected['arrival_action']
    assert selected['outbound_cost'] > 0 and len(selected['outbound_states']) == selected['outbound_cost'] + 1
    assert selected['actions'] == selected['outbound_actions'] + selected['return_actions']
    assert selected['states'] == selected['outbound_states'] + selected['return_states'][1:]
    assert selected['cost'] == len(selected['actions']) <= state['ledger'].remaining_budget
    assert selected['cost'] == selected['outbound_cost'] + selected['return_cost']


def local_route_adapter(components, old, mode, config):
    from nso.cpu_sensor_contract_v10 import digest

    module = load_source('nso_same_goal_route_choice', METHOD)
    original_get = components.get_selected_option
    backend = components._cpu_backend
    events = []

    def local_option(scene_index):
        native = original_get(scene_index)
        if native is None:
            return None
        state = backend.scenes[scene_index]
        before = digest(state['selected'])
        local, audit = module.choose_route(state, native, mode, old.method.route_union_area, config)
        assert digest(state['selected']) == before, 'global selection modified by local route ranking'
        check_local_sidecar(native, local, state)
        events.append(dict(paid_before=state['packet'].action_id, pose=[*state['packet'].position, state['packet'].heading],
                           frame_id=state['packet'].frame_id, packet_sha256=state['packet'].sha256(),
                           observed_belief_sha256=array_sha(state['mapper'].belief),
                           remaining_budget=state['ledger'].remaining_budget, global_candidate_id=native['candidate_id'],
                           global_mode=backend.args.cpu_score_mode, local_mode=mode, option_id=native['option_id'],
                           selection_call_id=native['selection_call_id'], same_global_target=native['pose'],
                           global_native_sha256=before, native_outbound_actions=native['outbound_actions'],
                           native_outbound_states=native['outbound_states'],
                           selected_outbound_actions=local['outbound_actions'],
                           selected_outbound_states=local['outbound_states'],
                           native_return_actions=native['return_actions'],
                           rebound_return_actions=local['return_actions'],
                           native_return_states=native['return_states'],
                           rebound_return_states=local['return_states'],
                           active_option_id=native['option_id'],
                           active_local_route_id=audit['selected_route_id'],
                           active_outbound_sha256=audit['chosen_outbound_sha256'],
                           differs_from_native=local['outbound_actions'] != native['outbound_actions'],
                           choice_audit=audit))
        return local

    components.get_selected_option = local_option
    return events


def frontier_adapter(components, runtime, world):
    sys.path.insert(0, str(STRICT))
    from strict_policy import FrontierPolicy
    from controller import FrontierDriver
    from env.grid_exploration import GridObservation, readonly_copy

    policy = FrontierPolicy(SimpleNamespace(resolution_m=world.config.resolution_m,
        robot_radius_m=world.config.robot_radius_m, sensor_range_m=world.config.max_depth_m,
        sensor_fov_deg=world.config.fov_deg), method='nearest_frontier', max_candidates=24)
    mapper = runtime.states[0]['mapper']

    def observation():
        current = runtime.states[0]['packet']
        return GridObservation(readonly_copy(mapper.belief), readonly_copy(mapper.visible),
                               current.position, current.heading, current.action_id, current.collision)

    return FrontierDriver(components, runtime, policy, observation)


def run(stage, layout, arm, protocol, out):
    old, values, initial, global_mode, local_mode = setup_runtime(stage, layout, arm, protocol)
    from runtime_binding import write

    world, transform, packet, components, runtime, effective, owner = values
    mapper = runtime.states[0]['mapper']
    backend = components._cpu_backend
    local_events = local_route_adapter(components, old, local_mode, protocol['local_route']) if local_mode else []
    frontier = frontier_adapter(components, runtime, world) if arm == 'F_strict' else None
    group = protocol['development' if stage == 'development' else 'heldout']
    native_budget = group['paid_action_budget']
    start_pose = [*packet.position, packet.heading]
    initial_packet_sha = packet.sha256()
    chain = chain_hash('0' * 64, initial_packet_sha)
    original_initial_belief_sha = array_sha(mapper.belief)
    if stage == 'development':
        source_initial = read(MECHANISM / 'layouts' / layout / 'runs/G/attempt_001/intervention.json')['intervention']
        assert original_initial_belief_sha == source_initial['belief_sha256']
    checkpoints = {'start': capture(out, 'start', mapper, packet, chain, 0, 'untouched_start')}
    frozen = read(HERE / f'{stage}_freeze.json')
    write(out / 'attempt.json', dict(status='started', stage=stage, arm=arm, layout=layout,
         local_policy=local_mode, global_policy=global_mode if frontier is None else None,
         initial_packet_sha256=initial_packet_sha, initial_belief_sha256=original_initial_belief_sha,
         initial_owner_sha256=owner['owner_sha256'], initial_pose=start_pose,
         source_protocol_sha256=sha(PROTOCOL), runner_freeze_sha256=sha(HERE / f'{stage}_freeze.json'),
         native_budget=native_budget, scene_files=frozen['world_files_sha256'], effective_runtime=effective))
    started = time.monotonic()
    steps = []
    belief_trace = [mapper.belief.copy()]
    active_paths = {}
    local_events_by_option_id = {}
    local_events_cursor = 0
    while not runtime.states[0]['closed']:
        decision_start = time.monotonic()
        action = frontier.next_action() if frontier is not None else runtime.next_local_action(0)
        decision_s = time.monotonic() - decision_start
        if action is None:
            break
        for choice in local_events[local_events_cursor:]:
            local_events_by_option_id[choice['option_id']] = choice
        local_events_cursor = len(local_events)
        state = runtime.states[0]
        pending = state['pending']
        assert pending['allowed'], 'recorded physical action bypassed native guard'
        active = state['option']
        phase = state['phase']
        target = None if active is None else list(active['pose'])
        active_id = None if active is None else active['option_id']
        if active is None:
            route_action_index = None
        else:
            route_action_index = active_paths.get(active_id, 0)
            assert active['outbound_actions'][route_action_index] == action
            active_paths[active_id] = route_action_index + 1
        picked = local_events_by_option_id.get(active_id)
        assert local_mode is None or active is None or picked is not None
        previous_belief = belief_trace[-1]
        raw, collision, done = world.step(action)
        packet = old.packet(world, transform, raw, layout, action, collision, done)
        owner = old.front.associate(mapper, raw, packet.frame_id)
        old.method.observe(mapper, packet.frame)
        runtime.observe(0, world.step_count, None, None, None, sensor_packet=packet)
        paid = packet.action_id
        packet_sha = packet.sha256()
        chain = chain_hash(chain, packet_sha)
        pose = [*world.position, world.heading]
        belief_trace.append(mapper.belief.copy())
        row = dict(step=paid, action=action, position=list(world.position), heading=world.heading,
                   collision=bool(collision), done=bool(done), packet_sha256=packet_sha,
                   packet_chain_sha256=chain, frame_sha256=old.method._frame_hash(raw),
                   nonsemantic_packet_sha256=old.base.nonsemantic_packet_sha(packet),
                   owner_sha256=owner['owner_sha256'], active_option_id=active_id,
                   active_candidate_id=None if active is None else active['candidate_id'],
                   selection_call_id=None if active is None else active['selection_call_id'],
                   route_action_index_zero_based=route_action_index,
                   local_route_id=None if picked is None else picked['active_local_route_id'],
                   active_outbound_sha256=None if picked is None else picked['active_outbound_sha256'],
                   selected_target_pose=target, phase_before_action=phase,
                   matched_target_pose_after_step=(pose == target) if target is not None else None,
                   guard_allowed=True, guard_reason=pending['reason'],
                   reserved_return_cost=pending['reserved_return_cost'],
                   observed_known_cells=int(np.count_nonzero(mapper.belief != -1)),
                   observed_free_cells=int(np.count_nonzero(mapper.belief == 0)),
                   newly_known_cells=int(np.count_nonzero((previous_belief == -1) & (mapper.belief != -1))),
                   path_length_m=world.moves * world.config.resolution_m, decision_wall_s=decision_s)
        steps.append(row)
        if paid in protocol.get('diagnostic_checkpoints', []):
            name = f'paid_{paid}'
            checkpoints[name] = capture(out, name, mapper, packet, chain, paid, 'paid_diagnostic_checkpoint', row)
        if paid % 10 == 0:
            write(out / 'heartbeat.json', dict(actual_paid=paid, native_budget=native_budget,
                local_choices=len(local_events), elapsed_s=time.monotonic() - started))
    summary = runtime.sensor_episode_summary(0)
    termination = summary['termination']
    assert summary['closed'] and termination is not None
    assert len(steps) == packet.action_id == termination['final_action_id'] <= native_budget
    final_step = steps[-1] if steps else None
    checkpoints['terminal'] = capture(out, 'terminal', mapper, packet, chain, native_budget, 'actual_native_terminal',
                                      final_step, copy_from='start' if not steps else None)
    global_events = [{key: event[key] for key in ('call_id', 'action_id', 'frame_id', 'input_sha256',
                      'output_sha256', 'inputs', 'outputs')}
                     for event in backend.calls if event['method'] == 'select_topo_target']
    assert all(event['outputs']['mode'] == global_mode for event in global_events)
    if frontier is not None:
        assert not global_events
        write(out / 'frontier_events.json', frontier.events)
    if local_mode:
        event_by_id = {event['call_id']: event for event in global_events}
        for local_event in local_events:
            observed_global = event_by_id[local_event['selection_call_id']]
            assert observed_global['outputs']['selected']['pose'] == local_event['same_global_target']
    write(out / 'global_choices.json', global_events)
    write(out / 'local_choices.json', local_events)
    write(out / 'steps.json', steps)
    write(out / 'audit.json', runtime.audit)
    write(out / 'runtime_summary.json', summary)
    np.savez_compressed(out / 'observed_belief_trace.npz', belief=np.asarray(belief_trace, dtype=np.int8),
                        paid=np.arange(len(belief_trace), dtype=np.int16))
    receipt = dict(status='completed_execution', stage=stage, layout=layout, arm=arm,
         native_budget=native_budget, paid_actions=packet.action_id,
         initial_packet_sha256=initial_packet_sha, initial_belief_sha256=original_initial_belief_sha,
         initial_pose=start_pose, final_pose=[*packet.position, packet.heading], final_packet_chain_sha256=chain,
         returned=termination['returned_to_anchor'], collisions=sum(row['collision'] for row in steps),
         observed_native_termination=termination, actual_closed_loop=True, forced_route_or_goal=False,
         evaluator_inputs_read_by_actor=False, local_route_only=local_mode is not None,
         global_mode=global_mode if frontier is None else None, local_mode=local_mode,
         native_global_choices=len(global_events), local_route_choices=len(local_events),
         path_length_m=0. if final_step is None else final_step['path_length_m'],
         checkpoints=checkpoints, elapsed_s=time.monotonic() - started,
         files_sha256={str(path.relative_to(out)): sha(path) for path in sorted(out.rglob('*')) if path.is_file()})
    write(out / 'manifest.json', receipt)
    print(json.dumps({key: receipt[key] for key in ('stage', 'layout', 'arm', 'paid_actions', 'returned', 'collisions')}), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', choices=('development', 'holdout'))
    parser.add_argument('layout')
    parser.add_argument('arm', choices=tuple(ARM_MODES))
    args = parser.parse_args()
    protocol = read(PROTOCOL)
    frozen = read(HERE / f'{args.stage}_freeze.json')
    assert dict(layout=args.layout, arm=args.arm) in frozen['schedule']
    assert all(sha(Path(path)) == digest for path, digest in frozen['files_sha256'].items())
    out = HERE / 'runs' / args.stage / args.layout / args.arm
    out.mkdir(parents=True, exist_ok=False)
    try:
        run(args.stage, args.layout, args.arm, protocol, out)
    except BaseException as error:
        (out / 'failure.json').write_text(json.dumps(dict(status='failed', layout=args.layout, arm=args.arm,
            stage=args.stage, error=repr(error), traceback=traceback.format_exc()), indent=2) + '\n')
        raise


if __name__ == '__main__':
    main()
