#!/usr/bin/env python3
"""Restore measured states from exact prior actions and score larger pools offline."""

import argparse
from copy import copy, deepcopy
import hashlib
import json
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
import time
import traceback

HERE = Path(__file__).resolve().parent
STUDY = HERE.parent
SOURCE = HERE.parents[1] / 'same_goal_route_choice_20261121'
SOURCE_RUNNER = SOURCE / 'runner'
PROTOCOL = STUDY / 'PROTOCOL.json'
HEAD = '02e9c5dcc0da7df61f23d8683fec2dd92c145c1e'
COUNTS = {'warehouse_left': 8, 'warehouse_right': 6, 'gallery_left': 5, 'gallery_right': 6}
SOURCE_FILES = ('manifest.json', 'attempt.json', 'steps.json', 'global_choices.json', 'local_choices.json',
                'start/checkpoint.json', 'start/native_mesh.ply', 'start/belief.npz', 'terminal/checkpoint.json',
                'terminal/native_mesh.ply', 'terminal/belief.npz', 'runtime_summary.json')
GLOBAL_FIELDS = ('call_id', 'action_id', 'frame_id', 'input_sha256', 'output_sha256', 'inputs', 'outputs')
sys.dont_write_bytecode = True
sys.path.insert(0, str(SOURCE_RUNNER))
from full_run import (B11, MECHANISM, YIELD, array_sha, chain_hash, load_source, local_route_adapter,
                      read, setup_runtime, sha)


def json_sha(value):
    raw = json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()
    return hashlib.sha256(raw).hexdigest()


def comparable_trace(trace):
    return [{key: value for key, value in step.items() if key != 'decision_wall_s'} for step in trace]


def normalize_local(choice):
    value = deepcopy(choice)
    del value['local_mode']
    del value['choice_audit']['selected_mode']
    return value


def read_archive(layout):
    archives = {}
    publication = SOURCE / 'publication/receipt.json'
    assert read(publication)['status'] == 'verified_published' and read(publication)['head'] == HEAD
    source_freeze = SOURCE_RUNNER / 'development_freeze.json'
    supervisor = read(SOURCE_RUNNER / 'development_supervisor_receipt.json')
    assert supervisor['status'] == 'completed' and supervisor['runner_freeze_sha256'] == sha(source_freeze)
    for mode in ('newG', 'newS'):
        directory = SOURCE_RUNNER / 'runs/development' / layout / mode
        record = read(directory / 'manifest.json')
        assert record['status'] == 'completed_execution' and record['actual_closed_loop']
        assert record['returned'] and record['collisions'] == 0 and record['paid_actions'] == 256
        assert record['global_mode'] == 'G' and record['local_mode'] == mode[-1]
        attempt = read(directory / 'attempt.json')
        assert attempt['runner_freeze_sha256'] == sha(source_freeze)
        assert attempt['source_protocol_sha256'] == sha(SOURCE / 'PROTOCOL.json')
        assert set(SOURCE_FILES) - {'manifest.json'} <= set(record['files_sha256'])
        for name, checksum in record['files_sha256'].items():
            assert sha(directory / name) == checksum, (layout, mode, name)
        archives[mode] = dict(directory=directory, manifest=record, attempt=attempt,
                             trace=read(directory / 'steps.json'),
                             global_events=read(directory / 'global_choices.json'),
                             local_events=read(directory / 'local_choices.json'))
    neutral, semantic = (archives[mode] for mode in ('newG', 'newS'))
    assert comparable_trace(neutral['trace']) == comparable_trace(semantic['trace'])
    assert neutral['global_events'] == semantic['global_events']
    assert [normalize_local(event) for event in neutral['local_events']] == [
           normalize_local(event) for event in semantic['local_events']]
    for checkpoint in ('start', 'terminal'):
        assert neutral['manifest']['checkpoints'][checkpoint] == semantic['manifest']['checkpoints'][checkpoint]
    sources = []
    for index, (g, s) in enumerate(zip(neutral['local_events'], semantic['local_events'], strict=True)):
        if g['choice_audit']['admitted_candidate_count'] > 1:
            assert g['paid_before'] == s['paid_before']
            assert g['choice_audit']['candidate_pool_sha256'] == s['choice_audit']['candidate_pool_sha256']
            assert g['choice_audit']['score_table_sha256'] == s['choice_audit']['score_table_sha256']
            sources.append(dict(layout=layout, ordinal=len(sources), source_decision_ordinal=index,
                              paid_before=g['paid_before'], global_call_id=g['selection_call_id'],
                              packet_sha256=g['packet_sha256'], belief_sha256=g['observed_belief_sha256'],
                              original_source_G_SHA=json_sha(g), original_source_S_SHA=json_sha(s)))
    assert len(sources) == COUNTS[layout]
    return archives, sources


def isolate_candidate_ranking(state, native, union, config, audit, consumer):
    """Share immutable measured geometry while isolating model/audit caches."""
    shared = state['mapper']._dual_template_state
    live = shared['cache']
    detached = dict(live, gains={name: dict(gains) for name, gains in live['gains'].items()})
    detached['models'] = {name: copy(model) for name, model in live['models'].items()}
    for model in detached['models'].values():
        model.audit = deepcopy(model.audit)
        if hasattr(model, '_occlusion_repair_summary'):
            model._occlusion_repair_summary = deepcopy(model._occlusion_repair_summary)
        model._b11_cache = detached
    shared['cache'] = detached
    try:
        return consumer.enlarge_pool(state, deepcopy(native), union, config, deepcopy(audit))
    finally:
        shared['cache'] = live


def verify_measured_snapshot(name, archive, mapper, packet, chain):
    import open3d as o3d

    expected = archive['manifest']['checkpoints'][name]
    assert expected['packet_sha256'] == packet.sha256() and expected['packet_chain_sha256'] == chain
    assert expected['belief_sha256'] == array_sha(mapper.belief)
    assert expected['pose'] == [*packet.position, packet.heading]
    native = mapper.mesh()
    native_sha = {f'{field}_sha256': array_sha(getattr(native, field)) for field in ('vertices', 'triangles')}
    assert native_sha == expected['native_mesh_fields']
    with TemporaryDirectory() as directory:
        p = Path(directory) / 'native.ply'
        assert o3d.io.write_triangle_mesh(str(p), native, write_ascii=False)
        msh_sha = sha(p)
    assert msh_sha == expected['files_sha256']['native_mesh.ply']
    return dict(checkpoint=name, paid=packet.action_id, packet_sha256=packet.sha256(),
                belief_sha256=array_sha(mapper.belief), packet_chain_sha256=chain,
                native_mesh_sha256=msh_sha, native_mesh_fields=native_sha)


def replay(layout, frozen, out):
    archives, locked_states = read_archive(layout)
    assert locked_states == frozen['states'][layout]
    source = archives['newG']
    protocol = read(SOURCE / 'PROTOCOL.json')
    old, values, initial, global_mode, local_mode = setup_runtime('development', layout, 'newG', protocol)
    from nso.cpu_sensor_contract_v10 import digest
    from runtime_binding import write

    world, transform, packet, components, runtime, _effective, owner = values
    mapper = runtime.states[0]['mapper']
    backend = components._cpu_backend
    source_manifest = source['manifest']
    assert global_mode == local_mode == 'G'
    assert packet.sha256() == initial['initial_packet_sha256'] == source_manifest['initial_packet_sha256']
    assert owner['owner_sha256'] == source['attempt']['initial_owner_sha256']
    assert array_sha(mapper.belief) == source_manifest['initial_belief_sha256']
    original_events = local_route_adapter(components, old, local_mode, protocol['local_route'])
    original_get = components.get_selected_option
    consumer = load_source('nso_same_goal_offline_enlargement', Path(frozen['consumer_path']))
    targets = {item['source_decision_ordinal']: item for item in locked_states}
    payload_paths = []
    global_index = 0
    actor_actions = 0
    started = time.monotonic()

    def score_frozen_branch(scene_index):
        nonlocal global_index
        result = original_get(scene_index)
        event = backend.calls[-1]
        assert event['method'] == 'select_topo_target'
        current = {key: event[key] for key in GLOBAL_FIELDS}
        assert current == source['global_events'][global_index], (layout, global_index, 'global')
        global_index += 1
        if result is None:
            return None
        source_index = len(original_events) - 1
        rebuilt = original_events[source_index]
        previous = source['local_events'][source_index]
        assert rebuilt == previous, (layout, source_index, 'local', [key for key in rebuilt if rebuilt[key] != previous[key]])
        if source_index in targets:
            locked = targets[source_index]
            state = backend.scenes[scene_index]
            paid = state['packet'].action_id
            assert paid == locked['paid_before'] and locked['global_call_id'] == current['call_id']
            assert rebuilt['packet_sha256'] == locked['packet_sha256']
            assert rebuilt['observed_belief_sha256'] == locked['belief_sha256']
            assert json_sha(rebuilt) == locked['original_source_G_SHA']
            selected_digest = digest(state['selected'])
            actual_pose = [*state['packet'].position, state['packet'].heading]
            expanded = isolate_candidate_ranking(state, state['selected'], old.method.route_union_area,
                protocol['local_route'], {'G': previous, 'S': archives['newS']['local_events'][source_index]}, consumer)
            assert digest(state['selected']) == selected_digest and array_sha(mapper.belief) == locked['belief_sha256']
            assert world.step_count == actor_actions == paid
            assert [*state['packet'].position, state['packet'].heading] == actual_pose
            path = out / 'states' / f'{locked["ordinal"]:02d}_paid_{paid:03d}.json'
            write(path, {**locked, 'state_id': f'{layout}:{paid}:{locked["global_call_id"]}',
                          'global_native_sha256': selected_digest, 'actual_pose': actual_pose,
                          'original_native_option': deepcopy(state['selected']),
                          'original_choice_audit': rebuilt['choice_audit'], 'expanded': expanded,
                          'used_for_action_selection': False, 'counterfactual_actions_executed': 0})
            payload_paths.append(path)
        return result

    components.get_selected_option = score_frozen_branch
    chain = chain_hash('0' * 64, packet.sha256())
    start_receipt = verify_measured_snapshot('start', source, mapper, packet, chain)
    for expected in source['trace']:
        assert not runtime.states[0]['closed']
        shadow_action = runtime.next_local_action(0)
        assert shadow_action == expected['action'], (layout, expected['step'], 'shadow_action')
        pending = runtime.states[0]['pending']
        assert pending['allowed'] == expected['guard_allowed'] and pending['reason'] == expected['guard_reason']
        assert pending['reserved_return_cost'] == expected['reserved_return_cost']
        assert runtime.states[0]['phase'] == expected['phase_before_action']
        option = runtime.states[0]['option']
        assert (option['option_id'] if option is not None else None) == expected['active_option_id']
        raw, collision, done = world.step(expected['action'])
        packet = old.packet(world, transform, raw, layout, expected['action'], collision, done)
        owner = old.front.associate(mapper, raw, packet.frame_id)
        old.method.observe(mapper, packet.frame)
        runtime.observe(0, world.step_count, None, None, None, sensor_packet=packet)
        actor_actions += 1
        chain = chain_hash(chain, packet.sha256())
        assert actor_actions == packet.action_id == expected['step']
        assert list(packet.position) == expected['position'] and packet.heading == expected['heading']
        assert bool(collision) == expected['collision'] and bool(done) == expected['done']
        assert packet.sha256() == expected['packet_sha256'] and chain == expected['packet_chain_sha256']
        assert old.method._frame_hash(raw) == expected['frame_sha256']
        assert old.base.nonsemantic_packet_sha(packet) == expected['nonsemantic_packet_sha256']
        assert owner['owner_sha256'] == expected['owner_sha256']
        assert int((mapper.belief == 0).sum()) == expected['observed_free_cells']
        assert int((mapper.belief != -1).sum()) == expected['observed_known_cells']
        if actor_actions % 10 == 0:
            write(out / 'heartbeat.json', dict(archived_paid_replayed=actor_actions,
                 states_scored=len(payload_paths), states_total=len(locked_states), elapsed_s=time.monotonic()-started))
    if not runtime.states[0]['closed']:
        assert runtime.next_local_action(0) is None
    assert runtime.states[0]['closed']
    assert len(original_events) == len(source['local_events']) and global_index == len(source['global_events'])
    assert len(payload_paths) == len(locked_states)
    terminal_receipt = verify_measured_snapshot('terminal', source, mapper, packet, chain)
    assert digest(runtime.sensor_episode_summary(0)) == digest(read(source['directory'] / 'runtime_summary.json'))
    manifest = dict(status='completed_exact_archived_replay', layout=layout, source_commit=HEAD,
                    protected_source_protocol_sha256=sha(SOURCE / 'PROTOCOL.json'),
                    restored_original_local_decisions=len(original_events), restored_original_global_decisions=global_index,
                    scored_decision_states=len(payload_paths), replayed_archived_actions=actor_actions,
                    counterfactual_actions_executed=0, new_trajectory_generated=False,
                    heldout_data_read=False, truth_used_for_ranking=False, original_guard_verified_per_action=True,
                    same_trace_and_online_choices_as_both_archives=True,
                    source_archives={key: dict(path=str(value['directory']),
                       files_sha256={name: sha(value['directory'] / name) for name in SOURCE_FILES})
                        for key,value in archives.items()},
                    start=start_receipt, terminal=terminal_receipt,
                    state_files=[str(path.relative_to(out)) for path in payload_paths],
                    replay_freeze_sha256=sha(HERE / 'freeze.json'),
                    elapsed_s=time.monotonic()-started,
                    files_sha256={str(path.relative_to(out)): sha(path) for path in sorted(out.rglob('*')) if path.is_file()})
    write(out / 'manifest.json', manifest)
    print(json.dumps({key: manifest[key] for key in ('layout','scored_decision_states','replayed_archived_actions')}), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('layout', choices=tuple(COUNTS))
    args = parser.parse_args()
    frozen = read(HERE / 'freeze.json')
    assert all(sha(Path(path)) == value for path, value in frozen['files_sha256'].items())
    out = HERE / 'runs' / args.layout
    (out / 'states').mkdir(parents=True, exist_ok=False)
    try:
        replay(args.layout, frozen, out)
    except BaseException as error:
        (out / 'failure.json').write_text(json.dumps(dict(status='failed_archived_replay', layout=args.layout,
            error=repr(error), traceback=traceback.format_exc()), indent=2) + '\n')
        raise


if __name__ == '__main__':
    main()
