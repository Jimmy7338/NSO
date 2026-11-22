#!/usr/bin/env python3
"""Verify the published S records and extract the two predeclared intervention anchors."""

from hashlib import sha1, sha256
import importlib.util
import json
from pathlib import Path
import sys
from types import SimpleNamespace
from zipfile import ZipFile

import numpy as np


HERE = Path(__file__).resolve().parent
STUDY = HERE.parent
WEIGHTED = STUDY.parent / 'weighted_geometric_control_20261122'
BUNDLE = WEIGHTED / 'publication/payload'
INSTANCE_INDEX = STUDY.parent.parent / 'workspaces/nso_hybrid_20261029/nso_bridge/nso_bridge/instance_support.py'
INSTANCE_ADAPTER = INSTANCE_INDEX.with_name('instance_asset_adapter.py')
FRONTEND = STUDY.parent / 'semantic_scene_redesign_20261119/instance_frontend_study/run.py'
WORLD = STUDY.parent / 'paired_unseen_gain_20261120/scene/unseen_world.py'
sys.dont_write_bytecode = True
sys.path.insert(0, str(WEIGHTED / 'runner'))
from weighted_score import Weights, rank_admitted


def require(condition, message):
    if not condition:
        raise ValueError(message)


def read(path):
    return json.loads(path.read_text())


def hash_file(path):
    return sha256(path.read_bytes()).hexdigest()


def record(path):
    return {'path': str(path), 'sha256': hash_file(path)}


def hash_array(value):
    return sha256(np.ascontiguousarray(value).tobytes()).hexdigest()


def digest(value):
    return sha256(json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def only(rows, message):
    selected = list(rows)
    require(len(selected) == 1, message)
    return selected[0]


def git_blob(path):
    raw = path.read_bytes()
    return sha1(f'blob {len(raw)}\0'.encode() + raw).hexdigest()


def verify_remote(protocol):
    remote = read(HERE / 'remote_readback.json')
    require((remote['repository'], remote['commit'], remote['remote_directory']) ==
            (protocol['repository'], protocol['source_commit'], protocol['source_path']), 'Wrong source commit')
    names = {'EVIDENCE_INDEX.json', 'fit_weights.json', 'holdout_freeze.json',
             'run_evidence/scenes_prepared.zip', 'run_evidence/scenes_evaluator_references.zip'}
    names.update('run_evidence/holdout_' + case['layout'] + '_S.zip' for case in protocol['cases'])
    require(set(remote['paths']) == names, 'Wrong source readback file roster')
    for name, source in remote['paths'].items():
        require(git_blob(BUNDLE / name) == source['git_sha'], f'Local archive differs from source commit: {name}')
    return remote, {entry['archive']: entry for entry in read(BUNDLE / 'EVIDENCE_INDEX.json')['bundles']}


def verify_bundle(entry, prefix=''):
    path = BUNDLE / 'run_evidence' / entry['archive']
    require(hash_file(path) == entry['archive_sha256'], f'Published archive SHA-256 mismatch: {path}')
    origin = Path(entry['original_path'])
    with ZipFile(path) as archive:
        require(set(archive.namelist()) == set(entry['released_files']), f'ZIP members differ: {path}')
        for name, descriptor in (entry['released_files'] | entry['native_meshes_on_workstation']).items():
            if not name.startswith(prefix):
                continue
            local = (origin / name).read_bytes()
            require(len(local) == descriptor['bytes'] and sha256(local).hexdigest() == descriptor['sha256'],
                    f'Working source differs from remote index: {origin / name}')
            if name in entry['released_files']:
                require(archive.read(name) == local, f'ZIP data differs from source: {path}:{name}')
    return origin


def replay_initial_owner(path, expected):
    with np.load(path, allow_pickle=False) as arrays:
        frame = SimpleNamespace(**dict(arrays))
    owner = np.zeros(frame.semantic.shape, dtype=np.int64)
    owner[frame.semantic == 2] = 1
    owner[frame.semantic == 3] = 2
    require(not owner[frame.depth_m == 0].any() and hash_array(owner) == expected, 'Original visible owner mask differs')
    spec = importlib.util.spec_from_file_location('source_frozen_instance_support', INSTANCE_INDEX)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    index = module.InstanceSupportIndex()
    return index.observe(frame, owner, 'frame-0')


def select_facility(layout, call, source_truth, initial, observed_owner, chosen):
    candidate_audit = call['outputs']['candidate_audit']
    measured_assets = candidate_audit['measured_assets']
    selected = call['outputs']['selected']
    source_objects = source_truth['source_object_records']
    stride = int(np.prod(initial['shape'])) + 1
    entry_binding = None
    if selected['asset_index'] is not None:
        measured = measured_assets[selected['asset_index']]
        selected_entry = only((entry for entry in candidate_audit['target_audit']
                               if entry['asset_index'] == selected['asset_index'] and entry['role'] == selected['group']
                               and entry['reason'] == 'selected'), f'Missing selected asset location for {layout}')
        require(measured['rear_boundary_xy'] == selected_entry['measured_point_rear_boundary_xy'],
                f'Archived entry does not identify the measured asset: {layout}')
        track = measured['group'] // stride
        entry_binding = 'The selected global asset index points to the archived measured AABB. This AABB is contained in exactly one pre-frozen facility AABB.'
    else:
        tags = {tag[0] for row in chosen.values() for tag in row['novel_view_tags']}
        require(len(tags) == 1 and tags.intersection(chosen['category']['novel_view_tags'][0][:1]) == tags,
                f'No unique pre-fork facility track: {layout}')
        track = next(iter(tags))
        owner = only((e for e in observed_owner['instances'] if e['assigned_track'] == track),
                     f'No first-frame owner matching sole novel track: {layout}')
        measured = only((row for row in measured_assets if row['group'] // stride == track),
                        f'No measured asset matching the sole novel track: {layout}')
        entry_binding = 'The coverage goal carries no asset pointer. Both routes tag the same track, bound to the archived first-frame owner by the original association, and to exactly one pre-frozen facility AABB.'

    measured_bounds = np.array([measured['observed_low'], measured['observed_high']])
    target = only((item for item in source_objects
                   if np.all(measured_bounds[0] >= np.asarray(item['bounds_world_xyz_m'][0]) - 1e-6)
                   and np.all(measured_bounds[1] <= np.asarray(item['bounds_world_xyz_m'][1]) + 1e-6)),
                  f'Measured target does not uniquely match a pre-frozen source asset: {layout}')
    if selected['asset_index'] is None:
        require(owner['instance_index'] == target['owner'], 'First-frame ownership disagrees with AABB match')
    key = {31: 'open31', 118: 'closed118'}[target['source_instance_id']]
    require(key in source_truth['metadata'], f'No fixed evaluation reference for primary target in {layout}')
    return {'target_facility_key': key, 'source_instance_id': target['source_instance_id'],
            'source_owner': target['owner'], 'track': track, 'archive_global_group': selected['group'],
            'archive_measured_asset_idx': selected['asset_index'], 'binding': entry_binding,
            'archive_measured_aabb': measured_bounds.tolist(), 'source_truth_aabb': target['bounds_world_xyz_m'],
            'archived_target_evidence_support_points': measured['support_points'],
            'first_frame_association_recomputed': observed_owner, 'asset_archive': measured,
            'other_facilities_measured': ['open31', 'closed118', 'facility_union']}


def freeze_case(case, protocol, weights, bundles):
    layout, paid, call_id = (case[k] for k in ('layout', 'paid_before', 'selection_call_id'))
    origin = verify_bundle(bundles['holdout_' + layout + '_S.zip'])
    manifest = read(origin / 'manifest.json')
    require(manifest['layout'] == layout and manifest['study_prior'] == 'S' and manifest['global_mode'] == 'G'
            and manifest['native_budget'] == protocol['native_budget_including_return']
            and manifest['actual_closed_loop'] and not manifest['forced_route_or_goal']
            and manifest['unknown_weight'] == manifest['action_cost'] == 0
            and manifest['runner_freeze_sha256'] == hash_file(BUNDLE / 'holdout_freeze.json'), 'Unexpected S lineage')
    for name, expected in manifest['files_sha256'].items():
        require(hash_file(origin / name) == expected, f'Old S lineage artifact changed: {layout}/{name}')

    source_inputs = verify_bundle(bundles['scenes_prepared.zip'], layout + '/') / layout
    initial = read(source_inputs / 'initial_state.json')
    require(initial['initial_packet_sha256'] == manifest['initial_packet_sha256']
            and initial['initial_owner_sha256'] == manifest['initial_owner_sha256']
            and initial['seed'] == manifest['actual_scene_seed'], 'Initial S observation differs from sealed source data')
    local = only((row for row in read(origin / 'local_choices.json') if row['selection_call_id'] == call_id
                  and row['paid_before'] == paid), 'Missing declared S recorded local call')
    global_call = only((row for row in read(origin / 'global_choices.json') if row['call_id'] == call_id
                        and row['action_id'] == paid), 'Missing declared recorded neutral global call')
    audit = local['choice_audit']
    choices = audit['candidates']
    allowed = [row for row in choices if row['admitted']]
    native = only((row for row in allowed if row['route_id'] == audit['original_route_id']), 'Invalid native route')
    tolerance = audit['score_equality_tolerance']
    require(len({row['outbound_sha256'] for row in choices}) == len(choices) == audit['assessed_candidate_count']
            and len(allowed) == audit['admitted_candidate_count'] and not local['full_pool']['unassessed_after_cap'],
            'Native pool was truncated or contains duplicates')
    require(digest([{k: v for k, v in row.items() if k != 'scores'} for row in choices])
            == audit['candidate_pool_sha256'] == local['full_pool']['full_candidate_pool_sha256']
            and digest([row['scores'] for row in choices]) == audit['score_table_sha256'] == local['full_pool']['full_score_sha256'],
            'Archived pool structural or S/G score fingerprints differ')
    require(digest(global_call['inputs']) == global_call['input_sha256']
            and digest(global_call['outputs']) == global_call['output_sha256'], 'Archived neutral global call fingerprint differs')
    selected = global_call['outputs']['selected']
    require(local['same_global_target'] == audit['native_goal_pose'] == selected['pose']
            and initial['anchor'] == audit['return_anchor'] == selected['return_anchor']
            and local['native_return_actions'] == local['rebound_return_actions'] == selected['return_actions'], 'Goal or return binding differs')

    chosen = {arm: only((row for row in allowed if row['route_id'] == case[arm + '_route']),
                        f'Declared route is absent or inadmissible: {layout}/{arm}') for arm in ('category', 'geometry')}
    cell_area = initial['grid_resolution_m'] ** 2
    policy_winner, _ = rank_admitted(allowed, native, 'S', Weights(0, 0), cell_area, tolerance)
    require(chosen['category']['route_id'] == audit['selected_route_id'] == policy_winner['route_id'], 'Category route was not actually selected')
    remaining = allowed.copy()
    order = []
    while remaining:
        winner, scores = rank_admitted(remaining, native, 'G', weights, cell_area, tolerance)
        score = only((entry for entry in scores if entry['route_id'] == winner['route_id']), 'Illegal repeated rank')
        order.append({'rank': len(order) + 1, **score})
        remaining.remove(winner)
    for row in chosen.values():
        require(digest({k: row[k] for k in ('outbound_states', 'outbound_actions')}) == row['outbound_sha256']
                and row['outbound_states'][0] == local['pose']
                and row['outbound_states'][-1] == selected['pose'] == row['goal_pose']
                and len(row['outbound_states']) == row['outbound_cost'] + 1 == len(row['outbound_actions']) + 1
                and row['outbound_cost'] == case['equal_nominal_outbound_actions']
                and row['outbound_cost'] + row['reserved_return_cost'] <= local['remaining_budget']
                and row['reserved_return_cost'] == len(selected['return_actions']), 'Declared nominal path differs')
    both = list(chosen.values())
    require(both[0]['unknown_union_cells'] - both[1]['unknown_union_cells'] == case['category_minus_geometry_forecast_unknown_cells']
            and both[0]['outbound_cost'] + paid == case['nominal_paid_arrival'], 'Unexpected nominated route conditions')
    first = next(i for i, (left, right) in enumerate(zip(*(row['outbound_actions'] for row in both), strict=True)) if left != right)
    first_state = next(i for i, (left, right) in enumerate(zip(*(row['outbound_states'] for row in both), strict=True)) if left != right)
    require(first + 1 == first_state, 'Nominal actions and poses differ at different clocks')

    initial_raw = read(origin / 'initial_raw_rgbd_scan_sha256.json')
    steps = read(origin / 'steps.json')
    common_steps = steps[:paid]
    with np.load(origin / 'observed_belief_trace.npz', allow_pickle=False) as trace:
        require(np.array_equal(trace['paid'][:paid + 1], np.arange(paid + 1)), 'Prefix is incomplete')
        belief_hashes = [hash_array(grid) for grid in trace['belief'][:paid + 1]]
    prefix_packet = common_steps[-1]['packet_sha256'] if paid else initial_raw['packet_sha256']
    events = [initial_raw['packet_sha256'], *(s['packet_sha256'] for s in common_steps)]
    packet_chain = '0' * 64
    for stamp, event in enumerate(events):
        packet_chain = sha256(bytes.fromhex(packet_chain) + bytes.fromhex(event)).hexdigest()
        require(stamp == 0 or common_steps[stamp - 1]['packet_chain_sha256'] == packet_chain, 'Prefix packet chain differs')
    require(prefix_packet == local['packet_sha256'] == audit['packet_sha256']
            and belief_hashes[0] == manifest['initial_belief_sha256']
            and belief_hashes[-1] == local['observed_belief_sha256'] == audit['belief_sha256'], 'Prefix fork hashes differ')

    evaluator_origin = verify_bundle(bundles['scenes_evaluator_references.zip'], layout + '/') / layout
    truth = read(evaluator_origin / 'RECEIPT.json')
    require(hash_file(evaluator_origin / 'fixed_reference.npz') == truth['reference_npz_sha256']
            and hash_file(evaluator_origin / 'source_mesh.ply') == truth['source_mesh_sha256'], 'Evaluator truth has changed')
    observed_owner = replay_initial_owner(source_inputs / 'initial_frame.npz', initial['initial_owner_sha256'])
    facility = select_facility(layout, global_call, truth, initial, observed_owner, chosen)
    facility.update({key: record(evaluator_origin / filename) for key, filename in {
        'evaluator_reference_receipt': 'RECEIPT.json', 'evaluator_truth': 'evaluator_truth.npz',
        'visible_truth_samples': 'fixed_reference.npz', 'source_mesh': 'source_mesh.ply'}.items()})

    ranked = {entry['route_id']: entry for entry in order}
    anchor = {'layout': layout, 'source_run_dir': str(origin), 'source_inputs_dir': str(source_inputs),
              'paid_before': paid, 'selection_call_id': call_id, 'option_id': local['option_id'],
              'prefix_pose': local['pose'], 'start_pose': local['pose'], 'goal_pose': selected['pose'],
              'return_anchor': initial['anchor'], 'frame_id': local['frame_id'], 'packet_sha256': prefix_packet,
              'packet_chain_sha256': packet_chain, 'belief_sha256': belief_hashes[-1],
              'candidate_pool_sha256': audit['candidate_pool_sha256'], 'archived_S_score_table_sha256': audit['score_table_sha256'],
              'score_table_sha256': audit['score_table_sha256'], 'global_native_sha256': local['global_native_sha256'],
              'global_inputs_sha256': global_call['input_sha256'], 'global_outputs_sha256': global_call['output_sha256'],
              'reserve_actions': selected['return_actions'], 'reserve_states': selected['return_states'],
              'remaining_budget': local['remaining_budget'], 'initial_state': initial,
              'native_route_id': native['route_id'], 'nominal_paid_arrival': case['nominal_paid_arrival'],
              'nominal_first_divergence': {'paid': paid + first + 1, 'category_action': both[0]['outbound_actions'][first],
                                           'geometry_action': both[1]['outbound_actions'][first]},
              'archive_no_exact_fork_mesh': (origin / f'paid_{paid}/native_mesh.ply').exists() is False and paid != 0,
              'routes': chosen, 'allowed_route_count': len(allowed), 'assessed_route_count': len(choices),
              'G_perf_rank': {arm: ranked[row['route_id']] for arm, row in chosen.items()},
              'G_perf_winner': order[0], 'G_perf_all_legal_ranks': order, 'prefix_belief_sha256_by_paid': belief_hashes,
              'verified_actor_sources': {name: record(origin / name) for name in ('manifest.json', 'steps.json',
                 'audit.json', 'global_choices.json', 'local_choices.json', 'initial_raw_rgbd_scan_sha256.json', 'observed_belief_trace.npz')},
              'original_initial_files': {name: record(source_inputs / name) for name in ('initial_state.json', 'initial_frame.npz', 'initial_scan.npz')}}
    return anchor, facility


def write_new(path, data):
    with path.open('x') as stream:
        json.dump(data, stream, indent=2, allow_nan=False)
        stream.write('\n')


def main():
    protocol = read(STUDY / 'PROTOCOL.json')
    remote, bundles = verify_remote(protocol)
    fit = read(WEIGHTED / 'fit_weights.json')
    require(hash_file(WEIGHTED / 'fit_weights.json') == hash_file(BUNDLE / 'fit_weights.json'), 'Frozen weight file changed')
    weights = Weights(**fit['performance'])
    require(weights == Weights(protocol['continuation_policy']['unknown_weight'], protocol['continuation_policy']['action_cost']),
            'Frozen G_perf differs from the declared continuation')
    anchors, evaluator = {}, {}
    for case in protocol['cases']:
        layout = case['layout']
        anchors[layout], evaluator[layout] = freeze_case(case, protocol, weights, bundles)
    common = {'repository': remote['repository'], 'commit': remote['commit'], 'remote_directory': remote['remote_directory'],
              'remote_readback': record(HERE / 'remote_readback.json'), 'source_evidence_index': record(BUNDLE / 'EVIDENCE_INDEX.json'),
              'source_holdout_freeze': record(BUNDLE / 'holdout_freeze.json'), 'source_G_perf_weights': record(BUNDLE / 'fit_weights.json')}
    write_new(HERE / 'frozen_anchors.json', common | {'scenes': anchors})
    write_new(HERE / 'evaluator_bindings.json', common | {'scenes': evaluator, 'evaluation_only': True,
              'identity_code_sources': [record(p) for p in (INSTANCE_INDEX, INSTANCE_ADAPTER, FRONTEND, WORLD)]})
    print({layout: (entry['candidate_pool_sha256'], entry['G_perf_rank'], evaluator[layout]['target_facility_key'])
           for layout, entry in anchors.items()})


if __name__ == '__main__':
    main()
