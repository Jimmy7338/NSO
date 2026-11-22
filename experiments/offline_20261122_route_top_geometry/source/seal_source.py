#!/usr/bin/env python3
"""Read back pinned source archives and seal the single analytic fork."""

import argparse
from copy import deepcopy
import csv
from hashlib import sha1, sha256
import json
from pathlib import Path
import sys
from zipfile import ZipFile


HERE = Path(__file__).resolve().parent
ANALYSES = HERE.parents[1]
PRIOR = ANALYSES / 'route_intervention_20261122'
WEIGHTED = ANALYSES / 'weighted_geometric_control_20261122'
PRIOR_REMOTE = 'experiments/offline_20261122_route_intervention'
WEIGHTED_REMOTE = 'experiments/offline_20261122_weighted_geometric_control'
PIN = '874bff785debcad6982059530cdeda892a26e7ba'
LAYOUT = 'paired_chain_17'
PAID = 105
CALL = 428
ARM_ROUTES = {'category': ('via_32_37_0', 67, 13), 'geometry': ('via_33_29_3', 67, 12),
              'native': ('native', 63, 1)}
RAW = ('depth_values_sha256', 'rgb_values_sha256', 'scan_ranges_sha256')
ACTOR_FILES = ('manifest.json', 'steps.json', 'audit.json', 'global_choices.json', 'local_choices.json',
               'initial_raw_rgbd_scan_sha256.json', 'observed_belief_trace.npz')


def require(condition, reason):
    if not condition:
        raise ValueError(reason)


def read(path):
    return json.loads(path.read_text())


def file_sha(path):
    return sha256(path.read_bytes()).hexdigest()


def record(path):
    return {'path': str(path), 'sha256': file_sha(path)}


def digest(value):
    return sha256(json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def unique(rows):
    rows = list(rows)
    require(len(rows) == 1, 'Expected one archived call, candidate, or evidence member')
    return rows[0]


def verify_git():
    remote = read(HERE / 'remote_readback.json')
    require(remote['commit'] == PIN and remote['repository'] == 'Jimmy7338/NSO', 'Incorrect source revision')
    local = {PRIOR_REMOTE: PRIOR, WEIGHTED_REMOTE: WEIGHTED}
    for path, expected in remote['paths'].items():
        directory, name = path.split('/', 2)[1:]
        root = local['experiments/' + directory]
        workspace = root / name
        archive = root / 'publication/payload' / name
        raw = (workspace if workspace.is_file() else archive).read_bytes()
        require(sha1(f'blob {len(raw)}\0'.encode() + raw).hexdigest() == expected,
                f'Source Git blob does not match pinned commit: {path}')
    return remote


def verify_bundle(root, dirname, prefix=''):
    package = root / 'publication/payload'
    bundle = unique(v for v in read(package / 'EVIDENCE_INDEX.json')['bundles'] if v['archive'] == dirname)
    directory = Path(bundle.get('native_directory') or bundle['original_path'])
    archive = package / 'run_evidence' / dirname
    require(file_sha(archive) == bundle['archive_sha256'], f'Archive digest differs: {dirname}')
    archived = bundle.get('included_files') or bundle['released_files']
    with ZipFile(archive) as data:
        require(set(data.namelist()) == set(archived), f'Archive member roster differs: {dirname}')
        for name, descriptor in (archived | bundle['native_meshes_on_workstation']).items():
            if not name.startswith(prefix):
                continue
            contents = (directory / name).read_bytes()
            require(len(contents) == descriptor['bytes'] and sha256(contents).hexdigest() == descriptor['sha256'],
                    f'Local evidence differs from pinned manifest: {directory / name}')
            if name in archived:
                require(data.read(name) == contents, f'Archive differs from local evidence: {archive}:{name}')
    return directory, record(archive)


def verify_manifest(directory):
    manifest = read(directory / 'manifest.json')
    for name, expected in manifest['files_sha256'].items():
        require(file_sha(directory / name) == expected, f'Observed run file changed: {directory / name}')
    return manifest


def archived_arm(arm, prior_anchor, source_trace):
    directory, archive = verify_bundle(PRIOR, f'{LAYOUT}_{arm}.zip')
    manifest = verify_manifest(directory)
    end = manifest['route_end']
    expected = ARM_ROUTES[arm]
    intervention = read(directory / 'intervention.json')
    require(manifest['layout'] == LAYOUT and manifest['arm'] == arm and manifest['paid_actions'] == 192
            and manifest['runner_freeze_sha256'] == file_sha(PRIOR / 'RUN_FREEZE.json')
            and manifest['returned'] is True and manifest['collisions'] == 0
            and end['status'] == 'arrived' and end['actual_paid'] == PAID + expected[1]
            and end['target_pose'] == prior_anchor['goal_pose'] == intervention['target_pose']
            and intervention['route_id'] == expected[0] and intervention['selection_call_id'] == CALL
            and intervention['outbound_sha256'] == prior_anchor['routes'][arm]['outbound_sha256'],
            f'Reused arm is not the declared same-state intervention: {arm}')
    trace = read(directory / 'steps.json')
    compared = ('step','action','position','heading','packet_sha256','packet_chain_sha256','owner_sha256', *RAW)
    for saved, archived in zip(trace[:PAID], source_trace[:PAID], strict=True):
        require(all(saved[key] == archived[key] for key in compared), f'Archived {arm} prefix drift')
    evidence = {'directory': str(directory), 'manifest': record(directory / 'manifest.json'),
                'fork_ready': record(directory / 'fork_ready.json'), 'steps': record(directory / 'steps.json'),
                'mesh_index': {stamp: record(directory / stamp / 'native_mesh.ply')
                               for stamp in ('fork','route_end','paid_172','terminal')}, 'archive': archive}
    for stamp, mesh in evidence['mesh_index'].items():
        checkpoint = manifest['checkpoints'][stamp]
        require(checkpoint['files_sha256']['native_mesh.ply'] == mesh['sha256'], 'Mesh receipt differs')
    return manifest, evidence


def canonical_rank(anchor, chosen_call, weights):
    sys.path.insert(0, str(WEIGHTED / 'runner'))
    from weighted_score import Weights, rank_admitted

    audited = chosen_call['choice_audit']
    candidates = audited['candidates']
    legal = [v for v in candidates if v['admitted']]
    native = unique(v for v in legal if v['route_id'] == 'native')
    tolerance = audited['score_equality_tolerance']
    require(len(legal) == len(candidates) == anchor['allowed_route_count'] == 34
            and not chosen_call['full_pool']['unassessed_after_cap']
            and len({v['outbound_sha256'] for v in candidates}) == 34, 'Unexpected candidate pool')
    require(digest([{k: val for k,val in row.items() if k != 'scores'} for row in candidates])
            == anchor['candidate_pool_sha256'] == audited['candidate_pool_sha256']
            and digest([row['scores'] for row in candidates]) == anchor['archived_S_score_table_sha256'],
            'Archived pool or score digest differs')
    scale = anchor['initial_state']['grid_resolution_m'] ** 2
    provisional, scores = rank_admitted(legal, native, 'G', Weights(**weights), scale, tolerance)
    order = []
    remaining = legal.copy()
    while remaining:
        winner, numbers = rank_admitted(remaining, native, 'G', Weights(**weights), scale, tolerance)
        order.append({'rank': len(order)+1, **unique(row for row in numbers if row['route_id'] == winner['route_id'])})
        remaining.remove(winner)
    require(provisional == native and order == anchor['G_perf_all_legal_ranks'], 'Canonical actual ranking changed')
    best = max(row['score_m2'] for row in scores)
    tied = [row['route_id'] for row in scores if best - row['score_m2'] <= tolerance]
    require(len(tied) > 1 and all(
        unique(row for row in legal if row['route_id'] == identifier)['outbound_cost'] == 63
        for identifier in tied), 'Recorded top score set does not bind native cost/tie')
    anchor['routes']['native'] = native
    anchor['G_perf_rank']['native'] = order[0]
    for arm,(identifier,cost,rank) in ARM_ROUTES.items():
        route = anchor['routes'][arm]
        require(route == unique(row for row in legal if row['route_id'] == identifier)
                and anchor['G_perf_rank'][arm]['rank'] == rank
                and route['outbound_cost'] == cost and len(route['outbound_actions']) == cost
                and route['outbound_states'][0] == anchor['prefix_pose']
                and route['goal_pose'] == route['outbound_states'][-1] == anchor['goal_pose']
                and digest({key: route[key] for key in ('outbound_states','outbound_actions')}) == route['outbound_sha256']
                and anchor['remaining_budget'] >= cost + route['reserved_return_cost'] == cost + 20,
                f'Archive changed the bound route {arm}')
    return {'candidate_count': 34, 'score_tolerance': tolerance, 'score_max_m2': best,
            'tie_break_order': ['fewest outbound actions', 'native first', 'outbound route digest'],
            'tied_route_ids': tied, 'all_tied_outbound_actions': 63, 'winner': order[0], 'weights': weights}


def cold_control(source_trace, cold_dir):
    cold_trace = read(cold_dir / 'steps.json')
    with (WEIGHTED / 'decision/heldout_actual_first_differences.csv').open(newline='') as stream:
        row = unique(item for item in csv.DictReader(stream) if item['layout'] == LAYOUT
                     and item['arm_left'] == 'G_perf' and item['arm_right'] == 'S')
    fields = {'action': ('action',), 'pose': ('position','heading'), 'raw_sensor': RAW}
    divergences = {}
    for name, keys in fields.items():
        s,g = next((s,g) for s,g in zip(source_trace,cold_trace,strict=True) if any(s[k] != g[k] for k in keys))
        require(s['step'] == g['step'] == int(row[f'first_{name}_diff']) == 44, 'Cold-start divergence clock changed')
        divergences[name] = {'paid': s['step'], 'S': {key:s[key] for key in keys},
                            'cold_G_perf': {key:g[key] for key in keys}}
    require(cold_trace[PAID-1]['packet_chain_sha256'] != source_trace[PAID-1]['packet_chain_sha256'],
            'Cold G_perf would unexpectedly be a shared history')
    return {'first_actual_difference': divergences, 'eligible_as_fork_comparator': False,
            'source_difference_table': record(WEIGHTED / 'decision/heldout_actual_first_differences.csv'),
            'cold_archive_manifest': record(cold_dir / 'manifest.json'), 'cold_archive_steps': record(cold_dir / 'steps.json')}


def save(path, data, verify_only):
    encoded = json.dumps(data, indent=2, ensure_ascii=False, allow_nan=False) + '\n'
    if verify_only:
        require(path.read_text() == encoded, f'Frozen source output differs: {path}')
    else:
        with path.open('x') as stream:
            stream.write(encoded)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--verify', action='store_true', help='Compare freshly verified sources to existing frozen receipts')
    args = parser.parse_args()
    remote = verify_git()
    source_anchors = read(PRIOR / 'source/frozen_anchors.json')
    anchor = deepcopy(source_anchors['scenes'][LAYOUT])
    observed_dir, _ = verify_bundle(WEIGHTED, f'holdout_{LAYOUT}_S.zip')
    cold_dir, _ = verify_bundle(WEIGHTED, f'holdout_{LAYOUT}_G_perf.zip')
    verify_manifest(observed_dir)
    verify_manifest(cold_dir)
    trace = read(observed_dir / 'steps.json')
    prepared, _ = verify_bundle(WEIGHTED, 'scenes_prepared.zip', LAYOUT + '/')
    reference, _ = verify_bundle(WEIGHTED, 'scenes_evaluator_references.zip', LAYOUT + '/')
    require(Path(anchor['source_run_dir']) == observed_dir and Path(anchor['source_inputs_dir']) == prepared / LAYOUT
            and anchor['paid_before'] == PAID and anchor['selection_call_id'] == CALL, 'Incorrect historical S fork')
    original_call = unique(item for item in read(observed_dir / 'local_choices.json')
                           if item['selection_call_id'] == CALL and item['paid_before'] == PAID)
    weights = read(WEIGHTED / 'fit_weights.json')['performance']
    require(weights == {'unknown_weight': .25, 'action_cost': .20}, 'Fitted geometry weights changed')
    ranking = canonical_rank(anchor, original_call, weights)

    _, category = archived_arm('category', anchor, trace)
    _, geometry = archived_arm('geometry', anchor, trace)
    category_fork = Path(category['fork_ready']['path'])
    geometry_fork = Path(geometry['fork_ready']['path'])
    fork = read(category_fork)
    identity = fork['identity']
    require(category_fork.read_bytes() == geometry_fork.read_bytes()
            and digest(identity) == fork['fork_identity_sha256']
            and identity['actual_paid'] == PAID and identity['selection_call_id'] == CALL
            and identity['target_pose'] == anchor['goal_pose'] and identity['return_anchor'] == anchor['return_anchor']
            and identity['pose'] == anchor['prefix_pose'] == [*trace[PAID-1]['position'],trace[PAID-1]['heading']]
            and identity['seed'] == anchor['initial_state']['seed']
            and identity['packet_chain_sha256'] == anchor['packet_chain_sha256'] == trace[PAID-1]['packet_chain_sha256']
            and identity['candidate_pool_sha256'] == anchor['candidate_pool_sha256']
            and identity['source_score_table_sha256'] == anchor['archived_S_score_table_sha256']
            and identity['belief_sha256'] == anchor['belief_sha256']
            and identity['mesh_sha256'] == category['mesh_index']['fork']['sha256'] == geometry['mesh_index']['fork']['sha256'],
            'Prior armed fork identity differs')

    actor_paths = [category_fork, geometry_fork, PRIOR / 'runner/run.py', PRIOR / 'source/frozen_anchors.json',
                   WEIGHTED / 'runner/weighted_score.py', WEIGHTED / 'fit_weights.json']
    actor_paths.extend(observed_dir / name for name in ACTOR_FILES)
    actor_paths.extend(prepared / LAYOUT / name for name in ('initial_state.json','initial_frame.npz','initial_scan.npz'))
    for stamp in ('start','paid_48','paid_96'):
        actor_paths.extend(observed_dir / stamp / key for key in ('checkpoint.json','belief.npz','native_mesh.ply'))
    actor = {'source_commit': PIN, 'layout': LAYOUT, 'anchor': anchor, 'ranking': ranking,
             'prior_common_fork': {'record': record(category_fork), 'identity': fork},
             'actor_files_sha256': {str(path):file_sha(path) for path in actor_paths}}
    actor_anchor = HERE / 'actor_anchor.json'
    save(actor_anchor, actor, args.verify)

    target = read(PRIOR / 'source/evaluator_bindings.json')['scenes'][LAYOUT]
    require(target['target_facility_key'] == 'open31'
            and target['visible_truth_samples'] == record(reference / LAYOUT / 'fixed_reference.npz'),
            'Original source shelf target or reference differs')
    measured = read(PRIOR / 'evaluation/results/measured.json')['cases']['paired_cycle_20']
    require(measured['G_perf_ranks']['geometry']['rank'] == 1 and measured['primary_target'] == 'closed118',
            'Cycle20 previously published competitor is not rank one')
    evaluator = {'source_commit': PIN, 'layout': LAYOUT, 'target': target, 'prior_category': category,
                 'prior_equal_cost_geometry': geometry, 'actor_anchor': record(actor_anchor),
                 'source_git_readback': record(HERE / 'remote_readback.json'),
                 'cold_start_control': cold_control(trace, cold_dir),
                 'cycle20_published_evidence': {'geometry_rank': measured['G_perf_ranks']['geometry'],
                                                'measured': record(PRIOR / 'evaluation/results/measured.json'),
                                                'publication': remote['commit']},
                 'evaluator_files_sha256': {str(path):file_sha(path) for path in (
                     PRIOR / 'source/evaluator_bindings.json', PRIOR / 'evaluation/MEASUREMENT_PROTOCOL.json',
                     PRIOR / 'evaluation/results/measured.json', PRIOR / 'evaluation/results/paired_primary.csv',
                     *(reference / LAYOUT / name for name in ('fixed_reference.npz','evaluator_truth.npz','source_mesh.ply','RECEIPT.json')))}}
    save(HERE / 'evaluator_bindings.json', evaluator, args.verify)
    print(f'{LAYOUT}: fork {fork["fork_identity_sha256"]}, rank 1 native tied with {len(ranking["tied_route_ids"])-1} alternatives; '
          'two archived arms and cold-start divergence verified')


if __name__ == '__main__':
    main()
