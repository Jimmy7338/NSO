#!/usr/bin/env python3
"""Frozen four-option V25r1 analysis: saved records only, no world or mapping."""
import os
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['PYTHONDONTWRITEBYTECODE'] = '1'
import argparse
import ast
from collections import deque
import csv
import hashlib
import io
import json
import math
from pathlib import Path
import shutil
import sys
import traceback
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT/'audit_results/facility_choice_v25r1_paid_p00_20260915'
OUTPUT = ROOT/'audit_results/facility_choice_v25r1_fixed_analysis_20260915'
PROTOCOL = ROOT/'docs/research/V25_P00_FIXED_OPTION_ANALYSIS_PROTOCOL_20260915.md'
PROTOCOL_SHA = 'b759c63252bf1f51a81f4721795ece6adb7c4df3e50a9d91a0f63bacd53d7b96'
CASE_SPEC = [('A_complex_B_simple', 'A'), ('A_complex_B_simple', 'B'),
             ('A_simple_B_complex', 'A'), ('A_simple_B_complex', 'B')]
PAIR_SPEC = ((0, 1), (2, 3), (0, 2), (1, 3))
PREFIX_KEYS = ('action_id', 'action', 'pose', 'nonsemantic', 'belief_sha256',
               'visible_sha256', 'marker_components')
RESERVE = 64*1024**2
OUTPUT_CAP = 2*1024**2


def require(condition, message):
    if not condition:
        raise ValueError(message)


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'),
                                     allow_nan=False).encode()).hexdigest()


def array_hash(array):
    array = np.ascontiguousarray(array)
    return hashlib.sha256(f'{array.dtype.str}:{array.shape}:'.encode()+array.tobytes()).hexdigest()


def near(a, b, label):
    require(isinstance(a, (int, float)) and not isinstance(a, bool) and math.isfinite(a), label)
    require(math.isclose(a, b, rel_tol=1e-10, abs_tol=1e-10), label+' arithmetic mismatch')


def verify_inventory(folder):
    inventory = read(folder/'artifact_hashes.json')
    actual = {str(p.relative_to(folder)) for p in folder.rglob('*')
              if p.is_file() and p != folder/'artifact_hashes.json'}
    require(set(inventory) == actual, 'Inventory file set mismatch: '+str(folder))
    for name, expected in inventory.items():
        require(sha(folder/name) == expected, 'Artifact changed: '+str(folder/name))
    return len(inventory)


def source_receipt(source, allow_stopped):
    manifest = read(source/'manifest.json')
    complete = manifest['status'] == 'complete'
    stopped = manifest['status'].startswith('stopped_')
    require(complete or (allow_stopped and stopped), 'Only a complete batch or explicitly requested stopped batch may be analyzed')
    require(not any(c[k] == 'running' for c in manifest['cases']
                    for k in ('physical_status', 'replay_status')), 'A physical/replay process is still active')
    require(sha(PROTOCOL) == PROTOCOL_SHA, 'Predeclared analysis protocol changed')
    rel = str(PROTOCOL.relative_to(ROOT))
    require(manifest['source_sha256'].get(rel) == PROTOCOL_SHA, 'Protocol was not frozen before collection')
    for name, expected in manifest['source_sha256'].items():
        require(sha(ROOT/name) == expected, 'Frozen collection source changed: '+name)
    for name, expected in manifest['input_sha256'].items():
        require(sha(ROOT/name) == expected, 'Frozen collection input changed: '+name)
    require(sha(source/'sources.zip') == manifest['source_archive_sha256'], 'Source archive changed')
    require(len(manifest['cases']) == 4 and [(c['assignment'], c['arm']) for c in manifest['cases']] == CASE_SPEC,
            'Four declared cases/order differ')
    require(manifest['config']['total_budget'] == 400 and manifest['config']['paid_prefix_actions'] == 234,
            'P00 budget/prefix changed')
    count = verify_inventory(source) if complete else None
    return manifest, dict(input_manifest_sha256=sha(source/'manifest.json'),
        input_root_inventory_sha256=sha(source/'artifact_hashes.json') if complete else None,
        root_inventory_verified=complete, verified_root_artifact_count=count,
        frozen_collection_sources=len(manifest['source_sha256']), frozen_collection_inputs=len(manifest['input_sha256']),
        protocol_sha256=PROTOCOL_SHA, complete_batch=complete,
        source_archive_sha256=manifest['source_archive_sha256'])


def subset_equal(original, enhanced):
    if isinstance(original, dict):
        return isinstance(enhanced, dict) and all(k in enhanced and subset_equal(v, enhanced[k]) for k, v in original.items())
    if isinstance(original, list):
        return isinstance(enhanced, list) and len(original) == len(enhanced) and all(subset_equal(a, b) for a, b in zip(original, enhanced))
    return original == enhanced


def audited_metric(metric, expected_coverage, expected_eligible, reference):
    require(metric['reference_signature'] == reference and metric['mission_asset_count'] == 2,
            'Metric reference or asset count mismatch')
    require([row['id'] for row in metric['instances']] == [0, 1], 'All two fixed assets must be scored')
    near(metric['coverage_2d'], expected_coverage, 'coverage')
    require(metric['eligible'] == expected_eligible, 'Qualification disagrees with paid execution')
    for tag in ('02cm', '05cm', '10cm'):
        values, boundary = [], []
        for item in metric['instances']:
            require(set(item['projections']) == {'xy', 'xz', 'yz'}, 'Three orthogonal projections required')
            quality = []
            for p in item['projections'].values():
                f1, iou = p[tag]['f1'], p['iou']
                require(0 <= f1 <= 1 and 0 <= iou <= 1, 'Projection quality out of range')
                quality.append(min(f1, iou))
            q = sum(quality)/3
            f = sum(p[tag]['f1'] for p in item['projections'].values())/3
            near(item[tag]['outline_quality'], q, 'instance quality')
            near(item[tag]['outline_f1'], f, 'instance boundary F1')
            if item['missing']:
                near(q, 0., 'Missing asset must score zero')
            values.append(q); boundary.append(f)
        q = sum(values)/2
        near(metric[tag]['outline_macro_quality'], q, 'macro quality')
        near(metric[tag]['outline_macro_f1'], sum(boundary)/2, 'macro boundary F1')
        near(metric[tag]['joint_outline'], expected_coverage*q, 'J=CQ')
    require(metric['missing_asset_count'] == sum(x['missing'] for x in metric['instances']), 'Missing asset count mismatch')
    return dict(C=expected_coverage, Q=metric['05cm']['outline_macro_quality'],
                J=metric['05cm']['joint_outline'], F1=metric['05cm']['outline_macro_f1'],
                completion_fraction=metric['completion_fraction'], eligible=metric['eligible'])


def marker_colors():
    """Read the frozen physical color table, without importing any world."""
    tree = ast.parse((ROOT/'env/virtual3d_inspection_v4.py').read_text())
    expr = next(n.value for n in tree.body if isinstance(n, ast.Assign)
                and any(isinstance(t, ast.Name) and t.id == 'MARKER_COLORS' for t in n.targets))
    require(isinstance(expr, ast.Dict), 'Unexpected frozen marker color syntax')
    colors = {ast.literal_eval(k): ast.literal_eval(v.args[0]) for k, v in zip(expr.keys, expr.values)}
    require(set(colors) == {2, 3}, 'Expected simple=2 / complex=3 marker codes')
    return colors


def seeded_component(mask, start):
    queue = deque([tuple(start)]); visited = {tuple(start)}
    while queue:
        r, c = queue.popleft()
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                q = (r+dr, c+dc)
                if (0 <= q[0] < mask.shape[0] and 0 <= q[1] < mask.shape[1]
                        and mask[q] and q not in visited):
                    visited.add(q); queue.append(q)
    return visited


def observed_class_cue(folder, case, config, colors):
    """Choose from actual prefix colors and observed left/right association.

    Hidden assignment/reference IDs do not select an arm. The evaluator's
    seed-window association is checked separately and cannot correct a seed.
    """
    if 'prefix' not in case['stages']:
        return dict(valid=False, reasons=['prefix unavailable'], complex_rule_arm=None, swap_rule_arm=None)
    prefix = case['stages']['prefix']; reasons, checked = [], []
    tracks = prefix['observed_track_summary']
    if len(tracks) != 2:
        reasons.append('not exactly two observed prefix tracks')
    ordered = sorted(tracks, key=lambda t: t['centre'][0])
    if len(ordered) == 2 and not ordered[0]['centre'][0] < ordered[1]['centre'][0]:
        reasons.append('observed left/right centers ambiguous')
    slot_roles = {t['observed_track_index']: 'AB'[i] for i, t in enumerate(ordered[:2])}
    for item in prefix['instances']:
        slot, seed = item['observed_slot'], item['seed']
        if seed is None:
            reasons.append('missing actual seed for slot '+str(slot)); continue
        action_id = seed['action_id']
        if not 0 <= action_id <= 234 or action_id >= len(case['trace']):
            reasons.append('seed outside available prefix'); continue
        path = folder/'packets'/f'{action_id:04d}.npz'
        with np.load(path, allow_pickle=False) as packet:
            metadata = json.loads(str(packet['metadata'].item()))
            rgb, depth = packet['frame__color_rgb'], packet['frame__depth_m']
            intrinsic, pose = packet['frame__intrinsic'], packet['frame__world_from_camera']
            pixel = tuple(seed['pixel']); z = float(depth[pixel])
            actual_codes = [code for code, color in colors.items() if np.array_equal(rgb[pixel], color)]
            binary = np.zeros(depth.shape, bool)
            for color in colors.values():
                binary |= np.all(rgb == color, axis=-1)
            binary &= np.isfinite(depth) & (depth > .15)
            supported = bool(binary[pixel])
            component = seeded_component(binary, pixel) if supported else set()
            local = np.asarray([(pixel[1]-intrinsic[0, 2])*z/intrinsic[0, 0],
                                (pixel[0]-intrinsic[1, 2])*z/intrinsic[1, 1], z])
            measured_seed = local@pose[:3, :3].T+pose[:3, 3]
            actual_code = actual_codes[0] if len(actual_codes) == 1 else None
            consistent = (metadata['action_id'] == action_id and supported and actual_code == seed['actual_marker_code']
                and len(component) == seed['component_pixels'] and len(component) >= 16
                and np.allclose(measured_seed, seed['observed_seed_xyz'], atol=1e-9, rtol=0)
                and seed['packet_sha256'] == case['trace'][action_id]['packet_sha256']
                and sha(path) == case['trace'][action_id]['raw_file_sha256'])
        if not consistent:
            reasons.append('actual saved RGB/depth seed receipt mismatch for slot '+str(slot))
        track = next((t for t in tracks if t['observed_track_index'] == slot), None)
        support = (track is not None and track['distinct_paid_poses'] >= config['minimum_supported_distinct_poses_per_track']
                   and track['max_valid_pixels'] >= 16)
        if not support:
            reasons.append('insufficient distinct paid prefix marker support for slot '+str(slot))
        checked.append(dict(observed_slot=slot, role_from_observed_x_order=slot_roles.get(slot),
            seed_action=action_id, seed_pixel=seed['pixel'], actual_color_code=actual_code,
            actual_rgb=colors.get(actual_code), component_pixels=len(component), raw_file_sha256=sha(path),
            actual_depth_seed_checked=consistent, distinct_paid_pose_support=support))
    if not prefix['evaluation_association_and_minimum_separation_passed']:
        reasons.append('prefix fixed association/minimum separation gate failed')
    for row in prefix['fixed_seed_association']['rows']:
        role = slot_roles.get(row['observed_slot'])
        if role is None or row['status'] != 'unique' or row['reference_id'] != 'AB'.index(role):
            reasons.append('observed left/right role disagrees with fixed evaluation association')
    by_role = {x['role_from_observed_x_order']: x['actual_color_code'] for x in checked}
    if set(by_role) != {'A', 'B'} or sorted(by_role.values(), key=lambda v: -1 if v is None else v) != [2, 3]:
        reasons.append('prefix does not supply exactly one simple and one complex role')
    valid = not reasons
    complex_arm = next((arm for arm, code in by_role.items() if code == 3), None) if valid else None
    swap_arm = next((arm for arm, code in by_role.items() if code == 2), None) if valid else None
    return dict(valid=valid, reasons=reasons, checked_seeds=checked, observed_codes_by_role=by_role,
        complex_rule_arm=complex_arm, swap_rule_arm=swap_arm,
        choice_used_hidden_assignment=False, choice_used_GT_reference_id=False,
        GT_association_used_only_as_validation=True, natural_semantic_network_tested=False)


def analyze_case(folder, declaration, manifest, colors):
    count = verify_inventory(folder)
    case, verify = read(folder/'result.json'), read(folder/'verification.json')
    require(case['status'] == 'complete' and verify['status'] == 'passed', 'Completed case and independent replay required')
    timing = read(folder/'timing.json')
    require(verify['physical_process_id'] == timing['physical_process_id']
            and verify['physical_process_id'] != verify['replay_process_id'], 'Replay PID is not independent')
    for flag in ('independent_process', 'all_per_step_metadata_equal', 'all_stage_geometry_arrays_equal',
                 'all_raw_and_observed_mesh_arrays_equal', 'all_frozen_metrics_equal'):
        require(verify[flag] is True, 'Missing independent verification: '+flag)
    require((case['assignment'], case['arm']) == (declaration['assignment'], declaration['arm']), 'Declared case identity mismatch')
    route = manifest['route_catalog'][case['arm']]
    require(case['route_sha256'] == declaration['route_sha256'] == digest(route), 'Case used a different route')
    require(case['inference_disabled'] and not case['autonomous_planner'] and not case['semantic_policy_executed'], 'Representation/policy scope changed')
    trace = case['trace']; paid = case['paid_actions']
    require(len(trace) == paid+1 == case['raw_frames'], 'Paid action/frame count mismatch')
    require(verify['actual_physical_replay_actions'] == paid and verify['saved_packets_verified'] == paid+1, 'Replay count mismatch')
    require([x['action_id'] for x in trace] == list(range(paid+1)), 'Noncontinuous paid history')
    require({p.name for p in (folder/'packets').iterdir()} == {f'{i:04d}.npz' for i in range(paid+1)}, 'Raw packet sequence differs')
    require(digest([(x['action_id'], x['action'], x['pose']) for x in trace]) == case['trajectory_sha256'], 'Trajectory digest mismatch')
    for i, row in enumerate(trace):
        expected_action = None if i == 0 else route['actions'][i-1]
        expected_phase = 'initial' if i == 0 else route['frame_schedule'][i-1]['phase']
        require(row['action'] == expected_action and row['phase'] == expected_phase, 'Paid action/phase differs from catalog')
        require(row['expected_state_matches'] == (row['pose'] == route['states'][i]), 'State-match receipt inconsistent')
        near(row['coverage_2d'], row['known_reachable_cells']/row['reachable_cells'], 'actual C')
    returned = trace[-1]['pose'] == manifest['anchor']
    require(case['returned_to_anchor'] == returned and case['remaining_budget'] == 400-paid, 'Terminal budget/return mismatch')
    eligible = paid <= 400 and case['collisions'] == 0 and returned and not case['execution_failed'] and trace[-1]['coverage_2d'] >= .8
    require(case['eligible'] == eligible, 'Case qualification inconsistent')
    stage_metrics = {}
    for name, stage in case['stages'].items():
        snapshot = read(folder/(name+'_snapshot.json'))
        require(subset_equal(snapshot, stage), 'Pre-GT representation receipt changed after evaluation')
        row = trace[stage['action_id']]
        failed = case['execution_failed'] if name == 'final' else False
        stage_eligible = (stage['action_id'] <= 400 and row['cumulative_collisions'] == 0
                          and row['returned_to_anchor'] and not failed and row['coverage_2d'] >= .8)
        stage_metrics[name] = dict(main=audited_metric(stage['main_observed'], row['coverage_2d'], stage_eligible, case['reference_signature']),
                                  raw=audited_metric(stage['raw_secondary'], row['coverage_2d'], stage_eligible, case['reference_signature']))
        for item in stage['instances']:
            with np.load(folder/item['observed_mesh_file'], allow_pickle=False) as mesh:
                actual = {key: array_hash(mesh[key]) for key in ('vertices', 'triangles')}
            require(actual == item['observed_mesh_sha256'], 'Saved measured geometry hash mismatch')
    with np.load(folder/'final_mesh.npz', allow_pickle=False) as mesh:
        actual = {key: array_hash(mesh[key]) for key in ('vertices', 'triangles', 'vertex_colors')}
    require(actual == case['final_mesh_sha256'], 'Saved final raw geometry mismatch')
    require(case['final_main_observed'] == case['stages']['final']['main_observed'], 'Main terminal alias mismatch')
    require(case['main_quality_05cm'] == case['final_main_observed']['05cm'], 'Main Q alias mismatch')
    for i, frame in enumerate(route['arrival_frame_indices']):
        expected = frame < len(trace) and trace[frame]['pose'] == route['selected_view_states'][i]
        require(case['paid_view_arrival_receipts'][i]['matches_declared_view'] == expected, 'Paid view receipt inconsistent')
    cue = observed_class_cue(folder, case, manifest['config'], colors)
    summary = dict(index=case['index'], assignment=case['assignment'], arm=case['arm'],
        paid_actions=paid, remaining_budget=400-paid, C=stage_metrics['final']['main']['C'],
        Q=stage_metrics['final']['main']['Q'], J=stage_metrics['final']['main']['J'],
        raw_Q=stage_metrics['final']['raw']['Q'], raw_J=stage_metrics['final']['raw']['J'],
        eligible=eligible, returned=returned, collisions=case['collisions'], execution_failed=case['execution_failed'],
        termination=case['termination'], fatal_for_unstarted_tasks=case['fatal_for_unstarted_tasks'],
        measured_only=True, prefix=stage_metrics.get('prefix'), final=stage_metrics['final'],
        all_declared_views_reached=case['all_declared_acquisition_views_physically_reached'],
        arrival_receipts=case['paid_view_arrival_receipts'], observed_class_cue=cue,
        verified_case_artifacts=count, physical_pid=verify['physical_process_id'], replay_pid=verify['replay_process_id'],
        prefix_association_passed=case['stages'].get('prefix', {}).get('evaluation_association_and_minimum_separation_passed', False),
        final_association_passed=case['stages']['final']['evaluation_association_and_minimum_separation_passed'])
    assets = []
    for identifier in (0, 1):
        final = case['stages']['final']['main_observed']['instances'][identifier]
        initial = case['stages'].get('prefix', {}).get('main_observed', {}).get('instances', [None, None])[identifier]
        q0 = None if initial is None else initial['05cm']['outline_quality']
        assets.append(dict(case_index=case['index'], assignment=case['assignment'], arm=case['arm'], asset='AB'[identifier],
            inspected_by_fixed_arm=case['arm'] == 'AB'[identifier], prefix_Q=q0,
            final_Q=final['05cm']['outline_quality'], delta_Q=None if q0 is None else final['05cm']['outline_quality']-q0,
            prefix_boundary_F1=None if initial is None else initial['05cm']['outline_f1'],
            final_boundary_F1=final['05cm']['outline_f1'], prefix_completed=None if initial is None else initial['completed'],
            final_completed=final['completed'], missing=final['missing'],
            final_size_errors_xyz_m=final['dimensions']['absolute_size_error_xyz_m'],
            final_center_error_inf_m=final['dimensions']['center_error_inf_m'],
            final_projection_iou={k: p['iou'] for k, p in final['projections'].items()},
            final_projection_hausdorff_upper_m={k: p['hausdorff_upper_m'] for k, p in final['projections'].items()}))
    return case, summary, assets


def mean_policy(rows, label):
    return dict(label=label, selected_case_indices=[r['index'] for r in rows],
                arms=[r['arm'] for r in rows], all_eligible=all(r['eligible'] for r in rows),
                **{'mean_'+key: sum(r[key] for r in rows)/len(rows) for key in ('C', 'Q', 'J', 'paid_actions')})


def comparison(rows, cues_valid):
    by = {(r['assignment'], r['arm']): r for r in rows}
    assignments = [CASE_SPEC[0][0], CASE_SPEC[2][0]]
    fixed = {arm: mean_policy([by[(x, arm)] for x in assignments], 'fixed_'+arm) for arm in ('A', 'B')}
    best_arm = min(('A', 'B'), key=lambda a: (-fixed[a]['mean_J'], a))
    best = fixed[best_arm]; denominator = best['mean_J']
    oracle_rows = [min([by[(x, a)] for a in ('A', 'B')], key=lambda r: (-r['J'], r['arm'])) for x in assignments]
    oracle = mean_policy(oracle_rows, 'two_tested_options_posthoc_oracle')
    def contrasted(policy):
        policy['absolute_J_vs_best_fixed'] = policy['mean_J']-denominator
        policy['relative_J_vs_best_fixed'] = None if denominator <= 0 else (policy['mean_J']-denominator)/denominator
        return policy
    semantic = swap = None
    if cues_valid:
        for field, label in (('complex_rule_arm', 'predeclared_observed_complex_rule'), ('swap_rule_arm', 'predeclared_swapped_rule')):
            chosen = [by[(x, by[(x, 'A')]['observed_class_cue'][field])] for x in assignments]
            value = contrasted(mean_policy(chosen, label))
            if field == 'complex_rule_arm': semantic = value
            else: swap = value
    gaps = [dict(assignment=x, delta_C_A_minus_B=by[(x, 'A')]['C']-by[(x, 'B')]['C'],
                 delta_Q_A_minus_B=by[(x, 'A')]['Q']-by[(x, 'B')]['Q'],
                 delta_J_A_minus_B=by[(x, 'A')]['J']-by[(x, 'B')]['J'],
                 delta_paid_actions_A_minus_B=by[(x, 'A')]['paid_actions']-by[(x, 'B')]['paid_actions']) for x in assignments]
    return dict(fixed_options=fixed, best_fixed_arm=best_arm, best_fixed=best,
                two_option_oracle=contrasted(oracle), observed_complex_rule=semantic, swapped_rule=swap,
                assignment_arm_gaps=gaps, interaction_J=gaps[0]['delta_J_A_minus_B']-gaps[1]['delta_J_A_minus_B'],
                means_of_J_are_not_products_of_mean_C_and_Q=True,
                oracle_is_not_an_executed_or_learned_policy=True, no_confidence_interval_from_two_assignments=True)


def actual_nonsemantic_divergence(left, right, budget=400, prefix_actions=234):
    """Exact saved-hash timing on the two actually collected same-arm paths.

    This is a diagnostic after collection. It supplies no recognition rule and
    does not substitute the ideal clean-ray witness for an actual observation.
    """
    a, b = left['trace'], right['trace']
    trajectory_a = [(r['action_id'], r['action'], r['pose']) for r in a]
    trajectory_b = [(r['action_id'], r['action'], r['pose']) for r in b]
    same = trajectory_a == trajectory_b
    result = dict(left=left['index'], right=right['index'], arm=left['arm'],
        complete_action_pose_sequences_identical=same,
        left_trajectory_sha256=digest(trajectory_a), right_trajectory_sha256=digest(trajectory_b),
        left_saved_frames=len(a), right_saved_frames=len(b), prefix_paid_actions=prefix_actions,
        total_budget=budget, first_divergence=None, first_divergence_by_field=None,
        no_nonsemantic_difference_through_end=None,
        field_scope='saved trace.nonsemantic only; class colors and GT coverage excluded',
        interpretation='exact hash difference for one paired noisy sample on this fixed acquisition path; '
            'not a learned/executable classifier, clean-ray distance, or adaptive-policy success')
    require(left['arm'] == right['arm'], 'Nonsemantic timing requires the same fixed arm')
    if not same:
        result['available'] = False
        result['reason'] = 'Full actual action/pose sequences differ; no paired-path timing claim'
        return result
    result['available'] = True
    first_by_field = {}
    for i, (x, y) in enumerate(zip(a, b)):
        require(x['action_id'] == y['action_id'] == i, 'Timing diagnostic requires continuous frame/action IDs')
        require(set(x['nonsemantic']) == set(y['nonsemantic']), 'Paired nonsemantic field schema differs')
        changed = sorted(k for k in x['nonsemantic'] if x['nonsemantic'][k] != y['nonsemantic'][k])
        for field in changed:
            first_by_field.setdefault(field, i)
        if changed and result['first_divergence'] is None:
            result['first_divergence'] = dict(frame_index=i, paid_step=i, action=x['action'], pose=x['pose'],
                phase=x['phase'], steps_after_prefix=i-prefix_actions, remaining_budget=budget-i,
                changed_fields=changed,
                left_field_hashes={k: x['nonsemantic'][k] for k in changed},
                right_field_hashes={k: y['nonsemantic'][k] for k in changed})
    result['first_divergence_by_field'] = {k: first_by_field.get(k) for k in sorted(a[0]['nonsemantic'])} if a else {}
    result['no_nonsemantic_difference_through_end'] = result['first_divergence'] is None
    result['last_compared_paid_step'] = len(a)-1
    return result


def analyze(source, manifest, provenance, allow_stopped):
    if manifest['status'] != 'complete':
        available = []
        for declaration in manifest['cases']:
            folder = source/f"case_{declaration['index']:02d}"
            item = dict(index=declaration['index'], assignment=declaration['assignment'], arm=declaration['arm'],
                physical_status=declaration['physical_status'], replay_status=declaration['replay_status'],
                result_available=(folder/'result.json').exists(),
                failure_receipts=[read(p) for p in sorted(folder.glob('failure_*.json'))] if folder.exists() else [])
            if (folder/'artifact_hashes.json').exists():
                item['verified_available_case_artifacts'] = verify_inventory(folder)
            if item['result_available']:
                saved = read(folder/'result.json')
                item.update(paid_actions=saved['paid_actions'], remaining_budget=saved['remaining_budget'],
                    C=saved['final_coverage_2d'], Q=saved['main_quality_05cm']['outline_macro_quality'],
                    J=saved['main_quality_05cm']['joint_outline'], eligible=saved['eligible'],
                    returned=saved['returned_to_anchor'], collisions=saved['collisions'],
                    execution_failed=saved['execution_failed'], termination=saved['termination'],
                    supplied_terminal_scores_not_an_information_comparison=True)
            available.append(item)
        return dict(status='stopped_batch_no_information_comparison', provenance=provenance, cases=available,
                    comparisons=None, semantic_efficacy_proven=False, reason='Four complete verified tasks are unavailable'), []
    raw, rows, assets = [], [], []
    colors = marker_colors()
    for declaration in manifest['cases']:
        require(declaration['physical_status'] == declaration['replay_status'] == 'complete', 'All four physical/replay cases must be complete')
        case, summary, individual = analyze_case(source/f"case_{declaration['index']:02d}", declaration, manifest, colors)
        raw.append(case); rows.append(summary); assets += individual
    pids = [p for row in rows for p in (row['physical_pid'], row['replay_pid'])]
    require(len(set(pids)) == 8, 'Eight distinct physical/replay processes required')
    aggregate = read(source/'result.json'); pairs = []
    for left, right in PAIR_SPEC:
        a = [{key: row[key] for key in PREFIX_KEYS} for row in raw[left]['trace'][:235]]
        b = [{key: row[key] for key in PREFIX_KEYS} for row in raw[right]['trace'][:235]]
        equal = len(a) == len(b) == 235 and a == b
        prior = next(p for p in aggregate['prefix_checks'] if p['left'] == left and p['right'] == right)
        require(prior['full_nonsemantic_prefix_identical'] == equal and prior['left_sha256'] == digest(a)
                and prior['right_sha256'] == digest(b), 'Aggregate prefix comparison disagrees')
        pairs.append(dict(left=left, right=right, passed=equal, left_sha256=digest(a), right_sha256=digest(b)))
    paired = all(p['passed'] for p in pairs)
    cue_consistent = all(rows[i]['observed_class_cue']['valid'] for i in range(4))
    for i, j in ((0, 1), (2, 3)):
        cue_consistent &= rows[i]['observed_class_cue'].get('observed_codes_by_role') == rows[j]['observed_class_cue'].get('observed_codes_by_role')
    arithmetic = comparison(rows, cue_consistent)
    all_eligible = all(r['eligible'] for r in rows)
    association = all(r['prefix_association_passed'] and r['final_association_passed'] for r in rows)
    qualified = all_eligible and paired and association
    semantic_valid = qualified and cue_consistent
    semantic = arithmetic['observed_complex_rule']; oracle = arithmetic['two_option_oracle']
    require(aggregate['physical_paid_actions'] == sum(r['paid_actions'] for r in rows), 'Aggregate action total mismatch')
    require(aggregate['replay_paid_actions'] == aggregate['physical_paid_actions'], 'Aggregate replay actions mismatch')
    require(aggregate['saved_raw_packets'] == sum(r['paid_actions']+1 for r in rows), 'Aggregate raw frame count mismatch')
    divergence = [actual_nonsemantic_divergence(raw[left], raw[right]) for left, right in ((0, 2), (1, 3))]
    return dict(status='complete_saved_record_analysis', provenance=provenance, cases=rows, per_asset_changes=assets,
        paired_prefix_checks=pairs, all_cases_eligible=all_eligible, all_instance_association_gates_passed=association,
        actual_same_arm_nonsemantic_divergence=divergence,
        all_observed_prefix_class_cues_valid_and_consistent=cue_consistent,
        raw_all_four_case_arithmetic=arithmetic,
        qualified_two_option_information=arithmetic if qualified else None,
        observed_rule_valid_for_qualified_comparison=semantic_valid,
        single_parent_oracle_strict_over_five_percent=bool(qualified and oracle['relative_J_vs_best_fixed'] is not None and oracle['relative_J_vs_best_fixed'] > .05),
        single_parent_observed_rule_strict_over_five_percent=bool(semantic_valid and semantic is not None and semantic['relative_J_vs_best_fixed'] is not None and semantic['relative_J_vs_best_fixed'] > .05),
        two_parent_development_goal_passed=False, semantic_efficacy_proven=False,
        policy_scope='one parent, two fixed scripted options, two paired assignments, one noise seed; no adaptive G or full ANS test',
        new_physical_actions=0, new_sensor_frames=0, new_tsdf_fusions=0, new_evaluator_or_backend_calls=0,
        main_tasks=4, independent_replays=4, physical_paid_actions=aggregate['physical_paid_actions'],
        replay_paid_actions=aggregate['replay_paid_actions'], saved_raw_packets=aggregate['saved_raw_packets'],
        unequal_actual_arm_costs_are_reported=True, static_oracle_information_is_not_strong_geometry_advantage=True), assets


def save(output, name, data):
    data = data.encode() if isinstance(data, str) else data
    used = sum(p.stat().st_size for p in output.rglob('*') if p.is_file())
    require(used+len(data)+4096 < OUTPUT_CAP, '2 MiB analysis cap exceeded')
    require(shutil.disk_usage(output).free-len(data)-8192 >= RESERVE, '64 MiB free reserve would be crossed')
    path = output/name; temporary = path.with_name(path.name+'.tmp')
    with temporary.open('xb') as stream:
        stream.write(data)
    os.replace(temporary, path)


def csv_text(rows, columns):
    buffer = io.StringIO(); writer = csv.DictWriter(buffer, fieldnames=columns, extrasaction='ignore')
    writer.writeheader(); writer.writerows(rows)
    return buffer.getvalue()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, default=SOURCE)
    parser.add_argument('--output', type=Path, default=OUTPUT)
    parser.add_argument('--analyze-stopped', action='store_true')
    args = parser.parse_args(); source, output = args.source.resolve(), args.output.resolve()
    require(not output.exists(), 'Existing independent analysis must not be overwritten')
    manifest, provenance = source_receipt(source, args.analyze_stopped)
    # All completed-case checks and cue choices occur only after the batch is
    # complete. A stopped batch gets no partial oracle/rule selection.
    result, assets = analyze(source, manifest, provenance, args.analyze_stopped)
    require(sha(source/'manifest.json') == provenance['input_manifest_sha256'], 'Input batch changed during analysis')
    for name, expected in manifest['source_sha256'].items():
        require(sha(ROOT/name) == expected, 'Source changed during analysis: '+name)
    require(shutil.disk_usage(output.parent).free >= RESERVE+1024**2, 'Analysis output capacity unavailable')
    output.mkdir(parents=True)
    try:
        save(output, 'result.json', json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False)+'\n')
        save(output, 'case_table.csv', csv_text(result['cases'], ['index', 'assignment', 'arm', 'paid_actions',
            'remaining_budget', 'C', 'Q', 'J', 'raw_Q', 'raw_J', 'eligible', 'returned', 'collisions', 'execution_failed', 'termination']))
        save(output, 'asset_changes.csv', csv_text(assets, ['case_index', 'assignment', 'arm', 'asset',
            'inspected_by_fixed_arm', 'prefix_Q', 'final_Q', 'delta_Q', 'prefix_boundary_F1', 'final_boundary_F1',
            'prefix_completed', 'final_completed', 'missing']))
        summary = '仅分析已保存数据；新动作、传感、TSDF 和评价调用均为 0。\n'
        if result['status'] == 'complete_saved_record_analysis':
            c = result['raw_all_four_case_arithmetic']; s = c['observed_complex_rule']
            summary += ('四条资格全部通过：'+str(result['all_cases_eligible'])+'；非语义前缀全部配对：'
                +str(all(p['passed'] for p in result['paired_prefix_checks']))+'。\n'
                +'最佳固定臂 '+c['best_fixed_arm']+'，平均 J='+str(c['best_fixed']['mean_J'])+'；两选项事后上限平均 J='
                +str(c['two_option_oracle']['mean_J'])+'，相对差='+str(c['two_option_oracle']['relative_J_vs_best_fixed'])+'。\n'
                +'实际前缀类别规则平均 J='+str(None if s is None else s['mean_J'])+'，相对最佳固定差='
                +str(None if s is None else s['relative_J_vs_best_fixed'])+'；合格规则比较有效：'
                +str(result['observed_rule_valid_for_qualified_comparison'])+'。\n')
            for item in result['actual_same_arm_nonsemantic_divergence']:
                event = item['first_divergence']
                summary += ('同臂 '+item['arm']+' 两排列完整动作/位姿相同：'
                    +str(item['complete_action_pose_sequences_identical'])+'；实际非类别记录首差付费步='
                    +str(None if event is None else event['paid_step'])+'，相对前缀末步234='
                    +str(None if event is None else event['steps_after_prefix'])+'，当时剩余预算='
                    +str(None if event is None else event['remaining_budget'])+'，字段='
                    +str(None if event is None else event['changed_fields'])+'。\n')
            summary += '实际首差只是单个配对噪声样本的哈希差异，不能据此声称已形成可执行识别器。\n'
        else:
            summary += '批次停止/不完整，不计算选臂规则或信息上限，不丢弃失败回执。\n'
        summary += '只有一个父布局和固定采集目录；这些结果不证明强几何对照、自适应语义策略、自然语义网络或完整 ANS 优势。\n'
        save(output, 'facts_zh.txt', summary)
        save(output, 'analysis_source.py', Path(__file__).read_bytes())
        save(output, 'provenance.json', json.dumps(dict(provenance, analysis_script_sha256=sha(Path(__file__)),
            input_root=str(source), analysis_protocol=str(PROTOCOL)), indent=2)+'\n')
        inventory = {p.name: sha(p) for p in sorted(output.iterdir()) if p.is_file()}
        save(output, 'artifact_hashes.json', json.dumps(inventory, indent=2)+'\n')
    except Exception as error:
        save(output, 'failure.json', json.dumps(dict(error=repr(error), traceback=traceback.format_exc()), indent=2)+'\n')
        raise
    print(json.dumps(dict(status=result['status'], output=str(output), result_sha256=sha(output/'result.json'))))


if __name__ == '__main__':
    main()
