#!/usr/bin/env python3
"""Bounded, read-only P00 endpoint decomposition; no simulator imports.

Only cases 00..03 and reference 2026 are interpreted. The running batch
manifest is snapshotted, not treated as evidence of whole-batch completion.
"""
import argparse
import csv
import hashlib
import io
import json
import math
import os
from pathlib import Path
import zipfile

ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / 'eval_results/facility_v22_early_coverage_20260915'
ASSIGNMENTS = ('A_complex_B_simple', 'A_simple_B_complex')


def digest(data):
    return hashlib.sha256(data).hexdigest()


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def require(condition, message):
    if not condition:
        raise AssertionError(message)


def close(a, b, label):
    require(math.isfinite(a) and math.isfinite(b) and abs(a-b) <= 1e-12, label)


def atomic(path, data):
    temporary = path.with_name(path.name + '.tmp')
    temporary.write_bytes(data)
    os.replace(temporary, path)


def json_bytes(value):
    return (json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False) + '\n').encode()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, default=SOURCE)
    parser.add_argument('--output', type=Path, default=Path(__file__).resolve().parent)
    args = parser.parse_args()
    source, output = args.source.resolve(), args.output.resolve()
    require(not (output / 'result.json').exists(), 'Existing diagnosis is immutable; choose a new output')
    require(not output.is_relative_to(source), 'Output must be separate from physical evidence')
    manifest_bytes = (source / 'manifest.json').read_bytes()
    manifest = json.loads(manifest_bytes)
    require(manifest['version'] == 'facility-early-coverage-v22-information-screen-1', 'Unexpected batch')
    inputs = {str((source/'manifest.json').relative_to(ROOT)): digest(manifest_bytes),
              str((source/'sources.zip').relative_to(ROOT)): sha(source/'sources.zip')}
    frozen = manifest['source_sha256']
    with zipfile.ZipFile(source/'sources.zip') as archive:
        for name, expected in frozen.items():
            require(sha(ROOT/name) == expected, 'Current frozen source mismatch: '+name)
            require(digest(archive.read(name)) == expected, 'Archived frozen source mismatch: '+name)
    references = {}
    for assignment in ASSIGNMENTS:
        name = f'references/D19-P00_{assignment}_2026.npz'
        path = Path(manifest['reference_source_root'])/name
        expected = manifest['reference_sha256'][name]
        require(sha(path) == expected, 'Reference changed: '+name)
        references[str(path.relative_to(ROOT))] = expected
    inputs.update(references)
    cases, raw = [], []
    for index in range(4):
        directory = source/f'case_{index:02d}'
        inventory_bytes = (directory/'artifact_hashes.json').read_bytes()
        inventory = json.loads(inventory_bytes)
        inputs[str((directory/'artifact_hashes.json').relative_to(ROOT))] = digest(inventory_bytes)
        values = {}
        for name in ('result.json', 'verification.json', 'timing.json'):
            data = (directory/name).read_bytes()
            require(digest(data) == inventory[name], 'Case inventory mismatch: '+str(directory/name))
            inputs[str((directory/name).relative_to(ROOT))] = digest(data)
            values[name] = json.loads(data)
        result, verification, timing = [values[k] for k in ('result.json', 'verification.json', 'timing.json')]
        case = result['case']
        require(case == manifest['cases'][index], 'Case differs from frozen manifest')
        require(case['index'] == index and case['parent'] == 'D19-P00', 'Only P00 cases 00..03 allowed')
        require(case['assignment'] == ASSIGNMENTS[index//2], 'Unexpected assignment order')
        require(case['first_option'] == ('N', 'A')[index % 2] and result['first_option'] == case['first_option'], 'Unexpected option')
        require(case['v22_first_choice']['option_map'] == {'N': 0, 'A': 2, 'B': 0}, 'P00 N=B alias changed')
        require(verification['status'] == 'passed' and verification['independent_process']
                and verification['fresh_world_sensor_action_metric_replay'], 'Independent replay incomplete')
        require(verification['physical_process_id'] == timing['process_id']
                and verification['replay_process_id'] != timing['process_id'], 'Replay process identity')
        require(verification['actions'] == result['paid_actions'] == len(result['actions']), 'Action count mismatch')
        require(result['primitive_budget_compliant'] and result['paid_actions'] <= case['budget'] == 160, 'Budget mismatch')
        endpoint = result['after']['2026']
        require(endpoint['eligible'] and endpoint['returned'] and not endpoint['failed']
                and endpoint['collisions'] == 0 and endpoint['coverage_2d'] >= .8, 'P00 endpoint not qualified')
        require(result['initial_option_outcome']['measured_arrived']
                and not result['initial_option_outcome']['explicit_cancel_or_denial'], 'Initial target did not arrive')
        instances = sorted(endpoint['instances'], key=lambda x: x['id'])
        require([x['id'] for x in instances] == list(range(6)), 'Fixed six facilities required')
        C = endpoint['coverage_2d']; Q = endpoint['05cm']['external_macro_f1']; J = endpoint['05cm']['joint_external']
        close(Q, sum(x['05cm']['f1'] for x in instances)/6, 'Macro F1 mismatch')
        close(J, C*Q, 'Joint reward mismatch')
        for row in instances:
            p, r, f = [row['05cm'][k] for k in ('precision', 'recall', 'f1')]
            close(f, 2*p*r/(p+r) if p+r else 0, 'Instance F1 mismatch')
            require(not row['missing'] or (f == 0 and row['predicted_samples'] == 0), 'Missing must score zero')
        cases.append(dict(index=index, parent=case['parent'], assignment=case['assignment'], first_option=case['first_option'],
            C=C, Q=Q, J=J, eligible=True, actions=result['paid_actions'], budget=case['budget'],
            initial_arrival_action_ids=result['initial_option_outcome']['arrival_action_ids'],
            missing_asset_ids=[x['id'] for x in instances if x['missing']],
            instances=[dict(id=x['id'], name=chr(65+x['id']), missing=x['missing'], **x['05cm'],
                weighted_J=C*x['05cm']['f1']/6) for x in instances],
            replay=verification, trajectory_sha256=digest(json_bytes([{'action':x['action'],'pose':x['pose']} for x in result['actions']]))))
        raw.append(result)
    rows, pairs = [], []
    for ni, ai in ((0, 1), (2, 3)):
        n, a = cases[ni], cases[ai]
        current = []
        for xn, xa in zip(n['instances'], a['instances']):
            item = dict(assignment=n['assignment'], asset_id=xn['id'], asset=xa['name'],
                group='key_A' if xn['id'] == 0 else 'key_B' if xn['id'] == 1 else 'background',
                C_N=n['C'], C_A=a['C'], F1_N=xn['f1'], F1_A=xa['f1'], delta_F1=xa['f1']-xn['f1'],
                precision_N=xn['precision'], precision_A=xa['precision'], recall_N=xn['recall'], recall_A=xa['recall'],
                missing_N=xn['missing'], missing_A=xa['missing'], J_i_N=xn['weighted_J'], J_i_A=xa['weighted_J'],
                delta_J_i=xa['weighted_J']-xn['weighted_J'],
                quality_component=a['C']*(xa['f1']-xn['f1'])/6,
                coverage_component=(a['C']-n['C'])*xn['f1']/6)
            close(item['delta_J_i'], item['quality_component']+item['coverage_component'], 'Per-asset decomposition mismatch')
            rows.append(item); current.append(item)
        total = sum(x['delta_J_i'] for x in current)
        close(total, a['J']-n['J'], 'Per-asset sum does not match actual delta J')
        pairs.append(dict(assignment=n['assignment'], N_case=ni, A_case=ai, delta_C=a['C']-n['C'],
            delta_Q=a['Q']-n['Q'], delta_J=a['J']-n['J'], delta_J_relative_to_N=(a['J']-n['J'])/n['J'],
            key_A_delta_J=current[0]['delta_J_i'], key_B_delta_J=current[1]['delta_J_i'],
            key_AB_delta_J=sum(x['delta_J_i'] for x in current[:2]), background_delta_J=sum(x['delta_J_i'] for x in current[2:]),
            quality_component=sum(x['quality_component'] for x in current), coverage_component=sum(x['coverage_component'] for x in current),
            decomposition_residual=total-(a['J']-n['J']), first_action_divergence=next((i+1 for i,(x,y) in enumerate(zip(raw[ni]['actions'],raw[ai]['actions'])) if (x['action'],x['pose']) != (y['action'],y['pose'])), None)))
    swaps = []
    for i0, i1 in ((0,2), (1,3)):
        c0, c1 = cases[i0], cases[i1]
        background = [dict(id=i, name=chr(65+i), delta_F1=c1['instances'][i]['f1']-c0['instances'][i]['f1'],
            delta_weighted_J=c1['instances'][i]['weighted_J']-c0['instances'][i]['weighted_J'],
            missing_before=c0['instances'][i]['missing'], missing_after=c1['instances'][i]['missing']) for i in range(2,6)]
        swaps.append(dict(first_option=c0['first_option'], before_case=i0, after_case=i1,
            actions_and_poses_identical=c0['trajectory_sha256'] == c1['trajectory_sha256'],
            background_F1_exactly_invariant=all(x['delta_F1'] == 0 for x in background),
            background_weighted_J_exactly_invariant=all(x['delta_weighted_J'] == 0 for x in background),
            delta_C=c1['C']-c0['C'], background=background))
    fixed_n = (cases[0]['J']+cases[2]['J'])/2
    fixed_a = (cases[1]['J']+cases[3]['J'])/2
    fixed_best = max(fixed_n, fixed_a)
    conditional = (max(cases[0]['J'],cases[1]['J'])+max(cases[2]['J'],cases[3]['J']))/2
    result = dict(status='complete_bounded_P00_diagnosis', entire_V22_batch_complete=False,
        source_batch_status_at_snapshot=manifest['status'], interpreted_case_indices=[0,1,2,3], excluded_parent='D19-P01',
        reference_seed=2026, threshold_m=.05, task_asset_count=6,
        formula='delta_J_i=(C_A*F1_A_i-C_N*F1_N_i)/6',
        secondary_decomposition='C_A*(F1_A_i-F1_N_i)/6 + (C_A-C_N)*F1_N_i/6; algebraic allocation, not causal estimate',
        cases=cases, pairs=pairs, per_asset=rows, class_swap_background_checks=swaps,
        P00_only_information=dict(canonical_options=['N=B','A'], both_assignments_feasible=True,
            best_fixed_value=fixed_best, conditional_value=conditional, VOI=conditional-fixed_best,
            relative_VOI=(conditional-fixed_best)/fixed_best, full_two_parent_gate_evaluated=False),
        key_AB_interaction=pairs[0]['key_AB_delta_J']-pairs[1]['key_AB_delta_J'],
        background_interaction=pairs[0]['background_delta_J']-pairs[1]['background_delta_J'],
        full_interaction=pairs[0]['delta_J']-pairs[1]['delta_J'],
        claims=dict(single_asset_response_observed=True, semantic_information_gain_proven=False,
            initial_view_count_causal_effect_identified=False, full_architecture_proven=False,
            new_physics_or_reconstruction_performed=False, background_strict_invariance_proven=False),
        provenance=dict(input_sha256=inputs, frozen_source_sha256=frozen, all_frozen_sources_match_archive_and_workspace=True,
            script_sha256=sha(Path(__file__)), raw_packets_replayed_by_this_diagnosis=False,
            raw_packet_inventory_retained_by_hash=True, independent_replay_receipts_validated=True,
            reference_arrays_not_loaded=True, no_case_after_03_result_read=True))
    output.mkdir(parents=True, exist_ok=True)
    atomic(output/'manifest_snapshot.json', manifest_bytes)
    atomic(output/'result.json', json_bytes(result))
    atomic(output/'input_sha256.json', json_bytes(inputs))
    atomic(output/'source_sha256.json', json_bytes({**frozen, str(Path(__file__).resolve().relative_to(ROOT)): sha(Path(__file__))}))
    csv_io = io.StringIO(); writer = csv.DictWriter(csv_io, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    atomic(output/'per_asset.csv', csv_io.getvalue().encode())
    print(json.dumps(dict(status=result['status'], delta_J=[x['delta_J'] for x in pairs], P00_VOI=conditional-fixed_best,
        residuals=[x['decomposition_residual'] for x in pairs], output=str(output)), indent=2))


if __name__ == '__main__':
    main()
