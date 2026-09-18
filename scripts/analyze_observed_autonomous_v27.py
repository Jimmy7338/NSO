#!/usr/bin/env python3
"""Read-only analysis of the frozen V27 feedback-effect autonomous batch."""
import os
os.environ['PYTHONDONTWRITEBYTECODE'] = '1'
import argparse
from collections import Counter
import csv
import gzip
import hashlib
import io
import json
import math
from pathlib import Path
import shutil
import sys
import zipfile

sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT/'audit_results/observed_autonomous_v27_20260916'
OLD = ROOT/'audit_results/observed_autonomous_v26_20260916'
REFERENCE = ROOT/'audit_results/observed_v27_geometry_reference_20260916'
OUTPUT = ROOT/'audit_results/observed_autonomous_v27_analysis_20260916'
PROTOCOL = ROOT/'docs/research/V27_AUTONOMOUS_ANALYSIS_PROTOCOL_20260916.md'
CASE_SPEC = ((0, 'A_complex_B_simple', 'S_no_feedback'),
             (1, 'A_complex_B_simple', 'S'),
             (2, 'A_simple_B_complex', 'S_no_feedback'),
             (3, 'A_simple_B_complex', 'S'))
CAP = 2*1024*1024
RESERVE = 64*1024*1024


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def read_gzip(path):
    with gzip.open(path, 'rt') as stream:
        return json.load(stream)


def near(actual, expected, label):
    require(math.isclose(actual, expected, rel_tol=1e-10, abs_tol=1e-10), label)


def verify_inventory(folder):
    inventory = read(folder/'artifact_hashes.json')
    actual = {str(path.relative_to(folder)) for path in folder.rglob('*')
              if path.is_file() and path.name != 'artifact_hashes.json'}
    require(actual == set(inventory), 'artifact set differs: '+str(folder))
    for name, expected in inventory.items():
        require(sha(folder/name) == expected, 'artifact hash differs: '+str(folder/name))
    return inventory


def metric(result):
    observed = result['final_main_observed']
    require(observed['contract'] == 'facility-three-projection-outline-v23-1', 'metric contract')
    C = result['final_coverage_2d']
    Q = observed['05cm']['outline_macro_quality']
    J = observed['05cm']['joint_outline']
    near(C, observed['coverage_2d'], 'coverage mismatch')
    near(J, C*Q, 'J arithmetic mismatch')
    per_instance = []
    for instance in observed['instances']:
        values = [min(row['05cm']['f1'], row['iou']) for row in instance['projections'].values()]
        near(instance['05cm']['outline_quality'], sum(values)/3, 'instance Q mismatch')
        if instance['missing']:
            near(instance['05cm']['outline_quality'], 0., 'missing instance is not zero')
        per_instance.append(dict(id=instance['id'], missing=instance['missing'],
                                 Q=instance['05cm']['outline_quality']))
    near(Q, sum(row['Q'] for row in per_instance)/2, 'macro Q mismatch')
    return C, Q, J, per_instance


def feedback_summary(calls, mode):
    events = [row for row in calls if row['module'] == 'IGCR'
              and row['operation'] == 'measured_transition_feedback']
    updates = [update for event in events for update in event['directional_updates']]
    counts = Counter(update['status'] for update in updates)
    table = {}
    for update in updates:
        if update['status'] == 'updated':
            key = f"{update['cue_id']}:{update['sector']}"
            table[key] = dict(count=update['feedback_count_after'],
                factor=update['feedback_factor_after'])
    if mode == 'S_no_feedback':
        require(not table and counts.get('updated', 0) == 0,
                'disabled feedback mode wrote a correction table')
    return dict(transition_events=len(events), endpoint_arrivals=sum(
        bool(event['selected_endpoint_reached']) for event in events),
        directional_attempts=len(updates), status_counts=dict(counts), final_table=table,
        saturated_updates=sum(update.get('observed_yield_proxy') == 1. for update in updates),
        comparable_patch_sum=sum(update.get('comparable_patches', 0) for update in updates),
        improved_patch_sum=sum(update.get('improved_common_patches', 0) for update in updates))


def analyze_case(source, declared):
    index, assignment, mode = declared
    folder = source/f'case_{index:02d}'
    inventory = verify_inventory(folder)
    result = read(folder/'result.json')
    verify = read(folder/'verification.json')
    require((result['index'], result['assignment'], result['mode']) == declared, 'case identity')
    require(result['autonomous_policy'] and result['imposed_prefix_actions'] == 0
            and not result['gt_service_catalogue_used'], 'policy boundary')
    trace = result['trace']
    require(len(trace) == result['paid_actions']+1 == result['raw_frames'], 'packet/action count')
    require([row['action_id'] for row in trace] == list(range(len(trace))), 'action sequence')
    require(trace[0]['action'] is None and trace[-1]['next_action'] is None, 'terminal trace')
    C, Q, J, instances = metric(result)
    eligible = (result['paid_actions'] <= 400 and C >= .8 and result['returned_to_anchor']
                and result['collisions'] == 0 and not result['final_main_observed']['failed'])
    require(eligible == result['eligible'], 'eligibility changed')
    replay = (verify['status'] == 'passed' and verify['independent_process']
        and verify['fresh_world'] and verify['fresh_autonomous_runtime']
        and not verify['saved_actions_used_to_drive_policy']
        and verify['saved_packets_verified'] == len(trace)
        and verify['independent_paid_actions'] == result['paid_actions']
        and verify['all_decisions_equal'] and verify['all_packets_equal']
        and verify['all_meshes_and_metrics_equal'])
    require(replay, 'independent replay receipt')
    calls = read_gzip(folder/result['calls_file'])
    plans = read_gzip(folder/result['plans_file'])
    require(len(calls) == result['summary']['module_calls'], 'module call count')
    require(len(plans) == result['summary']['global_plans'], 'plan count')
    feedback = feedback_summary(calls, mode)
    return dict(index=index, assignment=assignment, mode=mode, paid_actions=result['paid_actions'],
        C=C, Q=Q, J=J, eligible=eligible, replay_verified=replay,
        returned=result['returned_to_anchor'], collisions=result['collisions'],
        missing=result['final_main_observed']['missing_asset_count'], per_instance=instances,
        global_plans=len(plans), module_calls=len(calls), feedback=feedback,
        result_sha256=sha(folder/'result.json'), inventory_sha256=sha(folder/'artifact_hashes.json'),
        artifact_count=len(inventory), trace=trace)


def delta(left, right):
    return dict(delta_C=right['C']-left['C'], delta_Q=right['Q']-left['Q'],
        delta_J=right['J']-left['J'], relative_delta_J=(right['J']-left['J'])/left['J'])


def first_difference(left, right):
    common = min(len(left['trace']), len(right['trace']))
    command = next((i for i in range(common)
                    if left['trace'][i]['next_action'] != right['trace'][i]['next_action']), None)
    paid = next((i for i in range(1, common)
                 if left['trace'][i]['action'] != right['trace'][i]['action']), None)
    if paid is None and len(left['trace']) != len(right['trace']):
        paid = common
    history = next((i for i in range(common) if any(left['trace'][i][key] != right['trace'][i][key]
        for key in ('action', 'pose', 'nonsemantic', 'geometry_sha256'))), None)
    return dict(first_next_action_difference=command, first_paid_action_difference=paid,
        first_observation_history_difference=history,
        same_history_through_first_command=(command is not None and (history is None or history > command)))


def old_geometry_rows():
    reference = read(REFERENCE/'result.json')
    require(reference['status'] == 'complete_v26_v27_G_policy_equivalence'
            and reference['all_cases_equal'], 'G equivalence prerequisite')
    verify_inventory(REFERENCE)
    verify_inventory(OLD)
    rows = []
    for index in (0, 2):
        folder = OLD/f'case_{index:02d}'
        verify_inventory(folder)
        result = read(folder/'result.json')
        C, Q, J, instances = metric(result)
        rows.append(dict(index=index, assignment=result['assignment'], mode='G', C=C, Q=Q, J=J,
            paid_actions=result['paid_actions'], eligible=result['eligible'], per_instance=instances,
            result_sha256=sha(folder/'result.json')))
    return rows, reference


def save_bytes(output, name, payload):
    used = sum(path.stat().st_size for path in output.rglob('*') if path.is_file())
    require(used+len(payload)+4096 <= CAP-32768, 'analysis cap')
    require(shutil.disk_usage(output).free-len(payload)-4096 >= RESERVE, 'analysis reserve')
    temporary = output/(name+'.tmp')
    with temporary.open('xb') as stream:
        stream.write(payload)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, output/name)


def save_json(output, name, value):
    save_bytes(output, name, (json.dumps(value, ensure_ascii=False, indent=2,
        allow_nan=False)+'\n').encode())


def save_csv(output, name, rows):
    flat = []
    for row in rows:
        flat.append({key: json.dumps(value, ensure_ascii=False, separators=(',', ':'))
                     if isinstance(value, (dict, list)) else value
                     for key, value in row.items() if key != 'trace'})
    fields = sorted({key for row in flat for key in row})
    stream = io.StringIO()
    writer = csv.DictWriter(stream, fieldnames=fields)
    writer.writeheader(); writer.writerows(flat)
    save_bytes(output, name, stream.getvalue().encode())


def execute(source, output):
    require(not output.exists(), 'fresh analysis directory required')
    require(shutil.disk_usage(ROOT).free >= RESERVE+CAP, 'analysis capacity')
    root_inventory = verify_inventory(source)
    manifest = read(source/'manifest.json')
    require(manifest['status'] == 'complete', 'complete batch required')
    require([(row['index'], row['assignment'], row['mode']) for row in manifest['cases']]
            == list(CASE_SPEC), 'case matrix changed')
    require(manifest['config']['budget'] == 400 and manifest['config']['noise_seed'] == 1901
            and manifest['quota_start'] == 8 and manifest['main_attempts_started'] == 4,
            'frozen condition changed')
    for mapping in ('source_sha256', 'input_sha256'):
        for name, expected in manifest[mapping].items():
            require(sha(ROOT/name) == expected, mapping+' changed: '+name)
    own_name = str(Path(__file__).resolve().relative_to(ROOT))
    require(manifest['source_sha256'].get(own_name) == sha(Path(__file__)),
            'analyzer was not frozen before acquisition')
    require(manifest['source_sha256'].get(str(PROTOCOL.relative_to(ROOT))) == sha(PROTOCOL),
            'analysis protocol was not frozen before acquisition')
    with zipfile.ZipFile(source/'sources.zip') as archive:
        require(set(archive.namelist()) == set(manifest['source_sha256']), 'source archive members')
        for name, expected in manifest['source_sha256'].items():
            require(hashlib.sha256(archive.read(name)).hexdigest() == expected, 'archived source: '+name)
    output.mkdir()
    endpoints = [analyze_case(source, declared) for declared in CASE_SPEC]
    feedback_pairs = []
    for off, on in ((0, 1), (2, 3)):
        require(endpoints[off]['eligible'] and endpoints[on]['eligible'], 'ineligible feedback pair')
        feedback_pairs.append(dict(assignment=endpoints[off]['assignment'], disabled_index=off,
            enabled_index=on, **delta(endpoints[off], endpoints[on]),
            history=first_difference(endpoints[off], endpoints[on]), qualified=True))
    geometry, reference = old_geometry_rows()
    full_pairs = []
    for old, new_index in zip(geometry, (1, 3)):
        new = endpoints[new_index]
        require(old['assignment'] == new['assignment'] and old['eligible'] and new['eligible'],
                'ineligible full-method pair')
        full_pairs.append(dict(assignment=old['assignment'], G_index=old['index'],
            S_index=new_index, **delta(old, new), qualified=True))
    mean_feedback = sum(row['delta_J'] for row in feedback_pairs)/2
    mean_G = sum(row['J'] for row in geometry)/2
    mean_S = sum(endpoints[index]['J'] for index in (1, 3))/2
    mean_full_relative = (mean_S-mean_G)/mean_G
    feedback_gate = all(row['delta_J'] > 0 for row in feedback_pairs) and mean_feedback > 0
    full_gate = all(row['delta_J'] > 0 for row in full_pairs) and mean_full_relative > .05
    result = dict(status='complete_saved_evidence_v27_effect_analysis', endpoints=[
        {key: value for key, value in row.items() if key != 'trace'} for row in endpoints],
        feedback_pairs=feedback_pairs, historical_G=geometry, full_method_pairs=full_pairs,
        means=dict(feedback_delta_J=mean_feedback, G_J=mean_G, S_J=mean_S,
                   full_relative_delta_J=mean_full_relative),
        frozen_decision=dict(feedback_autonomous_benefit_passed=feedback_gate,
            full_semantic_advantage_over_5pct_passed=full_gate,
            stop_current_P00_expansion=not (feedback_gate and full_gate)),
        execution_cost=dict(new_main_attempts=4, cumulative_main_attempts=12,
            main_paid_actions=sum(row['paid_actions'] for row in endpoints),
            successful_replay_paid_actions=sum(row['paid_actions'] for row in endpoints),
            geometry_reference_offline_mapper_integrations=reference['total_offline_mapper_integrations']),
        scope=['One parent scene and one noise seed; no confidence interval or independent generalization claim.',
               'Independent replays verify determinism and evidence integrity; they are not new environments.',
               'S_no_feedback retains semantics and isolates only the feedback-table writeback.',
               'Q is the frozen exterior-outline proxy, not comprehensive 3D reconstruction accuracy.'],
        new_worlds_during_analysis=0, new_actions_during_analysis=0,
        mapper_updates_during_analysis=0, mesh_extractions_during_analysis=0,
        Q_evaluations_during_analysis=0)
    save_json(output, 'result.json', result)
    save_csv(output, 'endpoints.csv', endpoints)
    save_csv(output, 'feedback_pairs.csv', feedback_pairs)
    save_csv(output, 'full_method_pairs.csv', full_pairs)
    inputs = {str((source/'artifact_hashes.json').relative_to(ROOT)): sha(source/'artifact_hashes.json'),
              str((REFERENCE/'artifact_hashes.json').relative_to(ROOT)): sha(REFERENCE/'artifact_hashes.json'),
              str((OLD/'artifact_hashes.json').relative_to(ROOT)): sha(OLD/'artifact_hashes.json')}
    save_json(output, 'input_sha256.json', inputs)
    save_json(output, 'manifest.json', dict(status='complete', source=str(source),
        source_sha256={own_name: sha(Path(__file__)), str(PROTOCOL.relative_to(ROOT)): sha(PROTOCOL)},
        input_sha256=inputs, acquisition_root_artifacts=len(root_inventory)))
    inventory = {str(path.relative_to(output)): sha(path) for path in output.rglob('*')
                 if path.is_file() and path.name != 'artifact_hashes.json'}
    save_json(output, 'artifact_hashes.json', inventory)
    print(json.dumps(dict(status=result['status'], feedback_gate=feedback_gate,
        full_gate=full_gate, output=str(output))), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', action='store_true')
    parser.add_argument('--source', type=Path, default=SOURCE)
    parser.add_argument('--output', type=Path, default=OUTPUT)
    args = parser.parse_args()
    if not args.run:
        parser.error('explicit --run required after acquisition is complete')
    execute(args.source.resolve(), args.output.resolve())


if __name__ == '__main__':
    main()
