#!/usr/bin/env python3
"""Independently inspect the sealed V36 confirmation without executing a policy.

Reuses the frozen V35 saved-packet verifier, independently rebuilds nonsemantic
hashes, per-parent equal-weight statistics, and the preregistered confirmation
gate. No World, renderer, mapper, surface evaluator, or DP is constructed.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import traceback

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.analyze_semantic_chain_v35 import (
    close, first_divergence, inspect_case, read, require, sha, verify_seal)

BATCH = ROOT / 'audit_results/v36_online_confirmation_20260918'
OUTPUT = ROOT / 'audit_results/v36_online_confirmation_review_20260918'
PROTOCOL = ROOT / 'docs/research/V36_ONLINE_CONFIRMATION_PROTOCOL_20260918.md'
PLAN = ROOT / 'docs/research/V36_FROZEN_CONFIRMATION_PLAN_20260918.md'
PARENTS = ('P00', 'P01')
MODES = ('G', 'S')
THRESHOLDS = ('02cm', '05cm', '10cm')


def trace_array_hash(value):
    """The trace uses V13 array identities; posterior has its own encoding."""
    import numpy as np
    value = np.ascontiguousarray(value)
    return hashlib.sha256(f'{value.dtype.str}:{value.shape}:'.encode() + value.tobytes()).hexdigest()


def confirm_packet_fields(folder, details, parent_config):
    """Derive the nonsemantic comparison fields from the raw saved packets."""
    import numpy as np
    checked = 0
    recognized_positions = set()
    for observed, history in zip(details['trace'], details['controller']['history']):
        paid = observed['paid']
        with np.load(folder / 'packets' / f'{paid:03d}.npz', allow_pickle=False) as packet:
            rgb = packet['frame__color_rgb'].copy()
            counts = {2: int(np.all(rgb == (40, 100, 220), axis=-1).sum()),
                      3: int(np.all(rgb == (220, 60, 40), axis=-1).sum())}
            for color in ((40, 100, 220), (220, 60, 40)):
                rgb[np.all(rgb == color, axis=-1)] = (127, 127, 127)
            metadata = json.loads(str(packet['metadata']))
            fields = {
                'depth': packet['frame__depth_m'], 'rgb_nonsemantic': rgb,
                'intrinsic': packet['frame__intrinsic'],
                'camera_pose': packet['frame__world_from_camera'],
                'ranges': packet['scan__ranges_m'],
                'laser_pose': packet['scan__world_from_laser'],
                'scan_calibration': np.asarray([packet['scan__angle_min_rad'].item(),
                    packet['scan__angle_increment_rad'].item(), packet['scan__range_max_m'].item()]),
                'cell': np.asarray(metadata['position'], dtype=np.int64),
                'heading': np.asarray(metadata['heading'], dtype=np.int64)}
            require({key: trace_array_hash(value) for key, value in fields.items()} == observed['nonsemantic'],
                    'trace nonsemantic hashes differ from actual paid packet')
            xy = packet['frame__world_from_camera'][:2, 3] - np.asarray(parent_config['translation'][:2])
            require(np.allclose(xy, np.rint(xy), atol=1e-9, rtol=0.), 'observed pose is off the public grid')
            require([int(round(xy[0])), int(round(xy[1])), metadata['heading']] == observed['pose_v33'],
                    'controller pose differs from observed camera odometry')
            require(metadata['action_id'] == paid and metadata['action'] == observed['action'],
                    'saved paid packet action metadata differs from trace')
            supported = [code for code, count in counts.items() if count >= 16]
            decoded = supported[0] if len(supported) == 1 and sum(value > 0 for value in counts.values()) == 1 else None
            require(decoded == observed['cue']['class_id'], 'trace class differs from observed RGB')
            if paid <= 18 and decoded is not None:
                require(decoded == details['result']['physical_case']['hypothesis'] + 2,
                        'wrong observed controlled class within the common prefix')
                recognized_positions.add(tuple(observed['pose_v33'][:2]))
        require(history['state']['measured_map_sha256'] == observed['measured_map_sha256'],
                'controller map identity differs from saved observed map identity')
        checked += 1
    require(len(recognized_positions) >= 2, 'actual prefix RGB lacked two distinct recognized positions')
    for stage in ('prefix', 'final'):
        with np.load(folder / (stage + '_maps.npz'), allow_pickle=False) as maps:
            require(list(maps['belief'].shape) == parent_config['raster_shape'], 'parent raster shape differs')
    return {'raw_nonsemantic_packets_verified': checked,
            'controlled_prefix_distinct_positions': sorted(map(list, recognized_positions)),
            'observed_pose_and_parent_raster_verified': True}


def metrics(reports):
    """Compute the declared paired means without relying on collector contrasts."""
    if len(reports) != 8 or not all(value.get('complete') for value in reports.values()):
        return None, None
    by_parent = {}
    for parent in PARENTS:
        by_parent[parent] = {}
        for threshold in THRESHOLDS:
            values = {(h, mode): reports[parent, h, mode]['final_measurement'][threshold]['joint']
                      for h in (0, 1) for mode in MODES}
            means = {mode: sum(values[h, mode] for h in (0, 1)) / 2 for mode in MODES}
            differences = [values[h, 'S'] - values[h, 'G'] for h in (0, 1)]
            difference = sum(differences) / 2
            by_parent[parent][threshold] = {
                'mean_joint': means,
                'S_vs_G': {'treatment': 'S', 'baseline': 'G',
                    'paired_joint_differences': differences,
                    'paired_mean_joint_difference': difference,
                    'relative_mean_joint_difference': difference / means['G'] if means['G'] else None}}
    overall = {}
    for threshold in THRESHOLDS:
        means = {mode: sum(by_parent[parent][threshold]['mean_joint'][mode] for parent in PARENTS) / 2
                 for mode in MODES}
        differences = [by_parent[parent][threshold]['S_vs_G']['paired_mean_joint_difference'] for parent in PARENTS]
        difference = sum(differences) / 2
        overall[threshold] = {'mean_joint': means,
            'S_vs_G': {'treatment': 'S', 'baseline': 'G',
                'per_parent_mean_joint_differences': differences,
                'paired_mean_joint_difference': difference,
                'relative_mean_joint_difference': difference / means['G'] if means['G'] else None}}
    return by_parent, overall


def compare_statistics(actual, declared, parent):
    for threshold in THRESHOLDS:
        for mode in MODES:
            close(actual[threshold]['mean_joint'][mode], declared[threshold]['mean_joint'][mode],
                  parent + '/' + threshold + ' method mean differs')
        source, target = actual[threshold]['S_vs_G'], declared[threshold]['S_vs_G']
        for field in ('paired_mean_joint_difference', 'relative_mean_joint_difference'):
            if source[field] is None:
                require(target[field] is None, 'undefined relative mean differs')
            else:
                close(source[field], target[field], parent + '/' + threshold + '/' + field)
        if parent in PARENTS:
            require(len(source['paired_joint_differences']) == len(target['paired_joint_differences']) == 2,
                    'per-parent paired difference length differs')
            for actual_value, declared_value in zip(source['paired_joint_differences'], target['paired_joint_differences']):
                close(actual_value, declared_value, parent + '/' + threshold + ' per-hypothesis difference differs')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--batch', type=Path, default=BATCH)
    parser.add_argument('--output', type=Path, default=OUTPUT)
    args = parser.parse_args()
    for name in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
        require(os.environ.get(name) == '1', name + '=1 required')
    require(sys.dont_write_bytecode, '-B required')
    batch, output = args.batch.resolve(), args.output.resolve()
    require((batch / 'final_seal.json').exists(), 'collector must be terminal and sealed')
    verify_seal(batch, 'final_seal.json')
    from nso.research_evidence_v31 import freeze, write, seal, verify_sources, verify_inventory
    manifest = verify_sources(batch)
    for relative, expected in manifest['input_sha256'].items():
        require(sha(ROOT / relative) == expected, 'collector prerequisite input changed: ' + relative)
    config, summary = read(batch / 'config.json'), read(batch / 'result.json')
    require(summary['status'] in ('complete', 'stopped'), 'collector is not terminal')
    expected = [(parent, h, mode) for parent in PARENTS for h in (0, 1) for mode in MODES]
    require([(case['parent'], case['hypothesis'], case['mode']) for case in config['physical_cases']] == expected,
            'preregistered eight-case order differs')
    require(config['historical_main_used'] == 27 and config['maximum_new_main_starts'] == 8 and config['total_main_limit'] == 36,
            'main task quota differs from protocol')
    require(set(config['parents']) == set(PARENTS), 'public parent set differs')
    for parent in PARENTS:
        parameters = config['parents'][parent]
        require(parameters['parent'] == parent and parameters['noise_seed'] == 350918 and parameters['noise_model'] == 'iid_025px',
                'parent or held-out noise contract differs')
        require(parameters['budget'] == 42 and parameters['forced_prefix'] == 18 and parameters['voxel_m'] == .04 and
                parameters['truncation_m'] == .12 and parameters['exact_pose'] is True,
                'frozen execution parameters differ')
    output.mkdir(exist_ok=False)
    counts = {'new_worlds': 0, 'new_sensor_queries': 0, 'new_main_tasks': 0,
              'new_mapper_or_TSDF_calls': 0, 'new_surface_metric_calls': 0, 'new_DP_calls': 0,
              'saved_packets_verified': 0, 'complete_cases_verified': 0}
    reports, saved = {}, {}
    try:
        freeze(output, [Path(__file__), PROTOCOL, PLAN],
               input_sha256={str((batch / name).relative_to(ROOT)): sha(batch / name)
                   for name in ('config.json', 'result.json', 'final_seal.json', 'manifest.json')},
               replay_not_an_independent_sample=True, held_out_parent_geometry=False,
               new_DP_or_physical_calls_allowed=False)
        for case in config['physical_cases']:
            folder = batch / f"case{case['index']:02d}"
            key = case['parent'], case['hypothesis'], case['mode']
            if not folder.exists():
                reports[key] = {'case': case, 'complete': False, 'reason': 'not started after retained stop'}
                continue
            if not (folder / 'main_seal.json').exists() or (folder / 'failure.json').exists() or (folder / 'replay' / 'failure.json').exists():
                reports[key] = {'case': case, 'complete': False, 'reason': 'retained collector/replay implementation failure'}
                continue
            report, details = inspect_case(folder, case)
            report['raw_packet_and_parent_checks'] = confirm_packet_fields(folder, details, config['parents'][case['parent']])
            reports[key], saved[key] = report, details
            counts['saved_packets_verified'] += report['checked_paid_packets']
            counts['complete_cases_verified'] += 1
        parent_metrics, overall_metrics = metrics(reports)
        if parent_metrics is not None:
            for parent in PARENTS:
                compare_statistics(parent_metrics[parent], summary['prescribed_paired_comparison'][parent], parent)
            compare_statistics(overall_metrics, summary['overall_comparison'], 'equal-weight parents')
        divergences = []
        prefix_checks = {}
        for parent in PARENTS:
            rows = [saved.get((parent, h, mode)) for h in (0, 1) for mode in MODES]
            prefix_checks[parent] = all(row is not None and len(row['trace']) >= 19 for row in rows)
            if prefix_checks[parent]:
                reference = rows[0]['trace'][:19]
                prefix_checks[parent] = all(all(left['nonsemantic'] == right['nonsemantic']
                    for left, right in zip(reference, row['trace'][:19])) for row in rows)
            for h in (0, 1):
                if (parent, h, 'S') in saved and (parent, h, 'G') in saved:
                    row = first_divergence(saved[parent, h, 'S'], saved[parent, h, 'G'], 'S_vs_G')
                    row.update(parent=parent, hypothesis=h,
                        final_J5={mode: reports[parent, h, mode]['final_measurement']['05cm']['joint'] for mode in MODES})
                    divergences.append(row)
        causal_action = any(row.get('same_nonsemantic_observation_history') and row.get('same_pose') and
            row.get('same_masks') and not row.get('same_posterior') and
            row.get('interpretation') == 'same observed geometry with different RGB-derived semantic prior'
            for row in divergences)
        all_qualified = len(reports) == 8 and all(row.get('eligible', False) for row in reports.values())
        all_replayed = len(reports) == 8 and all(row.get('independent_replay_passed', False) for row in reports.values())
        cue_passed = len(reports) == 8 and all(row.get('controlled_prefix_cue_passed', False) for row in reports.values())
        effect_checks = {parent: {'J5_mean_strictly_positive': bool(parent_metrics and
            parent_metrics[parent]['05cm']['S_vs_G']['paired_mean_joint_difference'] > 0),
            'J2_J10_mean_nonnegative': bool(parent_metrics and all(
                parent_metrics[parent][threshold]['S_vs_G']['paired_mean_joint_difference'] >= 0
                for threshold in ('02cm', '10cm')))} for parent in PARENTS}
        checks = {'collector_qualification_and_replay_gate': bool(summary['qualification_and_replay_gate_passed']),
            'all_eight_terminal_qualified': all_qualified, 'all_eight_independent_online_replays_passed': all_replayed,
            'common_prefix_nonsemantic_equivalent_within_each_parent': all(prefix_checks.values()),
            'all_eight_actual_RGB_cue_prerequisites': cue_passed,
            'both_parents_primary_positive': all(row['J5_mean_strictly_positive'] for row in effect_checks.values()),
            'both_parents_secondary_nonnegative': all(row['J2_J10_mean_nonnegative'] for row in effect_checks.values()),
            'semantic_prior_changes_action_under_same_nonsemantic_history': bool(causal_action)}
        result = {'status': 'passed', 'saved_evidence_verified': True,
            'source_batch_result_sha256': sha(batch / 'result.json'),
            'source_batch_final_seal_sha256': sha(batch / 'final_seal.json'),
            'case_reports': list(reports.values()), 'per_parent_prefix_checks': prefix_checks,
            'per_parent_paired_metrics': parent_metrics, 'overall_equal_weight_metrics': overall_metrics,
            'first_semantic_action_divergences': divergences, 'per_parent_effect_checks': effect_checks,
            'checks': checks, 'complete_confirmation_gate_passed': all(checks.values()), 'counts': counts,
            'new_main_tasks': summary['new_main_tasks'], 'total_main_used': summary['total_main_used'],
            'surface_distance_metrics_recomputed': False,
            'quality_values_source': 'sealed actual TSDF P/R; independently checked F1/J arithmetic and saved coverage raster',
            'feedback_effect_reconfirmed': False, 'independent_unseen_geometry_test': False,
            'scope': 'frozen V35 CPU policy, two known parents in one designed family, one preregistered held-out noise seed; P01 ideal geometry was previously inspected; artificial RGB and exact pose, not natural semantics or original learned ANS'}
        write(output, output / 'result.json', result)
        seal(output); verify_sources(output); verify_inventory(output)
        verify_seal(batch, 'final_seal.json')
        print(json.dumps({'status': 'passed', 'complete_confirmation_gate_passed': result['complete_confirmation_gate_passed'],
                          'counts': counts, 'result_sha256': sha(output / 'result.json')}))
    except BaseException:
        write(output, output / 'failure.json', {'status': 'failed', 'error': traceback.format_exc(),
              'counts': counts, 'case_reports_completed': list(reports.values()), 'implicit_retry_allowed': False})
        if not (output / 'artifact_hashes.json').exists():
            seal(output)
        raise


if __name__ == '__main__':
    main()
