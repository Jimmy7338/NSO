#!/usr/bin/env python3
"""Read-only finite scene-matrix progress and already-reviewed metric summary.

No World, controller, mapper, replay or numerical surface evaluation is run.
Current complete archives are checked by the existing strict scene inspector;
only separately saved, bound and replay-verified reviews supply C/Q/J values.
"""
import argparse
from collections import Counter
from copy import deepcopy
import json
import math
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.summarize_semantic_integration import (
    EPS, _records, digest, pair_summary, parse_steps, read, sha,
)

STATES = ('not_attempted', 'reserved_or_running', 'failed_attempt',
          'complete_episode', 'integrity_error')
DEFAULT_PROTOCOL = ROOT/'configs/virtual3d/semantic_development_constructor_fixed_20260923.json'
REUSE_SCHEMA = 'semantic.scene_endpoint_evaluation_reused.v1'
SCIENTIFIC_FIELDS = ('slots', 'maximum_new_world_slots', 'analysis_blocks', 'mapper', 'evaluation',
    'development_gates', 'analysis_rules', 'asset_root', 'asset_manifest_sha256', 'navigation_root',
    'navigation_manifest_sha256', 'reference_index_path', 'reference_index_sha256',
    'maximum_episode_bytes', 'maximum_file_bytes', 'terminal_record_reserve_bytes',
    'maximum_elapsed_s', 'expected_batch_peak_bytes')


def contained(relative):
    value = Path(relative)
    if value.is_absolute() or '..' in value.parts:
        raise ValueError('repository-relative declared input required')
    path = ROOT/value
    if path.is_symlink() or not path.resolve().is_relative_to(ROOT):
        raise ValueError('plain contained input required')
    return path.resolve()


def review_metrics(review, episode, *, require_current_sources=True):
    """Validate existing review provenance without recomputing a numeric score."""
    manifest, protocol, result = episode['manifest'], episode['protocol'], episode['result']
    if (review.get('schema') != 'semantic.scene_experiment.review.v1'
            or review.get('run_id') != episode['started']['run_id']
            or review.get('phase_id') != protocol['phase_id']
            or review.get('episode_manifest_sha256') != episode['manifest_sha256']
            or review.get('source_sha256') != manifest['source_sha256']
            or review.get('input_sha256') != manifest['input_sha256']):
        raise ValueError('review identity/source/input/episode pin mismatch')
    if review.get('status') not in ('experiment_reviewed', 'experiment_replay_verified'):
        return None
    replay, evaluation, runtime = (review.get(k, {}) for k in ('replay', 'evaluation', 'runtime'))
    if (review.get('current_sources_match') is not True
            or (require_current_sources and episode['current_sources_match'] is not True)
            or replay.get('status') != 'verified' or replay.get('prediction_verified') is not True
            or replay.get('frames_verified') != result['acquired_and_saved_packets']
            or runtime.get('no_new_world_or_sensor_action') is not True
            or 'before' not in runtime or runtime['before'] != runtime.get('after')):
        raise ValueError('review did not verify the complete saved policy/prediction under bound sources')
    if review['status'] == 'experiment_replay_verified':
        return None
    prediction_pins = {name: manifest['files']['prediction/'+name]['sha256']
                      for name in ('mapper.json', 'occupancy.npz', 'mesh.npz')}
    if (evaluation.get('schema') != 'semantic_scene.complete_saved_prediction_evaluation.v1'
            or evaluation.get('asset_id') != episode['slot']['asset_id']
            or evaluation.get('asset_manifest_sha256') != protocol['asset_manifest_sha256']
            or evaluation.get('episode_manifest_sha256') != episode['manifest_sha256']
            or evaluation.get('reference_manifest_sha256') != episode['reference']['manifest_sha256']
            or evaluation.get('input_prediction_sha256') != prediction_pins
            or evaluation.get('evaluator_source_sha256') != manifest['source_sha256'].get('nso/complete_surface_evaluation.py')
            or evaluation.get('all_task_instances_in_macro_denominator') is not True
            or evaluation.get('prediction_roi_cropped') is not False
            or evaluation.get('prediction_reintegrated') is not False
            or any(evaluation.get(k) != 0 for k in ('new_worlds', 'new_sensor_packets', 'new_tsdf_integrations'))):
        raise ValueError('complete endpoint review metric/reference/prediction boundary mismatch')
    metrics = evaluation.get('metrics', {})
    for key in ('C_nav', 'Q', 'J_nav'):
        value = metrics.get(key)
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or not 0 <= value <= 1:
            raise ValueError('finite reviewed C/Q/J in [0,1] required')
    if abs(metrics['J_nav']-metrics['C_nav']*metrics['Q']) > EPS:
        raise ValueError('reviewed J differs from C*Q')
    return metrics


def historical_reference_record(episode):
    """Verify old reference bytes/source archives without importing old code."""
    reference = episode['reference']
    root = contained(reference['root'])
    if sha(root/'manifest.json') != reference['manifest_sha256']:
        raise ValueError('retired reference manifest changed')
    manifest = read(root/'manifest.json')
    expected = {'surface.npz', 'coverage_domain.npz', 'candidate_views.json', 'reference.json'}
    if set(manifest['files']) != expected:
        raise ValueError('retired reference file inventory differs')
    for name, metadata in manifest['files'].items():
        path = root/name
        if path.is_symlink() or sha(path) != metadata['sha256'] or path.stat().st_size != metadata['bytes']:
            raise ValueError('retired reference bytes changed')
    record = read(root/'reference.json')
    if (record['asset_id'] != episode['slot']['asset_id']
            or record['asset_manifest_sha256'] != episode['protocol']['asset_manifest_sha256']
            or record['evaluation_parameters'] != episode['protocol']['evaluation']
            or any(episode['manifest']['source_sha256'].get(k) != v for k, v in record['source_sha256'].items())):
        raise ValueError('retired reference differs from its archived executed sources or metric')
    return record


def reused_evaluation(receipt, episode, *, historical_sources=False):
    """Independently recheck a saved exact-input proof; never create a reuse.

    The producer's read-only helpers validate both original archived episodes,
    both independently completed policy/TSDF replays, and exact arrays. No call
    to its ``reuse`` writer, a replay driver or an evaluator is permitted here.
    """
    from scripts import reuse_semantic_scene_endpoint_evaluation as proof
    if historical_sources:
        from nso.semantic_scene_experiment import inspect_experiment as check_episode
    else:
        check_episode = proof._checked_episode
    root = Path(episode['root']).resolve()
    runtime = receipt.get('runtime', {})
    if (receipt.get('schema') != REUSE_SCHEMA
            or receipt.get('status') != 'experiment_endpoint_evaluation_reused'
            or receipt.get('phase_id') != episode['protocol']['phase_id']
            or receipt.get('run_id') != episode['started']['run_id']
            or receipt.get('method') != episode['slot']['method']
            or receipt.get('asset_id') != episode['slot']['asset_id']
            or Path(receipt.get('episode_root', '')).resolve() != root
            or receipt.get('episode_manifest_sha256') != episode['manifest_sha256']
            or receipt.get('source_sha256') != episode['manifest']['source_sha256']
            or receipt.get('input_sha256') != episode['manifest']['input_sha256']
            or receipt.get('numerical_evaluation_recomputed') is not False
            or receipt.get('independent_replay_performed_here') is not False
            or receipt.get('original_sources_or_outputs_modified') is not False
            or runtime.get('no_new_world_or_sensor_action') is not True
            or 'before' not in runtime or runtime['before'] != runtime.get('after')
            or any(receipt.get(k) != 0 for k in ('new_worlds', 'new_sensor_packets', 'new_tsdf_integrations', 'physical_actions'))):
        raise ValueError('reuse receipt target identity or read-only execution contract mismatch')
    source_root = Path(receipt['source_episode_root']).resolve()
    if source_root == root:
        raise ValueError('reuse source and target episodes must differ')
    code_copy = Path(receipt['supplemental_source_copy'])
    if (code_copy.is_symlink() or sha(code_copy) != receipt.get('supplemental_source_sha256')
            or (not historical_sources and sha(Path(proof.__file__)) != receipt.get('supplemental_source_sha256'))):
        raise ValueError('saved reuse producer source differs from its current checked implementation')
    source = check_episode(source_root, receipt['source_episode_manifest_sha256'])
    target = check_episode(root, episode['manifest_sha256'])
    sr_path, tr_path = (Path(receipt[key]) for key in ('source_review_path', 'target_review_path'))
    sr = proof._checked_review(sr_path, receipt['source_review_sha256'], source, complete_numerical=True)
    tr = proof._checked_review(tr_path, receipt['target_review_sha256'], target, complete_numerical=False)
    source_slot = {k: v for k, v in source['slot'].items() if k != 'method'}
    target_slot = {k: v for k, v in target['slot'].items() if k != 'method'}
    if (source['started']['run_id'] == target['started']['run_id']
            or receipt.get('source_run_id') != source['started']['run_id']
            or source['protocol'] != target['protocol'] or source_slot != target_slot
            or source['reference'] != target['reference']
            or source['manifest']['source_sha256'] != target['manifest']['source_sha256']
            or source['manifest']['input_sha256'] != target['manifest']['input_sha256']
            or receipt.get('source_replay') != sr['replay'] or receipt.get('target_replay') != tr['replay']
            or receipt.get('source_original_review_status') != sr['status']
            or receipt.get('target_original_review_status') != tr['status']
            or receipt.get('source_original_review_elapsed_s') != sr['elapsed_s']):
        raise ValueError('reuse source/target frozen configuration or independent replay chain mismatch')
    source_metrics = review_metrics(sr, source, require_current_sources=not historical_sources)
    if source_metrics is None:
        raise ValueError('reuse source lacks an original complete numerical evaluation')
    # These helpers only reopen pinned references and arrays, not numerical
    # surface scores. Reconstruct the whole proof instead of trusting its flags.
    if historical_sources:
        record = historical_reference_record(source)
        fingerprint = record['surface']['fingerprint']
    else:
        surface, _, record = proof._load_reference(source)
        fingerprint = surface.fingerprint
    arrays = proof._equal_arrays(source_root, root)
    public = {}
    for name in ('public_workspace.json', 'public_spec.json', 'public_graph.json'):
        left, right = sha(source_root/name), sha(root/name)
        if left != right:
            raise ValueError('reuse public input bytes differ')
        public[name] = dict(source_sha256=left, target_sha256=right, byte_identical=True)
    sm, tm = (read(p/'prediction/mapper.json') for p in (source_root, root))
    if (any(k not in sm or k not in tm or sm[k] != tm[k] for k in proof.MAPPER_FIELDS)
            or sm['backend_poisoned'] is not False
            or sm['frames'] != source['result']['acquired_and_saved_packets']
            or tm['frames'] != target['result']['acquired_and_saved_packets']
            or any(sm[k] != record['coverage'][k] for k in ('shape', 'resolution_m', 'origin_xy_m', 'grid_convention'))
            or arrays['prediction/occupancy.npz']['arrays']['belief']['array_sha256'] != sm['occupancy_sha256']):
        raise ValueError('reuse mapper numerical input/history binding differs')
    evaluator_sources = {n: source['manifest']['source_sha256'][n] for n in proof.EVALUATOR_SOURCES}
    expected_proof = dict(exact_original_arrays=arrays, byte_identical_public_files=public,
        identical_mapper_evaluation_fields={k: sm[k] for k in proof.MAPPER_FIELDS},
        identical_complete_source_closure=True, executed_source_count=len(source['manifest']['source_sha256']),
        evaluator_source_sha256=evaluator_sources, identical_frozen_protocol=True,
        identical_nonmethod_slot=source_slot, evaluation_configuration=proof.EVALUATION,
        asset_id=source['slot']['asset_id'], reference_spec=source['reference'],
        input_sha256=source['manifest']['input_sha256'], complete_source_and_target_saved_policy_TSDF_replays_verified=True)
    if receipt.get('equivalence_proof') != expected_proof:
        raise ValueError('saved exact-input equivalence proof does not match original inputs')
    evaluation = sr['evaluation']; coverage = evaluation.get('coverage', {})
    if (evaluation.get('task_success') != proof._task_success(source)
            or evaluation.get('original_episode_status') != source['result']['status']
            or evaluation.get('independent_replay_performed_here') is not False
            or source_metrics.get('reference_fingerprint') != fingerprint
            or source_metrics.get('prediction_mesh_validation') != 'strict_positive_area_without_absolute_area_floor'
            or coverage.get('denominator') != record['coverage']
            or source_metrics.get('prediction_sample_spacing_m') != proof.EVALUATION['sample_spacing_m']
            or source_metrics.get('prediction_seed') != proof.EVALUATION['seed']
            or source_metrics.get('threshold_m') != proof.EVALUATION['threshold_m']
            or source_metrics.get('C_nav') != coverage.get('C_nav')):
        raise ValueError('original source score does not bind the fixed complete-surface evaluation')
    expected = deepcopy(evaluation)
    expected.update(episode_manifest_sha256=target['manifest_sha256'],
        input_prediction_sha256=proof._prediction_pins(target), task_success=proof._task_success(target),
        original_episode_status=target['result']['status'], numerical_evaluation_recomputed=False,
        independent_replay_performed_here=False)
    if (receipt.get('evaluation') != expected or receipt.get('rebound_target_provenance_fields') != [
            'episode_manifest_sha256', 'input_prediction_sha256', 'task_success', 'original_episode_status']):
        raise ValueError('reused numeric values changed or target provenance was not rebound exactly')
    # Original artifacts/reviews remain immutable throughout the snapshot.
    check_episode(source_root, source['manifest_sha256'])
    check_episode(root, target['manifest_sha256'])
    if sha(sr_path) != receipt['source_review_sha256'] or sha(tr_path) != receipt['target_review_sha256']:
        raise ValueError('reuse original review changed during validation')
    return dict(metrics=source_metrics, evaluation=expected, evaluation_execution='reused',
        numerical_evaluation_recomputed=False, source_run_id=source['started']['run_id'],
        source_review_path=str(sr_path), source_review_sha256=receipt['source_review_sha256'],
        target_review_path=str(tr_path), target_review_sha256=receipt['target_review_sha256'],
        source_validation='archived_executed_sources' if historical_sources else 'current_and_archived_sources',
        independent_target_replay_verified=True, exact_numerical_input_chain_verified=True)


def reuse_index(root, slots):
    """Dedicated supplemental directory; refuse ambiguity, never pick latest."""
    index = {}
    if not root.exists():
        return index
    if root.is_symlink() or not root.is_dir():
        raise ValueError('plain supplemental receipt directory required')
    for path in sorted(root.glob('*.json')):
        if path.is_symlink():
            raise ValueError('plain supplemental receipt required')
        receipt = read(path)
        identity = receipt.get('run_id')
        if receipt.get('schema') != REUSE_SCHEMA or identity not in slots or identity in index:
            raise ValueError('unknown or duplicate target in supplemental reuse directory')
        index[identity] = dict(path=str(path.resolve()), sha256=sha(path), receipt=receipt)
    return index


def core_comparison(protocol, rows, *, name, condition, baseline, relative_minimum,
                    require_parent_consistency=False):
    """Pure aggregation: unavailable endpoints stay None, never measured zero."""
    required = protocol['development_gates']['development_parent_count']
    ids = set(protocol['analysis_blocks']['core']['run_ids'])
    candidates = [r for r in rows if r['run_id'] in ids and r['condition'] == condition]
    parents = sorted({r['parent_id'] for r in candidates})
    if len(parents) != required:
        raise ValueError('core comparison does not declare exactly six parent layouts')
    pairs = []
    arms = {baseline: [], 'shared_semantic': []}
    for parent in parents:
        selected = {}
        for method in arms:
            group = [r for r in candidates if r['parent_id'] == parent and r['method'] == method]
            if len(group) != 1:
                raise ValueError('core requires exactly one matched slot per parent and method')
            selected[method] = group[0]
            arms[method].append(group[0])
        b, s = selected[baseline], selected['shared_semantic']
        if any(b[k] != s[k] for k in ('budget', 'noise_seed', 'tie_rule', 'tie_seed')):
            raise ValueError('core comparison budget/noise/tie mismatch')
        available = b.get('metrics') is not None and s.get('metrics') is not None
        pair = dict(parent_id=parent, baseline_run_id=b['run_id'], candidate_run_id=s['run_id'],
                    complete_reviewed_pair=available, delta_J=None, relative_delta_J=None,
                    delta_C=None, delta_Q=None, return_difference=None)
        if available:
            bm, sm = b['metrics'], s['metrics']
            pair.update(delta_J=sm['J_nav']-bm['J_nav'],
                relative_delta_J=(sm['J_nav']-bm['J_nav'])/bm['J_nav'] if bm['J_nav'] else None,
                delta_C=sm['C_nav']-bm['C_nav'], delta_Q=sm['Q']-bm['Q'],
                return_difference=int(s['returned_xy_and_yaw'])-int(b['returned_xy_and_yaw']))
        pairs.append(pair)
    complete = [p for p in pairs if p['complete_reviewed_pair']]
    by_id = {r['run_id']: r for r in rows}
    baseline_mean = (sum(by_id[p['baseline_run_id']]['metrics']['J_nav'] for p in complete)/len(complete)
                     if complete else None)
    candidate_mean = (sum(by_id[p['candidate_run_id']]['metrics']['J_nav'] for p in complete)/len(complete)
                      if complete else None)
    difference = candidate_mean-baseline_mean if complete else None
    relative = difference/baseline_mean if baseline_mean else None
    mean_C = sum(p['delta_C'] for p in complete)/len(complete) if complete else None
    mean_return = sum(p['return_difference'] for p in complete)/len(complete) if complete else None
    positives = [p['delta_J'] for p in complete if p['delta_J'] > 0]
    max_share = max(positives)/sum(positives) if positives else None
    gates = protocol['development_gates']
    checks = dict(relative_J=None if relative is None else relative >= relative_minimum,
        coverage=None if mean_C is None else mean_C >= gates['mean_coverage_difference_minimum'],
        return_rate=None if mean_return is None else mean_return >= 0)
    if require_parent_consistency:
        checks.update(positive_parent_count=len(positives) >= gates['positive_parent_count_minimum'],
            positive_gain_concentration=None if max_share is None else max_share <= gates['maximum_one_parent_share_of_positive_J_differences'])
    status = ('insufficient_complete_pairs' if len(complete) != required else
              'undefined_zero_baseline' if relative is None else
              'numeric_thresholds_passed' if all(checks.values()) else 'numeric_thresholds_failed')
    # Hypothetical bounds for unobserved arms are explicitly NOT imputations.
    intervals = {}
    for metric in ('J_nav', 'C_nav'):
        limits = {}
        for method, group in arms.items():
            known = [r['metrics'][metric] for r in group if r.get('metrics') is not None]
            limits[method] = [sum(known)/required, (sum(known)+required-len(known))/required]
        intervals[metric+'_mean_candidate_minus_baseline'] = [
            limits['shared_semantic'][0]-limits[baseline][1],
            limits['shared_semantic'][1]-limits[baseline][0]]
    return dict(name=name, condition=condition, baseline_method=baseline,
        candidate_method='shared_semantic', required_parent_pairs=required,
        complete_reviewed_parent_pairs=len(complete), pairs=pairs,
        complete_pair_baseline_mean_J=baseline_mean, complete_pair_candidate_mean_J=candidate_mean,
        complete_pair_absolute_mean_J_difference=difference,
        relative_difference_of_means=relative, mean_coverage_difference=mean_C,
        mean_return_rate_difference=mean_return, positive_parent_count=len(positives),
        maximum_parent_share_of_positive_differences=max_share,
        relative_J_threshold=relative_minimum, numeric_checks=checks, status=status,
        full_parent_hypothetical_sensitivity_bounds=intervals,
        bounds_are_not_observed_or_imputed_scores=True, partial_pair_means_are_descriptive_only=len(complete) != required,
        full_development_gate_pass=None, attribution_not_decided_here=True)


def retained_correction(protocol):
    declaration = protocol.get('constructor_correction')
    if not declaration:
        raise ValueError('corrected phase must retain its explicit original failed attempt')
    path = contained(declaration['record'])
    if sha(path) != declaration['record_sha256']:
        raise ValueError('constructor correction record pin mismatch')
    record = read(path)
    pinned = {}
    for name in ('original_protocol', 'original_ledger', 'failed_attempt'):
        source = contained(record[name+'_path'])
        if sha(source) != record[name+'_sha256']:
            raise ValueError('retained correction source pin mismatch: '+name)
        pinned[name] = read(source)
    original, ledger, failure = (pinned[k] for k in ('original_protocol', 'original_ledger', 'failed_attempt'))
    if (original['slots'] != protocol['slots'] or len(protocol['slots']) != 96
            or ledger['slots'] != protocol['slots'] or len(ledger['entries']) != 1
            or failure.get('world_created') is not False
            or failure.get('status') != 'experiment_attempt_failed'
            or ledger['entries'][0]['result_sha256'] != record['failed_attempt_sha256']
            or ledger['entries'][0]['world_created'] is not False
            or ledger['protocol_sha256'] != record['original_protocol_sha256']):
        raise ValueError('original zero-World failure or identical finite matrix not retained')
    for key in ('controller', 'mapper', 'evaluation', 'development_gates', 'analysis_blocks',
                'asset_manifest_sha256', 'navigation_manifest_sha256', 'reference_index_sha256'):
        if original[key] != protocol[key]:
            raise ValueError('constructor correction changed experiment scope: '+key)
    return dict(record_path=str(path), record_sha256=declaration['record_sha256'],
        original_protocol_path=str(contained(record['original_protocol_path'])),
        original_ledger_path=str(contained(record['original_ledger_path'])),
        failed_attempt_path=str(contained(record['failed_attempt_path'])),
        failed_attempt_sha256=record['failed_attempt_sha256'], failure=failure,
        original_failed_attempts=1, original_worlds_created=0, original_paid_actions=0,
        same_96_slot_matrix_verified=True, superseded_unstarted_slots=95,
        total_attempt_ceiling_including_failure=97, total_primary_world_ceiling=96,
        correction_scope=record['scope'], method_candidate_revision_count=record['method_candidate_revision_count'])


def validate_method_layout(protocol, record, previous, ledger, summary):
    """Pure declaration checks; versions never share a core-comparison row."""
    declaration = protocol['method_correction']
    entries = ledger['entries']
    retained = record['retained_original_episodes']
    old_ids = [e['run_id'] for e in entries]
    paired = record['paired_before_after_run_ids']
    retired = record['retired_original_unstarted_run_ids']
    if (len(protocol['slots']) != 96 or protocol['slots'] != previous['slots']
            or ledger['slots'] != previous['slots'] or len(entries) != 5
            or len(set(old_ids)) != 5 or len(retained) != 5
            or {r['run_id'] for r in retained} != set(old_ids)
            or len(paired) != 5 or set(paired) != set(old_ids)
            or len(retired) != 91 or len(set(retired)) != 91 or set(retired) & set(old_ids)
            or set(retired) | set(old_ids) != set(previous['slots'])
            or protocol['controller'] != dict(previous['controller'], measurement_acquisition=True)):
        raise ValueError('single method correction must retain five old runs, retire 91, and preserve all 96 slots')
    if (tuple(record['scientific_matrix_fields']) != SCIENTIFIC_FIELDS
            or any(protocol[k] != previous[k] for k in SCIENTIFIC_FIELDS)
            or digest({k: previous[k] for k in SCIENTIFIC_FIELDS}) != record['scientific_matrix_sha256']):
        raise ValueError('method correction changed a scientific matrix field or its digest')
    if (record.get('schema') not in ('semantic.method_correction_draft.v1', 'semantic.method_correction.v1')
            or record['original_phase_id'] != previous['phase_id']
            or record['proposed_phase_id'] != protocol['phase_id']
            or declaration.get('original_phase_id') != previous['phase_id']
            or record['original_protocol_sha256'] != ledger['protocol_sha256']
            or summary['phase_id'] != previous['phase_id']
            or summary['protocol_sha256'] != ledger['protocol_sha256']
            or summary['ledger_snapshot_sha256'] != record['original_ledger_sha256']
            or summary['states'] != dict(complete_episode=5, failed_attempt=0, integrity_error=0,
                                        not_attempted=91, reserved_or_running=0)
            or summary['reviewed_metric_endpoints'] != 5):
        raise ValueError('pre-correction ledger/summary identity or five-negative-endpoint inventory differs')
    for value in (record, declaration):
        if value.get('method_candidate_revision_number') != 1 or value.get('maximum_method_candidate_revisions') != 1:
            raise ValueError('exactly one bounded method correction is allowed')
    accounting = record['world_budget_accounting']
    if (accounting.get('original_complete_worlds') != 5
            or accounting.get('corrected_matrix_maximum_new_worlds') != 96
            or accounting.get('total_primary_world_ceiling') != 101
            or accounting.get('net_additional_world_allowance_over_original_matrix') != 5
            or declaration.get('total_primary_world_ceiling') != 101
            or declaration.get('corrected_matrix_slots') != 96
            or declaration.get('retained_original_worlds') != 5
            or declaration.get('retired_original_unstarted_slots') != 91
            or declaration.get('separate_constructor_failure_attempts') != 1
            or declaration.get('total_attempt_ceiling_including_constructor_failure') != 102
            or record.get('total_attempt_ceiling_including_constructor_failure') != 102
            or record.get('constructor_failure_separate', {}).get('count') != 1
            or record['constructor_failure_separate'].get('worlds_created') != 0
            or record.get('second_method_revision_or_second_tuning_round_allowed') is not False
            or record.get('automatic_retry_allowed') is not False):
        raise ValueError('cumulative 5+96 World / 102 attempt ceiling or no-second-tuning rule differs')
    frozen = protocol['status'] == 'frozen'
    if (record.get('status') not in (('draft', 'frozen') if frozen else ('draft',))
            or declaration.get('status') != protocol['status']
            or record.get('frozen') is not frozen
            or record.get('execution_authorized') is not frozen
            or declaration.get('execution_authorized') is not frozen):
        raise ValueError('method correction draft/frozen authorization disagrees')
    if frozen:
        sources = protocol.get('execution_source_sha256')
        if not sources or record.get('execution_source_sha256') != sources:
            raise ValueError('frozen corrected execution sources must bind protocol and method record equally')
    summary_rows = [r for r in summary['episodes'] if r.get('complete_episode')]
    if (len(summary_rows) != 5 or {r['run_id'] for r in summary_rows} != set(old_ids)
            or any(r.get('metrics') is None for r in summary_rows)):
        raise ValueError('original five reviewed metrics are not retained')
    return dict(retained_primary_worlds=5, current_matrix_world_ceiling=96,
        total_primary_world_ceiling=101, prior_constructor_failure_attempts=1,
        total_attempt_ceiling=102, old_completed_attempts=5, retired_unstarted_slots=91,
        old_results_included_in_current_core_comparisons=False)


def retained_method_correction(protocol):
    """Validate old immutable archives while allowing current method changes."""
    from nso.semantic_scene_experiment import inspect_experiment
    declaration = protocol['method_correction']
    path = contained(declaration['record'])
    if sha(path) != declaration['record_sha256']:
        raise ValueError('method correction declaration pin mismatch')
    record = read(path)
    bound = {}
    for name in ('original_protocol', 'original_ledger', 'original_progress_summary'):
        source = contained(record[name+'_path'])
        if sha(source) != record[name+'_sha256']:
            raise ValueError('retired evidence pin changed: '+name)
        bound[name] = read(source)
    previous, ledger, summary = (bound[k] for k in ('original_protocol', 'original_ledger', 'original_progress_summary'))
    accounting = validate_method_layout(protocol, record, previous, ledger, summary)
    constructor = retained_correction(previous)
    old_failure = record['constructor_failure_separate']
    if (sha(contained(old_failure['record_path'])) != old_failure['record_sha256']
            or old_failure['record_sha256'] != constructor['record_sha256']
            or old_failure['failed_attempt_sha256'] != constructor['failed_attempt_sha256']
            or sha(contained(old_failure['failed_attempt_path'])) != constructor['failed_attempt_sha256']):
        raise ValueError('method revision lost its separate zero-World constructor failure')
    summaries = {r['run_id']: r for r in summary['episodes']}
    entries = {r['run_id']: r for r in ledger['entries']}
    verified = []
    for retained in record['retained_original_episodes']:
        identity = retained['run_id']; entry, saved = entries[identity], summaries[identity]
        directory = contained(retained['episode_path'])
        if (str(directory) != entry['metadata']['output']
                or entry['world_created'] is not True or entry['status'] != 'controller_stop'
                or entry['metadata']['source_sha256'] != record['original_executed_source_sha256']
                or entry['metadata']['slot'] != previous['slots'][identity]
                or retained['episode_manifest_sha256'] != saved['episode_manifest_sha256']
                or retained['result_sha256'] != entry['result_sha256']):
            raise ValueError('retired episode declaration differs from durable original ledger')
        episode = inspect_experiment(directory, retained['episode_manifest_sha256'])
        if (episode['manifest']['source_sha256'] != record['original_executed_source_sha256']
                or episode['manifest']['protocol_sha256'] != record['original_protocol_sha256']
                or episode['slot'] != previous['slots'][identity]):
            raise ValueError('retired execution source archive/protocol/slot binding differs')
        review_path = contained(retained['review_or_reuse_path'])
        if sha(review_path) != retained['review_or_reuse_sha256'] or sha(review_path) != saved['review']['sha256']:
            raise ValueError('retired review/reuse receipt pin changed')
        review = read(review_path)
        if review.get('schema') == REUSE_SCHEMA:
            historical = reused_evaluation(review, episode, historical_sources=True)
            metrics, execution = historical['metrics'], 'reused'
        else:
            metrics = review_metrics(review, episode, require_current_sources=False)
            execution = 'recomputed'
        if (metrics is None or metrics != saved['metrics']
                or {k: metrics[k] for k in ('C_nav', 'Q', 'J_nav')} != retained['metrics']
                or execution != retained['evaluation_execution']
                or execution != saved['evaluation_execution']):
            raise ValueError('retired numerical evidence differs from its pinned negative-result summary')
        result = episode['result']
        actions = [r['controller_action'] for r in result['actions']]
        poses = [read(directory/f'packets/{i:03d}_receipt.json')['execution']['pose_xyyaw_rad']
                 for i in range(result['acquired_and_saved_packets'])]
        if (digest(actions) != retained['action_sequence_sha256']
                or digest(poses) != retained['pose_sequence_sha256']
                or sha(directory/'prediction/mesh.npz') != retained['mesh_file_sha256']
                or sha(directory/'prediction/occupancy.npz') != retained['occupancy_file_sha256']
                or result['executed_paid_actions'] != retained['executed_paid_actions']
                or result['sensor_status'].get('returned_xy_and_yaw') != retained['returned_xy_and_yaw']):
            raise ValueError('retired action/pose/prediction or return evidence changed')
        verified.append(dict(retained, archived_execution_sources_verified=True,
            current_sources_match=episode['current_sources_match'],
            current_source_differences=episode['current_source_differences'],
            current_source_match_required=False, included_in_current_core_comparisons=False))
    for name in ('original_protocol', 'original_ledger', 'original_progress_summary'):
        if sha(contained(record[name+'_path'])) != record[name+'_sha256']:
            raise ValueError('retired evidence changed during summary verification')
    return dict(record_path=str(path), record_sha256=declaration['record_sha256'],
        status=protocol['status'], record_origin_status=record['status'],
        execution_authorized=record['execution_authorized'],
        original_phase_id=previous['phase_id'], current_phase_id=protocol['phase_id'],
        original_protocol_path=record['original_protocol_path'], original_protocol_sha256=record['original_protocol_sha256'],
        original_ledger_path=record['original_ledger_path'], original_ledger_sha256=record['original_ledger_sha256'],
        original_progress_summary_path=record['original_progress_summary_path'],
        original_progress_summary_sha256=record['original_progress_summary_sha256'],
        retained_negative_episodes=verified, original_paid_actions=sum(r['executed_paid_actions'] for r in verified),
        constructor_correction=constructor, accounting=accounting,
        method_candidate_revision_number=1, maximum_method_candidate_revisions=1,
        old_executed_sources_may_differ_from_current=True, scientific_matrix_unchanged=True,
        full_method_comparison_not_inferred_from_before_after_pair=True)


def parse_scene_steps(records):
    """Extend the shared parser with explicitly non-utility acquisition events."""
    spent = None
    cap = None
    first_planes, cancellations, selections = [], [], []
    feedback_events = []
    def observed_records():
        nonlocal spent, cap
        for index, step, receipt in records:
            evidence, decision = step['controller_evidence'], step['decision']
            for row in evidence.get('geometry_feedback', []):
                applied = row.get('applied')
                feedback_events.append(dict(paid_step=index, observed_instance_id=row.get('instance_id'),
                    applied=applied if type(applied) is bool else None, reason=row.get('reason')))
            amounts = [owner['initialization_actions_spent'] for owner in (evidence, decision)
                       if 'initialization_actions_spent' in owner]
            if amounts:
                if (any(type(value) is not int or value < 0 for value in amounts)
                        or len(set(amounts)) != 1 or (spent is not None and amounts[0] < spent)):
                    raise ValueError('saved actual initialization spending is inconsistent')
                spent = amounts[0]
            if 'initialization_action_cap' in decision:
                value = decision['initialization_action_cap']
                if type(value) is not int or value < 0 or (cap is not None and value != cap):
                    raise ValueError('saved initialization action allowance changed')
                cap = value
            if spent is not None and cap is not None and spent > cap:
                raise ValueError('saved actual initialization spending exceeds its allowance')
            for key in evidence.get('first_actual_reliable_planes', []):
                first_planes.append(dict(paid_step=index, observed_instance_id=key))
            cancelled = decision.get('initialization_cancelled')
            if cancelled is not None:
                cancellations.append(dict(paid_step=index, **cancelled))
            selection = decision.get('global_selection') or {}
            chosen = selection.get('selected') or {}
            if chosen.get('kind') == 'measurement_initialization':
                if (any(chosen.get(k) is not None for k in ('score', 'expected_gain', 'evi'))
                        or chosen.get('semantic_information_used') is not False
                        or chosen.get('counted_as_surface_gain') is not False):
                    raise ValueError('measurement initialization must not masquerade as surface utility or EVI')
                selections.append(dict(paid_step=index, **chosen))
            yield index, step, receipt
    parsed = parse_steps(observed_records())
    # No diagnostic choice is different from a measured/predicted zero value.
    if parsed['counts'].get('diagnostic_options_scored', 0) == 0:
        parsed['diagnostic_evi_max'] = None
    if parsed['counts'].get('selected_diagnose_then_observe', 0) == 0:
        parsed['selected_diagnostic_evi_max'] = None
    applied = sum(row['applied'] is True for row in feedback_events)
    refused = [row for row in feedback_events if row['applied'] is False]
    if applied != parsed['counts'].get('frontend_geometry_updates_applied', 0):
        raise ValueError('shared parser applied-feedback count disagrees with explicit saved boolean')
    parsed['feedback_application'] = dict(feedback_attempts=len(feedback_events),
        feedback_applied=applied, feedback_refused=len(refused),
        feedback_application_unknown=sum(row['applied'] is None for row in feedback_events),
        refusal_reasons=dict(Counter(row['reason'] or 'reason_not_recorded' for row in refused)),
        informative_residuals=parsed['counts'].get('informative_residuals', 0),
        accepted_residuals=parsed['counts'].get('accepted_residuals', 0),
        events=feedback_events, source='saved controller_evidence.geometry_feedback only',
        scope='Frontend ledger feedback attempts; applied means the saved applied flag is exactly true.',
        informative_residual_does_not_imply_applied_feedback=True,
        predicted_evi_is_not_actual_feedback=True,
        refusal_reason_is_saved_verbatim_not_an_inferred_capacity_cause=True)
    parsed['measurement_acquisition'] = dict(
        receipts_present=spent is not None, actual_initialization_actions_spent=spent,
        initialization_action_cap=cap, selected_initialization_macros=len(selections),
        selected_initialization_events=selections, first_actual_reliable_plane_events=first_planes,
        first_actual_reliable_plane_instance_count=len({r['observed_instance_id'] for r in first_planes}),
        initialization_cancellation_reasons=dict(Counter(r['reason'] for r in cancellations)),
        initialization_cancellations=cancellations, source='saved paid-step receipts only',
        initialization_score_and_expected_gain_are_undefined=True,
        initialization_counted_as_surface_gain=False, initialization_counted_as_diagnostic_evi=False)
    return parsed


def summarize_slot(protocol, protocol_sha, run_id, entry, reviews_root, reuse_record=None):
    from nso.semantic_scene_experiment import COMPLETE, inspect_experiment
    slot = protocol['slots'][run_id]
    parent, condition = slot['asset_id'].split('__')
    row = dict(run_id=run_id, parent_id=parent, condition=condition, **slot,
        state='not_attempted', status=None, metrics=None, C_nav=None, Q=None, J_nav=None,
        evaluation_execution=None, numerical_evaluation_recomputed=None,
        complete_episode=False, terminal_attempt=False, returned_xy_and_yaw=None,
        review=dict(available=False, evaluation_available=False, saved_policy_replay_verified=False))
    if entry is None:
        return row
    expected_root = contained(protocol['output_relative_path'])/run_id
    metadata = entry['metadata']
    if (metadata.get('slot') != slot or metadata.get('protocol_sha256') != protocol_sha
            or metadata.get('run_id') != run_id or metadata.get('output') != str(expected_root)):
        raise ValueError('durable reservation differs from declared slot/protocol/output')
    row.update(status=entry['status'], episode_root=str(expected_root),
               reservation_time_unix_s=entry['reservation_time_unix_s'])
    if entry['status'] == 'reserved_before_factory':
        row.update(state='reserved_or_running', world_creation_unknown_until_terminal=True)
        return row
    row.update(terminal_attempt=True, world_created=entry.get('world_created'))
    terminal = expected_root/('attempt_failure.json' if entry['status'] == 'experiment_attempt_failed' else 'result.json')
    if not terminal.exists() and entry['status'] == 'experiment_attempt_failed':
        terminal = contained(protocol['ledger_relative_path']).parent/'failures'/f'{run_id}.json'
    if sha(terminal) != entry['result_sha256']:
        raise ValueError('terminal result differs from durable ledger')
    result = read(terminal)
    if result.get('status') != entry['status']:
        raise ValueError('terminal status differs from durable ledger')
    row.update(terminal_path=str(terminal), terminal_sha256=entry['result_sha256'])
    if (entry['status'] not in COMPLETE or result.get('error') is not None
            or result.get('finalization_errors') or not (expected_root/'artifact_manifest.json').is_file()):
        row.update(state='failed_attempt', failure=result)
        return row
    review_path = reviews_root/f'{run_id}.json'
    if review_path.is_symlink():
        raise ValueError('plain review file required')
    review = read(review_path) if review_path.is_file() else None
    pin = (review['episode_manifest_sha256'] if review is not None else
           reuse_record['receipt']['episode_manifest_sha256'] if reuse_record is not None else
           sha(expected_root/'artifact_manifest.json'))
    episode = inspect_experiment(expected_root, pin)
    if episode['manifest']['protocol_sha256'] != protocol_sha or episode['slot'] != slot:
        raise ValueError('inspected episode differs from supplied matrix')
    parsed = parse_scene_steps(_records(expected_root, result, episode['manifest'], protocol))
    actions = [a['controller_action'] for a in result['actions']]
    if len(actions) != result['executed_paid_actions']:
        raise ValueError('saved action sequence length differs')
    row.update(state='complete_episode', complete_episode=True,
        episode_manifest_sha256=pin,
        manifest_pin_origin=('bound_saved_review' if review is not None else
            'supplemental_reuse_target_pin' if reuse_record is not None else 'current_manifest_snapshot_pending_review'),
        current_sources_match=episode['current_sources_match'],
        current_source_differences=episode['current_source_differences'],
        executed_paid_actions=result['executed_paid_actions'], saved_packets=result['acquired_and_saved_packets'],
        returned_xy_and_yaw=result['sensor_status'].get('returned_xy_and_yaw'),
        collisions=result['collisions'], execution_elapsed_s=result['elapsed_s'], timings=result['timings'],
        task_status=result['status'], mesh_file_sha256=sha(expected_root/'prediction/mesh.npz'),
        occupancy_file_sha256=sha(expected_root/'prediction/occupancy.npz'),
        action_sequence_sha256=digest(actions), pose_sequence_sha256=digest(parsed['_poses']),
        episode_bytes=sum(p.stat().st_size for p in expected_root.rglob('*') if p.is_file()),
        **parsed, _actions=actions)
    if review is not None:
        metrics = review_metrics(review, episode)
        info = dict(available=True, path=str(review_path), sha256=sha(review_path), status=review['status'],
            error=review.get('error'), saved_policy_replay_verified=review['status'] in (
                'experiment_reviewed', 'experiment_replay_verified'),
            evaluation_available=metrics is not None)
        if metrics is not None:
            info['evaluation'] = review['evaluation']
            info.update(evaluation_execution='recomputed', numerical_evaluation_recomputed=True)
            row.update(metrics=metrics, C_nav=metrics['C_nav'], Q=metrics['Q'], J_nav=metrics['J_nav'],
                evaluation_execution='recomputed', numerical_evaluation_recomputed=True)
        row['review'] = info
    if reuse_record is not None:
        reused = reused_evaluation(reuse_record['receipt'], episode)
        if sha(reuse_record['path']) != reuse_record['sha256']:
            raise ValueError('supplemental reuse receipt changed during summary')
        row['supplemental_evaluation_reuse'] = dict(path=reuse_record['path'], sha256=reuse_record['sha256'],
            **{k: v for k, v in reused.items() if k not in ('metrics', 'evaluation')})
        if row['metrics'] is not None:
            if row['metrics'] != reused['metrics']:
                raise ValueError('original target score differs from its declared exact-input reuse')
        else:
            row['original_review'] = row['review']
            row['review'] = dict(available=True, path=reuse_record['path'], sha256=reuse_record['sha256'],
                status='experiment_endpoint_evaluation_reused', saved_policy_replay_verified=True,
                evaluation_available=True, evaluation=reused['evaluation'],
                evaluation_execution='reused', numerical_evaluation_recomputed=False,
                source_run_id=reused['source_run_id'], target_independent_replay_verified=True)
            metrics = reused['metrics']
            row.update(metrics=metrics, C_nav=metrics['C_nav'], Q=metrics['Q'], J_nav=metrics['J_nav'],
                evaluation_execution='reused', numerical_evaluation_recomputed=False)
    return row


def summarize(protocol_path, *, reviews_root=None, reuse_root=None):
    from env.development_sensor_v41 import runtime_counts_v41
    from nso.semantic_scene_experiment import PhaseLedger, validate_protocol
    before = runtime_counts_v41()
    protocol_path = Path(protocol_path).resolve()
    protocol, protocol_sha = read(protocol_path), sha(protocol_path)
    validate_protocol(protocol)
    if protocol['status'] != 'frozen' or len(protocol['slots']) != 96:
        raise ValueError('the frozen corrected 96-slot matrix is required')
    method_declaration = retained_method_correction(protocol) if protocol.get('method_correction') else None
    declaration = (method_declaration['constructor_correction'] if method_declaration else retained_correction(protocol))
    accounting = (method_declaration['accounting'] if method_declaration else dict(
        retained_primary_worlds=0, current_matrix_world_ceiling=96, total_primary_world_ceiling=96,
        prior_constructor_failure_attempts=1, total_attempt_ceiling=97, old_completed_attempts=0,
        retired_unstarted_slots=95, old_results_included_in_current_core_comparisons=False))
    ledger = PhaseLedger(protocol, protocol_sha)
    ledger.check_existing()
    ledger_sha = None
    entries = []
    if ledger.path.exists():
        payload = ledger.path.read_bytes()
        import hashlib
        ledger_sha = hashlib.sha256(payload).hexdigest()
        saved = json.loads(payload)
        if (saved.get('schema') != 'semantic.phase_start_ledger.v1'
                or saved.get('phase_id') != protocol['phase_id']
                or any(saved.get(k) != v for k, v in ledger.binding.items())):
            raise ValueError('current finite ledger/protocol binding mismatch')
        entries = saved['entries']
    counts = Counter(e['run_id'] for e in entries)
    if any(n != 1 for n in counts.values()) or not set(counts) <= set(protocol['slots']):
        raise ValueError('duplicate/undeclared reservation')
    if entries and any(e['metadata']['source_sha256'] != entries[0]['metadata']['source_sha256'] for e in entries):
        raise ValueError('source closure differs across frozen phase')
    if (method_declaration and entries
            and any(e['metadata']['source_sha256'] != protocol['execution_source_sha256'] for e in entries)):
        raise ValueError('new phase reservation differs from its declared corrected execution sources')
    if len(entries)+accounting['old_completed_attempts']+1 > accounting['total_attempt_ceiling']:
        raise ValueError('cumulative attempt ceiling exceeded')
    reviews_root = Path(reviews_root).resolve() if reviews_root else ledger.path.parent/'reviews'
    reuse_root = Path(reuse_root).resolve() if reuse_root else ledger.path.parent/'evaluation_reuse'
    reuse_records = reuse_index(reuse_root, protocol['slots'])
    mapping = {e['run_id']: e for e in entries}
    rows = []
    for run_id in protocol['slots']:
        try:
            row = summarize_slot(protocol, protocol_sha, run_id, mapping.get(run_id), reviews_root,
                                 reuse_records.get(run_id))
            if run_id in reuse_records and row['state'] != 'complete_episode':
                raise ValueError('supplemental receipt exists without a completed target episode')
        except Exception as exc:
            slot = protocol['slots'][run_id]
            parent, condition = slot['asset_id'].split('__')
            row = dict(run_id=run_id, parent_id=parent, condition=condition, **slot,
                state='integrity_error', metrics=None, C_nav=None, Q=None, J_nav=None,
                evaluation_execution=None, numerical_evaluation_recomputed=None,
                complete_episode=False, terminal_attempt=mapping.get(run_id, {}).get('status') not in (None, 'reserved_before_factory'),
                error=dict(type=type(exc).__name__, message=str(exc)))
        rows.append(row)
    gates = protocol['development_gates']
    comparisons = [core_comparison(protocol, rows, name='semantic_information', condition='nominal_relationship',
        baseline='G', relative_minimum=gates['semantic_relative_J_vs_strong_G_minimum']),
        core_comparison(protocol, rows, name='candidate_increment', condition='structure_relationship_shift',
            baseline='bayes_semantic', relative_minimum=gates['candidate_relative_J_vs_ordinary_Bayes_on_primary_shift_group_minimum'],
            require_parent_consistency=True),
        core_comparison(protocol, rows, name='reliable_noninferiority', condition='nominal_relationship',
            baseline='bayes_semantic', relative_minimum=-gates['reliable_group_relative_J_degradation_maximum'])]
    pairs = []
    # Compare available action chains under identical physical/seed/budget/tie
    # settings, including core and ablations; method differences are descriptive.
    for index, left in enumerate(rows):
        if not left['complete_episode']:
            continue
        for right in rows[index+1:]:
            if (right['complete_episode'] and left['method'] != right['method']
                    and all(left[k] == right[k] for k in ('asset_id', 'budget', 'noise_seed', 'tie_rule', 'tie_seed'))):
                pairs.append(pair_summary(left, right))
    blocks = {}
    by_id = {r['run_id']: r for r in rows}
    for name, block in protocol['analysis_blocks'].items():
        selected = [by_id[k] for k in block['run_ids']]
        state_counts = Counter(r['state'] for r in selected)
        blocks[name] = dict(declared_slots=len(selected), states={k: state_counts[k] for k in STATES},
            reviewed_metric_endpoints=sum(r['metrics'] is not None for r in selected),
            independent_declared_parents=len({r['parent_id'] for r in selected}))
    for row in rows:
        for key in ('_actions', '_poses', '_decisions'):
            row.pop(key, None)
    after = runtime_counts_v41()
    if before != after or sha(protocol_path) != protocol_sha:
        raise ValueError('read-only summary runtime or frozen protocol changed')
    state_counts = Counter(r['state'] for r in rows)
    summary_sources = [str(Path(__file__).resolve().relative_to(ROOT)),
                       'scripts/summarize_semantic_integration.py']
    if reuse_records or (method_declaration and any(
            r['evaluation_execution'] == 'reused' for r in method_declaration['retained_negative_episodes'])):
        summary_sources.append('scripts/reuse_semantic_scene_endpoint_evaluation.py')
    return dict(schema='semantic.scene_matrix.progress_summary.v1',
        protocol_path=str(protocol_path), protocol_sha256=protocol_sha, phase_id=protocol['phase_id'],
        source_sha256={name: sha(ROOT/name) for name in summary_sources},
        ledger_path=str(ledger.path), ledger_snapshot_sha256=ledger_sha,
        ledger_changed_during_summary=ledger_sha != (sha(ledger.path) if ledger.path.exists() else None),
        reviews_root=str(reviews_root), reuse_root=str(reuse_root), declared_current_slots=96,
        states={k: state_counts[k] for k in STATES}, current_reserved_attempts=len(entries),
        total_attempts_including_retained_constructor_failure=len(entries)+accounting['old_completed_attempts']+1,
        cumulative_attempt_accounting=accounting,
        confirmed_terminal_worlds=sum(e.get('world_created') is True for e in entries if e['status'] != 'reserved_before_factory'),
        confirmed_terminal_worlds_all_versions=accounting['retained_primary_worlds']+sum(
            e.get('world_created') is True for e in entries if e['status'] != 'reserved_before_factory'),
        running_world_count_unknown=state_counts['reserved_or_running'] > 0,
        reviewed_metric_endpoints=sum(r['metrics'] is not None for r in rows),
        numerical_evaluation_sources=dict(
            recomputed=sum(r.get('evaluation_execution') == 'recomputed' for r in rows),
            reused=sum(r.get('evaluation_execution') == 'reused' for r in rows),
            supplemental_receipts=len(reuse_records)),
        all_declared_attempts_terminal=all(r['terminal_attempt'] for r in rows),
        all_declared_endpoints_reviewed=all(r['metrics'] is not None for r in rows),
        retained_constructor_correction=declaration, retained_method_correction=method_declaration,
        blocks=blocks, episodes=rows, core_comparison_phase_id=protocol['phase_id'],
        retired_episode_rows_in_current_core=0,
        matched_trajectory_comparisons=pairs, core_comparisons=comparisons,
        all_core_numeric_thresholds_passed=(all(c['status'] == 'numeric_thresholds_passed' for c in comparisons)
            if all(c['complete_reviewed_parent_pairs'] == c['required_parent_pairs'] for c in comparisons) else None),
        full_development_gate_pass=None, final_route_decision=None, attribution_review_required=True,
        analysis_rules=protocol['analysis_rules'], missing_metrics_imputed=False,
        new_worlds=0, new_sensor_packets=0, new_planner_executions=0, new_tsdf_integrations=0,
        new_numeric_surface_evaluations=0, runtime_before=before, runtime_after=after,
        limits=['This is finite development progress; no independent held-out confirmation.',
            'Metrics require an original complete review or a separately reverified exact-input reuse chain with independent target replay; missing values remain null.',
            'Reused endpoint numbers are labeled reused and do not claim another numerical evaluator execution.',
            'Retired negative episodes retain their own archived executed sources and are never mixed into current-phase core comparisons.',
            'Same-layout conditions/budgets/ties do not increase independent parent count.',
            'Positive total EVI plus cross-future enablement does not identify net cross-instance value.',
            'Action divergence or positive forecasts alone do not establish measured semantic benefit.',
            'Numeric core gates cannot settle noinfo/tie/error/ablation attribution or issue a final route decision.',
            'Unreviewed manifest pins are observed snapshots checked against durable result/source/input bindings, not independent replay evidence.',
            'P03 facility-0 fixed-view failures remain; the static 192-frame check is not a paid planner trajectory.'])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--protocol', type=Path, default=DEFAULT_PROTOCOL)
    parser.add_argument('--reviews-root', type=Path)
    parser.add_argument('--reuse-root', type=Path)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    result = summarize(args.protocol, reviews_root=args.reviews_root, reuse_root=args.reuse_root)
    encoded = json.dumps(result, indent=2, sort_keys=True, allow_nan=False)+'\n'
    if args.output:
        output = args.output.resolve()
        # Evidence trees and prior outputs must remain immutable.
        if output.is_relative_to(contained(read(args.protocol)['output_relative_path'])):
            raise ValueError('summary must remain outside immutable episode archives')
        output.parent.mkdir(parents=True, exist_ok=True)
        with output.open('x') as stream:
            stream.write(encoded)
        print(json.dumps(dict(output=str(output), sha256=sha(output), states=result['states'],
            reviewed_metric_endpoints=result['reviewed_metric_endpoints'], final_route_decision=None), sort_keys=True))
    else:
        print(encoded, end='')


if __name__ == '__main__':
    main()
