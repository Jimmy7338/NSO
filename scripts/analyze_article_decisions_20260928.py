#!/usr/bin/env python3
"""Audit reviewed, saved decisions; never run a world, planner or reconstruction.

The input summary is pinned. Every consumed step/result is checked against its
episode manifest. Novelty distances are recomputed only to attribute an existing
frontend gate; this does not replay a controller or estimate improved quality.
"""
from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from nso.exact_support_distance import exact_nearest_distances

SUMMARY = ROOT / 'audit_results/semantic_development_acquisition_20260923/progress_summary_010.json'
SUMMARY_SHA = 'ce26d8cc42464af8ebb981199dd48200e51d8c7588175f07c761624acbadfaff'
DEFAULT_OUTPUT = ROOT / 'audit_results/article_stage_20260928/diagnostics/decision_audit'
SOURCE_PATHS = (
    'nso/controller_semantic_mechanism.py', 'nso/controller_v43.py',
    'nso/semantic_reliability.py', 'nso/observed_instances_local.py',
    'nso/observed_instances_v41.py', 'nso/exact_support_distance.py',
    'nso/cpu_four_modules_v35.py', 'nso/online_planner_v35.py',
    'nso/observation_belief_v35.py',
    'scripts/analyze_article_decisions_20260928.py',
)
# These are the frozen local frontend defaults, not tuned diagnostic thresholds.
MAX_SUPPORT = 4096
MAX_EVIDENCE = 32
VOXEL = .05
MIN_POINTS = 4
MIN_FRACTION = .05


def sha(data):
    return hashlib.sha256(data).hexdigest()


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False)


def target(value):
    if not value:
        return ''
    return f"{value['node']}:{value['heading']}"


def first_set(row, name, step, condition=True):
    if condition and row.get(name) is None:
        row[name] = step


def class_qualified(instance):
    label = instance.get('observed_class')
    return (label is not None and not instance['association_uncertain']
            and instance['distinct_class_supports'].get(label, 0) >= 2)


def csv_write(path, rows):
    fields = sorted(set().union(*(r.keys() for r in rows))) if rows else ['empty']
    with path.open('w', newline='', encoding='utf-8') as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def json_write(path, value):
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + '\n')


class VerifiedEpisode:
    def __init__(self, row):
        self.root = Path(row['episode_root'])
        data = (self.root / 'artifact_manifest.json').read_bytes()
        assert sha(data) == row['episode_manifest_sha256'], self.root
        self.manifest = json.loads(data)
        self.consumed = []

    def read(self, relative):
        data = (self.root / relative).read_bytes()
        recorded = self.manifest['files'][relative]
        digest = sha(data)
        assert digest == recorded['sha256'] and len(data) == recorded['bytes'], relative
        self.consumed.append(dict(path=relative, sha256=digest, bytes=len(data)))
        return json.loads(gzip.decompress(data) if relative.endswith('.gz') else data)


def gate_diagnosis(previous, accepted):
    """Reproduce pre-insertion novelty; each failure flag is non-exclusive."""
    points = np.asarray(accepted['points_world_m'], dtype=np.float64).reshape(-1, 3)
    support = np.asarray(previous.get('support_points_world_m', []), dtype=np.float64).reshape(-1, 3)
    novelty = exact_nearest_distances(points, support) > VOXEL
    duplicate = bool(accepted['duplicate_measurement'])
    if duplicate:
        novelty[:] = False
    cells = np.floor(points[novelty] / VOXEL).astype(np.int64)
    novel_voxels = len({tuple(cell) for cell in cells})
    assert novel_voxels == accepted['novel_support_voxels'], 'Saved novelty mismatch'
    fraction = float(novelty.mean()) if len(novelty) else 0.
    frame_cap = previous.get('novel_support_frames', 0) >= MAX_EVIDENCE
    full = len(support) >= MAX_SUPPORT
    low_points = novel_voxels < MIN_POINTS
    low_fraction = fraction < MIN_FRACTION
    no_addition = bool(accepted['no_new_support'])
    eligible = not (frame_cap or low_points or low_fraction or no_addition)
    assert eligible == accepted['geometry_feedback_eligible'], 'Saved gate mismatch'
    cache_only = full and no_addition and not (duplicate or frame_cap or low_points or low_fraction)
    return dict(previous_support_points=len(support),
                previous_novel_support_frames=previous.get('novel_support_frames', 0),
                recomputed_novel_fraction=fraction, novel_support_voxels=novel_voxels,
                cache_full_before_observation=full, evidence_frame_cap=frame_cap,
                below_minimum_novel_points=low_points, below_minimum_novel_fraction=low_fraction,
                duplicate_measurement=duplicate, no_new_support=no_addition,
                cache_saturation_sole_failed_gate=cache_only)


def compact_profile(selection):
    if not selection:
        return None
    options = selection.get('direct_options', []) + selection.get('diagnostic_options', [])
    scores = {}
    per_target = {}
    for row in options:
        score = row.get('score')
        if score is None:
            continue
        key = '|'.join((row['kind'], target(row['target']), row.get('instance_id', '')))
        scores[key] = float(score)
        state = target(row['target'])
        per_target[state] = max(per_target.get(state, -float('inf')), float(score))
    selected = selection.get('selected') or {}
    chosen_target = target(selected.get('target'))
    alternatives = [s for t, s in per_target.items() if t != chosen_target]
    margin = (float(selected['score']) - max(alternatives)
              if selected.get('score') is not None and alternatives else None)
    gains = {f['instance_id']: [dict(view_id=c['view_id'],
        gain=c['structure_new_surface_area_m2'], fallback=c['fallback'],
        fallback_reason=c.get('fallback_reason'), repeated_view_excluded=c['repeated_view_excluded'])
        for c in f['candidates']] for f in selection.get('forecasts', [])}
    return dict(scores=scores, selected_target=chosen_target,
                selected_kind=selected.get('kind'), selected_score=selected.get('score'),
                distinct_target_margin=margin, forecasts=gains,
                direct_targets={target(c['target']) for c in selection.get('direct_options', [])})


def analyze_episode(row, decision_rows, feedback_rows, milestones):
    episode = VerifiedEpisode(row)
    result = episode.read('result.json')
    assert result['status'] == 'controller_stop', row['run_id']
    step_files = sorted(p for p in episode.manifest['files'] if p.startswith('steps/') and p.endswith('.json.gz'))
    assert len(step_files) == row['executed_paid_actions'] + 1, row['run_id']
    counters = Counter()
    previous = {}
    instance_rows = {}
    history = []
    poses = []
    first = {}
    run = row['run_id']
    initial_count = len(feedback_rows)
    for index, relative in enumerate(step_files):
        step = episode.read(relative)
        account = step['accounting']
        evidence, decision = step['controller_evidence'], step['decision']
        assert account['paid_step'] == index
        pose = step['mapper']['world_from_camera']
        poses.append(pose)
        if index:
            counters[account['action']] += 1
        association = evidence['association']
        current = {x['instance_id']: x for x in association['instances']}
        accepted = {x['instance_id']: x for x in association['accepted']}
        belief_rows = evidence.get('structure_belief', {}).get('instances', [])
        beliefs = {x['instance_id']: x for x in belief_rows}
        residuals = {x['instance_id']: x for x in evidence['observed_residual']['results']}
        feedback = {x['instance_id']: x for x in evidence['geometry_feedback']}
        for key, instance in current.items():
            milestone = instance_rows.setdefault(key, dict(run_id=run, instance_id=key,
                parent_id=row['parent_id'], condition=row['condition'], method=row['method'],
                first_seen=index))
            first_set(milestone, 'first_class_seen', index, instance['observed_class'] is not None)
            first_set(milestone, 'first_qualified_class', index, class_qualified(instance))
            first_set(milestone, 'first_cache_full', index, len(instance['support_points_world_m']) >= MAX_SUPPORT)
            residual = residuals.get(key, {})
            first_set(milestone, 'first_accepted_residual', index, residual.get('accepted', False))
            first_set(milestone, 'first_informative_residual', index, residual.get('informative', False))
            first_set(milestone, 'first_applied_feedback', index, feedback.get(key, {}).get('applied', False))
            b = beliefs.get(key, {})
            rho = b.get('rho')
            first_set(milestone, 'first_peer_adjusted_rho', index, rho is not None and abs(rho - .5) > 1e-12)
            milestone['last_observed_class'] = instance['observed_class']
            milestone['last_support_points'] = len(instance['support_points_world_m'])
            milestone['last_novel_support_frames'] = instance['novel_support_frames']
            # Association public state is pre-feedback in this paid frame.
            milestone['last_frontend_geometry_frames_before_current_update'] = instance['geometric_feedback_frames']
        for key, residual in residuals.items():
            if not residual.get('accepted'):
                continue
            assert key in accepted and key in feedback
            f = feedback[key]
            entry = dict(run_id=run, parent_id=row['parent_id'], condition=row['condition'], method=row['method'],
                paid_step=index, remaining_budget=row['budget'] - index, instance_id=key,
                informative=bool(residual['informative']), applied=bool(f['applied']), reason=f['reason'],
                association_geometry_feedback_eligible=bool(accepted[key]['geometry_feedback_eligible']),
                log_likelihood_span=float(np.ptp(residual['log_likelihoods'])),
                log_likelihoods=canonical(residual['log_likelihoods']),
                support_sha256=accepted[key]['support_sha256'])
            entry.update(gate_diagnosis(previous.get(key, {}), accepted[key]))
            assert f['applied'] == accepted[key]['geometry_feedback_eligible']
            feedback_rows.append(entry)
            counters['accepted_residuals'] += 1
            counters['informative_residuals'] += bool(residual['informative'])
            counters['applied_feedback'] += bool(f['applied'])
            counters['cache_only_refused'] += entry['cache_saturation_sole_failed_gate'] and not f['applied']
            counters['informative_cache_only_refused'] += entry['cache_saturation_sole_failed_gate'] and not f['applied'] and residual['informative']
            first_set(instance_rows[key], 'first_cache_only_refused_residual', index, entry['cache_saturation_sole_failed_gate'])
        qualified = sum(class_qualified(x) for x in current.values())
        semantic = sum(x.get('semantic_conditioning_used', False) for x in belief_rows)
        rho_adjusted = sum(x.get('rho') is not None and abs(x['rho'] - .5) > 1e-12 for x in belief_rows)
        first_set(first, 'first_class', index, qualified > 0)
        first_set(first, 'first_peer_rho_change', index, rho_adjusted > 0)
        selection = decision.get('global_selection')
        profile = compact_profile(selection)
        selected = (selection or {}).get('selected') or {}
        direct = (selection or {}).get('direct_options', [])
        diagnostics = (selection or {}).get('diagnostic_options', [])
        counters['global_replans_with_receipt'] += selection is not None
        counters['diagnostic_options'] += len(diagnostics)
        counters['positive_evi_options'] += sum(x.get('evi', 0.) > 1e-12 for x in diagnostics)
        counters['selected_diagnostics'] += selected.get('kind') == 'diagnose_then_observe'
        counters['selected_initializations'] += selected.get('kind') == 'measurement_initialization'
        first_set(first, 'first_positive_evi', index, any(x.get('evi', 0.) > 1e-12 for x in diagnostics))
        first_set(first, 'first_selected_diagnostic', index, selected.get('kind') == 'diagnose_then_observe')
        routing = decision.get('routing', {})
        first_set(first, 'first_return_action', index, routing.get('reason') == 'return')
        return_cost = routing.get('return_cost_after_action')
        remaining = row['budget'] - index
        route_margin = remaining - 1 - return_cost if return_cost is not None else None
        if route_margin is not None:
            assert route_margin >= 0, 'Saved return reserve violated'
        if selected.get('total_cost') is not None:
            assert selected['total_cost'] <= remaining
        decision_rows.append(dict(run_id=run, paid_step=index, remaining_budget=remaining,
            actual_action=account['action'], next_action=decision['action'], reason=decision.get('reason'),
            routing_reason=routing.get('reason'), global_replanned=bool(decision.get('global_replanned')),
            selected_kind=selected.get('kind'), selected_target=target(selected.get('target')),
            committed_target=target(decision.get('macro_target')), selected_score=selected.get('score'),
            selected_evi=selected.get('evi'), selected_total_cost=selected.get('total_cost'),
            direct_max=max((x['score'] for x in direct), default=None),
            diagnostic_max=max((x['score'] for x in diagnostics), default=None),
            positive_evi_options=sum(x.get('evi', 0.) > 1e-12 for x in diagnostics),
            distinct_target_margin=(profile or {}).get('distinct_target_margin'),
            qualified_instances=qualified, semantic_conditioned_instances=semantic,
            peer_rho_adjusted_instances=rho_adjusted,
            no_progress_paid_actions=evidence['no_progress_paid_actions'],
            new_occupancy_cells=step['mapper']['newly_known_cells'],
            return_cost_after_action=return_cost, return_reserve_margin_after_action=route_margin))
        history.append(dict(paid_step=index, actual_action=account['action'], pose=pose,
            next_action=decision['action'], committed_target=target(decision.get('macro_target')),
            profile=profile, beliefs=beliefs))
        previous = current
    for milestone in instance_rows.values():
        milestone['budget_at_first_class'] = (row['budget'] - milestone['first_qualified_class']
            if milestone.get('first_qualified_class') is not None else None)
        milestones.append(milestone)
    feedback_count = len(feedback_rows) - initial_count
    assert feedback_count == row['counts'].get('accepted_residuals', 0)
    assert counters['applied_feedback'] == row['counts'].get('frontend_geometry_updates_applied', 0)
    xyz = np.array(poses, dtype=float)[:, :3, 3]
    length = float(np.linalg.norm(np.diff(xyz[:, :2], axis=0), axis=1).sum())
    episode_row = dict(run_id=run, parent_id=row['parent_id'], condition=row['condition'],
        method=row['method'], budget=row['budget'], executed_paid_actions=row['executed_paid_actions'],
        saved_packets=len(history), C_nav=row['C_nav'], Q=row['Q'], J_nav=row['J_nav'],
        measured_xy_path_length_m=length, turns=counters['turn_left'] + counters['turn_right'],
        observes=counters['observe'], forwards=counters['forward'],
        final_reason=history[-1]['next_action'], returned_xy_and_yaw=row['returned_xy_and_yaw'],
        final_unused_budget=row['budget'] - row['executed_paid_actions'],
        planning_s=result['timings']['planning_s'], evidence_s=result['timings']['evidence_s'],
        action_sequence_sha256=row['action_sequence_sha256'], pose_sequence_sha256=row['pose_sequence_sha256'],
        mesh_file_sha256=row['mesh_file_sha256'], **first)
    for name in ('accepted_residuals', 'informative_residuals', 'applied_feedback', 'cache_only_refused',
                 'informative_cache_only_refused', 'global_replans_with_receipt', 'diagnostic_options',
                 'positive_evi_options', 'selected_diagnostics', 'selected_initializations'):
        episode_row[name] = counters[name]
    for name in ('planner_nonzero_geometry_evidence_instance_frames', 'peer_adjusted_reliability_instance_frames',
                 'view_forecast_fallbacks', 'view_forecast_nonfallbacks', 'positive_direct_options'):
        episode_row[name] = row['counts'].get(name, 0)
    provenance = dict(run_id=run, manifest_sha256=row['episode_manifest_sha256'],
        verified_files=episode.consumed, input_bytes=sum(x['bytes'] for x in episode.consumed),
        consumed_files_digest=sha(canonical(episode.consumed).encode()))
    return episode_row, history, provenance


def compare_pair(left, right, histories, replans, transfers):
    """Compare score profiles only while actual sensor/action history is common."""
    a, b = histories[left['run_id']], histories[right['run_id']]
    result = dict(parent_id=left['parent_id'], condition=left['condition'],
        method_a=left['method'], method_b=right['method'], run_a=left['run_id'], run_b=right['run_id'],
        delta_J_nav=left['J_nav'] - right['J_nav'], delta_Q=left['Q'] - right['Q'],
        delta_C_nav=left['C_nav'] - right['C_nav'],
        identical_full_actions=left['action_sequence_sha256'] == right['action_sequence_sha256'],
        identical_full_poses=left['pose_sequence_sha256'] == right['pose_sequence_sha256'],
        identical_final_mesh=left['mesh_file_sha256'] == right['mesh_file_sha256'],
        common_history_paid_frames=0, common_history_paired_replans=0,
        score_changed_replans=0, score_changed_same_selected_target=0,
        posterior_changed_paid_frames=0, max_common_candidate_score_difference=0.,
        distinct_common_candidate_sets=0)
    same_history = True
    for x, y in zip(a, b):
        i = x['paid_step']
        same_history = same_history and x['actual_action'] == y['actual_action'] and x['pose'] == y['pose']
        first_set(result, 'first_actual_history_difference', i, not same_history)
        first_set(result, 'first_next_action_difference', i, x['next_action'] != y['next_action'])
        first_set(result, 'first_committed_target_difference', i, x['committed_target'] != y['committed_target'])
        if not same_history:
            continue
        result['common_history_paid_frames'] += 1
        shared_keys = set(x['beliefs']) & set(y['beliefs'])
        belief_differences = {key: np.asarray(x['beliefs'][key]['structure_probabilities'])
            - np.asarray(y['beliefs'][key]['structure_probabilities']) for key in shared_keys}
        changed_instances = {key: delta for key, delta in belief_differences.items()
                             if np.abs(delta).max() > 1e-12}
        posterior_delta = max((float(np.abs(v).max()) for v in belief_differences.values()), default=0.)
        if posterior_delta > 1e-12:
            result['posterior_changed_paid_frames'] += 1
            first_set(result, 'first_posterior_difference', i)
        p, q = x['profile'], y['profile']
        if p is None or q is None:
            continue
        result['common_history_paired_replans'] += 1
        keys = set(p['scores']) & set(q['scores'])
        result['distinct_common_candidate_sets'] += set(p['scores']) != set(q['scores'])
        delta = max((abs(p['scores'][k] - q['scores'][k]) for k in keys), default=0.)
        max_key = max(sorted(keys), key=lambda k: abs(p['scores'][k] - q['scores'][k])) if keys else None
        replans.append(dict(run_a=left['run_id'], run_b=right['run_id'], paid_step=i,
            posterior_changed_instances=';'.join(sorted(changed_instances)),
            shared_scored_candidates=len(keys), same_scored_candidate_set=set(p['scores']) == set(q['scores']),
            selected_target_a=p['selected_target'], selected_target_b=q['selected_target'],
            selected_kind_a=p['selected_kind'], selected_kind_b=q['selected_kind'],
            selected_score_a=p['selected_score'], selected_score_b=q['selected_score'],
            distinct_target_margin_a=p['distinct_target_margin'], distinct_target_margin_b=q['distinct_target_margin'],
            max_score_difference=delta, max_difference_candidate=max_key,
            max_difference_candidate_score_a=p['scores'].get(max_key),
            max_difference_candidate_score_b=q['scores'].get(max_key)))
        if left['method'] == 'shared_semantic' and right['method'] == 'bayes_semantic':
            for key, belief_delta in changed_instances.items():
                f = p['forecasts'].get(key, [])
                feasible = [v for v in f if v['view_id'] in p['direct_targets']]
                transfers.append(dict(run_a=left['run_id'], run_b=right['run_id'], paid_step=i,
                    instance_id=key, rho=x['beliefs'][key]['rho'],
                    posterior_a=canonical(x['beliefs'][key]['structure_probabilities']),
                    posterior_b=canonical(y['beliefs'][key]['structure_probabilities']),
                    selected_kind=p['selected_kind'], forecasts=len(f),
                    fallback_forecasts=sum(v['fallback'] for v in f),
                    fallback_reasons=canonical(dict(Counter(v['fallback_reason'] for v in f if v['fallback']))),
                    repeated_view_exclusions=sum(v['repeated_view_excluded'] for v in f),
                    direct_feasible_forecasts=len(feasible),
                    nonzero_gain_forecasts=sum(any(abs(g) > 1e-12 for g in v['gain']) for v in f),
                    structure_dependent_forecasts=sum(np.ptp(v['gain']) > 1e-12 for v in f),
                    max_feasible_absolute_gain=max((max(abs(g) for g in v['gain']) for v in feasible), default=0.),
                    max_abs_posterior_times_feasible_gain=max((abs(float(belief_delta @ np.asarray(v['gain'])))
                                                            for v in feasible), default=0.)))
        result['max_common_candidate_score_difference'] = max(result['max_common_candidate_score_difference'], delta)
        if delta > 1e-12:
            result['score_changed_replans'] += 1
            result['score_changed_same_selected_target'] += p['selected_target'] == q['selected_target']
            first_set(result, 'first_common_candidate_score_difference', i)
        first_set(result, 'first_selected_target_difference_on_common_history', i,
                  p['selected_target'] != q['selected_target'])
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    raw = SUMMARY.read_bytes()
    assert sha(raw) == SUMMARY_SHA, 'Reviewed summary changed; do not broaden this audit implicitly'
    summary = json.loads(raw)
    selected = [x for x in summary['episodes'] if x.get('metrics') is not None]
    assert len(selected) == 58 and all(x['complete_episode'] for x in selected)
    source_hashes = {p: sha((ROOT / p).read_bytes()) for p in SOURCE_PATHS}
    decisions, feedback, milestones, episodes, provenance = [], [], [], [], []
    histories = {}
    for i, row in enumerate(selected, 1):
        episode, history, inputs = analyze_episode(row, decisions, feedback, milestones)
        episodes.append(episode)
        histories[row['run_id']] = history
        provenance.append(inputs)
        print(f"{i:02d}/58 {row['run_id']}: accepted={episode['accepted_residuals']} "
              f"applied={episode['applied_feedback']} cache-only={episode['cache_only_refused']}", flush=True)
    pairs, replans, transfers = [], [], []
    core = defaultdict(dict)
    for row in episodes:
        if row['run_id'].startswith('core_') and row['method'] in ('G', 'bayes_semantic', 'shared_semantic'):
            core[row['parent_id'], row['condition']][row['method']] = row
    for _, block in sorted(core.items()):
        for a, b in [('shared_semantic', 'G'), ('shared_semantic', 'bayes_semantic'), ('bayes_semantic', 'G')]:
            if a in block and b in block:
                pairs.append(compare_pair(block[a], block[b], histories, replans, transfers))
    assert sum(x['executed_paid_actions'] for x in episodes) == 6845
    assert len(decisions) == 6903
    gates = {key: sum(bool(x[key]) for x in feedback if not x['applied']) for key in (
        'cache_full_before_observation', 'evidence_frame_cap', 'below_minimum_novel_points',
        'below_minimum_novel_fraction', 'duplicate_measurement', 'no_new_support',
        'cache_saturation_sole_failed_gate')}
    witnessed = [x for x in feedback if x['cache_saturation_sole_failed_gate'] and x['informative']]
    first_witness = {}
    for row in witnessed:
        key = (row['parent_id'], row['condition'])
        # Prefer full semantic core episodes as readable examples.
        if row['run_id'].startswith('core_') and row['method'] == 'shared_semantic' and key not in first_witness:
            first_witness[key] = row
    report = dict(schema='article.saved_decision_diagnosis.v1',
        input_summary_sha256=SUMMARY_SHA, reviewed_episodes=len(episodes), saved_paid_actions=6845,
        saved_packets=6903, excluded_unreviewed_or_incomplete_slots=len(summary['episodes']) - len(selected),
        new_worlds=0, new_planner_executions=0, new_sensor_packets=0, new_tsdf_integrations=0,
        new_numeric_surface_evaluations=0, gate_distance_recomputations=len(feedback),
        gate_distance_definition='Original float64 support novelty, audited using exact nearest-support helper',
        accepted_residuals=len(feedback), informative_residuals=sum(x['informative'] for x in feedback),
        applied_feedback=sum(x['applied'] for x in feedback), refused_feedback=sum(not x['applied'] for x in feedback),
        refusal_gate_flags_nonexclusive=gates,
        informative_cache_only_refusals=len(witnessed),
        informative_cache_only_episodes=len({x['run_id'] for x in witnessed}),
        informative_cache_only_parent_ids=sorted({x['parent_id'] for x in witnessed}),
        complete_core_S_B_pairs=sum(p['method_a'] == 'shared_semantic' and p['method_b'] == 'bayes_semantic' for p in pairs),
        S_B_same_action_pose_mesh_pairs=sum(p['method_a'] == 'shared_semantic' and p['method_b'] == 'bayes_semantic'
            and p['identical_full_actions'] and p['identical_full_poses'] and p['identical_final_mesh'] for p in pairs),
        S_B_common_history_score_changed_replans=sum(p['score_changed_replans'] for p in pairs
            if p['method_a'] == 'shared_semantic' and p['method_b'] == 'bayes_semantic'),
        S_B_changed_instance_replan_records=len(transfers),
        S_B_changed_instance_replans_with_nonzero_feasible_gain=sum(t['max_feasible_absolute_gain'] > 1e-12 for t in transfers),
        S_B_changed_instance_replans_with_posterior_sensitive_feasible_gain=sum(t['max_abs_posterior_times_feasible_gain'] > 1e-12 for t in transfers),
        core_first_informative_cache_only_witnesses=list(first_witness.values()),
        limits=[
            'Counts are saved frames/events across repeated methods, not independent experimental units.',
            'Removing one gate does not predict changed action, quality, or a semantic advantage.',
            'Positive total EVI with a cross-instance flag does not identify net cross-instance information value.',
            'Scores are planning proxies; J_nav/Q/C_nav are reused reviewed terminal measurements.',
            'No online hard C_nav qualification gate exists in this scene controller; return reserve is checked separately.',
            'Pair score comparisons stop when actual paid action/pose histories diverge.',
        ])
    assert source_hashes == {p: sha((ROOT / p).read_bytes()) for p in SOURCE_PATHS}, 'Source changed during audit'
    assert sha(SUMMARY.read_bytes()) == SUMMARY_SHA
    args.output.mkdir(parents=True, exist_ok=False)
    for name, rows in [('episode_diagnostics', episodes), ('decision_steps', decisions),
                       ('feedback_events', feedback), ('instance_milestones', milestones), ('paired_decisions', pairs),
                       ('paired_replans', replans), ('transfer_forecasts', transfers)]:
        csv_write(args.output / (name + '.csv'), rows)
    json_write(args.output / 'summary.json', report)
    json_write(args.output / 'provenance.json', dict(schema='article.saved_decision_provenance.v1',
        input_summary=str(SUMMARY.relative_to(ROOT)), input_summary_sha256=SUMMARY_SHA,
        sources_sha256=source_hashes, consumed_episodes=provenance,
        thresholds=dict(maximum_support_points=MAX_SUPPORT, maximum_evidence_frames=MAX_EVIDENCE,
                        voxel_size_m=VOXEL, minimum_novel_points=MIN_POINTS, minimum_novel_fraction=MIN_FRACTION),
        source_and_summary_unchanged_at_end=True))
    outputs = {p.name: dict(bytes=p.stat().st_size, sha256=sha(p.read_bytes())) for p in sorted(args.output.iterdir())}
    json_write(args.output / 'artifact_manifest.json', dict(schema='article.diagnostic_artifacts.v1', files=outputs))
    total = sum(p.stat().st_size for p in args.output.iterdir())
    assert total < 10 * 1024 ** 2, 'Diagnostic output exceeds delegated storage bound'
    print(canonical(dict(output=str(args.output), bytes=total, summary=report)))


if __name__ == '__main__':
    main()
