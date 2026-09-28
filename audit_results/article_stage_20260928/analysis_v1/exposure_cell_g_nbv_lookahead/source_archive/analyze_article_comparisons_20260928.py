#!/usr/bin/env python3
"""Read-only comparison of sealed, independently reviewed article episodes.

Never opens live reserved episodes, replays policies, integrates meshes, or
remeasures quality. Every invocation writes a new immutable analysis snapshot.
"""
from __future__ import annotations
import argparse
from collections import Counter, defaultdict
import csv
from datetime import datetime, timezone
import gzip
import hashlib
import io
from itertools import combinations
import json
import math
from pathlib import Path
import sys
import uuid
import zipfile

ROOT = Path(__file__).resolve().parents[1]
STAGE = ROOT/'audit_results/article_stage_20260928'
METHODS = ('S', 'B', 'G', 'NBV')
METRICS = ('C_nav', 'P', 'R', 'F1', 'J_nav')
COSTS = ('paid_actions', 'path_length_m', 'turns', 'explicit_observe_actions',
         'observation_frames', 'planning_s', 'evidence_s', 'acquisition_s',
         'mapping_s', 'elapsed_episode_s')
TOL = 1e-12


def digest(data):
    return hashlib.sha256(data).hexdigest()


def canonical(data):
    return (json.dumps(data, sort_keys=True, indent=2, allow_nan=False)+'\n').encode()


def safe(base, name):
    path = (base/name).resolve()
    if not path.is_relative_to(base.resolve()):
        raise ValueError('input path escapes containing directory: '+str(name))
    return path


class Inputs:
    def __init__(self):
        self.pins = {}
        self.captured = {}

    def read(self, path, *, pin=None, mutable=False):
        path = Path(path).resolve()
        data = path.read_bytes()
        current = digest(data)
        if pin is not None and current != pin:
            raise ValueError('input SHA-256 differs: '+str(path))
        key = str(path.relative_to(ROOT)) if path.is_relative_to(ROOT) else str(path)
        row = dict(sha256=current, bytes=len(data), mutable_snapshot=mutable)
        if key in self.pins and self.pins[key] != row:
            raise ValueError('input changed within snapshot: '+key)
        self.pins[key] = row
        if mutable:
            self.captured[key] = data
        return data

    def json(self, path, **kwargs):
        return json.loads(self.read(path, **kwargs))

    def unchanged(self):
        for name, row in self.pins.items():
            if not row['mutable_snapshot'] and digest((ROOT/name).read_bytes()) != row['sha256']:
                raise ValueError('sealed input changed during analysis: '+name)


def status_without_review(entry):
    if entry is None:
        return 'unstarted'
    return 'running_reserved' if entry['status'] == 'reserved' else 'awaiting_independent_review'


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sealed_read(inputs, episode, manifest, name):
    row = manifest['files'][name]
    payload = inputs.read(safe(episode, name), pin=row['sha256'])
    require(len(payload) == row['bytes'], 'artifact byte count differs: '+name)
    return payload


def sensor_content_sha(rgbd_bytes, scan_bytes):
    """Compare all saved sensor arrays; exclude only run-namespaced frame ID.

    NPZ ZIP timestamps/compression and frame_id are not physical sensor data.
    NPY entries retain dtype, shape and exact payload, including pose and time.
    """
    h = hashlib.sha256()
    for kind, payload, expected in (
        ('rgbd', rgbd_bytes, {'frame_id.npy','paid_step.npy','rgb.npy','depth_m.npy','intrinsic.npy','world_from_camera.npy'}),
        ('scan', scan_bytes, {'timestamp_s.npy','ranges_m.npy','angle_min_rad.npy','angle_increment_rad.npy','range_max_m.npy','world_from_laser.npy'})):
        with zipfile.ZipFile(io.BytesIO(payload)) as archive:
            require(set(archive.namelist()) == expected and len(archive.namelist()) == len(expected), 'unknown sensor-array inventory')
            for name in sorted(expected-({'frame_id.npy'} if kind == 'rgbd' else set())):
                data = archive.read(name)
                h.update(canonical([kind, name, len(data)])); h.update(data)
    return h.hexdigest()


def read_episode(inputs, episode, manifest):
    result = json.loads(sealed_read(inputs, episode, manifest, 'result.json'))
    compact = {}
    mechanism = Counter()
    fallback = Counter()
    replans = []
    belief_trace = []
    association_transitions = []
    association_events = []
    previous_association = {}
    uncertain_counts, conflict_counts = Counter(), Counter()
    anchors, anchor_changes = {}, []
    first = {}
    instance_first = {}
    for name in sorted(manifest['files']):
        if not (name.startswith('steps/') and name.endswith('.json.gz')):
            continue
        row = json.loads(gzip.decompress(sealed_read(inputs, episode, manifest, name)))
        evidence, decision = row['controller_evidence'], row['decision']
        step = evidence['paid_step']
        content_sha = sensor_content_sha(
            sealed_read(inputs, episode, manifest, f'packets/{step:03d}_rgbd.npz'),
            sealed_read(inputs, episode, manifest, f'packets/{step:03d}_scan.npz'))
        beliefs = {x['instance_id']: dict(prob=x['structure_probabilities'], rho=x.get('rho'),
                    observed_class=x.get('observed_class'), peers=x.get('peer_instance_ids', []))
                   for x in evidence['structure_belief']['instances']}
        associations = {x['instance_id']: x for x in evidence['association'].get('instances', [])}
        association = evidence['association']
        for rejected in association.get('rejected', []):
            if rejected['reason'] in ('overlapping_local_support_proposals', 'multiple_markers_within_local_support',
                                      'ambiguous_geometric_association', 'multiple_components_claim_instance'):
                association_events.append(dict(paid_step=step, **rejected))
        for key, assoc in associations.items():
            uncertain_counts[key] += int(assoc['association_uncertain'])
            conflict_counts[key] += int(assoc['class_conflict'])
            if key in anchors and anchors[key] != assoc['anchor_world_m']:
                anchor_changes.append(dict(paid_step=step, instance_id=key, previous=anchors[key], current=assoc['anchor_world_m']))
            anchors[key] = assoc['anchor_world_m']
            state = (assoc['association_uncertain'], assoc['semantic_conditioning_used'], assoc['observed_class'])
            if previous_association.get(key) != state:
                points = assoc['support_points_world_m']
                association_transitions.append(dict(paid_step=step, instance_id=key,
                    association_uncertain=state[0], semantic_conditioning_used=state[1], measured_class=state[2],
                    distinct_class_supports=assoc['distinct_class_supports'], marker_anchor_world_m=assoc['anchor_world_m'],
                    support_sha256=assoc['support_sha256'], support_points=len(points),
                    near_ground_support_points_z_below_015m=sum(p[2] < .15 for p in points),
                    support_bbox_m=[[min(p[i] for p in points) for i in range(3)], [max(p[i] for p in points) for i in range(3)]] if points else None,
                    current_explicit_rejections=[r for r in association['rejected'] if key in r.get('candidate_instances', [])],
                    current_all_rejection_reason_counts=dict(Counter(r['reason'] for r in association['rejected'])),
                    current_local_proposal_audit=association.get('local_proposal_audit', []),
                    uniquely_accepted_current_measurements=[{n:r[n] for n in ('association','support_sha256','component_pixels','novel_support_voxels','geometry_feedback_eligible')}
                        for r in association['accepted'] if r['instance_id'] == key]))
                previous_association[key] = state
        for belief in evidence['structure_belief']['instances']:
            assoc = associations.get(belief['instance_id'], {})
            belief_trace.append(dict(paid_step=step, instance_id=belief['instance_id'],
                measured_class=assoc.get('observed_class'), qualified_class=belief['observed_class'],
                association_uncertain=assoc.get('association_uncertain'), class_conflict=assoc.get('class_conflict'),
                support_sha256=assoc.get('support_sha256'), support_points=len(assoc.get('support_points_world_m', [])),
                distinct_class_supports=assoc.get('distinct_class_supports'),
                geometry_log_evidence=belief['geometry_log_evidence'], structure_probabilities=belief['structure_probabilities'],
                rho=belief.get('rho'), peer_instance_ids=belief['peer_instance_ids'], action=decision['action'],
                routing_reason=decision.get('reason')))
        if decision.get('reason') == 'return':
            first.setdefault('return_decision', step)
        for key, belief in beliefs.items():
            instance_first.setdefault(key, dict(first_observed_step=step, first_class_step=None))
            if belief['observed_class'] is not None:
                if instance_first[key]['first_class_step'] is None:
                    instance_first[key]['first_class_step'] = step
                first.setdefault('semantic_class', step)
        for feedback in evidence.get('geometry_feedback', []):
            mechanism['feedback_attempts'] += 1
            if feedback['applied']:
                mechanism['feedback_applied'] += 1
                first.setdefault('applied_feedback', step)
        if evidence.get('first_actual_reliable_planes'):
            first.setdefault('reliable_plane', step)
        selection = decision.get('global_selection') if decision.get('global_replanned') else None
        forecasts = {}
        options = {}
        if selection:
            mechanism['replans'] += 1
            selected = selection.get('selected')
            mechanism['replans_with_selected_target'] += int(selected is not None)
            candidates = selection.get('direct_options', [])+selection.get('diagnostic_options', [])
            if selected and selected.get('kind') == 'measurement_initialization':
                candidates += selection.get('initialization_options', [selected])
            for candidate in candidates:
                target = candidate.get('target') or {}
                key = (candidate['kind'], target.get('node'), target.get('heading'), candidate.get('instance_id'))
                require(key not in options, 'nonunique candidate key')
                options[key] = {k: candidate.get(k) for k in ('score', 'expected_gain', 'evi', 'total_cost', 'scheduling_priority')}
            for forecast in selection.get('forecasts', []):
                key = forecast['instance_id']
                forecasts[key] = [dict(view_id=x['view_id'], gain=x['expected_new_surface_area_m2'],
                    structure_gain=x['structure_new_surface_area_m2'], fallback=x.get('fallback', False),
                    reason=x.get('fallback_reason'), repeated=x.get('repeated_view_excluded', False))
                    for x in forecast['candidates']]
                for item in forecasts[key]:
                    mechanism['forecast_candidates'] += 1
                    mechanism['positive_forecast_candidates'] += int(item['gain'] > TOL)
                    mechanism['zero_basis_forecast_candidates'] += int(all(abs(x) <= TOL for x in item['structure_gain']))
                    if item['gain'] > TOL:
                        first.setdefault('positive_forecast', step)
                    if item['fallback']:
                        fallback[str(item['reason'])] += 1
            selected = selection.get('selected') or {}
            replans.append(dict(paid_step=step, selected_kind=selected.get('kind'),
                selected_node=(selected.get('target') or {}).get('node'),
                selected_heading=(selected.get('target') or {}).get('heading'),
                selected_score=selected.get('score'), selected_expected_gain=selected.get('expected_gain'),
                selected_cost=selected.get('total_cost'), candidate_options=len(options),
                forecast_candidates=sum(map(len, forecasts.values())),
                positive_forecast_candidates=sum(c['gain'] > TOL for rows in forecasts.values() for c in rows),
                remaining=selection['remaining'], routing_reason=decision.get('reason'),
                feasible_direct_options=len(selection.get('direct_options', [])),
                feasible_diagnostic_options=len(selection.get('diagnostic_options', []))))
        compact[step] = dict(observation_sha=evidence['observation_sha256'], action=decision['action'],
                             observation_content_sha=content_sha,
                             beliefs=beliefs, forecasts=forecasts, options=options,
                             selected=selection.get('selected') if selection else None,
                             replanned=selection is not None, reason=decision.get('reason'),
                             remaining=selection.get('remaining') if selection else None)
    require(len(compact) == result['acquired_and_saved_packets'], 'saved step count differs from completed result')
    require(sorted(compact) == list(range(len(compact))), 'nonconsecutive saved steps')
    return dict(result=result, steps=compact, replans=replans, belief_trace=belief_trace,
                association_diagnostics=dict(transitions=association_transitions, ambiguity_events=association_events,
                    uncertain_instance_frames=dict(uncertain_counts), conflicting_class_instance_frames=dict(conflict_counts),
                    anchors=anchors, anchor_changes=anchor_changes,
                    caveat='Local-overlap rejected receipts lack participant instance IDs. Stable anchors/IDs and absent class conflicts do not prove every historical support point has correct ownership.'),
                mechanism=dict(counts=dict(mechanism), fallback_candidate_reasons=dict(fallback),
                               first_steps=first, instances=instance_first))


def forecast_state(rows):
    if rows is None or not rows:
        return dict(state='no_forecast_record', candidates=0, zero_basis=None, reasons=[])
    zero = all(all(abs(v) <= TOL for v in x['structure_gain']) for x in rows)
    return dict(state='all_structure_gains_zero' if zero else 'nonzero_structure_gain_available',
                candidates=len(rows), zero_basis=zero,
                reasons=sorted({x['reason'] for x in rows if x['reason']}),
                repeated=sum(x['repeated'] for x in rows),
                positive=sum(x['gain'] > TOL for x in rows))


def compare_episode(left, right):
    """Mechanism comparisons use identical paid observation prefixes only."""
    la, ra = left['result']['actions'], right['result']['actions']
    first_action = next((i+1 for i, (a, b) in enumerate(zip(la, ra))
                         if a['sensor_action'] != b['sensor_action']), None)
    if first_action is None and len(la) != len(ra):
        first_action = min(len(la), len(ra))+1
    left_steps, right_steps = left['steps'], right['steps']
    common_steps = sorted(set(left_steps) & set(right_steps))
    obs_divergence = next((i for i in common_steps
        if left_steps[i].get('observation_content_sha', left_steps[i]['observation_sha'])
        != right_steps[i].get('observation_content_sha', right_steps[i]['observation_sha'])), None)
    receipt_difference = next((i for i in common_steps if left_steps[i]['observation_sha'] != right_steps[i]['observation_sha']), None)
    prefix = [i for i in common_steps if (obs_divergence is None or i < obs_divergence)
              and (first_action is None or i < first_action)]
    posterior_rows, score_rows = [], []
    candidate_differences = []
    for i in prefix:
        a, b = left_steps[i], right_steps[i]
        for key in sorted(set(a['beliefs']) & set(b['beliefs'])):
            ap, bp = a['beliefs'][key], b['beliefs'][key]
            delta = max(abs(x-y) for x, y in zip(ap['prob'], bp['prob']))
            rho_delta = abs(ap['rho']-bp['rho']) if ap['rho'] is not None and bp['rho'] is not None else None
            if delta > TOL or (rho_delta is not None and rho_delta > TOL):
                af, bf = forecast_state(a['forecasts'].get(key)), forecast_state(b['forecasts'].get(key))
                posterior_rows.append(dict(paid_step=i, instance_id=key, posterior_max_abs_delta=delta,
                    structure_posterior_changed=delta > TOL,
                    rho_abs_delta=rho_delta, left_probabilities=ap['prob'], right_probabilities=bp['prob'],
                    left_forecast=af, right_forecast=bf,
                    both_forecast_bases_zero=(af['zero_basis'] is True and bf['zero_basis'] is True),
                    left_replanned=a['replanned'], right_replanned=b['replanned'],
                    left_feasible_options=len(a['options']), right_feasible_options=len(b['options']),
                    left_routing_reason=a.get('reason'), right_routing_reason=b.get('reason'),
                    left_remaining=a.get('remaining'), right_remaining=b.get('remaining')))
        if a['replanned'] and b['replanned']:
            keys = set(a['options']) & set(b['options'])
            def ranking(options):
                targets = {}
                for option_key, item in options.items():
                    if item['score'] is not None:
                        target = option_key[1:3]
                        targets[target] = max(targets.get(target, -math.inf), item['score'])
                return sorted(targets, key=lambda t: (-targets[t], str(t)))
            at, bt = (a.get('selected') or {}).get('target'), (b.get('selected') or {}).get('target')
            candidate_differences.append(dict(paid_step=i, left_only=len(set(a['options'])-keys),
                right_only=len(set(b['options'])-keys), common=len(keys),
                selected_target_equal=at == bt, selected_left=at, selected_right=bt,
                unique_target_score_order_equal=ranking(a['options']) == ranking(b['options'])))
            for key in sorted(keys, key=str):
                ar, br = a['options'][key], b['options'][key]
                numeric = all(isinstance(v, (int, float)) for v in (ar['score'], br['score']))
                delta = ar['score']-br['score'] if numeric else None
                score_rows.append(dict(paid_step=i, kind=key[0], target_node=key[1], target_heading=key[2],
                    instance_id=key[3], left_score=ar['score'], right_score=br['score'], score_delta=delta,
                    score_changed=abs(delta) > TOL if numeric else None,
                    left_expected_gain=ar['expected_gain'], right_expected_gain=br['expected_gain'],
                    left_total_cost=ar['total_cost'], right_total_cost=br['total_cost']))
    numeric_scores = [r for r in score_rows if r['score_delta'] is not None]
    changed_scores = [r for r in numeric_scores if r['score_changed']]
    changed_posteriors = [r for r in posterior_rows if r['structure_posterior_changed']]
    posterior_with_forecasts = [r for r in changed_posteriors if r['left_forecast']['candidates'] and r['right_forecast']['candidates']]
    return dict(first_action_divergence_paid_step=first_action, actions_identical=first_action is None,
        first_namespaced_receipt_hash_difference_paid_step=receipt_difference,
        observation_equality_rule='exact uncompressed NPY arrays: RGB-D excluding frame_id only, plus complete scan, including calibration/pose/time/paid_step',
        first_observation_divergence_paid_step=obs_divergence, common_observation_prefix_frames=len(prefix),
        common_replan_count=len(candidate_differences), numeric_common_candidate_scores=len(numeric_scores),
        common_selected_target_differences=sum(not r['selected_target_equal'] for r in candidate_differences),
        unique_target_score_order_differences=sum(not r['unique_target_score_order_equal'] for r in candidate_differences),
        changed_common_candidate_scores=len(changed_scores),
        first_changed_score_paid_step=min((r['paid_step'] for r in changed_scores), default=None),
        max_abs_common_score_delta=max((abs(r['score_delta']) for r in numeric_scores), default=None),
        changed_posterior_instance_steps=len(changed_posteriors),
        changed_reliability_instance_steps=sum(r['rho_abs_delta'] is not None and r['rho_abs_delta'] > TOL for r in posterior_rows),
        changed_posterior_with_both_forecasts=len(posterior_with_forecasts),
        changed_posterior_with_both_zero_forecast_bases=sum(r['both_forecast_bases_zero'] for r in posterior_with_forecasts),
        changed_posterior_at_replan_without_feasible_options=sum(r['left_replanned'] and r['right_replanned']
            and not r['left_feasible_options'] and not r['right_feasible_options'] for r in changed_posteriors),
        first_changed_posterior_paid_step=min((r['paid_step'] for r in changed_posteriors), default=None),
        candidate_set_differences=candidate_differences, posterior_rows=posterior_rows, score_rows=score_rows,
        scope='Same observation/action prefix only; after-divergence states are not a controlled mechanism comparison. Zero bases describe supplied candidates, not every physically possible view.')


def analyze_protocol(path, review_root, inputs):
    protocol = inputs.json(path)
    protocol_sha = digest(Path(path).read_bytes())
    require(protocol['schema'] == 'article.experiment_protocol.v1' and protocol['status'] == 'frozen', 'unfrozen or unknown protocol')
    phase = protocol['phase']
    output = safe(ROOT, protocol['output_relative_path'])
    ledger_path = output/'start_ledger.json'
    ledger = inputs.json(ledger_path, mutable=True) if ledger_path.exists() else None
    if ledger:
        require(ledger['protocol_sha256'] == protocol_sha and ledger['slots'] == protocol['slots'], 'ledger/protocol binding differs')
        entries = {x['run_id']: x for x in ledger['entries']}
        require(len(entries) == len(ledger['entries']) and set(entries) <= set(protocol['slots']), 'duplicate or unknown ledger slots')
    else:
        entries = {}
    rows, accepted, findings = [], {}, []
    for run_id, slot in sorted(protocol['slots'].items()):
        entry = entries.get(run_id)
        row = dict(phase=phase, protocol_sha256=protocol_sha, run_id=run_id, **slot,
                   status=status_without_review(entry), qualified=None, review_passed=None,
                   online_status=entry['status'] if entry else None)
        rows.append(row)
        # No probe of live episode or review directory: a reserved slot is not a failure.
        if not entry or entry['status'] == 'reserved':
            continue
        review_dir = review_root/run_id
        review_manifest_path = review_dir/'manifest.json'
        if not review_manifest_path.is_file():
            continue
        # The independent reviewer writes its manifest last. A partial last
        # write is pending, never evidence of a failed experiment or audit.
        try:
            ready_bytes = review_manifest_path.read_bytes()
            json.loads(ready_bytes)
        except (FileNotFoundError, json.JSONDecodeError):
            row['review_pending_reason'] = 'manifest_not_yet_complete'
            continue
        try:
            review_manifest = inputs.json(review_manifest_path, pin=digest(ready_bytes))
            require(review_manifest['schema'] == 'article.episode_review_manifest.v1', 'review manifest schema')
            for name, pin in review_manifest['files'].items():
                data = inputs.read(safe(review_dir, name), pin=pin['sha256'])
                require(len(data) == pin['bytes'], 'review byte count differs')
            review = inputs.json(review_dir/'review.json')
            require(review['run_id'] == run_id, 'review run identity differs')
            row.update(review_status=review['status'], review_passed=review.get('all_checks_passed', False),
                       qualified=review.get('qualified'))
            if review['status'] != 'reviewed' or not review.get('all_checks_passed'):
                if review['status'] == 'failed_attempt_preserved':
                    failure = inputs.json(output/'episodes'/run_id/'attempt_failure.json', pin=entry['result_sha256'])
                    require(failure == review['failure'], 'failure record differs from independent review')
                row['status'] = 'failed_attempt_reviewed' if review['status'] == 'failed_attempt_preserved' else 'review_findings'
                findings.append(dict(run_id=run_id, phase=phase, status=row['status'],
                                     failures=review.get('failures'), failure=review.get('failure'), error=review.get('error')))
                continue
            episode = output/'episodes'/run_id
            manifest = inputs.json(episode/'artifact_manifest.json', pin=entry['artifact_manifest_sha256'])
            require(entry['artifact_manifest_sha256'] == review_manifest['input_episode_manifest_sha256']
                    == review['input_manifest_sha256'], 'review/ledger/episode seal differs')
            require(manifest['protocol_sha256'] == review['protocol_sha256'] == protocol_sha, 'review protocol differs')
            require(review['method'] == slot['method'] and review['scene_id'] == slot['scene_id']
                    and review['phase'] == phase and review['budget'] == slot['budget'], 'review slot metadata differs')
            require(review['online_status'] == entry['status'] and review['qualified'] == entry['qualified'], 'review qualification/terminal status differs')
            data = read_episode(inputs, episode, manifest)
            require(digest(sealed_read(inputs, episode, manifest, 'result.json')) == entry['result_sha256'], 'result ledger pin differs')
            metrics = review['metrics']
            row.update(status='reviewed_qualified' if review['qualified'] else 'reviewed_unqualified',
                C_nav=metrics['C_nav'], P=metrics['macro_precision'], R=metrics['macro_completeness'],
                F1=metrics['macro_f1'], J_nav=metrics['J_nav'],
                paid_actions=review['paid_actions'], path_length_m=review['translation_m'],
                turns=review['turns'], explicit_observe_actions=review['extra_observes'],
                observation_frames=review['rgbd_frames'], collisions=review['collisions'],
                returned_xy_and_yaw=review['returned_xy_and_yaw'],
                minimum_return_slack_actions=review['minimum_return_slack_actions'],
                elapsed_episode_s=data['result']['elapsed_s'],
                planning_s=data['result']['timings'].get('planning_s'),
                evidence_s=data['result']['timings'].get('evidence_s'),
                acquisition_s=data['result']['timings'].get('acquisition_s'),
                mapping_s=data['result']['timings'].get('mapping_s'),
                reference_fingerprint=metrics['reference_fingerprint'])
            accepted[run_id] = data
            if not review['qualified']:
                findings.append(dict(run_id=run_id, phase=phase, status='reviewed_unqualified',
                    online_status=row['online_status'], collisions=row['collisions'], returned=row['returned_xy_and_yaw']))
        except (OSError, ValueError, KeyError, TypeError) as exc:
            row.update(status='analysis_binding_error', review_passed=False, qualified=None)
            for key in (*METRICS, 'planning_s', 'elapsed_episode_s'):
                row.pop(key, None)
            findings.append(dict(run_id=run_id, phase=phase, status='analysis_binding_error', error=str(exc)))
    return protocol, rows, accepted, findings


def make_pairs(rows, data):
    groups = defaultdict(dict)
    for row in rows:
        key = (row['phase'], row['protocol_sha256'], row['scene_id'], row['budget'], row['noise_seed'])
        require(row['method'] not in groups[key], 'duplicate matched method within protocol block')
        groups[key][row['method']] = row
    pairs, diagnostics = [], {}
    for key, members in sorted(groups.items()):
        for left_method, right_method in combinations([m for m in METHODS if m in members], 2):
            left, right = members[left_method], members[right_method]
            pair_id = left['run_id']+'__vs__'+right['run_id']
            pair = dict(pair_id=pair_id, phase=key[0], protocol_sha256=key[1], scene_id=key[2],
                        budget=key[3], noise_seed=key[4], left_method=left_method, right_method=right_method,
                        left_run=left['run_id'], right_run=right['run_id'], left_status=left['status'],
                        right_status=right['status'], status='unavailable')
            pairs.append(pair)
            if left['run_id'] not in data or right['run_id'] not in data:
                continue
            pair['status'] = 'both_reviewed_qualified' if left['qualified'] and right['qualified'] else 'qualification_failure'
            require(left['reference_fingerprint'] == right['reference_fingerprint'], 'paired evaluation reference differs')
            for metric in (*METRICS, *COSTS):
                pair['left_'+metric], pair['right_'+metric] = left[metric], right[metric]
                if pair['status'] == 'both_reviewed_qualified' and left[metric] is not None and right[metric] is not None:
                    pair['delta_'+metric] = left[metric]-right[metric]
            pair['mechanism_available'] = True
            comparison = compare_episode(data[left['run_id']], data[right['run_id']])
            diagnostics[pair_id] = comparison
            pair.update({k: v for k, v in comparison.items() if k not in ('candidate_set_differences', 'posterior_rows', 'score_rows', 'scope')})
    return pairs, diagnostics


def write_csv(path, rows, fallback):
    fields = list(dict.fromkeys(k for row in rows for k in row)) or fallback
    with path.open('x', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: json.dumps(v, sort_keys=True) if isinstance(v, (dict, list)) else v for k, v in row.items()})


def save_snapshot(output, protocols, rows, data, findings, pairs, diagnostics, inputs):
    output.mkdir(parents=True, exist_ok=False)
    def save(name, value):
        with (output/name).open('xb') as stream:
            stream.write(canonical(value))
    save('slots.json', rows); write_csv(output/'slots.csv', rows, ['phase', 'run_id', 'status'])
    save('qualification_and_failures.json', findings)
    save('pairs.json', pairs); write_csv(output/'pairs.csv', pairs, ['phase', 'pair_id', 'status'])
    save('mechanism_comparisons.json', diagnostics)
    save('per_run_mechanisms.json', {k: d['mechanism'] for k, d in data.items()})
    save('per_run_association_diagnostics.json', {k:d['association_diagnostics'] for k,d in data.items()})
    replans = [dict(run_id=k, **r) for k, d in data.items() for r in d['replans']]
    write_csv(output/'replans.csv', replans, ['run_id', 'paid_step'])
    belief_rows = [dict(run_id=k, **r) for k, d in data.items() for r in d['belief_trace']]
    write_csv(output/'belief_qualification_timeline.csv', belief_rows, ['run_id', 'paid_step', 'instance_id'])
    score_rows = [dict(pair_id=k, **r) for k, d in diagnostics.items() for r in d['score_rows']]
    write_csv(output/'paired_candidate_scores.csv', score_rows, ['pair_id', 'paid_step'])
    posterior_rows = [dict(pair_id=k, **r) for k, d in diagnostics.items() for r in d['posterior_rows']]
    write_csv(output/'paired_posterior_gates.csv', posterior_rows, ['pair_id', 'paid_step'])
    phases = sorted({r['phase'] for r in rows})
    phase_summary = {p: dict(status_counts=dict(Counter(r['status'] for r in rows if r['phase'] == p)),
        qualified_pairs=sum(r['phase'] == p and r['status'] == 'both_reviewed_qualified' for r in pairs),
        source_protocols=[dict(path=str(path.relative_to(ROOT)), sha256=digest(path.read_bytes()), slots=len(proto['slots']))
                          for path, proto in protocols if proto['phase'] == p]) for p in phases}
    for p in ('development', 'main'):
        phase_summary.setdefault(p, dict(status='no_frozen_protocol_supplied', status_counts={}, qualified_pairs=0))
    for i, (name, payload) in enumerate(inputs.captured.items()):
        (output/f'captured_ledger_{i:02d}.json').write_bytes(payload)
    inputs.unchanged()
    summary = dict(schema='article.comparison_snapshot.v1', created_utc=datetime.now(timezone.utc).isoformat(),
        phases=phase_summary, protocol_count=len(protocols), qualified_pair_count=sum(x['status']=='both_reviewed_qualified' for x in pairs),
        independent_reviewed_episodes=len(data), findings_count=len(findings),
        counters=dict(new_worlds=0, new_policy_runs=0, new_sensor_queries=0, new_tsdf_integrations=0, new_quality_evaluations=0),
        metric_definitions=dict(P='unweighted macro precision over four fixed devices', R='unweighted macro recall/completeness over four fixed devices',
            F1='mean of per-device F1, not harmonic mean of macro P and R', J_nav='C_nav * F1',
            observation_frames='initial frame plus every paid action; explicit observes reported separately'),
        limitations=['Development and main are separate; development results are engineering observations, not independent confirmation.',
            'Only reviewed, SHA-bound sealed records provide metrics. Missing/unstarted/running slots have blank metrics, never zero scores.',
            'Unqualified reviewed episodes remain visible but are excluded from qualified performance deltas.',
            'Mechanism comparisons use a common actual observation/action prefix; score differences alone do not establish improved reconstruction.',
            'Zero forecast diagnoses concern stored supplied candidates, not all possible views.',
            'Cross-run observation equality uses all saved sensor arrays except run-namespaced frame_id; original receipt hashes remain recorded.',
            'Elapsed/planning times are recorded wall times and can be affected by simultaneous jobs; no isolated runtime benchmark is claimed.',
            'All metric values are copied from completed independent reviews; no new mesh evaluation occurs.',
            'Positive gains are not a reporting or qualification filter. Every supplied frozen slot and failure is retained.',
            'Ledgers are captured once; later completions belong in a new snapshot.'],
        read_scope='All review files and used sealed result/step/RGB-D/scan artifacts rehashed; binary predictions rely on the independent review seal.')
    save('summary.json', summary)
    text = ['# Article comparison snapshot', '', 'Results are separated by phase; all supplied slots remain listed. No minimum-positive-gain selection is applied.', '']
    for phase, status in phase_summary.items():
        text += [f'## {phase}', '', json.dumps(status.get('status_counts', {}), sort_keys=True), '',
                 f"Qualified completed pairs: {status['qualified_pairs']}.", '']
    text += ['## Reviewed outcome rows', '', '| Run | Qualified | C_nav | P | R | F1 | J_nav |', '| --- | --- | --- | --- | --- | --- | --- |']
    for row in rows:
        if row['run_id'] in data:
            text += ['| '+ ' | '.join([row['run_id'], str(row['qualified'])]+[f'{row[m]:.6f}' for m in METRICS])+' |']
    text += ['', '## Interpretation boundaries', '']+['- '+s for s in summary['limitations']]
    (output/'README.md').write_text('\n'.join(text)+'\n')
    files = {str(p.relative_to(output)): dict(bytes=p.stat().st_size, sha256=digest(p.read_bytes())) for p in output.iterdir() if p.is_file()}
    save('manifest.json', dict(schema='article.comparison_manifest.v1', sources=inputs.pins, files=files,
                              script_sha256=digest(Path(__file__).read_bytes())))
    return summary


def self_test():
    require(status_without_review(None) == 'unstarted', 'unstarted classification')
    require(status_without_review({'status':'reserved'}) == 'running_reserved', 'live classification')
    require(forecast_state(None)['zero_basis'] is None, 'missing forecast must not mean zero')
    base = dict(result=dict(actions=[dict(sensor_action='forward')]), steps={0:dict(observation_sha='same',
        action='forward', beliefs={'i':dict(prob=[.25]*4, rho=.5)}, forecasts={}, options={}, selected=None, replanned=False)})
    c = compare_episode(base, base)
    require(c['actions_identical'] and c['changed_common_candidate_scores'] == 0, 'same trace identity')
    changed = json.loads(json.dumps(base)); changed['steps'] = {int(k):v for k,v in changed['steps'].items()}
    changed['steps'][0]['beliefs']['i']['prob'] = [.4,.2,.2,.2]
    c = compare_episode(changed, base)
    require(c['changed_posterior_instance_steps'] == 1 and c['changed_posterior_with_both_zero_forecast_bases'] == 0,
            'no forecast cannot establish posterior blocking')
    row = dict(phase='development', protocol_sha256='p', scene_id='s', budget=1, noise_seed=1,
               method='G', run_id='g', status='reviewed_qualified', qualified=True)
    pairs, diagnostics = make_pairs([row], {'g':base})
    require(not pairs and not diagnostics, 'single method cannot create comparisons')
    other = dict(row, method='S', run_id='s', status='unstarted', qualified=None)
    pairs, _ = make_pairs([row, other], {'g':base})
    require(len(pairs) == 1 and pairs[0]['status'] == 'unavailable' and 'delta_J_nav' not in pairs[0], 'missing pair metrics')
    zero_forecast = [dict(structure_gain=[0.]*4, gain=0., repeated=False, reason='missing_plane')]
    base['steps'][0]['forecasts']['i'] = zero_forecast
    changed['steps'][0]['forecasts']['i'] = zero_forecast
    c = compare_episode(changed, base)
    require(c['changed_posterior_with_both_zero_forecast_bases'] == 1, 'observed zero basis blocks every mixture')
    key = ('direct', 'home', 0, None)
    base['steps'][0]['options'][key] = dict(score=.1, expected_gain=1., evi=None, total_cost=10, scheduling_priority=None)
    changed['steps'][0]['options'][key] = dict(score=.2, expected_gain=2., evi=None, total_cost=10, scheduling_priority=None)
    base['steps'][0]['replanned'] = changed['steps'][0]['replanned'] = True
    c = compare_episode(changed, base)
    require(c['changed_common_candidate_scores'] == 1 and c['first_changed_score_paid_step'] == 0, 'same-input score change')
    changed['steps'][0]['observation_sha'] = 'different'
    require(compare_episode(changed, base)['numeric_common_candidate_scores'] == 0, 'different observations cannot be controlled evidence')
    row.update({k:1. for k in (*METRICS, *COSTS)})
    row['reference_fingerprint'] = 'fixed'
    other = dict(row, method='S', run_id='s', J_nav=.5)
    pairs, _ = make_pairs([row, other], {'g':base, 's':changed})
    require(pairs[0]['delta_J_nav'] == -.5 and pairs[0]['delta_path_length_m'] == 0., 'negative outcome must remain visible')
    def packet_fixture(kind, identity='run1', depth='depth'):
        entries = ({'frame_id':identity, 'paid_step':'0', 'rgb':'rgb', 'depth_m':depth, 'intrinsic':'K', 'world_from_camera':'T'}
                   if kind == 'rgbd' else {'timestamp_s':'0','ranges_m':'scan','angle_min_rad':'a','angle_increment_rad':'b','range_max_m':'c','world_from_laser':'T'})
        result = io.BytesIO()
        with zipfile.ZipFile(result, 'w') as archive:
            for name,value in entries.items():
                archive.writestr(name+'.npy', value.encode())
        return result.getvalue()
    scan = packet_fixture('scan')
    a = sensor_content_sha(packet_fixture('rgbd','run1'), scan)
    require(a == sensor_content_sha(packet_fixture('rgbd','run2'), scan), 'run namespace is not physical sensor difference')
    require(a != sensor_content_sha(packet_fixture('rgbd','run2','changed'), scan), 'actual depth difference remains protected')
    print('self-test passed: twelve missing/live/pair/forecast/common-input/negative-outcome/sensor-namespace integrity scenarios')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--protocol', action='append', type=Path)
    parser.add_argument('--review-root', type=Path, default=STAGE/'episode_reviews_v1')
    parser.add_argument('--output-root', type=Path, default=STAGE/'analysis_v1')
    parser.add_argument('--snapshot', default=None)
    parser.add_argument('--self-test', action='store_true')
    args = parser.parse_args()
    if args.self_test:
        self_test(); return
    paths = args.protocol or [ROOT/'configs/virtual3d/article_development_v1_20260928.json']
    name = args.snapshot or datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ_')+uuid.uuid4().hex[:8]
    require(Path(name).name == name and name not in ('.', '..'), 'snapshot must be a single new directory name')
    inputs = Inputs(); inputs.read(Path(__file__))
    rows, data, findings, protocols = [], {}, [], []
    for path in paths:
        path = path.resolve()
        protocol, group_rows, group_data, group_findings = analyze_protocol(path, args.review_root.resolve(), inputs)
        require(not (set(data) & set(group_data)), 'duplicate episode ID across protocols')
        protocols.append((path, protocol)); rows.extend(group_rows); data.update(group_data); findings.extend(group_findings)
    pairs, diagnostics = make_pairs(rows, data)
    output = args.output_root.resolve()/name
    require(not any(output.is_relative_to(safe(ROOT,p['output_relative_path'])) for _,p in protocols), 'analysis output must be outside experiment phase directories')
    result = save_snapshot(output, protocols, rows, data, findings, pairs, diagnostics, inputs)
    print(json.dumps(dict(output=str(output), phases=result['phases'], findings=len(findings),
                          bytes=sum(p.stat().st_size for p in output.iterdir())), indent=2))


if __name__ == '__main__':
    main()
