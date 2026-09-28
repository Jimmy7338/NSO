#!/usr/bin/env python3
"""Read-only GroundV2/ExposureV3 paired development-ablation snapshots.

No simulator, controller, fusion or quality evaluator is imported or invoked.
All 24 declared records remain visible; only sealed, independently reviewed
terminal records can supply metrics and traces. Real snapshots require an
explicit caller release. Main/held-out protocols are outside this interface.
"""
import argparse
from collections import Counter
from datetime import datetime, timezone
import gzip
import io
import json
import math
from pathlib import Path
import sys
import zipfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import analyze_article_ground_comparisons_20260928 as ground

old = ground.old
require = old.require
STAGE = ROOT/'audit_results/article_stage_20260928'
ARMS = ('GroundV2', 'ExposureV3')
METHODS = ('G', 'B', 'S', 'NBV')
SCENES = ('ART1_AISLE_DEV', 'ART1_CELL_DEV', 'ART1_LOOP_DEV')
VERSION = ground.VERSION
METRICS, COSTS = ground.METRICS, ground.COSTS
REVIEW_SCHEMAS = {
    'GroundV2': ('article.ground_episode_review_manifest.v2', 'article.saved_episode_review.ground_v2'),
    'ExposureV3': ('article.exposure_episode_review_manifest.v3', 'article.saved_episode_review.exposure_v3')}
MAX_OUTPUT_BYTES = 10*1024**2
GATE_DECLARATION = ROOT/'docs/research/ARTICLE_EXPOSURE_MAIN_MECHANISM_GATE_20260928.md'


def blank_row(arm, run_id, slot, entry):
    row = ground.blank_row(arm, run_id, slot, entry)
    row.update(observed_instance_count=None, qualified_category_instance_count=None,
        first_recorded_peer_step=None, first_informative_peer_step=None,
        macro_count=None, replans=None, final_budget_remaining=None,
        input_artifact_manifest_sha256=None, input_review_manifest_sha256=None,
        input_evaluation_sha256=None, input_protocol_sha256=None)
    return row


def validate_pairing(protocol, baseline):
    require(protocol.get('schema') == 'article.experiment_protocol.exposure_v3'
        and baseline.get('schema') == 'article.experiment_protocol.ground_v2', 'strict paired V3/V2 protocols')
    for source in (protocol, baseline):
        require(source.get('status') == 'frozen' and source.get('phase') == 'ablation'
            and len(source['slots']) == 12, 'only complete frozen twelve-slot development ablation')
        inventory = [(s['scene_id'], s['method']) for s in source['slots'].values()]
        require(len(set(inventory)) == 12 and set(inventory) == {(s,m) for s in SCENES for m in METHODS},
                'exact three development layouts by four methods')
    for field in ('evaluation', 'metric_preprocessing', 'references', 'mapper',
                  'asset_root', 'asset_manifest_sha256', 'numerical_runtime'):
        require(protocol[field] == baseline[field], 'paired common setting: '+field)
    require(protocol['controller'].get('same_center_exposure_dedup') is True
        and {k:v for k,v in protocol['controller'].items() if k != 'same_center_exposure_dedup'} == baseline['controller'],
        'only declared common exposure option differs')
    paired = []
    for run, slot in protocol['slots'].items():
        previous = slot['paired_baseline_run_id']; paired.append(previous)
        require(previous in baseline['slots'], 'known paired GroundV2 slot')
        require(all(slot[k] == baseline['slots'][previous][k]
                    for k in ('scene_id','method','budget','noise_seed')), 'same paired slot covariates')
    require(set(paired) == set(baseline['slots']) and len(set(paired)) == 12, 'bijective twelve-slot pairing')


def verify_sources(inputs, protocol):
    pins = protocol['source_sha256']
    payload = inputs.read(old.safe(ROOT, protocol['source_archive']), pin=protocol['source_archive_sha256'])
    with zipfile.ZipFile(io.BytesIO(payload)) as archive:
        require(len(archive.namelist()) == len(set(archive.namelist()))
            and set(archive.namelist()) == set(pins), 'exact archived execution-source inventory')
        for name, expected in pins.items():
            require(old.digest(archive.read(name)) == expected, 'archived source SHA: '+name)
            inputs.read(old.safe(ROOT,name), pin=expected)
    for reference in protocol['references'].values():
        inputs.read(old.safe(ROOT,reference['root'])/'manifest.json', pin=reference['manifest_sha256'])
    inputs.read(old.safe(ROOT,protocol['asset_root'])/'manifest.json', pin=protocol['asset_manifest_sha256'])


def verify_source_relationship(current, baseline):
    # The replacement CLI is not imported by the new runner. Its original
    # bytes remain verified in the independently sealed Ground archive.
    require(set(baseline)-set(current) <= {'scripts/run_article_ground_experiment_20260928.py'}
        and all(current[name] == baseline[name] for name in set(current)&set(baseline)),
        'shared Ground execution source remains unchanged')


def motion_complete(physics):
    required = ('paid_actions','budget','collisions','returned_xy_and_yaw','rgbd_frames')
    if not all(k in physics for k in required):
        return None
    return (physics['collisions'] == 0 and physics['returned_xy_and_yaw'] is True
            and 0 <= physics['paid_actions'] <= physics['budget']
            and physics['rgbd_frames'] == physics['paid_actions']+1)


def load_terminal(inputs, phase, review_root, row, entry, protocol, protocol_sha):
    folder = review_root/row['run_id']
    manifest = ground.sealed_folder(inputs,folder,{REVIEW_SCHEMAS[row['arm']][0]})
    if manifest is None:
        return None
    report = inputs.json(folder/'review.json')
    require(report['schema'] == REVIEW_SCHEMAS[row['arm']][1]
        and report['run_id'] == row['run_id'] and report['online_status'] == entry['status'], 'review identity and version')
    require(report['protocol_sha256'] == manifest['protocol_sha256'] == protocol_sha, 'review/protocol pin')
    row.update(review_status=report['status'], review_passed=report.get('all_checks_passed',False),
        input_review_manifest_sha256=old.digest(inputs.read(folder/'manifest.json')),
        input_protocol_sha256=protocol_sha)
    require(isinstance(report.get('reviewer_source_sha256'),dict)
        and bool(report['reviewer_source_sha256']), 'nonempty reviewer source pins')
    for name, pin in report['reviewer_source_sha256'].items():
        inputs.read(old.safe(ROOT,name),pin=pin)
    require(report.get('reviewer_source_sha256') == manifest.get('reviewer_source_sha256'), 'review source pins')
    if report['status'] == 'failed_attempt_preserved' and report.get('all_checks_passed'):
        failure = phase/'episodes'/row['run_id']/'attempt_failure.json'
        if not failure.is_file():
            failure = phase/'failures'/(row['run_id']+'.json')
        saved = inputs.json(failure,pin=entry['result_sha256'])
        require(saved == report['failure'] and entry['status'] == 'attempt_failed', 'original failure binding')
        row.update(status='failed_attempt_reviewed', original_end_to_end_qualified=False,
                   failure_detail=saved)
        return None
    if report['status'] != 'reviewed' or not report.get('all_checks_passed'):
        row.update(status='review_findings',failure_detail=report.get('failures',report.get('error')))
        return None
    episode = phase/'episodes'/row['run_id']
    artifacts = inputs.json(episode/'artifact_manifest.json',pin=entry['artifact_manifest_sha256'])
    require(entry['artifact_manifest_sha256'] == report['input_manifest_sha256']
        == manifest['input_episode_manifest_sha256'], 'ledger/review/artifact pin')
    require(artifacts['protocol_sha256'] == protocol_sha and artifacts['source_sha256'] == protocol['source_sha256'],
        'artifact protocol and execution sources')
    require(report['method'] == row['method'] and report['scene_id'] == row['scene_id']
        and report['phase'] == 'ablation' and report['qualified'] == entry['qualified'], 'reviewed slot and original qualification')
    for name in artifacts['files']:
        old.sealed_read(inputs,episode,artifacts,name)
    evaluation = json.loads(old.sealed_read(inputs,episode,artifacts,'evaluation.json'))
    require(evaluation['metrics'] == report['metrics'] and evaluation['metric_version'] == report['metric_version'] == VERSION,
            'canonical reviewed metric binding')
    require(report['metric_settings'] == protocol['evaluation'], 'common metric settings')
    result = json.loads(old.sealed_read(inputs,episode,artifacts,'result.json'))
    require(old.digest(old.sealed_read(inputs,episode,artifacts,'result.json')) == entry['result_sha256'], 'terminal result pin')
    verified_motion = motion_complete(report['physics_metrics'])
    ground.apply_metrics(row,report['metrics'],version=VERSION,motion=verified_motion is True,
                         mode='original_online_canonical_evaluation')
    row['motion_completion_verified'] = verified_motion
    ground.apply_costs(row,report['physics_metrics'],result)
    row.update(status='reviewed_qualified' if report['qualified'] else 'reviewed_unqualified',
        original_end_to_end_qualified=report['qualified'],
        trace_review_scope='complete_independent_'+row['arm']+'_review',
        input_artifact_manifest_sha256=entry['artifact_manifest_sha256'],
        input_evaluation_sha256=artifacts['files']['evaluation.json']['sha256'])
    if row['arm'] == 'ExposureV3':
        require(report['paired_baseline']['run_id'] == protocol['slots'][row['run_id']]['paired_baseline_run_id']
            and report['paired_baseline']['protocol'] == protocol['paired_baseline_protocol'], 'reviewed paired GroundV2 identity')
        row['exposure_audit'] = report['exposure_audit']
    return episode, artifacts


def finite(value, name):
    require(type(value) in (int,float) and math.isfinite(value) and value >= -1e-12, 'finite nonnegative '+name)
    return float(value)


def area_receipt(forecast, candidate, arm):
    """Separate pre-exposure, exposure residual and final exact-view gating."""
    names = forecast['structure_names']; probabilities = forecast['structure_probabilities']
    require(arm in ARMS and len(names) == len(set(names)) == len(probabilities) == 4
        and all(finite(p,'posterior') <= 1.+1e-12 for p in probabilities)
        and abs(sum(probabilities)-1.) <= 1e-12, 'four-structure posterior')
    before, excluded, after = [], [], []
    components = candidate['components']
    if candidate.get('fallback'):
        require(not components, 'fallback has no synthetic area components')
        before = excluded = after = [0.]*4
    else:
        require(len(components) == 12, 'four structures and three scales')
        for name in names:
            group = [c for c in components if c['structure'] == name]
            require(len(group) == 3 and {c['scale'] for c in group} == {.85,1.,1.15}, 'fixed equal scale mixture')
            b, e, a = [], [], []
            for c in group:
                residual = finite(c['new_surface_area_m2'],'residual area')
                if arm == 'ExposureV3':
                    raw = finite(c['new_surface_area_before_exposure_m2'],'before exposure')
                    removed = finite(c['excluded_previously_exposed_unseen_area_m2'],'excluded exposure')
                    require(c['exposure_exclusion_is_not_measured_surface'] is True, 'nominal exposure meaning')
                else:
                    raw, removed = residual, 0.
                require(abs(raw-residual-removed) <= 1e-12, 'exposure area partition')
                require(abs(finite(c['visible_area_m2'],'visible area')
                    -finite(c['already_observed_area_m2'],'observed area')-raw) <= 1e-12, 'measured-support partition')
                b.append(raw);e.append(removed);a.append(residual)
            before.append(sum(b)/3);excluded.append(sum(e)/3);after.append(sum(a)/3)
    effective = [0.]*4 if candidate['repeated_view_excluded'] else after
    require(len(candidate['structure_new_surface_area_m2']) == 4
        and max(abs(x-finite(y,'effective area')) for x,y in zip(effective,candidate['structure_new_surface_area_m2'])) <= 1e-12,
            'effective areas after exact-view rule')
    weighted = lambda values:sum(p*v for p,v in zip(probabilities,values))
    require(len(candidate['unexcluded_structure_new_surface_area_m2']) == 4
        and max(abs(x-finite(y,'pre exact-view area')) for x,y in zip(after,candidate['unexcluded_structure_new_surface_area_m2'])) <= 1e-12
        and abs(weighted(after)-finite(candidate['unexcluded_expected_new_surface_area_m2'],'pre exact-view expectation')) <= 1e-12,
        'inherited unexcluded fields are after exposure, before exact-view rule')
    require(abs(weighted(effective)-finite(candidate['expected_new_surface_area_m2'],'expectation')) <= 1e-12, 'saved posterior expectation')
    return dict(expected_before_exposure_m2=weighted(before), expected_excluded_exposure_m2=weighted(excluded),
        expected_after_exposure_m2=weighted(after), effective_new_surface_m2=weighted(effective),
        exact_repeated_view_excluded_m2=weighted(after)-weighted(effective),
        before_structure_areas_m2=before, excluded_structure_areas_m2=excluded, after_structure_areas_m2=after,
        structure_probabilities=probabilities, fallback=candidate.get('fallback',False),
        exact_repeated_view=candidate['repeated_view_excluded'])


def planning_evidence(evidence):
    """Compact actual post-feedback planning state, never support-point copies."""
    instances={r['instance_id']:dict(r) for r in evidence['association']['instances']}
    for feedback in evidence.get('geometry_feedback',[]):
        if feedback['applied']:
            key=feedback['instance_id']
            require(key in instances and feedback['instance']['instance_id']==key,'feedback identity')
            instances[key]=dict(feedback['instance'])
    result={}
    for belief in evidence['structure_belief']['instances']:
        key=belief['instance_id'];instance=instances[key];label=belief.get('observed_class')
        values=belief['geometry_log_evidence']
        logs=[-math.inf if v is None else v for v in values]
        qualified=(label is not None and label==instance.get('observed_class')
            and not instance['association_uncertain'] and not instance['class_conflict']
            and instance.get('distinct_class_supports',{}).get(label,0)>=2)
        result[key]=dict(instance_id=key,
            **{name:belief.get(name) for name in ('observed_class','structure_probabilities',
                'active_structure_prior','reliability_weights_leave_one_out','rho',
                'peer_instance_ids','geometry_log_evidence','semantic_conditioning_used')},
            **{name:instance.get(name) for name in ('anchor_world_m','marker_anchor_world_m',
                'support_sha256','marker_observation_sha256','marker_paid_step','distinct_class_supports',
                'geometric_feedback_frames','novel_support_frames','association_uncertain','class_conflict')},
            measured_category_qualified=qualified,geometry_log_evidence_informative=max(logs)-min(logs)>1e-12)
    peers=[]
    for key,record in result.items():
        for peer in record['peer_instance_ids'] or []:
            require(peer in result,'recorded peer has current planning state')
            other=result[peer]
            peers.append(dict(instance_id=key,peer_instance_id=peer,
                same_qualified_category=(record['measured_category_qualified'] and other['measured_category_qualified']
                    and record['observed_class']==other['observed_class']),
                peer_geometry_informative=other['geometry_log_evidence_informative']))
    return dict(instances=result,recorded_peer_links=peers,
        state_timing='association updated by applied current-frame geometry feedback; beliefs from post-feedback structure_belief',
        qualification_is_not_gt_identity_or_effect_proof=True)


def gate_decision(record, *, first_recorded_return_step):
    decision=record['decision'];evidence=record['controller_evidence']
    routing=decision.get('routing') or {};selection=decision.get('global_selection') or {}
    selected=selection.get('selected');target=decision.get('macro_target')
    remaining=(evidence.get('routing') or {}).get('remaining')
    matches=[r for r in routing.get('candidates',[]) if target is not None
        and r.get('node')==target.get('node') and r.get('heading')==target.get('heading')]
    require(len(matches)<=1,'unique committed route candidate')
    cost=None if not matches else matches[0].get('total_with_observation_and_return')
    reserve=routing.get('return_cost_after_action')
    compact_selected=None if selected is None else {k:v for k,v in selected.items()
        if v is None or isinstance(v,(str,bool,int,float)) or k=='target'}
    return dict(paid_step=evidence.get('paid_step'),next_action=decision.get('action'),
        global_replanned=decision.get('global_replanned'),selected=compact_selected,
        macro_id=decision.get('macro_id'),macro_target=target,
        macro_observe_submitted=decision.get('macro_observe_submitted'),
        initialization_cancelled=decision.get('initialization_cancelled'),
        decision_reason=decision.get('reason'),decision_guard=decision.get('safety_guard'),
        remaining_budget=remaining,routing_reason=routing.get('reason'),routing_target=routing.get('target'),
        selected_route_candidate=None if not matches else matches[0],
        committed_macro_budget_feasible=None if cost is None or remaining is None else
            matches[0].get('feasible') is True and cost<=remaining,
        return_cost_after_action=reserve,
        return_reserve_margin_after_action=None if reserve is None or remaining is None else remaining-1-reserve,
        return_latched_recorded=decision.get('return_latched',decision.get('watchdog_return_latched')),
        return_latch_recording='per-step latch absent in original decision schema; final latch is not back-filled',
        first_recorded_return_step=first_recorded_return_step,
        return_reason_seen_before_or_at=first_recorded_return_step is not None,
        configuration_sha256=decision.get('configuration_sha256'),
        source_observation_sha256=decision.get('source_observation_sha256'),
        planning_evidence=planning_evidence(evidence))


def trace_details(inputs, episode, manifest, arm, trace):
    exposures, replans, macros = [], [], []
    first_peer = first_informative = None
    instances = {}; previous_macro = object(); gate_steps={}; first_return=None
    for step, compact in sorted(trace['steps'].items()):
        record = json.loads(gzip.decompress(old.sealed_read(inputs,episode,manifest,f'steps/{step:03d}.json.gz')))
        evidence, decision = record['controller_evidence'], record['decision']
        if (decision.get('routing') or {}).get('reason')=='return' and first_return is None:first_return=step
        gate_steps[step]=gate_decision(record,first_recorded_return_step=first_return)
        associations = evidence['association']['instances']
        for row in associations:
            instances[row['instance_id']] = row
        beliefs = {r['instance_id']:r for r in evidence['structure_belief']['instances']}
        for belief in beliefs.values():
            peers = belief.get('peer_instance_ids',[])
            if peers and first_peer is None: first_peer = step
            for peer in peers:
                values = beliefs[peer]['geometry_log_evidence']
                values = [-math.inf if v is None else v for v in values]
                if max(values)-min(values) > 1e-12 and first_informative is None:
                    first_informative = step
        macro = dict(macro_id=decision.get('macro_id'), target=decision.get('macro_target'),
            initialization_cancelled=decision.get('initialization_cancelled'), reason=decision.get('reason'))
        compact['macro_target'] = macro['target']
        compact['macro_id'] = macro['macro_id']
        compact['macro_cancelled'] = macro['initialization_cancelled']
        if macro != previous_macro:
            macros.append(dict(paid_step=step,**macro));previous_macro=macro
        selection = decision.get('global_selection') if decision.get('global_replanned') else None
        if not selection: continue
        selected = selection.get('selected') or {}; target = selected.get('target') or {}
        target_id = None if not target else str(target['node'])+':'+str(target['heading'])
        selected_receipts = []; all_candidates = 0; positive_excluded = 0
        for forecast in selection.get('forecasts',[]):
            for candidate in forecast['candidates']:
                area = area_receipt(forecast,candidate,arm)
                all_candidates += 1
                positive_excluded += int(area['expected_excluded_exposure_m2'] > 1e-12)
                if candidate['view_id'] == target_id:
                    receipt = dict(paid_step=step,instance_id=forecast['instance_id'],view_id=target_id,
                        selected_kind=selected.get('kind'),**area)
                    exposures.append(receipt);selected_receipts.append(receipt)
        replans.append(dict(paid_step=step, selected_kind=selected.get('kind'), selected_target=target or None,
            selected_score=selected.get('score'), selected_expected_gain=selected.get('expected_gain'),
            selected_total_cost=selected.get('total_cost'), remaining=selection['remaining'],
            forecast_candidates=all_candidates,candidates_with_positive_exposure_exclusion=positive_excluded,
            selected_before_exposure_m2=sum(r['expected_before_exposure_m2'] for r in selected_receipts),
            selected_excluded_exposure_m2=sum(r['expected_excluded_exposure_m2'] for r in selected_receipts),
            selected_after_exposure_m2=sum(r['expected_after_exposure_m2'] for r in selected_receipts),
            selected_effective_new_surface_m2=sum(r['effective_new_surface_m2'] for r in selected_receipts),
            area_is_selection_utility=False))
    qualified = sum(not r['association_uncertain'] and not r['class_conflict']
        and r.get('observed_class') is not None
        and r.get('distinct_class_supports',{}).get(r['observed_class'],0) >= 2 for r in instances.values())
    return dict(observed_instance_count=len(instances), qualified_category_instance_count=qualified,
        first_recorded_peer_step=first_peer,first_informative_peer_step=first_informative,
        macro_count=len({r['macro_id'] for r in macros if r['macro_id'] is not None and r['macro_id'] > 0}),
        replans=replans,macros=macros,selected_exposure_receipts=exposures,
        gate_steps=gate_steps,episode_relative_path=str(episode.relative_to(ROOT)),episode_file_pins=manifest['files'],
        observed_instance_count_is_gt_discovery=False,
        area_semantics='nominal selected-view forecast, not additive measured surface or diagnostic utility')


def main_gate_witnesses(rows,traces,details,pairs):
    """Export six S/B evidence positions. This never authorizes a main run."""
    by={r['run_id']:r for r in rows};result=[]
    for pair in pairs:
        if pair['pair_kind']!='within_arm' or not pair['comparison'].endswith('/S-B'):continue
        left,right=by[pair['left_run']],by[pair['right_run']]
        witness=dict(comparison=pair['comparison'],scene_id=pair['scene_id'],left_run=left['run_id'],right_run=right['run_id'],
            status='unavailable',automatic_gate_pass=None,automatic_main_launch=False,
            side_status={r['method']:dict(status=r['status'],original_end_to_end_qualified=r['original_end_to_end_qualified'],
                motion_completion_verified=r['motion_completion_verified'],quality_measurement_available=r['quality_measurement_available'],
                protocol_sha256=r['input_protocol_sha256'],artifact_manifest_sha256=r['input_artifact_manifest_sha256'],
                review_manifest_sha256=r['input_review_manifest_sha256']) for r in (left,right)},
            endpoint_quality_is_not_an_entry_gate=True)
        result.append(witness)
        if not all(r['run_id'] in traces and r['run_id'] in details for r in (left,right)):continue
        first=pair['first_action_divergence_paid_step'];prefix=pair['common_observation_prefix_frames']
        witness.update(first_actual_action_divergence_paid_step=first,common_actual_prefix_frames=prefix,
            first_observation_divergence_paid_step=pair['first_observation_divergence_paid_step'],
            first_changed_posterior_paid_step=pair['first_changed_posterior_paid_step'],
            first_changed_score_paid_step=pair['first_changed_score_paid_step'])
        if first is None:
            witness['status']='no_actual_action_divergence';continue
        step=first-1
        if not all(first<=len(traces[r['run_id']]['result']['actions']) for r in (left,right)):
            witness['status']='one_execution_ended_before_action_pair';continue
        if step>=prefix:
            witness['status']='action_difference_after_sensor_divergence';continue
        witness.update(status='evidence_ready_for_manual_review',decision_paid_step=step,sides={})
        witness['observed_anchor_identity_pairs']=ground.observed_identity_pairs(
            traces[left['run_id']]['steps'][step]['observed_anchors'],traces[right['run_id']]['steps'][step]['observed_anchors'])
        for row in (left,right):
            run=row['run_id'];detail=details[run];decision=detail['gate_steps'][step];macro=decision['macro_id']
            commits=[k for k,v in detail['gate_steps'].items() if k<=step and v['macro_id']==macro
                and v['global_replanned'] and v['selected'] is not None and v['macro_target'] is not None]
            commit=max(commits) if commits else None
            supporting={}
            for at in {step,first,commit}-{None}:
                for name in (f'steps/{at:03d}.json.gz',f'packets/{at:03d}_receipt.json',
                             f'packets/{at:03d}_rgbd.npz',f'packets/{at:03d}_scan.npz'):
                    if name in detail['episode_file_pins']:
                        supporting[detail['episode_relative_path']+'/'+name]=detail['episode_file_pins'][name]
            for name in ('result.json','controller_final.json','protocol.json'):
                if name in detail['episode_file_pins']:
                    supporting[detail['episode_relative_path']+'/'+name]=detail['episode_file_pins'][name]
            witness['sides'][row['method']]=dict(current_decision=decision,
                macro_commit_paid_step=commit,macro_commit=None if commit is None else detail['gate_steps'][commit],
                macro_commit_within_common_prefix=None if commit is None else commit<prefix,
                actual_executed_sensor_action=traces[run]['result']['actions'][first-1]['sensor_action'],
                supporting_file_pins=supporting)
    require(len(result)==6,'six complete S/B evidence positions')
    return dict(schema='article.exposure_main_gate_witnesses.v1',pairs=result,
        automatic_gate_approval=False,new_main_runs=0,
        scope='Saved witness export only; source, peer-to-selected-action causality, guards and all-cohort qualification require independent review.',
        no_latch_imputation='Only recorded routing/guard/selection evidence is exported. Missing per-step latch remains null.')


def pair_rows(rows, traces):
    by = {(r['arm'],r['scene_id'],r['method']):r for r in rows}
    require(len(by) == len(rows) == 24, 'complete 24-row paired inventory')
    specifications = []
    for scene in SCENES:
        specifications.extend(('across_arm',f'ExposureV3-GroundV2/{m}',
            ('ExposureV3',scene,m),('GroundV2',scene,m)) for m in METHODS)
        specifications.extend(('within_arm',f'{arm}/{a}-{b}',(arm,scene,a),(arm,scene,b))
            for arm in ARMS for a,b in [('S','B'),('B','G')])
    pairs, witnesses = [], {}
    for kind, group, lk, rk in specifications:
        left,right=by[lk],by[rk]; key=left['run_id']+'__vs__'+right['run_id']
        pair=dict(pair_id=key,pair_kind=kind,comparison=group,scene_id=lk[1],
            left_run=left['run_id'],right_run=right['run_id'],left_status=left['status'],right_status=right['status'],
            quality_status='unavailable',cost_status='unavailable',mechanism_status='unavailable',
            both_original_end_to_end_qualified=left['original_end_to_end_qualified'] is True
                and right['original_end_to_end_qualified'] is True)
        if left['paid_actions'] is not None and right['paid_actions'] is not None:
            pair['cost_status']='reviewed_recorded_costs_available'
            for field in (*COSTS,'observed_instance_count','qualified_category_instance_count','macro_count','replans'):
                pair['left_'+field]=left[field];pair['right_'+field]=right[field]
                pair['delta_'+field]=None if left[field] is None or right[field] is None else left[field]-right[field]
        if left['quality_measurement_available'] and right['quality_measurement_available']:
            if left['metric_version'] != right['metric_version'] or left['reference_fingerprint'] != right['reference_fingerprint']:
                pair['quality_status']='incompatible_metric_or_reference'
            elif left['motion_completion_verified'] is True and right['motion_completion_verified'] is True:
                pair['quality_status']='motion_complete_common_derived_metric'
                for field in METRICS:
                    pair['left_'+field]=left[field];pair['right_'+field]=right[field]
                    pair['delta_'+field]=None if left[field] is None or right[field] is None else left[field]-right[field]
            else: pair['quality_status']='quality_available_motion_gate_not_met'
        if left['run_id'] in traces and right['run_id'] in traces:
            lt,rt=traces[left['run_id']],traces[right['run_id']]
            small, event = ground.compact_comparison(lt,rt)
            prefix=small['common_observation_prefix_frames']
            first_macro=next((i for i in range(prefix)
                if lt['steps'][i].get('macro_target') != rt['steps'][i].get('macro_target')),None)
            small['first_macro_target_divergence_paid_step']=first_macro
            first_peer=None
            for step in range(prefix):
                ls,rs=lt['steps'][step],rt['steps'][step]
                mapping=ground.observed_identity_pairs(ls['observed_anchors'],rs['observed_anchors'])
                for right_id,left_id in mapping.items():
                    lp=ls['beliefs'][left_id]['peers'];rp=rs['beliefs'][right_id]['peers']
                    if all(p in mapping for p in rp) and all(p in mapping.values() for p in lp):
                        mapped=sorted(mapping[p] for p in rp)
                        if sorted(lp) != mapped:
                            first_peer=dict(paid_step=step,left_instance_id=left_id,right_instance_id=right_id,
                                left_peer_ids=sorted(lp),right_peer_ids_in_left_identity=mapped)
                            break
                if first_peer is not None:break
            small['first_common_peer_set_divergence_paid_step']=None if first_peer is None else first_peer['paid_step']
            pair.update(small,mechanism_status='common_actual_prefix_only')
            if first_macro is not None:
                event['first_macro_target']=dict(paid_step=first_macro,
                    left=lt['steps'][first_macro].get('macro_target'),right=rt['steps'][first_macro].get('macro_target'))
            if first_peer is not None:event['first_common_peer_set']=first_peer
            witnesses[key]=event
        pairs.append(pair)
    return pairs,witnesses


def save_snapshot(output, rows, traces, details, pairs, witnesses, inputs, findings):
    require(not output.exists() and output.resolve().is_relative_to(STAGE.resolve()), 'exclusive external article-stage snapshot')
    gate=main_gate_witnesses(rows,traces,details,pairs)
    gate['declaration']=dict(path=str(GATE_DECLARATION.relative_to(ROOT)),sha256=old.digest(inputs.read(GATE_DECLARATION)))
    inputs.unchanged(); output.mkdir(parents=True,exist_ok=False)
    def save(name,value):
        with (output/name).open('xb') as stream: stream.write(old.canonical(value))
    def attach(run_id,record):
        row=next(r for r in rows if r['run_id']==run_id)
        return dict(run_id=run_id,arm=row['arm'],scene_id=row['scene_id'],method=row['method'],
            original_end_to_end_qualified=row['original_end_to_end_qualified'],
            motion_completion_verified=row['motion_completion_verified'],**record)
    old.write_csv(output/'slots.csv',rows,['arm','run_id','status'])
    old.write_csv(output/'paired_effects.csv',pairs,['comparison','scene_id','pair_kind','quality_status'])
    trajectories=[]
    for run_id,trace in traces.items():
        cumulative=0.;previous=None
        for point in trace['trajectory']:
            if previous is not None:cumulative+=math.hypot(point['x_m']-previous['x_m'],point['y_m']-previous['y_m'])
            trajectories.append(attach(run_id,dict(**point,cumulative_path_m=cumulative)))
            previous=point
        row=next(r for r in rows if r['run_id']==run_id)
        require(row['path_length_m'] is None or abs(cumulative-row['path_length_m']) <= 1e-9, 'recorded actual path agrees with independent review')
    old.write_csv(output/'trajectories.csv',trajectories,['run_id','paid_step','x_m','y_m','yaw_rad'])
    for name,field in [('selected_exposure.csv','selected_exposure_receipts'),('replans.csv','replans'),('macro_transitions.csv','macros')]:
        old.write_csv(output/name,[attach(run,row) for run,data in details.items() for row in data[field]],['run_id','paid_step'])
    save('first_mechanism_witnesses.json',witnesses);save('findings.json',findings);save('main_gate_witnesses.json',gate)
    summary=dict(schema='article.exposure_comparison_snapshot.v1',created_utc=datetime.now(timezone.utc).isoformat(),
        arms=list(ARMS),arm_records=24,paired_cells=12,within_arm_contrasts=12,layout_units=3,metric_version=VERSION,
        status_counts={arm:dict(Counter(r['status'] for r in rows if r['arm']==arm)) for arm in ARMS},
        actual_reserved_attempts={arm:sum(r['online_status'] is not None for r in rows if r['arm']==arm) for arm in ARMS},
        across_arm_common_metric_pairs=sum(r['pair_kind']=='across_arm' and r['quality_status']=='motion_complete_common_derived_metric' for r in pairs),
        within_arm_common_metric_pairs=sum(r['pair_kind']=='within_arm' and r['quality_status']=='motion_complete_common_derived_metric' for r in pairs),
        layout_effects=ground.layout_summary(pairs),findings=len(findings),
        main_gate_evidence_counts=dict(Counter(r['status'] for r in gate['pairs'])),automatic_main_gate_approval=False,
        new_worlds=0,new_policy_runs=0,new_tsdf_integrations=0,new_quality_evaluations=0,
        scope='Second post-diagnosis common-planner ablation on three development layouts; no held-out claim.',
        limits=['Every declared slot and failure retained; missing values are blank, not zero.',
            'Original pipeline qualification, measured-motion completion and quality availability are distinct.',
            'Mechanism differences only on exact actual sensor/action prefixes with observed-anchor identity matching.',
            'Before/excluded/after exposure are nominal opportunity areas, not measured TSDF gain or additive coverage.',
            'Selected-view area is not diagnostic lookahead utility; exact-camera exclusion is reported separately.',
            'Observed instance counts are frontend identities, not truth-certified discovered facility counts.',
            'Same-class peer fields describe recorded belief use, not all physically available sharing opportunities.',
            'Incomplete layout sets have no pooled mean; all signed layout effects are retained.',
            'Recorded wall time includes the original concurrent workload, not an isolated benchmark.'])
    save('summary.json',summary)
    for index,(_,payload) in enumerate(inputs.captured.items()):
        with (output/f'captured_ledger_{index:02d}.json').open('xb') as stream:stream.write(payload)
    files={p.name:dict(bytes=p.stat().st_size,sha256=old.digest(p.read_bytes())) for p in output.iterdir() if p.is_file()}
    sources={str(Path(p).relative_to(ROOT)):old.digest(Path(p).read_bytes())
        for p in (__file__,ground.__file__,old.__file__)}
    save('manifest.json',dict(schema='article.exposure_comparison_manifest.v1',files=files,
        input_sha256=inputs.pins,source_sha256=sources))
    require(sum(p.stat().st_size for p in output.iterdir()) <= MAX_OUTPUT_BYTES,'compact snapshot exceeds 10MiB')
    return summary


def analyze(protocol_path, output, exposure_reviews, ground_reviews):
    require(not output.exists(), 'no snapshot overwrite')
    require(output.resolve().is_relative_to(STAGE.resolve()), 'external article-stage output only')
    inputs=old.Inputs()
    for path in (__file__,ground.__file__,old.__file__):inputs.read(Path(path))
    protocol=inputs.json(protocol_path);protocol_sha=old.digest(inputs.read(protocol_path))
    pin=protocol['paired_baseline_protocol'];baseline=inputs.json(old.safe(ROOT,pin['path']),pin=pin['sha256'])
    validate_pairing(protocol,baseline)
    for source in (protocol,baseline):verify_sources(inputs,source)
    verify_source_relationship(protocol['source_sha256'],baseline['source_sha256'])
    new_entries,new_phase=ground.ledger_entries(inputs,protocol,protocol_sha)
    base_entries,base_phase=ground.ledger_entries(inputs,baseline,pin['sha256'])
    rows=[];traces={};details={};findings=[]
    for run_id,slot in sorted(protocol['slots'].items()):
        previous=slot['paired_baseline_run_id']
        for arm,name,entries,phase,reviews,source,source_sha in (
            ('GroundV2',previous,base_entries,base_phase,ground_reviews,baseline,pin['sha256']),
            ('ExposureV3',run_id,new_entries,new_phase,exposure_reviews,protocol,protocol_sha)):
            entry=entries.get(name);row=blank_row(arm,name,slot,entry);rows.append(row)
            if entry is None or entry['status']=='reserved':continue
            try:
                ready=load_terminal(inputs,phase,reviews,row,entry,source,source_sha)
                if ready:
                    trace=ground.read_trace(inputs,*ready)
                    detail=trace_details(inputs,*ready,arm,trace)
                    traces[name]=trace;details[name]=detail
                    row.update(trace_status='sealed_trace_available',
                        **{k:detail[k] for k in ('observed_instance_count','qualified_category_instance_count',
                           'first_recorded_peer_step','first_informative_peer_step','macro_count')},
                        replans=len(detail['replans']),final_budget_remaining=row['budget']-row['paid_actions'])
            except (OSError,ValueError,KeyError,TypeError) as exc:
                row.update(status='analysis_binding_error',quality_measurement_available=False,review_passed=False,error=str(exc),trace_status='unavailable')
                row.update({k:None for k in (*METRICS,*COSTS)})
                traces.pop(name,None);details.pop(name,None)
            if row['status'] in ('analysis_binding_error','review_findings','failed_attempt_reviewed'):
                findings.append(dict(run_id=name,arm=arm,status=row['status'],error=row.get('error'),failure=row.get('failure_detail')))
    pairs,witnesses=pair_rows(rows,traces)
    return save_snapshot(output,rows,traces,details,pairs,witnesses,inputs,findings)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--protocol',type=Path,default=ROOT/'configs/virtual3d/article_exposure_ablation_v3_20260928.json')
    parser.add_argument('--exposure-reviews',type=Path,default=STAGE/'episode_reviews_exposure_v3')
    parser.add_argument('--ground-reviews',type=Path,default=STAGE/'episode_reviews_ground_v2')
    parser.add_argument('--output',type=Path)
    parser.add_argument('--snapshot',action='store_true',help='Explicit caller release; performs no experiments')
    args=parser.parse_args()
    if not args.snapshot or args.output is None:parser.error('require explicit --snapshot and fresh --output')
    print(json.dumps(analyze(args.protocol,args.output,args.exposure_reviews,args.ground_reviews),ensure_ascii=False))


if __name__=='__main__':main()
