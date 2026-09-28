#!/usr/bin/env python3
"""Analyze all four declared V26 autonomous tasks, saved evidence only.

No world, mapper, policy, TSDF or metric evaluator is imported or executed.
Mission eligibility remains independent of missing detections and instance
association diagnostics. Input schemas are explicit; failures are not omitted.
"""
import os
os.environ['PYTHONDONTWRITEBYTECODE']='1'
os.environ['OMP_NUM_THREADS']='1'
os.environ['OPENBLAS_NUM_THREADS']='1'
import argparse
from collections import Counter
from copy import deepcopy
import csv
import gzip
import io
import shutil
import traceback
import zipfile
from time import perf_counter
import hashlib
import json
import math
from pathlib import Path
import sys
sys.dont_write_bytecode=True
ROOT=Path(__file__).resolve().parents[1]
SOURCE=ROOT/'audit_results/observed_autonomous_v26_20260916'
OUTPUT=ROOT/'audit_results/observed_autonomous_v26_analysis_20260916'
PROTOCOL=ROOT/'docs/research/V26_AUTONOMOUS_ANALYSIS_PROTOCOL_20260916.md'
CASE_SPEC=((0,'A_complex_B_simple','G'),(1,'A_complex_B_simple','S'),
           (2,'A_simple_B_complex','G'),(3,'A_simple_B_complex','S'))
PAIR_SPEC=((0,1),(2,3))
PROTOCOL_SHA='4ce70a867876329c4eb2c9bcc3a6c52466639f3adf655543bb15011197d14488'
CAP=2*1024*1024
RESERVE=64*1024*1024


def require(condition,message):
    if not condition:raise ValueError(message)


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream,'sha256').hexdigest()


def near(a,b,label):
    require(isinstance(a,(int,float)) and not isinstance(a,bool) and math.isfinite(a),label+' is not finite')
    require(math.isclose(a,b,rel_tol=1e-10,abs_tol=1e-10),label+' arithmetic mismatch')


def audited_v23_outline(metric):
    """Audit stored metric arithmetic only, never evaluate a mesh or infer Q."""
    require(metric['contract']=='facility-three-projection-outline-v23-1','Unexpected frozen metric')
    require(metric['mission_asset_count']==2 and [x['id'] for x in metric['instances']]==[0,1],
            'Both task facilities must be retained')
    coverage=metric['coverage_2d']
    require(0<=coverage<=1,'Invalid recorded C')
    for threshold in ('02cm','05cm','10cm'):
        values=[];boundary=[]
        for instance in metric['instances']:
            require(set(instance['projections'])=={'xy','xz','yz'},'Missing orthogonal projection')
            projected=[];f_scores=[]
            for projection in instance['projections'].values():
                f1=projection[threshold]['f1'];iou=projection['iou']
                require(0<=f1<=1 and 0<=iou<=1,'Projection metric outside [0,1]')
                projected.append(min(f1,iou));f_scores.append(f1)
            q=sum(projected)/3;f=sum(f_scores)/3
            near(instance[threshold]['outline_quality'],q,'instance Q')
            near(instance[threshold]['outline_f1'],f,'instance boundary F1')
            if instance['missing']:near(q,0.,'missing facility Q')
            values.append(q);boundary.append(f)
        q=sum(values)/2
        near(metric[threshold]['outline_macro_quality'],q,'all-facility Q')
        near(metric[threshold]['outline_macro_f1'],sum(boundary)/2,'all-facility boundary F1')
        near(metric[threshold]['joint_outline'],coverage*q,'J=CQ')
    return dict(C=coverage,Q=metric['05cm']['outline_macro_quality'],
                J=metric['05cm']['joint_outline'],missing=metric['missing_asset_count'])


def paired_numeric_difference(g,s):
    """Descriptive deltas; qualification is a separate mandatory caller gate."""
    if any(g.get(key) is None or s.get(key) is None for key in ('C','Q','J')):
        return dict(delta_C=None,delta_Q=None,delta_J=None,relative_delta_J=None)
    delta=s['J']-g['J']
    return dict(delta_C=s['C']-g['C'],delta_Q=s['Q']-g['Q'],delta_J=delta,
                relative_delta_J=delta/g['J'] if g['J']>0 else None)


def digest(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()


def inside(folder,name):
    path=(folder/name).resolve()
    require(path.is_relative_to(folder.resolve()),'Artifact path escapes evidence root')
    return path


def verify_inventory(folder,hash_bytes=True):
    inventory=read(folder/'artifact_hashes.json')
    actual={str(p.relative_to(folder)) for p in folder.rglob('*') if p.is_file() and p!=folder/'artifact_hashes.json'}
    require(set(inventory)==actual,'Artifact set differs: '+str(folder))
    if hash_bytes:
        for name,expected in inventory.items():require(sha(inside(folder,name))==expected,'Artifact hash differs: '+name)
    return inventory


def read_gzip(path):
    with gzip.open(path,'rt') as stream:return json.load(stream)


def source_check(source,allow_stopped):
    manifest=read(source/'manifest.json');complete=manifest['status']=='complete'
    require(complete or (allow_stopped and manifest['status'].startswith('stopped_')),'Only complete or explicitly stopped evidence is accepted')
    require(not any(c[k]=='running' for c in manifest['cases'] for k in ('physical_status','replay_status')),'An original process is still active')
    require([(c['index'],c['assignment'],c['mode']) for c in manifest['cases']]==list(CASE_SPEC),'Declared four-case matrix changed')
    cfg=manifest['config'];require(cfg['parent']=='D25-P00' and cfg['budget']==400 and cfg['replan_interval']==5 and cfg['candidate_limit']==12,'Frozen public protocol differs')
    require(cfg['sensor_model']=='iid_025px' and cfg['noise_seed']==1901,'Frozen sensor/seed differs')
    require(sha(PROTOCOL)==PROTOCOL_SHA and manifest['source_sha256'].get(str(PROTOCOL.relative_to(ROOT)))==PROTOCOL_SHA,'Analysis protocol was not frozen before collection')
    for field in ('source_sha256','input_sha256'):
        for name,expected in manifest[field].items():require(sha(ROOT/name)==expected,'Frozen '+field+' mismatch: '+name)
    require(sha(source/'sources.zip')==manifest['source_archive_sha256'],'Original source archive mismatch')
    with zipfile.ZipFile(source/'sources.zip') as z:
        require(set(z.namelist())==set(manifest['source_sha256']),'Source archive membership')
        for name,expected in manifest['source_sha256'].items():require(hashlib.sha256(z.read(name)).hexdigest()==expected,'Archived source bytes: '+name)
    inventory=verify_inventory(source) if complete else {}
    evidence={str((source/'manifest.json').relative_to(ROOT)):sha(source/'manifest.json')}
    if complete:evidence.update({str(inside(source,n).relative_to(ROOT)):h for n,h in inventory.items()})
    return manifest,evidence,dict(complete_batch=complete,root_inventory_verified=complete,
        root_artifacts=len(inventory),source_count=len(manifest['source_sha256']),
        source_archive_sha256=manifest['source_archive_sha256'],protocol_sha256=PROTOCOL_SHA)


def recovery_records(manifest,evidence):
    """Count preserved failed attempts without promoting them to verification.

    Only the externally recorded, single identical case-01 capacity retry is
    admitted. Read the recovery chain, not any current task performance data.
    """
    histories=manifest.get('operational_recovery_history',[])
    require(len(histories)<=1,'Unexpected additional recovery; explicit review required')
    records=[]
    def checked(path,expected=None):
        path=path.resolve();require(path.is_relative_to(ROOT),'Recovery reference outside project')
        h=sha(path)
        if expected is not None:require(h==expected,'Recovery evidence SHA: '+str(path))
        evidence[str(path.relative_to(ROOT))]=h
        return read(path)
    for history in histories:
        require(history['case']==1 and history['retry_limit']==1,'Only the declared case01 single retry is supported')
        require(history['failed_replay_paid_actions']==400 and history['failed_replay_process_id']==364228,'Preserved failed attempt identity/cost differs')
        receipt_path=inside(ROOT,history['receipt']);failed_path=inside(ROOT,history['failed_attempt'])
        receipt=checked(receipt_path,history['receipt_sha256'])
        failed=checked(failed_path,history['failed_attempt_sha256'])
        require(receipt['status']=='one_identical_full_replay_retry_scheduled','Undeclared recovery procedure')
        require(receipt['case']==failed['case']==history['case'] and failed['replay'] is True,'Recovery case/type differs')
        require(receipt['failed_replay_process_id']==failed['process_id']==history['failed_replay_process_id'],'Failed PID differs')
        require(receipt['failed_replay_paid_actions']==failed['paid_replay_actions_executed']==failed['last_logged_action']==history['failed_replay_paid_actions'],'Failed paid cost differs')
        require(receipt['failed_replay_exit_code']==failed['exit_code']!=0,'Failed attempt misclassified as success')
        require(not receipt['failed_replay_formally_verified'] and not failed['completion_scope']['formal_replay_verification_passed'],'Terminal transcription is not formal replay verification')
        require(failed['status']=='terminal_transcription_of_failed_receipt_write_not_formal_verification','Unexpected failure provenance')
        require(receipt['failed_replay_record']==history['failed_attempt'] and receipt['failed_replay_record_sha256']==history['failed_attempt_sha256'],'Receipt failure reference')
        require(not receipt['main_task_refunded'] and not receipt['frozen_source_changed'],'Recovery changed task accounting/source')
        require(receipt['unchanged_free_reserve_bytes']==64*1024*1024,'Recovery changed capacity reserve')
        require(failed['frozen_collector_sha256']==manifest['source_sha256']['scripts/run_observed_autonomous_v26.py'],'Recovery used different frozen collector')
        terminal=inside(failed_path.parent,failed['terminal_traceback_file'])
        require(sha(terminal)==failed['terminal_traceback_sha256'],'Failed terminal transcript SHA')
        evidence[str(terminal.relative_to(ROOT))]=failed['terminal_traceback_sha256']
        folder=receipt_path.parent
        before=checked(folder/'manifest_before_recovery.json',receipt['original_manifest_sha256'])
        require(failed['post_exit_read_only_observation']['manifest_sha256']==receipt['original_manifest_sha256'],'Original failed manifest identity')
        after=checked(folder/'manifest_after_recovery.json')
        expected_after=deepcopy(before)
        require(expected_after['cases'][1]['replay_status']=='running','Recovery original replay phase')
        expected_after['cases'][1]['replay_status']='unstarted'
        expected_after.setdefault('operational_recovery_history',[]).append(history)
        require(after==expected_after,'Recovery altered more than declared retry state and history')
        require(before['source_sha256']==after['source_sha256']==manifest['source_sha256'],'Frozen recovery source maps differ')
        cleanup=checked(folder/'cache_cleanup.json')
        require(not cleanup['opened_by_processes'] and cleanup['free_after']>=64*1024*1024,'Recorded capacity cleanup did not restore original reserve')
        records.append(dict(**history,failed_session_id=failed['session_id'],failed_exit_code=failed['exit_code'],
            formal_replay_verification_passed=False,cost_provenance=failed['source'],
            failure_scope=failed['completion_scope'],terminal_traceback=str(terminal.relative_to(ROOT)),
            terminal_traceback_sha256=failed['terminal_traceback_sha256'],source_and_manifest_recovery_chain_verified=True,
            interpretation='400 actually executed actions retained; operational receipt-write failure, not an additional valid result or environment sample'))
    return records


def replay_costs(endpoints,recoveries,complete):
    successful=[r for r in endpoints if r['replay_verified']]
    success_pids=[r['replay']['replay_process_id'] for r in successful]
    failed_pids=[r['failed_replay_process_id'] for r in recoveries]
    require(len(set(success_pids+failed_pids))==len(success_pids)+len(failed_pids),'Replay process double-counted or reused')
    for row in endpoints:
        failures=[r for r in recoveries if r['case']==row['index']]
        row['failed_replay_attempts']=len(failures)
        row['failed_replay_paid_actions']=sum(r['failed_replay_paid_actions'] for r in failures)
        row['all_recorded_replay_attempts']=int(row['replay_verified'])+len(failures)
        row['all_recorded_replay_paid_actions']=(row['paid_actions'] if row['replay_verified'] else 0)+row['failed_replay_paid_actions']
    verified_paid=sum(r['paid_actions'] for r in successful)
    failed_paid=sum(r['failed_replay_paid_actions'] for r in recoveries)
    if complete:require(len(successful)==4,'Complete batch lacks four successful independent replays')
    return dict(successful_formally_verified_replay_attempts=len(successful),successful_replay_process_ids=success_pids,
        successful_replay_paid_actions=verified_paid,failed_operational_replay_attempts=len(recoveries),
        failed_replay_process_ids=failed_pids,failed_replay_paid_actions=failed_paid,
        all_recorded_replay_attempts=len(successful)+len(recoveries),all_recorded_replay_paid_actions=verified_paid+failed_paid,
        complete_declared_batch=complete,additional_failed_attempts_are_not_environment_replicates=True,
        partial_batch_count_scope='Formally verified attempts plus explicitly preserved recovery failures; unrecorded partial attempts are not inferred')


def instance_diagnostics(result,snapshot):
    audit=result['evaluation_audit'];association=audit['fixed_seed_association'];issues=[]
    if snapshot['duplicate_seeded_components']:issues.append('duplicate_cleaned_component')
    if association['duplicate_reference_ids']:issues.append('duplicate_reference_assignment')
    if audit['foreign_window_nonempty_reference_ids']:issues.append('foreign_task_window_geometry')
    nonempty=0
    for row in audit['instances']:
        if not row['missing_observed_mesh']:
            nonempty+=1
            if row['fixed_seed_association']['status']!='unique':issues.append('nonempty_slot_without_unique_association_'+str(row['observed_slot']))
        if row['fixed_seed_association']['status']=='ambiguous':issues.append('ambiguous_seed_'+str(row['observed_slot']))
    return dict(instance_attribution_supported=(not issues) if nonempty else None,
        instance_attribution_issues=sorted(set(issues)),nonempty_observed_slots=nonempty,
        original_combined_instance_gate=audit['evaluation_association_and_minimum_separation_passed'],
        original_seed_association_gate=association['seed_association_gate_passed'],
        minimum_instance_separation_gate=snapshot['minimum_instance_separation_gate_passed'],
        observed_marker_support_gate=audit['observed_marker_support_gate_passed'],
        missing_seed_slots=audit['missing_seed_slots'],missing_observed_mesh_slots=audit['missing_observed_mesh_slots'],
        extra_observed_tracks=len(snapshot['extra_observed_track_receipts']),
        normal_missing_not_a_mission_disqualification=True)


def plan_pose(plan):
    selected=plan['selected']
    return None if selected is None else selected['pose']


def analyze_case(source,declared,root_verified,evidence,tables):
    index=declared['index'];folder=source/f'case_{index:02d}'
    endpoint=dict(index=index,assignment=declared['assignment'],mode=declared['mode'],
        physical_status=declared['physical_status'],replay_status=declared['replay_status'],
        C=None,Q=None,J=None,paid_actions=None,returned=None,collisions=None,mission_eligible=None,
        replay_verified=False,qualified_comparison_available=False,termination=None,instance_attribution_supported=None)
    if not folder.exists():
        require(declared['physical_status']=='unstarted','Claimed case has no evidence directory')
        endpoint['unavailable_reason']='declared_but_unstarted';return endpoint,None
    inventory=verify_inventory(folder,not root_verified)
    evidence.update({str(inside(folder,n).relative_to(ROOT)):h for n,h in inventory.items()})
    evidence[str((folder/'artifact_hashes.json').relative_to(ROOT))]=sha(folder/'artifact_hashes.json')
    if not (folder/'result.json').is_file():
        endpoint['unavailable_reason']='no_complete_endpoint'
        for name in ('failure.json','replay_failure.json','progress.json'):
            if (folder/name).is_file():endpoint[name.removesuffix('.json')]=read(folder/name)
        return endpoint,None
    result=read(folder/'result.json');summary=result['summary'];trace=result['trace']
    require(result['status']=='complete' and (result['index'],result['assignment'],result['mode'])==tuple(CASE_SPEC[index]),'Case identity mismatch')
    require(result['autonomous_policy'] and result['imposed_prefix_actions']==0 and not result['gt_service_catalogue_used'],'Unexpected imposed route/policy boundary')
    require(result['evaluation_after_last_decision'],'Evaluation timing')
    paid=result['paid_actions'];require(type(paid) is int and 0<=paid<=400,'Actual paid actions out of budget')
    require(summary['budget']==400 and summary['actual_paid_actions']==paid,'Runtime paid budget mismatch')
    require(len(trace)==paid+1==result['raw_frames'],'One initial and one packet per paid action')
    require([r['action_id'] for r in trace]==list(range(paid+1)),'Trace action sequence')
    require(trace[0]['action'] is None and trace[-1]['next_action'] is None,'Initial action/terminal command mismatch')
    require(result['trajectory_sha256']==digest([(r['action_id'],r['action'],r['pose']) for r in trace]),'Trajectory digest')
    require(trace[-1]['pose']==summary['current_pose'],'Terminal summary pose')
    returned=trace[-1]['pose']==summary['anchor']
    require(returned==result['returned_to_anchor']==trace[-1]['returned_to_anchor'],'Return position/heading mismatch')
    near(trace[-1]['coverage_2d'],result['final_coverage_2d'],'Final coverage')
    require(trace[-1]['cumulative_collisions']==result['collisions'],'Collision mismatch')
    main=result['final_main_observed'];raw=result['final_raw_secondary'];metrics=audited_v23_outline(main);raw_metrics=audited_v23_outline(raw)
    expected=(paid<=400 and main['budget_verified'] and main['primitive_budget_compliant'] and
        metrics['C']>=.8 and returned and result['collisions']==0 and not main['failed'])
    require(expected==main['eligible']==result['eligible']==raw['eligible'],'Mission eligibility changed')
    require(main['reference_signature']==raw['reference_signature']==result['evaluation_audit']['reference_signature'],'Reference differs by representation')
    near(metrics['C'],result['final_coverage_2d'],'Main C');near(raw_metrics['C'],metrics['C'],'Raw C')
    snapshot=read(inside(folder,result['measured_snapshot_file']))
    require(snapshot['snapshot_complete_before_reference'] and snapshot['inference_disabled'] and result['evaluation_audit']['inference_disabled'],'Measured-only representation contract')
    require(digest(snapshot)==result['evaluation_audit']['measurement_metadata_sha256'],'Snapshot digest')
    plans=read_gzip(inside(folder,result['plans_file']));calls=read_gzip(inside(folder,result['calls_file']))
    require(len(plans)==summary['global_plans'] and len(calls)==summary['module_calls'],'Plan/module counts')
    timing=read(folder/'timing.json');require(len(timing['planning_seconds'])==len(plans),'Plan timing vector')
    near(timing['total_planning_seconds'],sum(timing['planning_seconds']),'Total planning time')
    actual_plans=[];actual_calls=[];events=[];cue_first={};intent_paid=Counter();outbound_paid=Counter();fallback_paid=0
    active=None;replacements=0
    for action_id,row in enumerate(trace):
        require(row['file_sha256']==inventory[f'packets/{action_id:04d}.npz'],'Trace packet file SHA')
        if action_id:
            require(row['action']==trace[action_id-1]['next_action'],'Actual action differs from preceding issued command')
        event=read_gzip(inside(folder,row['audit_file']));require(digest(event)==row['decision_sha256'],'Per-step decision digest')
        require(event['action_id']==action_id and event['next_action']==row['next_action'] and event['geometry_sha256']==row['geometry_sha256'],'Per-step event/trace mismatch')
        require(len(event['cues'])==row['observed_semantic_cues'],'Cue count mismatch')
        actual_plans+=event['plans'];actual_calls+=event['module_calls'];events.append(event)
        for cue in event['cues']:
            src=json.loads(cue['source']);require(src['first_action']<=cue['action_id']<=action_id,'Future cue source')
            if cue['cue_id'] not in cue_first:
                cue_first[cue['cue_id']]=dict(cue_id=cue['cue_id'],first_observed_action=src['first_action'],first_consumed_action=action_id,class_id=cue['class_id'],source_kind=src['kind'])
        for plan in event['plans']:
            if active is not None and row['pose']!=active['pose'] and plan_pose(plan)!=active['pose']:replacements+=1
            active=plan['selected']
        guards=[c for c in event['module_calls'] if c['module']=='RPN-UQ' and c['operation']=='observed_action_and_return_guard']
        if action_id<paid:
            if active is not None:intent_paid[active['group']]+=1
            if guards and guards[-1]['reason']=='refresh_selected_target_on_current_observation':
                require(active is not None,'Target-following action without selected target');outbound_paid[active['group']]+=1
            else:fallback_paid+=1
        tables['coverage'].append(dict(index=index,mode=declared['mode'],assignment=declared['assignment'],**{k:row[k] for k in ('action_id','coverage_2d','known_public_roi_cells','cumulative_collisions','returned_to_anchor','observed_semantic_cues')}))
    require(actual_plans==plans and actual_calls==calls,'Per-step logs do not reconstruct complete plans/calls')
    require([c['call_id'] for c in calls]==list(range(1,len(calls)+1)),'Module call IDs not complete')
    if declared['mode']=='G':require(not cue_first and all(p['audit']['semantic_cue_count']==0 for p in plans),'G consumed semantic cues')
    sources=Counter();selected_sources=Counter();selected_groups=Counter();query_geometry=query_semantic=0;scored=0;max_candidates=max_queries=0
    for order,plan in enumerate(plans):
        audit=plan['audit'];candidates=plan['candidates'];selected=plan['selected'];action_id=audit['action_id']
        require(audit['mode']==declared['mode'] and audit['geometry_sha256']==trace[action_id]['geometry_sha256'],'Plan input identity')
        require(audit['candidate_limit']==12 and audit['candidate_count']==len(candidates)<=12,'Candidate capacity')
        require(sum(c['group'].startswith('coverage_') for c in candidates)<=4 and sum(not c['group'].startswith('coverage_') for c in candidates)<=8,'4+8 candidate capacity')
        require(not audit['route_catalogue_used'] and not audit['evaluation_truth_used'],'Forbidden candidate input')
        for c in candidates:
            for src in c['sources']:sources[src['kind']]+=1
        if selected is not None:
            require(selected in candidates,'Selected candidate absent from final pool');selected_groups[selected['group']]+=1
            for src in selected['sources']:selected_sources[src['kind']]+=1
        gq,sq=audit['geometric_proposal_queries'],audit['semantic_proposal_queries'];query_geometry+=gq;query_semantic+=sq;scored+=audit['scored_unique_poses'];max_queries=max(max_queries,gq+sq);max_candidates=max(max_candidates,len(candidates))
        tables['plans'].append(dict(index=index,assignment=declared['assignment'],mode=declared['mode'],plan_index=order,action_id=action_id,candidate_count=len(candidates),geometric_proposal_queries=gq,semantic_proposal_queries=sq,scored_unique_poses=audit['scored_unique_poses'],planning_seconds=timing['planning_seconds'][order],coverage_proposal_status=audit['coverage']['status'],coverage_evaluated_cells=audit['coverage']['evaluated_cells'],selected_pose=None if selected is None else selected['pose'],selected_group=None if selected is None else selected['group'],selected_sources=[] if selected is None else [s['kind'] for s in selected['sources']],selected_score=None if selected is None else selected['score']))
    feedback=[c for c in calls if c['module']=='IGCR' and c['operation']=='measured_transition_feedback']
    updates=[u for c in feedback for u in c['directional_updates']]
    guards=[c for c in calls if c['module']=='RPN-UQ' and c['operation']=='observed_action_and_return_guard']
    for c in calls:
        tables['events'].append(dict(index=index,call_id=c['call_id'],module=c['module'],operation=c['operation'],action_id=c.get('action_id'),selected_endpoint_reached=c.get('selected_endpoint_reached'),allowed=c.get('allowed'),proposed_action=c.get('proposed_action'),reason=c.get('reason'),improved_common_patches=c.get('improved_common_patches'),directional_updates=c.get('directional_updates')))
    verify_record=None;replay_ok=False
    if (folder/'verification.json').is_file():
        verify_record=read(folder/'verification.json')
        replay_ok=bool(verify_record['status']=='passed' and verify_record['independent_process'] and verify_record['fresh_world'] and verify_record['fresh_autonomous_runtime'] and not verify_record['saved_actions_used_to_drive_policy'] and verify_record['main_process_id']==timing['process_id'] and verify_record['main_process_id']!=verify_record['replay_process_id'] and verify_record['independent_paid_actions']==paid and verify_record['saved_packets_verified']==paid+1 and all(verify_record[k] for k in ('all_decisions_equal','all_packets_equal','all_meshes_and_metrics_equal')))
        require(replay_ok,'Existing replay receipt failed consistency')
    diagnostics=instance_diagnostics(result,snapshot)
    endpoint.update(**metrics,paid_actions=paid,remaining_budget=400-paid,returned=returned,collisions=result['collisions'],mission_eligible=expected,replay_verified=replay_ok,
        qualified_comparison_available=bool(expected and replay_ok and declared['physical_status']=='complete' and declared['replay_status']=='complete'),
        termination=summary['terminal_reason'],failed=main['failed'],raw_secondary=raw_metrics,instance_diagnostics=diagnostics,
        instance_attribution_supported=diagnostics['instance_attribution_supported'],per_instance=[dict(id=x['id'],missing=x['missing'],Q=x['05cm']['outline_quality'],completed=x['completed']) for x in main['instances']],
        global_plans=len(plans),module_calls=len(calls),modules_called=summary['modules_called'],total_candidate_occurrences=sum(p['audit']['candidate_count'] for p in plans),max_candidates=max_candidates,
        geometric_proposal_queries=query_geometry,semantic_proposal_queries=query_semantic,scored_unique_pose_sum=scored,max_proposal_queries_per_plan=max_queries,
        candidate_source_occurrences=dict(sources),selected_source_occurrences=dict(selected_sources),selected_groups=dict(selected_groups),
        empty_candidate_plans=sum(not p['candidates'] for p in plans),no_positive_selected_plans=sum(p['selected'] is None for p in plans),
        coverage_proposal_status_counts=dict(Counter(p['audit']['coverage']['status'] for p in plans)),
        guard_reason_counts=dict(Counter(c['reason'] for c in guards)),
        rejected_proposal_count=None,rejected_proposal_count_status='per-proposal rejection reasons not retained; no inferred budget/safety breakdown',
        paid_action_counts=dict(Counter(r['action'] for r in trace[1:])),
        selected_intent_paid_actions=dict(intent_paid),actual_target_following_paid_actions=dict(outbound_paid),fallback_return_or_no_selected_paid_actions=fallback_paid,
        selected_endpoint_arrivals=sum(bool(c['selected_endpoint_reached']) for c in feedback),target_replacements_before_arrival=replacements,
        feedback_update_status_counts=dict(Counter(u['status'] for u in updates)),explicit_motion_rejections=sum(c['proposed_action'] is not None and not c['allowed'] for c in guards),no_action_guard_events=sum(c['proposed_action'] is None for c in guards),
        first_planner_cue_action=min((x['first_observed_action'] for x in cue_first.values()),default=None),planner_cues=list(cue_first.values()),
        first_common_measurement_marker_action=min((x['first_action'] for x in result['observed_marker_tracks']),default=None),
        G_no_cue_not_equivalent_to_no_visible_marker=declared['mode']=='G',elapsed_s=timing['elapsed_s'],total_planning_seconds=timing['total_planning_seconds'],max_planning_seconds=max(timing['planning_seconds'],default=0.),
        reference_signature=main['reference_signature'],replay=verify_record,result_sha256=sha(folder/'result.json'))
    return endpoint,dict(trace=trace,plans=plans,events=events)


def compare_history(g,s):
    if g is None or s is None:return dict(status='unavailable_incomplete_case')
    gt,st=g['trace'],s['trace'];shared=min(len(gt),len(st))
    identity_keys=('action','pose','nonsemantic','geometry_sha256')
    mismatch=next((i for i in range(shared) if any(gt[i][k]!=st[i][k] for k in identity_keys)),None)
    first_paid=next((i for i in range(1,shared) if gt[i]['action']!=st[i]['action']),None)
    kind='different_paid_action' if first_paid is not None else None
    if first_paid is None and len(gt)!=len(st):first_paid=shared;kind='one_policy_stopped_before_the_other'
    first_command=next((i for i in range(shared) if gt[i]['next_action']!=st[i]['next_action']),None)
    gp={p['audit']['action_id']:p for p in g['plans']};sp={p['audit']['action_id']:p for p in s['plans']}
    choice=next((i for i in sorted(set(gp)&set(sp)) if plan_pose(gp[i])!=plan_pose(sp[i])),None)
    decision=first_command
    hist_equal=None if decision is None else (mismatch is None or mismatch>decision)
    cues=[] if decision is None else s['events'][decision]['cues']
    cue_available=any(json.loads(c['source'])['first_action']<=decision for c in cues)
    return dict(status='checked_saved_observations',common_recorded_observations=shared,first_nonsemantic_or_geometry_history_difference=mismatch,
        first_actual_paid_action_difference=first_paid,actual_difference_kind=kind,first_next_action_command_difference_after_observation=first_command,
        first_selected_pose_difference_at_common_plan_action=choice,
        G_first_selected_pose=None if choice is None else plan_pose(gp[choice]),S_first_selected_pose=None if choice is None else plan_pose(sp[choice]),
        first_command_G=None if decision is None else gt[decision]['next_action'],first_command_S=None if decision is None else st[decision]['next_action'],
        same_observation_history_through_first_command_difference=hist_equal,observed_cue_available_before_first_command_difference=cue_available,
        first_difference_consistent_with_observed_semantic_branch=bool(hist_equal and cue_available and decision is not None),
        attribution_limit='Only the joint V26 proposal/ranking/feedback intervention is compared; no mechanism isolation, no causal claim after history divergence')


def pair_record(g,s,gh,sh):
    qualified=g['qualified_comparison_available'] and s['qualified_comparison_available']
    raw=paired_numeric_difference(g,s)
    blocks=[]
    for row in (g,s):
        if row['C'] is None:blocks.append(row['mode']+'_endpoint_unavailable')
        elif not row['mission_eligible']:blocks.append(row['mode']+'_mission_ineligible')
        if not row['replay_verified']:blocks.append(row['mode']+'_independent_replay_unverified')
    return dict(assignment=g['assignment'],G_index=g['index'],S_index=s['index'],raw_difference=raw,
        qualified_pair=bool(qualified),qualified_difference=raw if qualified else None,
        action_difference=None if g['paid_actions'] is None or s['paid_actions'] is None else s['paid_actions']-g['paid_actions'],
        qualified_comparison_blockers=blocks,instance_attribution_supported=dict(G=g['instance_attribution_supported'],S=s['instance_attribution_supported']),
        history=compare_history(gh,sh),unqualified_positive_raw_delta_is_not_valid_S_advantage=True)


def save_bytes(out,name,blob,emergency=False):
    used=sum(p.stat().st_size for p in out.rglob('*') if p.is_file())
    require(used+len(blob)+4096<=CAP-(0 if emergency else 32768),'Analysis 2 MiB cap')
    require(shutil.disk_usage(out).free-len(blob)-4096>=RESERVE,'Analysis 64 MiB reserve')
    tmp=out/(name+'.tmp')
    with tmp.open('xb') as stream:stream.write(blob);stream.flush();os.fsync(stream.fileno())
    os.replace(tmp,out/name)


def save_json(out,name,value,emergency=False):
    save_bytes(out,name,(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+'\n').encode(),emergency)


def save_csv(out,name,rows):
    fields=sorted({key for row in rows for key in row});stream=io.StringIO();writer=csv.DictWriter(stream,fieldnames=fields);writer.writeheader()
    for row in rows:writer.writerow({k:json.dumps(v,ensure_ascii=False,separators=(',',':')) if isinstance(v,(dict,list,tuple)) else v for k,v in row.items()})
    save_bytes(out,name,stream.getvalue().encode())


def figures(out,endpoints,coverage):
    os.environ.setdefault('MPLCONFIGDIR','/dev/shm/nso_v26_analysis_mpl')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    colors={'G':'#2c6ca6','S':'#cf7021'}
    fig,axs=plt.subplots(1,3,figsize=(11.2,3.7));fig.subplots_adjust(top=.78,bottom=.18,wspace=.28)
    for ax,key in zip(axs,('C','Q','J')):
        for row in endpoints:
            if row[key] is None:continue
            x=row['index']//2+(-.18 if row['mode']=='G' else .18)
            ax.bar(x,row[key],width=.32,color=colors[row['mode']],alpha=.85,
                hatch='' if row['mission_eligible'] else '//',label=row['mode'] if row['index']<2 else None)
            ax.text(x,row[key]+.015,f"{row[key]:.3f}",ha='center',fontsize=8)
        ax.set(xticks=[0,1],xticklabels=['A complex','B complex'],ylim=(0,1.08),title={'C':'Reachable coverage C','Q':'Observed outline Q','J':'Joint J = C × Q'}[key]);ax.grid(axis='y',alpha=.2);ax.set_axisbelow(True)
        if key=='C':ax.axhline(.8,color='#a33b3b',ls='--',lw=.8)
    axs[0].legend(frameon=False);fig.suptitle('V26 autonomous endpoints — all declared cases retained',fontsize=13)
    fig.text(.5,.855,'Hatching = mission ineligible; missing endpoint is not replaced by zero',ha='center',fontsize=9)
    b=io.BytesIO();fig.savefig(b,format='png',dpi=150);plt.close(fig);save_bytes(out,'endpoints.png',b.getvalue())
    fig,axs=plt.subplots(1,2,figsize=(10,3.6),sharey=True);fig.subplots_adjust(top=.80,bottom=.19,wspace=.15)
    for pi,ax in enumerate(axs):
        for mode in ('G','S'):
            rows=[r for r in coverage if r['index']//2==pi and r['mode']==mode]
            if rows:ax.plot([r['action_id'] for r in rows],[r['coverage_2d'] for r in rows],color=colors[mode],label=mode,lw=1.2)
        ax.axhline(.8,color='#a33b3b',ls='--',lw=.8);ax.set(xlim=(0,400),ylim=(0,1.02),xlabel='Paid action',title=['A complex / B simple','A simple / B complex'][pi]);ax.grid(alpha=.2);ax.legend(frameon=False)
    axs[0].set_ylabel('True evaluation coverage C');fig.suptitle('Recorded coverage only; no interpolated quality trajectory',fontsize=12)
    b=io.BytesIO();fig.savefig(b,format='png',dpi=150);plt.close(fig);save_bytes(out,'coverage.png',b.getvalue())


def execute(source,out,allow_stopped):
    manifest,evidence,receipt=source_check(source,allow_stopped)
    require(not out.exists() and not out.is_relative_to(source),'Fresh analysis directory outside acquisition required')
    require(shutil.disk_usage(ROOT).free>=RESERVE+CAP,'Analysis capacity unavailable')
    out.mkdir(parents=True);started=perf_counter()
    own_sources={str(Path(__file__).resolve().relative_to(ROOT)):sha(Path(__file__)),str(PROTOCOL.relative_to(ROOT)):PROTOCOL_SHA}
    try:
        save_json(out,'manifest.json',dict(status='running',source=str(source),source_receipt=receipt,source_sha256=own_sources))
        recoveries=recovery_records(manifest,evidence)
        tables=dict(coverage=[],plans=[],events=[]);endpoints=[];histories=[]
        for case in manifest['cases']:
            row,history=analyze_case(source,case,receipt['complete_batch'],evidence,tables);endpoints.append(row);histories.append(history)
        operational_cost=replay_costs(endpoints,recoveries,receipt['complete_batch'])
        pairs=[pair_record(endpoints[g],endpoints[s],histories[g],histories[s]) for g,s in PAIR_SPEC]
        all_available=all(r['C'] is not None for r in endpoints);qualified=all(r['qualified_comparison_available'] for r in endpoints)
        means=None
        if all_available:
            means={mode:{k:sum(r[k] for r in endpoints if r['mode']==mode)/2 for k in ('C','Q','J')} for mode in ('G','S')}
        overall=dict(all_four_endpoints_available=all_available,all_four_qualified_for_comparison=qualified,
            mean_endpoints=means,raw_mean_difference=None if means is None else paired_numeric_difference(means['G'],means['S']),
            qualified_mean_difference=paired_numeric_difference(means['G'],means['S']) if qualified else None,
            mission_eligible_count=sum(r['mission_eligible'] is True for r in endpoints),replay_verified_count=sum(r['replay_verified'] for r in endpoints),
            new_main_attempts=manifest['main_attempts_started'],total_completed_main_paid_actions=sum(r['paid_actions'] or 0 for r in endpoints),
            total_verified_replay_paid_actions=sum(r['paid_actions'] for r in endpoints if r['replay_verified']),
            failed_operational_replay_paid_actions=operational_cost['failed_replay_paid_actions'],
            all_recorded_replay_attempts=operational_cost['all_recorded_replay_attempts'],
            all_recorded_replay_paid_actions=operational_cost['all_recorded_replay_paid_actions'],
            completed_main_plus_all_recorded_replay_paid_actions=sum(r['paid_actions'] or 0 for r in endpoints)+operational_cost['all_recorded_replay_paid_actions'])
        result=dict(status='complete_saved_evidence_analysis' if receipt['complete_batch'] else 'complete_stopped_batch_analysis',
            source_receipt=receipt,endpoints=endpoints,pairs=pairs,overall=overall,
            operational_recovery_history=recoveries,replay_execution_cost=operational_cost,
            new_physical_actions=0,new_sensor_packets=0,new_mapper_or_tsdf_replays=0,new_policy_calls=0,new_mesh_evaluations=0,
            notes=['All four declared cases retained; normal missing facilities score zero and never alter mission eligibility.',
             'Instance attribution diagnostics remain separate from mission qualification and displayed numeric deltas.',
             'A positive raw delta from an ineligible pair is not a qualified S advantage.',
             'One parent, one noise seed, two assignments; independent process replays are not independent environments.',
             'Failed operational replay actions remain in total execution cost. A terminal transcription is not a successful verification; the unchanged collector aggregate counts successful replays only.',
             'V26 S is an untrained visible-marker template prototype, not a validated conditional Q head or natural semantic frontend.',
             'Candidate source counts are occurrences; multi-source candidates can contribute more than one source.',
             'Arrivals and local feedback proxies do not establish per-option Q improvement; Q is measured only at the terminal snapshot.',
             'G has no planner semantic memory; absence of G cues does not imply no marker was visible to its sensors.'])
        save_csv(out,'endpoints.csv',endpoints);save_csv(out,'paired_comparisons.csv',[{k:v for k,v in p.items() if k!='history'} for p in pairs])
        for key,rows in tables.items():save_csv(out,key+'.csv',rows)
        figures(out,endpoints,tables['coverage'])
        for name,expected in evidence.items():require(sha(ROOT/name)==expected,'Input changed during analysis: '+name)
        for name,expected in {**manifest['source_sha256'],**manifest['input_sha256'],**own_sources}.items():require(sha(ROOT/name)==expected,'Source changed during analysis: '+name)
        save_json(out,'result.json',result)
        save_json(out,'input_sha256.json',evidence)
        stream=io.BytesIO()
        with zipfile.ZipFile(stream,'w',zipfile.ZIP_DEFLATED) as z:
            for name in own_sources:z.writestr(name,(ROOT/name).read_bytes())
        save_bytes(out,'sources.zip',stream.getvalue())
        save_json(out,'manifest.json',dict(status='complete',source=str(source),source_receipt=receipt,source_sha256=own_sources,
            elapsed_s=perf_counter()-started,input_hashes_rechecked=True,output_bytes_before_final_manifest=sum(p.stat().st_size for p in out.iterdir())))
        print(json.dumps(dict(status=result['status'],mission_eligible=overall['mission_eligible_count'],replays=overall['replay_verified_count'],output=str(out))),flush=True)
    except BaseException as error:
        failure=dict(status='failed',error=repr(error),traceback=traceback.format_exc(),original_evidence_modified=False)
        print(json.dumps(failure),file=sys.stderr)
        for tmp in out.glob('*.tmp'):tmp.rename(tmp.with_name('interrupted_'+tmp.name))
        save_json(out,'failure.json',failure,True);save_json(out,'manifest.json',dict(status='failed',source=str(source),source_sha256=own_sources),True)
        raise
    finally:
        inventory={str(p.relative_to(out)):sha(p) for p in sorted(out.rglob('*')) if p.is_file() and p.name!='artifact_hashes.json'}
        save_json(out,'artifact_hashes.json',inventory,True)


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--run',action='store_true')
    parser.add_argument('--source',type=Path,default=SOURCE);parser.add_argument('--output',type=Path,default=OUTPUT)
    parser.add_argument('--allow-stopped',action='store_true',help='Read only an explicitly stopped, nonactive batch; preserve missing declared cases')
    args=parser.parse_args()
    if not args.run:parser.error('Explicit --run required after root confirms the batch is complete or stopped')
    execute(args.source.resolve(),args.output.resolve(),args.allow_stopped)

if __name__=='__main__':main()
