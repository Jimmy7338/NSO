#!/usr/bin/env python3
"""Compact read-only GroundOn/Off comparisons, retaining every declared slot.

No World, TSDF, controller replay or quality measurement is invoked. Only
terminal, sealed and reviewed records can supply scores or decision traces.
The caller must explicitly request a snapshot after study-release approval.
"""
import argparse
from collections import Counter,defaultdict
from copy import deepcopy
from datetime import datetime,timezone
import hashlib
import json
import math
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from scripts import analyze_article_comparisons_20260928 as old

STAGE=ROOT/'audit_results/article_stage_20260928'
VERSION='article.common_numeric_face_evaluation.v1'
METRICS=old.METRICS
COSTS=old.COSTS
ARMS=('GroundOff','GroundOn')
METHODS=('G','B','S','NBV')
HEAVY={'candidate_set_differences','posterior_rows','score_rows','scope'}
require=old.require


def sealed_folder(inputs,path,schemas=None):
    manifest_path=path/'manifest.json'
    if not manifest_path.is_file():return None
    try:payload=manifest_path.read_bytes();json.loads(payload)
    except (FileNotFoundError,json.JSONDecodeError):return None
    manifest=inputs.json(manifest_path,pin=old.digest(payload))
    if schemas is not None:require(manifest.get('schema') in schemas,'unexpected sealed-folder schema: '+str(path))
    for name,pin in manifest['files'].items():
        data=inputs.read(old.safe(path,name),pin=pin['sha256'])
        require(len(data)==pin['bytes'],'sealed-folder byte mismatch: '+name)
    return manifest


def ledger_entries(inputs,protocol,protocol_sha):
    phase=old.safe(ROOT,protocol['output_relative_path']);path=phase/'start_ledger.json'
    if not path.is_file():return {},phase
    ledger=inputs.json(path,mutable=True)
    require(ledger['protocol_sha256']==protocol_sha and ledger['slots']==protocol['slots'],'ledger/protocol binding')
    entries={r['run_id']:r for r in ledger['entries']}
    require(len(entries)==len(ledger['entries']) and set(entries)<=set(protocol['slots']),'unique declared ledger reservations')
    return entries,phase


def blank_row(arm,run_id,slot,entry):
    return dict(arm=arm,run_id=run_id,scene_id=slot['scene_id'],method=slot['method'],budget=slot['budget'],noise_seed=slot['noise_seed'],
        status=old.status_without_review(entry),online_status=None if entry is None else entry['status'],
        original_end_to_end_qualified=None if entry is None else entry.get('qualified',False),
        review_status=None,review_passed=None,motion_completion_verified=None,quality_measurement_available=False,
        metric_version=None,quality_mode=None,trace_status='unavailable',trace_review_scope=None,
        **{k:None for k in (*METRICS,*COSTS)},reference_fingerprint=None)


def apply_metrics(row,metrics,*,version,motion,mode):
    require(version==VERSION,'refuse mixed original/derived metric versions')
    require([r['instance_id'] for r in metrics['per_instance']]==[0,1,2,3],'four fixed devices required')
    f1=sum(r['f1'] for r in metrics['per_instance'])/4
    require(abs(metrics['macro_f1']-f1)<=1e-12 and abs(metrics['Q']-f1)<=1e-12
        and abs(metrics['J_nav']-metrics['C_nav']*f1)<=1e-12,'macro F1 and joint arithmetic')
    row.update(C_nav=metrics['C_nav'],P=metrics['macro_precision'],R=metrics['macro_completeness'],
        F1=metrics['macro_f1'],J_nav=metrics['J_nav'],metric_version=version,
        motion_completion_verified=bool(motion),quality_measurement_available=True,quality_mode=mode,
        reference_fingerprint=metrics['reference_fingerprint'])


def apply_costs(row,motion,result):
    actions=motion.get('action_counts',{})
    row.update(paid_actions=motion.get('paid_actions'),path_length_m=motion.get('translation_m'),
        turns=actions.get('turn_left',0)+actions.get('turn_right',0),explicit_observe_actions=actions.get('observe',0),
        observation_frames=motion.get('rgbd_frames'),elapsed_episode_s=result.get('elapsed_s'))
    row.update({key:result.get('timings',{}).get(key) for key in ('planning_s','evidence_s','acquisition_s','mapping_s')})


def read_trace(inputs,episode,manifest):
    # Reuse the established packet comparator: exact physical NPY entries,
    # excluding only run-namespaced frame_id (never calibration/time/pose).
    loaded=old.read_episode(inputs,episode,manifest)
    starts=defaultdict(dict);changes=defaultdict(dict)
    for event in loaded['association_diagnostics']['transitions']:
        starts[event['paid_step']][event['instance_id']]=event['marker_anchor_world_m']
    for event in loaded['association_diagnostics']['anchor_changes']:
        changes[event['paid_step']][event['instance_id']]=event['current']
    anchors={}
    trajectory=[]
    for step,row in sorted(loaded['steps'].items()):
        anchors.update(starts[step]);anchors.update(changes[step])
        row['observed_anchors']={key:list(anchors[key]) for key in row['beliefs']}
        packet=json.loads(old.sealed_read(inputs,episode,manifest,f'packets/{step:03d}_receipt.json'))
        receipt=packet['execution'];pose=receipt['pose_xyyaw_rad']
        trajectory.append(dict(paid_step=step,x_m=pose[0],y_m=pose[1],yaw_rad=pose[2],
            actual_sensor_action=receipt['action'],collision=receipt['collision'],
            observation_sha256=packet['observation_sha256'],next_controller_action=row['action']))
    return dict(**{key:loaded[key] for key in ('result','steps','mechanism')},trajectory=trajectory)


def load_on(inputs,phase,review_root,row,entry,protocol_sha):
    folder=review_root/row['run_id'];manifest=sealed_folder(inputs,folder,{'article.ground_episode_review_manifest.v2'})
    if manifest is None:return None
    report=inputs.json(folder/'review.json')
    require(report['run_id']==row['run_id'] and report['online_status']==entry['status'],'ground review run/terminal identity')
    row.update(review_status=report['status'],review_passed=report.get('all_checks_passed',False))
    if report['status']!='reviewed' or not report.get('all_checks_passed'):
        row['status']='failed_attempt_reviewed' if report['status']=='failed_attempt_preserved' else 'review_findings'
        row['failure_detail']=report.get('failure',report.get('failures',report.get('error')))
        return None
    episode=phase/'episodes'/row['run_id']
    artifacts=inputs.json(episode/'artifact_manifest.json',pin=entry['artifact_manifest_sha256'])
    require(entry['artifact_manifest_sha256']==manifest['input_episode_manifest_sha256']==report['input_manifest_sha256'],
        'ground review/ledger/artifact seal')
    require(artifacts['protocol_sha256']==report['protocol_sha256']==manifest['protocol_sha256']==protocol_sha,'ground protocol seal')
    require(report['method']==row['method'] and report['scene_id']==row['scene_id'] and report['phase']=='ablation','ground reviewed slot')
    require(report['qualified']==entry['qualified'],'ground qualification unchanged')
    evaluation=json.loads(old.sealed_read(inputs,episode,artifacts,'evaluation.json'))
    require(evaluation['metrics']==report['metrics'] and evaluation['metric_version']==report['metric_version']==VERSION,
        'ground reviewed derived metric binding')
    row['status']='reviewed_qualified' if report['qualified'] else 'reviewed_unqualified'
    apply_metrics(row,report['metrics'],version=report['metric_version'],motion=report['qualified'],mode='new_online_derived_measurement')
    result=json.loads(old.sealed_read(inputs,episode,artifacts,'result.json'))
    require(old.digest(old.sealed_read(inputs,episode,artifacts,'result.json'))==entry['result_sha256'],'ground result ledger pin')
    apply_costs(row,report['physics_metrics'],result)
    row['trace_review_scope']='complete_independent_ground_review'
    return episode,artifacts


def load_off(inputs,phase,review_root,common_root,plan,plan_sha,row,entry,protocol_sha):
    review_dir=review_root/row['run_id'];review_manifest=sealed_folder(inputs,review_dir,{'article.episode_review_manifest.v1'})
    if review_manifest is None:return None
    report=inputs.json(review_dir/'review.json')
    require(report['run_id']==row['run_id'] and report['online_status']==entry['status'],'baseline review run/terminal identity')
    row.update(review_status=report['status'],review_passed=report.get('all_checks_passed',False))
    permitted=report.get('all_checks_passed') and report['status'] in ('reviewed','failed_attempt_preserved')
    if not permitted:
        row.update(status='review_findings',failure_detail=report.get('failures',report.get('error')))
        return None
    if report['status']=='failed_attempt_preserved':
        row.update(status='original_failed_awaiting_derived_measurement',failure_detail=report.get('failure'))
    else:row['status']='awaiting_common_derived_measurement'
    folder=common_root/row['run_id'];prepared=folder/'prepared';measurement=folder/'measurement'
    prep_manifest=sealed_folder(inputs,prepared,{'article.posthoc_forensic_manifest.v1'})
    measure_manifest=sealed_folder(inputs,measurement,{'article.common_numeric_measurement_manifest.v1',None})
    if prep_manifest is None or measure_manifest is None:return None
    frozen=inputs.json(prepared/'frozen_plan.json')
    require(old.digest(inputs.read(prepared/'frozen_plan.json'))==plan_sha and frozen==plan,'same predeclared common measurement plan')
    snapshot=inputs.json(prepared/'input_snapshot.json');captured=inputs.json(prepared/'original_ledger_entry.json')
    require(captured==entry and snapshot['run_id']==row['run_id'] and snapshot['frozen_plan_sha256']==plan_sha,
        'posthoc terminal baseline/plan binding')
    result=inputs.json(measurement/'result.json');attempt=inputs.json(measurement/'attempt.json')
    require(attempt['frozen_plan_sha256']==plan_sha and attempt['input_snapshot_sha256']==old.digest(inputs.read(prepared/'input_snapshot.json')),
        'measurement attempted against sealed forensic input')
    require(result['run_id']==row['run_id'] and result['measurement_version']==plan['measurement_version']==VERSION,
        'same derived metric version')
    require(result['original_end_to_end_status']==entry['status']==snapshot['original_end_to_end_status'] and
        result['original_end_to_end_qualified']==entry.get('qualified',False)==snapshot['original_end_to_end_qualified'],
        'original failed/unqualified status must not be repaired by derived quality')
    require(result['original_prediction_seal']==snapshot['original_prediction_seal'],'original prediction seal carried into supplement')
    row['common_measurement_status']=result['status'];row['motion_completion_verified']=result['motion_completion_verified']
    if not result['quality_measurement_available']:
        row.update(status='common_derived_quality_unavailable',failure_detail=result.get('error',row.get('failure_detail')))
        return None
    require(result['status']=='derived_quality_available' and result['reference_manifest']==plan['references'][row['scene_id']],
        'common measurement reference')
    require(result['adapter']==snapshot['adapter'] and result['adapter']['schema']=='article.prediction_numeric_face_adapter.v1',
        'prepared and measured adapter equality')
    require(result['mode'] in ('verified_exact_input_reuse','new_derived_measurement'),'explicit reuse or new derived measurement mode')
    if result['mode']=='verified_exact_input_reuse':
        proof=inputs.json(measurement/'reuse_proof.json')
        require(old.safe(ROOT,proof['source_review'])==review_dir.resolve(),'exact reuse source review directory')
        pinned=proof['source_review_manifest'];payload=inputs.read(review_dir/'manifest.json',pin=pinned['sha256'])
        require(len(payload)==pinned['bytes'] and result['metrics']==report['metrics'] and result['new_surface_evaluations']==0,
            'exact reuse original reviewed metric and zero evaluator calls')
        inputs.read(phase/'episodes'/row['run_id']/'evaluation.json',pin=proof['source_evaluation_sha256'])
    else:require(result['new_surface_evaluations']==1,'one declared derived evaluation call')
    if entry['status']=='attempt_failed':
        require(snapshot['forensic_capture_after_terminal_failure'] is True and snapshot['original_artifact_manifest_available'] is False
            and result['original_failed_attempt_remains_failed'] is True,'failed original kept separate from later measurement')
        row['status']='original_failed_motion_complete_derived_quality' if result['motion_completion_verified'] else 'original_failed_derived_quality_motion_unverified'
        row['trace_review_scope']='terminal_failure_forensic_capture_with_motion_audit_only'
    else:
        require(report['status']=='reviewed' and report['input_manifest_sha256']==entry['artifact_manifest_sha256']
            and report['protocol_sha256']==protocol_sha,'baseline complete independent review')
        row['status']='reviewed_common_derived_quality';row['trace_review_scope']='complete_independent_v1_review_plus_common_derived_metric'
    episode=old.safe(ROOT,snapshot['original_episode']);artifacts=dict(files=snapshot['input_files'])
    original_result=json.loads(old.sealed_read(inputs,episode,artifacts,'result.json'))
    apply_costs(row,snapshot['motion_completion'],original_result)
    apply_metrics(row,result['metrics'],version=result['measurement_version'],motion=result['motion_completion_verified'],mode=result['mode'])
    return episode,artifacts


def observed_identity_pairs(left,right):
    distances={(a,b):sum((x-y)**2 for x,y in zip(av,bv))**.5
        for a,av in left.items() for b,bv in right.items()}
    result={}
    for a in left:
        candidates=[b for b in right if distances[a,b]<=.25]
        if len(candidates)==1 and sum(distances[x,candidates[0]]<=.25 for x in left)==1:
            result[candidates[0]]=a
    return result


def compact_comparison(left,right):
    # Current/previously observed marker anchors only; never class or asset ID.
    a=dict(result=left['result'],steps={});b=dict(result=right['result'],steps={})
    identity_counts={}
    for step in sorted(set(left['steps'])|set(right['steps'])):
        if step not in left['steps'] or step not in right['steps']:
            if step in left['steps']:a['steps'][step]=left['steps'][step]
            if step in right['steps']:b['steps'][step]=right['steps'][step]
            continue
        l=left['steps'][step];r=right['steps'][step]
        mapping=observed_identity_pairs(l['observed_anchors'],r['observed_anchors'])
        identity_counts[step]=(len(l['beliefs'])-len(mapping),len(r['beliefs'])-len(mapping))
        for target,row,rename in ((a,l,{key:key if key in mapping.values() else 'left_unmatched:'+key for key in l['beliefs']}),
                                  (b,r,{key:mapping.get(key,'right_unmatched:'+key) for key in r['beliefs']})):
            changed=dict(row)
            changed['beliefs']={rename[key]:value for key,value in row['beliefs'].items()}
            changed['forecasts']={rename.get(key,'unknown:'+key):value for key,value in row['forecasts'].items()}
            changed['options']={(key[0],key[1],key[2],rename.get(key[3],key[3])):value for key,value in row['options'].items()}
            target['steps'][step]=changed
    comparison=old.compare_episode(a,b)
    small={key:value for key,value in comparison.items() if key not in HEAVY}
    for field in ('xy','full_pose'):
        small['first_'+field+'_divergence_paid_step']=next((l['paid_step'] for l,r in
            zip(left.get('trajectory',[]),right.get('trajectory',[]))
            if any(abs(l[k]-r[k])>1e-12 for k in ('x_m','y_m')) or
                (field=='full_pose' and abs((l['yaw_rad']-r['yaw_rad']+math.pi)%(2*math.pi)-math.pi)>1e-12)),None)
    prefix=small['common_observation_prefix_frames']
    small['unmatched_left_instance_prefix_frames']=sum(v[0] for k,v in identity_counts.items() if k<prefix)
    small['unmatched_right_instance_prefix_frames']=sum(v[1] for k,v in identity_counts.items() if k<prefix)
    small['identity_rule']='bidirectionally unique observed marker anchors within 0.25m at each paid step; no class or GT; unmatched identities not compared'
    first=small['first_action_divergence_paid_step']
    witness=None
    if first is not None and first-1 in left['steps'] and first-1 in right['steps']:
        l=left['steps'][first-1];r=right['steps'][first-1]
        witness=dict(decision_paid_step=first-1,executed_action_paid_step=first,left_next_action=l['action'],right_next_action=r['action'],
            same_physical_observation=l['observation_content_sha']==r['observation_content_sha'])
    events=dict(first_action_witness=witness,
        first_changed_score=next((r for r in comparison['score_rows'] if r['score_changed']),None),
        first_changed_posterior=next((r for r in comparison['posterior_rows'] if r['structure_posterior_changed']),None),
        first_changed_target=next((r for r in comparison['candidate_set_differences'] if not r['selected_target_equal']),None),
        first_changed_rank=next((r for r in comparison['candidate_set_differences'] if not r['unique_target_score_order_equal']),None))
    return small,events


def pair_rows(rows,traces):
    by={(r['arm'],r['scene_id'],r['method']):r for r in rows};pairs=[];events={}
    scenes=sorted({r['scene_id'] for r in rows})
    specs=[]
    for scene in scenes:
        specs.extend((f'GroundOn-GroundOff/{m}',('GroundOn',scene,m),('GroundOff',scene,m)) for m in METHODS)
        specs.extend((f'{arm}/{l}-{r}',(arm,scene,l),(arm,scene,r)) for arm in ARMS for l,r in [('S','B'),('B','G')])
    for group,lk,rk in specs:
        left,right=by[lk],by[rk];key=left['run_id']+'__vs__'+right['run_id']
        row=dict(pair_id=key,comparison=group,scene_id=lk[1],left_run=left['run_id'],right_run=right['run_id'],
            left_status=left['status'],right_status=right['status'],quality_status='unavailable',mechanism_status='unavailable',
            both_original_end_to_end_qualified=left['original_end_to_end_qualified'] is True and right['original_end_to_end_qualified'] is True)
        pairs.append(row)
        if left['quality_measurement_available'] and right['quality_measurement_available']:
            if left['metric_version']!=right['metric_version'] or left['reference_fingerprint']!=right['reference_fingerprint']:
                row['quality_status']='incompatible_metric_or_reference'
            elif left['motion_completion_verified'] is True and right['motion_completion_verified'] is True:
                row['quality_status']='motion_complete_common_derived_metric'
                row['interpretation']='original online failures remain failures; paired score concerns motion-complete saved predictions under one derived metric'
                for metric in (*METRICS,*COSTS):
                    row['left_'+metric]=left[metric];row['right_'+metric]=right[metric]
                    row['delta_'+metric]=None if left[metric] is None or right[metric] is None else left[metric]-right[metric]
            else:row['quality_status']='quality_available_motion_gate_not_met'
        if left['run_id'] in traces and right['run_id'] in traces:
            small,witness=compact_comparison(traces[left['run_id']],traces[right['run_id']])
            row.update(small,mechanism_status='common_actual_prefix_only',left_trace_review_scope=left['trace_review_scope'],right_trace_review_scope=right['trace_review_scope'])
            events[key]=witness
    return pairs,events


def layout_summary(pairs):
    result={}
    for group in sorted({r['comparison'] for r in pairs}):
        all_rows=[r for r in pairs if r['comparison']==group]
        good=[r for r in all_rows if r['quality_status']=='motion_complete_common_derived_metric']
        require(len({r['scene_id'] for r in all_rows})==len(all_rows)==3,'three independent layout units, no pseudo-replicates')
        result[group]=dict(expected_layout_units=3,available_layout_units=len(good),all_three_available=len(good)==3,
            unavailable_layouts=[r['scene_id'] for r in all_rows if r not in good],
            original_pipeline_failure_present=any(not r['both_original_end_to_end_qualified'] for r in good),
            all_layouts_mean_delta={m:sum(r['delta_'+m] for r in good)/3 if len(good)==3 else None for m in METRICS},
            paired_layout_deltas=[dict(scene_id=r['scene_id'],**{m:r['delta_'+m] for m in METRICS}) for r in good],
            inference='Three development layouts; descriptive paired effects only. Devices, frames and methods are not independent layout replicates.')
    return result


def analyze(protocol_path,plan_path,output,ground_reviews,baseline_reviews):
    require(not output.exists(),'exclusive new snapshot output')
    require(output.resolve().is_relative_to(STAGE),'analysis output under external article stage')
    inputs=old.Inputs();protocol=inputs.json(protocol_path);protocol_sha=old.digest(inputs.read(protocol_path))
    require(protocol['schema']=='article.experiment_protocol.ground_v2' and protocol['phase']=='ablation' and len(protocol['slots'])==12,
        'only frozen twelve-cell ground ablation supported')
    pin=protocol['paired_baseline_protocol'];baseline_path=old.safe(ROOT,pin['path'])
    baseline=inputs.json(baseline_path,pin=pin['sha256'])
    plan=inputs.json(plan_path);plan_sha=old.digest(inputs.read(plan_path))
    require(plan['measurement_version']==VERSION and plan['protocol_sha256']==pin['sha256'] and plan['slots']==baseline['slots'],
        'common derived plan belongs to paired baseline cohort')
    require(plan['evaluation']==protocol['evaluation']==baseline['evaluation'] and plan['references']==protocol['references']==baseline['references'],
        'same settings/reference in both arms')
    for source in (protocol,baseline,plan):
        for name,expected_sha in source['source_sha256'].items():inputs.read(old.safe(ROOT,name),pin=expected_sha)
        if 'source_archive' in source:inputs.read(old.safe(ROOT,source['source_archive']),pin=source['source_archive_sha256'])
    on_entries,on_phase=ledger_entries(inputs,protocol,protocol_sha)
    off_entries,off_phase=ledger_entries(inputs,baseline,pin['sha256'])
    common_root=old.safe(ROOT,plan['output_root']);rows=[];traces={};findings=[];mechanisms={}
    for on_id,slot in sorted(protocol['slots'].items()):
        off_id=slot['paired_baseline_run_id'];off_slot=baseline['slots'][off_id]
        require(all(slot[k]==off_slot[k] for k in ('method','scene_id','budget','noise_seed')),'per-slot paired covariates')
        for arm,run_id,entry in [('GroundOff',off_id,off_entries.get(off_id)),('GroundOn',on_id,on_entries.get(on_id))]:
            row=blank_row(arm,run_id,slot,entry);rows.append(row)
            if not entry or entry['status']=='reserved':continue
            try:
                ready=(load_on(inputs,on_phase,ground_reviews,row,entry,protocol_sha) if arm=='GroundOn' else
                    load_off(inputs,off_phase,baseline_reviews,common_root,plan,plan_sha,row,entry,pin['sha256']))
                if ready:
                    try:
                        trace=read_trace(inputs,*ready);traces[run_id]=trace
                        row['trace_status']='sealed_trace_available';mechanisms[run_id]=trace['mechanism']
                    except (OSError,ValueError,KeyError,TypeError) as exc:
                        row['trace_status']='trace_unavailable';row['trace_error']=str(exc)
            except (OSError,ValueError,KeyError,TypeError) as exc:
                row.update(status='analysis_binding_error',quality_measurement_available=False,review_passed=False,error=str(exc))
                row.update({m:None for m in (*METRICS,*COSTS)})
            if row['status'] in ('analysis_binding_error','review_findings','failed_attempt_reviewed','common_derived_quality_unavailable') or row.get('trace_error'):
                findings.append(dict(run_id=run_id,arm=arm,status=row['status'],error=row.get('error',row.get('trace_error')),
                    failure=row.get('failure_detail')))
    require(len(rows)==24 and len({(r['arm'],r['scene_id'],r['method']) for r in rows})==24,'all twelve paired cells retained')
    pairs,events=pair_rows(rows,traces);summary=layout_summary(pairs)
    inputs.unchanged();output.mkdir(parents=True,exist_ok=False)
    def save(name,value):
        with (output/name).open('xb') as stream:stream.write(old.canonical(value))
    old.write_csv(output/'slots.csv',rows,['arm','run_id','status'])
    old.write_csv(output/'paired_effects.csv',pairs,['comparison','scene_id','quality_status'])
    save('first_mechanism_witnesses.json',events);save('per_run_mechanism_summary.json',mechanisms)
    save('findings.json',findings)
    by_run={r['run_id']:r for r in rows}
    trajectories=[dict(run_id=run_id,arm=by_run[run_id]['arm'],scene_id=by_run[run_id]['scene_id'],method=by_run[run_id]['method'],
        original_online_status=by_run[run_id]['online_status'],original_end_to_end_qualified=by_run[run_id]['original_end_to_end_qualified'],
        motion_completion_verified=by_run[run_id]['motion_completion_verified'],trace_review_scope=by_run[run_id]['trace_review_scope'],**point)
        for run_id,trace in traces.items() for point in trace['trajectory']]
    old.write_csv(output/'trajectories.csv',trajectories,['run_id','paid_step','original_online_status'])
    result=dict(schema='article.ground_comparison_snapshot.v1',created_utc=datetime.now(timezone.utc).isoformat(),
        layout_units=3,paired_cells=12,arm_records=24,metric_version=VERSION,
        status_counts={arm:dict(Counter(r['status'] for r in rows if r['arm']==arm)) for arm in ARMS},
        actual_reserved_attempts={arm:sum(r['online_status'] is not None for r in rows if r['arm']==arm) for arm in ARMS},
        common_metric_motion_complete_pairs=sum(r['quality_status']=='motion_complete_common_derived_metric' for r in pairs),
        layout_effects=summary,findings=len(findings),new_worlds=0,new_policy_runs=0,new_tsdf_integrations=0,new_quality_evaluations=0,
        scope='Post-diagnosis component ablation on three development layouts, not independent held-out confirmation.',
        limits=['All 12 GroundOn/Off cells retained; missing metrics blank, never zero-filled.',
            'Original end-to-end qualification, saved-motion completeness and derived quality are separate columns.',
            'CELL_G original pipeline failure cannot be relabeled as an online success by posthoc evaluation.',
            'Same-prefix mechanism witnesses use exact physical arrays excluding frame_id only and observed anchors only.',
            'After the first executed-action/physical-observation divergence, no controlled per-frame causal attribution is made.',
            'Only compact first witnesses and summary counts are exported; full evidence remains in sealed original records.',
            'Recorded planning wall time is not an isolated runtime benchmark; no quality remeasurement occurs.'])
    save('summary.json',result)
    for i,(name,payload) in enumerate(inputs.captured.items()):
        with (output/f'captured_ledger_{i:02d}.json').open('xb') as stream:stream.write(payload)
    files={p.name:dict(bytes=p.stat().st_size,sha256=old.digest(p.read_bytes())) for p in output.iterdir() if p.is_file()}
    save('manifest.json',dict(schema='article.ground_comparison_manifest.v1',files=files,input_sha256=inputs.pins,
        source_sha256={str(Path(__file__).relative_to(ROOT)):old.digest(Path(__file__).read_bytes()),
            str(Path(old.__file__).relative_to(ROOT)):old.digest(Path(old.__file__).read_bytes())}))
    require(sum(p.stat().st_size for p in output.iterdir())<=10*1024**2,'compact snapshot exceeded 10MiB cap')
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--protocol',type=Path,default=ROOT/'configs/virtual3d/article_ground_ablation_v2_20260928.json')
    parser.add_argument('--common-plan',type=Path,default=STAGE/'common_evaluation_plan_v1.json')
    parser.add_argument('--ground-reviews',type=Path,default=STAGE/'episode_reviews_ground_v2')
    parser.add_argument('--baseline-reviews',type=Path,default=STAGE/'episode_reviews_v1')
    parser.add_argument('--output',type=Path)
    parser.add_argument('--snapshot',action='store_true',help='Explicit caller release; never runs experiments or metric evaluation')
    args=parser.parse_args()
    if not args.snapshot or args.output is None:parser.error('require explicit --snapshot and new --output after caller releases real analysis')
    result=analyze(args.protocol,args.common_plan,args.output,args.ground_reviews,args.baseline_reviews)
    print(json.dumps(result,ensure_ascii=False,allow_nan=False))


if __name__=='__main__':main()
