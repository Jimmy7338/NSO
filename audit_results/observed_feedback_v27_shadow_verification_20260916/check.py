#!/usr/bin/env python3
"""Independent stored-JSON/hash/arithmetic verification; no project imports."""
import gzip
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
from time import perf_counter

ROOT=Path(__file__).resolve().parents[2]
OUT=Path(__file__).resolve().parent
REC=ROOT/'audit_results/observed_feedback_v27_shadow_recovered_20260916'
OLD=ROOT/'audit_results/observed_feedback_v27_shadow_20260916'
FAIL=ROOT/'audit_results/observed_feedback_v27_shadow_failure_20260916'
STACK=ROOT/'audit_results/observed_feedback_v27_shadow_failure_stack_20260916'
PRIOR=ROOT/'audit_results/observed_v26_feedback_support_20260916'
CASE=ROOT/'audit_results/observed_autonomous_v26_20260916/case_01'
EVENTS=[44,136,261,272,284,296,306,316,330,344,355,363,365]
PINS={
    OLD/'artifact_hashes.json':'91de924953e0df1a3e6d15e0bbd38a267fd53899ba9a85886a33e7fc87f7d1a0',
    FAIL/'artifact_hashes.json':'58d9e08a575d288b2eeb9ea903b58485199743c6dc560e12284cf79a9b2c7e27',
    STACK/'artifact_hashes.json':'f6d70bfefa276248bde1eb969b2d01fbf3892c33ed9d0d1d66c274bff6680649',
    PRIOR/'result.json':'e57558ad2caddd124e2908148e5713e829c2447cb042a7861ddd793ab5c21724',
    ROOT/'scripts/audit_v27_feedback_shadow.py':'0430504591e7805dfbcabbd29f491c38d9fafda864f8275bcebeaf054c2463db',
    ROOT/'docs/research/V27_STABLE_FEEDBACK_DESIGN_20260916.md':'ca0e8ca71cac936e72ea8f32334027f4c1a522418f3d998f9d31eb868c8aa6d3',
    ROOT/'docs/research/V27_SHADOW_SERIALIZATION_RECOVERY_20260916.md':'7f89f2944457f4c24d822d92d150a640b606cffed930c51a92b23d03418cd9cd'}


def require(condition,message):
    if not condition:raise ValueError(message)


def sha(path):
    with path.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()


def read(path):return json.loads(path.read_text())


def zipped(path):
    with gzip.open(path,'rt') as f:return json.load(f)


def digest(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()


def mapping(values):
    for name,h in values.items():require(sha(ROOT/name)==h,'Input/source hash differs: '+name)


def inventory(folder):
    values=read(folder/'artifact_hashes.json')
    actual={str(p.relative_to(folder)) for p in folder.rglob('*') if p.is_file() and p.name!='artifact_hashes.json'}
    require(set(values)==actual,'Artifact membership differs: '+str(folder))
    for name,h in values.items():require(sha(folder/name)==h,'Artifact bytes differ: '+name)
    return values


def near(a,b,label):require(math.isfinite(a) and abs(a-b)<1e-12,label+' arithmetic differs')


def table_rows(table):
    return [dict(cue_id=k[0],sector=k[1],count=n,total_measured_proxy=t,correction_factor=(1+t)/(1+n))
            for k,(n,t) in sorted(table.items())]


def run():
    started=perf_counter()
    require(not (OUT/'result.json').exists(),'Do not overwrite an existing verification')
    require(shutil.disk_usage(ROOT).free>=64*1024**2+100*1024,'64 MiB reserve required')
    for path,h in PINS.items():require(sha(path)==h,'Previously pinned evidence changed: '+str(path))
    checked={str(p.relative_to(ROOT)):inventory(p) for p in (REC,OLD,FAIL,STACK,PRIOR)}
    manifest=read(REC/'manifest.json');inputs=read(REC/'input_sha256.json');result=read(REC/'result.json')
    require(manifest['status']=='complete' and result['status']=='complete_matched_fixed_feedback_shadow','Recovered shadow incomplete')
    require(manifest['output_cap_bytes']==512*1024 and manifest['free_reserve_bytes']==64*1024**2,'Capacity contract changed')
    mapping(manifest['source_sha256']);mapping(manifest['original_source_sha256']);mapping(inputs)
    prior=read(PRIOR/'result.json');failure=read(FAIL/'failure_record.json')
    require(read(OLD/'manifest.json')['status']=='running' and not (OLD/'result.json').exists(),'Original failed state was overwritten')
    require(failure['process_id']==377855 and failure['session_id']==73595 and failure['exit_code']==1,'Failed attempt identity changed')
    require(failure['saved_packet_offline_mapper_and_tsdf_updates_executed']==401,'Lost failed offline execution cost')
    require(prior['counts']['offline_mapper_updates']==401,'Prior support audit cost')
    counts=result['counts'];require(counts==manifest['counts'],'Manifest/result counts differ')
    expected_counts=dict(saved_packets_loaded=401,offline_mapper_updates=401,
        offline_tsdf_integrations_from_existing_packets=401,planning_geometry_snapshots=401,
        full_feedback_support_captures=26,feedback_transition_calls=39,original_sampled_semantic_gate_calls=39)
    for name,n in counts.items():require(n==expected_counts.get(name,0),'Unexpected executed operation/count: '+name)
    require(result['old_geometry_sha256_reproduced_observations']==401,'Planning identity count')
    require(result['old_unavailable']==result['new_updated']==13 and result['disabled_table_unchanged'],'Feedback summary changed')
    rows=result['rows'];require([r['action_id'] for r in rows]==EVENTS,'Original 13 ordered event list differs')
    prior_rows={r['action_id']:r for r in prior['rows']}
    original=read(CASE/'result.json');trace=original['trace'];plans=zipped(CASE/'plans.json.gz')
    calls=zipped(CASE/'module_calls.json.gz')
    gates={r['action_id']:r for r in calls if r['operation']=='measured_transition_feedback'
           and r['selected_endpoint_reached'] and r['directional_updates']}
    require(list(gates)==EVENTS,'Original actual gate list changed')
    table={};compact=[];saturated=0
    for row in rows:
        action=row['action_id'];a=row['new_full_update'];off=row['disabled_full_update'];old=row['old_sampled_update']
        prior_row=prior_rows[action];p=prior_row['full_whitelist']
        require(row['before_action_id']==action-1,'Nonadjacent feedback transition')
        plan=[x for x in plans if x['audit']['action_id']<action][-1]
        require(row['original_selection_action']==plan['audit']['action_id'] and row['selected_candidate_sha256']==digest(plan['selected']),'Original installed selected target differs')
        require(plan['selected']['pose']==trace[action]['pose'],'Recorded endpoint was not reached')
        before=zipped(CASE/trace[action-1]['audit_file'])
        require(row['previous_observed_cues_sha256']==digest(before['cues']),'Previous cue history differs')
        cue=[c for c in before['cues'] if c['cue_id']==a['cue_id']]
        require(len(cue)==1 and cue[0]['action_id']<=action-1,'Missing/future cue')
        require(digest(cue[0])==prior_row['prior_observed_cue_sha256'],'Cue center/history differs from prior support audit')
        require(row['original_geometry_sha256_before']==trace[action-1]['geometry_sha256']
            and row['original_geometry_sha256_after']==trace[action]['geometry_sha256'],'Saved geometry identity differs')
        require(old==gates[action]['directional_updates'][0],'Old feedback row differs')
        require(old['status']=='unavailable_insufficient_common_measured_support','Original unavailable status changed')
        require(a['status']=='updated' and off['status']=='disabled_no_directional_feedback','New/disabled status changed')
        require(a['cue_id']==off['cue_id']==old['cue_id']==prior_row['cue_id'] and a['sector']==off['sector']==old['sector']==prior_row['sector'],'Directional identity changed')
        for actual,expected in (('comparable_patches','local_common_count'),('improved_common_patches','local_union_improved_count'),
            ('direction_improved_patches','local_direction_improved_count'),('range_improved_patches','local_range_improved_count'),
            ('both_improved_patches','local_both_improved_count'),('new_keys','after_only_keys')):
            require(a[actual]==off[actual]==p[expected],'Full support count differs: '+actual)
        key=a['cue_id'],a['sector'];n,total=table.get(key,(0,0.));observed=min(1.,p['local_union_improved_count']/16)
        require(a['feedback_count_before']==n and a['feedback_count_after']==n+1,'Count recurrence differs')
        near(a['observed_yield_proxy'],observed,'Original /16')
        near(a['feedback_factor_before'],(1+total)/(1+n),'Before factor')
        near(a['feedback_factor_after'],(1+total+observed)/(2+n),'After factor')
        table[key]=n+1,total+observed;saturated+=observed==1.
        require(row['three_tables_after']==dict(old_V26_S=[],new_V27_S=table_rows(table),new_V27_S_no_feedback=[]),'Intermediate table differs')
        compact.append(dict(action_id=action,cue_id=a['cue_id'],sector=a['sector'],sampled_common=old['comparable_patches'],
            full_common=a['comparable_patches'],improved_union=a['improved_common_patches'],observed_proxy=observed,
            count_after=n+1,factor_after=a['feedback_factor_after']))
    require(result['final_tables']==dict(old_V26_S=[],new_V27_S=table_rows(table),new_V27_S_no_feedback=[]),'Final three tables differ')
    require(saturated==result['saturated_original_div16_events']==8,'Original /16 saturation count')
    require(sum(v[0] for v in table.values())==13,'Total feedback entries lost')
    mapping(manifest['source_sha256']);mapping(manifest['original_source_sha256']);mapping(inputs)
    for path,h in PINS.items():require(sha(path)==h,'Evidence changed during verification')
    for folder,values in checked.items():require(inventory(ROOT/folder)==values,'Inventory changed during verification')
    return dict(status='passed_independent_readonly_shadow_verification',scope='Stored bytes, original logs and arithmetic only; no project module imports',
        recovered_source_count=len(manifest['source_sha256']),original_source_count=len(manifest['original_source_sha256']),
        recovered_input_count=len(inputs),recovered_artifact_count=len(checked[str(REC.relative_to(ROOT))]),
        recovered_result_sha256=sha(REC/'result.json'),recovered_manifest_sha256=sha(REC/'manifest.json'),
        recovered_inventory_sha256=sha(REC/'artifact_hashes.json'),
        checked_inventory_sha256={folder:sha(ROOT/folder/'artifact_hashes.json') for folder in checked},
        old_failure_unchanged=True,original_status_running_is_stale_failure_not_active_process=True,
        row_count=13,rows=compact,final_tables=result['final_tables'],saturation_count=8,
        recovery_reported_elapsed_s=result['elapsed_s'],verification_elapsed_s=perf_counter()-started,
        offline_cost=dict(first_failed_shadow=401,recovered_shadow=401,all_shadow=802,prior_support_diagnosis=401,
            total_feedback_diagnosis_and_shadow=1203,additional_fusion_in_this_verification=0),
        verification_new_worlds=0,verification_new_actions=0,verification_policy_calls=0,verification_mapper_updates=0,
        verification_mesh_extractions=0,verification_Q_evaluations=0,
        limitation='Feedback support and writeback on the fixed original history are verified; no V27 autonomous runtime, new C/Q/J or whole-architecture gain is established')


def save(name,value):
    blob=(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+'\n').encode()
    require(shutil.disk_usage(ROOT).free-len(blob)>=64*1024**2,'64 MiB reserve')
    require(sum(p.stat().st_size for p in OUT.iterdir())+len(blob)<100*1024,'100 KiB verification limit')
    with (OUT/name).open('xb') as f:f.write(blob);f.flush();os.fsync(f.fileno())


if __name__=='__main__':
    try:
        result=run();result['verification_script_sha256']=sha(Path(__file__))
        save('result.json',result)
        print(json.dumps({k:result[k] for k in ('status','recovered_source_count','original_source_count','recovered_input_count','recovered_artifact_count','row_count','final_tables','offline_cost','recovery_reported_elapsed_s','recovered_result_sha256','recovered_inventory_sha256')}))
    except BaseException as error:
        save('failure.json',dict(status='failed_independent_readonly_verification',error=repr(error)));raise
    finally:save('artifact_hashes.json',{p.name:sha(p) for p in sorted(OUT.iterdir()) if p.is_file() and p.name!='artifact_hashes.json'})
