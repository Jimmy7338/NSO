#!/usr/bin/env python3
"""One offline full-support audit of the 13 recorded case-01 V26 feedback gates.

All 401 existing packets are integrated once into one mapper; no new world,
sensor, policy, mesh extraction, Q evaluation or counterfactual trajectory.
The existing arrival/prediction/visibility gates are reused, never re-planned.
"""
import os
for name in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):
    os.environ[name]='1'
os.environ['PYTHONDONTWRITEBYTECODE']='1'
import argparse
import gzip
import hashlib
import json
from pathlib import Path
import shutil
import signal
import statistics
import sys
from time import perf_counter
import traceback
from types import SimpleNamespace
sys.dont_write_bytecode=True
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
SOURCE=ROOT/'audit_results/observed_autonomous_v26_20260916'
CASE=SOURCE/'case_01'
OUTPUT=ROOT/'audit_results/observed_v26_feedback_support_20260916'
CONFIG_FIELDS=('resolution_m','robot_radius_m','camera_height_m','width_px','height_px',
    'fov_deg','max_depth_m','depth_sigma_m','dropout','action_duration_s','voxel_m',
    'laser_height_m','laser_rays','width_m','height_m','truncation_m','pose_noise_m',
    'stereo_model','stereo_reference_fx_px','stereo_baseline_m')
EVENT_ACTIONS=(44,136,261,272,284,296,306,316,330,344,355,363,365)
CAP=512*1024
RESERVE=64*1024*1024
COUNTERS=dict(saved_packets_loaded=0,offline_mapper_updates=0,
    offline_tsdf_integrations_from_existing_packets=0,geometry_adapter_snapshots=0,
    new_world_instances=0,new_physical_tasks=0,new_physical_actions=0,
    new_sensor_packets=0,new_policy_calls=0,new_mesh_extractions=0,
    new_quality_evaluations=0,forbidden_calls=0)


def require(value,message):
    if not value:raise ValueError(message)


def sha(path):
    with Path(path).open('rb') as stream:return hashlib.file_digest(stream,'sha256').hexdigest()


def read(path):return json.loads(Path(path).read_text())


def zipped_read(path):
    with gzip.open(path,'rt') as stream:return json.load(stream)


def digest(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()


def verify(mapping):
    for name,expected in mapping.items():require(sha(ROOT/name)==expected,'Changed input/source: '+name)


def write(out,name,value):
    blob=(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+'\n').encode()
    used=sum(p.stat().st_size for p in out.rglob('*') if p.is_file())
    require(used+len(blob)+4096<=CAP,'512 KiB output hard limit')
    require(shutil.disk_usage(out).free-len(blob)-4096>=RESERVE,'64 MiB free reserve')
    tmp=out/(name+'.tmp')
    with tmp.open('xb') as stream:stream.write(blob);stream.flush();os.fsync(stream.fileno())
    os.replace(tmp,out/name)


def denied(*args,**kwargs):
    COUNTERS['forbidden_calls']+=1
    raise RuntimeError('World/physical/sensor/policy/mesh/evaluation call forbidden')


def tripwires():
    guarded=[];seen=set()
    for module in tuple(sys.modules.values()):
        if not getattr(module,'__name__','').startswith('env.'):continue
        for value in tuple(vars(module).values()):
            if not isinstance(value,type) or id(value) in seen or not value.__module__.startswith('env.') or 'World' not in value.__name__:continue
            seen.add(id(value));guarded.append(value.__module__+'.'+value.__name__)
            value.__init__=denied
            for name in ('step','sense','scan'):
                if hasattr(value,name):setattr(value,name,denied)
    return sorted(guarded)


def inputs():
    # Root manifest / case inventory are currently being extended by replay.
    # Snapshot their fixed mappings; do NOT lock or mutate these moving files.
    manifest=read(SOURCE/'manifest.json');inventory=read(CASE/'artifact_hashes.json')
    require(manifest['cases'][1]['physical_status']=='complete','Completed physical case01 required')
    original_sources=dict(manifest['source_sha256']);verify(original_sources)
    used={}
    def checked(path,compressed=False):
        relative=str(path.relative_to(CASE));h=sha(path)
        require(inventory.get(relative)==h,'Original sealed input mismatch: '+relative)
        used[str(path.relative_to(ROOT))]=h
        return zipped_read(path) if compressed else read(path)
    record=checked(CASE/'result.json')
    require(record['status']=='complete' and record['index']==1 and record['mode']=='S' and record['paid_actions']==400,'Fixed case01 contract')
    config={name:record['public_config'][name] for name in CONFIG_FIELDS}
    shape=tuple(record['shape']);trace=record['trace'];anchor=record['summary']['anchor']
    require(shape==(50,80) and len(trace)==401 and [r['action_id'] for r in trace]==list(range(401)),'Fixed saved history')
    calls=checked(CASE/'module_calls.json.gz',True)
    events={c['action_id']:c for c in calls if c['module']=='IGCR' and c['operation']=='measured_transition_feedback'
            and c['selected_endpoint_reached'] and c['directional_updates']}
    require(tuple(events)==EVENT_ACTIONS and sum(len(e['directional_updates']) for e in events.values())==13,'Recorded event scope differs; no replacement cases')
    audits={}
    for action in sorted({i for a in events for i in (a-1,a)}):
        audit=checked(CASE/trace[action]['audit_file'],True)
        require(digest(audit)==trace[action]['decision_sha256'],'Decision evidence SHA')
        require(audit['action_id']==action and audit['geometry_sha256']==trace[action]['geometry_sha256'],'Audit state identity')
        audits[action]=audit
    for action,event in events.items():
        require(event in audits[action]['module_calls'],'Gate event missing from exact saved observation boundary')
        require(not event['evaluation_Q_used'],'Feedback unexpectedly consumed Q')
    for action,row in enumerate(trace):
        path=CASE/'packets'/f'{action:04d}.npz';h=sha(path)
        require(inventory.get(str(path.relative_to(CASE)))==h==row['file_sha256'],'Original packet file SHA')
        used[str(path.relative_to(ROOT))]=h
    used[str((SOURCE/'sources.zip').relative_to(ROOT))]=manifest['source_archive_sha256']
    verify(used)
    return dict(config=config,shape=shape,trace=trace,anchor=anchor,events=events,audits=audits,
        original_sources=original_sources,input_sha256=used,
        configuration_access='Opened case01/result.json; only public_config whitelist, shape, anchor and trace used; no Q/mesh/reference/route fields forwarded',
        mutable_inventory_policy='Read fixed input mappings once; no before/after lock on replay-appended verification, root manifest or case inventory')


def compact_patch(patch):
    return dict(key=list(patch.key),point=list(patch.point),bits=patch.bits,best_range=patch.best_range)


def patch_hash(patches):return digest([compact_patch(p) for p in patches])


def support(before,after,center):
    old={p.key:p for p in before}
    common=[p for p in after if p.key in old]
    local=[p for p in common if sum((p.point[i]-center[i])**2 for i in (0,1))**.5<=1.5]
    direction=[p for p in local if bool(p.bits & ~old[p.key].bits)]
    distance=[p for p in local if p.best_range<old[p.key].best_range-.01]
    both=set(p.key for p in direction)&set(p.key for p in distance)
    global_improved=[p for p in common if bool(p.bits & ~old[p.key].bits) or p.best_range<old[p.key].best_range-.01]
    return dict(before_count=len(before),after_count=len(after),common_key_count=len(common),
        after_only_keys=len(after)-len(common),local_common_count=len(local),
        local_direction_improved_count=len(direction),local_range_improved_count=len(distance),
        local_both_improved_count=len(both),local_union_improved_count=len(direction)+len(distance)-len(both),
        global_union_improved_count=len(global_improved),
        before_whitelist_sha256=patch_hash(before),after_whitelist_sha256=patch_hash(after),
        local_common_keys_sha256=digest([list(p.key) for p in local]))


def source_files():
    paths={Path(__file__).resolve(),ROOT/'nso/observed_planner_v26.py'}
    for module in tuple(sys.modules.values()):
        name=getattr(module,'__file__',None)
        if name:
            path=Path(name).resolve()
            if path.suffix=='.py' and path.is_relative_to(ROOT) and not any(p.startswith('.venv') for p in path.relative_to(ROOT).parts):paths.add(path)
    return {str(p.relative_to(ROOT)):sha(p) for p in sorted(paths)}


def execute(out):
    require(not out.exists(),'Fresh audit directory required; failed first attempt is not overwritten')
    require(shutil.disk_usage(ROOT).free>=RESERVE+CAP,'Insufficient bounded audit capacity')
    public=inputs()
    import numpy as np
    from nso.decision_replay_v13 import load_packet
    from nso.observed_runtime_mapper_v10 import ObservedRuntimeMapperV10
    from nso.observed_state_v26 import geometry_state_v26
    guarded=tripwires();sources=source_files()
    for name,h in sources.items():
        if name in public['original_sources']:require(h==public['original_sources'][name],'Imported original source changed')
    out.mkdir(parents=True);started=perf_counter();rows=[]
    try:
        write(out,'manifest.json',dict(status='running',source_case=str(CASE),source_sha256=sources,
            original_source_sha256=public['original_sources'],guarded_world_classes=guarded,
            output_cap_bytes=CAP,free_reserve_bytes=RESERVE,counts=COUNTERS))
        write(out,'input_sha256.json',public['input_sha256'])
        mapper=ObservedRuntimeMapperV10(public['shape'],SimpleNamespace(**public['config']))
        mapper.mesh=denied
        required={i for a in public['events'] for i in (a-1,a)};snapshots={}
        for action in range(401):
            packet=load_packet(CASE/'packets'/f'{action:04d}.npz');COUNTERS['saved_packets_loaded']+=1
            row=public['trace'][action]
            require(packet.action_id==action and packet.sha256()==row['packet_sha256'],'Loaded packet identity')
            require([*packet.position,packet.heading]==row['pose'] and packet.action==row['action'],'Saved pose/action mismatch')
            mapper.update(packet.frame,packet.scan);COUNTERS['offline_mapper_updates']+=1
            COUNTERS['offline_tsdf_integrations_from_existing_packets']+=1
            if action in required:
                sampled=geometry_state_v26(mapper,packet,public['anchor'],400-action,max_patches=256)
                full=geometry_state_v26(mapper,packet,public['anchor'],400-action,max_patches=2**31-1)
                COUNTERS['geometry_adapter_snapshots']+=2
                require(sampled.geometry_sha256==row['geometry_sha256'],'Offline sampled state differs from original')
                expected=(tuple(full.patches[i] for i in np.linspace(0,len(full.patches)-1,256,dtype=int))
                          if len(full.patches)>256 else full.patches)
                require(expected==sampled.patches,'Uniform index thinning is not reproduced')
                snapshots[action]=(sampled.patches,full.patches)
            if action in public['events']:
                event=public['events'][action];previous=public['audits'][action-1]
                before_sample,before_full=snapshots[action-1];after_sample,after_full=snapshots[action]
                for update in event['directional_updates']:
                    matching=[c for c in previous['cues'] if c['cue_id']==update['cue_id']]
                    require(len(matching)==1,'Previously observed cue identity missing/ambiguous')
                    cue=matching[0];provenance=json.loads(cue['source'])
                    require(provenance['first_action']<=cue['action_id']<=action-1,'Cue came from a future observation')
                    a=support(before_sample,after_sample,cue['center']);b=support(before_full,after_full,cue['center'])
                    require(a['local_common_count']==update['comparable_patches'],'Original local sampled support mismatch')
                    require(a['after_only_keys']==event['sampled_key_entries'],'Original sampled membership turnover mismatch')
                    require(a['global_union_improved_count']==event['improved_common_patches'],'Original global sampled improvement mismatch')
                    if update['status']=='updated':require(a['local_union_improved_count']==update['improved_common_patches'],'Original accepted local improvement mismatch')
                    rows.append(dict(action_id=action,cue_id=cue['cue_id'],sector=update['sector'],
                        cue_center=list(cue['center']),cue_last_observation_action=cue['action_id'],
                        cue_first_observation_action=provenance['first_action'],prior_observed_cue_sha256=digest(cue),
                        original_status=update['status'],recorded_comparable_patches=update['comparable_patches'],
                        original_arrival_prediction_visibility_gates_reused=True,
                        sampled=a,full_whitelist=b,full_support_meets_original_minimum4=b['local_common_count']>=4,
                        full_support_has_direction_or_range_improvement=b['local_union_improved_count']>0))
                del snapshots[action-1]
                if action not in {a-1 for a in public['events']}:del snapshots[action]
            if action%50==0 or action==400:
                print(json.dumps(dict(action=action,rows_completed=len(rows),offline_mapper_updates=COUNTERS['offline_mapper_updates'])),flush=True)
        require(len(rows)==13 and COUNTERS['offline_mapper_updates']==401 and mapper.frames==401,'Bounded complete audit counts')
        verify(public['input_sha256']);verify(public['original_sources']);verify(sources)
        summary=dict(recorded_directional_updates=len(rows),recorded_unavailable_updates=sum(r['original_status']!='updated' for r in rows),
            sampled_local_common_counts=[r['sampled']['local_common_count'] for r in rows],
            full_local_common_counts=[r['full_whitelist']['local_common_count'] for r in rows],
            full_local_union_improved_counts=[r['full_whitelist']['local_union_improved_count'] for r in rows],
            full_original_support_gate_count=sum(r['full_support_meets_original_minimum4'] for r in rows),
            full_nonzero_improvement_count=sum(r['full_support_has_direction_or_range_improvement'] for r in rows),
            selected_events_sampled_entries_median=statistics.median(r['sampled']['after_only_keys'] for r in rows),
            selected_events_sampled_local_common_median=statistics.median(r['sampled']['local_common_count'] for r in rows),
            selected_events_full_local_common_median=statistics.median(r['full_whitelist']['local_common_count'] for r in rows))
        result=dict(status='complete_readonly_recorded_gate_support_diagnosis',rows=rows,summary=summary,
            counts=COUNTERS,elapsed_s=perf_counter()-started,original_input_source_hashes_unchanged=True,
            public_config=public['config'],shape=list(public['shape']),configuration_access=public['configuration_access'],
            mutable_inventory_policy=public['mutable_inventory_policy'],
            support_definition='All finite geometry-state whitelist patches inside public ROI; common voxel keys; after-point horizontal radius<=1.5m about previously observed cue; new bits OR best_range improvement>0.01m',
            limitation='Post-hoc support diagnosis on original S trajectory; no feedback state/score update, no replayed policy, no new Q or proof that adding support improves final task quality',
            full_quality_or_tsdf_arrays_saved=False)
        write(out,'result.json',result)
        write(out,'manifest.json',dict(status='complete',source_case=str(CASE),source_sha256=sources,
            original_source_sha256=public['original_sources'],input_hashes_rechecked=True,counts=COUNTERS,
            guarded_world_classes=guarded,elapsed_s=perf_counter()-started,output_cap_bytes=CAP,free_reserve_bytes=RESERVE))
        print(json.dumps(dict(status=result['status'],summary=summary,output=str(out))),flush=True)
    except BaseException as error:
        failure=dict(status='failed',error=repr(error),traceback=traceback.format_exc(),counts=COUNTERS,
            completed_rows=rows,original_evidence_modified=False)
        print(json.dumps(failure),file=sys.stderr,flush=True)
        write(out,'failure.json',failure)
        write(out,'manifest.json',dict(status='failed',source_sha256=sources,counts=COUNTERS))
        raise
    finally:
        inventory={str(p.relative_to(out)):sha(p) for p in sorted(out.rglob('*')) if p.is_file() and p.name!='artifact_hashes.json'}
        write(out,'artifact_hashes.json',inventory)


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--run',action='store_true')
    parser.add_argument('--output',type=Path,default=OUTPUT);args=parser.parse_args()
    if not args.run:parser.error('Explicit --run required for this single bounded offline replay')
    if hasattr(os,'sched_setaffinity'):
        allowed=os.sched_getaffinity(0);os.sched_setaffinity(0,{min(allowed)})
    def stop(signum,frame):raise RuntimeError('Audit stopped by signal '+str(signum))
    for signum in (signal.SIGTERM,signal.SIGINT,signal.SIGALRM):signal.signal(signum,stop)
    signal.alarm(600)
    try:execute(args.output.resolve())
    finally:signal.alarm(0)


if __name__=='__main__':main()
