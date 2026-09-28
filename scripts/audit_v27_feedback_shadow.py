#!/usr/bin/env python3
"""V27-A matched-fixed feedback tables on one saved 401-packet history.

Exactly one offline mapper/TSDF pass; three feedback-only objects process the
13 original eligible arrival events. Original choices/actions remain fixed.
No world, new sensing, plan/local_action, mesh or Q evaluation is permitted.
This does not execute the V27 runtime or establish a closed-loop policy gain.
"""
import os
for name in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):
    os.environ[name]='1'
os.environ['PYTHONDONTWRITEBYTECODE']='1'
import argparse
from copy import deepcopy
import json
from pathlib import Path
import shutil
import signal
import sys
from time import perf_counter
import traceback
from types import SimpleNamespace
sys.dont_write_bytecode=True
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from scripts import audit_v26_feedback_support as base

OUTPUT=ROOT/'audit_results/observed_feedback_v27_shadow_20260916'
PRIOR=ROOT/'audit_results/observed_v26_feedback_support_20260916'
DESIGN=ROOT/'docs/research/V27_STABLE_FEEDBACK_DESIGN_20260916.md'
DESIGN_SHA='ca0e8ca71cac936e72ea8f32334027f4c1a522418f3d998f9d31eb868c8aa6d3'
PRIOR_RESULT_SHA='e57558ad2caddd124e2908148e5713e829c2447cb042a7861ddd793ab5c21724'
FIXED_NEW_SOURCES={
    'nso/observed_feedback_v27.py':'8e028f0232befd32063d772bb562d74402713ff0fe0098201c08cdb14a6196bb',
    'nso/observed_planner_v27.py':'29119e9759f4f3be70e39954406765cd38df7f8c677101a3eb858bbfe8e71615',
    'nso/observed_runtime_v27.py':'d4e2fac41c8ae07b598d98ca8256a36630f4dc7bf2a38d409289efbffc014579',
    'tests/virtual3d/test_observed_feedback_v27.py':'0b3b18cd389c2870303ecec0bcb631260188440386ea4d673de01e2d333a4dd2'}
CAP=512*1024
RESERVE=64*1024*1024
COUNTERS=dict(saved_packets_loaded=0,offline_mapper_updates=0,
    offline_tsdf_integrations_from_existing_packets=0,planning_geometry_snapshots=0,
    full_feedback_support_captures=0,feedback_transition_calls=0,
    original_sampled_semantic_gate_calls=0,new_world_instances=0,new_physical_tasks=0,
    new_physical_actions=0,new_sensor_packets=0,new_planning_calls=0,
    new_local_action_calls=0,new_mesh_extractions=0,new_quality_evaluations=0,
    autonomous_runtime_executions=0,forbidden_calls=0)


def require(value,message):base.require(value,message)


def denied(*args,**kwargs):
    COUNTERS['forbidden_calls']+=1
    raise RuntimeError('New physical/policy/mesh/evaluation execution forbidden')


def checked_inputs():
    require(base.sha(DESIGN)==DESIGN_SHA,'Frozen V27 shadow design changed')
    public=base.inputs()
    require(base.sha(PRIOR/'result.json')==PRIOR_RESULT_SHA,'Prior support diagnostic result changed')
    inventory=base.read(PRIOR/'artifact_hashes.json')
    actual={str(p.relative_to(PRIOR)) for p in PRIOR.rglob('*') if p.is_file() and p.name!='artifact_hashes.json'}
    require(set(inventory)==actual,'Prior diagnostic artifact set changed')
    for name,h in inventory.items():
        path=PRIOR/name;require(base.sha(path)==h,'Prior diagnostic artifact SHA')
        public['input_sha256'][str(path.relative_to(ROOT))]=h
    public['input_sha256'][str((PRIOR/'artifact_hashes.json').relative_to(ROOT))]=base.sha(PRIOR/'artifact_hashes.json')
    prior=base.read(PRIOR/'result.json')
    require(prior['status']=='complete_readonly_recorded_gate_support_diagnosis','Prior bounded diagnosis incomplete')
    require(prior['summary']['full_original_support_gate_count']==13,'Expected fixed original 13 support gates')
    for name,h in base.read(PRIOR/'input_sha256.json').items():
        require(public['input_sha256'].get(name)==h,'Existing history differs from prior full-support audit')
    plans_path=base.CASE/'plans.json.gz';case_inventory=base.read(base.CASE/'artifact_hashes.json')
    require(base.sha(plans_path)==case_inventory['plans.json.gz'],'Original fixed selected plans SHA')
    public['input_sha256'][str(plans_path.relative_to(ROOT))]=case_inventory['plans.json.gz']
    plans=base.zipped_read(plans_path)
    require([p['audit']['action_id'] for p in plans]==sorted({p['audit']['action_id'] for p in plans}),'Original plan chronology/uniqueness')
    selected={}
    for action in base.EVENT_ACTIONS:
        available=[p for p in plans if p['audit']['action_id']<action]
        require(available,'No original selection before arrival')
        plan=available[-1];choice=plan['selected']
        require(choice is not None and choice in plan['candidates'],'Original selected target missing from saved pool')
        require(choice['pose']==public['trace'][action]['pose'],'Fixed original choice is not the arrived target')
        selected[action]=dict(plan_action=plan['audit']['action_id'],candidate=deepcopy(choice))
    public.update(prior_rows={r['action_id']:r for r in prior['rows']},selected=selected)
    require(tuple(public['prior_rows'])==base.EVENT_ACTIONS,'Original diagnostic row identity')
    base.verify(public['input_sha256']);base.verify(FIXED_NEW_SOURCES)
    return public


def source_snapshot():
    sources=base.source_files()
    for path in (Path(__file__).resolve(),DESIGN,*(ROOT/name for name in FIXED_NEW_SOURCES)):
        sources[str(path.relative_to(ROOT))]=base.sha(path)
    require(all(sources[k]==h for k,h in FIXED_NEW_SOURCES.items()),'V27 implementation/test changed before shadow freeze')
    return sources


def table_rows(planner):
    return [dict(cue_id=key[0],sector=key[1],count=count,total_measured_proxy=total,
        correction_factor=(1.+total)/(1.+count)) for key,(count,total) in sorted(planner.feedback.items())]


def near(a,b,label):require(abs(a-b)<=1e-12,label+' arithmetic mismatch')


def execute(out):
    require(not out.exists(),'Fresh output required; never overwrite or retry failed shadow')
    require(shutil.disk_usage(ROOT).free>=RESERVE+CAP,'64 MiB reserve plus 512 KiB output required')
    public=checked_inputs()
    from nso.decision_replay_v13 import load_packet
    from nso.observed_runtime_mapper_v10 import ObservedRuntimeMapperV10
    from nso.observed_state_v26 import geometry_state_v26,SemanticCueV26
    from nso.observed_planner_v26 import ObservedPlannerV26
    from nso.observed_planner_v27 import ObservedPlannerV27
    from nso.observed_feedback_v27 import capture_feedback_support_v27
    require(ObservedPlannerV27.plan is ObservedPlannerV26.plan
        and ObservedPlannerV27.local_action is ObservedPlannerV26.local_action
        and ObservedPlannerV27._semantic_gain is ObservedPlannerV26._semantic_gain,
        'V27 changed original planning or sampled visibility/gain functions')
    guarded=base.tripwires();sources=source_snapshot()
    for name,h in sources.items():
        if name in public['original_sources']:require(h==public['original_sources'][name],'Original imported source changed')
    out.mkdir(parents=True);started=perf_counter();rows=[]
    try:
        base.write(out,'manifest.json',dict(status='running',source_sha256=sources,
            original_source_sha256=public['original_sources'],source_case=str(base.CASE),
            declared_events=list(base.EVENT_ACTIONS),output_cap_bytes=CAP,free_reserve_bytes=RESERVE,
            scope='matched-fixed shadow feedback only; no new autonomous runtime or actions',guarded_world_classes=guarded))
        base.write(out,'input_sha256.json',public['input_sha256'])
        mapper=ObservedRuntimeMapperV10(public['shape'],SimpleNamespace(**public['config']))
        mapper.mesh=denied
        branches={'old_V26_S':ObservedPlannerV26('S'),'new_V27_S':ObservedPlannerV27('S'),
                  'new_V27_S_no_feedback':ObservedPlannerV27('S_no_feedback')}
        current_gate_state=[None]
        for planner in branches.values():
            planner.plan=denied;planner.local_action=denied
            original=planner._semantic_gain
            def checked_gain(state,pose,cues,masks,original=original):
                require(state is current_gate_state[0] and len(state.patches)<=256,'Feedback gate received nonoriginal/full state')
                COUNTERS['original_sampled_semantic_gate_calls']+=1
                return original(state,pose,cues,masks)
            planner._semantic_gain=checked_gain
        boundary={i for a in base.EVENT_ACTIONS for i in (a-1,a)}
        snapshots={};independent_table={}
        for action in range(401):
            packet=load_packet(base.CASE/'packets'/f'{action:04d}.npz');COUNTERS['saved_packets_loaded']+=1
            trace=public['trace'][action]
            require(packet.action_id==action and packet.sha256()==trace['packet_sha256'],'Original saved packet identity')
            mapper.update(packet.frame,packet.scan);COUNTERS['offline_mapper_updates']+=1
            COUNTERS['offline_tsdf_integrations_from_existing_packets']+=1
            state=geometry_state_v26(mapper,packet,public['anchor'],400-action,max_patches=256)
            COUNTERS['planning_geometry_snapshots']+=1
            require(state.geometry_sha256==trace['geometry_sha256'],'Original 256-patch planning state not reproduced')
            if action in boundary:
                support=capture_feedback_support_v27(mapper,packet,public['anchor'],400-action,state)
                COUNTERS['full_feedback_support_captures']+=1
                snapshots[action]=(state,support)
            if action in public['events']:
                before,old_support=snapshots[action-1];after,new_support=snapshots[action]
                current_gate_state[0]=after
                selected=public['selected'][action];choice=selected['candidate']
                raw_cues=public['audits'][action-1]['cues']
                for cue in raw_cues:
                    provenance=json.loads(cue['source'])
                    require(provenance['first_action']<=cue['action_id']<=action-1,'Future cue passed to fixed feedback transition')
                cues=tuple(SemanticCueV26(**cue) for cue in raw_cues)
                outputs={}
                for name,planner in branches.items():
                    extra={} if name=='old_V26_S' else dict(support_before=old_support,support_after=new_support)
                    outputs[name]=planner.observe_transition(before,after,deepcopy(choice),cues,**extra)
                    COUNTERS['feedback_transition_calls']+=1
                    require(planner.feedback_action==action,'Feedback sequence was not consumed')
                old=outputs['old_V26_S'];frozen=public['events'][action]
                require(old=={k:v for k,v in frozen.items() if k not in ('module','operation','call_id')},'Original sampled feedback log not strictly reproduced')
                require(len(old['directional_updates'])==1 and old['directional_updates'][0]['status']=='unavailable_insufficient_common_measured_support','Original 13 unavailable events changed')
                prior=public['prior_rows'][action]
                require(base.patch_hash(old_support.patches)==prior['full_whitelist']['before_whitelist_sha256']
                    and base.patch_hash(new_support.patches)==prior['full_whitelist']['after_whitelist_sha256'],'Full support whitelist differs from prior exact diagnosis')
                main=outputs['new_V27_S'];disabled=outputs['new_V27_S_no_feedback']
                for value in (main,disabled):
                    require(value['selected_endpoint_reached'] and value['transition_scope']=='endpoint_last_paid_step','Feedback scope changed')
                    require(len(value['directional_updates'])==1,'New/removed directional gate compared with original event')
                    for key in ('known_cell_delta','sampled_key_entries','improved_common_patches'):
                        require(value[key]==old[key],'Original sampled diagnostic changed: '+key)
                    row=value['directional_updates'][0]
                    require(row['cue_id']==prior['cue_id'] and row['sector']==prior['sector'],'Shadow changed cue or directional gate')
                    for actual,expected in (('comparable_patches','local_common_count'),
                        ('improved_common_patches','local_union_improved_count'),
                        ('direction_improved_patches','local_direction_improved_count'),
                        ('range_improved_patches','local_range_improved_count'),
                        ('both_improved_patches','local_both_improved_count'),('new_keys','after_only_keys')):
                        require(row[actual]==prior['full_whitelist'][expected],'V27 full support differs from prior diagnosis: '+actual)
                update=main['directional_updates'][0];off=disabled['directional_updates'][0]
                require(update['status']=='updated' and off['status']=='disabled_no_directional_feedback','New/disabled feedback application contract')
                key=(update['cue_id'],update['sector']);count,total=independent_table.get(key,(0,0.))
                observed=min(1.,prior['full_whitelist']['local_union_improved_count']/16.)
                require(update['feedback_count_before']==count and update['feedback_count_after']==count+1,'Independent update count')
                near(update['observed_yield_proxy'],observed,'Measured /16 rule')
                near(update['feedback_factor_before'],(1.+total)/(1.+count),'Factor before')
                near(update['feedback_factor_after'],(1.+total+observed)/(2.+count),'Factor after')
                independent_table[key]=(count+1,total+observed)
                require(branches['new_V27_S'].feedback==independent_table,'Full shadow table arithmetic mismatch')
                require(not branches['old_V26_S'].feedback and not branches['new_V27_S_no_feedback'].feedback,'Old/disabled table unexpectedly written')
                rows.append(dict(action_id=action,before_action_id=action-1,original_selection_action=selected['plan_action'],
                    selected_candidate_sha256=base.digest(choice),previous_observed_cues_sha256=base.digest(raw_cues),
                    original_geometry_sha256_before=before.geometry_sha256,original_geometry_sha256_after=after.geometry_sha256,
                    support_sha256_before=old_support.support_sha256,support_sha256_after=new_support.support_sha256,
                    old_sampled_update=old['directional_updates'][0],new_full_update=update,disabled_full_update=off,
                    three_tables_after={name:table_rows(planner) for name,planner in branches.items()},
                    original_256_gates_and_full_support_counts_verified=True))
                del snapshots[action-1]
                if action not in {a-1 for a in base.EVENT_ACTIONS}:del snapshots[action]
            if action%50==0 or action==400:
                print(json.dumps(dict(action=action,rows_completed=len(rows),offline_mapper_updates=COUNTERS['offline_mapper_updates'])),flush=True)
        require(len(rows)==13 and mapper.frames==401 and COUNTERS['feedback_transition_calls']==39
            and COUNTERS['original_sampled_semantic_gate_calls']==39 and COUNTERS['full_feedback_support_captures']==26,'Single bounded shadow scope/counts')
        require(not COUNTERS['forbidden_calls'] and not base.COUNTERS['forbidden_calls'],'A forbidden method was attempted')
        base.verify(public['input_sha256']);base.verify(public['original_sources']);base.verify(sources)
        result=dict(status='complete_matched_fixed_feedback_shadow',rows=rows,counts=COUNTERS,
            final_tables={name:table_rows(planner) for name,planner in branches.items()},
            old_unavailable=13,new_updated=13,disabled_table_unchanged=True,
            saturated_original_div16_events=sum(r['new_full_update']['observed_yield_proxy']==1. for r in rows),
            old_geometry_sha256_reproduced_observations=401,original_input_source_hashes_unchanged=True,
            elapsed_s=perf_counter()-started,public_config=public['config'],shape=list(public['shape']),
            configuration_access=public['configuration_access'],
            limitations=['One existing S trajectory and 13 original gates; no new world or autonomous task.',
                'Original actions and selected options stay fixed; shadow tables never affect route selection.',
                'Only the original 13 update-applicable transitions call feedback; no new gate-selected samples.',
                'Full support captured only at 26 relevant boundaries; this does not execute the complete V27 runtime.',
                'The original last-step >=4 and /16 proxy is unchanged and uncalibrated; no Q is computed.',
                'Successful support/table wiring is not evidence of better final coverage, shape quality or semantic architecture.'])
        base.write(out,'result.json',result)
        base.write(out,'manifest.json',dict(status='complete',source_sha256=sources,
            original_source_sha256=public['original_sources'],source_case=str(base.CASE),counts=COUNTERS,
            output_cap_bytes=CAP,free_reserve_bytes=RESERVE,input_hashes_rechecked=True,elapsed_s=perf_counter()-started,
            guarded_world_classes=guarded))
        print(json.dumps(dict(status=result['status'],old_unavailable=13,new_updated=13,disabled_table_unchanged=True,
            final_tables=result['final_tables'],output=str(out))),flush=True)
    except BaseException as error:
        failure=dict(status='failed',error=repr(error),traceback=traceback.format_exc(),counts=COUNTERS,
            completed_rows=rows,original_evidence_modified=False)
        print(json.dumps(failure),file=sys.stderr,flush=True)
        base.write(out,'failure.json',failure)
        base.write(out,'manifest.json',dict(status='failed',source_sha256=sources,counts=COUNTERS))
        raise
    finally:
        inventory={str(p.relative_to(out)):base.sha(p) for p in sorted(out.rglob('*')) if p.is_file() and p.name!='artifact_hashes.json'}
        base.write(out,'artifact_hashes.json',inventory)


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--run',action='store_true')
    parser.add_argument('--output',type=Path,default=OUTPUT);args=parser.parse_args()
    if not args.run:parser.error('Wait for explicit root approval, then --run once only')
    if hasattr(os,'sched_setaffinity'):
        allowed=os.sched_getaffinity(0);os.sched_setaffinity(0,{min(allowed)})
    def stop(signum,frame):raise RuntimeError('Shadow stopped by signal '+str(signum))
    for signum in (signal.SIGTERM,signal.SIGINT,signal.SIGALRM):signal.signal(signum,stop)
    signal.alarm(600)
    try:execute(args.output.resolve())
    finally:signal.alarm(0)


if __name__=='__main__':main()
