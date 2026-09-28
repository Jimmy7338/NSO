#!/usr/bin/env python3
"""All declared second-view probes after exact recorded first approaches.

Evaluation has world access; the runtime receives SensorPackets only. This is
an option feasibility/observation test, not a full-budget policy comparison.
"""
import os
os.environ.setdefault('OMP_NUM_THREADS','1')
os.environ.setdefault('OPENBLAS_NUM_THREADS','1')
import argparse
from copy import deepcopy
import gzip
import json
from pathlib import Path
import shutil
import subprocess
import sys
from time import perf_counter
from types import SimpleNamespace
import zipfile
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import numpy as np
from env.canonical_rgbd_v15 import CanonicalRGBDInspectionWorldV15
from env.virtual3d_inspection_v4 import InspectionConfigV4
from nso.cpu_four_modules_v16_3 import viewpoint_components_v16_3
from nso.staged_approach_runtime_v16 import StagedApproachRuntimeV16
from nso.cpu_sensor_contract_v10 import GridTransform,digest,json_value
from nso.decision_replay_v13 import load_packet,save_packet,decision_state,array_hash
from nso.surface_feedback_v16 import SurfaceFeedbackV16,capture_support
from nso.semantic_opportunities_v14 import observed_descriptors,route_instance_features
from nso.response_features_v7 import observing_camera
from scripts.collect_semantic_gain_v13_history import packet,sha,write
from utils.reconstruction_metrics import ReconstructionEvaluator
from utils.counterfactual_surface_visibility import reference_visible


def read(p):return json.loads(p.read_text())
def seal(p):write(p/'artifact_hashes.json',{str(x.relative_to(p)):sha(x) for x in sorted(p.rglob('*'))
    if x.is_file() and x.name!='artifact_hashes.json'})
def verify_sources(m):
    for n,v in m['source_sha256'].items():
        if sha(ROOT/n)!=v:raise ValueError('changed frozen source: '+n)
def gzwrite(p,x):p.write_bytes(gzip.compress(json.dumps(json_value(x),sort_keys=True,allow_nan=False).encode(),mtime=0))


def run(root,protocol,declaration,replay=False):
    candidate=declaration['first_candidate']; second=declaration['second_candidate']
    cid=second['candidate_id']; first_arrival_action=declaration['arrival_action_id']
    folder=root/f"first_{candidate['candidate_id']}_second_{cid}"
    if not replay:folder.mkdir();(folder/'packets').mkdir()
    history=ROOT/protocol['history'];hp=read(history/'manifest.json')['protocol']
    doc=read(ROOT/hp['scene_protocol']);c=next(x for x in doc['contexts'] if x['id']==hp['context'])
    config=InspectionConfigV4(**{**doc['shared_conditions'],**{k:v for k,v in c.items() if k not in ('id','seed')}})
    world=CanonicalRGBDInspectionWorldV15(config,seed=protocol['structure_seed'],semantic_condition='aligned')
    hf=history/f"structure_{protocol['structure_seed']}";step=protocol['action_id']
    transform=GridTransform(tuple(world.shape),config.resolution_m)
    args=SimpleNamespace(nso_backend='cpu_v10',eval=True,train_global=False,
        use_open_vocab_semantic=True,use_topo_graph=True,use_rpn_uq=True,use_igcr=True,
        cpu_score_mode='G',cpu_disable_feedback=False,cpu_max_candidates=5,cpu_coverage_slots=4,
        cpu_planner_revision='v10_3_1',cpu_semantic_source_schema='inspection_v4',cpu_measured_novelty_floor=.25,
        cpu_viewpoint_start_action=first_arrival_action)
    comp=viewpoint_components_v16_3(args,world.shape);runtime=StagedApproachRuntimeV16(comp,1,world.shape)
    first=packet(world,dict(parent=hp['context']),None)
    if first.sha256()!=load_packet(hf/'packets/0000.npz').sha256():raise ValueError('initial sensor mismatch')
    runtime.start_sensor_episode(0,config=config,transform=transform,packets=[first],
        total_budget=protocol['total_task_budget'],return_anchor=(*first.position,first.heading))
    s=runtime.states[0];mapper=s['mapper'];b=comp._cpu_backend
    ep=protocol['evaluation'];evaluators={seed:ReconstructionEvaluator(world,count=ep['reference_count'],seed=seed)
        for seed in ep['reference_seeds']};e=evaluators[ep['reference_seeds'][0]]
    weight=float(world.mesh.get_surface_area())/ep['reference_count']
    visible=reference_visible(e.reference,first.frame,e.truth,first.frame.world_from_camera,config.max_depth_m)
    def evaluate():
        coverage=float(np.count_nonzero((mapper.belief!=-1)&world.reachable)/world.reachable.sum())
        return {str(seed):dict(coverage_2d=coverage,**v.evaluate(mapper.mesh(),coverage,thresholds=ep['thresholds_m']))
            for seed,v in evaluators.items()}
    actions=[];events=[];observer=None;started=perf_counter()
    expected=read(folder/'actions.json') if replay else None
    def consume(action,phase):
        nonlocal visible
        frame,collision,done=world.step(action)
        measured=packet(world,dict(parent=hp['context']),action,frame,collision,done).validate(transform,config)
        path=hf/f'packets/{world.step_count:04d}.npz' if world.step_count<=step else folder/f'packets/{world.step_count:04d}.npz'
        if step < world.step_count <= first_arrival_action:
            recorded=ROOT/protocol['stage_source']/f"candidate_{candidate['candidate_id']}/packets/{world.step_count:04d}.npz"
            if measured.sha256()!=load_packet(recorded).sha256():raise ValueError('first approach sensor history differs')
        if world.step_count<=step or replay:
            if measured.sha256()!=load_packet(path).sha256():raise ValueError('physical packet differs')
        else:save_packet(path,measured)
        try:
            runtime.observe(0,world.step_count,None,None,None,sensor_packet=measured)
        except Exception as error:
            if not replay:
                write(folder/'failure_after_physical_action.json',dict(action_id=world.step_count,phase=phase,
                    packet_sha256=measured.sha256(),error=repr(error),termination=s.get('termination')))
            raise
        visible |= reference_visible(e.reference,frame,e.truth,frame.world_from_camera,config.max_depth_m)
        row=dict(action_id=world.step_count,action=action,phase=phase,pose=[*world.position,world.heading],
            collision=bool(collision),packet_sha256=measured.sha256())
        if replay and row!=expected[len(actions)]:raise ValueError('physical action differs')
        actions.append(row)
        if observer is not None:
            events.append(dict(phase=phase,**observer.observe(mapper,action_id=world.step_count,coordinate_epoch='fixed_world_pose')))
        if not replay:write(folder/'actions.json',actions)
    for aid in range(1,step+1):
        action=load_packet(hf/f'packets/{aid:04d}.npz').action
        if runtime.authorize_probe_action(action,phase='external_prefix') is None:raise ValueError('recorded prefix denied')
        consume(action,'external_prefix')
    old=next(x for x in read(hf/'all_decisions.json') if x['action_id']==step)
    actual=decision_state(runtime)['evidence']
    for key in ('map_arrays','mesh'):
        if actual[key]!=old['state_before']['evidence'][key]:raise ValueError('prefix reconstruction differs')
    receipt=runtime.install_candidate(candidate,protocol['candidate_pool_sha256'])
    prefix_state=decision_state(runtime);before=evaluate();prefix_visible=visible.copy()
    frozen_asset=deepcopy(b.scenes[0]['assets'][candidate['asset_index']])
    observer=SurfaceFeedbackV16(mapper,action_id=step,coordinate_epoch='fixed_world_pose')
    prefix_support=capture_support(mapper)[2]
    reached=False;denial=None
    while s['active_actions'] and not s['closed']:
        action=s['active_actions'][0]
        if runtime.authorize_probe_action(action,phase='outbound') is None:
            denial=deepcopy(runtime.audit[-1]);break
        consume(action,'outbound')
    reached=bool([*s['packet'].position,s['packet'].heading]==candidate['pose'] and not s['packet'].collision)
    first_reached=reached
    if not first_reached or world.step_count!=first_arrival_action:
        raise ValueError('previously replayed first approach no longer reaches the declared second decision')
    first_arrival_state=decision_state(runtime);first_arrival_metrics=evaluate();first_visible=visible.copy()
    old_first=json.loads(gzip.decompress((ROOT/protocol['stage_source']/f"candidate_{candidate['candidate_id']}/deterministic.json.gz").read_bytes()))
    for key in ('map_arrays','mesh'):
        if first_arrival_state['evidence'][key]!=old_first['arrival_state']['evidence'][key]:
            raise ValueError('first arrival reconstruction differs')
    frozen_asset=deepcopy(b.scenes[0]['assets'][second['asset_index']])
    prefix_support=capture_support(mapper)[2]
    second_receipt=runtime.install_candidate(second,declaration['arrival_pool_sha256'])
    denial=None
    while s['active_actions'] and not s['closed']:
        action=s['active_actions'][0]
        if runtime.authorize_probe_action(action,phase='outbound') is None:
            denial=deepcopy(runtime.audit[-1]);break
        consume(action,'second_view')
    reached=bool([*s['packet'].position,s['packet'].heading]==second['pose'] and not s['packet'].collision)
    arrival_state=decision_state(runtime);arrival_metrics=evaluate();arrival_visible=visible.copy()
    # Fixed pre-attempt measured points, actual arrival depth; no new group ID assumption.
    _,depth_support=observing_camera(SimpleNamespace(keyframes=[s['packet'].frame],config=config),frozen_asset['points'])
    arrival_support=capture_support(mapper)[2]
    new_keys=set(arrival_support)-set(prefix_support)
    prior_direction_gain=sum(bin(arrival_support[k][1]&~prefix_support[k][1]).count('1') for k in set(arrival_support)&set(prefix_support))
    post_pool=None
    if not s['closed']:
        b.select_target(0)
        post_pool=deepcopy(b.scenes[0]['last_selection'])
        assets,desc=observed_descriptors(runtime,post_pool['candidates'])
        post_pool['features']=route_instance_features(mapper,post_pool['candidates'],assets,desc)
        post_pool['executed_next_inspection']=False
        runtime.begin_return()
        while not s['closed']:
            action=runtime.next_local_action(0)
            if action is None:break
            consume(action,'return')
    terminal=runtime.sensor_episode_summary(0)
    outcome=dict(candidate_id=cid,group=second['group'],first_candidate_id=candidate['candidate_id'],
        first_target_reached=first_reached,second_receipt=second_receipt,first_arrival=first_arrival_metrics,
        first_arrival_state_sha256=first_arrival_state['sha256'],structure_seed=protocol['structure_seed'],
        prefix_state_sha256=prefix_state['sha256'],arrival_state_sha256=arrival_state['sha256'],
        final_state_sha256=decision_state(runtime)['sha256'],candidate_receipt=receipt,
        target_reached=reached,outbound_denial=denial,
        second_view_actions=sum(x['phase']=='second_view' for x in actions),
        outbound_actions=sum(x['phase']=='outbound' for x in actions),
        return_actions=sum(x['phase']=='return' for x in actions),total_paid_actions=world.step_count,
        total_task_budget=protocol['total_task_budget'],collisions=int(world.collisions),termination=terminal['termination'],
        before=before,arrival=arrival_metrics,after_return=evaluate(),
        actual_second_arrival_depth_support_on_initial_face=int(depth_support),initial_face_descriptor_points=len(frozen_asset['points']),
        second_view_new_support_keys=len(new_keys),second_view_new_directions_on_initial_support=prior_direction_gain,
        observed_support_counts_are_not_surface_area=True,instance_attribution_available=False,
        two_stage_new_visible_surface_m2=float(np.count_nonzero(arrival_visible&~prefix_visible))*weight,
        second_view_new_visible_surface_m2=float(np.count_nonzero(arrival_visible&~first_visible))*weight,
        all_new_visible_surface_m2=float(np.count_nonzero(visible&~prefix_visible))*weight,
        visibility_reference_seed=ep['reference_seeds'][0],
        path_distance_m=sum(x['action']=='forward' for x in actions)*config.resolution_m,
        second_view_path_distance_m=sum(x['action']=='forward' and x['phase']=='second_view' for x in actions)*config.resolution_m,
        outbound_path_distance_m=sum(x['action']=='forward' and x['phase']=='outbound' for x in actions)*config.resolution_m,
        action_time_s=world.step_count*config.action_duration_s,
        post_arrival_candidates=0 if post_pool is None else len(post_pool['candidates']),
        post_arrival_back_candidates=0 if post_pool is None else sum(x['group'].endswith(('_entry','_deep')) for x in post_pool['candidates']),
        modules_called=sorted({c['module'] for c in b.calls}),future_rewards_used_by_planner=False,
        external_prefix_actions=step,full_budget_policy_comparison=False,semantic_efficacy_proven=False)
    deterministic=dict(outcome=outcome,events=events,arrival_pool=post_pool,runtime_audit=runtime.audit,
        module_calls=b.calls,prefix_state=prefix_state,first_arrival_state=first_arrival_state,arrival_state=arrival_state)
    # Profiling durations are excluded from exact module-call replay identity.
    deterministic.pop('module_calls')
    mesh=mapper.mesh();mesh_arrays={k:np.asarray(getattr(mesh,k)) for k in ('vertices','triangles','vertex_colors')}
    if replay:
        if len(actions)!=len(expected):raise ValueError('action length differs')
        saved=json.loads(gzip.decompress((folder/'deterministic.json.gz').read_bytes()))
        if digest(deterministic)!=digest(saved):raise ValueError('state, observations or physical metrics differ on fresh replay')
        with np.load(folder/'final_mesh.npz',allow_pickle=False) as stored:
            for name,v in mesh_arrays.items():np.testing.assert_array_equal(v,stored[name])
        write(folder/'verification.json',dict(status='passed',physical_actions=world.step_count,
            sensor_packets_actions_states_feedback_candidates_metrics_mesh_exact=True))
    else:
        write(folder/'result.json',outcome);gzwrite(folder/'deterministic.json.gz',deterministic)
        gzwrite(folder/'module_calls.json.gz',b.calls)
        write(folder/'timing.json',dict(execution_mapping_and_evaluation_wall_time_s=perf_counter()-started,
            planning_runtime_is_not_isolated_benchmark=True))
        np.savez_compressed(folder/'final_mesh.npz',**mesh_arrays)
        np.savez_compressed(folder/'visibility.npz',prefix=prefix_visible,first_arrival=first_visible,arrival=arrival_visible,final=visible,
            reference=e.reference,classes=e.classes,sample_weight_m2=weight)
    return outcome


def main():
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True)
    p.add_argument('--replay-index',type=int);a=p.parse_args()
    if a.replay_index is not None:
        m=read(a.output/'manifest.json');verify_sources(m)
        run(a.output,m['protocol'],m['protocol']['declarations'][a.replay_index],True);return
    preflight=ROOT/'audit_results/feasible_aperture_v16_3_preflight_20260914'
    scope_path=preflight/'second_view_declarations.json';scope=read(scope_path)
    prior=read(preflight/'manifest.json')
    if prior['status']!='complete':raise ValueError('complete shared-pool preflight required')
    original=scope['source_protocol'];history=original['history']
    inputs=[history,scope['source'],str(preflight.relative_to(ROOT))]
    inventories={}
    for name in inputs:
        folder=ROOT/name;inventories[name]=sha(folder/'artifact_hashes.json')
        for n,v in read(folder/'artifact_hashes.json').items():
            if sha(folder/n)!=v:raise ValueError('input artifact changed')
    protocol={**original,'stage_source':scope['source'],'declarations':scope['declarations'],
        'execution_scope':'exact first approach, one declared second viewpoint, guarded return; no full-budget policy comparison'}
    for n,v in prior['source_sha256'].items():
        if sha(ROOT/n)!=v:raise ValueError('preflight source changed: '+n)
    names=set(prior['source_sha256']) | {str(Path(__file__).relative_to(ROOT)),str(scope_path.relative_to(ROOT))}
    sources={n:sha(ROOT/n) for n in sorted(names)}
    m=dict(status='running',protocol=protocol,source_sha256=sources,input_inventory_sha256=inventories,
        training_allowed=False,parent_layouts=1,worlds_before_freeze=0,
        historical_test_fixture_revision='V16.2 preflight retains old test; current unit fix does not change imported runtime code')
    a.output.mkdir(parents=True,exist_ok=False);write(a.output/'manifest.json',m)
    with zipfile.ZipFile(a.output/'sources.zip','x',zipfile.ZIP_DEFLATED) as z:
        for n in sources:z.write(ROOT/n,n)
    rows=[]
    try:
        for index,candidate in enumerate(protocol['declarations']):
            if shutil.disk_usage(ROOT).free<64*1024**2:raise RuntimeError('storage guard below 64 MiB before next branch')
            row=run(a.output,protocol,candidate)
            subprocess.run([sys.executable,str(Path(__file__)),'--output',str(a.output),'--replay-index',str(index)],check=True)
            rows.append(row);write(a.output/'partial.json',rows)
            print(json.dumps(dict(first=row['first_candidate_id'],candidate_id=row['candidate_id'],replayed=True,reached=row['target_reached'],
                new_support=row['second_view_new_support_keys'],back_candidates=row['post_arrival_back_candidates'])),flush=True)
        verify_sources(m)
        write(a.output/'summary.json',dict(status='complete',physical_probes=len(rows),independent_replays=len(rows),
            reached=sum(x['target_reached'] for x in rows),returns=sum(x['termination']['returned_to_anchor'] for x in rows),
            collisions=sum(x['collisions'] for x in rows),failed=sum(x['termination']['failed'] for x in rows),
            training_allowed=False,semantic_efficacy_proven=False))
        m['status']='complete'
    except Exception as error:m.update(status='failed',error=repr(error));raise
    finally:write(a.output/'manifest.json',m);seal(a.output)


if __name__=='__main__':main()
