#!/usr/bin/env python3
"""Replay one sealed action/target sequence; no policy selection or hardware.

The old runtime still owns fusion and execution accounting. Its target selector
is replaced by a lookup of the sealed target/candidates. Scores are recomputed
only to assert equality, never to choose an alternative action or target.
"""
import argparse
import copy
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys
import time
import traceback
from types import MethodType

for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):
    os.environ[key]='1'
HERE=Path(__file__).resolve().parent
OLD=HERE.parents[1]/'dual_template_exploration_20261119'
ATTEMPT=OLD/'layouts/shelf_left/runs/S/attempt_001'
REFERENCE=OLD/'audit/references/shelf_left'
CHECKPOINTS=(0,24,54,109,127,167,208,256)

def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def write(path,value):
    path=Path(path);temporary=path.with_suffix(path.suffix+'.tmp')
    temporary.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n');temporary.replace(path)
def load(name,path):
    spec=importlib.util.spec_from_file_location(name,path)
    module=importlib.util.module_from_spec(spec);sys.modules[name]=module;spec.loader.exec_module(module);return module

class IndependentReference:
    """Read-only evaluator, never assigned to mapper/runtime/backend."""
    def __init__(self,config):
        import numpy as np
        import open3d as o3d
        sys.path.insert(0,str(OLD/'audit'))
        from surface_metrics import evaluate_surface,ray_scene,visible_reference
        self.evaluate_surface=evaluate_surface;self.ray_scene=ray_scene;self.visible_reference=visible_reference
        receipt=json.loads((REFERENCE/'RECEIPT.json').read_text())
        for name,key in [('fixed_reference.npz','reference_npz_sha256'),('source_mesh.ply','source_mesh_sha256'),('PROTOCOL.json','protocol_sha256')]:
            assert sha(REFERENCE/name)==receipt[key],name
        self.reference=dict(np.load(REFERENCE/'fixed_reference.npz',allow_pickle=False))
        self.source=o3d.io.read_triangle_mesh(str(REFERENCE/'source_mesh.ply'));self.source_ray=ray_scene(self.source)
        self.weights={k:receipt['metadata'][k]['sampled_source_area_m2']/receipt['metadata'][k]['drawn_samples'] for k in ('all','open31','closed118')}
        self.visible={k:np.zeros(len(self.reference[k]),bool) for k in self.weights}
        self.previous={};self.previous_action=None;self.config=config;self.checkpoints=[]
        self.receipt=receipt

    def observe_pose(self,pose):
        import numpy as np
        for key,visible in self.visible.items():
            ids=np.flatnonzero(~visible)
            if len(ids):
                newly,_,_=self.visible_reference(self.reference[key][ids],self.source_ray,np.asarray([pose]),self.config)
                visible[ids[newly]]=True

    def checkpoint(self,mapper,world,out):
        import numpy as np
        import open3d as o3d
        action=int(world.step_count);folder=out/'checkpoints'/f'action_{action:03d}';folder.mkdir(parents=True)
        mesh=mapper.mesh();assert len(mesh.triangles)
        assert o3d.io.write_triangle_mesh(str(folder/'native_tsdf_mesh.ply'),mesh,write_ascii=False)
        ray=self.ray_scene(mesh);metrics=self.evaluate_surface(mesh,self.source,self.reference)
        arrays={};groups={}
        for key,weight in self.weights.items():
            distance=ray.compute_distance(o3d.core.Tensor(self.reference[key]),nthreads=1).numpy()
            visible=self.visible[key];arrays[key+'_distance_m']=distance;arrays[key+'_visible']=visible
            row={'reference_samples':len(distance),'area_weight_per_sample_m2':weight,
                'fixed_reference_visible_area_estimate_m2':weight*len(distance),
                'acquired_pose_continuous_FOV_visible_samples':int(visible.sum()),
                'acquired_pose_continuous_FOV_visible_area_estimate_m2':weight*int(visible.sum())}
            for threshold in (.05,.10):
                tag=f'{int(round(threshold*100)):02d}cm';correct=distance<=threshold;arrays[key+'_correct_'+tag]=correct
                row[tag]={'correct_samples':int(correct.sum()),'recall':float(correct.mean()),
                    'correctly_reconstructed_area_estimate_m2':weight*int(correct.sum())}
                if self.previous_action is not None:
                    before=self.previous[key+'_correct_'+tag];gained=correct&~before;lost=before&~correct
                    row[tag]['since_previous_checkpoint']={
                        'start_action':self.previous_action,'end_action':action,
                        'new_correct_samples':int(gained.sum()),'lost_correct_samples':int(lost.sum()),
                        'net_correct_samples':int(correct.sum())-int(before.sum()),
                        'new_correct_area_estimate_m2':weight*int(gained.sum()),
                        'lost_correct_area_estimate_m2':weight*int(lost.sum()),
                        'net_correct_area_estimate_m2':weight*(int(correct.sum())-int(before.sum())),
                        'recall_delta_percentage_points':100*float(correct.mean()-before.mean())}
            if self.previous_action is not None:
                old=self.previous[key+'_visible'];row['new_continuous_FOV_visible_area_estimate_m2']=weight*int((visible&~old).sum())
            groups[key]=row
        np.savez_compressed(folder/'fixed_reference_distances_masks.npz',**arrays)
        row={'paid_action':action,'mapper_fused_frames':mapper.frames,'position':list(world.position),'heading':world.heading,
            'native_mesh_sha256':sha(folder/'native_tsdf_mesh.ply'),'mesh_vertices':len(mesh.vertices),'mesh_triangles':len(mesh.triangles),
            'groups':groups,'surface_metrics':metrics,'fixed_reference_sha256':self.receipt['reference_npz_sha256']}
        write(folder/'metrics.json',row);self.checkpoints.append(row)
        self.previous={key:value.copy() for key,value in arrays.items()};self.previous_action=action
        write(out/'checkpoints.json',self.checkpoints)
        print(json.dumps({'checkpoint':action,'all_correct_area_5cm':groups['all']['05cm']['correctly_reconstructed_area_estimate_m2'],
                          'open_recall_5cm':groups['open31']['05cm']['recall'],'closed_recall_5cm':groups['closed118']['05cm']['recall']}),flush=True)

def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--run',action='store_true');parser.add_argument('--output',type=Path,default=HERE/'attempt_001')
    args=parser.parse_args()
    if not args.run:parser.print_help();return
    out=args.output;out.mkdir(parents=True,exist_ok=False);started=time.monotonic();step_rows=[];score_checks=[];hooks=[]
    try:
        import numpy as np
        import open3d as o3d
        b=load('sealed_dual_template_replay_helpers',OLD/'run.py');b.dependencies();initial=b.verify('shelf_left')
        frozen_manifest=json.loads((ATTEMPT/'artifact_manifest.json').read_text())
        for name,digest in frozen_manifest['files_sha256'].items():assert sha(ATTEMPT/name)==digest,name
        steps=json.loads((ATTEMPT/'steps.json').read_text());choices=json.loads((ATTEMPT/'choices.json').read_text())
        expected_result=json.loads((ATTEMPT/'result.json').read_text());assert len(steps)==256
        world,tr,first,comp,runtime,effective,owner=b.bootstrap('shelf_left','S')
        assert first.sha256()==initial['initial_packet_sha256']==expected_result['initial_packet_sha256']
        mapper=runtime.states[0]['mapper'];backend=comp._cpu_backend;state=backend.scenes[0]
        evaluator=IndependentReference(world.config);evaluator.observe_pose(first.frame.world_from_camera)
        evaluator.checkpoint(mapper,world,out)
        hook_path=HERE.parent/'model_audit/replay_hook.py'
        assert hook_path.exists(),'Declared independent hook not yet ready; retry explicitly with a new attempt only'
        hook=load('independent_replay_gain_hook',hook_path)
        from nso.cpu_sensor_contract_v10 import digest,json_value
        cursor=[0]
        def sealed_selection(self,scene_idx):
            s=self.scenes[scene_idx];p=s['packet'];i=cursor[0];record=copy.deepcopy(choices[i]);expected=record['outputs'];paid=p.action_id
            assert s['graph_version']==s['semantic_version']==mapper.frames
            inputs=dict(semantic_version=s['semantic_version'],graph_version=s['graph_version'],
                planning_feedback_version=s['feedback']['planning_feedback_version'],actual_feedback_version=s['feedback']['feedback_version'],
                planning_camera_poses_sha256=digest(s['ledger'].planning_camera_poses()),remaining_budget=s['ledger'].remaining_budget)
            assert inputs==record['inputs'],('runtime input mismatch',paid)
            attempted=s['gain'].attempted_camera_mask(mapper,s['ledger'].planning_camera_poses())
            scores,rows=self._scores_v10_1(s,expected['candidates'],attempted)
            errors={}
            for mode in scores:
                a=np.asarray(scores[mode],float);e=np.asarray(expected['scores'][mode],float)
                np.testing.assert_allclose(a,e,rtol=0,atol=1e-10,err_msg=f'paid={paid} mode={mode}')
                errors[mode]=float(np.max(np.abs(a-e)))if len(a)else 0.
            assert s['gain'].snapshot()==expected['gain_calibration']
            s['plans']+=1;assert s['plans']==expected['plan_number'];s['selected']=copy.deepcopy(expected['selected'])
            if s['selected'] is not None:assert s['selected']['selection_call_id']==len(self.calls)+1
            s['last_selection']=copy.deepcopy(expected)
            self._record(scene_idx,'STGHP','select_topo_target',inputs,expected,time.perf_counter())
            selected=s['selected'];selected_id=None if selected is None else selected['candidate_id']
            selected_row=next((r for r in rows if r['candidate_id']==selected_id),None)
            check={'paid_action':paid,'recorded_plan_number':s['plans'],'candidate_count':len(expected['candidates']),
                'selected_candidate_id':selected_id,'scores_max_absolute_error':errors,'selected_score_audit':selected_row,
                'selected_outbound_actions':None if selected is None else selected['outbound_actions'],
                'no_candidate_generation_or_argmax':True}
            score_checks.append(json_value(check));write(out/'score_checks.json',score_checks)
            if paid in CHECKPOINTS and selected is not None:
                cache=b.method._cache(mapper,'S');before_frames=mapper.frames;before_belief=digest(mapper.belief)
                value=hook.audit_state(mapper,cache['models'],cache['ray'],selected,paid)
                assert mapper.frames==before_frames and digest(mapper.belief)==before_belief
                hooks.append(json_value(value));write(out/'gain_hook_checks.json',hooks)
            cursor[0]+=1
            return None if selected is None else np.asarray(selected['pose'][:2],int)
        backend.select_target=MethodType(sealed_selection,backend)
        for old in steps:
            action=runtime.next_local_action(0);assert action==old['action'],('action mismatch',world.step_count,action,old['action'])
            active=runtime.states[0]['option'];phase=runtime.states[0]['phase']
            assert phase==old['phase_before_action']
            assert (None if active is None else active['option_id'])==old['active_option_id']
            raw,collision,done=world.step(action);p=b.packet(world,tr,raw,'shelf_left',action,collision,done)
            assert p.sha256()==old['packet_sha256'],('packet mismatch',old['step'])
            assert list(world.position)==old['position'] and world.heading==old['heading']
            assert bool(collision)==old['collision'] and bool(done)==old['done']
            owner=b.front.associate(mapper,raw,p.frame_id);assert owner['owner_sha256']==old['owner_sha256']
            b.method.observe(mapper,p.frame);runtime.observe(0,world.step_count,None,None,None,sensor_packet=p)
            assert mapper.frames==world.step_count+1
            measured=b.base.evaluate(mapper.belief,world.reachable)
            for key in ['coverage_fraction','free_correct_fraction','false_occupied_fraction']:assert measured[key]==old[key]
            assert len(mapper.quality)==old['measured_patch_count'] and len(state['assets'])==old['measured_asset_count']
            assert state['gain'].snapshot()['posterior_mean']==old['gain_posterior']
            evaluator.observe_pose(raw.world_from_camera)
            step_rows.append({'step':world.step_count,'action':action,'position':list(world.position),'heading':world.heading,
                'packet_sha256':p.sha256(),'mapper_frames':mapper.frames,'coverage_fraction':measured['coverage_fraction'],
                'old_packet_pose_action_owner_quality_asset_feedback_match':True})
            if world.step_count in CHECKPOINTS:evaluator.checkpoint(mapper,world,out)
            if world.step_count%16==0 or world.step_count in CHECKPOINTS:
                write(out/'progress.json',{'paid_action':world.step_count,'elapsed_s':time.monotonic()-started,'score_calls_verified':cursor[0]})
        assert cursor[0]==len(choices)
        final=out/'checkpoints/action_256/native_tsdf_mesh.ply';oldmesh=ATTEMPT/'final_tsdf_mesh.ply'
        assert sha(final)==sha(oldmesh),'Final native TSDF bytes differ'
        np.testing.assert_array_equal(mapper.belief,np.load(ATTEMPT/'final_belief.npz',allow_pickle=False)['belief'])
        assert b.method.frame_receipts(mapper)==json.loads((ATTEMPT/'template_frame_receipts.json').read_text())
        assert runtime.sensor_episode_summary(0)['termination']==expected_result['termination']
        b.verify('shelf_left')
        write(out/'replayed_steps.json',step_rows)
        write(out/'result.json',{'status':'completed_fixed_action_and_target_replay','actions':256,'fused_frames':mapper.frames,
            'verified_recorded_score_calls':cursor[0],'checkpoints':list(CHECKPOINTS),'new_policy_decisions':0,
            'old_packet_pose_action_owner_quality_asset_feedback_match_all_steps':True,'final_native_TSDF_exact_bytes_match':True,
            'final_native_TSDF_sha256':sha(final),'final_belief_exact_match':True,'template_frame_receipts_exact_match':True,
            'source_attempt':str(ATTEMPT),'source_manifest_sha256':sha(ATTEMPT/'artifact_manifest.json'),
            'reference_sha256':evaluator.receipt['reference_npz_sha256'],'script_sha256':sha(__file__),
            'diagnostic_hook_sha256':sha(hook_path),'elapsed_s':time.monotonic()-started,
            'area_interpretation':'Monte Carlo area estimates on the SAME frozen uniform source-area samples; gross gain/loss/net separately. Not exact physical area.',
            'visibility_interpretation':'Evaluator-only continuous FOV source-mesh first-hit visibility along actual acquired poses, not pixel-resolved depth measurements or TSDF completeness.',
            'template_reward_interpretation':'Hypothetical area-weighted visibility score; neither measured surface area nor measured TSDF gain.',
            'hardware':False,'source_files_modified':False})
        print(json.dumps({'status':'completed_fixed_replay','seconds':time.monotonic()-started,'output':str(out)}),flush=True)
    except BaseException as exc:
        write(out/'failure.json',{'status':'failed_or_interrupted_replay_not_experimental_result','error':repr(exc),
            'traceback':traceback.format_exc(),'completed_actions':len(step_rows),'verified_score_calls':len(score_checks),
            'elapsed_s':time.monotonic()-started});raise

if __name__=='__main__':main()
