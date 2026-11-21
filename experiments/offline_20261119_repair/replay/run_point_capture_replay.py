#!/usr/bin/env python3
"""Capture weighted hypothesis points at fixed states54/109; old target/action replay only."""
import copy,hashlib,importlib.util,json,os,sys,time,traceback
from pathlib import Path
from types import MethodType
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ[key]='1'
B=Path('/home/ouc/nso_robot/analyses/dual_template_exploration_20261119')
OLD=B/'layouts/shelf_left/runs/S/attempt_001'
OUT=Path(__file__).resolve().parents[2]
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(p,v):Path(p).write_text(json.dumps(v,indent=2,allow_nan=False)+'\n')
def load(name,p):
 s=importlib.util.spec_from_file_location(name,p);m=importlib.util.module_from_spec(s);sys.modules[name]=m;s.loader.exec_module(m);return m
def capture_points(repair,models,measured_ray,route,paid,out):
 import numpy as np
 from nso.semantic_completion_v3 import ObjectCompletionModel
 arrays={};value={'paid_action':paid,'candidate_id':route['candidate_id'],'outbound_actions':len(route['outbound_actions']),'models':{}}
 states=list(dict.fromkeys(tuple(map(int,x))for x in route['outbound_states'][1:]))
 for name,model in models.items():
  old=np.zeros(len(model.points));new=old.copy()
  for st in states:
   v,receipt=repair.trace_gain(model,st[:2],st[2],measured_ray,repair_enabled=False)
   np.testing.assert_array_equal(v,ObjectCompletionModel.gain(model,st[:2],st[2],measured_ray))
   changed,receipt=repair.trace_gain(model,st[:2],st[2],measured_ray,repair_enabled=True)
   assert np.all(changed>=v)
   np.maximum(old,v,out=old);np.maximum(new,changed,out=new)
  arrays.update({name+'_points':model.points.copy(),name+'_hypothesis_ids':model.hypothesis_ids.copy(),name+'_weights':model.weights.copy(),name+'_original_union_gain':old,name+'_repaired_union_gain':new})
  value['models'][name]={'original_route_area_m2':float(old.sum()),'repaired_route_area_m2':float(new.sum()),'points':len(model.points),'original_gain_elementwise_identical':True,'object_geometry_diagnostics':model.audit['objects']}
 target=out/f'state_{paid:03d}_predictions.npz';np.savez_compressed(target,**arrays)
 value['predictions_path']=str(target);value['predictions_sha256']=sha(target)
 value['evaluation_truth_loaded']=False
 return value

def main():
 import numpy as np
 started=time.monotonic();rows=[];scores_log=[];hooks=[];freeze=json.loads((OUT/'freeze.json').read_text())
 try:
  for p,h in freeze['snapshot_sha256'].items():assert sha(p)==h
  b=load('versioned_replay_old_runner',B/'run.py');b.dependencies();first_expected=b.verify('shelf_left')
  hook_path=OUT/'code/method/occlusion_repair.py';hook=load('versioned_point_capture_repair',hook_path)
  assert sha(hook.__file__)==freeze['repair_sha256']
  world,tr,first,comp,runtime,effective,owner=b.bootstrap('shelf_left','S')
  assert first.sha256()==first_expected['initial_packet_sha256']
  mapper=runtime.states[0]['mapper'];backend=comp._cpu_backend;state=backend.scenes[0]
  choices=json.loads((OLD/'choices.json').read_text());steps=json.loads((OLD/'steps.json').read_text());cursor=[0]
  from nso.cpu_sensor_contract_v10 import digest,json_value
  def fixed(self,scene_idx):
   s=self.scenes[scene_idx];paid=s['packet'].action_id;record=copy.deepcopy(choices[cursor[0]]);expected=record['outputs']
   inputs=dict(semantic_version=s['semantic_version'],graph_version=s['graph_version'],planning_feedback_version=s['feedback']['planning_feedback_version'],actual_feedback_version=s['feedback']['feedback_version'],planning_camera_poses_sha256=digest(s['ledger'].planning_camera_poses()),remaining_budget=s['ledger'].remaining_budget)
   assert inputs==record['inputs']
   attempted=s['gain'].attempted_camera_mask(mapper,s['ledger'].planning_camera_poses());scores,_=self._scores_v10_1(s,expected['candidates'],attempted)
   errors={}
   for mode,value in scores.items():
    np.testing.assert_allclose(value,expected['scores'][mode],rtol=0,atol=1e-10);errors[mode]=float(np.max(np.abs(np.asarray(value)-expected['scores'][mode])))if len(value)else 0.
   assert s['gain'].snapshot()==expected['gain_calibration']
   s['plans']+=1;assert s['plans']==expected['plan_number'];s['selected']=copy.deepcopy(expected['selected'])
   if s['selected']is not None:assert s['selected']['selection_call_id']==len(self.calls)+1
   s['last_selection']=expected;self._record(scene_idx,'STGHP','select_topo_target',inputs,expected,time.perf_counter())
   scores_log.append({'paid_action':paid,'max_absolute_score_error':errors,'recorded_candidate_id':None if s['selected']is None else s['selected']['candidate_id']})
   if paid in (54,109):
    cache=b.method._cache(mapper,'S');before=(mapper.frames,digest(mapper.belief))
    value=capture_points(hook,cache['models'],cache['ray'],s['selected'],paid,OUT)
    assert (mapper.frames,digest(mapper.belief))==before
    value.update(repair_source_sha256=freeze['repair_sha256'],hook_source_sha256=freeze['hook_sha256'],independent_replay_attempt='attempt_003')
    hooks.append(json_value(value));write(OUT/'gain_hook_checks.json',hooks)
    print(json.dumps({'fixed_state':paid,'original_S':value['models']['S']['original_route_area_m2'],'repaired_S':value['models']['S']['repaired_route_area_m2']}),flush=True)
   cursor[0]+=1
   return None if s['selected']is None else np.asarray(s['selected']['pose'][:2],int)
  backend.select_target=MethodType(fixed,backend)
  for old in steps[:109]:
   action=runtime.next_local_action(0);assert action==old['action']
   raw,col,done=world.step(action);p=b.packet(world,tr,raw,'shelf_left',action,col,done);assert p.sha256()==old['packet_sha256']
   assert list(world.position)==old['position']and world.heading==old['heading']and col==old['collision']
   owner=b.front.associate(mapper,raw,p.frame_id);assert owner['owner_sha256']==old['owner_sha256']
   b.method.observe(mapper,p.frame);runtime.observe(0,world.step_count,None,None,None,sensor_packet=p)
   assert mapper.frames==world.step_count+1 and len(mapper.quality)==old['measured_patch_count']and len(state['assets'])==old['measured_asset_count']
   assert state['gain'].snapshot()['posterior_mean']==old['gain_posterior']
   rows.append({'step':world.step_count,'packet_sha256':p.sha256(),'old_action_pose_owner_quality_assets_feedback_match':True})
  # Saved state109 capture only: action110 is NOT requested or executed.
  backend.select_target(0)
  assert [x['paid_action']for x in hooks]==[54,109]
  assert world.step_count==109 and mapper.frames==110 and runtime.states[0]['pending']is None
  for p,h in freeze['snapshot_sha256'].items():assert sha(p)==h
  for p,h in freeze['live_source_sha256'].items():assert sha(p)==h
  b.verify('shelf_left');write(OUT/'replayed_packets.json',rows);write(OUT/'score_checks.json',scores_log)
  write(OUT/'result.json',{'status':'completed_same_domain_point_capture_replay','paid_actions_replayed':109,'fused_frames':110,'fixed_states':[54,109],'original_score_calls_matched':len(scores_log),'new_policy_decisions':0,'action_110_executed':False,'repair_installed_into_actor':False,'repair_source_sha256':freeze['repair_sha256'],'hook_source_sha256':freeze['hook_sha256'],'source_snapshots_unchanged':True,'live_sources_unchanged':True,'original_packets_actions_poses_owners_quality_assets_feedback_all_match':True,'native_mesh_or_reference_recomputed':False,'actual_surface_increment_source':'attempt_001 separate replay with final 256-step native TSDF byte identity','elapsed_s':time.monotonic()-started})
  print(json.dumps({'status':'completed_point_capture_replay','elapsed_s':time.monotonic()-started}),flush=True)
 except BaseException as exc:
  write(OUT/'failure.json',{'error':repr(exc),'traceback':traceback.format_exc(),'completed_actions':len(rows),'elapsed_s':time.monotonic()-started});raise
if __name__=='__main__':main()
