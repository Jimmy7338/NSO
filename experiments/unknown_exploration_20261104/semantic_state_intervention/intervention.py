import os
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ[k]='1'
import sys,json,hashlib,time,traceback
from pathlib import Path
from types import SimpleNamespace
import numpy as np
E=Path(__file__).resolve().parent.parent;P=E/'evidence';R=E/'source';sys.path.insert(0,str(R));sys.path.insert(0,str(E/'geometric_weight_control'))
from env.virtual3d_competition_v9 import create_competition_world
from nso.components import NSO_Components
from weight2 import GWeight2
from nso_bridge.exploration_joint_policy import ExplorationJointRuntimeIntegration
from nso.cpu_sensor_contract_v10 import GridTransform,SensorPacket
from scripts.eval_competition_v9_1 import restore_evaluator,snapshot_metrics
from scripts.eval_counterfactual_views import coverage
from utils.counterfactual_surface_visibility import reference_visible,surface_increment
from utils.reconstruction_metrics import ReconstructionEvaluator
from nso_bridge.exploration_joint_policy import ExplorationJointCPUFourModules
class Switchable(GWeight2):
 policy_variant=ExplorationJointCPUFourModules.policy_variant
 def _scores_v10_1(self,s,routes,attempted):
  if self.args.intervention_weight2:return super()._scores_v10_1(s,routes,attempted)
  return ExplorationJointCPUFourModules._scores_v10_1(self,s,routes,attempted)
D=Path(__file__).parent;OUT=Path(os.environ['NSO_OUTPUT_ROOT'])/'execution';OUT.mkdir(exist_ok=False);sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest();protocol=json.loads((D/'portable_protocol.json').read_text());assert sha(Path(__file__))==protocol['harness_sha256'] and sha(E/'geometric_weight_control/weight2.py')==protocol['wrapper_sha256']
arch=P/'deployed_policy_matrix';oldsteps=json.loads((arch/'shelf_west_96_S_steps.json').read_text());oldchoices=json.loads((arch/'shelf_west_96_S_choices.json').read_text());oldresult=json.loads((arch/'shelf_west_96_S_result.json').read_text());assert oldchoices[2]['inputs']['remaining_budget']==55

def run(condition):
 target=OUT/condition;target.mkdir();w=create_competition_world('Q0','shelf_west');tr=GridTransform(tuple(w.shape),w.config.resolution_m);frame=w.sense();scan=w.scan();p=SensorPacket('Q0_shelf_west','fresh-first-frame','frame-0',0,frame,scan,tuple(map(int,w.position)),int(w.heading),'deterministic_simulator_rgbd_and_scan','simulator_exact_discrete_odometry',None,False,False).validate(tr,w.config);assert p.sha256()==oldresult['initial_packet_sha256'];anchor=(*p.position,p.heading);assert list(anchor)==oldresult['anchor']
 count=json.loads((R/'configs/virtual3d/competition_v9_validation_protocol.json').read_text())['reference']['samples_requested'];e=ReconstructionEvaluator(w,count=count,seed=2026);weight=float(w.mesh.get_surface_area())/count;mid=w.config.width_m/2;ref=dict(points=e.reference,classes=e.classes,weights=np.full(len(e.reference),weight),prefix_seen=reference_visible(e.reference,frame,e.truth,frame.world_from_camera,w.config.max_depth_m),west_slot=(e.classes!=1)&(e.reference[:,0]<mid),east_slot=(e.classes!=1)&(e.reference[:,0]>=mid),reachable=w.reachable,vertices=np.asarray(w.mesh.vertices),triangles=np.asarray(w.mesh.triangles));e=restore_evaluator(w,ref);future=np.zeros_like(ref['prefix_seen'])
 args=SimpleNamespace(nso_backend='cpu_v10',eval=True,train_global=False,use_open_vocab_semantic=True,use_topo_graph=True,use_rpn_uq=True,use_igcr=True,cpu_score_mode='S',cpu_disable_feedback=False,cpu_max_candidates=12,cpu_coverage_slots=4,cpu_planner_revision='v10_3_1',cpu_measured_novelty_floor=.25,run_id='fresh-S',intervention_weight2=False)
 c=NSO_Components(args);c.initialize('cpu',1,*w.shape,*w.shape);c._cpu_backend=Switchable(args,1,w.shape);c.capabilities=dict(c._cpu_backend.capabilities);rt=ExplorationJointRuntimeIntegration(c,1,w.shape);rt.start_sensor_episode(0,config=w.config,transform=tr,packets=[p],total_budget=96,return_anchor=anchor,paid_prefix_actions=0);_,before=snapshot_metrics(rt.states[0]['mapper'],w,e,ref,(.05,.10));assert before==oldresult['before_metrics'];steps=[];switched=False;boundary=None
 while not rt.states[0]['closed']:
  if len(steps)==41 and not switched:
   s=rt.states[0];ledger=c._cpu_backend.scenes[0]['ledger'];assert not s['pending'] and not s['active_actions'] and ledger.remaining_budget==55 and s['packet'].sha256()==oldsteps[40]['packet_sha256'];assert len([x for x in c._cpu_backend.calls if x['method']=='select_topo_target'])==2
   for a,b in zip([dict(inputs=x['inputs'],outputs=x['outputs']) for x in c._cpu_backend.calls if x['method']=='select_topo_target'],oldchoices[:2]):assert a==b
   boundary=dict(action=41,remaining_budget=ledger.remaining_budget,packet_sha256=p.sha256(),pose=[*p.position,p.heading],anchor=anchor,unpaid_observations=0,active_actions=s['active_actions'],pending=s['pending'],prior_planning_calls=2,surface_at_boundary_m2=steps[-1]['new_visible_surface_area_m2'],coverage_at_boundary_m2=steps[-1]['covered_area_m2']);(target/'boundary.json').write_text(json.dumps(boundary,indent=2));args.cpu_score_mode='S' if condition=='S' else 'G';args.intervention_weight2=condition=='G_weight2';switched=True
  action=rt.next_local_action(0);choices=[dict(inputs=x['inputs'],outputs=x['outputs']) for x in c._cpu_backend.calls if x['method']=='select_topo_target'];(target/'choices.json').write_text(json.dumps(choices,indent=2));(target/'runtime_audit.json').write_text(json.dumps(rt.audit,indent=2))
  if switched and len(choices)>=3 and not (target/'boundary_check.json').exists():
   fresh=choices[2];old=oldchoices[2];assert fresh['inputs']==old['inputs'] and fresh['outputs']['candidate_audit']==old['outputs']['candidate_audit'] and fresh['outputs']['candidates']==old['outputs']['candidates'];a=fresh['outputs']['score_audit'];b=old['outputs']['score_audit'];assert len(a)==len(b)
   for x,y in zip(a,b):assert {k:v for k,v in x.items() if k!='posthoc_G_weight2'}==y
   assert all(fresh['outputs']['scores'][k]==old['outputs']['scores'][k] for k in old['outputs']['scores'] if not (condition=='G_weight2' and k=='G'))
   (target/'boundary_check.json').write_text(json.dumps(dict(candidate_map_components_match=True,selected=fresh['outputs']['selected']['pose'] if fresh['outputs']['selected'] else None),indent=2))
  if action is None:break
  frame,col,done=w.step(action);scan=w.scan();p=SensorPacket('Q0_shelf_west','fresh-first-frame',f'frame-{w.step_count}',int(w.step_count),frame,scan,tuple(map(int,w.position)),int(w.heading),'deterministic_simulator_rgbd_and_scan','simulator_exact_discrete_odometry',action,bool(col),bool(done)).validate(tr,w.config);future|=reference_visible(e.reference,frame,e.truth,frame.world_from_camera,w.config.max_depth_m);rt.observe(0,w.step_count,None,None,None,sensor_packet=p);cov=coverage(rt.states[0]['mapper'],w.reachable);steps.append(dict(step=w.step_count,action=action,position=list(w.position),heading=w.heading,collision=bool(col),done=bool(done),packet_sha256=p.sha256(),covered_area_m2=cov*int(w.reachable.sum())*w.config.resolution_m**2,new_visible_surface_area_m2=surface_increment(ref['prefix_seen'],future,ref['weights'])));(target/'steps.json').write_text(json.dumps(steps,indent=2))
  if len(steps)<=41:assert steps[-1]==oldsteps[len(steps)-1],f'prefix mismatch at {len(steps)}'
 _,after=snapshot_metrics(rt.states[0]['mapper'],w,e,ref,(.05,.10));term=rt.sensor_episode_summary(0)['termination'];assert switched and (target/'boundary_check.json').exists()
 if condition=='S':assert steps==oldsteps and after==oldresult['after_metrics'] and term==oldresult['termination'], 'S full reproduction mismatch'
 result=dict(condition=condition,steps=len(steps),paid_prefix=41,remaining_at_switch=55,before=before,after=after,boundary=boundary,additional_surface_m2=steps[-1]['new_visible_surface_area_m2']-boundary['surface_at_boundary_m2'],additional_coverage_m2=after['covered_area_m2']-boundary['coverage_at_boundary_m2'],post_switch_path_m=.2*sum(x['action']=='forward' and not x['collision'] for x in steps[41:]),collisions=w.collisions,termination=term,archived_S_exact_reproduction=condition=='S');(target/'result.json').write_text(json.dumps(result,indent=2));return result
results=[]
for name in ('S','G','G_weight2'):
 try:
  result=run(name);results.append(result);(OUT/'partial.json').write_text(json.dumps(results,indent=2));print(name,'done',flush=True)
 except Exception as ex:
  (OUT/'failure.json').write_text(json.dumps(dict(condition=name,error=repr(ex),traceback=traceback.format_exc()),indent=2));raise
(OUT/'results.json').write_text(json.dumps(results,indent=2))
