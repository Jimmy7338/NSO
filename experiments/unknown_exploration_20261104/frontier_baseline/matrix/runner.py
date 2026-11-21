import os
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'): os.environ[key]='1'
import sys,json,hashlib
from pathlib import Path
from types import SimpleNamespace
import numpy as np
P=Path(__file__).resolve().parents[2]
S=P/'source';sys.path.insert(0,str(S))
from env.virtual3d_competition_v9 import create_competition_world
from env.grid_exploration import GridObservation, readonly_copy
from nso.frontier_policy import FrontierPolicy
from nso.execution_guard_v8 import ObservedExecutionGuard
from nso.observed_runtime_mapper_v10 import ObservedRuntimeMapperV10
from nso.cpu_sensor_contract_v10 import GridTransform,SensorPacket
from scripts.eval_competition_v9_1 import restore_evaluator,snapshot_metrics
from scripts.eval_counterfactual_views import coverage
from utils.reconstruction_metrics import ReconstructionEvaluator
from utils.counterfactual_surface_visibility import reference_visible,surface_increment

out=Path(__file__).parent / os.environ["RUN_KEY"];out.mkdir(exist_ok=False)
arr=os.environ["ARRANGEMENT"];budget=int(os.environ["BUDGET"])
w=create_competition_world('Q0',arr);config=w.config;tr=GridTransform(tuple(w.shape),config.resolution_m)
f=w.sense();sc=w.scan();anchor=(*map(int,w.position),int(w.heading))
p=SensorPacket('Q0_'+arr,'fresh-first-frame','frame-0',0,f,sc,tuple(map(int,w.position)),int(w.heading),'deterministic_simulator_rgbd_and_scan','simulator_exact_discrete_odometry',None,False,False).validate(tr,config)
expected=next(x for x in json.loads((P/'evidence/fresh_matrix/results.json').read_text()) if x['arrangement']==arr)
assert p.sha256()==expected['initial_packet_sha256'] and list(anchor)==list(expected['anchor'])
mapper=ObservedRuntimeMapperV10(w.shape,config);mapper.update(f,sc)
policy=FrontierPolicy(SimpleNamespace(resolution_m=config.resolution_m,robot_radius_m=config.robot_radius_m,sensor_range_m=config.max_depth_m,sensor_fov_deg=config.fov_deg),method='nearest_frontier',max_candidates=24)
guard=ObservedExecutionGuard(config.resolution_m,config.robot_radius_m)
count=json.loads((S/'configs/virtual3d/competition_v9_validation_protocol.json').read_text())['reference']['samples_requested'];e=ReconstructionEvaluator(w,count=count,seed=2026);weight=float(w.mesh.get_surface_area())/count;mid=config.width_m/2
ref=dict(points=e.reference,classes=e.classes,weights=np.full(len(e.reference),weight),prefix_seen=reference_visible(e.reference,f,e.truth,f.world_from_camera,config.max_depth_m),west_slot=(e.classes!=1)&(e.reference[:,0]<mid),east_slot=(e.classes!=1)&(e.reference[:,0]>=mid),reachable=w.reachable,vertices=np.asarray(w.mesh.vertices),triangles=np.asarray(w.mesh.triangles))
e=restore_evaluator(w,ref);future=np.zeros_like(ref['prefix_seen']);_,before=snapshot_metrics(mapper,w,e,ref,(.05,.10))
assert before==expected['before_metrics'], 'initial metric mismatch'
for matrix in ('fresh_matrix','deployed_policy_matrix','coverage_ablation_matrix'):
 for prior in json.loads((P/'evidence'/matrix/'results.json').read_text()):
  if prior['arrangement']==arr and prior['budget']==budget:
   assert before==prior['before_metrics'] and list(anchor)==list(prior['anchor']) and p.sha256()==prior['initial_packet_sha256'], 'prior matrix mismatch'
steps=[]; interventions=[];phase='explore';reason=None;collision=False
while len(steps)<budget:
 pos=tuple(map(int,w.position));heading=int(w.heading);remaining=budget-len(steps)
 # Only the immutable observed map and odometry enter the policy.
 obs=GridObservation(readonly_copy(mapper.belief),readonly_copy(mapper.visible),pos,heading,len(steps),collision)
 decision=None
 if phase=='explore':
  decision=policy.act(obs);action=decision.action
  if action=='stop': phase='return';interventions.append(dict(step=len(steps),event='policy_stop_return'))
 if phase=='return':
  ret=guard.return_plan(mapper.belief,pos,heading,anchor,remaining)
  if not ret.available: reason=ret.reason;break
  if not ret.actions: reason='returned_to_anchor';break
  action=ret.actions[0]
 check=guard.assess(mapper.belief,pos,heading,action,remaining)
 nextheading=(heading+(1 if action=='right' else -1 if action=='left' else 0))%4
 ret=guard.return_plan(mapper.belief,check.target,nextheading,anchor,remaining-1) if check.allowed else None
 if not check.allowed or not ret.available:
  rejection=check.reason if not check.allowed else ret.reason
  interventions.append(dict(step=len(steps),event='action_rejected_return',action=action,reason=rejection,phase=phase))
  if phase=='return': reason=rejection;break
  phase='return';continue
 f,collision,done=w.step(action);sc=w.scan()
 p=SensorPacket('Q0_'+arr,'fresh-first-frame',f'frame-{w.step_count}',int(w.step_count),f,sc,tuple(map(int,w.position)),int(w.heading),'deterministic_simulator_rgbd_and_scan','simulator_exact_discrete_odometry',action,bool(collision),bool(done)).validate(tr,config)
 mapper.update(f,sc);future|=reference_visible(e.reference,f,e.truth,f.world_from_camera,config.max_depth_m)
 cov=coverage(mapper,w.reachable)
 steps.append(dict(step=len(steps)+1,phase=phase,action=action,position=list(w.position),heading=int(w.heading),collision=bool(collision),packet_sha256=p.sha256(),covered_area_m2=cov*int(w.reachable.sum())*config.resolution_m**2,new_visible_surface_area_m2=surface_increment(ref['prefix_seen'],future,ref['weights']),decision_goal=None if decision is None or decision.goal is None else list(decision.goal),reserved_return_cost=ret.paid_cost))
 if collision or done:reason='collision' if collision else 'sensor_terminal';break
if reason is None:reason='budget_exhausted'
_,after=snapshot_metrics(mapper,w,e,ref,(.05,.10));returned=(*map(int,w.position),int(w.heading))==anchor
result=dict(scope='Q0 controlled first-frame simulation; related arrangements, not independent scenes' ,arrangement=arr,baseline='local FrontierPolicy nearest_frontier plus observed-action and return guard',initial_packet_sha256=expected['initial_packet_sha256'],anchor=anchor,budget=budget,steps=len(steps),path_m=sum(s['action']=='forward' for s in steps)*config.resolution_m,collisions=int(w.collisions),termination=dict(reason=reason,returned_to_anchor=returned,failed=not returned or bool(w.collisions),current_footprint_known_safe=bool(guard.safe_grid(mapper.belief)[tuple(w.position)])),return_trigger=(interventions[0] if interventions else None),interventions=interventions,before_metrics=before,after_metrics=after,coverage_gain_m2=after['covered_area_m2']-before['covered_area_m2'],new_surface_m2=surface_increment(ref['prefix_seen'],future,ref['weights']),policy_source_sha256=hashlib.sha256((S/'nso/frontier_policy.py').read_bytes()).hexdigest())
(out/'steps.json').write_text(json.dumps(steps,indent=2));(out/'result.json').write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2))
