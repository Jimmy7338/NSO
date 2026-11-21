import os
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS','VECLIB_MAXIMUM_THREADS','BLIS_NUM_THREADS'):os.environ[k]='1'
import sys,json,hashlib,time,traceback,gc,resource
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace
import numpy as np
R=Path('/home/ouc/nso_robot/workspaces/nso_research_20261029/NSO');sys.path.insert(0,str(R))
from env.virtual3d_competition_v9 import create_competition_world
from nso.components import NSO_Components
B=Path('/home/ouc/nso_robot/workspaces/nso_hybrid_20261029/nso_bridge');sys.path.insert(1,str(B))
from nso_bridge.exploration_joint_policy import ExplorationJointCPUFourModules, ExplorationJointRuntimeIntegration
from nso.cpu_sensor_contract_v10 import GridTransform,SensorPacket
from scripts.eval_competition_v9_1 import restore_evaluator,snapshot_metrics
from scripts.eval_counterfactual_views import coverage
from scripts.run_cpu_semantic_inspection_v10_2 import write_json
from utils.counterfactual_surface_visibility import reference_visible,surface_increment
from utils.reconstruction_metrics import ReconstructionEvaluator
from nso.cpu_sensor_contract_v10 import json_value,digest
D=Path(__file__).resolve().parent/'Q2_execution';D.mkdir(exist_ok=False)
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
class FixedWeight7Comparator(ExplorationJointCPUFourModules):
    """Offline named comparator; native generator/scorer columns remain unchanged."""
    def select_target(self,scene_idx):
        super().select_target(scene_idx)
        s=self.scenes[scene_idx];result=deepcopy(s['last_selection']);routes=result['candidates']
        weight_scores=[float(row['common_total_proxy']+7.*row['prior_potentials']['G']) for row in result['score_audit']]
        result['scores']['G_lambda_dev7']=weight_scores
        native_S_selected=(result.get('selected') or {}).get('candidate_id')
        index=min(range(len(routes)),key=lambda i:(-weight_scores[i],routes[i]['cost'],routes[i]['candidate_id'])) if routes else None
        if index is not None and weight_scores[index]<=0:index=None
        selected=None if index is None else deepcopy(routes[index])
        if selected is not None:
            selected.update(option_id=f'{s["scene_id"]}/{s["episode_id"]}/plan-{s["plans"]}',selection_call_id=len(self.calls),selected_map_version=s['mapper'].frames,selected_feedback_version=s['feedback']['feedback_version'],parent_region=s['graph']['current_region'])
        s['selected']=selected
        result.update(selected=selected,mode='G_lambda_dev7',comparator=dict(name='G_lambda_dev7',lambda_fixed=7.,native_bookkeeping_dispatch='S',native_S_candidate_before_comparator=native_S_selected,formula='common+7*prior_G',native_score_columns_unchanged=True))
        s['last_selection']=json_value(result)
        event=self.calls[-1];assert event['method']=='select_topo_target'
        event['outputs']=json_value(result);event['output_sha256']=digest(result)
        return None if selected is None else np.asarray(selected['pose'][:2],int)
sources=sorted(p for d in ('env','nso','utils','scripts','configs/virtual3d') for p in (R/d).glob('*.py' if d!='configs/virtual3d' else '*.json'))
hashes={str(p.relative_to(R)):sha(p) for p in sources}
policy_files=[B/'nso_bridge/exploration_joint_policy.py',B/'nso_bridge/sensor_range_adapter.py'];policy_hashes={str(p):sha(p) for p in policy_files}
order=[('shelf_west',m) for m in ['C','G','S','X','G_lambda_dev7']]+[('shelf_east',m) for m in ['G_lambda_dev7','X','S','G','C']]
protocol=dict(schema='nso.fixed_Q2_ten_run.v1',status='frozen_before_world_construction',context='Q2',outer_seed=953,budget=96,run_order=order,
    source_sha256=hashes,policy_sha256=policy_hashes,harness_sha256=sha(Path(__file__)),approved_plan_sha256=sha(Path(__file__).parent/'controlled_plan.json'),
    native_modes=['C','G','S','X'],separate_comparator='G_lambda_dev7',comparator_formula='common+7*prior_G; native candidates, quality, feedback and return unchanged',
    comparator_weight_origin='ceil(Q1development actualS saved-state common lowerbound6.0379685851)=7, frozen before anyQ2world; no sweep or held-out choice',
    dispatch_note='Comparator calls nativeS planning bookkeeping once, then replaces only that comparator selectedtarget using saved nativecommon/priorG; all native score columns preserved and archived mode renamed explicitly.',
    simulator_radius_m=.2,physical_radius_m=.7212185868375829,physical_configuration_untouched=True,physical_adapters_disabled=True,
    initial='single actual simulator firstframe, zero paid prefix; no V35 publicposegraph/template',
    primary=['successful full-pose return/completepaidledger','unknowncoverage curve,AUCover96 and first90/95/99% thresholds'],
    AUC_rule='terminal achieved coverage carried to96 without newobservations; return/failure reported separately; failedshorter run never efficiencywin',
    secondary=['referencevisible surface','meshF1@5/10cm','distance/collisions','same-state scores and nonnegativeuniformweight feasibility'],
    statistical_unit='one syntheticparent, two related categoryplacements; ten trajectories not ten independent scenes',
    no_parameter_search=True,no_outcome_dependent_retries=True,threads=1,nice=10)
write_json(D/'protocol.json',protocol);(D/'executed_runner.py').write_bytes(Path(__file__).read_bytes())
t0=time.monotonic();initial={};references={};summaries=[]
def beat(status):write_json(D/'heartbeat.json',dict(status=status,elapsed_s=time.monotonic()-t0,pid=os.getpid(),completed=len(summaries),peak_rss_MiB=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024))
def run(arr,condition):
 key='Q2_'+arr+'_96_'+condition;beat('starting '+key)
 w=create_competition_world('Q2',arr);tr=GridTransform(tuple(w.shape),w.config.resolution_m);assert w.config.robot_radius_m==.2
 f=w.sense();sc=w.scan();p=SensorPacket('Q2_'+arr,'fresh-first-frame','frame-0',0,f,sc,tuple(map(int,w.position)),int(w.heading),'deterministic_simulator_rgbd_and_scan','simulator_exact_discrete_odometry',None,False,False).validate(tr,w.config)
 if arr not in initial:initial[arr]=p.sha256()
 else:assert initial[arr]==p.sha256(),'initialpacket mismatch within arrangement'
 if arr not in references:
  count=json.loads((R/'configs/virtual3d/competition_v9_validation_protocol.json').read_text())['reference']['samples_requested'];e=ReconstructionEvaluator(w,count=count,seed=2026);weight=float(w.mesh.get_surface_area())/count;mid=w.config.width_m/2
  references[arr]=dict(points=e.reference,classes=e.classes,weights=np.full(len(e.reference),weight),prefix_seen=reference_visible(e.reference,f,e.truth,f.world_from_camera,w.config.max_depth_m),west_slot=(e.classes!=1)&(e.reference[:,0]<mid),east_slot=(e.classes!=1)&(e.reference[:,0]>=mid),reachable=w.reachable,vertices=np.asarray(w.mesh.vertices),triangles=np.asarray(w.mesh.triangles))
 reference=references[arr];e=restore_evaluator(w,reference);future=np.zeros_like(reference['prefix_seen'])
 dispatch='S' if condition=='G_lambda_dev7' else condition
 args=SimpleNamespace(nso_backend='cpu_v10',eval=True,train_global=False,use_open_vocab_semantic=True,use_topo_graph=True,use_rpn_uq=True,use_igcr=True,cpu_score_mode=dispatch,cpu_disable_feedback=False,cpu_max_candidates=12,cpu_coverage_slots=4,cpu_planner_revision='v10_3_1',cpu_measured_novelty_floor=.25,run_id='Q2-'+condition)
 c=NSO_Components(args);c.initialize('cpu',1,*w.shape,*w.shape);backend=FixedWeight7Comparator if condition=='G_lambda_dev7' else ExplorationJointCPUFourModules;c._cpu_backend=backend(args,1,w.shape);c.capabilities=dict(c._cpu_backend.capabilities)
 rt=ExplorationJointRuntimeIntegration(c,1,w.shape);anchor=(*p.position,p.heading);rt.start_sensor_episode(0,config=w.config,transform=tr,packets=[p],total_budget=96,return_anchor=anchor,paid_prefix_actions=0)
 _,before=snapshot_metrics(rt.states[0]['mapper'],w,e,reference,(.05,.10));steps=[]
 while not rt.states[0]['closed']:
  action=rt.next_local_action(0)
  choices=[dict(inputs=z['inputs'],outputs=z['outputs'],output_sha256=z['output_sha256']) for z in c._cpu_backend.calls if z['method']=='select_topo_target'];write_json(D/(key+'_choices.json'),choices)
  if action is None:break
  f,col,done=w.step(action);sc=w.scan();p=SensorPacket('Q2_'+arr,'fresh-first-frame',f'frame-{w.step_count}',int(w.step_count),f,sc,tuple(map(int,w.position)),int(w.heading),'deterministic_simulator_rgbd_and_scan','simulator_exact_discrete_odometry',action,bool(col),bool(done)).validate(tr,w.config)
  future|=reference_visible(e.reference,f,e.truth,f.world_from_camera,w.config.max_depth_m);rt.observe(0,w.step_count,None,None,None,sensor_packet=p)
  cov=coverage(rt.states[0]['mapper'],w.reachable);steps.append(dict(step=w.step_count,action=action,position=list(w.position),heading=w.heading,collision=bool(col),done=bool(done),packet_sha256=p.sha256(),coverage_fraction=cov,covered_area_m2=cov*int(w.reachable.sum())*w.config.resolution_m**2,new_visible_surface_area_m2=surface_increment(reference['prefix_seen'],future,reference['weights'])))
  write_json(D/(key+'_steps.json'),steps);beat(key+' step '+str(w.step_count))
 _,after=snapshot_metrics(rt.states[0]['mapper'],w,e,reference,(.05,.10));term=rt.sensor_episode_summary(0);s=c._cpu_backend.scenes[0]
 write_json(D/(key+'_runtime_audit.json'),rt.audit);write_json(D/(key+'_ledger.json'),dict(snapshot=s['ledger'].snapshot(),events=json_value(s['ledger']._events),planning_camera_poses=json_value(s['ledger'].planning_camera_poses())))
 choices=[dict(inputs=z['inputs'],outputs=z['outputs'],output_sha256=z['output_sha256']) for z in c._cpu_backend.calls if z['method']=='select_topo_target'];write_json(D/(key+'_choices.json'),choices)
 assert len(steps)==term['modules']['feedback']['paid_actions']
 row=dict(context='Q2',arrangement=arr,budget=96,condition=condition,native_dispatch_mode=dispatch,status='completed',initial_packet_sha256=initial[arr],anchor=list(anchor),final_pose=[*map(int,w.position),int(w.heading)],steps=len(steps),before_metrics=before,after_metrics=after,coverage_gain_m2=after['covered_area_m2']-before['covered_area_m2'],new_surface_m2=surface_increment(reference['prefix_seen'],future,reference['weights']),termination=term['termination'],collisions=w.collisions,choices_count=len(choices),feedback=term['modules']['observed_gain_calibration'])
 write_json(D/(key+'_result.json'),row);summaries.append(row);write_json(D/'results.json',summaries);print(key,'completed',len(steps),flush=True)
try:
 for arr,condition in order:
  try:run(arr,condition)
  except Exception as error:
   row=dict(context='Q2',arrangement=arr,condition=condition,status='execution_exception',error=repr(error),traceback=traceback.format_exc());write_json(D/('Q2_'+arr+'_96_'+condition+'_exception.json'),row);summaries.append(row);write_json(D/'results.json',summaries);print(arr,condition,'exception retained',repr(error),flush=True)
  gc.collect()
 assert all(sha(R/k)==v for k,v in hashes.items());assert all(sha(Path(k))==v for k,v in policy_hashes.items());beat('completed_all_ten');print('all ten completed',flush=True)
except BaseException as error:
 write_json(D/'batch_failure.json',dict(error=repr(error),traceback=traceback.format_exc()));beat('batch_failed');raise
