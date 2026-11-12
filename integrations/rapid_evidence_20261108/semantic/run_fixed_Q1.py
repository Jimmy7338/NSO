import os
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ[k]='1'
import sys,json,hashlib,time,traceback
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
D=Path(__file__).resolve().parent/'execution';D.mkdir(exist_ok=False)
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
sources=sorted(p for d in ('env','nso','utils','scripts','configs/virtual3d') for p in (R/d).glob('*.py' if d!='configs/virtual3d' else '*.json'))
hashes={str(p.relative_to(R)):sha(p) for p in sources}
policy_files=[B/'nso_bridge/exploration_joint_policy.py',B/'nso_bridge/sensor_range_adapter.py']
policy_hashes={str(p):sha(p) for p in policy_files}
protocol=dict(schema='nso.rapid.semantic_Q1.v1',frozen_before_world_construction=True,
    context='Q1',outer_seed=952,arrangement='shelf_west',budget=96,modes=['G','S','X'],run_order=['G','S','X'],
    initial_information='one actual simulated first frame, zero paid prefix',
    policy='unchanged exploration_joint_v1/v10_3_1; no physical range/extent adapters',
    X='existing native swapped class prior score, not new formula',
    no_parameter_search=True,no_outcome_based_retries=True,
    selection_reason='First unused declared parent Q1 after priorQ0 matrix; west is first original arrangement; budget96 fixed before this world was constructed, three native modes once each',
    endpoints=['full anchor return','collision count','coverage curve and final','reference visible surface curve/final','mesh F1 5cm and10cm initial/final','all same-state G/S/X scores and choices'],
    primary_comparison='S versus G and S versus native X; retain mixed/tied/negative results',
    limits=['One synthetic declared parent, not independent natural scenes or autonomous recognition benefit.',
            'NativeX is a counterfactual prior-score comparison; no actual object labels or hidden geometry changed.',
            'A G/S improvement alone can be reproduced by generic weighting and does not isolate category-specific advantage.'],
    source_sha256=hashes,policy_sha256=policy_hashes,harness_sha256=sha(Path(__file__)),threads=1)
write_json(D/'protocol.json',protocol)
t0=time.monotonic();initial_hash=None;reference=None;summaries=[]
def beat(status):write_json(D/'heartbeat.json',dict(status=status,elapsed_s=time.monotonic()-t0,pid=os.getpid()))
def run(mode):
 global initial_hash,reference
 key='Q1_shelf_west_96_'+mode;beat('starting '+key)
 w=create_competition_world('Q1','shelf_west');tr=GridTransform(tuple(w.shape),w.config.resolution_m)
 f=w.sense();sc=w.scan();p=SensorPacket('Q1_shelf_west','fresh-first-frame','frame-0',0,f,sc,tuple(map(int,w.position)),int(w.heading),'deterministic_simulator_rgbd_and_scan','simulator_exact_discrete_odometry',None,False,False).validate(tr,w.config)
 if initial_hash is None:initial_hash=p.sha256()
 else:assert initial_hash==p.sha256(),'initial packets differ'
 if reference is None:
  count=json.loads((R/'configs/virtual3d/competition_v9_validation_protocol.json').read_text())['reference']['samples_requested']
  e=ReconstructionEvaluator(w,count=count,seed=2026);weight=float(w.mesh.get_surface_area())/count;mid=w.config.width_m/2
  reference=dict(points=e.reference,classes=e.classes,weights=np.full(len(e.reference),weight),prefix_seen=reference_visible(e.reference,f,e.truth,f.world_from_camera,w.config.max_depth_m),west_slot=(e.classes!=1)&(e.reference[:,0]<mid),east_slot=(e.classes!=1)&(e.reference[:,0]>=mid),reachable=w.reachable,vertices=np.asarray(w.mesh.vertices),triangles=np.asarray(w.mesh.triangles))
 e=restore_evaluator(w,reference);future=np.zeros_like(reference['prefix_seen'])
 args=SimpleNamespace(nso_backend='cpu_v10',eval=True,train_global=False,use_open_vocab_semantic=True,use_topo_graph=True,use_rpn_uq=True,use_igcr=True,cpu_score_mode=mode,cpu_disable_feedback=False,cpu_max_candidates=12,cpu_coverage_slots=4,cpu_planner_revision='v10_3_1',cpu_measured_novelty_floor=.25,run_id='rapid-Q1-'+mode)
 c=NSO_Components(args);c.initialize('cpu',1,*w.shape,*w.shape);c._cpu_backend=ExplorationJointCPUFourModules(args,1,w.shape);c.capabilities=dict(c._cpu_backend.capabilities)
 rt=ExplorationJointRuntimeIntegration(c,1,w.shape);anchor=(*p.position,p.heading)
 rt.start_sensor_episode(0,config=w.config,transform=tr,packets=[p],total_budget=96,return_anchor=anchor,paid_prefix_actions=0)
 _,before=snapshot_metrics(rt.states[0]['mapper'],w,e,reference,(.05,.10));steps=[]
 while not rt.states[0]['closed']:
  action=rt.next_local_action(0)
  choices=[dict(inputs=x['inputs'],outputs=x['outputs']) for x in c._cpu_backend.calls if x['method']=='select_topo_target']
  write_json(D/(key+'_choices.json'),choices)
  if action is None:break
  f,col,done=w.step(action);sc=w.scan()
  p=SensorPacket('Q1_shelf_west','fresh-first-frame',f'frame-{w.step_count}',int(w.step_count),f,sc,tuple(map(int,w.position)),int(w.heading),'deterministic_simulator_rgbd_and_scan','simulator_exact_discrete_odometry',action,bool(col),bool(done)).validate(tr,w.config)
  future|=reference_visible(e.reference,f,e.truth,f.world_from_camera,w.config.max_depth_m)
  rt.observe(0,w.step_count,None,None,None,sensor_packet=p)
  cov=coverage(rt.states[0]['mapper'],w.reachable)
  steps.append(dict(step=w.step_count,action=action,position=list(w.position),heading=w.heading,collision=bool(col),done=bool(done),packet_sha256=p.sha256(),covered_area_m2=cov*int(w.reachable.sum())*w.config.resolution_m**2,new_visible_surface_area_m2=surface_increment(reference['prefix_seen'],future,reference['weights'])))
  write_json(D/(key+'_steps.json'),steps);beat(key+' step '+str(w.step_count))
 _,after=snapshot_metrics(rt.states[0]['mapper'],w,e,reference,(.05,.10));term=rt.sensor_episode_summary(0)
 choices=[dict(inputs=x['inputs'],outputs=x['outputs']) for x in c._cpu_backend.calls if x['method']=='select_topo_target']
 write_json(D/(key+'_choices.json'),choices);write_json(D/(key+'_runtime_audit.json'),rt.audit)
 row=dict(context='Q1',arrangement='shelf_west',budget=96,condition=mode,status='completed',initial_packet_sha256=initial_hash,anchor=anchor,final_pose=[*map(int,w.position),int(w.heading)],steps=len(steps),before_metrics=before,after_metrics=after,coverage_gain_m2=after['covered_area_m2']-before['covered_area_m2'],new_surface_m2=surface_increment(reference['prefix_seen'],future,reference['weights']),termination=term['termination'],collisions=w.collisions,choices_count=len(choices),feedback=term['modules']['observed_gain_calibration'])
 write_json(D/(key+'_result.json'),row);summaries.append(row);write_json(D/'results.json',summaries);print(key,'completed',flush=True)
try:
 for mode in ['G','S','X']:run(mode)
 assert all(sha(R/k)==v for k,v in hashes.items())
 assert all(sha(Path(k))==v for k,v in policy_hashes.items())
 beat('completed');print('all completed',flush=True)
except Exception as ex:
 write_json(D/'failure.json',dict(error=repr(ex),traceback=traceback.format_exc()));beat('failed');raise
