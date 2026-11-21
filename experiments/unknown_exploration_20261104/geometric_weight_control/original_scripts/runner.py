import os
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ[k]='1'
import sys,json,hashlib,time,traceback
from pathlib import Path
from types import SimpleNamespace
import numpy as np
R=Path('/home/ouc/nso_robot/analyses/reproduction_20261104/simulation_publication_stage/experiments/unknown_exploration_20261104/source');sys.path.insert(0,str(R))
sys.path.insert(0,'/home/ouc/nso_robot/analyses/independent_layout_20261104')
from two_room_world import TwoRoomWorld
sys.path.insert(0,'/home/ouc/nso_robot/analyses/geometric_weight_control_20261104')
from weight2 import GWeight2
from nso.components import NSO_Components
B=Path('/home/ouc/nso_robot/analyses/reproduction_20261104/simulation_publication_stage/experiments/unknown_exploration_20261104/source');sys.path.insert(1,str(B))
from nso_bridge.exploration_joint_policy import ExplorationJointCPUFourModules, ExplorationJointRuntimeIntegration
from nso.cpu_sensor_contract_v10 import GridTransform,SensorPacket
from scripts.eval_competition_v9_1 import restore_evaluator,snapshot_metrics
from scripts.eval_counterfactual_views import coverage
from scripts.run_cpu_semantic_inspection_v10_2 import write_json
from utils.counterfactual_surface_visibility import reference_visible,surface_increment
from utils.reconstruction_metrics import ReconstructionEvaluator
D=Path(os.environ['NSO_OUTPUT_ROOT']) / 'G_weight2';D.mkdir(exist_ok=False)
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
sources=sorted(p for d in ('env','nso','utils','scripts','configs/virtual3d') for p in (R/d).glob('*.py' if d!='configs/virtual3d' else '*.json')); hashes={str(p.relative_to(R)):sha(p) for p in sources}
protocol=dict(scope='post-hoc G_weight2 control: single TWO_ROOM_01 run, lambda 2 after observed G/S and lambda audit',arrangements=['two_room_shelf_west'],budgets=[96],modes=['G_weight2'],prefix_actions=0,run_order=['two_room_96_G_weight2'],measurement='coverage and truth-visible area each step; mesh F1 first and last only',policy_variant='G_weight2 post-hoc, G common+2*prior_G', physical_adapters_disabled=True, source_probe_receipt_sha256=sha(Path('/home/ouc/nso_robot/analyses/reproduction_20261104/simulation_publication_stage/experiments/unknown_exploration_20261104/evidence/deployed_variant_probe_receipt.json')), policy_sha256=sha(B/'nso_bridge/exploration_joint_policy.py'), range_adapter_sha256=sha(B/'nso_bridge/sensor_range_adapter.py'), original_matrix_protocol_sha256=sha(Path('/home/ouc/nso_robot/analyses/reproduction_20261104/simulation_publication_stage/experiments/unknown_exploration_20261104/evidence/fresh_matrix/protocol.json')), original_matrix_results_sha256=sha(Path('/home/ouc/nso_robot/analyses/reproduction_20261104/simulation_publication_stage/experiments/unknown_exploration_20261104/evidence/fresh_matrix/results.json')),seed='Q0 fixed by original generator',source_sha256=hashes,harness_sha256=sha(Path(__file__)),threads=1)
write_json(D/'protocol.json',protocol)
t0=time.monotonic(); known={}; initial={}; ref={}; summaries=[]
frozen=json.loads((Path('/home/ouc/nso_robot/analyses/independent_layout_20261104/preflight.json')).read_text())
control=json.loads((Path(__file__).resolve().parent/'frozen_protocol.json').read_text())
assert sha(Path(__file__))==control['runner_sha256'] and sha(Path(__file__).resolve().parent/'weight2.py')==control['wrapper_sha256']
assert sha(Path('/home/ouc/nso_robot/analyses/independent_layout_20261104/two_room_world.py'))==frozen['world_sha256']
assert all(sha(R/k)==v for k,v in frozen['source_sha256'].items()), 'frozen source changed'
def beat(status):write_json(D/'heartbeat.json',dict(status=status,elapsed_s=time.monotonic()-t0,pid=os.getpid()))
def run(arr,budget,mode):
 key=f'{arr}_{budget}_G_weight2';beat('starting '+key); w=TwoRoomWorld();tr=GridTransform(tuple(w.shape),w.config.resolution_m);f=w.sense();sc=w.scan();p=SensorPacket('TWO_ROOM_01','fresh-first-frame','frame-0',0,f,sc,tuple(map(int,w.position)),int(w.heading),'deterministic_simulator_rgbd_and_scan','simulator_exact_discrete_odometry',None,False,False).validate(tr,w.config)
 assert p.sha256() == frozen['initial_packet_sha256'] and list((*p.position,p.heading)) == frozen['anchor'], 'initial packet or anchor mismatch'
 if arr in initial:assert initial[arr]==p.sha256()
 else:initial[arr]=p.sha256()
 if arr not in ref:
  count=json.loads((R/'configs/virtual3d/competition_v9_validation_protocol.json').read_text())['reference']['samples_requested'];e=ReconstructionEvaluator(w,count=count,seed=2026);weight=float(w.mesh.get_surface_area())/count; mid=w.config.width_m/2
  ref[arr]=dict(points=e.reference,classes=e.classes,weights=np.full(len(e.reference),weight),prefix_seen=reference_visible(e.reference,f,e.truth,f.world_from_camera,w.config.max_depth_m),west_slot=(e.classes!=1)&(e.reference[:,0]<mid),east_slot=(e.classes!=1)&(e.reference[:,0]>=mid),reachable=w.reachable,vertices=np.asarray(w.mesh.vertices),triangles=np.asarray(w.mesh.triangles))
 reference=ref[arr];e=restore_evaluator(w,reference);future=np.zeros_like(reference['prefix_seen'])
 args=SimpleNamespace(nso_backend='cpu_v10',eval=True,train_global=False,use_open_vocab_semantic=True,use_topo_graph=True,use_rpn_uq=True,use_igcr=True,cpu_score_mode=mode,cpu_disable_feedback=False,cpu_max_candidates=12,cpu_coverage_slots=4,cpu_planner_revision='v10_3_1',cpu_measured_novelty_floor=.25,run_id='fresh-'+mode)
 c=NSO_Components(args);c.initialize('cpu',1,*w.shape,*w.shape);c._cpu_backend=GWeight2(args,1,w.shape);c.capabilities=dict(c._cpu_backend.capabilities);rt=ExplorationJointRuntimeIntegration(c,1,w.shape);anchor=(*p.position,p.heading);rt.start_sensor_episode(0,config=w.config,transform=tr,packets=[p],total_budget=budget,return_anchor=anchor,paid_prefix_actions=0);_,before=snapshot_metrics(rt.states[0]['mapper'],w,e,reference,(.05,.10));old=json.loads(Path('/home/ouc/nso_robot/analyses/independent_layout_20261104/frozen_execution/two_room_joint/two_room_shelf_west_96_G_result.json').read_text());assert before==old['before_metrics'] and initial[arr]==old['initial_packet_sha256'] and list(anchor)==old['anchor'], 'initial equality mismatch';steps=[]
 while not rt.states[0]['closed']:
  action=rt.next_local_action(0)
  choices=[dict(inputs=x['inputs'],outputs=x['outputs']) for x in c._cpu_backend.calls if x['method']=='select_topo_target'];write_json(D/(key+'_choices.json'),choices);write_json(D/(key+'_runtime_audit.json'),rt.audit)
  if action is None:break
  f,col,done=w.step(action);sc=w.scan();p=SensorPacket('TWO_ROOM_01','fresh-first-frame',f'frame-{w.step_count}',int(w.step_count),f,sc,tuple(map(int,w.position)),int(w.heading),'deterministic_simulator_rgbd_and_scan','simulator_exact_discrete_odometry',action,bool(col),bool(done)).validate(tr,w.config);future|=reference_visible(e.reference,f,e.truth,f.world_from_camera,w.config.max_depth_m);rt.observe(0,w.step_count,None,None,None,sensor_packet=p)
  cov=coverage(rt.states[0]['mapper'],w.reachable);steps.append(dict(step=w.step_count,action=action,position=list(w.position),heading=w.heading,collision=bool(col),done=bool(done),packet_sha256=p.sha256(),covered_area_m2=cov*int(w.reachable.sum())*w.config.resolution_m**2,new_visible_surface_area_m2=surface_increment(reference['prefix_seen'],future,reference['weights'])));write_json(D/(key+'_steps.json'),steps);beat(key+' step '+str(w.step_count))
 _,after=snapshot_metrics(rt.states[0]['mapper'],w,e,reference,(.05,.10));term=rt.sensor_episode_summary(0);choices=[dict(inputs=x['inputs'],outputs=x['outputs']) for x in c._cpu_backend.calls if x['method']=='select_topo_target'];write_json(D/(key+'_choices.json'),choices)
 row=dict(arrangement=arr,budget=budget,condition='G_weight2',status='completed',initial_packet_sha256=initial[arr],anchor=anchor,steps=len(steps),before_metrics=before,after_metrics=after,coverage_gain_m2=after['covered_area_m2']-before['covered_area_m2'],new_surface_m2=surface_increment(reference['prefix_seen'],future,reference['weights']),termination=term['termination'],collisions=w.collisions,modules_called=sorted(set(x['module'] for x in c._cpu_backend.calls)),feedback=term['modules']['observed_gain_calibration'],choices_count=len(choices))
 write_json(D/(key+'_result.json'),row)
 summaries.append(row);write_json(D/'partial.json',summaries);print(key,'done',flush=True)
try:
 for mode in ('G',):run('two_room_shelf_west',96,mode)
 assert all(sha(R/k)==v for k,v in hashes.items());assert sha(B/'nso_bridge/exploration_joint_policy.py')==protocol['policy_sha256'];write_json(D/'results.json',summaries);beat('complete')
except Exception as ex:
 write_json(D/'failure.json',dict(error=repr(ex),traceback=traceback.format_exc()));beat('failed');raise
