import os
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ[k]='1'
import sys,json,hashlib,traceback
from pathlib import Path
from types import SimpleNamespace
import numpy as np
R=Path(__file__).resolve().parent / 'source';sys.path.insert(0,str(R))
from env.virtual3d_competition_v9 import create_competition_world
from nso.components import NSO_Components
from nso.runtime_integration import NSORuntimeIntegration
from nso.cpu_sensor_contract_v10 import GridTransform,SensorPacket
from nso.competition_candidates_v8 import measured_assets
from scripts.eval_competition_v9_1 import restore_evaluator
from scripts.eval_counterfactual_views import coverage
from scripts.run_cpu_semantic_inspection_v10_2 import write_json
from utils.counterfactual_surface_visibility import reference_visible,surface_increment
P=Path(__file__).resolve().parent/'evidence/fresh_matrix';D=Path(os.environ['NSO_OUTPUT_ROOT'])/'outbound_counterfactual';D.mkdir(exist_ok=False)
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
sources=sorted(p for d in ('env','nso','utils','scripts','configs/virtual3d') for p in (R/d).glob('*.py' if d!='configs/virtual3d' else '*.json')); hashes={str(p.relative_to(R)):sha(p) for p in sources}
write_json(D/'manifest.json',dict(status='running',source_sha256=hashes,harness_sha256=sha(Path(__file__)),scope='two original selected outbound trajectories from independently reconstructed common 19-step checkpoint; no replan before arrival'))
base=np.load(P.parent/'current_source_pair/prepared/Q0/shelf_west/reference.npz');reference={k:base[k] for k in base.files};rows=[]
try:
 for mode in ('G','S'):
  w=create_competition_world('Q0','shelf_west');tr=GridTransform(tuple(w.shape),w.config.resolution_m);f=w.sense();sc=w.scan();p=SensorPacket('Q0_shelf_west','fresh-first-frame','frame-0',0,f,sc,tuple(map(int,w.position)),int(w.heading),'deterministic_simulator_rgbd_and_scan','simulator_exact_discrete_odometry',None,False,False).validate(tr,w.config)
  oldresult=json.loads((P/f'shelf_west_96_{mode}_result.json').read_text());assert p.sha256()==oldresult['initial_packet_sha256']
  ev=restore_evaluator(w,reference);seen=reference_visible(ev.reference,f,ev.truth,f.world_from_camera,w.config.max_depth_m)
  args=SimpleNamespace(nso_backend='cpu_v10',eval=True,train_global=False,use_open_vocab_semantic=True,use_topo_graph=True,use_rpn_uq=True,use_igcr=True,cpu_score_mode=mode,cpu_disable_feedback=False,cpu_max_candidates=12,cpu_coverage_slots=4,cpu_planner_revision='v10_3_1',cpu_measured_novelty_floor=.25,run_id='fresh-'+mode)
  c=NSO_Components(args);c.initialize('cpu',1,*w.shape,*w.shape);rt=NSORuntimeIntegration(c,1,w.shape);rt.start_sensor_episode(0,config=w.config,transform=tr,packets=[p],total_budget=96,return_anchor=(*p.position,p.heading),paid_prefix_actions=0)
  old=json.loads((P/f'shelf_west_96_{mode}_steps.json').read_text());prior=json.loads((P/f'shelf_west_96_{mode}_choices.json').read_text())[1]['outputs']; checkpoint_seen=None;curve=[]; assets=None
  for step in range(1,44 if mode=='S' else 42):
   act=rt.next_local_action(0)
   if step==20:
    call=[x for x in c._cpu_backend.calls if x['method']=='select_topo_target'][-1]['outputs'];assert call['candidates']==prior['candidates'] and call['scores']==prior['scores'] and call['selected']['candidate_id']==prior['selected']['candidate_id'];
    assets=[{k:v for k,v in a.items() if k!='points'} for a in measured_assets(rt.states[0]['mapper'])];write_json(D/(mode+'_assets.json'),assets)
    checkpoint_seen=seen.copy();cov=coverage(rt.states[0]['mapper'],w.reachable)*int(w.reachable.sum())*w.config.resolution_m**2;curve.append(dict(outbound_step=0,paid_step=19,covered_area_m2=cov,new_surface_since_checkpoint_m2=0.0));outbound=prior['selected']['outbound_actions'];assert act==outbound[0]
   if step>=20:assert act==outbound[step-20],f'unexpected action or replan at step {step}'
   assert act==old[step-1]['action'];f,col,done=w.step(act);sc=w.scan();p=SensorPacket('Q0_shelf_west','fresh-first-frame',f'frame-{w.step_count}',int(w.step_count),f,sc,tuple(map(int,w.position)),int(w.heading),'deterministic_simulator_rgbd_and_scan','simulator_exact_discrete_odometry',act,bool(col),bool(done)).validate(tr,w.config);assert p.sha256()==old[step-1]['packet_sha256'];seen|=reference_visible(ev.reference,f,ev.truth,f.world_from_camera,w.config.max_depth_m);rt.observe(0,w.step_count,None,None,None,sensor_packet=p)
   if step>=20:
    cov=coverage(rt.states[0]['mapper'],w.reachable)*int(w.reachable.sum())*w.config.resolution_m**2;curve.append(dict(outbound_step=step-19,paid_step=step,covered_area_m2=cov,new_surface_since_checkpoint_m2=surface_increment(checkpoint_seen,seen,reference['weights'])));write_json(D/(mode+'_curve.json'),curve)
  assert [*map(int,w.position),int(w.heading)]==prior['selected']['pose']; choices=[x for x in c._cpu_backend.calls if x['method']=='select_topo_target'];assert len(choices)==2
  rows.append(dict(mode=mode,candidate_id=prior['selected']['candidate_id'],pose=prior['selected']['pose'],outbound_steps=len(outbound),verified_packets=step,curve=curve,assets=assets,predicted_score_audit=next(x for x in prior['score_audit'] if x['candidate_id']==prior['selected']['candidate_id']),collision_count=w.collisions));write_json(D/'partial.json',rows)
 assert all(sha(R/k)==v for k,v in hashes.items());write_json(D/'results.json',rows);m=json.loads((D/'manifest.json').read_text());m['status']='complete';write_json(D/'manifest.json',m)
except Exception as e:
 write_json(D/'failure.json',dict(error=repr(e),traceback=traceback.format_exc()));raise
