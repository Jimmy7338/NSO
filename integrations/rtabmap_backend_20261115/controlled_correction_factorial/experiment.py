"""Isolated actual simulated V35 correction factorial. No robot dependencies."""
import os
for name in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):os.environ[name]='1'
import argparse,hashlib,json,sys,time,traceback
from pathlib import Path
HERE=Path(__file__).resolve().parent
ROOT=Path('/home/ouc/nso_robot/workspaces/nso_research_20261029/NSO')
sys.path.insert(0,str(ROOT));sys.path.insert(0,str(HERE))
PROTOCOL=HERE/'protocol.v2.json'
RUNS=HERE/'runs_v2'
EQUIVALENCE=HERE/'equivalence.v2.json'
SCENE=ROOT/'configs/virtual3d/v33_direction_scene_r1_20260917.json'
CASES=[dict(id=f'h{h}_observed{o}_forecast{f}',hypothesis=h,observed_correction=bool(o),forecast_diagnostic=bool(f))for h in (0,1)for o,f in ((0,0),(0,1),(1,0),(1,1))]
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def save(p,v):Path(p).write_text(json.dumps(v,indent=2,ensure_ascii=False,allow_nan=False)+'\n')
def bindings():
 paths=[SCENE,*HERE.glob('*.py')]
 for name in ('nso','env','utils'):paths.extend((ROOT/name).glob('*.py'))
 paths+=[ROOT/'scripts/run_online_routes_v35.py',ROOT/'scripts/preflight_pixel_information_v34.py']
 return {str(p):sha(p)for p in sorted(set(paths))}
def check():
 d=json.loads((PROTOCOL).read_text())
 for p,h in d['source_sha256'].items():
  if sha(p)!=h:raise ValueError('Frozen source changed: '+p)
 return d
def freeze():
 if(PROTOCOL).exists():raise FileExistsError('Protocol already frozen; never overwrite')
 save(PROTOCOL,dict(schema='nso.V35.correction_factorial.v2',frozen_unix_s=time.time(),status='frozen_before_new_simulation; awaiting_root_review',independent_layout_units=1,parent='P00',configurations=[0,1],cases=CASES,wrong_prior='Original swapped category interpretation for every cell',budget=42,common_prefix=18,noise_model='iid_025px',noise_seed=1901,sensing='Original InformationPixelWorldV34; exact simulated poses',fusion='Original ObservedRuntimeMapperV10 voxel.04/trunc.12',evaluation='Original extract_observed_asset_mesh and SurfaceMeasurementV34; thresholds2/5/10cm,5cmprimary; reference accessed only after terminal predictions/actions sealed',original_controls=['bothON original swapped(X)','bothOFF original swapped_no_feedback(Xnf)'],new_routes=8,endpoint_pairing='All eight main trajectories use the same regenerated frozen public cache; old published endpoints remain contextual references only and are never merged into this factorial. Equivalence trajectories are implementation checks, not efficacy replicates.',supersedes_protocol_sha256=sha(HERE/'protocol.json'),qualification='Keep all failures, collisions, no-return, early ends and ties; never replace failed slots; do not pool2configurations as independent layouts',mechanism_scope='Only separates actual measured correction from forecast diagnostic opportunity; no new optimizer/scorer/candidate/return rule',public_assets='Historical P00 geometry/pixel public cache absent locally; public-cache command regenerates original public table and clean template sensor arrays from frozen original source, not selected hidden outcome. Equivalence command compares original and adapter on identical actual simulated observation objects, with every action/posterior checked.',source_sha256=bindings(),actual_batch_started=False))
 print(PROTOCOL)
def assets():
 check();import numpy as np
 from nso.direction_information_v33 import DirectionGeometryV33
 from env.information_pixel_v34 import InformationPixelWorldV34
 from scripts.preflight_pixel_information_v34 import raw_fields
 d=HERE/'public_cache';d.mkdir(exist_ok=False);parent=next(p for p in json.loads(SCENE.read_text())['parents']if p['id']=='P00')
 tables=[DirectionGeometryV33(parent,h).tables()for h in (0,1)];g=d/'audit_results/v33_direction_information_r1_20260917';g.mkdir(parents=True);save(g/'P00_geometry.json',dict(tables=tables))
 for h,table in enumerate(tables):
  w=InformationPixelWorldV34('P00',h,'factorial-public-'+str(h),noise_model='clean');columns={}
  for n,pose in enumerate(table['poses']):
   p=w.packet_at(tuple(pose),step=n,noise_model='clean')
   for k,v in raw_fields(p).items():
    if k!='rgb':columns.setdefault(k,[]).append(np.array(v,copy=True))
  np.savez_compressed(d/f'P00_h{h}_pixels.npz',**{k:np.stack(v)for k,v in columns.items()})
 save(d/'manifest.json',dict(protocol_sha256=sha(PROTOCOL),kind='regenerated original public templates, no policy trial',files={str(p.relative_to(d)):sha(p)for p in d.rglob('*')if p.is_file()}))
 print(d/'manifest.json')
def setup(h,observed,forecast,equivalence=False):
 from nso.online_planner_v35 import load_public_models_v35
 from nso.observation_belief_v35 import PublicTemplatesV35
 from nso.cpu_four_modules_v35 import CPUFourModuleControllerV35
 from factorial_controller import FactorialController
 from env.information_pixel_v34 import InformationPixelWorldV34
 from nso.observed_runtime_mapper_v10 import ObservedRuntimeMapperV10
 d=HERE/'public_cache'
 manifest=json.loads((d/'manifest.json').read_text())
 for rel,expected in manifest['files'].items():
  if sha(d/rel)!=expected:raise ValueError('Frozen public cache changed: '+rel)
 if manifest.get('protocol_sha256')!=sha(PROTOCOL):raise ValueError('Public cache was generated under a different protocol version')
 models,prefix=load_public_models_v35(d,'P00');templates=PublicTemplatesV35.from_saved(d,'P00',models[0].poses)
 c=FactorialController(models,templates,prefix,observed_correction=observed,forecast_diagnostic=forecast,total_budget=42)
 base=CPUFourModuleControllerV35(models,templates,prefix,mode='swapped'if observed else'swapped_no_feedback',total_budget=42)if equivalence else None
 w=InformationPixelWorldV34('P00',h,'factorial-'+str(h),noise_model='iid_025px');assert w.noise_seed==1901
 return models,c,base,w,ObservedRuntimeMapperV10(w.shape,w.config,truncation_m=.12)
def classify_end(paid, following, returned, collisions, terminal_reason):
 if collisions:return dict(classification='collision',failed=True)
 if following is not None:return dict(classification='budget_exhausted_with_pending_continuation' if paid>=42 else 'unexpected_unfinished_loop',failed=True)
 if not returned:return dict(classification='budget_end_without_return' if paid>=42 else 'stopped_without_return',failed=True)
 if terminal_reason not in ('sensor_or_budget_end_returned','returned_no_affordable_action'):
  return dict(classification='unrecognized_terminal_reason',failed=True)
 return dict(classification='returned_terminal_stop',failed=False)

def trajectory(h,observed,forecast,folder=None,equivalence=False):
 import numpy as np
 from nso.cpu_four_modules_v35 import ObservationV35
 from nso.decision_replay_v13 import array_hash
 from scripts.run_online_routes_v35 import observed_pose
 from nso.surface_measurement_v34 import extract_observed_asset_mesh,SurfaceMeasurementV34
 models,c,base,w,mapper=setup(h,observed,forecast,equivalence);trace=[];history=[];action=None;collisions=0;snapshots={};started=time.monotonic()
 for paid in range(43):
  packet=w.packet()if paid==0 else w.step(action);mapper.update(packet.frame,packet.scan);collisions+=int(packet.collision);pose=observed_pose(packet,w.shift)
  obs=ObservationV35(frame_id=f'opaque-{paid}',step=paid,pose=pose,action=action,depth=packet.frame.depth_m,rgb=packet.frame.color_rgb,ranges=packet.scan.ranges_m,collision=packet.collision,measured_map_sha256=array_hash(mapper.belief))
  c.accept(obs);following=c.next_action()
  if base is not None:
   # Exact same acquired packet arrays, pose and measured map digest are reused in RAM.
   base.accept(obs);reference_action=base.next_action()
   assert following==reference_action,(h,observed,paid,'action mismatch')
   assert c.posterior_receipts[-1]==base.posterior_receipts[-1],(h,observed,paid,'posterior mismatch')
   assert c.plans==base.plans,(h,observed,paid,'planner trace mismatch')
  trace.append(dict(paid=paid,issued_action=action,next_action=following,pose=list(pose),packet_sha256=packet.sha256(),collision=packet.collision))
  history.append(c.posterior_receipts[-1])
  if folder is not None:
   with (folder/'partial_trace.jsonl').open('a') as stream:stream.write(json.dumps(trace[-1])+'\n')
  if not equivalence and(paid==18 or following is None or paid==42):
   bounds=np.asarray([box for hyp in w.parent['hypotheses']for asset in hyp['assets']for box in asset['boxes']]).reshape(-1,3,2)
   public_roi=np.stack((bounds[:,:,0].min(axis=0),bounds[:,:,1].max(axis=0)))+w.shift
   mesh,crop=extract_observed_asset_mesh(mapper.mesh(),public_roi)
   snapshot=(mesh,crop,np.array(mapper.belief,copy=True),paid)
   if paid==18:snapshots['prefix']=snapshot
   if following is None or paid==42:snapshots['final']=snapshot
  if following is None:break
  action=following
 returned=tuple(trace[-1]['pose'])==models[0].poses[models[0].anchor]
 end=classify_end(paid,following,returned,collisions,c.terminal_reason)
 result=dict(end_classification=end['classification'],failed=end['failed'],hypothesis=h,observed_correction=observed,forecast_diagnostic=forecast,paid_actions=paid,observations=len(trace),collisions=collisions,returned=returned,terminal_reason=c.terminal_reason,elapsed_s=time.monotonic()-started,trajectory_sha256=hashlib.sha256(json.dumps(trace,sort_keys=True).encode()).hexdigest())
 if not equivalence and 'final' not in snapshots:raise AssertionError('A terminal snapshot must exist even at budget exhaustion')
 if equivalence:return dict(**result,equivalence_passed=True,comparison='original X/Xnf versus adapter, exact same actual simulated observations/action+posterior+plan receipt at every step',independent_efficacy_run=False)
 save(folder/'trace.json',trace);save(folder/'beliefs.json',history);save(folder/'prediction_seal.json',dict(before_reference_access=True,trace_sha256=sha(folder/'trace.json'),source_protocol_sha256=sha(PROTOCOL),public_cache_manifest_sha256=sha(HERE/'public_cache/manifest.json'),prediction_array_sha256={name:dict(vertices=array_hash(np.asarray(mesh.vertices)),triangles=array_hash(np.asarray(mesh.triangles)),belief=array_hash(belief))for name,(mesh,crop,belief,n)in snapshots.items()}))
 # Independent reference access only after acquired predictions and decisions are sealed.
 floor=w.evaluation_floor();evaluator=SurfaceMeasurementV34(w.instance_mesh(0),w.instance_mesh(0,vertical_only=True));stages={}
 for name,(mesh,crop,belief,n)in snapshots.items():
  coverage=float(np.mean(belief[floor['reachable']]!=-1));measurement=evaluator.evaluate(mesh,coverage,returned=tuple(trace[n]['pose'])==models[0].poses[models[0].anchor],collisions=sum(t['collision']for t in trace[:n+1]),failed=bool(end['failed'] if name=='final' else True),paid_actions=n,budget=42)
  stages[name]=dict(measurement=measurement,crop=crop,stage_role='actual_terminal'if name=='final'else'nonterminal_prefix_checkpoint_not_qualified',end_classification=end['classification']if name=='final'else'nonterminal_checkpoint')
 result['stages']=stages;save(folder/'result.json',result);return result
def main():
 ap=argparse.ArgumentParser();ap.add_argument('command',choices=['freeze','public-cache','equivalence','run-case']);ap.add_argument('--case',choices=[c['id']for c in CASES]);args=ap.parse_args()
 if args.command=='freeze':freeze();return
 check()
 if args.command=='public-cache':assets();return
 if args.command=='equivalence':
  if(EQUIVALENCE).exists():raise FileExistsError('Equivalence receipt exists')
  result=[trajectory(h,b,b,equivalence=True)for h in (0,1)for b in (True,False)];save(EQUIVALENCE,dict(all_passed=True,cases=result,source_protocol_sha256=sha(PROTOCOL)));print('equivalence PASS');return
 eq=json.loads((EQUIVALENCE).read_text());assert eq['all_passed']and eq['source_protocol_sha256']==sha(PROTOCOL)
 if args.case is None:ap.error('--case required')
 case=next(c for c in CASES if c['id']==args.case);folder=RUNS/case['id'];folder.mkdir(parents=True,exist_ok=False)
 save(folder/'started.json',dict(case=case,unix_s=time.time(),protocol_sha256=sha(PROTOCOL)))
 try:print(json.dumps(trajectory(case['hypothesis'],case['observed_correction'],case['forecast_diagnostic'],folder=folder)))
 except BaseException as e:save(folder/'failure.json',dict(error=repr(e),traceback=traceback.format_exc()));raise
if __name__=='__main__':main()
