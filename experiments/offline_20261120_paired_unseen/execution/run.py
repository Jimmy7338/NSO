"""B11 prepared worlds and inherited complete unknown-exploration runner."""
from pathlib import Path
import argparse,dataclasses,json,time,sys
import numpy as np
import runtime_binding as R
HERE=Path(__file__).resolve().parent;ROOT=HERE.parent

def frozen_paths():
 paths=[HERE/n for n in ('run.py','runtime_binding.py','full_posterior_forecast.py','short_branches.py','protocol.json','test_full_posterior_forecast.py','test_cost_selection.py','supervisor.py')]
 paths += [ROOT/'PREFIX_PREPARATION_V2.json',ROOT/'preparation_v2/prefix_override.json',ROOT/'preparation_v2/selector_freeze.json',ROOT/'performance/observed_prefix_preflight.json']
 paths += sorted((ROOT/'preparation_v2').glob('*.py'))
 paths += sorted((ROOT/'preparation_v2').glob('*/*.json'))
 paths += [ROOT/'STUDY_SCOPE.json',ROOT/'USER_PROTOCOL.txt',ROOT/'scene/freeze.json',ROOT/'performance/fork_forecast.py',ROOT/'performance/audit/independent_fork_review.json',ROOT/'performance/fallback_validation.json',HERE/'fork_existing_prefix_validation.json']
 scene=json.loads((ROOT/'scene/freeze.json').read_text())
 for key in ('files_sha256','source_sha256','frozen_files_sha256','code_and_data_sha256'):
  if isinstance(scene.get(key),dict):paths += [Path(p)for p in scene[key]]
 if (ROOT/'evaluation/freeze.json').exists():
  ef=json.loads((ROOT/'evaluation/freeze.json').read_text());paths.append(ROOT/'evaluation/freeze.json')
  for key in ('files_sha256','code_sha256'):
   if isinstance(ef.get(key),dict):paths += [Path(p)for p in ef[key]]
 paths += [R.B/'run.py',R.B/'method/adapter.py',R.B9/'structure/replay.py',R.B9/'structure/model.py',R.B9/'structure/audit/calibration_attempt_001/selected_temperature.json',R.B8/'audit/calibration_attempt_001/selected_scale.json']
 inherited=json.loads((R.B8/'calibration/inference_freeze.json').read_text());paths += [Path(p)for p in inherited['files_sha256']]
 b10=R.BASE/'dual_template_gain_alignment_20261120'
 paths += [b10/'geometry_audit/forecast_development/acquisition_forecast.py',b10/'execution_audit/acquisition_forecast/render_observed_free.py']
 return sorted(set(paths))

def freeze():
 assert not (HERE/'freeze.json').exists()
 assert R.protocol()['status']=='frozen_before_B11_predictions_and_policy_runs'
 R.write(HERE/'freeze.json',dict(schema='B11_execution_freeze_v1',frozen_unix_s=time.time(),files_sha256={str(p):R.sha(p)for p in frozen_paths()},new_scene_predictions_before_freeze=0,complete_policy_runs_before_freeze=0,staging='first chain_a/cycle_a12 branches table, then chain_b/cycle_b12 branches, then8full'))

def verify(layout=None):
 from short_branches import verify as check
 check()
 if layout is None:return
 path=prepared(layout);f=json.loads((path/'freeze.json').read_text())
 assert R.sha(HERE/'freeze.json')==f['execution_freeze_sha256']
 for name,h in f['prepared_files_sha256'].items():assert R.sha(path/name)==h,name
 return json.loads((path/'initial_state.json').read_text())

def prepared(layout):return HERE/'layouts'/layout/'prepared'
def dependencies():
 old=R.install();old.HERE=HERE;old.prepared=prepared;old.verify=verify;return old

def prepare(layout):
 verify();old=dependencies();out=prepared(layout);out.mkdir(parents=True,exist_ok=False)
 from nso.observed_runtime_mapper_v10 import ObservedRuntimeMapperV10
 world,tr,p=R.world_and_packet(layout);mapper=ObservedRuntimeMapperV10(world.shape,world.config);mapper.update(p.frame,p.scan)
 first=dict(layout=layout,scene_id=R.protocol()['layouts'][layout]['scene_id'],scene_role='new_paired_four_layout_two_topology_unknown_exploration',shape=list(world.shape),anchor=[*world.position,world.heading],actual_config=dataclasses.asdict(world.config),initial_packet_sha256=p.sha256(),initial_nonsemantic_packet_sha256=old.base.nonsemantic_packet_sha(p),initial_owner_sha256=__import__('hashlib').sha256(old.front.owner(p.frame).tobytes()).hexdigest(),evaluation=old.base.evaluate(mapper.belief,world.reachable),current_footprint_conflict=bool(mapper.current_footprint_conflict),partial_ideal_semantics=True,natural_detector=False,geometry_receipt=world.geometry_receipt,simulator_actions=0)
 R.write(out/'initial_state.json',first);p.frame.save(out/'initial_frame.npz');p.scan.save(out/'initial_scan.npz')
 R.write(out/'freeze.json',dict(execution_freeze_sha256=R.sha(HERE/'freeze.json'),prepared_files_sha256={p.name:R.sha(p)for p in out.iterdir()if p.is_file()},created_unix_s=time.time()))
 print(json.dumps(dict(status='prepared',layout=layout,simulator_actions=0)),flush=True)

def selfcheck(layout):
 first=verify(layout);old=dependencies();rows={}
 from nso.cpu_sensor_contract_v10 import digest
 shared=R.world_and_packet(layout)
 # Zero-action shared acquisition: inspect native candidate/common state without
 # paying for repeated full C scoring or treating it as a method evaluation.
 for arm in R.ARMS:
  world,tr,p,comp,runtime,effective,owner=old.bootstrap(layout,arm,shared)
  state=comp._cpu_backend.scenes[0];context=dict(state=state,backend=comp._cpu_backend)
  routes,common,audit=R.native_candidates(context)
  cache=old.method._cache(state['mapper'],arm);g,s=cache['models']['G'],cache['models']['S']
  assert p.sha256()==first['initial_packet_sha256'];assert world.step_count==0
  rows[arm]=dict(candidate_sha256=digest(routes),common_sha256=digest(common),initial_packet_sha256=p.sha256(),owner_sha256=owner['owner_sha256'],belief_sha256=digest(state['mapper'].belief),points_sha256=digest(g.points),G_base_sha256=digest(g.base_weights),S_base_sha256=digest(s.base_weights),history_keep_sha256=digest(g.history_keep),candidate_count=len(routes),points=len(g.points),simulator_actions=0,full_posterior=True)
 assert rows['S_visibility']==rows['S_acquisition']
 R.write(HERE/'layouts'/layout/'selfcheck.json',dict(status='passed_zero_action_binding_and_candidates',arms=rows,new_gain_scored=False))
 print(json.dumps(dict(status='passed',layout=layout,simulator_actions=0,initial_candidate_count=len(routes))),flush=True)

def main():
 p=argparse.ArgumentParser();p.add_argument('command',choices=('freeze','prepare','selfcheck','run','verify'));p.add_argument('--layout');p.add_argument('--arm',choices=tuple(R.ARMS));a=p.parse_args()
 if a.command=='freeze':freeze();return
 if not a.layout:p.error('layout required')
 if a.command=='prepare':prepare(a.layout)
 elif a.command=='selfcheck':selfcheck(a.layout)
 elif a.command=='verify':verify(a.layout)
 else:
  if a.arm is None:p.error('run requires arm')
  old=dependencies();parent=HERE/'layouts'/a.layout/'runs'/a.arm
  before=set(parent.glob('attempt_*')) if parent.exists() else set()
  old.run(a.layout,a.arm)
  new=set(parent.glob('attempt_*'))-before;assert len(new)==1
  output=new.pop();R.write(output/'B11_execution_binding.json',dict(wrapper=R.resource(Path(__file__)),execution_freeze=R.resource(HERE/'freeze.json'),predictor=R.ARMS[a.arm]['predictor'],full_posterior_same_quadrature=True,short_cost_band_used=False,geometry_from_acquired_frames_only=True))
  R.write(output/'artifact_manifest.json',dict(status='completed_execution',files_sha256={p.name:R.sha(p)for p in output.iterdir()if p.is_file()and p.name!='artifact_manifest.json'}))
if __name__=='__main__':main()
