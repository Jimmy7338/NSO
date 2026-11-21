#!/usr/bin/env python3
"""B8 five-arm predictions on six sealed prefixes and 66 reused branches.

Reconstruct only already acquired prefixes. No candidate branch or new policy
is executed. Independent evaluation alone opens source surface truth.
"""
import argparse,hashlib,importlib.util,json,os,sys,time,traceback
from pathlib import Path
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ[key]='1'
HERE=Path(__file__).resolve().parent;STUDY=HERE.parent;BASE=STUDY.parent
V1=BASE/'dual_template_geometry_20261120';B5=BASE/'dual_template_objectfit_20261120';B6=BASE/'dual_template_parts_20261120'
STATES={'shelf_left':(24,54,109),'shelf_right':(24,53,110)}
ARMS={
 'G_fixed':dict(semantic=False,extent_mode='fixed',group='G'),
 'S_fixed':dict(semantic=True,extent_mode='fixed',group='S'),
 'G_uncertainty':dict(semantic=False,extent_mode='uncertainty',group='G'),
 'S_uncertainty':dict(semantic=True,extent_mode='uncertainty',group='S'),
 'S_mean_collapse':dict(semantic=True,extent_mode='mean_collapse',group='S')}
SCALE=STUDY/'audit/calibration_attempt_001/selected_scale.json'
SCALE_SHA='c92367d5c17719b42e235d42af3123c28c3970ccdcb97bb79fd0a16715beeeb1'

def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def resource(path):return dict(path=str(Path(path).resolve()),sha256=sha(path))
def write(path,value):
 path=Path(path);temporary=path.with_suffix(path.suffix+'.tmp');temporary.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n');temporary.replace(path)
def load(name,path):
 spec=importlib.util.spec_from_file_location(name,path);module=importlib.util.module_from_spec(spec);sys.modules[name]=module;spec.loader.exec_module(module);return module

def declared_arms():
 rows=json.loads((STUDY/'policy_design.json').read_text())['arms']
 assert set(rows)==set(ARMS)
 for arm,flags in ARMS.items():
  assert rows[arm]['extent_mode']==flags['extent_mode']
  assert (rows[arm]['class_prior']=='acquired_category')==flags['semantic']
 return rows

def frozen_paths():
 prior=json.loads((B6/'replay/freeze.json').read_text())
 paths=[Path(p) for p in prior['files_sha256']]
 paths += [Path(__file__),HERE/'protocol.json',HERE/'test_wrapper.py',HERE/'tests.log',HERE/'run_batch.py',STUDY/'policy_design.json',B6/'replay/run_parts_predictions.py',B6/'replay/freeze.json',SCALE,
  STUDY/'audit/summary.json',STUDY/'audit/counterfactual_check.json',STUDY/'calibration/inference_validation.json']
 paths += [STUDY/'method'/name for name in ('model.py','infer.py','test_method.py','tests.log','METHOD_SPEC.md','validation.json')]
 paths += [STUDY/'audit'/name for name in ('evaluate_particle_replay.py','test_particle_replay_evaluator.py','particle_replay_validation.json','particle_replay_tests.log')]
 for layout,states in STATES.items():
  for paid in states:
   root=B6/'replay'/f'{layout}_{paid}_attempt_001'
   paths += [root/name for name in ('result.json','manifest.json','outer_and_layers_predictions.json','outer_and_layers_predictions.npz')]
 return sorted(set(paths))

def freeze():
 assert not (HERE/'freeze.json').exists(),'Never overwrite a prediction freeze'
 assert sha(SCALE)==SCALE_SHA
 scale=json.loads(SCALE.read_text());assert scale['sigma_m']==.01 and scale['opened_truth_splits']==['train']
 declared_arms();protocol=json.loads((HERE/'protocol.json').read_text())
 assert protocol['status']=='frozen_before_five_arm_prediction_replay' and protocol['states']=={k:list(v) for k,v in STATES.items()}
 validation=json.loads((STUDY/'method/validation.json').read_text())
 assert validation['status']=='frozen_before_independent_geometry_calibration'
 for name,expected in validation['files'].items():assert sha(STUDY/'method'/name)==expected
 old=load('B8_verify_B6',B6/'replay/run_parts_predictions.py');old.verify()
 write(HERE/'freeze.json',dict(schema='B8_fixed_prefix_five_arm_freeze_v1',created_unix_s=time.time(),
  files_sha256={str(p):sha(p) for p in frozen_paths()},states=STATES,arms=ARMS,selected_scale=resource(SCALE),
  sigma_m=scale['sigma_m'],reused_candidate_branches=66,reused_candidate_actions=1569,new_candidate_actions=0,hardware=False))
 return HERE/'freeze.json'

def verify():
 frozen=json.loads((HERE/'freeze.json').read_text());assert {str(p):sha(p) for p in frozen_paths()}==frozen['files_sha256']
 return frozen

def dependencies():
 old=load('B8_sealed_prefix_runner',V1/'replay/run_staged_branches.py');old.verify()
 b,repair,_=old.dependencies()
 parts=load('B8_reuse_B6_helpers',B6/'replay/run_parts_predictions.py');parts.verify()
 model=load('B8_frozen_particle_model',STUDY/'method/model.py')
 return old,b,repair,parts,model

def snapshot(mapper,model):
 import numpy as np
 from nso.cpu_sensor_contract_v10 import digest
 values=list(mapper.quality.values());quality={}
 if values:
  for key in values[0]:quality[key]=hashlib.sha256(np.ascontiguousarray([row[key] for row in values]).tobytes()).hexdigest()
 return dict(frames=int(mapper.frames),belief=digest(mapper.belief),quality=quality,
  observation_signature=model.observed_input_signature(mapper))

def capture_models(model_module,context,out,sigma):
 import numpy as np
 from nso.cpu_sensor_contract_v10 import json_value
 from utils.reconstruction_metrics import ray_scene
 mapper,routes=context['mapper'],context['routes'];before=snapshot(mapper,model_module)
 mesh=mapper.mesh();measured_ray=ray_scene(mesh) if len(mesh.triangles) else None
 started=time.monotonic();bank=model_module.build_particle_bank(mapper)
 bank_time=time.monotonic()-started
 models={arm:model_module.FullTemplateModel(mapper,semantic=flags['semantic'],sigma_m=sigma,
  extent_mode=flags['extent_mode'],bank=bank) for arm,flags in ARMS.items()}
 parity={}
 for mode in ('fixed','uncertainty'):
  g,s=models['G_'+mode],models['S_'+mode]
  np.testing.assert_array_equal(g.points,s.points);np.testing.assert_array_equal(g.hypothesis_ids,s.hypothesis_ids)
  np.testing.assert_array_equal(g.object_ids,s.object_ids);np.testing.assert_array_equal(g.particle_ids,s.particle_ids)
  assert g.bank is s.bank is bank
  parity[mode]=dict(same_points=True,same_hypothesis_ids=True,same_object_particle_ids=True,
   shared_bank=True,points_sha256=model_module.digest(g.points),points=len(g.points),marginal_depth_weights_may_differ=True)
 write(out/'common_bank_audit.json',json_value(dict(schema='B8_common_bank_state_audit_v1',bank_build_seconds=bank_time,
  observed_input_sha256=bank.observed_input_sha256,mode_basis_parity=parity,
  neutral_distribution=bank.distribution(sigma_m=sigma,semantic=False),new_sensor_data_acquired=False)))
 captures={}
 for arm,flags in ARMS.items():
  start=time.monotonic();model=models[arm];name=flags['group'];cache={};rows=[];unions=[]
  support_lookup=np.asarray([int(o['support_track']) for o in model.objects],np.int64)
  support=support_lookup[model.object_ids] if len(model.object_ids) else np.empty(0,np.int64)
  arrays={name+'_points':model.points,name+'_weights':model.weights,name+'_base_weights':model.base_weights,
   name+'_history_keep':model.history_keep,name+'_hypothesis_ids':model.hypothesis_ids,
   name+'_object_ids':model.object_ids,name+'_particle_ids':model.particle_ids,
   name+'_shape_ids':model.hypothesis_ids%2,name+'_support_tracks':support}
  for route,common in zip(routes,context['common_rows']):
   cid=int(route['candidate_id']);union=np.zeros(len(model.points))
   for raw in route['outbound_states'][1:]:
    state=tuple(map(int,raw))
    if state not in cache:cache[state]=model.gain(state[:2],state[2],measured_ray)
    np.maximum(union,cache[state],out=union)
   area=float(union.sum());assert np.isfinite(area) and 0<=area<=model.weights.sum()+1e-8
   arrays[name+f'_gain_{cid}']=union;unions.append(union)
   rows.append(dict(candidate_id=cid,template_area={name:area},common_total_proxy=common['common_total_proxy'],
    coverage_only_proxy=common['coverage_only_proxy'],scores={name:common['common_total_proxy']+area},
    outbound_actions=len(route['outbound_actions']),return_actions_excluded=len(route['return_actions'])))
  arrays['candidate_ids']=np.asarray([r['candidate_id'] for r in rows],np.int64)
  arrays[name+'_route_union_gain_matrix']=np.stack(unions) if unions else np.zeros((0,len(model.points)))
  path=out/(arm+'_predictions.npz');np.savez_compressed(path,**arrays)
  receipt=dict(variant=arm,flags=flags,sigma_m=sigma,selected_scale=resource(SCALE),model_audits={name:model.audit},routes=rows,
   common_bank_audit=resource(out/'common_bank_audit.json'),mode_basis_parity=parity,
   mean_collapse_basis_separate=flags['extent_mode']=='mean_collapse',source_geometry_loaded_by_model=False,
   candidate_acquisition_reused=True,elapsed_s=time.monotonic()-start,
   occlusion_repair_summary=getattr(model,'_occlusion_repair_summary',{}),
   matrix=dict(candidate_ids_key='candidate_ids',gain_matrix_key=name+'_route_union_gain_matrix',
    named_gain_vectors_preserved_for_evaluator=True),points=count_or_zero(model.points))
  write(out/(arm+'_predictions.json'),json_value(receipt))
  captures[arm]=dict(predictions=resource(path),receipt=resource(out/(arm+'_predictions.json')))
  print(json.dumps(dict(arm=arm,candidates=len(rows),points=len(model.points),seconds=receipt['elapsed_s'])),flush=True)
 after=snapshot(mapper,model_module);assert before==after
 write(out/'observed_mapper_unchanged.json',dict(before=before,after=after,unchanged=True,
  shared_bank_cache_is_not_new_observation=True,new_candidate_actions=0))
 return captures

def count_or_zero(points):return int(len(points))

def main():
 parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--freeze',action='store_true')
 parser.add_argument('--layout',choices=tuple(STATES));parser.add_argument('--state',type=int);parser.add_argument('--output',type=Path)
 args=parser.parse_args()
 if args.freeze:print(freeze());return
 if args.layout is None or args.state not in STATES[args.layout] or args.output is None:parser.error('Supply a declared layout/state and fresh output')
 frozen=verify();out=args.output;out.mkdir(parents=True,exist_ok=False);started=time.monotonic()
 try:
  old,b,repair,parts,model=dependencies();context=old.sealed_prefix(b,args.layout,args.state)
  previous=V1/'replay'/f'{args.layout}_{args.state}_attempt_001'
  assert context['prefix_receipt']==json.loads((previous/'prefix_execution.json').read_text())
  assert context['routes']==json.loads((previous/'fixed_candidates.json').read_text())
  check=out/'prefix_reconstructed_check.ply';rebuilt=old.save_mesh(context['mapper'],check)
  original=resource(previous/'prefix_native_tsdf_mesh.ply');assert rebuilt['sha256']==original['sha256'];check.unlink()
  write(out/'prefix_execution.json',dict(**context['prefix_receipt'],native_mesh_rebuilt_sha256=rebuilt['sha256'],
   previous_native_mesh=original,native_prefix_byte_identical_to_v1=True,newly_executed_candidate_actions=0))
  write(out/'fixed_candidates.json',context['routes']);observed=parts.save_observed_support(context['mapper'],out)
  import numpy as np
  with np.load(out/'observed_support.npz',allow_pickle=False) as current,np.load(B5/'replay'/f'{args.layout}_{args.state}_attempt_001/observed_support.npz',allow_pickle=False) as original_support:
   assert set(current.files)==set(original_support.files)
   for key in current.files:np.testing.assert_array_equal(current[key],original_support[key])
  captures=capture_models(model,context,out,float(frozen['sigma_m']))
  after_check=out/'prefix_after_prediction_check.ply';after=old.save_mesh(context['mapper'],after_check);after_check.unlink()
  assert after['sha256']==rebuilt['sha256']
  reused=parts.reused_branches(args.layout,args.state);assert set(reused)=={int(r['candidate_id']) for r in context['routes']}
  baseline=B6/'replay'/f'{args.layout}_{args.state}_attempt_001'
  stages=[]
  for route in context['routes']:
   cid=int(route['candidate_id']);source=reused[cid]
   for arm,flags in ARMS.items():
    group=flags['group']
    stages.append(dict(stage_id=f'{args.layout}_{args.state}_{arm}_{cid}',layout=args.layout,variant=arm,state_action=args.state,candidate_id=cid,
     before_mesh=source['before_mesh'],after_mesh=source['after_mesh'],predictions=captures[arm]['predictions'],prediction_receipt=captures[arm]['receipt'],
     baseline_predictions=resource(baseline/'outer_and_layers_predictions.npz'),baseline_prediction_receipt=resource(baseline/'outer_and_layers_predictions.json'),
     fit_support_basis='full',prediction_groups=[dict(name=group,points_key=group+'_points',weights_key=group+f'_gain_{cid}',
      object_ids_key=group+'_object_ids',support_tracks_key=group+'_support_tracks',particle_ids_key=group+'_particle_ids',shape_ids_key=group+'_shape_ids')],
     execution_receipt=source['execution_receipt'],execution_reused=True))
  write(out/'manifest.json',dict(schema='B8_five_arm_reused_branch_manifest_v1',layout=args.layout,mode='candidate_branch',stages=stages,
   all_candidates_complete=True,candidate_acquisition_reused=True,observed_support=observed['arrays'],observed_support_metadata=observed['metadata'],
   baseline_variant='B6_outer_and_layers',acquired_support_array_identical_to_B5=True,acquisition_manifest=resource(previous/'manifest.json'),
   newly_executed_candidate_actions=0,arms=ARMS,selected_scale=resource(SCALE),common_bank_audit=resource(out/'common_bank_audit.json')))
  verify();old.verify();b.verify(args.layout)
  write(out/'result.json',dict(status='completed_five_arm_fixed_prefix_reused_branches',layout=args.layout,state=args.state,
   candidates=len(reused),variants=list(ARMS),fixed_prefix_exact=True,native_prefix_byte_identical_to_v1=True,native_mesh_unchanged_after_prediction=True,
   candidate_acquisition_reused=True,newly_executed_candidate_actions=0,prefix_reconstruction_actions=args.state,
   manifest=resource(out/'manifest.json'),freeze=resource(HERE/'freeze.json'),elapsed_s=time.monotonic()-started,hardware=False))
 except BaseException as exc:
  write(out/'failure.json',dict(error=repr(exc),traceback=traceback.format_exc(),elapsed_s=time.monotonic()-started,
   complete_experiment=False,newly_executed_candidate_actions=0));raise

if __name__=='__main__':main()
