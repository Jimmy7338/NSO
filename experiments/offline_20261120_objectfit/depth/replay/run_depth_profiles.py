#!/usr/bin/env python3
"""Profile already acquired depth at six sealed prefixes, without policy updates.

Only the frozen original sensor prefix is reconstructed. This runner does not
score new candidates, acquire a branch, or load source/reference meshes for the
profile. The unchanged AB model is checked directly against frozen B6 arrays.
"""
import argparse,hashlib,importlib.util,json,os,sys,time,traceback
from pathlib import Path
for name in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ[name]='1'
HERE=Path(__file__).resolve().parent;ROOT=HERE.parent
B3=ROOT.parent/'dual_template_geometry_20261120'
B5=ROOT.parent/'dual_template_objectfit_20261120'
B6=ROOT.parent/'dual_template_parts_20261120'
STATES={'shelf_left':(24,54,109),'shelf_right':(24,53,110)}

def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def read(path):return json.loads(Path(path).read_text())
def resource(path):return dict(path=str(Path(path).resolve()),sha256=sha(path))
def write(path,data):
 path=Path(path);temp=path.with_suffix(path.suffix+'.tmp');temp.write_text(json.dumps(data,indent=2,allow_nan=False)+'\n');temp.replace(path)
def load(name,path):
 spec=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(spec);sys.modules[name]=m;spec.loader.exec_module(m);return m

def frozen_paths():
 paths=[Path(__file__),ROOT/'protocol.json',B6/'replay/run_parts_predictions.py',B6/'replay/freeze.json']
 paths += [Path(p) for p in read(B6/'replay/freeze.json')['files_sha256']]
 paths += sorted((ROOT/'method').glob('*.py'))
 paths += [ROOT/'method/METHOD_SPEC.md',ROOT/'method/validation.json',ROOT/'method/tests.log',ROOT/'audit/independent_profile_review.json']
 for layout,states in STATES.items():
  for paid in states:
   folder=B6/'replay'/f'{layout}_{paid}_attempt_001'
   paths += [folder/name for name in ('result.json','manifest.json','observed_support.npz','observed_support.json','outer_and_layers_predictions.npz','outer_and_layers_predictions.json')]
 return sorted(set(paths))

def verify():
 frozen=read(HERE/'freeze.json');assert {str(p):sha(p)for p in frozen_paths()}==frozen['files_sha256'];return frozen

def freeze():
 p=read(ROOT/'protocol.json');assert p['status']=='frozen_before_depth_profile';assert p['states']=={k:list(v)for k,v in STATES.items()}
 path=HERE/'freeze.json';assert not path.exists()
 parent=load('B7_frozen_B6',B6/'replay/run_parts_predictions.py');parent.verify()
 write(path,dict(schema='B7_observed_depth_profile_v1',created_unix_s=time.time(),states=STATES,
  files_sha256={str(p):sha(p)for p in frozen_paths()},candidate_scoring=False,new_candidate_actions=0,parameter_update_applied=False,hardware=False))
 return path

def array_hash(array):
 import numpy as np
 a=np.ascontiguousarray(array);h=hashlib.sha256();h.update(str(a.dtype).encode());h.update(str(a.shape).encode());h.update(a.tobytes());return h.hexdigest()

def mapper_snapshot(mapper):
 import numpy as np
 values=list(mapper.quality.values());keys=np.asarray(list(mapper.quality),dtype=np.int64).reshape(-1,3)
 quality={key:array_hash(np.asarray([row[key]for row in values]))for key in values[0]}
 frames=[{key:array_hash(getattr(f,key))for key in ('depth_m','intrinsic','world_from_camera')}for f in mapper.keyframes]
 return dict(frames=mapper.frames,quality_keys=array_hash(keys),quality=quality,belief=array_hash(mapper.belief),
  tracks=array_hash(mapper.instance_support_index.memberships_for_keys(keys)),acquired_keyframes=frames)

def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--freeze',action='store_true');p.add_argument('--layout',choices=tuple(STATES));p.add_argument('--state',type=int);p.add_argument('--output',type=Path);args=p.parse_args()
 if args.freeze:print(freeze());return
 if args.layout is None or args.state not in STATES[args.layout] or args.output is None:p.error('Declared --layout --state and fresh --output required')
 verify();out=args.output;out.mkdir(parents=True,exist_ok=False);started=time.monotonic()
 try:
  import numpy as np
  parent=load('B7_frozen_B6',B6/'replay/run_parts_predictions.py');parent.verify()
  old,b,repair,model_module=parent.dependencies()
  profile_module=load('B7_observed_depth_profile',ROOT/'method/profile.py')
  context=old.sealed_prefix(b,args.layout,args.state);mapper=context['mapper']
  prior=B3/'replay'/f'{args.layout}_{args.state}_attempt_001'
  assert context['prefix_receipt']==read(prior/'prefix_execution.json')
  assert context['routes']==read(prior/'fixed_candidates.json')
  rebuilt=old.save_mesh(mapper,out/'prefix_native_check.ply');expected=resource(prior/'prefix_native_tsdf_mesh.ply')
  assert rebuilt['sha256']==expected['sha256'];(out/'prefix_native_check.ply').unlink()
  write(out/'prefix_execution.json',dict(**context['prefix_receipt'],native_mesh_byte_identical=True,native_mesh=expected,new_candidate_actions=0))
  baseline_dir=B6/'replay'/f'{args.layout}_{args.state}_attempt_001'
  baseline_arrays=resource(baseline_dir/'outer_and_layers_predictions.npz');baseline_receipt=resource(baseline_dir/'outer_and_layers_predictions.json')
  cls=model_module.configure(object_support=True,full_observed_support=True,geometry_fit=True,outer_orientation=True,observed_layers=True)
  models={name:cls(mapper,semantic=name=='S')for name in ('G','S')}
  before=mapper_snapshot(mapper);parity={}
  with np.load(baseline_arrays['path'],allow_pickle=False)as arrays:
   for name,model in models.items():
    parity[name]={}
    for key,value in (('points',model.points),('weights',model.weights),('hypothesis_ids',model.hypothesis_ids)):
     np.testing.assert_array_equal(value,arrays[name+'_'+key]);parity[name][key]=array_hash(value)
  np.testing.assert_array_equal(models['G'].points,models['S'].points)
  from nso.cpu_sensor_contract_v10 import json_value
  model_before={name:{key:array_hash(getattr(m,key))for key in ('points','weights','hypothesis_ids')}for name,m in models.items()}
  object_audit_before={name:hashlib.sha256(json.dumps(json_value(dict(objects=m.objects,audit=m.audit)),sort_keys=True,allow_nan=False).encode()).hexdigest()for name,m in models.items()}
  # The geometry profile receives the neutral-class model. No alternative
  # branch mesh or reference geometry is passed to this method.
  profile=profile_module.profile_model(mapper,models['G'])
  expected_frames=[hashlib.sha256(np.asarray(f.world_from_camera,float).tobytes()+np.asarray(f.intrinsic,float).tobytes()+np.ascontiguousarray(f.depth_m).tobytes()).hexdigest()for f in mapper.keyframes]
  object_lookup={int(o['support_track']):o for o in models['G'].objects}
  assert {int(o['support_track'])for o in profile['objects']}==set(object_lookup)
  for obj in profile['objects']:
   assert obj['fit_input_points_sha256']==object_lookup[int(obj['support_track'])]['fit_input_points_sha256']
   if 'fixed_ray_audit' in obj:
    assert obj['fixed_ray_audit']['keyframe_fingerprints']==expected_frames
    assert obj['fixed_ray_audit']['denominator_fixed_across_depth_and_hypotheses']
    assert obj['shape_mixture_prior']==[.5,.5]
    assert obj['current_depth_m'] in obj['depths_m']
    assert np.asarray(obj['surface_mean_clipped_distance_squared_m2']).shape==(len(obj['depths_m']),2)
    assert np.asarray(obj['free_mean_clipped_depth_gap_squared_m2']).shape==(len(obj['depths_m']),2)
  from nso.cpu_sensor_contract_v10 import json_value
  write(out/'profile.json',json_value(profile))
  assert mapper_snapshot(mapper)==before,'Profile mutated acquired mapper state'
  for name,m in models.items():
   assert {key:array_hash(getattr(m,key))for key in ('points','weights','hypothesis_ids')}==model_before[name]
   assert hashlib.sha256(json.dumps(json_value(dict(objects=m.objects,audit=m.audit)),sort_keys=True,allow_nan=False).encode()).hexdigest()==object_audit_before[name]
  write(out/'baseline_model_parity.json',dict(G_S_geometry_shared=True,baseline_arrays=baseline_arrays,baseline_receipt=baseline_receipt,
   per_arm_array_sha256=parity,exact_frozen_B6_AB_arrays=True,profile_did_not_mutate_mapper_or_model=True,
   no_new_candidate_scores_computed=True,interpretation='The current AB model is unchanged because this phase is diagnostic only. This does not assert that an identified future depth update would have no effect.'))
  write(out/'observed_input_hashes.json',before)
  verify();parent.verify();old.verify();b.verify(args.layout)
  write(out/'result.json',dict(status='completed_fixed_prefix_observed_depth_profile',layout=args.layout,state_action=args.state,
   profile=resource(out/'profile.json'),baseline_parity=resource(out/'baseline_model_parity.json'),observed_input_hashes=resource(out/'observed_input_hashes.json'),
   freeze=resource(HERE/'freeze.json'),native_prefix_byte_identical=True,new_candidate_actions=0,new_candidate_scores=0,
   parameter_update_applied=False,hardware=False,elapsed_s=time.monotonic()-started))
  print(json.dumps(dict(status='completed',layout=args.layout,state_action=args.state,elapsed_s=time.monotonic()-started)),flush=True)
 except BaseException as exc:
  write(out/'failure.json',dict(error=repr(exc),traceback=traceback.format_exc(),elapsed_s=time.monotonic()-started,
    new_candidate_actions=0,new_candidate_scores=0,parameter_update_applied=False));raise

if __name__=='__main__':main()
