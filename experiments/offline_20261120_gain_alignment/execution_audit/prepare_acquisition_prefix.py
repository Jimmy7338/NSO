#!/usr/bin/env python3
"""Prepare acquired-only TSDF forecasting inputs; execute no future candidate.

Reconstruct one sealed development prefix. Export all acquired RGB-D frames,
current measured mesh, fixed outbound poses, and shared per-structure geometry
at the conditional mean of the frozen B9 posterior. This is an explicit point
approximation for the proposed acquisition operator, not the full B8 integral.
"""
import argparse,hashlib,importlib.util,json,os,sys,time
from pathlib import Path
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ[key]='1'
import numpy as np
HERE=Path(__file__).resolve().parent;ROOT=HERE.parent;BASE=ROOT.parent
B8=BASE/'dual_template_uncertainty_20261120';B9=BASE/'dual_template_followup_20261120'
def load(name,path):
 spec=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(spec);sys.modules[name]=m;spec.loader.exec_module(m);return m
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def arrhash(a):return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()
def write(p,x):Path(p).write_text(json.dumps(x,indent=2,allow_nan=False)+'\n')
def resource(p):return dict(path=str(p),sha256=sha(p))
def plain(x):
 if isinstance(x,np.ndarray):return x.tolist()
 if isinstance(x,np.generic):return x.item()
 if isinstance(x,dict):return {k:plain(v)for k,v in x.items()}
 if isinstance(x,(list,tuple)):return [plain(v)for v in x]
 return x

def prepare(layout,paid,output):
 output.mkdir(parents=True,exist_ok=False);start=time.monotonic()
 adapter=load('C_acquired_prefix_adapter',HERE/'prefix_adapter.py')
 context=adapter.rebuild_observed_prefix(layout,paid);prefix_seconds=time.monotonic()-start
 mapper=context.mapper;frames=context.acquired_frames
 model=load('C_frozen_B8_model',B8/'method/model.py');shape=load('C_frozen_B9_shape',B9/'structure/model.py')
 import open3d as o3d
 from env.virtual3d import camera_pose
 immutable_before=model.observed_input_signature(mapper)
 arrays={name:np.stack([np.asarray(getattr(f,name))for f in frames])for name in ('depth_m','color_rgb','intrinsic','world_from_camera','semantic')}
 arrays['timestamp_s']=np.asarray([f.timestamp_s for f in frames],float)
 original_frame_hashes=[{name:arrhash(getattr(f,name))for name in ('depth_m','color_rgb','intrinsic','world_from_camera','semantic')}for f in frames]
 np.savez_compressed(output/'acquired_frames.npz',**arrays)
 native=mapper.mesh();o3d.io.write_triangle_mesh(str(output/'native_prefix.ply'),native,write_ascii=False)
 assert sha(output/'native_prefix.ply')==context.receipt['native_mesh_sha256']
 # Exact independent replay of the actual TSDF integrator, retaining real RGB.
 ts=time.monotonic();vol=o3d.pipelines.integration.ScalableTSDFVolume(voxel_length=mapper.config.voxel_m,sdf_trunc=.12,color_type=o3d.pipelines.integration.TSDFVolumeColorType.RGB8)
 for f in frames:
  f.validate();k=f.intrinsic;h,w=f.depth_m.shape
  rgbd=o3d.geometry.RGBDImage.create_from_color_and_depth(o3d.geometry.Image(np.ascontiguousarray(f.color_rgb,np.uint8)),o3d.geometry.Image(np.ascontiguousarray(f.depth_m,np.float32)),depth_scale=1.,depth_trunc=mapper.config.max_depth_m,convert_rgb_to_intensity=False)
  intr=o3d.camera.PinholeCameraIntrinsic(w,h,k[0,0],k[1,1],k[0,2],k[1,2]);vol.integrate(rgbd,intr,np.linalg.inv(f.world_from_camera))
 replayed=vol.extract_triangle_mesh();replay_seconds=time.monotonic()-ts
 for field in ('vertices','triangles','vertex_colors'):
  np.testing.assert_array_equal(np.asarray(getattr(native,field)),np.asarray(getattr(replayed,field)))
 # Check the independent forecast helper's gray RGB does not alter geometry.
 helper_path=ROOT/'geometry_audit/forecast_development/acquisition_forecast.py';helper=load('C_forecast_helper_check',helper_path)
 past=[helper.DepthFrame(f.depth_m,f.world_from_camera,f.intrinsic,'acquired')for f in frames]
 ts=time.monotonic();gray=helper.integrate(past,mapper.config.voxel_m,.12,mapper.config.max_depth_m);gray_seconds=time.monotonic()-ts
 for field in ('vertices','triangles'):np.testing.assert_array_equal(np.asarray(getattr(native,field)),np.asarray(getattr(gray,field)))
 # The class-independent conditional geometry is obtained directly from B9
 # components, avoiding numerical division by a vanishing shape probability.
 ts=time.monotonic();bank=model.build_particle_bank(mapper);bank_seconds=time.monotonic()-ts
 objects=[]
 for oi,item in enumerate(bank.objects):
  v=item['variants']['uncertainty'];q,loge,_=shape.components(v['energy_m2'],v['prior_mass'],.01)
  ds=np.asarray([p['depth_m']for p in v['particles']]);lo=np.asarray([p['lower_m']for p in v['particles']])
  probabilities={}
  for arm,prior in [('G',[.5,.5]),('S',[1-item['shelf_prior_probability'],item['shelf_prior_probability']])]:
   posterior=shape.posterior(v['energy_m2'],v['prior_mass'],prior,16.,.01)
   np.testing.assert_array_equal(q,posterior['conditional_geometry_mass'])
   probabilities[arm]=posterior['shape_probability_mass'].tolist()
  hyp=[]
  for hi in (0,1):
   depth=float(q[:,hi]@ds);lower=float(q[:,hi]@lo)
   parts=model.primitive_parts(item['base'],depth,lower,item['bits'])[hi]
   mesh,_=model.mesh_scene(parts,item['base']['rotation']);meshpath=output/f'object_{oi}_shape_{hi}.ply'
   assert o3d.io.write_triangle_mesh(str(meshpath),mesh,write_ascii=False)
   hyp.append(dict(shape_id=hi,shape_name=('closed','open')[hi],conditional_mean_depth_m=depth,conditional_mean_lower_m=lower,parts=plain(parts),mesh=resource(meshpath),vertices=len(mesh.vertices),triangles=len(mesh.triangles)))
  objects.append(dict(object_index=oi,support_track=item['support_track'],base_geometry=plain(item['base']),conditional_geometry_mass=q.tolist(),particles=plain(v['particles']),shape_probability_mass=probabilities,hypotheses=hyp))
 write(output/'conditional_hypotheses.json',dict(schema='C_acquired_conditional_mean_hypotheses_preparation_v1',sigma_m=.01,structural_temperature=16.,point_estimate_approximation=True,likelihood_recomputed=False,geometry_shared_G_S=True,objects=objects,source_reference_consumed=False,observed_input_sha256=bank.observed_input_sha256))
 routes=[]
 for route,common in zip(context.routes,context.common_rows):
  poses=[camera_pose(s[:2],s[2],mapper.config,mapper.shape[0]).tolist()for s in route['outbound_states'][1:]]
  routes.append(dict(candidate_id=route['candidate_id'],original_route=route,common=common,future_camera_poses=poses))
 write(output/'fixed_candidates.json',routes)
 assert immutable_before==model.observed_input_signature(mapper)
 assert original_frame_hashes==[{name:arrhash(getattr(f,name))for name in ('depth_m','color_rgb','intrinsic','world_from_camera','semantic')}for f in frames]
 history=[i for i,f in enumerate(frames)if any(f is k for k in mapper.keyframes)]
 predicted_frames=sum(len(x['future_camera_poses'])for x in routes)*sum(len(o['hypotheses'])for o in objects)
 report=dict(status='complete_acquired_prefix_preparation_no_candidate_forecast',layout=layout,state_action=paid,
  prefix_receipt=context.receipt,acquired_frames=len(frames),history_keyframes=len(mapper.keyframes),retained_keyframe_indices=history,
  all_past_frames_not_only_keyframes=True,unchanged_mapper_and_frames=True,actor_given_source_reference=False,new_future_acquisitions=0,candidate_scores_computed=0,
  arrays={k:dict(shape=list(v.shape),dtype=str(v.dtype),sha256=arrhash(v))for k,v in arrays.items()},
  actual_native_replay=dict(vertices=len(native.vertices),triangles=len(native.triangles),vertices_sha256=arrhash(np.asarray(native.vertices)),triangles_sha256=arrhash(np.asarray(native.triangles)),same_actual_RGB_geometry_colors=True,constant_RGB_helper_geometry_exact=True),
  fixed_route_count=len(routes),objects=len(objects),mutually_exclusive_hypotheses_per_object=2,
  proposed_forecast_work=dict(hypothesis_candidate_pairs=len(routes)*sum(len(o['hypotheses'])for o in objects),future_depth_frames_if_all_pairs=predicted_frames,one_before_volume_fusion_frames=len(frames),before_volume_can_be_cached=True,scene='current measured mesh plus one conditional-mean hypothesis only; never both mutually exclusive shapes'),
  timing_s=dict(sealed_prefix_rebuild=prefix_seconds,actual_RGB_TSDF_replay=replay_seconds,constant_RGB_TSDF_replay=gray_seconds,conditional_bank=bank_seconds,total=time.monotonic()-start),
  limitations=['No future render, forecast score, actual branch outcome or target change evaluated here.','A conditional-mean shape replaces within-structure particle integration; this approximation must be separately declared.','Gray RGB helper check is geometry only; visual colors are not reproduced.'],
  sources={str(p):sha(p)for p in (Path(__file__),HERE/'prefix_adapter.py',B8/'method/model.py',B9/'structure/model.py',helper_path)},
  outputs={p.name:resource(p)for p in output.iterdir() if p.is_file()})
 write(output/'receipt.json',report);print(json.dumps({k:report[k]for k in ('status','acquired_frames','history_keyframes','fixed_route_count','objects','proposed_forecast_work','timing_s')}),flush=True)
 return report

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--layout',default='shelf_left');p.add_argument('--state',type=int,default=54);p.add_argument('--output',type=Path,required=True);a=p.parse_args();prepare(a.layout,a.state,a.output)
