"""Independent 12-object acquisition-metric validation, not a policy trial.

Known procedural template geometry isolates reward accounting. The actor-side
mask reads only acquired native TSDF, never future meshes or truth samples.
"""
import os
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ[key]='1'
import argparse,hashlib,importlib.util,json,sys,time
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import open3d as o3d
HERE=Path(__file__).resolve().parent;B10=HERE.parents[1];BASE=B10.parent
SOURCE=BASE/'reproduction_20261104/simulation_publication_stage/experiments/unknown_exploration_20261104/source'
REPAIR=BASE/'dual_template_repair_20261119/method/occlusion_repair.py'
METHOD=B10/'method/measured_novelty.py'
sys.path.insert(0,str(SOURCE));sys.path.insert(0,str(HERE.parent))
import parts_audit

def load(name,p):
 s=importlib.util.spec_from_file_location(name,p);m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m
repair=load('B10_frozen_B2',REPAIR);novelty=load('B10_measured_novelty',METHOD)
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def digest(a):return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()
def read(p):return json.loads(Path(p).read_text())
def write(p,x):Path(p).write_text(json.dumps(x,indent=2,allow_nan=False)+'\n')
def mesh(parts):
 out=o3d.geometry.TriangleMesh()
 for lo,size in parts:out+=o3d.geometry.TriangleMesh.create_box(*size).translate(lo)
 return out

def scene(m):
 if not len(m.triangles):return None
 s=o3d.t.geometry.RaycastingScene(nthreads=1);s.add_triangles(o3d.t.geometry.TriangleMesh.from_legacy(m));return s

def pose(angle,target_z,range_m=1.6):
 a=np.deg2rad(angle);eye=np.array([range_m*np.sin(a),-range_m*np.cos(a),1.3]);target=np.array([0.,0.,target_z]);z=target-eye;z/=np.linalg.norm(z);x=np.cross(z,[0,0,1.]);x/=np.linalg.norm(x);y=np.cross(z,x)
 p=np.eye(4);p[:3,:3]=np.column_stack([x,y,z]);p[:3,3]=eye;return p

K=np.array([[60.,0.,59.5],[0.,60.,44.5],[0.,0.,1.]])
def render(s,p):
 v,u=np.mgrid[:90,:120];d=np.column_stack([(u.ravel()-K[0,2])/60,(v.ravel()-K[1,2])/60,np.ones(10800)])@p[:3,:3].T
 r=np.column_stack([np.tile(p[:3,3],(10800,1)),d]).astype(np.float32);z=s.cast_rays(o3d.core.Tensor(r),nthreads=1)['t_hit'].numpy().reshape(90,120)
 z=np.where(np.isfinite(z)&(z>.2)&(z<8.),z,0).astype(np.float32)
 return SimpleNamespace(depth_m=z,world_from_camera=p,intrinsic=K)

def integrate(frames):
 volume=o3d.pipelines.integration.ScalableTSDFVolume(voxel_length=.06,sdf_trunc=.12,color_type=o3d.pipelines.integration.TSDFVolumeColorType.RGB8)
 intr=o3d.camera.PinholeCameraIntrinsic(120,90,60.,60.,59.5,44.5)
 for frame in frames:
  rgb=o3d.geometry.Image(np.full((90,120,3),127,np.uint8));depth=o3d.geometry.Image(np.ascontiguousarray(frame.depth_m))
  rgbd=o3d.geometry.RGBDImage.create_from_color_and_depth(rgb,depth,depth_scale=1.,depth_trunc=8.,convert_rgb_to_intensity=False)
  volume.integrate(rgbd,intr,np.linalg.inv(frame.world_from_camera))
 return volume.extract_triangle_mesh()

def distance(s,pts):
 return s.compute_distance(o3d.core.Tensor(np.asarray(pts,np.float32)),nthreads=1).numpy()if s is not None else np.full(len(pts),np.inf)

def surface_reference(parts,truth,poses,seed):
 # Non-overlapping component volumes share contact planes only. Uniform
 # triangle-area source sampling and independent first-hit visibility remove
 # these interior contact surfaces without using actor geometry or routes.
 m=mesh(parts);vs=np.asarray(m.vertices);tris=vs[np.asarray(m.triangles)];area=np.linalg.norm(np.cross(tris[:,1]-tris[:,0],tris[:,2]-tris[:,0]),axis=1)/2
 rng=np.random.default_rng(seed);idx=rng.choice(len(area),30000,p=area/area.sum());uv=rng.random((30000,2));rt=np.sqrt(uv[:,0]);pts=(1-rt[:,None])*tris[idx,0]+(rt*(1-uv[:,1]))[:,None]*tris[idx,1]+(rt*uv[:,1])[:,None]*tris[idx,2]
 visible=np.zeros(len(pts),bool)
 for p in poses:
  local=(pts-p[:3,3])@p[:3,:3];z=local[:,2];mask=(z>.2)&(z<8)&(abs(local[:,0])<z)&(abs(local[:,1])<z*.75);ids=np.flatnonzero(mask)
  delta=pts[ids]-p[:3,3];r=np.column_stack([np.tile(p[:3,3],(len(ids),1)),delta]).astype(np.float32);hit=truth.cast_rays(o3d.core.Tensor(r),nthreads=1)['t_hit'].numpy();d=np.linalg.norm(delta,axis=1)
  visible[ids]|=abs(hit-1)*d<=.001
 return pts[visible],float(area.sum()/30000),dict(drawn=30000,visible=int(visible.sum()),reference_points_sha256=digest(pts[visible]),fixed_pose_count=len(poses),sample_area_m2=float(area.sum()/30000))

def generate():
 rng=np.random.default_rng(202611201007);objects=[]
 for i in range(12):
  family='closed'if i<6 else'open';w=float(rng.uniform(.65,1.2));d=float(rng.uniform(.35,.9));h=float(rng.uniform(.75,1.5));x=-w/2;y=-d/2
  if family=='closed':parts=[([x,y,0.],[w,d,h])]
  else:
   thick=float(rng.uniform(.035,.055));post=float(rng.uniform(.055,.09));level=float(rng.uniform(.38,.58)*h);levels=[0.,level,h-thick]
   parts=[([x,y,z],[w,d,thick])for z in levels]
   for a,b in zip(levels[:-1],levels[1:]):
    parts += [([x+dx,y+dy,a+thick],[post,post,b-a-thick])for dx in (0.,w-post)for dy in (0.,d-post)]
  objects.append(dict(object_id=f'object_{i:02d}',family=family,width=w,depth=d,height=h,parts=parts,source_seed=202611201007,reference_seed=202611201100+i))
 spec=dict(status='frozen_before_new_acquisition_and_metric_outcomes',objects=objects,objects_count=12,independent_object_draws=True,
  past_angles_deg=[0.,6.,12.],routes={'revisit':[0.,6.,12.],'side':[24.,45.,70.],'around':[100.,140.,180.]},reference_angles_deg=list(range(0,360,30)),
  dimensions_distributions={'width':[.65,1.2],'depth':[.35,.9],'height':[.75,1.5]},sensor=dict(width=120,height=90,fx=60.,fy=60.,max_depth_m=8.,voxel_m=.06,truncation_m=.12),
  scope='Known synthetic geometry reward-accounting test; not fitted-geometry, full NSO policy, natural sensor or hardware validation. No G/S effect claim.',
  thresholds=dict(native_novelty_m=.05,actual_source_reconstruction_m=.05,history_axial_m=.10),
  primary='Old-versus-new route predicted duplicate mass under current native mesh; true-source gross newly reconstructed area reported separately, never assumed equal to template mass.',
  secondary='Absolute error to actual gross source surface, rank across the3routes, all negative outcomes; no parameter fitting or scene retry.',
  sampling='Fixed independent source-surface MC30000/object, reference visibility over12predeclared orbital poses. Native TSDF integrates all acquired frames.',
  history='Original sparsification: keep first then displacement>.45m or forward-axisdot<.8, independent of the reward change.',
  method_sha256=sha(METHOD),code_sha256={str(p):sha(p)for p in [Path(__file__),HERE.parent/'parts_audit.py',REPAIR,METHOD]})
 write(HERE/'protocol.json',spec);print(json.dumps(dict(status=spec['status'],objects=len(objects),protocol_sha256=sha(HERE/'protocol.json'))))

def run():
 spec=read(HERE/'protocol.json');assert not(HERE/'result.json').exists()
 for p,h in spec['code_sha256'].items():assert sha(p)==h,p
 from env import virtual3d
 original_pose=virtual3d.camera_pose
 rows=[];start=time.monotonic()
 try:
  for obj in spec['objects']:
   parts=obj['parts'];truth=scene(mesh(parts));target=obj['height']/2
   past=[render(truth,pose(a,target))for a in spec['past_angles_deg']];history=[]
   for f in past:
    if not history or np.linalg.norm(f.world_from_camera[:3,3]-history[-1].world_from_camera[:3,3])>.45 or np.dot(f.world_from_camera[:3,2],history[-1].world_from_camera[:3,2])<.8:history.append(f)
   measured=integrate(past);measured_scene=scene(measured);points,areas,_,_=parts_audit.quad(parts)
   hist=repair.classify_history(points,history);history_keep=~hist['strict_free']&~hist['measured_surface'];weights=areas*history_keep
   keep,dist=novelty.unmeasured_mask(points,measured_scene)
   config=SimpleNamespace(fov_deg=90.,height_px=90,width_px=120,max_depth_m=8.);mapper=SimpleNamespace(config=config,keyframes=history,shape=(1,1))
   model=SimpleNamespace(mapper=mapper,points=points,weights=weights,hypothesis_ids=np.zeros(len(points),int),hypothesis_rays=[truth],audit={'history_rejection':True})
   refs,area,refaudit=surface_reference(parts,truth,[pose(a,target)for a in spec['reference_angles_deg']],obj['reference_seed']);before=distance(measured_scene,refs)<=np.float32(.05)
   candidate_rows=[];matrices={}
   for name,angles in spec['routes'].items():
    poses=[pose(a,target)for a in angles];virtual3d.camera_pose=lambda cell,heading,config,shape,poses=poses:poses[int(cell[1])]
    gains=[repair.trace_gain(model,[0,i],0,measured_scene,repair_enabled=True)[0]for i in range(len(poses))];old=np.max(gains,axis=0);new=novelty.consume(old,keep)
    future=[render(truth,p)for p in poses];after_mesh=integrate(past+future);after=distance(scene(after_mesh),refs)<=np.float32(.05)
    gross=float((after&~before).sum()*area);lost=float((before&~after).sum()*area);net=gross-lost
    assert np.all(new<=old)and np.all(new[~keep]==0)
    rowspec=dict(route=name,predicted_before_m2=float(old.sum()),predicted_after_m2=float(new.sum()),duplicate_reward_before_m2=float(old[~keep].sum()),duplicate_reward_after_m2=float(new[~keep].sum()),
     actual_source_gross_new_m2=gross,actual_source_lost_m2=lost,actual_source_net_new_m2=net,absolute_error_before_m2=abs(old.sum()-gross),absolute_error_after_m2=abs(new.sum()-gross),
     before_native_vertices=len(measured.vertices),after_native_vertices=len(after_mesh.vertices),predicted_hypothesis_not_fused=True,
     future_depth_hashes=[digest(f.depth_m)for f in future],after_native_vertex_sha256=digest(np.asarray(after_mesh.vertices)),after_native_triangle_sha256=digest(np.asarray(after_mesh.triangles)))
    candidate_rows.append(rowspec);matrices[name+'_old']=old;matrices[name+'_new']=new
   np.savez_compressed(HERE/(obj['object_id']+'_predictions.npz'),points=points,areas=areas,history_keep=history_keep,native_keep=keep,native_distance=dist,**matrices)
   row=dict(object_id=obj['object_id'],family=obj['family'],past_frames=len(past),history_keyframes=len(history),reference=refaudit,
    past_depth_hashes=[digest(f.depth_m)for f in past],before_native_vertex_sha256=digest(np.asarray(measured.vertices)),before_native_triangle_sha256=digest(np.asarray(measured.triangles)),candidates=candidate_rows)
   rows.append(row);write(HERE/'progress.json',dict(completed=len(rows),planned=12));print(json.dumps(dict(object=obj['object_id'],completed=len(rows))),flush=True)
 finally:virtual3d.camera_pose=original_pose
 flat=[r for o in rows for r in o['candidates']]
 result=dict(status='completed_independent_twelve_object_novelty_validation',protocol_sha256=sha(HERE/'protocol.json'),objects=rows,objects_completed=len(rows),routes_evaluated=len(flat),no_actions_or_policy_trials=True,known_geometry_fixture=True,
  summary=dict(duplicate_before_mean_m2=float(np.mean([r['duplicate_reward_before_m2']for r in flat])),duplicate_after_mean_m2=float(np.mean([r['duplicate_reward_after_m2']for r in flat])),
   source_gross_absolute_error_before_mean_m2=float(np.mean([r['absolute_error_before_m2']for r in flat])),source_gross_absolute_error_after_mean_m2=float(np.mean([r['absolute_error_after_m2']for r in flat])),
   absolute_error_improved=sum(r['absolute_error_after_m2']<r['absolute_error_before_m2']-1e-12 for r in flat),absolute_error_worsened=sum(r['absolute_error_after_m2']>r['absolute_error_before_m2']+1e-12 for r in flat)),
  elapsed_s=time.monotonic()-start,limitations=['Duplicate reward elimination follows the pointwise mask and is not evidence of policy improvement.',
   'Known exact procedural geometry isolates acquisition accounting; fitted/unknown shape misspecification and category inference are not tested.',
   'The source-reference query direction differs from template-to-mesh distance. Actual gross surface error can worsen and all routes are retained.',
   'Native TSDF implementation and discretization are realistic software components, but sensor poses/depth are ideal and there is no collision/navigation simulation.'])
 for p,h in spec['code_sha256'].items():assert sha(p)==h,p
 write(HERE/'result.json',result);print(json.dumps(dict(status=result['status'],summary=result['summary'],sha256=sha(HERE/'result.json'))))

def main():
 a=argparse.ArgumentParser();a.add_argument('command',choices=['freeze','run']);args=a.parse_args();generate()if args.command=='freeze'else run()
if __name__=='__main__':main()
