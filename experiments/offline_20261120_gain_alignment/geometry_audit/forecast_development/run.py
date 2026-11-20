"""Frozen known-geometry development and independent acquisition probes."""
import os
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ[key]='1'
import argparse,hashlib,importlib.util,json,time,sys
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import acquisition_forecast as c
from surface_basis import build_surface_basis
HERE=Path(__file__).resolve().parent;FIX=HERE.parent/'independent_validation'
s=importlib.util.spec_from_file_location('fixture_helpers',FIX/'validate.py');v=importlib.util.module_from_spec(s);s.loader.exec_module(v)

def write(p,x):Path(p).write_text(json.dumps(x,indent=2,allow_nan=False,default=lambda a:a.item()if isinstance(a,np.generic)else a.tolist())+'\n')
def generate(seed):
 rng=np.random.default_rng(seed);objects=[]
 for i in range(12):
  family='closed'if i<6 else'open';w=float(rng.uniform(.65,1.2));d=float(rng.uniform(.35,.9));h=float(rng.uniform(.75,1.5));x=-w/2;y=-d/2
  if family=='closed':parts=[([x,y,0.],[w,d,h])]
  else:
   thick=float(rng.uniform(.035,.055));post=float(rng.uniform(.055,.09));level=float(rng.uniform(.38,.58)*h);levels=[0.,level,h-thick];parts=[([x,y,z],[w,d,thick])for z in levels]
   for a,b in zip(levels[:-1],levels[1:]):parts += [([x+dx,y+dy,a+thick],[post,post,b-a-thick])for dx in (0.,w-post)for dy in (0.,d-post)]
  objects.append(dict(object_id=f'object_{i:02d}',family=family,width=w,depth=d,height=h,parts=parts,source_seed=seed,reference_seed=202611201200+i))
 return objects

def freeze():
 old=v.read(FIX/'protocol.json');files=[Path(__file__),HERE/'acquisition_forecast.py',HERE/'surface_basis.py',HERE/'test_forecast.py',FIX/'validate.py',HERE.parent/'parts_audit.py',v.REPAIR]
 spec=dict(schema='acquisition_forward_known_geometry_v1',status='frozen_before_development_and_independent_test_outcomes',code_sha256={str(p):v.sha(p)for p in files},development_objects=old['objects'],independent_test_objects=generate(202611201113),independent_test_seed=202611201113,past_angles_deg=old['past_angles_deg'],routes=old['routes'],reference_angles_deg=old['reference_angles_deg'],independent_objects_per_split=12,routes_per_object=3,
 formula='Integral on hypothesis-own fixed .06m external-visible surface quadrature: area * not historical strict_free * (distance to acquired-past TSDF > .05m) * (distance to acquired-past plus rendered-hypothesis-future TSDF <= .05m). Gross is the gain; lost/net separately reported.',
 comparison='Matched same fine surface basis: original B2 point-visibility with10cm measured-surface history; point-visibility replacing measurement history with native5cm; forward estimator. Coarse prior variants are historical development only.',
 future_render='Ideal axial depth, >.15m and <=8m,120x90 fx=fy60; no simulated noise. In these explicitly known-geometry fixtures hypothesis equals generating geometry; predicted and actual channels are independently rendered and checked, so this is an acquisition/fusion consistency upper-bound probe, not unseen-shape prediction.',
 past='All3 acquired real frames fused; sparse old history independently retained only for strict_free and comparator. No mapper or posterior changes.',
 basis='Deterministic.06 quadrature, primitive outward internal rejection and coincident face area/multiplicity,12own orbital first-hit visibility; source evaluation independently MC30000 with independent seeds. Both are discrete area estimates, not exact boolean mesh area.',
 source_evaluation='Source-to-native5cm before/after gross/lost/net on independent source Monte Carlo; source reference created only after prediction arrays. Actual future depths/meshes never passed into forecast.',
 count='12 independent dimensional draws per split, balanced6closed6open;36 dependent routes. New test seed selected once before outcomes; no resampling/tuning. Old12 are development.',
 limitation='Known geometry and ideal sensor; no fitted-template posterior, online unknown exploration or G/S efficiency evidence. Fitted hypothesis first-hit rendering needs separate B2 strict-free intersection repair to avoid contradicted predicted occluders; the toy does not establish that capability.',old_development_protocol=dict(path=str(FIX/'protocol.json'),sha256=v.sha(FIX/'protocol.json')))
 write(HERE/'protocol.json',spec);print(json.dumps({'status':'frozen','sha256':v.sha(HERE/'protocol.json')}))

def run(split):
 spec=v.read(HERE/'protocol.json')
 for p,h in spec['code_sha256'].items():assert v.sha(p)==h,p
 out=HERE/(split+'_attempt_001');out.mkdir(exist_ok=False);rows=[];start=time.monotonic()
 old=v.read(FIX/'serialization_retry_002/result.json');oldobjs={o['object_id']:o for o in old['objects']}
 from env import virtual3d
 original=virtual3d.camera_pose
 try:
  for obj in spec[split+'_objects']:
   target=obj['height']/2;hyp=v.scene(v.mesh(obj['parts']));past_raw=[v.render(hyp,v.pose(a,target))for a in spec['past_angles_deg']]
   past=[c.DepthFrame(f.depth_m,f.world_from_camera,f.intrinsic,'acquired')for f in past_raw];history=[]
   for f in past_raw:
    if not history or np.linalg.norm(f.world_from_camera[:3,3]-history[-1].world_from_camera[:3,3])>.45 or np.dot(f.world_from_camera[:3,2],history[-1].world_from_camera[:3,2])<.8:history.append(f)
   native=c.integrate(past);native_scene=c.ray_scene(native);basis=build_surface_basis(obj['parts'],orbital_poses=[v.pose(a,target)for a in spec['reference_angles_deg']]);pts=basis['points'];areas=basis['areas'];hist=v.repair.classify_history(pts,history);allowed=areas*(~hist['strict_free']);nativekeep=c.distance(native,pts)>np.float32(.05)
   mapper=SimpleNamespace(config=SimpleNamespace(fov_deg=90.,height_px=90,width_px=120,max_depth_m=8.),keyframes=history,shape=(1,1));model=SimpleNamespace(mapper=mapper,points=pts,weights=allowed,hypothesis_ids=np.zeros(len(pts),int),hypothesis_rays=[hyp],audit={'history_rejection':True})
   provisional=[];arrays=dict(points=pts,areas=areas,strict_free=hist['strict_free'],history_measured_surface=hist['measured_surface'],native_keep=nativekeep)
   for name,angles in spec['routes'].items():
    poses=[v.pose(a,target)for a in angles];virtual3d.camera_pose=lambda cell,heading,config,shape,poses=poses:poses[int(cell[1])]
    trace=np.max([v.repair.trace_gain(model,[0,i],0,native_scene,repair_enabled=True)[0]for i in range(len(poses))],axis=0)
    original_fine=trace*(~hist['measured_surface']);replace_fine=trace*nativekeep
    future=c.render(hyp,poses,v.K,120,90);pred=c.forecast(past,future,pts,allowed)
    row=dict(route=name,matched_fine_original_visibility_m2=float(original_fine.sum()),matched_fine_native_visibility_m2=float(replace_fine.sum()),forecast=pred['receipt'],future_predicted_depth_hashes=[c.digest(f.depth_m)for f in future]);provisional.append((row,poses,future))
    arrays[name+'_matched_fine_original']=original_fine;arrays[name+'_matched_fine_native']=replace_fine;arrays[name+'_forecast_gain']=pred['gain'];arrays[name+'_forecast_loss']=pred['loss']
   # All actor predictions complete before independent reference samples and actual-future evaluation.
   truth=v.scene(v.mesh(obj['parts']));refs,area,refaudit=v.surface_reference(obj['parts'],truth,[v.pose(a,target)for a in spec['reference_angles_deg']],obj['reference_seed']);before=v.distance(native_scene,refs)<=np.float32(.05);candidates=[]
   for row,poses,predicted in provisional:
    actual=[v.render(truth,p)for p in poses];assert all(np.array_equal(a.depth_m,b.depth_m)for a,b in zip(actual,predicted));after=v.integrate(past_raw+actual);after_mask=v.distance(v.scene(after),refs)<=np.float32(.05);gross=float((~before&after_mask).sum()*area);lost=float((before&~after_mask).sum()*area)
    assert row['forecast']['after_vertices_sha256']==c.digest(np.asarray(after.vertices));assert row['forecast']['after_triangles_sha256']==c.digest(np.asarray(after.triangles))
    if split=='development':
     oldrow=next(r for r in oldobjs[obj['object_id']]['candidates']if r['route']==row['route']);assert oldrow['after_native_vertex_sha256']==c.digest(np.asarray(after.vertices));assert oldrow['actual_source_gross_new_m2']==gross
    row.update(actual_source_gross_new_m2=gross,actual_source_lost_m2=lost,actual_source_net_new_m2=gross-lost,exact_predicted_and_actual_acquisition_mesh=True)
    for label,value in [('fine_original',row['matched_fine_original_visibility_m2']),('fine_native',row['matched_fine_native_visibility_m2']),('forward',row['forecast']['predicted_gross_new_m2'])]:row[label+'_signed_error_m2']=value-gross;row[label+'_absolute_error_m2']=abs(value-gross)
    candidates.append(row)
   np.savez_compressed(out/(obj['object_id']+'_predictions.npz'),**arrays)
   rows.append(dict(object_id=obj['object_id'],family=obj['family'],past_frames=len(past),history_frames=len(history),basis=basis['receipt'],reference=refaudit,candidates=candidates));write(out/'progress.json',dict(completed=len(rows),total=12));print(json.dumps(dict(split=split,completed=len(rows))),flush=True)
 finally:virtual3d.camera_pose=original
 flat=[r for o in rows for r in o['candidates']]
 def summarize(rr):return dict(rows=len(rr),actual_gross_mean_m2=float(np.mean([r['actual_source_gross_new_m2']for r in rr])),**{m:dict(mean_signed_error_m2=float(np.mean([r[m+'_signed_error_m2']for r in rr])),mean_absolute_error_m2=float(np.mean([r[m+'_absolute_error_m2']for r in rr])))for m in ('fine_original','fine_native','forward')},forecast_wall_s=sum(r['forecast']['elapsed_s']for r in rr))
 result=dict(status='completed_known_geometry_acquisition_forward_probe',split=split,protocol_sha256=v.sha(HERE/'protocol.json'),independent_objects=12,dependent_route_rows=36,objects=rows,summary=summarize(flat),by_route={name:summarize([r for r in flat if r['route']==name])for name in spec['routes']},elapsed_s=time.monotonic()-start,all_predicted_native_meshes_match_actual_under_known_geometry=all(r['exact_predicted_and_actual_acquisition_mesh']for r in flat),new_unknown_exploration_policy_actions=0,not_fitted_geometry=True)
 for p,h in spec['code_sha256'].items():assert v.sha(p)==h,p
 write(out/'result.json',result);print(json.dumps(dict(split=split,status=result['status'],summary=result['summary'],sha256=v.sha(out/'result.json'))),flush=True)
if __name__=='__main__':
 a=argparse.ArgumentParser();a.add_argument('command',choices=['freeze','development','independent_test']);args=a.parse_args();freeze()if args.command=='freeze'else run(args.command)
