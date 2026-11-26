#!/usr/bin/env python3
"""All 11 original L54 candidates, four acquired conditional-mean hypotheses.

No actual future frames or source reference are read. Native future images are
rendered from current measured mesh plus exactly one hypothetical object. This
is a declared conditional-mean and sensor-fusion prediction revision.
"""
import argparse,hashlib,importlib.util,json,os,sys,time
from pathlib import Path
from types import SimpleNamespace
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ[key]='1'
import numpy as np
import open3d as o3d
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[1]
PREFIX=HERE.parent/'acquisition_prefix/shelf_left_54_attempt_001'
CORE=ROOT/'geometry_audit/forecast_development'
BASE=ROOT.parent;B8=BASE/'dual_template_uncertainty_20261120'
SOURCE=BASE/'reproduction_20261104/simulation_publication_stage/experiments/unknown_exploration_20261104/source'
sys.path.insert(0,str(CORE));sys.path.insert(0,str(SOURCE))
import acquisition_forecast as C
from surface_basis import build_surface_basis
import render_observed_free as R

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def resource(p):return dict(path=str(p),sha256=sha(p))
def write(p,x):Path(p).write_text(json.dumps(x,indent=2,allow_nan=False,default=lambda a:a.item()if isinstance(a,np.generic)else a.tolist())+'\n')
def arrayhash(a):return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()
def protocol():return json.loads((HERE/'protocol.json').read_text())
def verify():
 cfg=protocol()
 for p,h in cfg['files_sha256'].items():assert sha(p)==h,p
 assert cfg['states']==[{'layout':'shelf_left','paid_actions':54}]
 assert cfg['hypothesis_candidate_pairs']==44
 return cfg

def run(output):
 cfg=verify();output.mkdir(exist_ok=False);start=time.monotonic()
 pref=json.loads((PREFIX/'receipt.json').read_text());objects=json.loads((PREFIX/'conditional_hypotheses.json').read_text())['objects'];routes=json.loads((PREFIX/'fixed_candidates.json').read_text())
 assert len(objects)==2 and len(routes)==11
 with np.load(PREFIX/'acquired_frames.npz',allow_pickle=False)as z:
  past=[C.DepthFrame(z['depth_m'][i].copy(),z['world_from_camera'][i].copy(),z['intrinsic'][i].copy(),'acquired')for i in range(len(z['depth_m']))]
 history=[past[i]for i in pref['retained_keyframe_indices']]
 assert len(past)==55 and len(history)==21
 immutable=[C.frame_hash(f)for f in past]
 native=o3d.io.read_triangle_mesh(str(PREFIX/'native_prefix.ply'));native_ray=C.ray_scene(native)
 assert arrayhash(np.asarray(native.vertices))==pref['actual_native_replay']['vertices_sha256']
 assert arrayhash(np.asarray(native.triangles))==pref['actual_native_replay']['triangles_sha256']
 poses={}
 for route in routes:
  assert len(route['future_camera_poses'])==len(route['original_route']['outbound_states'])-1
  for state,pose in zip(route['original_route']['outbound_states'][1:],route['future_camera_poses']):
   key=tuple(map(int,state))
   if key in poses:np.testing.assert_array_equal(poses[key],pose)
   poses[key]=np.asarray(pose,float)
 mapper=SimpleNamespace(config=SimpleNamespace(fov_deg=90.,height_px=90,width_px=120,max_depth_m=8.,resolution_m=.2,camera_height_m=1.2),shape=(40,50),keyframes=history)
 from env.virtual3d import camera_pose
 for state,pose in poses.items():np.testing.assert_array_equal(pose,camera_pose(state[:2],state[2],mapper.config,mapper.shape[0]))
 all_results=[];complete_pairs=0
 for obj in objects:
  transform=np.eye(4);transform[:3,:3]=np.asarray(obj['base_geometry']['rotation'])
  for hyp in obj['hypotheses']:
   ts=time.monotonic();hi=hyp['shape_id'];oi=obj['object_index'];name=f'object_{oi}_shape_{hi}'
   parts=[(np.asarray(lo,float),np.asarray(size,float))for lo,size in hyp['parts']]
   basis=build_surface_basis(parts,world_from_local=transform);points=basis['points'];areas=basis['areas'];hist=R.REPAIR.classify_history(points,history)
   allowed=areas*(~hist['strict_free']);nativekeep=C.distance(native,points)>np.float32(.05)
   hyp_path=Path(hyp['mesh']['path']);assert sha(hyp_path)==hyp['mesh']['sha256'];mesh=o3d.io.read_triangle_mesh(str(hyp_path));scene=C.ray_scene(mesh)
   proxy=SimpleNamespace(mapper=mapper,points=points,weights=allowed,hypothesis_ids=np.zeros(len(points),int),hypothesis_rays=[scene],audit={'history_rejection':True})
   cache={};visibility={};ray_audits={}
   for state,pose in poses.items():
    depth,audit=R.corrected_depth(scene,native_ray,pose,past[0].intrinsic,120,90,history,min_depth_m=.15,max_depth_m=8.)
    cache[state]=C.DepthFrame(depth,pose.copy(),past[0].intrinsic.copy(),'predicted_hypothesis');ray_audits[str(state)]=audit
    visibility[state]=R.REPAIR.trace_gain(proxy,state[:2],state[2],native_ray,repair_enabled=True)[0]
   arrays=dict(points=points,areas=areas,strict_free=hist['strict_free'],history_measured_surface=hist['measured_surface'],native_keep=nativekeep)
   rows=[]
   for route in routes:
    cid=int(route['candidate_id']);states=[tuple(map(int,s))for s in route['original_route']['outbound_states'][1:]]
    future=[cache[s]for s in states];prediction=C.forecast(past,future,points,allowed)
    assert prediction['receipt']['before_vertices_sha256']==pref['actual_native_replay']['vertices_sha256']
    assert prediction['receipt']['before_triangles_sha256']==pref['actual_native_replay']['triangles_sha256']
    union=np.maximum.reduce([visibility[s]for s in states])if states else np.zeros(len(points))
    old_fine=union*(~hist['measured_surface']);native_fine=union*nativekeep
    rows.append(dict(candidate_id=cid,predicted_gross_new_m2=float(prediction['gain'].sum()),predicted_lost_m2=float(prediction['loss'].sum()),predicted_net_new_m2=float(prediction['gain'].sum()-prediction['loss'].sum()),matched_fine_original_visibility_m2=float(old_fine.sum()),matched_fine_native_visibility_m2=float(native_fine.sum()),forecast_receipt=prediction['receipt'],future_hypothesis_depth_hashes=[arrayhash(f.depth_m)for f in future]))
    arrays[f'candidate_{cid}_forward_gain']=prediction['gain'];arrays[f'candidate_{cid}_forward_loss']=prediction['loss'];arrays[f'candidate_{cid}_fine_original']=old_fine;arrays[f'candidate_{cid}_fine_native']=native_fine
    complete_pairs+=1;write(output/'progress.json',dict(completed_pairs=complete_pairs,total_pairs=44,object_index=oi,shape_id=hi,candidate_id=cid));print(json.dumps(dict(event='prediction_complete',completed=complete_pairs,total=44,object=oi,shape=hi,candidate=cid)),flush=True)
   np.savez_compressed(output/(name+'.npz'),**arrays)
   row=dict(object_index=oi,support_track=obj['support_track'],shape_id=hi,conditional_mean_depth_m=hyp['conditional_mean_depth_m'],conditional_mean_lower_m=hyp['conditional_mean_lower_m'],shape_probability_mass={arm:obj['shape_probability_mass'][arm][hi]for arm in ('G','S')},basis=basis['receipt'],strict_free_area_m2=float(areas[hist['strict_free']].sum()),area_after_strict_free_m2=float(allowed.sum()),rows=rows,arrays=resource(output/(name+'.npz')),unique_future_camera_poses=len(cache),future_render_audits=ray_audits,elapsed_s=time.monotonic()-ts)
   write(output/(name+'.json'),row);all_results.append(row)
 arms={};comparators={m:{}for m in ('fine_original','fine_native')}
 for arm in ('G','S'):
  rows=[];comparison={m:[]for m in comparators}
  for route in routes:
   cid=int(route['candidate_id']);gross=lost=0.;fine={'fine_original':0.,'fine_native':0.}
   for hyp in all_results:
    w=hyp['shape_probability_mass'][arm];r=next(x for x in hyp['rows']if x['candidate_id']==cid)
    gross+=w*r['predicted_gross_new_m2'];lost+=w*r['predicted_lost_m2'];fine['fine_original']+=w*r['matched_fine_original_visibility_m2'];fine['fine_native']+=w*r['matched_fine_native_visibility_m2']
   common=float(route['common']['common_total_proxy']);cost=route['original_route']['cost']
   row=dict(candidate_id=cid,template_gain_m2=gross,predicted_lost_m2=lost,predicted_net_new_m2=gross-lost,common_total_proxy=common,score=common+gross,cost=cost,outbound_actions=len(route['original_route']['outbound_actions']))
   rows.append(row)
   for m in fine:comparison[m].append(dict(candidate_id=cid,template_gain_m2=fine[m],common_total_proxy=common,score=common+fine[m],cost=cost))
  def winner(rr):return min(rr,key=lambda r:(-r['score'],r['cost'],r['candidate_id']))['candidate_id']
  arms[arm]=dict(rows=rows,selected_candidate=winner(rows))
  for m in comparison:comparators[m][arm]=dict(rows=comparison[m],selected_candidate=winner(comparison[m]))
 assert immutable==[C.frame_hash(f)for f in past]
 verify()
 result=dict(status='completed_fitted_acquisition_prediction_all_original_candidates',layout='shelf_left',state_action=54,protocol=resource(HERE/'protocol.json'),hypothesis_candidate_pairs=complete_pairs,original_candidate_count=len(routes),acquired_past_frames=55,history_keyframes=21,arms=arms,comparators=comparators,hypothesis_outputs=[resource(output/f'object_{h["object_index"]}_shape_{h["shape_id"]}.json')for h in all_results],current_native_and_past_unchanged=True,future_reference_geometry_or_actual_frames_read=False,new_acquisition_actions=0,template_geometry='B9 per-structure conditional mean at sigma.01/T16; shared G/S geometry, original different structural priors',acquisition_revision='gross own-surface native5cm before/forecastafter, retains B2 strict-free surface/ray exclusion and acquired-surface veto; no10cm measured-surface consumption',future_unknown_occlusion_limit='Only current acquired mesh and one mutually exclusive object hypothesis; other unseen geometry is absent.',elapsed_s=time.monotonic()-start)
 write(output/'result.json',result);print(json.dumps({k:result[k]for k in ('status','hypothesis_candidate_pairs','elapsed_s')}),flush=True)

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);a=p.parse_args();run(a.output)
