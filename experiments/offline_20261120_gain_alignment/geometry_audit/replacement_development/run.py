"""B10b old12 development only. No new policy actions or tuning."""
import os
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ[k]='1'
from pathlib import Path
import json,hashlib,importlib.util,sys,time
from types import SimpleNamespace
import numpy as np
from scipy.stats import spearmanr
HERE=Path(__file__).resolve().parent;FIX=HERE.parent/'independent_validation';sys.path.insert(0,str(HERE))
import replacement
spec=importlib.util.spec_from_file_location('fixed12_helpers',FIX/'validate.py');v=importlib.util.module_from_spec(spec);spec.loader.exec_module(v)
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p):return json.loads(Path(p).read_text())
def write(p,x):Path(p).write_text(json.dumps(x,indent=2,allow_nan=False,default=lambda a:a.item()if isinstance(a,np.generic)else a.tolist())+'\n')
def main():
 assert not(HERE/'result.json').exists()
 inputs=[Path(__file__),HERE/'replacement.py',HERE/'test_replacement.py',HERE/'tests.log',FIX/'protocol.json',FIX/'validate.py',FIX/'serialization_retry_002/result.json',FIX/'serialization_retry_002/retry_receipt.json']
 freeze=dict(status='frozen_B1_B2_development_formula_before_outcomes',source_objects='Same old12 now development; not independent test',variants={'A_append':dict(resolution_m=.22,formula='area * old_history_keep * native_keep'),'B1_replace_coarse':dict(resolution_m=.22,formula='area * not_strict_free * native_keep'),'B2_replace_fine':dict(resolution_m=.06,formula='area * not_strict_free * native_keep')},native_threshold_m=.05,source_threshold_m=.05,files_sha256={str(p):sha(p)for p in inputs})
 write(HERE/'freeze.json',freeze)
 cfg=read(FIX/'protocol.json');old=read(FIX/'serialization_retry_002/result.json');lookup={o['object_id']:o for o in old['objects']}
 from env import virtual3d
 original_pose=virtual3d.camera_pose;rows=[];start=time.monotonic()
 try:
  for obj in cfg['objects']:
   parts=obj['parts'];truth=v.scene(v.mesh(parts));target=obj['height']/2;past=[v.render(truth,v.pose(a,target))for a in cfg['past_angles_deg']];history=[]
   for f in past:
    if not history or np.linalg.norm(f.world_from_camera[:3,3]-history[-1].world_from_camera[:3,3])>.45 or np.dot(f.world_from_camera[:3,2],history[-1].world_from_camera[:3,2])<.8:history.append(f)
   native=v.integrate(past);assert v.digest(np.asarray(native.vertices))==lookup[obj['object_id']]['before_native_vertex_sha256'];measured_scene=v.scene(native)
   parent_arrays=np.load(FIX/'serialization_retry_002'/(obj['object_id']+'_predictions.npz'),allow_pickle=False);variants={}
   for name,definition in freeze['variants'].items():
    points,area=replacement.quadrature(parts,definition['resolution_m']);hist=v.repair.classify_history(points,history);keep,dist=v.novelty.unmeasured_mask(points,measured_scene)
    if name=='A_append':weights=area*(~hist['strict_free']&~hist['measured_surface']&keep)
    else:weights=replacement.weights(area,hist['strict_free'],keep)
    config=SimpleNamespace(fov_deg=90.,height_px=90,width_px=120,max_depth_m=8.);mapper=SimpleNamespace(config=config,keyframes=history,shape=(1,1))
    model=SimpleNamespace(mapper=mapper,points=points,weights=weights,hypothesis_ids=np.zeros(len(points),int),hypothesis_rays=[truth],audit={'history_rejection':True})
    candidates=[];arrays=dict(points=points,areas=area,history_strict_free=hist['strict_free'],history_measured_surface=hist['measured_surface'],native_keep=keep,native_distance=dist,weights=weights)
    for route,angles in cfg['routes'].items():
     poses=[v.pose(a,target)for a in angles];virtual3d.camera_pose=lambda cell,heading,config,shape,poses=poses:poses[int(cell[1])]
     gains=[v.repair.trace_gain(model,[0,i],0,measured_scene,repair_enabled=True)[0]for i in range(len(poses))];gain=np.max(gains,axis=0);arrays[route+'_gain']=gain
     prior=next(r for r in lookup[obj['object_id']]['candidates']if r['route']==route);actual=prior['actual_source_gross_new_m2']
     if name=='A_append':np.testing.assert_array_equal(gain,parent_arrays[route+'_new'])
     restored=(hist['measured_surface']&~hist['strict_free']&keep)
     candidates.append(dict(route=route,predicted_m2=float(gain.sum()),actual_gross_new_m2=actual,actual_net_new_m2=prior['actual_source_net_new_m2'],
       absolute_error_m2=float(abs(gain.sum()-actual)),signed_error_m2=float(gain.sum()-actual),remaining_native_duplicate_m2=float(gain[~keep].sum()),
       restored_depth_band_reward_m2=float(gain[restored].sum()),original_predicted_m2=prior['predicted_before_m2'],original_absolute_error_m2=prior['absolute_error_before_m2']))
    pred=[r['predicted_m2']for r in candidates];actual=[r['actual_gross_new_m2']for r in candidates]
    rho=float(spearmanr(pred,actual).statistic)if np.ptp(pred)>0 and np.ptp(actual)>0 else None
    variants[name]=dict(candidates=candidates,points=len(points),base_area_m2=float(area.sum()),history_keep_count=int((~hist['strict_free']&~hist['measured_surface']).sum()),native_keep_count=int(keep.sum()),rank_rho_vs_gross=rho)
    np.savez_compressed(HERE/f"{obj['object_id']}_{name}.npz",**arrays)
   rows.append(dict(object_id=obj['object_id'],family=obj['family'],variants=variants));print(json.dumps(dict(completed=len(rows))),flush=True)
 finally:virtual3d.camera_pose=original_pose
 summary={}
 for name in freeze['variants']:
  summary[name]={}
  for route in ['all',*cfg['routes']]:
   rr=[r for obj in rows for r in obj['variants'][name]['candidates']if route=='all'or r['route']==route]
   summary[name][route]=dict(n=len(rr),mean_prediction_m2=float(np.mean([r['predicted_m2']for r in rr])),mean_actual_gross_m2=float(np.mean([r['actual_gross_new_m2']for r in rr])),mean_absolute_error_m2=float(np.mean([r['absolute_error_m2']for r in rr])),mean_signed_error_m2=float(np.mean([r['signed_error_m2']for r in rr])),mean_restored_depth_band_reward_m2=float(np.mean([r['restored_depth_band_reward_m2']for r in rr])),
    improved_vs_original=sum(r['absolute_error_m2']<r['original_absolute_error_m2']-1e-12 for r in rr),worsened_vs_original=sum(r['absolute_error_m2']>r['original_absolute_error_m2']+1e-12 for r in rr))
 for p,h in freeze['files_sha256'].items():assert sha(p)==h,p
 result=dict(status='completed_old12_replacement_development_diagnostic',independent_new_test=False,new_policy_actions=0,old_exact_native_prefix_reproduced=True,A_exact_original_corrected_arrays=True,objects=rows,summary=summary,elapsed_s=time.monotonic()-start,freeze_sha256=sha(HERE/'freeze.json'),
  limits=['Known exact geometry and the same12oldfixtures are development data; no new generalization sample count.',
   'B1 only replaces depth-band measured consumption; strict-free with measured-surface veto and future B2 visibility are unchanged.',
   'B2 adds finer0.06m quadrature with exact primitive area conservation; this is isolated from the B1 formula and not silently substituted.',
   'Source-reference actual metrics are reused independently; no future data enters the predictor. Existing geometry-oracle fixture limits remain.'])
 write(HERE/'result.json',result);print(json.dumps(dict(status=result['status'],summary=summary,result_sha256=sha(HERE/'result.json')),default=lambda a:a.item()))
if __name__=='__main__':main()
