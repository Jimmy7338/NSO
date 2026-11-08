"""Fixed S0/S6/S13 ROI plane diagnostic; raw depth/K/TF only, no policy/ROS."""
import os
os.environ.update(OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1')
from pathlib import Path
import json,hashlib,math
import numpy as np
P=Path('/home/ouc/nso_robot/reports/nso-hybrid-20261108/fresh_calibrated_pair_20261108_01_S');D=Path(__file__).resolve().parent
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
protocol=dict(action_ids=[0,6,13],crop_uv=[360,480,1100,710],stride=4,seed=0,trials=120,tolerance_m=.01,plane_refinement='SVD least squares on highest-inlier-count RANSAC support; strictcountimprovement, firsttie retained; upward odomnormal',neighborhood_centers_uv=[[656,488],[664,488],[672,488],[528,504],[536,504],[544,504]],neighborhood_offsets=[-8,-4,0,4,8],scope='Same fixed ROI and settings for all3predeclared frames; no choicebased on fit outcome. Floor-dominated ROI is a visual hypothesis, not ground truth.')
(D/'ground_frames_protocol.json').write_text(json.dumps(protocol,ensure_ascii=False,indent=2)+'\n')
rows=[json.loads(l) for l in open(P/'joint/decisions.jsonl')];obs={r['request']['params']['action_id']:r for r in rows if r['request']['method']=='observe' and r['response'].get('ok')};cfg=json.load(open(P/'joint_configuration.json'));Tplan=np.asarray(cfg['configuration']['planning_from_odom']);floor=cfg['calibration']['initial_grid']['floor_odom_z_m']
def project(depth,K,uv):
 uv=np.asarray(uv,int);z=depth[uv[:,1],uv[:,0]];good=np.isfinite(z)&(z>0);rays=np.linalg.inv(K)@np.column_stack([uv[good],np.ones(good.sum())]).T;xyz=(rays*(z[good]/rays[2])).T
 return xyz,uv[good],z[good]
def plane(points):
 rng=np.random.default_rng(0);best=None;bestn=-1
 for _ in range(120):
  q=points[rng.choice(len(points),3,replace=False)];n=np.cross(q[1]-q[0],q[2]-q[0]);length=np.linalg.norm(n)
  if length<1e-12:continue
  n/=length;d=-float(n@q[0]);mask=np.abs(points@n+d)<.01
  if mask.sum()>bestn:bestn=int(mask.sum());best=mask
 if best is None:raise ValueError('no nondegenerate fixedROIplane')
 center=points[best].mean(axis=0);_,_,vt=np.linalg.svd(points[best]-center,full_matrices=False);n=vt[-1]
 if n[2]<0:n=-n
 d=-float(n@center);res=np.abs(points@n+d)
 return n,d,res,best
out=[]
for aid in [0,6,13]:
 r=obs[aid];path=Path(r['request']['params']['path']);receipt=r['response']['result']['summary']['semantic_diagnostics']['receipt'];assert sha(path)==receipt['source_file_sha256']
 with np.load(path,allow_pickle=False) as z:
  depth=z['depth_m'].copy();K=z['intrinsic_raw'].copy();T=z['world_from_camera'].copy();base=z['base_odom_pose'].copy();meta=json.loads(str(z['receipt_json']));stamp=float(z['rgbd_timestamp_s'])
 assert depth.shape==(720,1280)
 uu,vv=np.meshgrid(np.arange(360,1100,4),np.arange(480,710,4));camera,uv,depths=project(depth,K,np.column_stack([uu.ravel(),vv.ravel()]));world=camera@T[:3,:3].T+T[:3,3]
 n,d,res,mask=plane(world);nc=T[:3,:3].T@n;dc=float(n@T[:3,3]+d);nbase=np.array([[math.cos(base[2]),math.sin(base[2]),0],[-math.sin(base[2]),math.cos(base[2]),0],[0,0,1]])@n
 zorigin=-d/n[2];zbase=-(n[0]*base[0]+n[1]*base[1]+d)/n[2];zcam=-(n[0]*T[0,3]+n[1]*T[1,3]+d)/n[2]
 neighbors=[]
 for u,v in protocol['neighborhood_centers_uv']:
  grid=np.array([(u+du,v+dv) for dv in [-8,-4,0,4,8] for du in [-8,-4,0,4,8]]);pc,pix,zs=project(depth,K,grid);pw=pc@T[:3,:3].T+T[:3,3];pp=pw@Tplan[:3,:3].T+Tplan[:3,3];center=np.flatnonzero(np.all(pix==[u,v],axis=1));i=int(center[0]) if len(center) else None
  neighbors.append(dict(center_uv=[u,v],valid_depth_neighbors=len(pc),depth_min_m=float(zs.min()) if len(zs) else None,depth_max_m=float(zs.max()) if len(zs) else None,median_abs_plane_residual_m=float(np.median(np.abs(pw@n+d))) if len(pc) else None,planning_z_min_m=float(pp[:,2].min()) if len(pc) else None,planning_z_max_m=float(pp[:,2].max()) if len(pc) else None,above_original_height_threshold_count=int(np.count_nonzero((pp[:,2]>.15)&(pp[:,2]<1.2))),center_depth_m=None if i is None else float(zs[i]),center_planning_xyz=None if i is None else pp[i].tolist()))
 row=dict(action_id=aid,source_packet=str(path),source_sha256=sha(path),RGBD_stamp_s=stamp,base_odom_xy_yaw=base.tolist(),raw_depth_shape=list(depth.shape),raw_intrinsic=K.tolist(),odom_from_camera=T.tolist(),ROI_valid_points=len(camera),ransac_support_count=int(mask.sum()),inlier_fraction_after_refinement=float(np.mean(res<.01)),median_abs_residual_m=float(np.median(res)),p95_abs_residual_m=float(np.percentile(res,95)),plane_normal_odom=n.tolist(),plane_d_odom_m=d,plane_z_at_odom_origin_m=float(zorigin),plane_z_at_current_base_xy_m=float(zbase),apparent_offset_from_declared_floor_at_base_m=float(zbase-floor),declared_floor_odom_z_m=floor,slope_deg=float(np.rad2deg(np.arccos(np.clip(n[2],-1,1)))),plane_normal_base_yaw_frame=nbase.tolist(),plane_normal_camera=nc.tolist(),plane_d_camera_m=dc,camera_perpendicular_distance_to_plane_m=abs(dc),camera_vertical_height_above_plane_m=float(T[2,3]-zcam),camera_vertical_height_from_declared_floor_m=float(T[2,3]-floor),pixel_neighborhoods=neighbors)
 out.append(row)
# Explicit reproduction of root S13 fixedROI result, not an independent measurement.
root=json.load(open(D/'floor_projection_diagnostic.json'));s13=out[-1];comparison=dict(normal_max_abs_difference=float(np.max(np.abs(np.asarray(root['plane_normal_odom'])-s13['plane_normal_odom']))),d_abs_difference_m=abs(root['plane_d_m']-s13['plane_d_odom_m']),median_residual_abs_difference_m=abs(root['median_residual_m']-s13['median_abs_residual_m']))
summary=dict(schema='nso.fixed3frame_ground_projection.v1',protocol=protocol,frames=out,S13_root_reproduction=comparison,script_sha256=sha(__file__),no_runtime_change=True,no_thesis_change=True,limits=['Depth/K/TF-derived plane in a visually floor-dominated ROI is not independently measured ground truth.','The z at odomorigin is an extrapolation; local base/camera heights are reported separately.','Camera-relative plane distance is derived from same depthsensor, not an external calibration reference.','A stable plane does not identify which extrinsic, base mounting, fixedflooroffset or depthmodel is physically correct.','FixedUV neighborhoods acrossframes do not necessarily view the same physical ground patch.'])
(D/'ground_frames_result.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2)+'\n')
print(json.dumps(dict(frames=[{k:r[k] for k in ['action_id','ROI_valid_points','inlier_fraction_after_refinement','median_abs_residual_m','plane_normal_odom','plane_z_at_odom_origin_m','plane_z_at_current_base_xy_m','apparent_offset_from_declared_floor_at_base_m','camera_perpendicular_distance_to_plane_m','camera_vertical_height_above_plane_m']} for r in out],S13_root_reproduction=comparison),ensure_ascii=False))
