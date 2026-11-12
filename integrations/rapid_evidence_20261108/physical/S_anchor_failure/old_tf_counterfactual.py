"""Same fixed raw3frames/ROI under oldmount TF; no actual map or runtime change."""
import os
os.environ.update(OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1')
from pathlib import Path
import json,hashlib,xml.etree.ElementTree as ET
import numpy as np
D=Path(__file__).resolve().parent;old=Path('/home/ouc/nso_robot/analyses/calibrated_mount_20261108/hybrid_camera_neural.before.launch');actual=json.load(open(D/'ground_frames_result.json'))
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
args={r.attrib['name']:r.attrib.get('value',r.attrib.get('default')) for r in ET.parse(old).getroot().iter('arg') if r.attrib.get('name','').startswith('cam_')}
values={k:float(args[k]) for k in ['cam_pos_x','cam_pos_y','cam_pos_z','cam_roll','cam_pitch','cam_yaw']};assert [values[k] for k in values]==[.071,-.0009,.288,0.,0.,0.]
# Actual resolved ZED2i macro: mount->center z=.015; left xyz=-.01,+.06,0;
# leftoptical rpy=-pi/2,0,-pi/2. Both workspace macros were hash-identical.
Ropt=np.array([[0.,0.,1.],[-1.,0.,0.],[0.,-1.,0.]])
base_from_oldcamera=np.eye(4);base_from_oldcamera[:3,:3]=Ropt;base_from_oldcamera[:3,3]=[.061,.0591,.303]
protocol=json.load(open(D/'ground_frames_protocol.json'));out=[]
def plane(points):
 rng=np.random.default_rng(0);best=None;count=-1
 for _ in range(120):
  q=points[rng.choice(len(points),3,replace=False)];n=np.cross(q[1]-q[0],q[2]-q[0]);norm=np.linalg.norm(n)
  if norm<1e-12:continue
  n/=norm;d=-float(n@q[0]);mask=np.abs(points@n+d)<.01
  if mask.sum()>count:best=mask;count=int(mask.sum())
 center=points[best].mean(axis=0);_,_,v=np.linalg.svd(points[best]-center,full_matrices=False);n=v[-1]
 if n[2]<0:n=-n
 d=-float(n@center);res=np.abs(points@n+d)
 return n,d,res
for existing in actual['frames']:
 path=Path(existing['source_packet']);assert sha(path)==existing['source_sha256']
 with np.load(path,allow_pickle=False) as z:
  depth=z['depth_m'];K=z['intrinsic_raw'];receipt=json.loads(str(z['receipt_json']));odom_from_base=np.array(receipt['historical_tf']['odom_from_base']['matrix']);base=z['base_odom_pose'];meta=json.loads(str(z['joint_metadata_json']));planning=np.array(meta['planning_from_odom'])
 uu,vv=np.meshgrid(np.arange(360,1100,4),np.arange(480,710,4));uv=np.column_stack([uu.ravel(),vv.ravel()]);zs=depth[uv[:,1],uv[:,0]];good=np.isfinite(zs)&(zs>0);uv=uv[good];zs=zs[good];rays=np.linalg.inv(K)@np.column_stack([uv,np.ones(len(uv))]).T;camera=(rays*(zs/rays[2])).T
 T=odom_from_base@base_from_oldcamera;world=camera@T[:3,:3].T+T[:3,3];n,d,res=plane(world);zbase=-(n[0]*base[0]+n[1]*base[1]+d)/n[2];zorigin=-d/n[2];zcam=-(n[0]*T[0,3]+n[1]*T[1,3]+d)/n[2];dc=float(n@T[:3,3]+d);pixel_z=[]
 for u,v in protocol['neighborhood_centers_uv']:
  z=depth[v,u];ray=np.linalg.inv(K)@[u,v,1.];pc=ray*(z/ray[2]);pw=T[:3,:3]@pc+T[:3,3];pp=planning@np.r_[pw,1.];pixel_z.append(float(pp[2]))
 out.append(dict(action_id=existing['action_id'],source_sha256=sha(path),points=len(camera),old_odom_from_camera=T.tolist(),old_plane_normal_odom=n.tolist(),old_plane_z_at_origin_m=float(zorigin),old_plane_z_at_current_base_xy_m=float(zbase),old_apparent_offset_from_floor_at_base_m=float(zbase+.233),old_plane_slope_deg=float(np.rad2deg(np.arccos(np.clip(n[2],-1,1)))),old_camera_vertical_height_above_plane_m=float(T[2,3]-zcam),old_camera_perpendicular_distance_to_plane_m=abs(dc),old_camera_height_from_declared_floor_m=float(T[2,3]+.233),old_inlier_fraction=float(np.mean(res<.01)),old_median_residual_m=float(np.median(res)),old_six_fixed_pixel_planning_z_m=pixel_z,new_plane_normal_odom=existing['plane_normal_odom'],new_plane_z_at_current_base_xy_m=existing['plane_z_at_current_base_xy_m'],new_camera_perpendicular_distance_to_plane_m=existing['camera_perpendicular_distance_to_plane_m'],perpendicular_distance_invariance_error_m=abs(abs(dc)-existing['camera_perpendicular_distance_to_plane_m'])))
result=dict(schema='nso.old_mount_same_raw_counterfactual.v1',scope='Same originalS0/S6/S13rawdepth/K, fixedROI/seed/trials/tolerance; oldmountTF only, no live execution/TSDF reconstruction/threshold change',old_launch=str(old),old_launch_sha256=sha(old),old_launch_mount=values,old_base_from_left_camera_optical=base_from_oldcamera.tolist(),ZED_macro_path='/home/ouc/scout_nav/src/zed-ros/zed-ros-wrapper/zed_wrapper/urdf/zed_macro.urdf.xacro',ZED_macro_sha256=sha('/home/ouc/scout_nav/src/zed-ros/zed-ros-wrapper/zed_wrapper/urdf/zed_macro.urdf.xacro'),protocol_sha256=sha(D/'ground_frames_protocol.json'),frames=out,script_sha256=sha(__file__),limits=['Counterfactual reprojection is not an actual oldTF map build or evidence that old calibration is physically more accurate.','A rigid TF change preserves camera-to-plane perpendicular distance; fixedfloororigin inconsistency cannot be resolved by a visually flatter reprojection alone.','Actualheight/extrinsicaccuracy requires independent physicalmeasurement.','No scientificpolicy or runtime was changed by this script.'])
(D/'old_tf_counterfactual_result.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n');print(json.dumps(out,ensure_ascii=False))
