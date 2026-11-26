import os
os.environ.update(OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1')
import pathlib,json,hashlib,sys,numpy as np
D=pathlib.Path(__file__).resolve().parent;W=pathlib.Path('/home/ouc/nso_robot/workspaces/nso_hybrid_20261029');sys.path.insert(0,str(W/'nso_bridge'))
from nso_bridge.joint_engine import _MeasuredMapMapper
from utils.rgbd_contract import RGBDFrame
r=json.loads((D/'result.json').read_text());P=pathlib.Path(r['report']);cfg=json.loads((P/'joint_configuration.json').read_text())['configuration'];T=np.array(cfg['planning_from_odom']);p=pathlib.Path(r['first_unsafe_packet'])
with np.load(p,allow_pickle=False) as z:
 f=RGBDFrame(float(z['rgbd_timestamp_s']),z['depth'],z['color_rgb'],z['intrinsic'],T@z['world_from_camera'],np.zeros(z['depth'].shape,dtype=np.int32));pts,_=f.points(stride=2)
 # Native grid_cell: reversed y row, x column; exact shape origin.
 cell=r['final_footprint_cells'][0]['cell'];cells=np.floor(pts[:,:2]/.2).astype(int)[:,::-1];cells[:,0]=cfg['shape'][0]-1-cells[:,0];good=(pts[:,2]>.15)&(pts[:,2]<1.2)&np.all(cells==cell,axis=1);pp=pts[good];cp=(pp-f.world_from_camera[:3,3])@f.world_from_camera[:3,:3];q=cp@f.intrinsic.T;uv=q[:,:2]/q[:,2,None]*4
 assert len(pp)==139
 base=T@np.asarray(json.loads(str(z['receipt_json']))['historical_tf']['odom_from_base']['matrix']);logical=np.array([(r['audited_fixed_current_cell'][1]+.5)*.2,(cfg['shape'][0]-1-r['audited_fixed_current_cell'][0]+.5)*.2]);dist=np.linalg.norm(pp[:,:2]-base[:2,3],axis=1)
 out=dict(actual_base_planning_xyz=base[:3,3].tolist(),logical_cell_center_planning_xy=logical.tolist(),actual_vs_logical_xy_offset_m=float(np.linalg.norm(base[:2,3]-logical)),support_point_actual_base_horizontal_distance_minmax_m=[float(dist.min()),float(dist.max())],closest_support_raw_pixel=uv[int(np.argmin(dist))].tolist(),closest_support_planning_xyz=pp[int(np.argmin(dist))].tolist(),source_packet=str(p),source_sha256=hashlib.sha256(p.read_bytes()).hexdigest(),cell=cell,depth_support_count=len(pp),raw_pixel_u_minmax=[float(uv[:,0].min()),float(uv[:,0].max())],raw_pixel_v_minmax=[float(uv[:,1].min()),float(uv[:,1].max())],planning_height_minmax_m=[float(pp[:,2].min()),float(pp[:,2].max())],optical_depth_minmax_m=[float(cp[:,2].min()),float(cp[:,2].max())],pixel_stride_raw=8,source_rgbd_scan_stamp_ns={k:json.loads(str(z['receipt_json']))['headers'][k]['stamp_ns'] for k in ['rgb','depth','scan']})
 np.savez_compressed(D/'depth_support139.npz',pixel_raw=uv,planning_xyz=pp,camera_xyz=cp)
(D/'support_pixels.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out))
