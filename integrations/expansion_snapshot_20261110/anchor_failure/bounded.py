import os
os.environ.update(OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1')
import sys,json,math,hashlib
from pathlib import Path
import numpy as np
W=Path('/home/ouc/nso_robot/workspaces/nso_hybrid_20261029');sys.path.insert(0,str(W/'nso_bridge'))
from nso_bridge.joint_engine import _MeasuredMapMapper
from nso_bridge.sensor_range_adapter import adapt_scan
sys.path.insert(0,str(W/'nso_bridge/joint_vendor'))
from env.virtual3d_v2 import VirtualConfigV2
from utils.rgbd_contract import RGBDFrame,PlanarScan
P=Path('/home/ouc/nso_robot/reports/nso-hybrid-20261110/expand_history_S96_20261110_01');O=Path(__file__).parent
cfg=json.loads((P/'joint_configuration.json').read_text())['configuration'];c=VirtualConfigV2(**dict(cfg['sensor_config'],resolution_m=.2,robot_radius_m=cfg['robot_radius_m']));T=np.array(cfg['planning_from_odom']);m=_MeasuredMapMapper(tuple(cfg['shape']),c);m.sensor_range_profile=cfg['sensor_range_profile'];m.measured_extent_profile=cfg['measured_extent_profile']
results=[]
for aid,f in [(60,'1794292115971439305'),(61,'1794292130668983305')]:
 z=np.load(P/'joint_inputs'/('joint-'+f+'.npz'));frame=RGBDFrame(float(z['rgbd_timestamp_s']),z['depth'],z['color_rgb'],z['intrinsic'],T@z['world_from_camera'],np.zeros(z['depth'].shape,np.int32));scan=PlanarScan(float(z['scan_timestamp_s']),z['ranges'],float(z['angle_min']),float(z['angle_increment']),float(z['range_max']),T@z['world_from_laser']);scan,_=adapt_scan(scan,cfg['sensor_range_profile']);m.input_observed_grid=z['joint_belief_grid'];m.update(frame,scan)
 center=(200,200);bad=[];pts,_=frame.points(stride=2);pc=np.array([m.grid_cell(p) for p in pts]);v,u=np.mgrid[0:180:2,0:320:2];uv=np.stack([u,v],-1)[z['depth'][::2,::2]>0]
 for r,co in np.argwhere(m.belief!=0):
  dist=math.hypot(r-center[0],co-center[1])*.2
  if dist>c.robot_radius_m+.2/math.sqrt(2)+1e-12:continue
  cell=(int(r),int(co));ids=np.flatnonzero(np.all(pc==cell,axis=1)&(pts[:,2]>.15)&(pts[:,2]<1.2));hits=[]
  for i,d in enumerate(scan.ranges_m):
   if not np.isfinite(d) or d<=0 or d>=scan.range_max_m-1e-3:continue
   a=scan.angle_min_rad+i*scan.angle_increment_rad;p=scan.world_from_laser[:3,:3]@np.array([np.cos(a)*d,np.sin(a)*d,0])+scan.world_from_laser[:3,3]
   if m.grid_cell(p)==cell:hits.append(dict(beam=i,range_m=float(d),point=p.tolist()))
  bad.append(dict(cell=cell,distance_m=dist,value=int(m.belief[cell]),nav=int(z['joint_belief_grid'][cell]),depth_count=len(ids),depth_points=pts[ids[:20]].tolist(),depth_uv=uv[ids[:20]].tolist(),scan_hits=hits))
 results.append(dict(action=aid,frame=f,bad=bad,conflict=m.current_footprint_conflict_details))
(O/'bounded.json').write_text(json.dumps(results,indent=2));print(json.dumps(results))
