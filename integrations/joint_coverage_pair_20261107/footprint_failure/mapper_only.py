#!/usr/bin/env python3
"""Bounded33-frame native mapper reconstruction; no session/runtime/planner/ROS."""
import os
os.environ.update(OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1')
import hashlib,inspect,json,math,sys,time
from pathlib import Path
import numpy as np
W=Path('/home/ouc/nso_robot/workspaces/nso_hybrid_20261029')
sys.path.insert(0,str(W/'nso_bridge'))
from nso_bridge.joint_engine import _MeasuredMapMapper
from nso_bridge.sensor_range_adapter import adapt_scan
from env.virtual3d_v2 import VirtualConfigV2
from utils.rgbd_contract import RGBDFrame,PlanarScan
from utils.grid_geometry import inflated_obstacles
from nso.mapping3d import SensorMapper
P=Path('/home/ouc/nso_robot/analyses/joint_coverage_pair_20261107/publication/S')
OUT=Path(__file__).resolve().parent
EXPECTED={31:'9c189bdaed600110b57ab19d74bcf316b6c7dfce21f24608417e329145ae3567',32:'6fe5e81db30afe0cbe1d28fe34f023cee2ca88dde63815f3f3803d1fba37a146'}
def sha(a):return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()
def filesha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
config=json.loads((P/'source/joint_configuration.json').read_text())['configuration']
manifest=json.loads((P/'frames_manifest.json').read_text())['frames']
source=json.loads((P/'source/source_manifest.json').read_text())['files'];source={r['path']:r['sha256'] for r in source}
for rel in ['nso_bridge/nso_bridge/joint_engine.py','nso_bridge/joint_vendor/nso/mapping3d.py','nso_bridge/joint_vendor/nso/observed_runtime_mapper_v10.py']:
 assert filesha(W/rel)==source[rel],rel
sensor=dict(config['sensor_config']);sensor.update(resolution_m=.2,robot_radius_m=config['robot_radius_m'])
c=VirtualConfigV2(**sensor);mapper=_MeasuredMapMapper(tuple(config['shape']),c);transform=np.asarray(config['planning_from_odom']);beliefs={};stages={};last_data={};start=time.monotonic()
mapper.sensor_range_profile=config['sensor_range_profile'];mapper.measured_extent_profile=config['measured_extent_profile']
sensor_lines,offset=inspect.getsourcelines(SensorMapper.update); sensor_path=inspect.getfile(SensorMapper.update)
engine_lines,eng_offset=inspect.getsourcelines(_MeasuredMapMapper.update);engine_path=inspect.getfile(_MeasuredMapMapper.update)
targets={}
for i,line in enumerate(sensor_lines):
 if 'points,labels=frame.points' in line:targets[(sensor_path,offset+i)]='after_scan_before_depth'
 if 'if 0<=origin[0]<' in line:targets[(sensor_path,offset+i)]='after_depth_before_origin_free'
 if 'keys=np.floor(points/.15)' in line:targets[(sensor_path,offset+i)]='after_sensor_origin_free'
for i,line in enumerate(engine_lines):
 if 'self.belief[:] = np.maximum' in line:targets[(engine_path,eng_offset+i)]='after_observed_mapper_before_nav'
 if 'self.visible[:] |=' in line:targets[(engine_path,eng_offset+i)]='after_nav_union'
def trace(frame,event,arg):
 if event=='line':
  key=(frame.f_code.co_filename,frame.f_lineno)
  if key in targets and frame.f_locals.get('self') is mapper:stages[targets[key]]=mapper.belief.copy()
 return trace
for item in manifest:
 aid=item['action_id'];path=P/item['file'];assert filesha(path)==item['sha256']
 with np.load(path,allow_pickle=False) as z:
  frame=RGBDFrame(float(z['rgbd_timestamp_s']),z['depth'].copy(),z['color_rgb'].copy(),z['intrinsic'].copy(),transform@z['world_from_camera'],z['semantic'].copy())
  scan=PlanarScan(float(z['scan_timestamp_s']),z['ranges'].copy(),float(z['angle_min']),float(z['angle_increment']),float(z['range_max']),transform@z['world_from_laser'])
  scan,_=adapt_scan(scan,config['sensor_range_profile']);nav=z['joint_belief_grid'].copy();mapper.input_observed_grid=nav
  if aid==32:last_data=dict(frame=frame,scan=scan,nav=nav,packet_path=str(path),frame_id=item['frame_id'])
  if aid==32:sys.settrace(trace)
  try:mapper.update(frame,scan)
  finally:sys.settrace(None)
  if aid in [30,31,32]:beliefs[aid]=mapper.belief.copy()
  if aid in EXPECTED:assert sha(mapper.belief)==EXPECTED[aid],(aid,sha(mapper.belief))
center=(195,201);threshold=c.robot_radius_m/c.resolution_m+math.sqrt(2)/2
bad=[]
for r,col in np.argwhere(mapper.belief!=0):
 dist=math.hypot(r-center[0],col-center[1])
 if dist<=threshold+1e-12:
  cell=(int(r),int(col));bad.append(dict(cell=list(cell),center_distance_m=dist*c.resolution_m,values={str(k):int(v[cell]) for k,v in beliefs.items()},stages32={k:int(v[cell]) for k,v in stages.items()},nav32=int(last_data['nav'][cell])))
# Original point and laser endpoint functions identify actual insertion support.
frame,scan=last_data['frame'],last_data['scan'];points,labels=frame.points(stride=2)
obstacle=(points[:,2]>.15)&(points[:,2]<1.2)
point_cells=np.array([mapper.grid_cell(p) for p in points],int)
for row in bad:
 cell=tuple(row['cell']);ids=np.flatnonzero(obstacle&np.all(point_cells==cell,axis=1))
 row['frame32_depth_obstacle_points_count']=len(ids)
 row['depth_obstacle_points_sample_planning']=points[ids[:5]].tolist()
 row['depth_obstacle_labels_sample']=labels[ids[:5]].tolist()
 hits=[]
 for i,distance in enumerate(scan.ranges_m):
  if not np.isfinite(distance) or distance<=0 or distance>=scan.range_max_m-1e-3:continue
  a=scan.angle_min_rad+i*scan.angle_increment_rad;local=np.array([np.cos(a)*distance,np.sin(a)*distance,0.]);point=scan.world_from_laser[:3,:3]@local+scan.world_from_laser[:3,3]
  if mapper.grid_cell(point)==cell:hits.append(dict(index=i,range_m=float(distance),planning_point=point.tolist()))
 row['frame32_laser_hit_count']=len(hits);row['laser_hits_sample']=hits[:5]
result=dict(schema='nso.mapper_only_footprint_source.v1',scope='Original33 canonical measured frames; native mapper.update only, no policy/runtime/scorer/inference/ROS',source=str(P),frames=mapper.frames,
 matched_belief_sha256={str(k):sha(beliefs[k]) for k in EXPECTED},expected_belief_sha256=EXPECTED,
 center=list(center),robot_radius_m=c.robot_radius_m,resolution_m=c.resolution_m,original_cell_center_blocking_threshold_m=threshold*c.resolution_m,
 safe_by_action={str(k):bool((~inflated_obstacles(v!=0,c.robot_radius_m/c.resolution_m))[center]) for k,v in beliefs.items()},
 frame32_stage_sha256={k:sha(v) for k,v in stages.items()},nearby_nonfree_blocking_cells=bad,
 frame32_observed_current_cell_conflict=mapper.current_footprint_conflict_details,
 nav32_center_safe=bool((~inflated_obstacles(last_data['nav']!=0,c.robot_radius_m/c.resolution_m))[center]),
 native_source_sha256={rel:filesha(W/rel) for rel in source if rel in ['nso_bridge/nso_bridge/joint_engine.py','nso_bridge/joint_vendor/nso/mapping3d.py','nso_bridge/joint_vendor/nso/observed_runtime_mapper_v10.py']},
 elapsed_s=time.monotonic()-start,limits=['Attribution is to actual native grid insertion, not a verified physical object identity or calibrated reconstruction accuracy.','Camera retained extrinsics remain unverified; no source filtered or runtime corrected.','Occupied cells persist under native fusion; input navigation map safety alone does not imply fused belief safety.'])
(OUT/'mapper_source_result.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
print(json.dumps(result,ensure_ascii=False))
