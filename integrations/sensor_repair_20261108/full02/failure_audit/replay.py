"""One20-frame native mapper-only attribution; no policy, detector or ROS."""
import os
os.environ.update(OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1')
import json,sys,hashlib,inspect,math,time
from pathlib import Path
import numpy as np
W=Path('/home/ouc/nso_robot/workspaces/nso_hybrid_20261029');sys.path.insert(0,str(W/'nso_bridge'))
from nso_bridge.joint_engine import _MeasuredMapMapper
from nso_bridge.sensor_range_adapter import adapt_scan
from nso_bridge.measured_extent_adapter import install_dispatcher
from env.virtual3d_v2 import VirtualConfigV2
from utils.rgbd_contract import RGBDFrame,PlanarScan
from utils.grid_geometry import inflated_obstacles
from nso.mapping3d import SensorMapper
P=Path('/home/ouc/nso_robot/reports/nso-hybrid-20261108/depth_ground_S40_20261108_02');OUT=Path(__file__).resolve().parent
sha=lambda a:hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()
fsha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
cfg=json.load(open(P/'joint_configuration.json'))['configuration'];source={r['path']:r['sha256'] for r in json.load(open(P/'source_manifest.json'))['files']}
rels=['nso_bridge/nso_bridge/joint_engine.py','nso_bridge/joint_vendor/nso/mapping3d.py','nso_bridge/joint_vendor/nso/observed_runtime_mapper_v10.py','nso_bridge/nso_bridge/sensor_range_adapter.py']
for rel in rels:assert fsha(W/rel)==source[rel],rel
rows=[json.loads(l) for l in open(P/'joint/decisions.jsonl')];obs=[r for r in rows if r['request']['method']=='observe' and r['response'].get('ok')];assert [r['request']['params']['action_id'] for r in obs]==list(range(20))
selections={}
for r in rows:
 if r['request']['method']=='observe':aid=r['request']['params']['action_id']
 if r['request']['method']=='next_action' and r['response']['result'].get('selection'):
  audit=r['response']['result']['selection']['candidate_audit']
  if audit['remaining_budget']==cfg['total_budget']-aid:selections[aid]=audit
sensor=dict(cfg['sensor_config']);sensor.update(resolution_m=.2,robot_radius_m=cfg['robot_radius_m']);c=VirtualConfigV2(**sensor);mapper=_MeasuredMapMapper(tuple(cfg['shape']),c);mapper.sensor_range_profile=cfg['sensor_range_profile'];mapper.measured_extent_profile=cfg['measured_extent_profile'];install_dispatcher();T=np.array(cfg['planning_from_odom']);original_anchor=tuple(cfg['return_anchor'][:2]);anchor=tuple(obs[-1]['response']['result']['state']['pose'][:2]);start=time.monotonic();history={};stages={};inputs={};hashes={}
targets={}
for method in [SensorMapper.update,_MeasuredMapMapper.update]:
 lines,offset=inspect.getsourcelines(method);path=inspect.getfile(method)
 for i,line in enumerate(lines):
  for needle,name in [('points,labels=frame.points','after_scan_before_depth'),('if 0<=origin[0]<','after_depth_before_origin_free'),('keys=np.floor(points/.15)','after_sensor_origin_free'),('self.belief[:] = np.maximum','after_observed_mapper_before_nav'),('self.visible[:] |=','after_nav_union')]:
   if needle in line:targets[(path,offset+i)]=name
current_stage={}
def trace(frame,event,arg):
 if event=='line' and frame.f_locals.get('self') is mapper:
  name=targets.get((frame.f_code.co_filename,frame.f_lineno))
  if name:current_stage[name]=mapper.belief.copy()
 return trace
for record in obs:
 aid=record['request']['params']['action_id'];path=Path(record['request']['params']['path']);rec=record['response']['result']['summary']['semantic_diagnostics']['receipt'];assert fsha(path)==rec['source_file_sha256']
 mask=Path(rec['mask_file']);assert fsha(mask)==rec['mask_file_sha256']
 with np.load(mask,allow_pickle=False) as z:semantic=z['semantic'].copy()
 assert sha(semantic)==rec['canonical_semantic_sha256']
 with np.load(path,allow_pickle=False) as z:
  frame=RGBDFrame(float(z['rgbd_timestamp_s']),z['depth'].copy(),z['color_rgb'].copy(),z['intrinsic'].copy(),T@z['world_from_camera'],semantic)
  scan=PlanarScan(float(z['scan_timestamp_s']),z['ranges'].copy(),float(z['angle_min']),float(z['angle_increment']),float(z['range_max']),T@z['world_from_laser']);scan,_=adapt_scan(scan,cfg['sensor_range_profile']);nav=z['joint_belief_grid'].copy();mapper.input_observed_grid=nav
 current_stage={};sys.settrace(trace)
 try:mapper.update(frame,scan)
 finally:sys.settrace(None)
 history[aid]=mapper.belief.copy();stages[aid]=current_stage;inputs[aid]=dict(frame=frame,scan=scan,nav=nav,packet=str(path));hashes[aid]=sha(mapper.belief)
 if aid in selections:assert hashes[aid]==selections[aid]['belief_sha256'],(aid,hashes[aid],selections[aid]['belief_sha256'])
safe=lambda b:~inflated_obstacles(b!=0,c.robot_radius_m/c.resolution_m)
status=[dict(action_id=a,anchor_safe=bool(safe(b)[anchor]),navigation_anchor_safe=bool(safe(inputs[a]['nav'])[anchor])) for a,b in history.items()];first=next((r['action_id'] for r in status if not r['anchor_safe']),None);assert first is not None
threshold=c.robot_radius_m/c.resolution_m+math.sqrt(.5);bad=[]
for row,col in np.argwhere(history[first]!=0):
 if math.hypot(row-anchor[0],col-anchor[1])<=threshold+1e-12:
  cell=(int(row),int(col));aid=first;frame=inputs[aid]['frame'];scan=inputs[aid]['scan'];points,labels=frame.points(stride=2);cells=np.array([mapper.grid_cell(p) for p in points]);support=np.flatnonzero((points[:,2]>.15)&(points[:,2]<1.2)&np.all(cells==cell,axis=1));hits=[]
  for i,d in enumerate(scan.ranges_m):
   if not np.isfinite(d) or d<=0 or d>=scan.range_max_m-1e-3:continue
   angle=scan.angle_min_rad+i*scan.angle_increment_rad;p=scan.world_from_laser[:3,:3]@np.array([np.cos(angle)*d,np.sin(angle)*d,0.])+scan.world_from_laser[:3,3]
   if mapper.grid_cell(p)==cell:hits.append(dict(index=i,range_m=float(d),planning_point=p.tolist()))
  bad.append(dict(cell=list(cell),anchor_center_distance_m=math.dist(cell,anchor)*c.resolution_m,values_by_action={str(a):int(b[cell]) for a,b in history.items()},stages_at_first_unsafe={k:int(v[cell]) for k,v in stages[aid].items()},nav_at_first_unsafe=int(inputs[aid]['nav'][cell]),depth_obstacle_support_count=len(support),depth_points_sample=points[support[:3]].tolist(),laser_hits=hits))
result=dict(schema='nso.full02_current_footprint_source.v1',scope='One native20frame mapper-only reconstruction with exact cached semantic arrays; no fullpolicy/detector/ROS or source change',report=str(P),frames=mapper.frames,anchor=list(anchor),radius_m=c.robot_radius_m,first_unsafe_action=first,status_by_action=status,belief_sha256_by_action=hashes,logged_hash_matches=list(selections),blocking_cells=bad,first_unsafe_packet=inputs[first]['packet'],first_unsafe_RGBD_stamp_s=inputs[first]['frame'].timestamp_s,first_unsafe_scan_stamp_s=inputs[first]['scan'].timestamp_s,current_discrete_state_at_failure=obs[-1]['response']['result']['state']['pose'],source_sha256={r:fsha(W/r) for r in rels},elapsed_s=time.monotonic()-start,limits=['Insertion source attribution does not identify a physical object, sensor noise cause or localization accuracy.','Raw ring preserved separately; this result uses the exact accepted historical packet, not later scans.','No clearing, filter, geometry, scoring or threshold changes.'])
(OUT/'result.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n');print(json.dumps(dict(first_unsafe_action=first,blocking_cells=len(bad),hash_matches=list(selections),elapsed_s=result['elapsed_s'])))

# Compact support trace at final occupied circular-footprint cells, retaining all modalities.
final=19; extra=[]
for row,col in np.argwhere(history[final]!=0):
 if math.hypot(row-anchor[0],col-anchor[1])>threshold+1e-12:continue
 cell=(int(row),int(col));records=[]
 for aid in range(20):
  if aid not in (18,19) and (aid==0 or history[aid][cell]==history[aid-1][cell]):continue
  frame=inputs[aid]['frame'];scan=inputs[aid]['scan'];points,labels=frame.points(stride=2);cells=np.floor(points[:,:2]/c.resolution_m).astype(int)[:,::-1];cells[:,0]+=cfg['shape'][0]//2;cells[:,1]+=cfg['shape'][1]//2
  # Native grid_cell convention is used explicitly, avoiding guessed grid origin.
  support=np.flatnonzero((points[:,2]>.15)&(points[:,2]<1.2)&np.array([mapper.grid_cell(p)==cell for p in points]));hits=[]
  for j,d in enumerate(scan.ranges_m):
   if not np.isfinite(d) or d<=0 or d>=scan.range_max_m-1e-3:continue
   angle=scan.angle_min_rad+j*scan.angle_increment_rad;p=scan.world_from_laser[:3,:3]@np.array([np.cos(angle)*d,np.sin(angle)*d,0.])+scan.world_from_laser[:3,3]
   if mapper.grid_cell(p)==cell:hits.append(dict(index=j,range_m=float(d),planning_point=p.tolist()))
  # Transform depth-supported points back to sampled camera pixels; raw is 4x sampled coordinates.
  cp=(points[support]-frame.world_from_camera[:3,3])@frame.world_from_camera[:3,:3];uvh=cp@frame.intrinsic.T;uv=uvh[:,:2]/uvh[:,2,None]
  records.append(dict(action_id=aid,value=int(history[aid][cell]),stages={k:int(v[cell]) for k,v in stages[aid].items()},nav=int(inputs[aid]['nav'][cell]),depth_support_n=len(support),depth_support_sample_planning_xyz=points[support[:6]].tolist(),depth_support_raw_pixel_sample=(uv[:6]*4).tolist(),laser_hits=hits))
 extra.append(dict(cell=list(cell),distance_to_current_center_m=math.dist(cell,anchor)*c.resolution_m,value18=int(history[18][cell]),value19=int(history[19][cell]),records=records))
result['final_footprint_cells']=extra;result['actual_return_anchor']=list(original_anchor);result['audited_fixed_current_cell']=list(anchor)
result['script_sha256']=fsha(Path(__file__));(OUT/'result.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
print(json.dumps(dict(final_bad_cells=len(extra),transitions=[dict(cell=x['cell'],before=x['value18'],after=x['value19']) for x in extra])))
