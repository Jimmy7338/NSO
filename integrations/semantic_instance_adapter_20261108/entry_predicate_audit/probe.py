#!/usr/bin/env python3
"""Recorded-state baseline/instance partition native candidate/scorer probe. Offline only."""
import os
os.environ.update(OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1')
import collections,hashlib,json,pathlib,sys,tempfile,copy
import numpy as np
W=pathlib.Path('/home/ouc/nso_robot/workspaces/nso_hybrid_20261029');sys.path.insert(0,str(W/'nso_bridge'));sys.path.insert(0,str(pathlib.Path(__file__).resolve().parents[1]));from association import MeasuredAssociation,quality_samples,quality_keys,associated_patches
from nso_bridge.joint_engine import JointSession
from nso_bridge.exploration_joint_policy import generate_options
from nso.cpu_sensor_contract_v10 import json_value
from utils.rgbd_contract import RGBDFrame
import nso.response_features_v7 as rf
import nso.competition_candidates_v8_1 as v81
import nso.competition_candidates_v8 as v8
REPORT=pathlib.Path('/home/ouc/nso_robot/reports/nso-hybrid-20261108/depth_ground_S40_20261108_01');OUT=pathlib.Path(__file__).resolve().parent
rows=[json.loads(x) for x in (REPORT/'joint/decisions.jsonl').read_text().splitlines()];cfg=json.loads((REPORT/'joint_configuration.json').read_text());cfg['execution_allowed']=False;cfg.pop('semantic_configuration',None);session=JointSession(**cfg);assoc=MeasuredAssociation();out=[];last_action=None
sha=lambda p:hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest()

def entry_trace(mapper,packet,s,routes,assets):
 from scipy.sparse.csgraph import dijkstra
 from utils.grid_geometry import inflated_obstacles
 from nso.route_coverage_v2 import orientation_graph
 from env.virtual3d import camera_pose
 from nso.competition_candidates_v8_1 import XY_DIRECTIONS,aperture_support,navigation_rear_boundary
 safe=~inflated_obstacles(mapper.belief!=0,mapper.config.robot_radius_m/mapper.config.resolution_m);graph,cells,ids=orientation_graph(safe);start=int(ids[packet.position])*4+packet.heading;anchor=s['return_anchor'];home=int(ids[tuple(anchor[:2])])*4+anchor[2];a=dijkstra(graph,directed=True,indices=start);b=dijkstra(graph.T.tocsr(),directed=True,indices=home);cost=a+b;reachable=np.isfinite(cost)&(a>0);budget=s['ledger'].remaining_budget;used={tuple(x['pose']) for x in routes if x['group'].startswith('coverage_') or x['group']=='old_surface_rotation'};details=[];need80=False
 for i,asset in enumerate(assets):
  wanted=int(np.argmax(XY_DIRECTIONS@asset['front_axis']));desired=navigation_rear_boundary(mapper,asset)+.4*asset['back_axis'];counts={'safe_oriented':len(cost),'reachable_paid_return':int(reachable.sum()),'wanted_heading':0,'backside':0,'aperture_positive':0,'within_budget':0,'unused':0,'projection_within_035':0};min_any=None;min_eligible=None;sample=None
  for node in np.flatnonzero(reachable):
   if node%4!=wanted:continue
   counts['wanted_heading']+=1;cell=tuple(map(int,cells[node//4]));pose=(*cell,wanted);cam=camera_pose(cell,wanted,mapper.config,mapper.shape[0]);depth=float((cam[:2,3]-asset['rear_boundary_xy'])@asset['back_axis'])
   if depth<=.15:continue
   counts['backside']+=1;ap=float(aperture_support(asset,cam,mapper.config).max())
   if ap<=0:continue
   counts['aperture_positive']+=1;full=float(cost[node]);err=float(np.linalg.norm(cam[:2,3]-desired));min_any=full if min_any is None else min(min_any,full)
   if pose not in used and err<=.35:
    if min_eligible is None or full<min_eligible:min_eligible=full;sample={'pose':pose,'cost':full,'projection_error_m':err,'aperture_max':ap}
   if full>budget:continue
   counts['within_budget']+=1
   if pose in used:continue
   counts['unused']+=1
   if err>.35:continue
   counts['projection_within_035']+=1
  possible=bool(min_any is not None and budget<min_any<=80-packet.action_id);need80|=possible
  details.append({'asset_index':i,'group':int(asset['group']),'marked_points':int(asset['marked_points']),'front_axis':asset['front_axis'].tolist(),'rear_boundary_xy':asset['rear_boundary_xy'].tolist(),'wanted_heading':wanted,'counts':counts,'min_return_feasible_aperture_cost':min_any,'min_unused_projected_entry_cost':min_eligible,'min_unused_projected_entry_sample':sample,'aperture_pose_beyond_budget_within_total80':possible})
 return {'paid':int(packet.action_id),'actual_remaining_budget':int(budget),'total80_remaining':80-int(packet.action_id),'used_common_poses':sorted(used),'assets':details,'single_total80_hypothesis_warranted':need80,'scope':'Counts are original safe graph and native backside/FOV support/heading/projection. FOV is not an occlusion certificate. Duplicate approximated by actual common poses; no entry was selected in recorded cases.'}

def diagnostic(s,variant):
 m=s['mapper'];backend=session.backend;packet=s['packet'];attempt=s['gain'].attempted_camera_mask(m,s['ledger'].planning_camera_poses());saved=[(mod,mod.observed_patches) for mod in (rf,v81,v8)];sc=dict(s)
 try:
  if variant:
   evidence=m.quality_evidence(max_points=10000);keys=quality_keys(m,evidence);patches,_,_=associated_patches(m,evidence,keys,assoc,rf.observed_patches)
   def patched(mapper,*args,**kwargs):
    if mapper is not m: return saved[0][1](mapper,*args,**kwargs)
    return patches
   for mod,_ in saved:mod.observed_patches=patched
   sc['assets']=v81.measured_assets(m)
  routes,audit=generate_options(m,packet.position,packet.heading,s['ledger'].remaining_budget,s['return_anchor'],max_candidates=int(getattr(backend.args,'cpu_max_candidates',12)),coverage_strategy='total_diverse',coverage_slots=int(getattr(backend.args,'cpu_coverage_slots',4)),attempted_camera_mask=attempt)
  g=s['graph'];routes=[r for r in routes if g['current_region']>0 and g['current_region']==g['anchor_region']==int(g['regions'][tuple(r['pose'][:2])])]
  scores,sa=backend._scores_v10_1(sc,routes,attempt) if routes else ({'G':[],'S':[],'X':[]},[])
  choice={}
  for mode in ('G','S','X'):
   i=min(range(len(routes)),key=lambda i:(-scores[mode][i],routes[i]['cost'],routes[i]['candidate_id'])) if routes else None
   choice[mode]=None if i is None or scores[mode][i]<=0 else {'candidate_id':routes[i]['candidate_id'],'pose':routes[i]['pose'],'score':scores[mode][i]}
  detail=entry_trace(m,packet,s,routes,sc['assets']) if variant else None
  resource80=None
  if detail and detail['single_total80_hypothesis_warranted']:
   rr,aa=generate_options(m,packet.position,packet.heading,80-int(packet.action_id),s['return_anchor'],max_candidates=int(getattr(backend.args,'cpu_max_candidates',12)),coverage_strategy='total_diverse',coverage_slots=int(getattr(backend.args,'cpu_coverage_slots',4)),attempted_camera_mask=attempt)
   resource80={'scope':'single predeclared total80 resource counterfactual; no new observations','remaining':80-int(packet.action_id),'candidate_count':len(rr),'candidate_audit':aa}
  return dict(resource80=resource80,candidates=routes,candidate_audit=audit,scores=scores,score_audit=sa,choices=choice,assets_count=len(sc['assets']),entry_detail=detail)
 finally:
  for mod,original in saved:mod.observed_patches=original
with tempfile.TemporaryDirectory(prefix='nso_instance_scoring_') as tmp:
 for row in rows:
  method=row['request']['method'];expected=row['response'].get('result')
  if method=='observe':
   params=dict(row['request']['params']);aid=params['action_id'];path=pathlib.Path(params['path']);receipt=expected['summary']['semantic_diagnostics']['receipt'];mask=pathlib.Path(receipt['mask_file']);assert sha(path)==receipt['source_file_sha256'] and sha(mask)==receipt['mask_file_sha256']
   with np.load(path,allow_pickle=False) as z,np.load(mask,allow_pickle=False) as masks:
    data={k:z[k].copy() for k in z.files if k not in ('rgb_received_ros_serialized','depth_received_ros_serialized','camera_info_received_ros_serialized','scan_received_ros_serialized')};labels=masks['semantic'].copy();valid=np.isfinite(z['depth_m'])&(z['depth_m']>0)&(z['depth_m']<=session.config.max_depth_m);owner=np.zeros(valid.shape,np.int64);owned=np.zeros(valid.shape,bool)
    for i in sorted(range(len(receipt['detections'])),key=lambda i:(-receipt['detections'][i]['detector_score'],i)):
     det=receipt['detections'][i];mask_i=masks[f'mask_{i}'];assert hashlib.sha256(mask_i.tobytes()).hexdigest()==det['mask_sha256'];measured=mask_i&valid;assign=measured&~owned;owner[assign]=i+1;owned|=measured
    frame=RGBDFrame(float(z['rgbd_timestamp_s']),z['depth'].copy(),z['color_rgb'].copy(),z['intrinsic'].copy(),np.asarray(cfg['configuration']['planning_from_odom'])@z['world_from_camera'],labels);keys,owners,cat=quality_samples(frame,owner[::receipt['pixel_stride'],::receipt['pixel_stride']]);assoc.observe(receipt['frame_id'],keys,owners,cat);data['semantic']=labels
   tmp_path=pathlib.Path(tmp)/'packet.npz';np.savez_compressed(tmp_path,**data);params['path']=str(tmp_path);res=session.observe(**params);assert res['accepted'] and res['state']['pose']==expected['state']['pose'];
  elif method=='next_action':
   if aid in (0,18):
    state=session.backend.scenes[0];base=diagnostic(state,False);variant=diagnostic(state,True);recorded=expected.get('selection');assert recorded is not None;assert base['candidate_audit']['belief_sha256']==recorded['candidate_audit']['belief_sha256'];assert base['scores']['G']==recorded['scores']['G'] and base['scores']['S']==recorded['scores']['S'];out.append({'action_id':aid,'baseline':base,'instance_variant':variant})
   actual=session.next_action();assert actual['action']==expected['action'],(aid,actual['action'],expected['action'])
   if aid==18:break
  elif method in ('init','shutdown'):continue
  else:raise ValueError(method)
(OUT/'entry_predicate_audit.json').write_text(json.dumps(json_value({'schema':'nso.instance_partition_scoring_probe.v1','status':'offline_proposal_not_deployed','states':out,'scope':'Original actual trace replay, cached labels and masks; baseline native selection checked; variant regenerated only at actual states. No hypothetical observations or changed thresholds/weights.','limits':['Instance geometry is a proposed method variant, not object identity ground truth.','No physical rollout or semantic effect proven.']}),indent=2)+'\n')
print('states',len(out))
