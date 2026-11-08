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
from adapter import offline_generator,offsets_for_radius
import nso_bridge.exploration_joint_policy as policy
source_observed_patches=rf.observed_patches
def diagnostic(s,variant,scaled=False):
 m=s['mapper'];backend=session.backend;packet=s['packet'];attempt=s['gain'].attempted_camera_mask(m,s['ledger'].planning_camera_poses());saved=[(mod,mod.observed_patches) for mod in (rf,v81,v8)];sc=dict(s)
 try:
  if variant:
   evidence=m.quality_evidence(max_points=10000);keys=quality_keys(m,evidence);patches,_,_=associated_patches(m,evidence,keys,assoc,rf.observed_patches)
   def patched(mapper,*args,**kwargs):
    if mapper is not m: return saved[0][1](mapper,*args,**kwargs)
    return patches
   for mod,_ in saved:mod.observed_patches=patched
   sc['assets']=v81.measured_assets(m)
  generator=offline_generator(generate_options,m.config.robot_radius_m) if scaled else generate_options
  routes,audit=generator(m,packet.position,packet.heading,s['ledger'].remaining_budget,s['return_anchor'],max_candidates=int(getattr(backend.args,'cpu_max_candidates',12)),coverage_strategy='total_diverse',coverage_slots=int(getattr(backend.args,'cpu_coverage_slots',4)),attempted_camera_mask=attempt)
  g=s['graph'];routes=[r for r in routes if g['current_region']>0 and g['current_region']==g['anchor_region']==int(g['regions'][tuple(r['pose'][:2])])]
  scores,sa=backend._scores_v10_1(sc,routes,attempt) if routes else ({'G':[],'S':[],'X':[]},[])
  choice={}
  for mode in ('G','S','X'):
   i=min(range(len(routes)),key=lambda i:(-scores[mode][i],routes[i]['cost'],routes[i]['candidate_id'])) if routes else None
   choice[mode]=None if i is None or scores[mode][i]<=0 else {'candidate_id':routes[i]['candidate_id'],'pose':routes[i]['pose'],'score':scores[mode][i]}
  return dict(candidates=routes,candidate_audit=audit,scores=scores,score_audit=sa,choices=choice,assets_count=len(sc['assets']),offsets=offsets_for_radius(m.config.robot_radius_m) if scaled else policy.VIEW_OFFSETS_M)
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
   if aid in (0,18,33,38):
    state=session.backend.scenes[0];base=diagnostic(state,False,False);recorded=expected.get('selection');assert recorded is not None;assert base['candidate_audit']['belief_sha256']==recorded['candidate_audit']['belief_sha256'];assert base['scores']['G']==recorded['scores']['G'] and base['scores']['S']==recorded['scores']['S'];cells={'native_original':base,'native_scaled':diagnostic(state,False,True),'instance_original':diagnostic(state,True,False),'instance_scaled':diagnostic(state,True,True)};assert policy.VIEW_OFFSETS_M==(('entry',.4),('deep',.8)) and rf.observed_patches is source_observed_patches;out.append({'action_id':aid,'remaining_budget':state['ledger'].remaining_budget,'cells':cells});print('completed actual state',aid,flush=True)
   actual=session.next_action();assert actual['action']==expected['action'],(aid,actual['action'],expected['action'])
  elif method in ('init','shutdown'):continue
  else:raise ValueError(method)
(OUT/'full01_result.json').write_text(json.dumps(json_value({'schema':'nso.clearance_matched_entries_full01_factorial.v1','status':'offline_proposal_not_deployed','states':out,'scope':'Original actual trace replay, cached labels and masks; baseline native selection checked; variant regenerated only at actual states. No hypothetical observations; only partition and/or predeclared radius-shifted entry/deep offsets differ. Original score weights/tolerance/circle/return kept.','limits':['Instance geometry is a proposed method variant, not object identity ground truth.','No physical rollout or semantic effect proven.']}),indent=2)+'\n')
print('states',len(out))
