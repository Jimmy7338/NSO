#!/usr/bin/env python3
"""Cross-check recorded route factors using the preserved original geometry functions."""
import argparse,json,math,pathlib,sys
from types import SimpleNamespace
import numpy as np
p=argparse.ArgumentParser();p.add_argument('--decisions',type=pathlib.Path,required=True);p.add_argument('--audit',type=pathlib.Path,required=True);p.add_argument('--vendor',type=pathlib.Path,required=True);a=p.parse_args();sys.path.insert(0,str(a.vendor.resolve()))
from env.virtual3d import camera_pose
from nso.competition_candidates_v8 import aperture_support
rows=[json.loads(x) for x in a.decisions.read_text().splitlines()];cfg=next(x['request']['params']['configuration'] for x in rows if x['request']['method']=='init');s=cfg['sensor_config'];config=SimpleNamespace(resolution_m=s.get('resolution_m',.2),camera_height_m=s['camera_height_m'],width_px=s['width_px'],height_px=s['height_px'],fov_deg=s['fov_deg'],max_depth_m=s['max_depth_m']);audit=json.loads(a.audit.read_text());cases=[]
for plan in audit['plans']:
 call=plan['selection_call_id'];q=next(x['response']['result'] for x in rows if x['request']['method']=='next_action' and x['response'].get('ok') and (x['response']['result'].get('selected') or {}).get('selection_call_id')==call);selection=q['selection'];assets=selection['candidate_audit']['measured_assets'];computed=0;camera_checks=0
 for candidate,score in zip(selection['candidates'],selection['score_audit']):
  for i,raw in enumerate(assets):
   asset={k:np.asarray(v) if isinstance(v,list) else v for k,v in raw.items()};factor=0.;depths=[];behind=[];max_visible=0
   for state in candidate['outbound_states'][1:]:
    row,col,heading=state;pose=camera_pose((row,col),heading,config,cfg['shape'][0]);expected=np.array([(col+.5)*config.resolution_m,(cfg['shape'][0]-row-.5)*config.resolution_m,config.camera_height_m]);np.testing.assert_allclose(pose[:3,3],expected,rtol=0,atol=1e-12);camera_checks+=1
    dep=float((pose[:2,3]-asset['rear_boundary_xy'])@asset['back_axis']);depths.append(dep)
    if dep>.15:behind.append(state)
    sup=aperture_support(asset,pose,config);max_visible=max(max_visible,int(np.count_nonzero(sup)))
    delta=pose[:2,3]-asset['aabb_center'][:2];sector=int(np.floor((np.arctan2(delta[1],delta[0])+np.pi)/(2*np.pi)*8))%8;support=score['measured_direction_support'];novelty=1.-support[i][sector] if i<len(support) else 1.;floor=score['measured_novelty_floor'];factor=max(factor,float(sup.mean())*(floor+(1-floor)*novelty))
   assert math.isclose(factor,score['incremental_aperture'][i],rel_tol=0,abs_tol=1e-12),(call,candidate['candidate_id'],i,factor,score['incremental_aperture'][i]);computed+=1
   marked=next((v for v in plan['marked_groups'] if v['index']==i),None)
   if marked:
    manual=next(v for v in marked['candidate_checks'] if v['candidate_id']==candidate['candidate_id']);assert math.isclose(max(depths),manual['max_back_depth_m'],abs_tol=1e-12);assert behind==manual['states_passing_back_gate'];assert max_visible==manual['max_grid_points_after_vertical_gate']
 cases.append({'selection_call_id':call,'camera_state_asset_checks':camera_checks,'candidate_asset_factors_matched':computed})
print(json.dumps({'scope':'original camera_pose and aperture_support, recorded paths and measured geometry; no detector, mapper or planning run','cases':cases,'result':'pass'},indent=2))
