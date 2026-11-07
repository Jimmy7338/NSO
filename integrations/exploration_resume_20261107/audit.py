#!/usr/bin/env python3
"""Scalar audit of two recorded selections; never invokes planning or detection."""
import argparse,json,math
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('decisions',type=Path);args=p.parse_args()
rows=[json.loads(x) for x in args.decisions.read_text().splitlines()]
config=next(x['request']['params']['configuration'] for x in rows if x['request']['method']=='init'); sensor=config['sensor_config'];resolution=sensor.get('resolution_m',.2);grid_height=config['shape'][0]
configs=[(0,4),(20,85)];out=[]
for action,call in configs:
 obs=next(x['response']['result'] for x in rows if x['request']['method']=='observe' and x['response'].get('ok') and x['response']['result'].get('accepted') and x['response']['result']['state']['action_id']==action)
 q=next(x['response']['result'] for x in rows if x['request']['method']=='next_action' and x['response'].get('ok') and (x['response']['result'].get('selected') or {}).get('selection_call_id')==call)
 sel=q['selection'];assets=sel['candidate_audit']['measured_assets'];markers=obs['summary']['semantic_diagnostics']['measured_assets'];assert [a['group'] for a in assets]==[a['group'] for a in markers]
 marked=[]
 for i,(a,m) in enumerate(zip(assets,markers)):
  if not m['marked_points']:continue
  checks=[]
  for cand,score in zip(sel['candidates'],sel['score_audit']):
   depths=[];behind=[];max_z=max_h=max_v=0
   for row,col,h in cand['outbound_states'][1:]:
    dr,dc=[(-1,0),(0,1),(1,0),(0,-1)][h];forward=(dc,-dr);right=(forward[1],-forward[0]);camera=((col+.5)*resolution,(grid_height-row-.5)*resolution,sensor['camera_height_m']);rear=a['rear_boundary_xy'];back=a['back_axis'];side=a['side_axis'];d=(camera[0]-rear[0])*back[0]+(camera[1]-rear[1])*back[1];depths.append(d)
    if d<=.15:continue
    behind.append([row,col,h]);zcount=hcount=vcount=0
    for oi in range(5):
     off=(oi/4-.5)*a['measured_width_m'];px=rear[0]+off*side[0];py=rear[1]+off*side[1];dx=px-camera[0];dy=py-camera[1];z=dx*forward[0]+dy*forward[1];horizontal=dx*right[0]+dy*right[1]
     for vi in range(5):
      point_height=a['observed_low'][2]+vi/4*(a['observed_high'][2]-a['observed_low'][2]);vertical=camera[2]-point_height;tangent=math.tan(math.radians(sensor['fov_deg']/2));rangeok=.15<z<=sensor['max_depth_m'];horizontalok=abs(horizontal)<z*tangent;verticalok=abs(vertical)<z*tangent*sensor['height_px']/sensor['width_px'];zcount+=rangeok;hcount+=rangeok and horizontalok;vcount+=rangeok and horizontalok and verticalok
    max_z=max(max_z,zcount);max_h=max(max_h,hcount);max_v=max(max_v,vcount)
   checks.append({'candidate_id':cand['candidate_id'],'max_back_depth_m':max(depths),'states_passing_back_gate':behind,'max_grid_points_in_depth_range':max_z,'max_grid_points_after_horizontal_gate':max_h,'max_grid_points_after_vertical_gate':max_v,'logged_aperture_factor':score['incremental_aperture'][i]})
  targets=[x for x in sel['candidate_audit']['target_audit'] if x['asset_index']==i]
  marked.append({'index':i,'group':a['group'],'center':a['aabb_center'],'width_depth_height_m':[a['measured_width_m'],a['measured_depth_m'],a['measured_height_m']],'support_points':a['support_points'],'marked_points':m['marked_points'],'class_vote':m['class_vote'],'candidate_checks':checks,'target_audit':targets})
 out.append({'observation_action_id':action,'selection_call_id':call,'selected_candidate':q['selected']['candidate_id'],'scores_G':sel['scores']['G'],'scores_S':sel['scores']['S'],'marked_groups':marked})
print(json.dumps({'schema':'nso.recorded_semantic_aperture_scalar_audit.v1','scope':'recorded geometry and candidates; model/planner not invoked','limits':['candidate audit group/asset ordering verified against observation','hypothetical rear plane and FOV are model geometry, not physical reachability','association error, object identity and accuracy not independently assessed'],'plans':out},indent=2))
