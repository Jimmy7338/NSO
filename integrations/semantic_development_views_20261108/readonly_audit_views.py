"""Recorded-view audit only; no ROS/model/mapper/planner execution."""
import os
os.environ.update(OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1')
import pathlib,json,hashlib,sys,collections,numpy as np
D=pathlib.Path(__file__).resolve().parent;W=pathlib.Path('/home/ouc/nso_robot/workspaces/nso_hybrid_20261029');sys.path.insert(0,str(W/'nso_bridge'))
from nso_bridge.joint_engine import _MeasuredMapMapper
from env.virtual3d_v2 import VirtualConfigV2
from env.virtual3d import camera_pose
from nso.competition_candidates_v8_1 import aperture_support
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest();protocol=json.loads((D/'protocol.json').read_text());views=[];missing=[]
for view in protocol['views']:
 name=view['id'];summary=D/name/'score/summary.json'
 if not summary.exists():missing.append(name);continue
 s=json.loads(summary.read_text());cfgpath=pathlib.Path(s['configuration']);sourcepath=pathlib.Path(s['source']);packet=pathlib.Path(s['joint_packet']);assert sha(cfgpath)==s['configuration_sha256'];assert sha(sourcepath)==s['source_sha256'];assert sha(packet)==s['joint_packet_sha256'];settings=json.loads(cfgpath.read_text());cfg=settings['configuration'];c=VirtualConfigV2(**dict(cfg['sensor_config'],resolution_m=.2,robot_radius_m=cfg['robot_radius_m']));sel=s['selection'];diag=s['post_plan_semantic_diagnostics'];votes={x['group']:x for x in diag['measured_assets']};assets=[]
 with np.load(packet,allow_pickle=False) as z:meta=json.loads(str(z['receipt_json']));base=np.array(meta['historical_tf']['odom_from_base']['matrix']);pose=[float(base[0,3]),float(base[1,3]),float(np.arctan2(base[1,0],base[0,0]))];stamp=meta['headers']['rgb']['stamp_ns']
 poses=sorted({tuple(p) for r in sel['candidates'] for p in r['outbound_states'][1:]}) if sel else []
 for ai,raw in enumerate(sel['candidate_audit']['measured_assets'] if sel else []):
  a={k:np.array(v) if isinstance(v,list) else v for k,v in raw.items()};counts=collections.Counter();maxback=-float('inf')
  for state in poses:
   cp=camera_pose(state[:2],state[2],c,cfg['shape'][0]);back=float((cp[:2,3]-a['rear_boundary_xy'])@a['back_axis']);maxback=max(maxback,back)
   if back<=.15:counts['rear_reject']+=1;continue
   counts['rear_pass']+=1
   h=np.linspace(-.5,.5,5)*a['measured_width_m'];v=np.linspace(a['observed_low'][2],a['observed_high'][2],5);xx,zz=np.meshgrid(h,v);pp=np.column_stack([a['rear_boundary_xy'][0]+xx.ravel()*a['side_axis'][0],a['rear_boundary_xy'][1]+xx.ravel()*a['side_axis'][1],zz.ravel()]);local=(pp-cp[:3,3])@cp[:3,:3];z=local[:,2];t=np.tan(np.deg2rad(c.fov_deg/2));forward=(z>.15)&(z<=c.max_depth_m);horizontal=forward&(np.abs(local[:,0])<z*t);vertical=horizontal&(np.abs(local[:,1])<z*t*c.height_px/c.width_px);counts['forward_range_pass']+=int(forward.any());counts['horizontal_pass']+=int(horizontal.any());counts['vertical_pass']+=int(vertical.any());assert bool(aperture_support(a,cp,c).max()>0)==bool(vertical.any())
  assets.append(dict(index=ai,group=raw['group'],vote=votes.get(raw['group']),dimensions_m=[raw[k] for k in ['measured_width_m','measured_depth_m','measured_height_m']],support_points=raw['support_points'],route_pose_count=len(poses),route_predicates=dict(counts),max_back_offset_m=None if not poses else maxback,entry_diagnostics=[x for x in sel['candidate_audit']['target_audit'] if x['asset_index']==ai]))
 delta=np.array(pose)-view['target_odom'];delta[2]=np.arctan2(np.sin(delta[2]),np.cos(delta[2]));views.append(dict(view=name,planned_yaw_offset_deg=view['yaw_offset_deg'],actual_odom_pose=pose,target_odom=view['target_odom'],actual_target_xy_error_m=float(np.linalg.norm(delta[:2])),actual_target_yaw_error_rad=float(delta[2]),rgb_stamp_ns=stamp,semantic_pixels=diag['receipt']['semantic_pixels'],quality_labels=diag['quality_labels'],post_plan_assets=assets,within_state_choices=s['within_state_choices'],candidate_count=0 if sel is None else len(sel['candidates']),all_G_S_X_equal=sel is not None and sel['scores']['G']==sel['scores']['S']==sel['scores']['X'],nonzero_aperture_factors=0 if sel is None else sum(v!=0 for x in sel['score_audit'] for v in x['incremental_aperture']),preplan_measured_asset_count=len(s['semantic_diagnostics']['measured_assets']),postplan_measured_asset_count=len(diag['measured_assets']),input_hashes={str(p):sha(p) for p in [summary,cfgpath,sourcepath,packet]}))
out=dict(status='complete_three_predeclared_views' if not missing else 'partial_waiting_for_remaining_actual_views',views=views,missing_views=missing,protocol_sha256=sha(D/'protocol.json'),script_sha256=sha(pathlib.Path(__file__)),limits=['Actual capture poses reported, not requested angles; setup errors/XY variation are retained.','Post-plan semantic diagnostics used: pre-plan asset list may be empty before native backend materializes assets.','Single-view same-state scores are development diagnostics, not executed method arms or unknown-coverage efficiency evidence.','Category predictions and geometry groups are not independently annotated object truth. No TF, candidate, entry, score or grouping modifications.'])
(D/'readonly_science_audit.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps({k:out[k] for k in ['status','missing_views']}));print(json.dumps([{k:v[k] for k in ['view','actual_odom_pose','quality_labels','candidate_count','all_G_S_X_equal','nonzero_aperture_factors','post_plan_assets']} for v in views]))
