"""Read-only recorded global-state semantic/aperture audit; no mapper/model/ROS."""
import os
os.environ.update(OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1')
import pathlib,json,hashlib,collections,sys,numpy as np
B=pathlib.Path('/home/ouc/nso_robot');W=B/'workspaces/nso_hybrid_20261029';D=pathlib.Path(__file__).resolve().parent;sys.path.insert(0,str(W/'nso_bridge'))
from nso_bridge.joint_engine import _MeasuredMapMapper
from env.virtual3d_v2 import VirtualConfigV2
from env.virtual3d import camera_pose
from nso.competition_candidates_v8_1 import aperture_support
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
rels=['nso_bridge/joint_vendor/nso/cpu_four_modules_v10.py','nso_bridge/joint_vendor/nso/hierarchical_options_v10.py','nso_bridge/joint_vendor/nso/competition_candidates_v8_1.py','nso_bridge/joint_vendor/nso/response_features_v7.py']
def audit(name):
 P=B/'reports/nso-hybrid-20261108'/name;rows=[json.loads(x) for x in (P/'joint/decisions.jsonl').read_text().splitlines()];cfg=json.loads((P/'joint_configuration.json').read_text())['configuration'];manifest={r['path']:r['sha256'] for r in json.loads((P/'source_manifest.json').read_text())['files']}
 for rel in rels:assert sha(W/rel)==manifest[rel]
 c=VirtualConfigV2(**dict(cfg['sensor_config'],resolution_m=.2,robot_radius_m=cfg['robot_radius_m']));obs={};plans={};aid=None
 for r in rows:
  if r['request']['method']=='observe':
   aid=r['request']['params']['action_id'];obs[aid]=r
  if r['request']['method']=='next_action' and r['response']['result'].get('selection'):
   s=r['response']['result']['selection'];plans.setdefault(s['plan_number'],(aid,s))
 result=[]
 for number,(aid,s) in plans.items():
  diag=obs[aid]['response']['result']['summary']['semantic_diagnostics'];votes={x['group']:x for x in diag['measured_assets']};assets=[];poses=sorted({tuple(p) for r in s['candidates'] for p in r['outbound_states'][1:]})
  for ai,raw in enumerate(s['candidate_audit']['measured_assets']):
   asset={k:np.array(v) if isinstance(v,list) else v for k,v in raw.items()};counts=collections.Counter();maxdepth=-float('inf')
   for p in poses:
    pose=camera_pose(p[:2],p[2],c,cfg['shape'][0]);depth=float((pose[:2,3]-asset['rear_boundary_xy'])@asset['back_axis']);maxdepth=max(maxdepth,depth)
    if depth<=.15:counts['rear_standoff_rejected_pose']+=1;continue
    counts['rear_standoff_pass_pose']+=1
    h=np.linspace(-.5,.5,5)*asset['measured_width_m'];v=np.linspace(asset['observed_low'][2],asset['observed_high'][2],5);xx,zz=np.meshgrid(h,v);pts=np.column_stack([asset['rear_boundary_xy'][0]+xx.ravel()*asset['side_axis'][0],asset['rear_boundary_xy'][1]+xx.ravel()*asset['side_axis'][1],zz.ravel()]);loc=(pts-pose[:3,3])@pose[:3,:3];z=loc[:,2];t=np.tan(np.deg2rad(c.fov_deg/2));forward=(z>.15)&(z<=c.max_depth_m);horizontal=forward&(np.abs(loc[:,0])<z*t);vertical=horizontal&(np.abs(loc[:,1])<z*t*c.height_px/c.width_px)
    counts['forward_range_pass_pose']+=int(forward.any());counts['horizontal_FOV_pass_pose']+=int(horizontal.any());counts['vertical_FOV_pass_pose']+=int(vertical.any());native=aperture_support(asset,pose,c);assert bool(native.max()>0)==bool(vertical.any())
   target=[x for x in s['candidate_audit']['target_audit'] if x['asset_index']==ai]
   assets.append(dict(index=ai,group=raw['group'],votes=votes.get(raw['group']),width_depth_height_m=[raw[k] for k in ['measured_width_m','measured_depth_m','measured_height_m']],support_points=raw['support_points'],route_pose_n=len(poses),route_predicates=dict(counts),max_route_back_axis_offset_m=maxdepth if poses else None,entry_audit=[{k:x[k] for k in ['role','reason','nearest_error_m','aperture_support_fraction']} for x in target]))
  result.append(dict(plan_number=number,paid_prefix=aid,current_pose=s['candidate_audit']['current_pose'],remaining_budget=s['candidate_audit']['remaining_budget'],quality_labels=diag['quality_labels'],candidate_count=len(s['candidates']),G_S_X_exactly_equal=s['scores']['G']==s['scores']['S']==s['scores']['X'],nonzero_aperture_factor_count=sum(float(v)!=0 for x in s['score_audit'] for v in x['incremental_aperture']),quality_proxy_nonzero_candidates=sum(x['quality_total_proxy']!=0 for x in s['score_audit']),assets=assets,entry_reason_counts=dict(collections.Counter(x['reason'] for x in s['candidate_audit']['target_audit'] if x['role'].endswith('_entry')))))
 first=obs[0];path=pathlib.Path(first['request']['params']['path']);receipt=first['response']['result']['summary']['semantic_diagnostics']['receipt'];assert sha(path)==receipt['source_file_sha256']
 with np.load(path,allow_pickle=False) as z:meta=json.loads(str(z['receipt_json']));base=meta['historical_tf']['odom_from_base']['matrix'];mount=meta['historical_tf']['base_from_camera']['matrix']
 observations=[dict(action_id=i,quality_labels=r['response']['result']['summary']['semantic_diagnostics']['quality_labels'],semantic_pixels=r['response']['result']['summary']['semantic_diagnostics']['receipt']['semantic_pixels']) for i,r in obs.items()]
 return dict(report=str(P),global_plans=result,observations=observations,first_packet_sha256=sha(path),first_actual_odom_from_base=base,first_actual_base_from_camera=mount,source_sha256={rel:manifest[rel] for rel in rels},grouping_profile=cfg.get('asset_grouping_profile'),measured_extent_profile=cfg.get('measured_extent_profile'))
current=audit('nov06_timing_S40_20261108_02');previous=audit('nov06_sameground_S40_20261108_01');assert np.allclose(current['first_actual_base_from_camera'],previous['first_actual_base_from_camera'],atol=1e-12,rtol=0)
out=dict(status='readonly_recorded_global_states_no_new_policy_or_detection',current=current,previous_first_global=previous['global_plans'][0],previous_first_actual_odom_from_base=previous['first_actual_odom_from_base'],same_captured_base_from_camera=True,limits=['Counterfactual scores are same-state logged scores, not executed G/X trajectories.','Only recorded accepted candidates and paths evaluated; missing-entry diagnostic reports original predicates but does not reconstruct/filter the entire native feasible pool.','Quality labels refer to original quality-evidence voxel samples, not full image mask pixel counts.','Different episode start positions/headings/FOV and real detections prevent isolating a single causal pose variable; same TF disproves an intervening camera-mount change as explanation.'])
(D/'semantic_opportunity_audit.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(dict(current_plans=len(current['global_plans']),current_summary=[dict(plan=x['plan_number'],paid=x['paid_prefix'],labels=x['quality_labels'],aperture=x['nonzero_aperture_factor_count'],entry=x['entry_reason_counts']) for x in current['global_plans']],previous_first=previous['global_plans'][0]),indent=2))
