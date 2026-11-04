"""Offline reconstruction from recorded measured packets; no ROS, no motion."""
import hashlib,json,pathlib,sys
from types import SimpleNamespace
import numpy as np
from scipy.sparse.csgraph import dijkstra
W=pathlib.Path('/home/ouc/nso_robot/workspaces/nso_hybrid_20261029/nso_bridge');sys.path.insert(0,str(W))
from nso_bridge.joint_engine import JointSession,RGBDFrame,PlanarScan,json_value
from nso_bridge.sensor_range_adapter import adapt_scan
from nso.frontier_policy import FrontierPolicy
from env.grid_exploration import GridObservation,readonly_copy
from nso.execution_guard_v8 import ObservedExecutionGuard
from nso.route_coverage_v2 import orientation_graph
R=pathlib.Path('/home/ouc/nso_robot/reports/nso-hybrid-20261104/default_S40_02');D=pathlib.Path(__file__).parent
source=pathlib.Path('/home/ouc/nso_robot/analyses/instance_support_20261103/replay_joint_profile.py').read_text().split('\ndef replay(name):')[0];ns={'__file__':str(D/'cache_reader.py')};exec(compile(source,'cached_frontend','exec'),ns);frontend=ns['CacheFrontend'](R)
sha=lambda p:hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest();rows=[json.loads(x) for x in (R/'joint/decisions.jsonl').read_text().splitlines()];init=rows[0]['request']['params'];conf=init['configuration'];session=JointSession(conf,calibration=init['calibration']);mapper=session.mapper;outputs=[];inputs=[]
class Instrumented(FrontierPolicy):
 def reset(self):super().reset();self.audit=[]
 def _candidate_score(self,obs,cell,heading,gain,distance,parent,base_score):
  self.audit.append(dict(cell=list(cell),heading=heading,predicted_unknown_camera_cells=gain,distance_cells=distance,score=base_score));return base_score
for ix,row in enumerate(rows):
 req=row['request'];resp=row['response'].get('result',{})
 if req['method']!='observe' or not resp.get('accepted'):continue
 params=req['params'];path=pathlib.Path(params['path']);rawsha=sha(path)
 with np.load(path,allow_pickle=False) as z:
  meta=json.loads(str(z['joint_metadata_json'].item()));record,_=frontend.cache[str(meta['frame_id'])];assert rawsha==record['source_file_sha256'];depth=z['depth_m'];valid=np.isfinite(depth)&(depth>0)&(depth<=session.config.max_depth_m)
  if meta.get('approved_self_mask_applied'):
   mr=meta.get('approved_self_mask_receipt') or {};asset=pathlib.Path(mr.get('asset_path') or (pathlib.Path(init['calibration']['self_mask_asset_directory'])/'fixed-imu-mask.npy'));assert sha(asset)==(mr.get('file_sha256') or init['calibration']['self_mask_file_sha256']);valid &= ~np.load(asset,allow_pickle=False)
  v=frontend.predict(z['rgb_raw'],valid,frame_id=str(meta['frame_id']));stride=record['pixel_stride'];assert np.allclose(np.where(valid[::stride,::stride],depth[::stride,::stride],0),z['depth'],rtol=1e-6,atol=1e-7)
  frame=RGBDFrame(float(z['rgbd_timestamp_s']),z['depth'].copy(),z['color_rgb'].copy(),z['intrinsic'].copy(),session.planning_from_odom@z['world_from_camera'],v['semantic'][::stride,::stride]);scan=PlanarScan(float(z['scan_timestamp_s']),z['ranges'].copy(),float(z['angle_min']),float(z['angle_increment']),float(z['range_max']),session.planning_from_odom@z['world_from_laser']);scan,range_audit=adapt_scan(scan,session.sensor_range_profile);mapper.input_observed_grid=z['joint_belief_grid'].copy();mapper.update(frame,scan)
 action=params['action_id'];pose=resp['state']['pose'];inputs.append(dict(action=action,frame_id=meta['frame_id'],raw_sha256=rawsha,frontend_cache_sha256=record['mask_file_sha256']))
 if action not in (0,20,40):continue
 belief_hash=hashlib.sha256(mapper.belief.tobytes()).hexdigest();historical_hash=None
 # Compare an archived immediately following planning event only if one actually exists.
 for later in rows[ix+1:]:
  if later['request']['method']=='observe':break
  selection=later['response'].get('result',{}).get('selection') or {}
  if (selection.get('selected') or {}).get('selected_map_version') == action+1 and selection.get('candidate_audit',{}).get('belief_sha256'):historical_hash=selection['candidate_audit']['belief_sha256'];break
 if historical_hash is not None:assert belief_hash==historical_hash,'recorded joint map mismatch'
 obs=GridObservation(readonly_copy(mapper.belief),readonly_copy(mapper.visible),tuple(pose[:2]),int(pose[2]),action,False);cfg=SimpleNamespace(resolution_m=session.config.resolution_m,robot_radius_m=session.config.robot_radius_m,sensor_range_m=session.config.max_depth_m,sensor_fov_deg=session.config.fov_deg);pol=Instrumented(cfg,method='nearest_frontier',max_candidates=24);decision=pol.act(obs)
 guard=ObservedExecutionGuard(cfg.resolution_m,cfg.robot_radius_m);safe=guard.safe_grid(mapper.belief);anchor=tuple(conf['return_anchor']);remaining=conf['total_budget']-action;pos=tuple(pose[:2]);start_safe=bool(safe[pos]);anchor_safe=bool(safe[anchor[:2]]);ret=guard.return_plan(mapper.belief,pos,pose[2],anchor,remaining);assess=None
 if decision.action!='stop' and remaining>0:
  test=guard.assess(mapper.belief,pos,pose[2],decision.action,remaining);heading=(pose[2]+(1 if decision.action=='right' else -1 if decision.action=='left' else 0))%4;back=guard.return_plan(mapper.belief,test.target,heading,anchor,remaining-1) if test.allowed else None;assess=dict(action=decision.action,footprint_allowed=test.allowed,reason=test.reason,return_available=None if back is None else back.available,return_reason=None if back is None else back.reason,return_cost=None if back is None else back.paid_cost)
 goal=decision.goal;goal_info=None
 if start_safe and anchor_safe and goal is not None:
  graph,cells,ids=orientation_graph(safe);start=int(ids[pos])*4+pose[2];home=int(ids[anchor[:2]])*4+anchor[2]
  if safe[goal]:
   dest=int(ids[goal])*4+decision.goal_heading;out=dijkstra(graph,directed=True,indices=start);back=dijkstra(graph.T.tocsr(),directed=True,indices=home);cost=out[dest]+back[dest];goal_info=dict(known_safe=True,roundtrip_cost=None if not np.isfinite(cost) else int(cost),within_recorded_remaining=bool(np.isfinite(cost) and cost<=remaining))
  else:goal_info=dict(known_safe=False,roundtrip_cost=None,within_recorded_remaining=False)
 outputs.append(dict(action=action,frame_id=meta['frame_id'],pose=pose,actual_base_pose_planning=resp['state']['actual_base_pose_planning'],remaining_budget=remaining,belief_sha256=belief_hash,archived_immediate_planning_belief_sha256=historical_hash,exact_archived_map_hash_match=None if historical_hash is None else True,observed_cells=int(np.count_nonzero(mapper.belief!=-1)),safe_cells=int(safe.sum()),current_safe=start_safe,anchor_safe=anchor_safe,return_plan=dict(available=ret.available,reason=ret.reason,cost=ret.paid_cost),frontier_decision=json_value(decision.__dict__),frontier_scored_candidates=pol.audit,goal_feasibility=goal_info,first_action_assessment=assess));print('action',action,'done',flush=True)
 if action==40:break
result=dict(scope='offline replay of one recorded physical integration episode; no ROS, movement, complete explore_lite, new policy rollout or effectiveness claim',configuration=conf,original_source_manifest_sha256=sha(R/'source_manifest.json'),journal_sha256=sha(R/'joint/decisions.jsonl'),cache_reader_sha256=sha('/home/ouc/nso_robot/analyses/instance_support_20261103/replay_joint_profile.py'),input_packets=inputs,states=outputs,limitation='Mapper reconstructed from actual accepted RGB-D/scan, source occupancy grid and cached semantic frontend; exact archived map hash checked only where a recorded immediate planning selection exposes one. Frontier policy instantiated fresh at each snapshot; this is same-state selection probe, not a replay of a persistent frontier trajectory. The budget is the recorded S40 budget and step40 has zero remaining.')
(D/'result.json').write_text(json.dumps(json_value(result),indent=2))
