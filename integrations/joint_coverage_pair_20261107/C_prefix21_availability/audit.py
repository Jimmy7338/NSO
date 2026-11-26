#!/usr/bin/env python3
"""Exact22-observation mapper-only C state; original directed view/return availability."""
import os
os.environ.update(OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1')
import hashlib,inspect,json,math,sys,time
from pathlib import Path
import numpy as np
from scipy.sparse.csgraph import dijkstra
W=Path('/home/ouc/nso_robot/workspaces/nso_hybrid_20261029');sys.path.insert(0,str(W/'nso_bridge'))
from nso_bridge.joint_engine import _MeasuredMapMapper
from nso_bridge.sensor_range_adapter import adapt_scan
from nso_bridge.measured_extent_adapter import install_dispatcher
from nso_bridge.exploration_joint_policy import generate_options,VIEW_OFFSETS_M,TARGET_ERROR_LIMIT_M
from env.virtual3d_v2 import VirtualConfigV2
from env.virtual3d import camera_pose
from nso.competition_candidates_v8_1 import measured_assets,aperture_support,navigation_rear_boundary,XY_DIRECTIONS
from nso.route_coverage_v2 import orientation_graph
from nso.response_candidates_v7 import _path,_actions
from utils.rgbd_contract import RGBDFrame,PlanarScan
from utils.grid_geometry import inflated_obstacles
P=Path('/home/ouc/nso_robot/analyses/joint_coverage_pair_20261107/publication/C');O=Path(__file__).resolve().parent
sha=lambda a:hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()
fsha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
settings=json.loads((P/'source/joint_configuration.json').read_text());cfg=settings['configuration'];manifest=json.loads((P/'frames_manifest.json').read_text())['frames'];recorded={x['path']:x['sha256'] for x in json.loads((P/'source/source_manifest.json').read_text())['files']}
source_paths=['nso_bridge/nso_bridge/joint_engine.py','nso_bridge/nso_bridge/exploration_joint_policy.py','nso_bridge/joint_vendor/nso/mapping3d.py','nso_bridge/joint_vendor/nso/observed_runtime_mapper_v10.py','nso_bridge/joint_vendor/nso/competition_candidates_v8_1.py','nso_bridge/joint_vendor/nso/route_coverage_v2.py','nso_bridge/joint_vendor/utils/grid_geometry.py']
for rel in source_paths:assert fsha(W/rel)==recorded[rel],rel
selection=None;last_observe=None;paid=0
for line in (P/'decisions.jsonl').read_text().splitlines():
    x=json.loads(line)
    if x['request']['method']=='observe':paid=x['request']['params']['action_id'];last_observe=x['response']['result']
    if x['request']['method']=='next_action' and paid==21:
        selection=x['response']['result']['selection'];break
assert selection is not None
sensor=dict(cfg['sensor_config']);sensor.update(resolution_m=.2,robot_radius_m=cfg['robot_radius_m']);config=VirtualConfigV2(**sensor);shape=tuple(cfg['shape']);T=np.asarray(cfg['planning_from_odom']);mapper=_MeasuredMapMapper(shape,config);mapper.sensor_range_profile=cfg['sensor_range_profile'];mapper.measured_extent_profile=cfg['measured_extent_profile'];install_dispatcher();started=time.monotonic()
for item in manifest[:22]:
    p=P/item['file'];assert fsha(p)==item['sha256']
    with np.load(p,allow_pickle=False) as z:
        frame=RGBDFrame(float(z['rgbd_timestamp_s']),z['depth'].copy(),z['color_rgb'].copy(),z['intrinsic'].copy(),T@z['world_from_camera'],z['semantic'].copy())
        scan=PlanarScan(float(z['scan_timestamp_s']),z['ranges'].copy(),float(z['angle_min']),float(z['angle_increment']),float(z['range_max']),T@z['world_from_laser']);scan,_=adapt_scan(scan,cfg['sensor_range_profile']);mapper.input_observed_grid=z['joint_belief_grid'].copy();mapper.update(frame,scan)
        if item['action_id']==21:base=z['base_odom_pose'].copy();stamp=float(z['rgbd_timestamp_s'])
assert mapper.frames==22 and sha(mapper.belief)==selection['candidate_audit']['belief_sha256']
safe=~inflated_obstacles(mapper.belief!=0,config.robot_radius_m/config.resolution_m);assert sha(safe)==selection['candidate_audit']['safe_sha256']
current=tuple(selection['candidate_audit']['current_pose']);anchor=tuple(cfg['return_anchor']);point=T@np.r_[base[:2],0.,1.];actual_yaw=float(base[2]+math.atan2(T[1,0],T[0,0]));derived=(shape[0]-1-int(np.floor(point[1]/.2)),int(np.floor(point[0]/.2)),int(math.floor((math.pi/2-actual_yaw)/(math.pi/2)+.5))%4);assert derived==current
assets=measured_assets(mapper);logged=last_observe['summary']['semantic_diagnostics']['measured_assets'];assert [(a['group'],a['marked_points'],a['class_vote']) for a in assets]==[(a['group'],a['marked_points'],a['class_vote']) for a in logged]
for a,b in zip(assets,selection['candidate_audit']['measured_assets']):
    for k in b:assert np.allclose(a[k],b[k],rtol=0,atol=1e-12),k
native_descriptors=[{k:(v.tolist() if isinstance(v,np.ndarray) else v) for k,v in a.items() if k not in ['points','bits']} for a in assets]
graph,cells,ids=orientation_graph(safe);start=int(ids[current[:2]])*4+current[2];home=int(ids[anchor[:2]])*4+anchor[2];assert ids[current[:2]]>=0 and ids[anchor[:2]]>=0
out,prev=dijkstra(graph,directed=True,indices=start,return_predecessors=True);back,backprev=dijkstra(graph.T.tocsr(),directed=True,indices=home,return_predecessors=True);cost=out+back;connected=np.isfinite(cost)&(out>0)
poses=[camera_pose(tuple(cell),h,config,shape[0]) for cell in cells for h in range(4)];state=lambda n:[int(cells[n//4,0]),int(cells[n//4,1]),int(n%4)]
def best(mask):
    nodes=np.flatnonzero(mask&connected)
    if not len(nodes):return None
    n=min(nodes,key=lambda n:(cost[n],out[n],int(n)));forward=_path(prev,start,int(n),cells);returning=_path(backprev,home,int(n),cells,reverse=True);full=forward+returning[1:];actions=_actions(full)
    assert len(actions)==round(cost[n]) and len(forward)-1==round(out[n]) and len(returning)-1==round(back[n])
    return dict(state=state(n),full_cost=int(round(cost[n])),outbound_cost=int(round(out[n])),return_cost=int(round(back[n])),outbound_states=[list(z) for z in forward],return_states=[list(z) for z in returning],full_actions=actions,within_original_remaining19=bool(cost[n]<=19))
results=[]
for ai,a in enumerate(assets):
    if not a['marked_points']:continue
    support=np.array([float(aperture_support(a,p,config).mean()) for p in poses]);rear=np.array([float((p[:2,3]-a['rear_boundary_xy'])@a['back_axis'])>.15 for p in poses]);wanted=int(np.argmax(XY_DIRECTIONS@a['front_axis']));head=np.arange(len(poses))%4==wanted;navrear=navigation_rear_boundary(mapper,a);desired=navrear+.4*a['back_axis'];error=np.array([np.linalg.norm(p[:2,3]-desired) for p in poses]);view=support>0;eligible=view&head&rear&(error<=TARGET_ERROR_LIMIT_M)
    row=dict(asset_index=ai,group=a['group'],marked_points=a['marked_points'],class_vote=a['class_vote'],known_safe_aperture_states=int(view.sum()),reachable_aperture_states=int((view&connected).sum()),reachable_wanted_heading_aperture_states=int((view&head&connected).sum()),reachable_entry_geometry_states=int((eligible&connected).sum()),native_wanted_heading=wanted,navigation_rear_boundary_xy=navrear.tolist(),native_desired_entry_xy=desired.tolist(),native_entry_projection_limit_m=TARGET_ERROR_LIMIT_M,any_heading_aperture_minimum=best(view),wanted_heading_aperture_minimum=best(view&head),native_entry_geometry_minimum=best(eligible),nearest_wanted_projection_error_m=float(np.min(error[view&head&connected])) if np.any(view&head&connected) else None,original_entry_audit=[r for r in selection['candidate_audit']['target_audit'] if r['asset_index']==ai],original_remaining19_any_aperture_states=int(np.count_nonzero(view&connected&(cost<=19))))
    # No generator is called unless one exact entry-eligible cost exists. At most one derived cost, no sweep.
    if row['native_entry_geometry_minimum'] is not None:
        B=row['native_entry_geometry_minimum']['full_cost'];routes,audit=generate_options(mapper,current[:2],current[2],B,anchor,max_candidates=12,coverage_strategy='total_diverse',coverage_slots=4,attempted_camera_mask=np.zeros(shape,bool));row['one_exact_derived_budget_native_candidate_check']=dict(hypothetical_remaining_budget=B,attempted_mask_scope='v10_3_1 consume_predicted_attempts=false: originalzero mask',target_audit=[r for r in audit['target_audit'] if r['asset_index']==ai],selected_asset_entry_routes=[r for r in routes if r['asset_index']==ai])
    else:row['one_exact_derived_budget_native_candidate_check']=None
    results.append(row)
np.savez_compressed(O/'partial_state.npz',belief=mapper.belief,safe=safe)
actual=json.loads((P/'source/result.json').read_text());run=json.loads((P/'source/running.json').read_text());arrival=actual['successful_arrivals'][20]
result=dict(schema='nso.actual_C_prefix21_native_availability.v1',scope='22actual observations, mapper-only; no full policy, scoring, detector, ROS or runtime change',mapper_frames=mapper.frames,paid_prefix_actions=21,current_state=list(current),actual_base_pose_planning=[float(point[0]),float(point[1]),actual_yaw],anchor=list(anchor),original_remaining_budget=19,belief_sha256=sha(mapper.belief),safe_sha256=sha(safe),logged_hashes_exact=True,native_assets_match_recorded=True,native_assets=native_descriptors,known_safe_cells=len(cells),current_anchor_connected_states=int(connected.sum()),minimum_current_return_cost=int(round(back[start])),assets=results,prefix_acquisition=dict(CAN_distance_m=arrival['can_end']['distance_m']-actual['successful_arrivals'][0]['can_start']['distance_m'],elapsed_to_arrival21_pose_source_s=arrival['after']['source_stamp_s']-run['started_unix_s'],timing_scope='source-pose timestamp minus recorded wall start, not exact monotonic mission runtime'),native_source_sha256={rel:fsha(W/rel) for rel in source_paths},script_sha256=fsha(__file__),elapsed_s=time.monotonic()-started,limits=['Any-heading support, wanted-heading support and nativeentry projection eligibility are different levels. Exact geometric eligibility is not a selected entry until the original generator check is reported.','Same group ID is not persistent physical identity. Assets are partial observed groups; no hidden rear geometry or unknown free space inferred.','The snapshot uses already paid actual21-action history, not a free initialization or a new physical experiment.','No score or policy counter is advanced; no map/asset grouping/radius/return contract is modified.'])
(O/'availability.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n');print(json.dumps(dict(current=result['current_state'],return_cost=result['minimum_current_return_cost'],assets=results,elapsed_s=result['elapsed_s']),ensure_ascii=False))
