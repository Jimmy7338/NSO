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
sensor=dict(cfg['sensor_config']);sensor.update(resolution_m=.2,robot_radius_m=cfg['robot_radius_m']);config=VirtualConfigV2(**sensor);shape=tuple(cfg['shape']);T=np.asarray(cfg['planning_from_odom']);mapper=_MeasuredMapMapper(shape,config);mapper.sensor_range_profile=cfg['sensor_range_profile'];mapper.measured_extent_profile=cfg['measured_extent_profile'];install_dispatcher();started=time.monotonic();history=[]
for item in manifest[:22]:
    p=P/item['file'];assert fsha(p)==item['sha256']
    with np.load(p,allow_pickle=False) as z:
        frame=RGBDFrame(float(z['rgbd_timestamp_s']),z['depth'].copy(),z['color_rgb'].copy(),z['intrinsic'].copy(),T@z['world_from_camera'],z['semantic'].copy())
        scan=PlanarScan(float(z['scan_timestamp_s']),z['ranges'].copy(),float(z['angle_min']),float(z['angle_increment']),float(z['range_max']),T@z['world_from_laser']);scan,_=adapt_scan(scan,cfg['sensor_range_profile']);mapper.input_observed_grid=z['joint_belief_grid'].copy();mapper.update(frame,scan);history.append(frame.world_from_camera.copy())
        if item['action_id']==21:base=z['base_odom_pose'].copy();stamp=float(z['rgbd_timestamp_s'])
assert mapper.frames==22 and sha(mapper.belief)==selection['candidate_audit']['belief_sha256']
safe=~inflated_obstacles(mapper.belief!=0,config.robot_radius_m/config.resolution_m);assert sha(safe)==selection['candidate_audit']['safe_sha256']
current=tuple(selection['candidate_audit']['current_pose']);anchor=tuple(cfg['return_anchor']);point=T@np.r_[base[:2],0.,1.];actual_yaw=float(base[2]+math.atan2(T[1,0],T[0,0]));derived=(shape[0]-1-int(np.floor(point[1]/.2)),int(np.floor(point[0]/.2)),int(math.floor((math.pi/2-actual_yaw)/(math.pi/2)+.5))%4);assert derived==current
assets=measured_assets(mapper);logged=last_observe['summary']['semantic_diagnostics']['measured_assets'];assert [(a['group'],a['marked_points'],a['class_vote']) for a in assets]==[(a['group'],a['marked_points'],a['class_vote']) for a in logged]
for a,b in zip(assets,selection['candidate_audit']['measured_assets']):
    for k in b:assert np.allclose(a[k],b[k],rtol=0,atol=1e-12),k
native_descriptors=[{k:(v.tolist() if isinstance(v,np.ndarray) else v) for k,v in a.items() if k not in ['points','bits']} for a in assets]

from types import SimpleNamespace
from scipy.ndimage import label
from nso_bridge.exploration_joint_policy import ExplorationJointCPUFourModules,ExplorationObservedGainCalibration
from nso.cpu_sensor_contract_v10 import json_value
from nso.competition_candidates_v8 import measured_assets as scoring_assets
modules=last_observe['summary']['runtime']['modules'];feedback=modules['feedback'];g=modules['observed_gain_calibration']
assert len(history)==feedback['planning_camera_pose_count']==22
assert sha(np.asarray(history,dtype=np.float64))==feedback['planning_camera_poses_sha256']
gain=ExplorationObservedGainCalibration(shape,feedback_enabled=g['feedback_enabled'],consume_predicted_attempts=g['consume_predicted_attempts'])
gain.alpha=dict(g['alpha']);gain.beta=dict(g['beta']);gain.events=g['events'];assert gain.snapshot()==g
class History:
    def planning_camera_poses(self):return [p.copy() for p in history]
args=SimpleNamespace(cpu_planner_revision='v10_3_1',cpu_score_mode='C',cpu_max_candidates=12,cpu_coverage_slots=4,cpu_measured_novelty_floor=.25)
backend=ExplorationJointCPUFourModules(args,1,shape)
assets_for_score=scoring_assets(mapper)
support=[[float(np.mean((np.asarray(a['bits'],dtype=np.int64)&(1<<sector))!=0)) for sector in range(8)] for a in assets_for_score]
s=dict(mapper=mapper,assets=assets_for_score,ledger=History(),gain=gain,planning_direction_support=support)
attempted=gain.attempted_camera_mask(mapper,history)
assert sha(attempted)==selection['candidate_audit']['attempted_camera_sha256']
regions,n=label(safe,structure=np.array([[0,1,0],[1,1,1],[0,1,0]]));region=int(regions[current[:2]]);assert region>0 and region==int(regions[anchor[:2]])
def evaluate(budget):
    routes,audit=generate_options(mapper,current[:2],current[2],budget,anchor,max_candidates=12,coverage_strategy='total_diverse',coverage_slots=4,attempted_camera_mask=attempted)
    generated=[r['candidate_id'] for r in routes]
    routes=[r for r in routes if int(regions[tuple(r['pose'][:2])])==region]
    scores,rows=backend._scores_v10_1(s,routes,attempted)
    choices={}
    for mode in ('C','G','S','X'):
        i=min(range(len(routes)),key=lambda i:(-scores[mode][i],routes[i]['cost'],routes[i]['candidate_id'])) if routes else None
        if i is not None and scores[mode][i]<=0:i=None
        choices[mode]=None if i is None else dict(candidate_id=routes[i]['candidate_id'],group=routes[i]['group'],pose=routes[i]['pose'],cost=routes[i]['cost'],outbound_cost=routes[i]['outbound_cost'],return_cost=routes[i]['return_cost'],score=scores[mode][i],first_action=routes[i]['outbound_actions'][0])
    return dict(remaining_budget=budget,generated_ids=generated,retained_ids=[r['candidate_id'] for r in routes],candidates=routes,candidate_audit=audit,scores=scores,score_audit=rows,choices=choices)
def same(a,b,path=''):
    if isinstance(a,dict):
        assert a.keys()==b.keys(),(path,a.keys(),b.keys())
        for k in a:same(a[k],b[k],path+'.'+k)
    elif isinstance(a,list):
        assert len(a)==len(b),(path,len(a),len(b))
        for i,(u,v) in enumerate(zip(a,b)):same(u,v,path+f'[{i}]')
    elif isinstance(a,(int,float)) and not isinstance(a,bool):assert np.isclose(a,b,atol=1e-8,rtol=1e-8),(path,a,b)
    else:assert a==b,(path,a,b)
control=evaluate(19)
for k in ('candidates','candidate_audit','scores','score_audit'):same(json_value(control[k]),selection[k],k)
print('native19 exact recorded control passed',flush=True)
result=evaluate(57)
summary=[]
for r,a in zip(result['candidates'],result['score_audit']):
    i=result['retained_ids'].index(r['candidate_id']);summary.append(dict(candidate_id=r['candidate_id'],group=r['group'],pose=r['pose'],outbound_cost=r['outbound_cost'],return_cost=r['return_cost'],cost=r['cost'],G=result['scores']['G'][i],S=result['scores']['S'][i],G_minus_S=result['scores']['G'][i]-result['scores']['S'][i],incremental_aperture=a['incremental_aperture'],prior_potentials=a['prior_potentials']))
report=dict(schema='nso.C21.original_remaining57.v1',status='completed',scope='One posthoc resource counterfactual, unchanged deployed exploration_joint_v1 with declared existing range/extent adapters; no full policy replay, detector, ROS or physical execution',observations=22,paid_prefix=21,original_remaining=19,hypothetical_remaining=57,hypothetical_total=78,radius_m=config.robot_radius_m,control=dict(native19_all_candidate_audit_routes_scores_score_audit_match_recorded=True,recorded_mode=selection['mode'],recorded_selection=selection['selected'],choices=control['choices']),history_count=len(history),history_sha256=sha(np.asarray(history,dtype=np.float64)),gain_snapshot=gain.snapshot(),planning_direction_support=support,attempted_sha256=sha(attempted),belief_sha256=sha(mapper.belief),safe_sha256=sha(safe),camera_seen_sha256=sha(mapper.camera_seen),result=result,compact_candidates=summary,source_sha256={rel:fsha(W/rel) for rel in recorded},script_sha256=fsha(__file__),elapsed_s=time.monotonic()-started,limits=['Not executed S78 or evidence of semantic causal benefit.','Native aperture is hypothetical far-plane image support, not verified unobstructed visibility.','Resource condition57 chosen once from earlier native wanted-heading full-return minimum; no budget sweep.','History/gain restored from recorded state; native19 full candidate and score equivalence internally validates reconstruction.'])
(O/'result.json').write_text(json.dumps(json_value(report),ensure_ascii=False,indent=2,allow_nan=False)+'\n')
print(json.dumps(json_value(dict(choices=result['choices'],candidates=summary,elapsed_s=report['elapsed_s'])),ensure_ascii=False),flush=True)
