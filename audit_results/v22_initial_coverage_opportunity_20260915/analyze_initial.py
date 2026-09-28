#!/usr/bin/env python3
"""Four action-zero opportunity audits; no future packets, rewards or GT geometry.

A/B mean left/right marked observed instances, not simulator IDs. Diagnose the
existing eight coverage routes and, separately, paid terminal-turn alternatives
at those same endpoints. Alternatives retain the original outbound path and
its positive unknown-cell union; no independent near-visit is relabelled.
"""
import os
os.environ.setdefault('OMP_NUM_THREADS','1');os.environ.setdefault('OPENBLAS_NUM_THREADS','1')
os.environ.setdefault('PYTHONDONTWRITEBYTECODE','1')
from copy import deepcopy
import csv,gzip,hashlib,json,sys,zipfile
from pathlib import Path
from types import SimpleNamespace
import numpy as np
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from nso.observed_runtime_mapper_v10 import ObservedRuntimeMapperV10
from nso.decision_replay_v13 import load_packet
from nso.cpu_sensor_contract_v10 import digest,json_value
from nso.observed_asset_axes_v16 import measured_assets_v16
from nso.semantic_taxonomy_v15 import canonical_observed_records
from nso.axis_history_view_v16 import AxisHistoryViewV16
from nso.facility_candidates_v20 import ObservedRouteSpaceV20
from nso.observed_precision_response_v19 import ObservedPrecisionResponseV19
from nso.visibility_corrected_scores_v17 import ObservedMeshAperture
from env.virtual3d import camera_pose
from utils.grid_geometry import DIRECTIONS
OUT=Path(__file__).resolve().parent
BATCH=ROOT/'eval_results/facility_v20_coverage_probe_20260915'


def read(path):return json.loads(path.read_text())
def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def require(ok,message):
 if not ok:raise ValueError(message)
def dump(name,value):(OUT/name).write_text(json.dumps(json_value(value),indent=2,sort_keys=True,allow_nan=False)+'\n')

def initial_calls(path):
 """Stream only action-zero records; stop on the first later envelope.

 We never open case result.json or any future packet. Hashing the compressed
 log for provenance does not supply any of its later observations to scoring.
 """
 decoder=json.JSONDecoder();buffer='';started=False;calls=[]
 with gzip.open(path,'rt') as stream:
  while True:
   chunk=stream.read(4096)
   if not chunk:break
   buffer+=chunk
   while True:
    buffer=buffer.lstrip()
    if not started:
     if not buffer:break
     require(buffer[0]=='[','Expected JSON array log');buffer=buffer[1:];started=True
    buffer=buffer.lstrip(' \n\r\t,')
    if not buffer or buffer[0]==']':break
    # The sorted record envelope begins with action_id. Stop before parsing
    # any nonzero-action record body, including later rewards.
    if buffer.startswith('{'):
     import re
     match=re.match(r'\{\s*"action_id"\s*:\s*(\d+)',buffer)
     if match and int(match.group(1))>0:return calls
    try:row,offset=decoder.raw_decode(buffer)
    except json.JSONDecodeError:break
    if row['action_id']!=0:return calls
    calls.append(row);buffer=buffer[offset:]
 return calls

def validate_route(route,space):
 states=route['states'];actions=route['actions'];out=route['outbound_cost']
 require(len(states)==len(actions)+1 and len(actions)==route['cost']==out+route['return_cost'],'Route accounting differs')
 require(states[0]==list(space.current) and states[-1]==list(space.anchor),'Route start/return heading differs')
 require(route['cost']<=space.budget and out>0,'Route budget invalid')
 require(states[out]==route['pose'],'Endpoint does not match paid arrival')
 for pose in states:require(space.safe[tuple(pose[:2])],'Unknown/unsafe route cell')
 for before,after,action in zip(states,states[1:],actions):
  dr,dc=DIRECTIONS[before[2]] if action=='forward' else (0,0)
  expected=[int(before[0]+dr),int(before[1]+dc),(before[2]+(1 if action=='right' else -1 if action=='left' else 0))%4]
  require(expected==after,'Invalid paid action transition')


def turn_variant(route,heading,space):
 endpoint=route['pose'];delta=(heading-endpoint[2])%4
 turns=[] if delta==0 else ['right'] if delta==1 else ['left','left'] if delta==2 else ['left']
 updated=space.route((*endpoint[:2],heading),group=route['group'],candidate_id=route['candidate_id'])
 if updated is None:return None
 states=deepcopy(route['outbound_states']);actions=list(route['outbound_actions'])
 for action in turns:
  states.append([*states[-1][:2],(states[-1][2]+(1 if action=='right' else -1))%4]);actions.append(action)
 updated.update(outbound_states=states,outbound_actions=actions,outbound_cost=len(actions),arrival_action=len(actions),
  states=states+updated['return_states'][1:],actions=actions+updated['return_actions'],cost=len(actions)+updated['return_cost'])
 if updated['cost']>space.budget:return None
 validate_route(updated,space)
 require(np.array_equal(space.route_mask(updated),space.route_mask(route)),'Turning changed the shared 2D unknown union')
 return updated,len(turns)


def route_metrics(route,ai,assets,response,aperture,mapper,initial_pose):
 asset=assets[ai];states=route['outbound_states'][1:];part=response.slices[ai]
 pose_responses=[response._pose_response(s) for s in states]
 measurements=[int(r['visible'][part].sum()) for r in pose_responses]
 unknown=[int((r['visible']&r['no_hit'])[part].sum()) for r in pose_responses]
 raw_poses=[camera_pose(s[:2],s[2],mapper.config,mapper.shape[0]) for s in states]
 support=[float(aperture.support(asset,p).mean()) for p in raw_poses]
 end_pose=raw_poses[-1];offset=end_pose[:2,3]-asset['aabb_center'][:2]
 sector=int(np.floor((np.arctan2(offset[1],offset[0])+np.pi)/(2*np.pi)*8))%8
 xyz=np.array([p[:3,3] for p in raw_poses]);center=np.asarray(asset['aabb_center'])
 precision=response.score_route(route)['per_asset_expected_precision_change'][ai]
 start_support=float(aperture.support(asset,initial_pose).mean())
 return dict(observed_label='A' if ai==0 else None, observed_asset_index=ai,
  endpoint_xy=end_pose[:2,3].tolist(),endpoint_heading=int(route['pose'][2]),endpoint_center_distance_xy_m=float(np.linalg.norm(offset)),
  closest_route_center_distance_xy_m=float(np.linalg.norm(xyz[:,:2]-center[:2],axis=1).min()),
  endpoint_observed_points_in_predicted_view=measurements[-1],max_observed_points_in_predicted_view=max(measurements),
  predicted_point_acquisitions=sum(measurements),predicted_visible_points_with_no_known_mesh_hit=sum(unknown),
  observed_support_points=part.stop-part.start,endpoint_azimuth_sector=sector,
  endpoint_sector_historical_fraction=float(np.mean((np.asarray(asset['bits'],np.int64)&(1<<sector))!=0)),
  observed_precision_proxy_change=precision,initial_hypothetical_rear_plane_support=start_support,
  maximum_hypothetical_rear_plane_support=max(support),endpoint_hypothetical_rear_plane_support=support[-1],
  first_positive_rear_plane_action=next((i+1 for i,x in enumerate(support) if x>0),None),
  observed_mesh_only=True,unknown_is_not_visibility_certificate=True,calibrated_future_F1=False)


def audit_case(manifest,case):
 folder=BATCH/f'case_{case["index"]:02d}';inventory=read(folder/'artifact_hashes.json')
 for name in ['module_calls.json.gz','packets/0000.npz']:require(sha(folder/name)==inventory[name],'Initial evidence hash changed')
 calls=initial_calls(folder/'module_calls.json.gz')
 require(calls and all(c['action_id']==0 for c in calls),'Noninitial record entered audit')
 for call in calls:require(digest(call['inputs'])==call['input_sha256'] and digest(call['outputs'])==call['output_sha256'],'Initial log digest differs')
 recorded=next(c for c in calls if c['method']=='select_topo_target')['outputs']
 observed=next(c for c in calls if c['method']=='update_semantic')['outputs']['measured_assets']
 name=f'references/{case["parent"]}_{case["assignment"]}_2026.npz';ref=Path(manifest['reference_source_root'])/name
 require(sha(ref)==manifest['reference_sha256'][name],'Frozen public configuration source differs')
 with np.load(ref,allow_pickle=False) as cached:public=json.loads(str(cached['metadata'].item()))['signature_payload']['config']
 config=SimpleNamespace(**public);shape=(round(config.height_m/config.resolution_m),round(config.width_m/config.resolution_m))
 packet=load_packet(folder/'packets/0000.npz');require(packet.action_id==0 and packet.sha256()==case['initial_packet_sha256'],'Only frame zero required')
 mapper=ObservedRuntimeMapperV10(shape,config);mapper.update(packet.frame,packet.scan)
 assets=canonical_observed_records(measured_assets_v16(AxisHistoryViewV16(mapper,[packet.frame])),'inspection_v4')
 require(json_value([{k:v for k,v in a.items() if k!='points'} for a in assets])==observed,'Observed instances differ from initial module log')
 marked=sorted([i for i,a in enumerate(assets) if a['marked_points']>0],key=lambda i:tuple(assets[i]['aabb_center'][:2]))
 require(len(marked)==2,'Exactly two left/right observed marked instances required')
 initial_guard=next(c for c in calls if c['method']=='assess_local_action')
 require(initial_guard['inputs']['belief_sha256']==digest(mapper.belief),'Initial guarded map differs')
 space=ObservedRouteSpaceV20(mapper,packet.position,packet.heading,(*packet.position,packet.heading),case['budget'])
 routes,caudit=space.coverage_candidates(slots=8)
 recorded_routes=[r for r in recorded['candidates'] if r['group'].startswith('coverage_')]
 require(json_value(routes)==recorded_routes and json_value(caudit)==recorded['candidate_audit']['coverage'],'Shared coverage candidates differ')
 response=ObservedPrecisionResponseV19(mapper,assets,intrinsic=packet.frame.intrinsic)
 aperture=ObservedMeshAperture(mapper.mesh(),config);initial_pose=packet.frame.world_from_camera
 rows=[];variants=[];paths={};boundaries=[]
 for route in routes:
  validate_route(route,space);ci=route['candidate_id'];mask=space.route_mask(route)
  old=next(r for r in recorded['score_audit'] if r['candidate_id']==ci)['v20_coverage']
  require(digest(mask)==old['predicted_mask_sha256'] and int(mask.sum())==old['unique_predicted_unknown_cells'],'Initial unique-coverage prediction differs')
  require(mask.any(),'Coverage candidate has no common coverage role')
  common=dict(case_index=case['index'],parent=case['parent'],assignment=case['assignment'],candidate_id=ci,
   endpoint_pose=route['pose'],outbound_cost=route['outbound_cost'],return_cost=route['return_cost'],roundtrip_cost=route['cost'],
   original_unique_predicted_unknown_cells=int(mask.sum()),original_unknown_union_sha256=digest(mask),
   original_N_score=recorded['scores']['N'][ci],actual_future_coverage_unmeasured=True)
  paths[str(ci)]=route
  for role,ai in zip(['A','B'],marked):
   row=route_metrics(route,ai,assets,response,aperture,mapper,initial_pose);row['observed_label']=role
   rows.append(dict(common,**row,terminal_turns_added=0,kind='existing_coverage'))
  for heading in range(4):
   if heading==route['pose'][2]:continue
   proposal=turn_variant(route,heading,space)
   if proposal is None:continue
   updated,turns=proposal
   for role,ai in zip(['A','B'],marked):
    row=route_metrics(updated,ai,assets,response,aperture,mapper,initial_pose);row['observed_label']=role
    variants.append(dict(common,**row,kind='same_endpoint_paid_turn_diagnostic',terminal_turns_added=turns,
     outbound_cost=updated['outbound_cost'],return_cost=updated['return_cost'],roundtrip_cost=updated['cost'],
     candidate_path_sha256=digest(updated),original_outbound_prefix_preserved=True,coverage_union_unchanged=True))
  for other in routes:
   if ci>=other['candidate_id']:continue
   left=route['outbound_actions'];right=other['outbound_actions'];shared=0
   while shared<min(len(left),len(right)) and left[shared]==right[shared]:shared+=1
   boundaries.append(dict(first_candidate=ci,second_candidate=other['candidate_id'],common_outbound_actions=shared,
    first_different_action=shared+1 if shared<max(len(left),len(right)) else None,one_route_ends_at_common_prefix=shared==min(len(left),len(right))))
 geometry={k:v for k,v in observed[marked[0]].items() if k not in ['class_vote','raw_class_vote','semantic_encoding']}
 return dict(case={k:case[k] for k in ['index','parent','assignment','budget']},observed_instances=observed,
  observed_A_B=[dict(role=role,index=ai,center=assets[ai]['aabb_center'],class_vote=assets[ai]['class_vote'],marked_points=assets[ai]['marked_points']) for role,ai in zip(['A','B'],marked)],
  geometry_packet_sha256=digest(dict(depth=packet.frame.depth_m,scan=packet.scan.ranges_m,intrinsic=packet.frame.intrinsic,world_from_camera=packet.frame.world_from_camera)),
  observed_geometry_sha256=digest([{k:v for k,v in a.items() if k not in ['class_vote','raw_class_vote','semantic_encoding']} for a in observed]),
  initial_map_sha256=digest(mapper.belief),candidate_geometry_sha256=digest(routes),candidate_routes=paths,
  recorded_selected_candidate_id=recorded['selected']['candidate_id'],existing_rows=rows,terminal_heading_diagnostics=variants,
  route_divergence=boundaries,initial_calls_read=len(calls),future_case_result_opened=False,future_packet_loaded=False,
  reference_public_config_only=True,source_inputs=dict(raw_frame0_sha256=inventory['packets/0000.npz'],
  module_log_file_sha256=inventory['module_calls.json.gz'],reference_metadata_source_sha256=sha(ref)),
  paid_actions_executed=0,world_instantiated=False,semantic_gain_proven=False)


def compact(case):
 result=dict(case=case['case'],observed_A_B=case['observed_A_B'],candidate_count=len(case['candidate_routes']))
 for source,name in [('existing_rows','existing'),('terminal_heading_diagnostics','paid_terminal_turns')]:
  rows=case[source];summary={}
  for role in ['A','B']:
   selected=[r for r in rows if r['observed_label']==role]
   best=max(selected,key=lambda r:(r['maximum_hypothetical_rear_plane_support'],-r['outbound_cost'],-r['candidate_id'],-r['endpoint_heading']))
   best_precision=max(selected,key=lambda r:(r['observed_precision_proxy_change'],-r['outbound_cost'],-r['candidate_id'],-r['endpoint_heading']))
   summary[role]=dict(positive_rear_plane_rows=sum(r['maximum_hypothetical_rear_plane_support']>0 for r in selected),
    best_rear_plane={k:best[k] for k in ['candidate_id','endpoint_heading','outbound_cost','return_cost','roundtrip_cost','original_unique_predicted_unknown_cells','maximum_hypothetical_rear_plane_support','endpoint_hypothetical_rear_plane_support','first_positive_rear_plane_action','terminal_turns_added']},
    best_observed_precision={k:best_precision[k] for k in ['candidate_id','endpoint_heading','outbound_cost','return_cost','roundtrip_cost','observed_precision_proxy_change','predicted_point_acquisitions']})
  result[name]=summary
 return result


def main():
 require(not (OUT/'result.json').exists(),'Existing audit cannot be overwritten')
 manifest=read(BATCH/'manifest.json');require(manifest['status']=='complete','Frozen initial evidence required')
 sources=dict(manifest['source_sha256']);sources[str(Path(__file__).relative_to(ROOT))]=sha(Path(__file__))
 for name,wanted in sources.items():require(sha(ROOT/name)==wanted,'Frozen observed-side source changed: '+name)
 protocol=dict(status='declared_before_diagnostic',input_actions=[0],cases=[0,1,2,3],observed_instances='left/right marked clusters only',
  existing_coverage_candidates='all eight recorded shared coverage candidates per initial scene',
  heading_diagnostic='all other 3 cardinal headings at each same endpoint; preserve original outbound prefix, pay shortest turns, reserve current observed-map return',
  opportunity_metrics=['observed-point projected acquisition and scalar-depth precision proxy','known-mesh-masked hypothetical rear-plane support','observed distance/azimuth','positive unique unknown-cell union','full paid route and return'],
  no_new_score_weights=True,no_training=True,physical_trajectory_count=0,
  limitations='Observed-mesh non-occlusion and no-hit rays do not certify real future visibility or full surface F1. Same endpoint turns do not prove semantic causal gain.')
 dump('protocol.json',protocol)
 cases=[audit_case(manifest,c) for c in manifest['cases']]
 for start in [0,2]:
  left,right=cases[start:start+2]
  for key in ['geometry_packet_sha256','observed_geometry_sha256','initial_map_sha256','candidate_geometry_sha256']:
   require(left[key]==right[key],'Paired initial geometric diagnostic changed: '+key)
  require([x['class_vote'] for x in left['observed_A_B']]==[-x['class_vote'] for x in right['observed_A_B']],'Paired observed labels did not flip')
  for key in ['existing_rows','terminal_heading_diagnostics']:
   def strip(rows):return [{k:v for k,v in row.items() if k not in ['case_index','assignment']} for row in rows]
   require(strip(left[key])==strip(right[key]),'Observed opportunity depends on semantic label')
 summary=[compact(c) for c in cases]
 for c in cases:dump(f'case_{c["case"]["index"]:02d}.json',c)
 flat=[r for c in cases for key in ['existing_rows','terminal_heading_diagnostics'] for r in c[key]]
 columns=list(dict.fromkeys(k for row in flat for k in row))
 with (OUT/'opportunities.csv').open('w',newline='') as stream:
  writer=csv.DictWriter(stream,fieldnames=columns);writer.writeheader()
  for row in flat:writer.writerow({k:json.dumps(v) if isinstance(v,list) else v for k,v in row.items()})
 result=dict(status='complete',scope=protocol,initial_packet_and_candidate_pairing_exact=True,
  class_votes_flip_while_geometry_opportunities_remain_identical=True,summary=summary,
  candidate_selection_uses_future_return=False,future_case_result_opened=False,physical_trajectories=0,semantic_gain_proven=False)
 dump('result.json',result);dump('source_sha256.json',sources)
 with zipfile.ZipFile(OUT/'sources.zip','x',zipfile.ZIP_DEFLATED) as archive:
  for name in sources:archive.write(ROOT/name,name)
 for name,wanted in sources.items():require(sha(ROOT/name)==wanted,'Source changed during initial audit')
 dump('artifact_hashes.json',{p.name:sha(p) for p in OUT.iterdir() if p.is_file() and p.name!='artifact_hashes.json'})
 print(json.dumps(json_value(summary),indent=2),flush=True)

if __name__=='__main__':main()
