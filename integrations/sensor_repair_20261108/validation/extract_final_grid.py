#!/usr/bin/env python3
"""Native mapper replay of original 0..13 packets with frozen floor candidate; no ROS/policy."""
import os
os.environ.update(OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1')
import argparse,hashlib,json,pathlib,sys
import numpy as np
p=argparse.ArgumentParser();p.add_argument('--report',type=pathlib.Path,required=True);p.add_argument('--workspace',type=pathlib.Path,required=True);p.add_argument('--profile',type=pathlib.Path,required=True);p.add_argument('--epoch',required=True);p.add_argument('--historical-old-mount-reprojection',action='store_true',help='Explicit cross-epoch hypothetical: old mount on historical raw frames');p.add_argument('--output',type=pathlib.Path,required=True);a=p.parse_args()
if a.output.exists():raise FileExistsError(a.output)
profile=json.loads(a.profile.read_text());assert profile['schema']=='nso.floor_reference_candidate.v1' and profile['epoch']==a.epoch
assert profile['status']=='offline_internal_consistency_only_not_physical_truth_not_deployed'
mode=profile['camera_transform_mode'];assert mode in ('captured','old_mount_counterfactual')
if a.historical_old_mount_reprojection: raise ValueError('this actual-epoch replay uses captured TF only')
mode='captured_actual_legacy_epoch'
W=a.workspace.resolve();sys.path.insert(0,str(W/'nso_bridge'))
from nso_bridge.joint_engine import _MeasuredMapMapper
from nso_bridge.sensor_range_adapter import adapt_scan
from nso_bridge.measured_extent_adapter import install_dispatcher
from env.virtual3d_v2 import VirtualConfigV2
from utils.rgbd_contract import RGBDFrame,PlanarScan
from utils.grid_geometry import inflated_obstacles
sha=lambda p:hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest()
ah=lambda x:hashlib.sha256(np.ascontiguousarray(x).tobytes()).hexdigest()
report=a.report.resolve();cfg=json.loads((report/'joint_configuration.json').read_text())['configuration'];sources={r['path']:r['sha256'] for r in json.loads((report/'source_manifest.json').read_text())['files']}
rels=['nso_bridge/nso_bridge/joint_engine.py','nso_bridge/joint_vendor/nso/mapping3d.py','nso_bridge/joint_vendor/nso/observed_runtime_mapper_v10.py','nso_bridge/nso_bridge/sensor_range_adapter.py']
for r in rels:assert sha(W/r)==sources[r],r
rows=[json.loads(x) for x in (report/'joint/decisions.jsonl').read_text().splitlines()];obs=[x for x in rows if x['request']['method']=='observe' and x['response'].get('ok')];assert [x['request']['params']['action_id'] for x in obs]==list(range(23))
# The profile's training/heldout packets must be part of this exact sequence; epoch is explicit.
packet_hashes=[sha(x['request']['params']['path']) for x in obs]
assert profile['source_training']['sha256'] not in packet_hashes, 'pretrial training packet must differ from episode packets'
params=dict(cfg['sensor_config']);params.update(resolution_m=.2,robot_radius_m=cfg['robot_radius_m']);c=VirtualConfigV2(**params);install_dispatcher();old=None
original=np.asarray(cfg['planning_from_odom'],dtype=float);candidate=original.copy()
with np.load(obs[0]['request']['params']['path'],allow_pickle=False) as z: base_z=float(np.asarray(json.loads(str(z['receipt_json']))['historical_tf']['odom_from_base']['matrix'])[2,3])
assumed_floor=base_z+float(profile['floor_relative_to_training_base_m']);candidate[2,3]=-assumed_floor;assert np.array_equal(candidate[:2,:],original[:2,:])
ms=[]
for T in (original,candidate):
 m=_MeasuredMapMapper(tuple(cfg['shape']),c);m.sensor_range_profile=cfg['sensor_range_profile'];m.measured_extent_profile=cfg['measured_extent_profile'];ms.append(m)
result=[]
for i,row in enumerate(obs):
 path=pathlib.Path(row['request']['params']['path']);rec=row['response']['result']['summary']['semantic_diagnostics']['receipt'];assert sha(path)==rec['source_file_sha256'];mask=pathlib.Path(rec['mask_file']);assert sha(mask)==rec['mask_file_sha256']
 with np.load(mask,allow_pickle=False) as z: labels=z['semantic'].copy()
 assert ah(labels)==rec['canonical_semantic_sha256']
 with np.load(path,allow_pickle=False) as z:
  actual_camera=z['world_from_camera'].copy();camera=actual_camera.copy();laser=z['world_from_laser'].copy();nav=z['joint_belief_grid'].copy();depth=z['depth'].copy();color=z['color_rgb'].copy();K=z['intrinsic'].copy();ranges=z['ranges'].copy();rtime=float(z['rgbd_timestamp_s']);stime=float(z['scan_timestamp_s']);amin=float(z['angle_min']);inc=float(z['angle_increment']);rmax=float(z['range_max'])
  if old is not None: camera=np.asarray(json.loads(str(z['receipt_json']))['historical_tf']['odom_from_base']['matrix'])@old
 for m,T,cam in zip(ms,(original,candidate),(actual_camera,actual_camera)):
  f=RGBDFrame(rtime,depth,color,K,T@cam,labels);scan=PlanarScan(stime,ranges,amin,inc,rmax,T@laser);scan,_=adapt_scan(scan,cfg['sensor_range_profile']);m.input_observed_grid=nav;m.update(f,scan)
 x,y=(m.belief.copy() for m in ms);anchor=tuple(cfg['return_anchor'][:2]);cur=tuple(row['response']['result']['state']['pose'][:2]);
 from scipy.ndimage import label
 def topology(b):
  safe=~inflated_obstacles(b!=0,c.robot_radius_m/c.resolution_m);lab,_=label(safe,structure=np.array([[0,1,0],[1,1,1],[0,1,0]]));return {'anchor_safe':bool(safe[anchor]),'current_safe':bool(safe[cur]),'connected':bool(lab[anchor] and lab[anchor]==lab[cur]),'anchor_component_cells':int(np.count_nonzero(lab==lab[anchor])) if lab[anchor] else 0}
 result.append(dict(action_id=i,packet_sha256=packet_hashes[i],baseline_topology=topology(x),candidate_topology=topology(y),baseline_belief_sha256=ah(x),candidate_belief_sha256=ah(y),baseline_counts={str(k):int((x==k).sum()) for k in (-1,0,1)},candidate_counts={str(k):int((y==k).sum()) for k in (-1,0,1)},transitions={f'{j}_to_{k}':int(((x==j)&(y==k)).sum()) for j in (-1,0,1) for k in (-1,0,1)},nav_sha256=ah(nav)))
expected={}
for rec in rows:
 if rec['request']['method']=='observe':aid=rec['request']['params']['action_id']
 if rec['request']['method']=='next_action' and rec['response']['result'].get('selection'):
  audit=rec['response']['result']['selection']['candidate_audit']
  if audit['remaining_budget']==cfg['total_budget']-aid:expected[aid]=audit['belief_sha256']
for aid,digest in expected.items():assert result[aid]['baseline_belief_sha256']==digest,(aid,digest)
out=dict(logged_actual_belief_hash_checks=expected,schema='nso.floor_mapper_counterfactual.v1',profile_sha256=sha(a.profile),mode=mode,epoch=a.epoch,report=str(report),source_manifest_sha256=sha(report/'source_manifest.json'),source_sha256={r:sha(W/r) for r in rels},baseline_floor_odom_z_m=-float(original[2,3]),candidate_floor_odom_z_m=assumed_floor,base_relative_floor_m=float(profile['floor_relative_to_training_base_m']),frames=result,scope='Native mapper only, same 23 actual raw inputs/cached labels and fixed thresholds; no detector, planner, ROS or runtime changes.',limits=['Floor candidate is sensor-internal consistency, not independent physical truth; cross-epoch replay assumes the same base-relative floor height and old mount. ','Original occupied measurements are preserved; transition counts do not certify free space.','No scientific or physical success is inferred from changed cells.'])
np.savez_compressed(a.output.with_suffix('.npz'),baseline=x,candidate=y,nav=nav)
a.output.write_text(json.dumps(out,indent=2)+'\n');print(json.dumps({'frames':len(result),'profile_sha256':out['profile_sha256']}))
