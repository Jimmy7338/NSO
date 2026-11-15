import sys,time,json,copy,hashlib
from pathlib import Path
import numpy as np
W=Path('/home/ouc/nso_robot/workspaces/nso_hybrid_20261029');O=Path(__file__).parent;sys.path.insert(0,str(W/'nso_bridge'));sys.path.insert(0,str(O/'candidate'))
from nso_bridge import joint_engine
from nso_bridge.rtabmap_viewpoint_session import RtabmapViewpointSession
from nso_bridge.exact_candidate_batch_v1 import install as install_batch
from nso_bridge.exact_visibility_counts import ExactVisibilityCounts as Original
from exact_visibility_parallel_v1 import ParallelExactVisibilityCounts,install as install_parallel
from nso_bridge.sensor_range_adapter import lidar_range_cells
from utils.grid_geometry import inflated_obstacles
R=Path('/home/ouc/nso_robot/reports/nso-hybrid-20261115/NSO_rtabmap_integrated_20261115_02')
lines=[json.loads(l) for l in (R/'joint/decisions.jsonl').read_text().splitlines()]
settings=copy.deepcopy(lines[0]['request']['params']);settings.pop('execution_allowed',None);saved_semantic=copy.deepcopy(settings['semantic_configuration']);settings['semantic_configuration']=None
first=next(x for x in lines if x['request']['method']=='observe_viewpoint_result')['request']['params']
baseline=next(x for x in lines if x['request']['method']=='next_viewpoint')['response']['result']
packet=Path(first['path'])
with np.load(packet,allow_pickle=False)as z:meta=json.loads(str(z['joint_metadata_json'].item()))
fid=str(meta['frame_id']);receipt=None
for f in (R/'joint/semantics').glob('*.json'):
 r=json.loads(f.read_text())
 if str(r['frame_id'])==fid:receipt=r;break
if receipt is None:raise ValueError('No real first-frame cached semantic receipt')
with np.load(receipt['mask_file'],allow_pickle=False)as z:labels=z['semantic'].copy()
install_batch('exact_candidate_batch_v1');session=RtabmapViewpointSession(**settings)
session.semantic_configuration=saved_semantic
session.semantic_cache[fid]=dict(labels=labels,owner=None,receipt=receipt)
started=time.monotonic();session.observe_viewpoint_result(**first);observe_s=time.monotonic()-started
s=session.inner.backend.scenes[0];mapper=s['mapper'];gain=s['gain'];history=s['ledger'].planning_camera_poses();attempted=gain.attempted_camera_mask(mapper,history)
belief=np.asarray(mapper.belief);occ=belief==1;radius=mapper.config.robot_radius_m/mapper.config.resolution_m
radar=(belief==-1)&~inflated_obstacles(occ,radius);camera=(belief==-1)&~np.asarray(mapper.camera_seen,bool)&~attempted
args=(occ,radar,camera,lidar_range_cells(mapper),int(mapper.config.max_depth_m/mapper.config.resolution_m),mapper.config.fov_deg)
receipt_out=dict(scope='Actual NSO02 first packet frozen; no ROS/inference/new motion; original eight full tables and original total candidate domain',source_packet=str(packet),source_sha256=hashlib.sha256(packet.read_bytes()).hexdigest(),source_frame=fid,semantic_mask_sha256=receipt['mask_file_sha256'],shape=list(belief.shape),radii=[args[3],args[4]],fov=args[5],occupancy_count=int(occ.sum()),observe_s=observe_s,worker_threads=3,cpu_affinity=sorted(__import__('os').sched_getaffinity(0)))
def save(): (O/'actual_receipt.json').write_text(json.dumps(receipt_out,indent=2,allow_nan=False)+'\n')
save();print('START_ORIGINAL_TABLES',flush=True)
started=time.monotonic();old=Original(*args);receipt_out['original_tables_s']=time.monotonic()-started;save();print('ORIGINAL_TABLES',receipt_out['original_tables_s'],flush=True)
started=time.monotonic();new=ParallelExactVisibilityCounts(*args);receipt_out['parallel_tables_s']=time.monotonic()-started
checks={}
for key in old.tables:
 a,b=old.tables[key],new.tables[key]
 exact=all(np.array_equal(a[i],b[i])for i in [0,1,4])and a[2:4]==b[2:4]
 checks[str(key)]=dict(exact=exact,counts_sha256=hashlib.sha256(b[0].tobytes()).hexdigest(),hits_sha256=hashlib.sha256(b[1].tobytes()).hexdigest())
 if not exact:raise ValueError('actual full table mismatch '+str(key))
receipt_out.update(full_eight_tables_exact=True,table_checks=checks,parallel_table_job_s=new.table_build_seconds);save();print('PARALLEL_TABLES',receipt_out['parallel_tables_s'],flush=True)
del old,new
install_parallel();started=time.monotonic();result=session.next_viewpoint(remaining_time_s=None,return_requested=False);receipt_out['full_candidate_choose_s']=time.monotonic()-started
receipt_out['full_selection_exact_equal']=result['selection']==baseline['selection'];receipt_out['full_target_exact_equal']=result['target']==baseline['target']
receipt_out['candidate_count']=len(result['selection']['candidates']);receipt_out['selected_target']=result['target']['target_odom_pose'];receipt_out['full_selection_sha256']=hashlib.sha256(json.dumps(result['selection'],sort_keys=True,allow_nan=False).encode()).hexdigest();receipt_out['baseline_selection_sha256']=hashlib.sha256(json.dumps(baseline['selection'],sort_keys=True,allow_nan=False).encode()).hexdigest();receipt_out['original_policy_or_guard_modified']=False
(O/'actual_parallel_result.json').write_text(json.dumps(result,indent=2));save();print(json.dumps({k:v for k,v in receipt_out.items()if k not in ['table_checks','parallel_table_job_s']},indent=2),flush=True)
if not receipt_out['full_selection_exact_equal'] or not receipt_out['full_target_exact_equal']:raise ValueError('full chooser output differs from actual original live first selection')
