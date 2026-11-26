#!/usr/bin/env python3
"""One paired cached-realS3 replay benchmark; no model/ROS/actuator use."""
import copy,hashlib,importlib.util,json,math,os,sys,time
from pathlib import Path
import numpy as np
O=Path('/home/ouc/nso_robot/analyses/resume_20261115/primitive_feedback_optimized_replay');W=Path('/home/ouc/nso_robot/workspaces/nso_hybrid_20261029');B=Path(__file__).resolve().parent
sys.path.insert(0,str(W/'nso_bridge'));sys.path.insert(0,str(O))
import nso_bridge
from nso_bridge import joint_engine
nso_bridge.__path__=[str(O/'candidate')]+list(nso_bridge.__path__)
from actual_history_fixture import build_original_actual_pilot,REPORT
from nso_bridge.primitive_geometry_rebase_v1 import rebuild_geometry_preserving_actual_feedback as baseline,stats_fingerprint
p=B/'candidate/primitive_geometry_rebase_transient_io_v1.py';s=importlib.util.spec_from_file_location('transient_rebase',p);fast=importlib.util.module_from_spec(s);s.loader.exec_module(fast)
pilot,observations,cache,settings=build_original_actual_pilot(B/'source_fixture_IO_comparison')
with np.load(observations[-1]['request']['params']['path'],allow_pickle=False)as z:
 meta=json.loads(str(z['joint_metadata_json'].item()));snapshot=meta['rtabmap_graph_snapshot'];grid=z['joint_belief_grid'].copy()
originals={r['original_path']:hashlib.sha256(Path(r['original_path']).read_bytes()).hexdigest()for r in pilot.records}
start=time.monotonic();old,old_receipt=baseline(pilot,snapshot,grid,output=B/'IO_compare_baseline',semantic_cache=cache);old_s=time.monotonic()-start
start=time.monotonic();new,new_receipt=fast.rebuild_geometry_preserving_actual_feedback(pilot,snapshot,grid,output=B/'IO_compare_candidate',semantic_cache=cache);new_s=time.monotonic()-start
assert stats_fingerprint(pilot.inner)==stats_fingerprint(old.inner)==stats_fingerprint(new.inner)
assert old.inner.backend.scenes[0]['ledger'].snapshot()==new.inner.backend.scenes[0]['ledger'].snapshot()
def equal(a,b):
 if isinstance(a,np.ndarray):return isinstance(b,np.ndarray)and a.dtype==b.dtype and a.shape==b.shape and a.tobytes()==b.tobytes()
 if isinstance(a,dict):return a.keys()==b.keys()and all(equal(a[k],b[k])for k in a)
 if isinstance(a,(list,tuple)):return len(a)==len(b)and all(equal(x,y)for x,y in zip(a,b))
 return a==b
mapper_keys=['belief','visible','camera_seen','surface','quality','frames','current_footprint_conflict','current_footprint_conflict_details','current_footprint_conflict_count','input_observed_grid']
for key in mapper_keys:assert equal(getattr(old.inner.mapper,key),getattr(new.inner.mapper,key)),key
for a,b in zip(old.inner.mapper.keyframes,new.inner.mapper.keyframes):
 for key in ['timestamp_s','depth_m','color_rgb','intrinsic','world_from_camera','semantic']:assert equal(getattr(a,key),getattr(b,key)),key
clouds=[x.inner.mapper.volume.extract_voxel_point_cloud()for x in [old,new]]
for key in ['points','colors']:assert equal(np.asarray(getattr(clouds[0],key)),np.asarray(getattr(clouds[1],key))),key
manifest=Path(new_receipt['derived_geometry_storage']['manifest']);m=json.loads(manifest.read_text());archive_tests=[]
for i,item in enumerate(m['archives']):
 assert not Path(item['transient_path']).exists()
 regenerated=B/'IO_compare_candidate'/f'regenerated_test_{i:06d}.npz';rr=fast.regenerate_transient_archive(manifest,i,regenerated);assert rr['encoded_byte_identical']
 base=B/'IO_compare_baseline'/f'geometry_{i:06d}.npz'
 with np.load(base,allow_pickle=False)as a,np.load(regenerated,allow_pickle=False)as b:
  assert a.files==b.files
  for k in a.files:assert equal(a[k],b[k]),(i,k)
 archive_tests.append(dict(frame=i,array_count=len(a.files),source_original_unchanged=True,regenerated_stored_zip_byte_identical=True,compressed_baseline_zip_sha256=hashlib.sha256(base.read_bytes()).hexdigest(),stored_zip_sha256=item['transient_sha256'],ZIP_hash_expected_to_differ=True))
 regenerated.unlink()
# Existing selected option still has a route, so these predictions do not call
# a new expensive global chooser or invent any measured observation.
a=old.next_action();b=new.next_action()
for key in ['action','pose','target_odom_pose','target_prediction_epoch_pose','assessment']:assert equal(a['pending'][key],b['pending'][key]),key
pa=old.inner.backend.scenes[0]['gain']._pending;pb=new.inner.backend.scenes[0]['gain']._pending
assert equal(pa,pb)
for path,sha in originals.items():assert hashlib.sha256(Path(path).read_bytes()).hexdigest()==sha
retained=sum(p.stat().st_size for p in (B/'IO_compare_candidate').glob('*')if p.is_file()and p.name not in ['prediction_inputs_000004.npz'])
base_archive_bytes=sum(p.stat().st_size for p in (B/'IO_compare_baseline').glob('geometry_*.npz'))
receipt=dict(schema='transient_geometry_IO_actual_S3_paired_check/1',success=True,source_report=str(REPORT),source_actual_frames=4,new_actual_measurements=False,new_gain_events=False,model_reinference=False,robot_actuation=False,active_source_edits=False,affinity=sorted(os.sched_getaffinity(0)),baseline_rebuild_s=old_s,candidate_rebuild_s=new_s,speedup_ratio=old_s/new_s,feedback_events_and_statistics_exact=True,geometry_derived_ledger_snapshot_exact=True,mapper_fields_exact=mapper_keys,keyframe_arrays_exact=True,TSDF_voxel_points_and_colors_byte_exact=True,next_original_action_goal_assessment_and_gain_masks_exact=True,all_original_source_files_SHA_unchanged=True,source_original_archive_count=len(originals),derived_archive_checks=archive_tests,old_compressed_geometry_bytes_retained=base_archive_bytes,new_transient_geometry_archives_retained=0,new_manifest_and_receipt_bytes=retained,peak_transient_geometry_archive_bytes=max(x['transient_bytes']for x in m['archives']),no_TSDF_clone=True,computed_full_original_history=True,source_sha256=hashlib.sha256(p.read_bytes()).hexdigest(),limits='One pairedcachedactualS3 software replay, not new independentrobot samples. Sourcezip identities differ for transient serialization; all canonicalarrays/order and frozenactualfeedback checked. Current O(N)fusion remains, O(N²)large derivedimage retention removed.')
(B/'transient_IO_comparison_receipt.json').write_text(json.dumps(receipt,indent=2)+'\n');print(json.dumps({k:v for k,v in receipt.items()if k!='derived_archive_checks'},indent=2))
