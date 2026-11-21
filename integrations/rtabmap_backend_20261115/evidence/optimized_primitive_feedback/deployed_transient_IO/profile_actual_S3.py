#!/usr/bin/env python3
"""One cached real S3 profile; no model inference, actuator, or active edits."""
import copy,cProfile,hashlib,io,json,os,pstats,sys,time
from pathlib import Path
import numpy as np
O=Path('/home/ouc/nso_robot/analyses/resume_20261115/primitive_feedback_optimized_replay');W=Path('/home/ouc/nso_robot/workspaces/nso_hybrid_20261029');B=Path(__file__).resolve().parent
sys.path.insert(0,str(W/'nso_bridge'));sys.path.insert(0,str(O))
import nso_bridge
from nso_bridge import joint_engine
nso_bridge.__path__=[str(O/'candidate')]+list(nso_bridge.__path__)
from actual_history_fixture import build_original_actual_pilot,REPORT
from nso_bridge.primitive_geometry_rebase_v1 import rebuild_geometry_preserving_actual_feedback,stats_fingerprint
from nso_bridge.rtabmap_geometry import corrected_geometry
import open3d as o3d

def cloud_hash(volume):
 cloud=volume.extract_point_cloud();return hashlib.sha256(np.asarray(cloud.points).tobytes()+np.asarray(cloud.colors).tobytes()).hexdigest()
# Populated TSDF clone independence: different RGB over the same exact depth.
v=o3d.pipelines.integration.ScalableTSDFVolume(.03,.12,o3d.pipelines.integration.TSDFVolumeColorType.RGB8)
d=np.ones((48,64),np.float32);a=np.zeros((48,64,3),np.uint8);a[:,:,1]=255;c=np.zeros_like(a);c[:,:,0]=255
intr=o3d.camera.PinholeCameraIntrinsic(64,48,60.,60.,31.5,23.5)
def integrate(vol,rgb):
 frame=o3d.geometry.RGBDImage.create_from_color_and_depth(o3d.geometry.Image(rgb),o3d.geometry.Image(d),depth_scale=1.,depth_trunc=4.,convert_rgb_to_intensity=False);vol.integrate(frame,intr,np.eye(4))
integrate(v,a);original_hash=cloud_hash(v);clone=copy.deepcopy(v);integrate(clone,c)
clone_test=dict(type=str(type(v)),copy_supported=True,original_before=original_hash,original_after=cloud_hash(v),clone_after=cloud_hash(clone),deepcopy_independent=(cloud_hash(v)==original_hash),synthetic_probe_only=True)
start=time.monotonic();pilot,observations,cache,settings=build_original_actual_pilot(B/'source_fixture');build_elapsed=time.monotonic()-start
last=observations[-1]['request']['params']['path']
with np.load(last,allow_pickle=False)as z:
 metadata=json.loads(str(z['joint_metadata_json'].item()));snapshot=metadata['rtabmap_graph_snapshot'];grid=z['joint_belief_grid'].copy()
mapper=pilot.inner.mapper
state_types={k:str(type(v)) for k,v in mapper.__dict__.items()}
start=time.monotonic()
try:mapper_clone=copy.deepcopy(mapper);mapper_copy=dict(supported=True,elapsed_s=time.monotonic()-start)
except Exception as exc:mapper_copy=dict(supported=False,error=type(exc).__name__+':'+str(exc))
# Count exactly unchanged canonical geometry across this actual graph evolution.
packets=[]
for record in pilot.records:
 with np.load(record['original_path'],allow_pickle=False)as z:
  arrays={k:z[k].copy()for k in ['receipt_json','joint_metadata_json']}
 meta=json.loads(str(arrays['joint_metadata_json'].item()));old_updates,_=corrected_geometry(arrays,meta['rtabmap_graph_snapshot']);new_updates,_=corrected_geometry(arrays,snapshot)
 packets.append(dict(action_id=record['action_id'],all_three_matrices_exact=all(np.array_equal(old_updates[k],new_updates[k]) for k in old_updates),matrix_max_delta={k:float(np.max(np.abs(old_updates[k]-new_updates[k]))) for k in old_updates}))
profile=cProfile.Profile();start=time.monotonic();profile.enable()
new,receipt=rebuild_geometry_preserving_actual_feedback(pilot,snapshot,grid,output=B/'profile_full_rebuild',semantic_cache=cache)
profile.disable();elapsed=time.monotonic()-start
assert stats_fingerprint(new.inner)==stats_fingerprint(pilot.inner)
s=pstats.Stats(profile);entries=[]
for (filename,line,name),(cc,nc,tt,ct,callers)in s.stats.items():entries.append(dict(file=filename,line=line,name=name,primitive_calls=cc,calls=nc,self_s=tt,cumulative_s=ct))
entries.sort(key=lambda x:-x['cumulative_s'])
result=dict(schema='actual_S3_geometry_replay_performance_profile/1',source_report=str(REPORT),actual_original_packets=len(pilot.records),new_measurement_or_feedback_or_inference=False,actuation=False,active_source_edits=False,affinity=sorted(os.sched_getaffinity(0)),source_fixture_reconstruction_s=build_elapsed,profiled_full_rebuild_s=elapsed,TSDF_clone_probe=clone_test,mapper_copy=mapper_copy,mapper_attribute_types=state_types,actual_canonical_geometry_equality=packets,feedback_fingerprint_exact=True,top_cumulative_functions=entries[:35],top_self_functions=sorted(entries,key=lambda x:-x['self_s'])[:20],no_model_reinference=True,limits='One actual cached four-frame profile; cProfile overhead and current CPUload affect elapsedtime. Copy construction alone doesnot prove safe independent checkpointing.')
(B/'profile_receipt.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps({k:v for k,v in result.items()if k not in ['top_cumulative_functions','top_self_functions','mapper_attribute_types']},indent=2));print('TOP',json.dumps(entries[:18],indent=2))
