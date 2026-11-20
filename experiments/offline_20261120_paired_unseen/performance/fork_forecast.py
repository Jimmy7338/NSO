"""Exact Linux process-snapshot backend for prediction-only TSDF fusion.

Open3D copy/deepcopy aliases internal volume units. This helper instead
uses process copy-on-write: a child receives the acquired volume, mutates only
its process-local pages, and returns mesh hashes/distances. No physics or
reward approximation, future-frame deletion, or posterior truncation.
"""
from __future__ import annotations
from pathlib import Path
import hashlib,importlib.util,multiprocessing as mp,sys,time
import numpy as np
CORE=Path('/home/ouc/nso_robot/analyses/dual_template_gain_alignment_20261120/geometry_audit/forecast_development/acquisition_forecast.py')
CORE_SHA='0a912e44f474d6a97c62c674f822e3b9b216bddc30e4b29c91ac07a3c646d259'
assert hashlib.sha256(CORE.read_bytes()).hexdigest()==CORE_SHA
_s=importlib.util.spec_from_file_location('B11_frozen_C_core',CORE);core=importlib.util.module_from_spec(_s);sys.modules[_s.name]=core;_s.loader.exec_module(core)

def mesh_hashes(mesh):return {name:core.digest(np.asarray(getattr(mesh,name)))for name in ('vertices','triangles','vertex_colors')}
def append_frames(volume,frames,max_depth_m=8.):
 import open3d as o3d
 for f in frames:
  k=np.asarray(f.intrinsic);h,w=f.depth_m.shape;d=np.where(np.isfinite(f.depth_m)&(f.depth_m>0),f.depth_m,0).astype(np.float32)
  rgb=o3d.geometry.Image(np.full((h,w,3),127,np.uint8));rgbd=o3d.geometry.RGBDImage.create_from_color_and_depth(rgb,o3d.geometry.Image(np.ascontiguousarray(d)),depth_scale=1.,depth_trunc=max_depth_m,convert_rgb_to_intensity=False)
  intrinsic=o3d.camera.PinholeCameraIntrinsic(w,h,k[0,0],k[1,1],k[0,2],k[1,2]);volume.integrate(rgbd,intrinsic,np.linalg.inv(f.world_from_camera))

def make_volume(past,voxel_m,truncation_m,max_depth_m):
 import open3d as o3d
 v=o3d.pipelines.integration.ScalableTSDFVolume(voxel_length=voxel_m,sdf_trunc=truncation_m,color_type=o3d.pipelines.integration.TSDFVolumeColorType.RGB8)
 append_frames(v,past,max_depth_m);return v

class ForkSnapshot:
 def __init__(self,past_frames,*,voxel_m=.06,truncation_m=.12,max_depth_m=8.,tolerance_m=.05,worker_timeout_s=120.):
  if 'fork'not in mp.get_all_start_methods():raise RuntimeError('Exact fork backend requires Linux/POSIX fork')
  for f in past_frames:core.validate_frame(f,'acquired')
  if not np.isfinite(worker_timeout_s)or worker_timeout_s<=0:raise ValueError('positive finite computational worker timeout required')
  self.worker_timeout_s=float(worker_timeout_s)
  self.context=mp.get_context('fork');self.past=tuple(past_frames);self.past_hash=[core.frame_hash(f)for f in self.past]
  self.voxel=voxel_m;self.truncation=truncation_m;self.max_depth=max_depth_m;self.tolerance=tolerance_m
  self._volume=make_volume(self.past,voxel_m,truncation_m,max_depth_m);self.before=self._volume.extract_triangle_mesh();self.before_hashes=mesh_hashes(self.before);self.calls=0
 def before_distance(self,points):return core.distance(self.before,points)
 def forecast(self,future_frames,points,area_weights,before_distance=None):
  pts=np.asarray(points,float);areas=np.asarray(area_weights,float)
  if pts.ndim!=2 or pts.shape[1]!=3 or areas.shape!=(len(pts),)or not np.isfinite(pts).all()or not np.isfinite(areas).all()or np.any(areas<0):raise ValueError('aligned finite surface points and nonnegative weights required')
  for f in future_frames:core.validate_frame(f,'predicted_hypothesis')
  hashes=[core.frame_hash(f)for f in future_frames];d0=self.before_distance(pts)if before_distance is None else np.asarray(before_distance)
  if d0.shape!=(len(pts),)or np.isnan(d0).any():raise ValueError('before-distance vector must match surface basis')
  reader,writer=self.context.Pipe(duplex=False);begun=time.monotonic()
  def child():
   try:
    append_frames(self._volume,future_frames,self.max_depth);mesh=self._volume.extract_triangle_mesh()
    writer.send(dict(distance=core.distance(mesh,pts),mesh_hashes=mesh_hashes(mesh),frame_hashes=[core.frame_hash(f)for f in future_frames]))
   except BaseException as e:writer.send(dict(error=repr(e)))
   finally:writer.close()
  process=self.context.Process(target=child);process.start();writer.close()
  result=None;fallback_reason=None
  try:
   if reader.poll(self.worker_timeout_s):
    try:result=reader.recv()
    except EOFError:fallback_reason='child_pipe_closed_without_result'
   else:fallback_reason='computational_child_timeout'
  finally:reader.close()
  if fallback_reason is not None:
   process.terminate();process.join(2)
   if process.is_alive():process.kill();process.join()
  else:
   process.join()
   if process.exitcode!=0 or 'error'in result:fallback_reason=f'child_failure_{process.exitcode}: {result.get("error")}'
  # A computation failure retries exactly the same observations/formula.
  # It never skips a candidate, modifies a posterior, or stops navigation.
  if fallback_reason is not None:
   fresh=make_volume(self.past,self.voxel,self.truncation,self.max_depth)
   append_frames(fresh,future_frames,self.max_depth);fresh_mesh=fresh.extract_triangle_mesh()
   result=dict(distance=core.distance(fresh_mesh,pts),mesh_hashes=mesh_hashes(fresh_mesh),frame_hashes=[core.frame_hash(f)for f in future_frames])
  assert hashes==result['frame_hashes'];d1=result['distance'];tau=np.float32(self.tolerance);b=d0<=tau;a=d1<=tau;gain=areas*(~b&a);lost=areas*(b&~a);self.calls+=1
  return dict(gain=gain,loss=lost,before_distance=d0,after_distance=d1,receipt=dict(schema='B11_exact_process_snapshot_forecast_v1',parent_mesh_hashes=self.before_hashes,after_mesh_hashes=result['mesh_hashes'],past_frames=len(self.past),future_frames=len(future_frames),predicted_gross_new_m2=float(gain.sum()),predicted_lost_m2=float(lost.sum()),predicted_net_new_m2=float(gain.sum()-lost.sum()),backend='fork_copy_on_write'if fallback_reason is None else'exact_fresh_replay_fallback',computational_fallback_reason=fallback_reason,computational_worker_timeout_s=self.worker_timeout_s,new_TSDF_never_shared_back_to_parent=True,positive_mass_truncation=False,future_frame_sequence_preserved=True,elapsed_s=time.monotonic()-begun))
 def verify_parent_unchanged(self):
  assert mesh_hashes(self._volume.extract_triangle_mesh())==self.before_hashes
  assert [core.frame_hash(f)for f in self.past]==self.past_hash
  return dict(status='parent_volume_and_acquired_frames_unchanged',calls=self.calls)
