"""Test exact process-isolated TSDF forecasts; no policy/robot actions."""
import os
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):os.environ[k]='1'
import copy,hashlib,json,multiprocessing as mp,time
from pathlib import Path
import numpy as np
import open3d as o3d
HERE=Path(__file__).resolve().parent
K=o3d.camera.PinholeCameraIntrinsic(120,90,60.,60.,59.5,44.5)
def volume():return o3d.pipelines.integration.ScalableTSDFVolume(.06,.12,o3d.pipelines.integration.TSDFVolumeColorType.RGB8)
def integrate(v,depth):
 rgb=o3d.geometry.Image(np.full((90,120,3),127,np.uint8));dep=o3d.geometry.Image(np.ascontiguousarray(depth,np.float32));rgbd=o3d.geometry.RGBDImage.create_from_color_and_depth(rgb,dep,depth_scale=1.,depth_trunc=8.,convert_rgb_to_intensity=False);v.integrate(rgbd,K,np.eye(4))
def snapshot(v):
 m=v.extract_triangle_mesh();return (np.asarray(m.vertices).copy(),np.asarray(m.triangles).copy(),np.asarray(m.vertex_colors).copy())
def hashes(arrays):return [hashlib.sha256(a.tobytes()).hexdigest()for a in arrays]
def main():
 past=[np.full((90,120),1.+i*.001,np.float32) for i in range(55)];base=volume()
 for d in past:integrate(base,d)
 before=snapshot(base);out=[];context=mp.get_context('fork')
 for case in range(8):
  future=[np.full((90,120),z,np.float32)for z in [1.01+case*.015,1.04+case*.01,1.+case*.004]]
  if case==3:future=[past[0]]*3
  if case==4:future=[]
  if case==5:
   d=future[0].copy();d[:30,:40]=0;future=[d,*future]
  started=time.monotonic();reader,writer=context.Pipe(duplex=False)
  def child():
   try:
    for depth in future:integrate(base,depth)
    writer.send({'hashes':hashes(snapshot(base))})
   except BaseException as e:writer.send({'error':repr(e)})
   finally:writer.close()
  p=context.Process(target=child);p.start();writer.close()
  if not reader.poll(20):p.terminate();p.join();raise RuntimeError('Forked Open3D TSDF hung')
  result=reader.recv();p.join(5);reader.close();assert p.exitcode==0 and 'error'not in result
  fork_s=time.monotonic()-started;assert hashes(snapshot(base))==hashes(before)
  t=time.monotonic();reference=volume()
  for depth in [*past,*future]:integrate(reference,depth)
  expected=hashes(snapshot(reference));ref_s=time.monotonic()-t;assert result['hashes']==expected,(case,result,expected)
  out.append({'case':case,'future_frames':len(future),'fork_s':fork_s,'fresh_replay_s':ref_s,'exact_vertices_triangles_colors':True,'parent_unchanged':True,'exit_code':p.exitcode})
 result={'status':'all_process_isolated_forecasts_exact','cases':out,'past_frames':len(past),'threads':1,'start_method':'fork','not_Open3D_copy_constructor':True,'source_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'median_fork_s':float(np.median([x['fork_s']for x in out])),'median_fresh_replay_s':float(np.median([x['fresh_replay_s']for x in out]))}
 (HERE/'fork_volume_probe.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result))
if __name__=='__main__':main()
