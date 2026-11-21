"""Validate exact process snapshot on real acquired L54, no new study outcomes."""
import os
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ[k]='1'
import sys,json,time,hashlib,importlib.util
from pathlib import Path
import numpy as np
HERE=Path(__file__).resolve().parent;BASE=HERE.parents[1]
def load(n,p):
 s=importlib.util.spec_from_file_location(n,p);m=importlib.util.module_from_spec(s);sys.modules[n]=m;s.loader.exec_module(m);return m
start=time.monotonic();adapter=load('B11_fork_L54_adapter',BASE/'dual_template_gain_alignment_20261120/execution_audit/prefix_adapter.py');c=adapter.rebuild_observed_prefix('shelf_left',54)
model=load('B11_fork_L54_model',BASE/'dual_template_uncertainty_20261120/method/model.py');bank=model.build_particle_bank(c.mapper);data=bank.materialize('uncertainty')
from full_posterior_forecast import CORE,RENDER,digest
fork=load('B11_fork_verification',HERE.parent/'performance/fork_forecast.py')
from env.virtual3d import camera_pose
past=[CORE.DepthFrame(f.depth_m,f.world_from_camera,f.intrinsic,'acquired')for f in c.acquired_frames];snapshot=fork.ForkSnapshot(past);native=CORE.ray_scene(c.mapper.mesh());rows=[]
for hi in (0,1):
 item=bank.objects[0];v=item['variants']['uncertainty'];p=v['particles'][0];shape=item['geometry'][(p['depth_m'],p['lower_m'])][hi]
 oi,ki,_,lo,end=next(s for s in data['slices']if s[:3]==(0,0,hi));points=data['points'][lo:end];area=data['areas'][lo:end]*(~data['history']['strict_free'][lo:end]);d0=snapshot.before_distance(points)
 for route in c.routes[:2]:
  future=[]
  for state in route['outbound_states'][1:]:
   pose=camera_pose(state[:2],state[2],c.mapper.config,c.mapper.shape[0]);depth,audit=RENDER.corrected_depth(shape['scene'],native,pose,past[0].intrinsic,120,90,c.mapper.keyframes)
   future.append(CORE.DepthFrame(depth,pose,past[0].intrinsic,'predicted_hypothesis'))
  a=snapshot.forecast(future,points,area,before_distance=d0);b=CORE.forecast(past,future,points,area)
  for key in ('gain','loss','before_distance','after_distance'):np.testing.assert_array_equal(a[key],b[key])
  assert a['receipt']['after_mesh_hashes']['vertices']==b['receipt']['after_vertices_sha256'];assert a['receipt']['after_mesh_hashes']['triangles']==b['receipt']['after_triangles_sha256']
  rows.append(dict(shape_id=hi,candidate_id=int(route['candidate_id']),future_frames=len(future),exact_gain_loss_distances_and_mesh=True,fork_seconds=a['receipt']['elapsed_s'],fresh_replay_seconds=b['receipt']['elapsed_s'],after_mesh_sha256=a['receipt']['after_mesh_hashes']))
parent=snapshot.verify_parent_unchanged()
receipt=dict(status='four_real_prefix_exact_fork_forecasts_passed',layout='old_shelf_left',paid_actions=54,rows=rows,parent=parent,new_study_outcomes=False,elapsed_s=time.monotonic()-start,code_sha256={str(p):hashlib.sha256(p.read_bytes()).hexdigest()for p in (Path(__file__),HERE.parent/'performance/fork_forecast.py',HERE/'full_posterior_forecast.py')})
(HERE/'fork_existing_prefix_validation.json').write_text(json.dumps(receipt,indent=2)+'\n');print(json.dumps(receipt,indent=2))
