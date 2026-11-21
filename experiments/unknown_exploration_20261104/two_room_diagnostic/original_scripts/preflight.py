import os
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ[k]='1'
import sys,json,hashlib,time
from pathlib import Path
from dataclasses import asdict
import numpy as np
from scipy.ndimage import label
D=Path(__file__).parent;S=Path('/home/ouc/nso_robot/analyses/reproduction_20261104/simulation_publication_stage/experiments/unknown_exploration_20261104/source');sys.path.insert(0,str(S))
from two_room_world import TwoRoomWorld,WALL_BOXES
from env.virtual3d_competition_v9 import create_competition_world
from nso.cpu_sensor_contract_v10 import SensorPacket,GridTransform
from nso.observed_runtime_mapper_v10 import ObservedRuntimeMapperV10
from nso.execution_guard_v8 import ObservedExecutionGuard
from utils.reconstruction_metrics import ReconstructionEvaluator
import open3d as o3d
from scripts.eval_counterfactual_views import coverage
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest();started=time.monotonic();w=TwoRoomWorld();tr=GridTransform(tuple(w.shape),w.config.resolution_m);f=w.sense();sc=w.scan()
p=SensorPacket('TWO_ROOM_01','fresh-first-frame','frame-0',0,f,sc,tuple(map(int,w.position)),int(w.heading),'deterministic_simulator_rgbd_and_scan','simulator_exact_discrete_odometry',None,False,False).validate(tr,w.config)
m=ObservedRuntimeMapperV10(w.shape,w.config);m.update(f,sc);g=ObservedExecutionGuard(w.config.resolution_m,w.config.robot_radius_m);anchor=(*map(int,w.position),int(w.heading));assert g.safe_grid(m.belief)[w.position];assert g.return_plan(m.belief,w.position,w.heading,anchor,96).available
# Divider row fully blocked after sealing doorway: count disconnected start vs north.
row=w._cell(1.,3.7)[0];sealed=w._blocked.copy();sealed[row,:]=True;comp,n=label(~sealed);south=int(comp[w.start]);north_candidates=np.argwhere((~sealed)&(np.indices(w.shape)[0]<row));north_ids=set(comp[tuple(north_candidates.T)]);assert south and south not in north_ids
assert w.reachable[:row,:].sum() > 100 and w.reachable[row+1:,:].sum() > 100
old=create_competition_world('Q0','shelf_west');assert np.any(w.occupancy!=old.occupancy)
# Same fixed physical wall must be in the camera and lidar ray scene and evaluator mesh.
rays=o3d.core.Tensor(np.array([[1.1,2.9,.25,0,1,0],[1.1,2.9,.8,0,1,0]],dtype=np.float32))
sensor_hits=w._ray.cast_rays(rays,nthreads=1)['t_hit'].numpy();old_hits=old._ray.cast_rays(rays,nthreads=1)['t_hit'].numpy()
evaluator=ReconstructionEvaluator(w,count=100,seed=2026);eval_hits=evaluator.truth.cast_rays(rays,nthreads=1)['t_hit'].numpy()
assert np.allclose(sensor_hits,.7,atol=1e-5) and np.allclose(eval_hits,sensor_hits,atol=1e-5)
assert np.all(old_hits>sensor_hits+.1) and w._blocked[w._cell(1.1,3.5)]

files=sorted(p for p in S.rglob('*') if p.is_file() and p.suffix in ('.py','.json'))
res=dict(scene='TWO_ROOM_01',description='Q0 shelf-west assets and same sensor stream plus fixed y=3.6m cross-divider, 1.6m doorway x=3.0..4.6m',scope='one mechanism-oriented Q0-derived related scene; no random sample or representativeness',conditions=[dict(policy='exploration_joint_v1',mode=m,budget=96) for m in ('G','S')],sensor_config=asdict(w.config),wall_boxes=WALL_BOXES,anchor=anchor,initial_packet_sha256=p.sha256(),world_sha256=sha(D/'two_room_world.py'),runner_sha256=sha(D/'runner.py'),source_sha256={str(p.relative_to(S)):sha(p) for p in files},geometry=dict(shape=w.shape,reachable_cells=int(w.reachable.sum()),reachable_north_cells=int(w.reachable[:row,:].sum()),reachable_south_cells=int(w.reachable[row+1:,:].sum()),changed_occupancy_cells=int(np.count_nonzero(w.occupancy!=old.occupancy)),divider_row=int(row),sealing_divider_separates_start_from_north=True,free_components_after_sealing=int(n),start_observed_footprint_safe=True),wall_consistency=dict(lidar_height_ray_m=float(sensor_hits[0]),camera_height_ray_m=float(sensor_hits[1]),reference_rays_m=eval_hits.tolist(),old_world_rays_m=old_hits.tolist(),inflated_collision_at_south_wall=True),sensor=dict(depth_valid_pixels=int(np.count_nonzero(f.depth_m>0)),depth_total_pixels=int(f.depth_m.size),scan_finite=int(np.isfinite(sc.ranges_m).sum()),scan_rays=len(sc.ranges_m),initial_reachable_coverage=float(coverage(m,w.reachable)),mapper_frames=int(m.frames)),preflight_seconds=time.monotonic()-started,policy_runs=0)
assert res['sensor']['depth_valid_pixels']>0 and res['sensor']['scan_finite']>0
(D/'preflight.json').write_text(json.dumps(res,indent=2));print(json.dumps({k:v for k,v in res.items() if k!='source_sha256'},indent=2))
