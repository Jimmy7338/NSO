#!/usr/bin/env python3
"""Frozen CPU counterfactual test of repeated / novel-view TSDF observations."""
import os
os.environ.setdefault('OMP_NUM_THREADS','2')
os.environ.setdefault('OPENBLAS_NUM_THREADS','1')
import argparse
from dataclasses import replace
import json
from pathlib import Path
import resource
import sys
import time
import zipfile
import numpy as np
import open3d as o3d
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from env.virtual3d import VirtualConfig
from nso.mapping3d import SensorMapper
from utils.inspection_benchmark import InspectionWorld, TargetEvaluator, angular_proxy
from utils.cpu_protocol import file_hash, digest_json


def write_json(path, data):
    path.write_text(json.dumps(data,indent=2,ensure_ascii=False,allow_nan=False)+'\n')


def travel_distance(world, start, end):
    """Offline fixture shortest 4-neighbour distance, not a planner oracle."""
    from collections import deque
    queue = deque([(start,0)])
    visited = {start}
    while queue:
        cell, distance = queue.popleft()
        if cell == end:
            return distance*world.config.resolution_m
        for dr,dc in ((1,0),(-1,0),(0,1),(0,-1)):
            nxt = cell[0]+dr,cell[1]+dc
            if 0<=nxt[0]<world.shape[0] and 0<=nxt[1]<world.shape[1] and world.reachable[nxt] and nxt not in visited:
                visited.add(nxt)
                queue.append((nxt,distance+1))
    raise ValueError('unreachable fixture')


def run(config, output):
    output.mkdir(parents=True,exist_ok=False)
    # Include transitive local dependencies from the preceding frozen pipeline.
    dependencies = {'env/virtual3d.py','env/grid_exploration.py','nso/mapping3d.py',
        'nso/quality_coverage3d.py','nso/coverage_planner_v2.py','nso/navigable_frontier_v2.py',
        'nso/route_coverage_v2.py','nso/frontier_policy.py','nso/grid_mechanisms.py',
        'nso/grid_topology.py','utils/grid_geometry.py','utils/rgbd_contract.py',
        'utils/reconstruction_metrics.py','utils/paper_eval.py','utils/cpu_protocol.py',
        'requirements-3d.lock.txt'}
    names = sorted(dependencies | {
        'utils/inspection_benchmark.py','scripts/eval_inspection_mechanism.py',
        'scripts/analyze_inspection_mechanism.py','tests/virtual3d/test_inspection.py',
        'configs/virtual3d/inspection_v1.json'})
    hashes = {name:file_hash(ROOT/name) for name in names}
    write_json(output/'config.json',config)
    metadata = dict(status='running',files_sha256=hashes,config_sha256=digest_json(config),
        expected_branches=len(config['seeds'])*len(config['geometries'])*len(config['noise_models'])*len(config['branches']),
        scope='controlled endpoint sensing; not navigation, SLAM or semantic efficacy validation',
        pose_source='perfect',cpu_only=True,open3d=o3d.__version__)
    with zipfile.ZipFile(output/'sources.zip','w',zipfile.ZIP_DEFLATED) as archive:
        for name in names:
            archive.write(ROOT/name,name)
    write_json(output/'run_metadata.json',metadata)
    c = VirtualConfig(**config['environment'])
    records = []
    started = time.perf_counter()
    frame_count = 0
    for seed in config['seeds']:
        for geometry in config['geometries']:
            scene_name = f'{geometry}_{seed}'
            world = InspectionWorld(c,seed,geometry)
            evaluator = TargetEvaluator(world,config['reference_samples'],config['prediction_samples'])
            scene_dir = output/scene_name
            scene_dir.mkdir()
            o3d.io.write_triangle_mesh(str(scene_dir/'truth.ply'),world.target)
            o3d.io.write_triangle_mesh(str(scene_dir/'scene.ply'),world.mesh)
            np.savez_compressed(scene_dir/'reference.npz',points=evaluator.reference,roi=evaluator.roi)
            specs = dict(no_update=(1.4,-90),duplicate_last=(1.4,-90),repeat_fresh=(1.4,-90),
                         near=(.9,-90),far=(2.2,-90),side=(1.4,-45),opposite=(1.4,90))
            poses = {name:world.pose(*specs[name]) for name in config['branches']}
            initial_pose,initial_cell = world.pose(1.4,-90)
            for noise in config['noise_models']:
                group = scene_dir/noise
                group.mkdir()
                (group/'prefix').mkdir()
                prefix = [world.capture(initial_pose,i,noise) for i in range(config['prefix_frames'])]
                baseline_mapper = SensorMapper(world.shape,c)
                for i,frame in enumerate(prefix):
                    frame.save(group/'prefix'/f'{i:03d}.npz')
                    baseline_mapper.update(frame)
                    frame_count += 1
                baseline_mesh = baseline_mapper.mesh()
                _,distances = evaluator.distances(baseline_mesh)
                support = distances<=.05
                np.savez_compressed(group/'support.npz',mask=support)
                baseline = evaluator.evaluate(baseline_mesh,support)
                write_json(group/'baseline.json',baseline)
                o3d.io.write_triangle_mesh(str(group/'baseline.ply'),baseline_mesh)
                for branch in config['branches']:
                    branch_dir = group/branch
                    branch_dir.mkdir()
                    (branch_dir/'frames').mkdir()
                    mapper = SensorMapper(world.shape,c)
                    for frame in prefix:
                        mapper.update(frame)
                    pose,cell = poses[branch]
                    proxy = angular_proxy(mapper,pose,cell)
                    cumulative = dict.fromkeys(proxy,0.)
                    delta_xy = pose[:2,3]-initial_pose[:2,3]
                    path_m = travel_distance(world,initial_cell,cell)
                    a,b = initial_pose[:2,2],pose[:2,2]
                    yaw = float(np.arccos(np.clip(a@b,-1,1)))
                    # Only a lower bound: does not charge intermediate path turns
                    # or continuous sensor acquisition while moving.
                    travel_lower_s = path_m/c.resolution_m*c.action_duration_s + yaw/(np.pi/2)*c.action_duration_s
                    capture_started = time.perf_counter()
                    for added in range(1,max(config['checkpoints'])+1):
                        index = config['prefix_frames']+added-1
                        if branch != 'no_update':
                            marginal = angular_proxy(mapper,pose,cell)
                            for key,value in marginal.items():
                                cumulative[key] += value
                            frame = replace(prefix[-1],timestamp_s=float(index)) if branch=='duplicate_last' else world.capture(pose,index,noise)
                            frame.save(branch_dir/'frames'/f'{added:03d}.npz')
                            mapper.update(frame)
                            frame_count += 1
                        if added not in config['checkpoints']:
                            continue
                        mesh = mapper.mesh()
                        mesh_path = branch_dir/f'{added:03d}.ply'
                        o3d.io.write_triangle_mesh(str(mesh_path),mesh)
                        metrics = evaluator.evaluate(mesh,support)
                        row = dict(scene=scene_name,seed=seed,geometry=geometry,noise=noise,branch=branch,
                            split='development' if seed in config['development_seeds'] else 'validation',
                            added_frames=added if branch!='no_update' else 0,checkpoint=added,
                            capture_budget_s=added*c.action_duration_s,travel_distance_m=path_m,
                            travel_time_lower_bound_s=travel_lower_s,camera_radius_m=float(np.linalg.norm(pose[:2,3]-world.center[:2])),
                            yaw_change_deg=float(np.rad2deg(yaw)),displacement_m=float(np.linalg.norm(delta_xy)),
                            initial_proxy=proxy,cumulative_proxy=cumulative.copy(),metrics=metrics,
                            delta={key:float(value-baseline[key]) for key,value in metrics.items()
                                   if value is not None and baseline[key] is not None and key not in ('predicted_roi_samples','reference_samples')},
                            mesh=str(mesh_path.relative_to(output)),elapsed_wall_s=time.perf_counter()-capture_started)
                        records.append(row)
                print(f'{scene_name} {noise}: {len(records)} checkpoints',flush=True)
    with (output/'measurements.jsonl').open('w') as stream:
        for row in records:
            stream.write(json.dumps(row,allow_nan=False)+'\n')
    assert all(file_hash(ROOT/name)==sha for name,sha in hashes.items()), 'source changed during experiment'
    metadata.update(status='complete',branches=len(records)//len(config['checkpoints']),checkpoints=len(records),
        stored_frames=frame_count,wall_time_s=time.perf_counter()-started,
        peak_rss_mib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024)
    write_json(output/'run_metadata.json',metadata)
    # Content hashes make replay inputs and metric/mesh evidence tamper-evident.
    artifacts = {str(p.relative_to(output)):file_hash(p) for p in sorted(output.rglob('*')) if p.is_file()}
    write_json(output/'artifacts_sha256.json',artifacts)
    print(json.dumps(metadata,ensure_ascii=False),flush=True)


if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config',type=Path,default=ROOT/'configs/virtual3d/inspection_v1.json')
    parser.add_argument('--output',type=Path,required=True)
    args = parser.parse_args()
    run(json.loads(args.config.read_text()),args.output)
