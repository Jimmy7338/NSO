#!/usr/bin/env python3
"""Run CPU depth+laser -> map -> planner -> motion -> TSDF -> GT evaluation."""
import os
os.environ.setdefault('OMP_NUM_THREADS','2')
os.environ.setdefault('OPENBLAS_NUM_THREADS','1')
import csv
from dataclasses import asdict
from datetime import datetime,timezone
import hashlib
import json
from pathlib import Path
import resource
import sys
import time
import zipfile
import numpy as np
import open3d as o3d
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from env.virtual3d import VirtualConfig,VirtualWorld
from env.grid_exploration import GridConfig
from nso.mapping3d import SensorMapper
from nso.quality_coverage3d import QualityCoveragePolicy,METHODS
from utils.reconstruction_metrics import ReconstructionEvaluator
from utils.cpu_protocol import file_hash,digest_json
from utils.grid_geometry import DIRECTIONS


def write_json(path,data):path.write_text(json.dumps(data,indent=2,ensure_ascii=False,allow_nan=False)+'\n')


def run_episode(world,evaluator,method,config,output):
    output.mkdir();(output/'frames').mkdir();(output/'scans').mkdir();(output/'meshes').mkdir()
    c=world.config;mapper=SensorMapper(world.shape,c)
    grid=GridConfig(resolution_m=c.resolution_m,robot_radius_m=c.robot_radius_m,
        sensor_range_m=c.max_depth_m,sensor_fov_deg=360,max_steps=c.max_steps)
    policy=QualityCoveragePolicy(grid,c,method,quality_weight=config['quality_weight'])
    rows=[];metrics=[];poses=[];known_masks=[];decisions=[];latency=[];mapping=[]
    frame=world.sense();collision=False;done=False;action='reset';started=time.perf_counter()
    while True:
        index=world.step_count;scan=world.scan()
        frame.save(output/'frames'/f'{index:05d}.npz');scan.save(output/'scans'/f'{index:05d}.npz')
        t=time.perf_counter();mapper.update(frame,scan);mapping.append(time.perf_counter()-t)
        if collision:
            dr,dc=DIRECTIONS[world.heading];r,col=world.position[0]+dr,world.position[1]+dc
            if 0<=r<world.shape[0] and 0<=col<world.shape[1]:mapper.belief[r,col]=1
        obs=mapper.observation(world.position,world.heading,index,collision)
        if index:policy.observe_outcome(obs,done)
        coverage=float(np.count_nonzero((mapper.belief!=-1)&world.reachable)/world.reachable.sum())
        rows.append(dict(step=index,time_s=frame.timestamp_s,action=action,coverage_2d=coverage,
            explored_area_m2=coverage*world.reachable.sum()*c.resolution_m**2,
            collisions=world.collisions,path_length_m=world.moves*c.resolution_m))
        poses.append((*world.position,world.heading));known_masks.append(np.packbits((mapper.belief!=-1).ravel()))
        if index%config['evaluation_interval']==0 or done:
            mesh=mapper.mesh();o3d.io.write_triangle_mesh(str(output/'meshes'/f'{index:05d}.ply'),mesh)
            metric=dict(step=index,time_s=frame.timestamp_s,coverage_2d=coverage,
                        **evaluator.evaluate(mesh,coverage,tuple(config['thresholds_m'])))
            metrics.append(metric)
        if done:break
        policy.set_surface_evidence(mapper.evidence())
        t=time.perf_counter();decision=policy.act(obs);latency.append(1000*(time.perf_counter()-t))
        decisions.append(dict(step=index,action=decision.action,goal=decision.goal,
            goal_heading=decision.goal_heading,new_goal=decision.new_goal))
        action=decision.action;frame,collision,done=world.step(action)
    mesh=mapper.mesh();o3d.io.write_triangle_mesh(str(output/'reconstruction.ply'),mesh)
    np.savez_compressed(output/'map_observations.npz',occupancy=world.occupancy,reachable=world.reachable,
                        known_packed=np.stack(known_masks),poses=poses,final_belief=mapper.belief)
    with (output/'steps.csv').open('w') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    for name,records in [('metrics',metrics),('decisions',decisions),('candidates',policy.selection_audit),('attempts',policy.attempts)]:
        (output/(name+'.jsonl')).write_text(''.join(json.dumps(r,allow_nan=False)+'\n' for r in records))
    result=dict(method=method,**metrics[-1],collisions=world.collisions,steps=world.step_count,
        frames=mapper.frames,path_length_m=world.moves*c.resolution_m,goal_count=policy.goal_count,
        wall_time_s=time.perf_counter()-started,decision_ms_p95=float(np.percentile(latency,95)),
        mapping_ms_mean=1000*float(np.mean(mapping)),artifact_dir=output.name,
        pose_source='perfect_simulated_odometry',semantic_source='visible_synthetic_class_labels',
        joint_auc_definition='trapezoidal checkpoint approximation, held constant after early stop')
    result['coverage_auc']=float(np.trapz(np.interp(np.arange(c.max_steps+1),np.arange(len(rows)),[r['coverage_2d'] for r in rows]))/c.max_steps)
    for threshold in config['thresholds_m']:
        tag=f'{round(threshold*100):02d}cm';x=[m['step'] for m in metrics];y=[m[f'joint_{tag}'] for m in metrics]
        result[f'joint_auc_{tag}']=float(np.trapz(np.interp(np.arange(c.max_steps+1),x,y))/c.max_steps)
    write_json(output/'episode.json',result)
    return result


def run(config,output):
    if set(config['methods'])-set(METHODS) or len(set(config['methods']))!=len(config['methods']):raise ValueError('invalid methods')
    c=VirtualConfig(**config['environment'])
    output=Path(output);output.mkdir(parents=True,exist_ok=False)
    names=['env/virtual3d.py','env/grid_exploration.py','nso/mapping3d.py','nso/quality_coverage3d.py',
           'nso/coverage_planner_v2.py','nso/navigable_frontier_v2.py','nso/route_coverage_v2.py',
           'nso/frontier_policy.py','nso/grid_mechanisms.py','nso/grid_topology.py',
           'utils/grid_geometry.py','utils/rgbd_contract.py','utils/reconstruction_metrics.py',
           'utils/paper_eval.py','utils/cpu_protocol.py','scripts/eval_virtual3d.py','requirements-3d.lock.txt']
    signatures={n:file_hash(ROOT/n) for n in names}
    write_json(output/'config.json',config)
    meta=dict(status='running',created_at=datetime.now(timezone.utc).isoformat(),files_sha256=signatures,
              config_sha256=digest_json(config),open3d=o3d.__version__,cpu_only=True,
              expected_episodes=len(config['scenes'])*len(config['methods']),scope='pipeline feasibility, not efficacy confirmation',
              sensors='simulated monocular depth equivalent to processed stereo depth + planar laser; no stereo matching simulation',
              robot_target='ROS1, ZED stereo, Leishen planar lidar, IMU, GPS')
    write_json(output/'run_metadata.json',meta)
    with zipfile.ZipFile(output/'sources.zip','w',zipfile.ZIP_DEFLATED) as z:
        for n in names:z.write(ROOT/n,n)
    records=[]
    for entry in config['scenes']:
        name=f"{entry['layout']}_seed{entry['seed']}_{entry.get('semantic_condition','aligned')}"
        reference_world=VirtualWorld(c,**entry)
        evaluator=ReconstructionEvaluator(reference_world,count=config['reference_samples'])
        o3d.io.write_triangle_mesh(str(output/(name+'_truth.ply')),reference_world.mesh)
        np.savez_compressed(output/(name+'_reference.npz'),points=evaluator.reference,classes=evaluator.classes,
                            occupancy=reference_world.occupancy,reachable=reference_world.reachable)
        truth_hash=hashlib.sha256(np.asarray(reference_world.mesh.vertices).tobytes()+np.asarray(reference_world.mesh.triangles).tobytes()).hexdigest()
        for method in config['methods']:
            world=VirtualWorld(c,**entry)
            result=run_episode(world,evaluator,method,config,output/(name+'_'+method))
            result.update(scene=name,truth_sha256=truth_hash)
            write_json(output/result['artifact_dir']/'episode.json',result);records.append(result)
            with (output/'episodes.jsonl').open('a') as f:f.write(json.dumps(result,allow_nan=False)+'\n')
            print(name,method,'coverage',round(result['coverage_2d'],3),'F1',round(result['f1_05cm'],3),
                  'joint',round(result['joint_05cm'],3),'collisions',result['collisions'],'seconds',round(result['wall_time_s'],1),flush=True)
    if signatures!={n:file_hash(ROOT/n) for n in names}:raise RuntimeError('source changed during run')
    meta.update(status='complete',source_verified=True,peak_rss_mib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024)
    write_json(output/'run_metadata.json',meta)
    write_json(output/'summary.json',{m:{k:float(np.mean([r[k] for r in records if r['method']==m]))
        for k in ('coverage_2d','f1_05cm','joint_05cm','joint_auc_05cm','collisions','wall_time_s')} for m in config['methods']})


if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--config',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();run(json.loads(a.config.read_text()),a.output)
