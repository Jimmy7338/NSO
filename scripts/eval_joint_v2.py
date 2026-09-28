#!/usr/bin/env python3
"""Compact, replayable full closed-loop evaluation on larger CPU worlds."""
import os
os.environ.setdefault('OMP_NUM_THREADS','2');os.environ.setdefault('OPENBLAS_NUM_THREADS','1')
import argparse
import hashlib
import json
from pathlib import Path
import resource
import sys
import time
import zipfile
from concurrent.futures import ProcessPoolExecutor,as_completed
import numpy as np
import open3d as o3d
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from utils.virtual_scene_factory import create_scene,scene_configuration,scene_key
from env.grid_exploration import GridConfig
from nso.mapping3d_v2 import QualityMapperV2
from nso.quality_coverage3d import QualityCoveragePolicy
from nso.joint_planner_v2 import JointPlannerV2
from nso.camera_mapping_v2 import CameraQualityMapperV2
from nso.camera_joint_planner_v2 import CameraJointPlannerV2
from nso.semantic_completion_v3 import SemanticHistoryMapperV3,ObjectJointPlannerV3
from nso.feedback_joint_planner_v4 import FeedbackJointPlannerV4
from utils.reconstruction_metrics import ReconstructionEvaluator,ray_scene
from utils.cpu_protocol import file_hash,digest_json
from utils.grid_geometry import DIRECTIONS

METHODS=('coverage','legacy_geometry','legacy_semantic','geometry_v2','full_v2','no_route_v2','no_camera_v2','no_hypotheses_v3',
         'full_v4','geometry_v4','no_feedback_v4','no_guard_v4','no_route_v4','objectness_v4','no_hypotheses_v4')


def write_json(path,value):path.write_text(json.dumps(value,indent=2,ensure_ascii=False,allow_nan=False)+'\n')


def create_policy(method,c,parameters,camera_coverage=False,object_hypotheses=False):
    grid=GridConfig(resolution_m=c.resolution_m,robot_radius_m=c.robot_radius_m,
        sensor_range_m=c.max_depth_m,sensor_fov_deg=360,max_steps=c.max_steps)
    if method in ('coverage','legacy_geometry','legacy_semantic'):
        mode={'coverage':'coverage','legacy_geometry':'geometric_quality','legacy_semantic':'semantic_quality'}[method]
        return QualityCoveragePolicy(grid,c,mode)
    cls=CameraJointPlannerV2 if camera_coverage and method!='no_camera_v2' else JointPlannerV2
    parameters=parameters.copy()
    if object_hypotheses and method not in ('no_hypotheses_v3','no_camera_v2'):cls=ObjectJointPlannerV3
    else:
        parameters.pop('hypothesis_weight',None);parameters.pop('fine_categories',None)
    if method.endswith('_v4'):
        if not camera_coverage or not object_hypotheses:
            raise ValueError('V4 development guard requires camera coverage and object hypotheses')
        cls=FeedbackJointPlannerV4
        parameters['feedback_guard']=method!='no_feedback_v4'
        parameters['coverage_guard']=method!='no_guard_v4'
        parameters['use_object_hypotheses']=method!='no_hypotheses_v4'
        if method=='objectness_v4':
            parameters['fine_categories']=False;parameters['semantic_weight']=0.
    else:
        for key in ('coverage_retention','inspection_fraction','reserve_fraction','minimum_evidence_rate'):
            parameters.pop(key,None)
    return cls(grid,c,semantic=method not in ('geometry_v2','geometry_v4'),route=method not in ('no_route_v2','no_route_v4'),**parameters)


def episode(world,evaluator,method,config,output):
    output.mkdir();(output/'frames').mkdir();(output/'scans').mkdir()
    c=world.config
    cls=CameraQualityMapperV2 if config.get('camera_coverage',False) else QualityMapperV2
    if config.get('object_hypotheses',False):cls=SemanticHistoryMapperV3
    mapper=cls(world.shape,c,c.truncation_m)
    policy=create_policy(method,c,config['planner'],config.get('camera_coverage',False),config.get('object_hypotheses',False))
    metrics=[];steps=[];known=[];poses=[];decisions=[];collision=False;done=False;action='reset'
    frame=world.sense();started=time.perf_counter();planning=[];mapping=[]
    while True:
        index=world.step_count;scan=world.scan()
        if config.get('save_frames',True):
            frame.save(output/'frames'/f'{index:04d}.npz');scan.save(output/'scans'/f'{index:04d}.npz')
        timer=time.perf_counter();mapper.update(frame,scan);mapping.append(time.perf_counter()-timer)
        if collision:
            dr,dc=DIRECTIONS[world.heading];r,col=world.position[0]+dr,world.position[1]+dc
            if 0<=r<world.shape[0] and 0<=col<world.shape[1]:mapper.belief[r,col]=1
        obs=mapper.observation(world.position,world.heading,index,collision)
        if index:policy.observe_outcome(obs,done)
        coverage=float(np.count_nonzero((mapper.belief!=-1)&world.reachable)/world.reachable.sum())
        steps.append(dict(step=index,action=action,coverage=coverage,collisions=world.collisions))
        known.append(np.packbits((mapper.belief!=-1).ravel()));poses.append([*world.position,world.heading])
        if index%config['evaluation_interval']==0 or done:
            mesh=mapper.mesh()
            row=dict(step=index,coverage_2d=coverage,**evaluator.evaluate(mesh,coverage,config['thresholds_m']))
            metrics.append(row)
            if config.get('save_checkpoint_meshes',False):
                np.savez_compressed(output/f'mesh_{index:04d}.npz',vertices=np.asarray(mesh.vertices),triangles=np.asarray(mesh.triangles))
        if done:break
        if isinstance(policy,JointPlannerV2):policy.set_mapping(mapper)
        else:policy.set_surface_evidence(mapper.evidence())
        timer=time.perf_counter();decision=policy.act(obs);planning.append(time.perf_counter()-timer)
        decisions.append(dict(step=index,action=decision.action,goal=decision.goal,heading=decision.goal_heading))
        action=decision.action;frame,collision,done=world.step(action)
    mesh=mapper.mesh()
    np.savez_compressed(output/'final_mesh.npz',vertices=np.asarray(mesh.vertices),triangles=np.asarray(mesh.triangles))
    np.savez_compressed(output/'maps.npz',known_packed=np.stack(known),poses=poses,final_belief=mapper.belief)
    for name,rows in [('metrics',metrics),('steps',steps),('decisions',decisions),('candidates',policy.selection_audit),('attempts',policy.attempts)]:
        with (output/(name+'.jsonl')).open('w') as stream:
            for row in rows:stream.write(json.dumps(row,allow_nan=False)+'\n')
    result=dict(method=method,**metrics[-1],collisions=world.collisions,frames=mapper.frames,
        path_length_m=world.moves*c.resolution_m,wall_time_s=time.perf_counter()-started,
        planning_ms_p95=float(np.percentile(planning,95))*1000,mapping_ms_mean=float(np.mean(mapping))*1000,
        artifact_dir=output.name,worker_peak_rss_mib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024)
    if isinstance(policy,FeedbackJointPlannerV4):result['guard_diagnostics']=policy.diagnostics()
    result['coverage_auc']=float(np.trapz(np.interp(np.arange(c.max_steps+1),[r['step'] for r in steps],[r['coverage'] for r in steps]))/c.max_steps)
    for threshold in config['thresholds_m']:
        tag=f'{round(threshold*100):02d}cm'
        result[f'joint_auc_{tag}']=float(np.trapz(np.interp(np.arange(c.max_steps+1),[r['step'] for r in metrics],[r[f'joint_{tag}'] for r in metrics]))/c.max_steps)
    return result


def episode_task(config,entry,scene,method,output):
    world=create_scene(config,entry);c=world.config
    reference=np.load(output/(scene+'_reference.npz'),allow_pickle=False)
    evaluator=ReconstructionEvaluator.__new__(ReconstructionEvaluator)
    evaluator.reference=reference['points'];evaluator.classes=reference['classes'];evaluator.truth=ray_scene(world.mesh)
    result=episode(world,evaluator,method,config,output/(scene+'_'+method))
    truth_hash=hashlib.sha256(np.asarray(world.mesh.vertices).tobytes()+np.asarray(world.mesh.triangles).tobytes()).hexdigest()
    result.update(scene=scene,seed=entry['seed'],layout=entry.get('layout','inspection'),semantic_condition=entry.get('semantic_condition','aligned'),
        depth_sigma_m=c.depth_sigma_m,pose_noise_m=c.pose_noise_m,occluded_objects=c.occluded_objects,truth_sha256=truth_hash)
    result['world_family']=scene_configuration(config,entry)[0]
    write_json(output/result['artifact_dir']/'episode.json',result)
    return result


def run(config,output):
    if set(config['methods'])-set(METHODS):raise ValueError('unknown method')
    if any(method.endswith('_v4') for method in config['methods']):
        if config.get('full_method') not in config['methods']:
            raise ValueError('V4 matrix requires an explicit full_method present in methods')
    output.mkdir(parents=True,exist_ok=False)
    # Freeze all top-level project modules needed by this CPU path, including
    # package initializers; optional GPU subpackages are not executed.
    names=sorted({str(p.relative_to(ROOT)) for folder in ('env','nso','utils') for p in (ROOT/folder).glob('*.py')} |
        {'scripts/eval_joint_v2.py','scripts/analyze_joint_v2.py','scripts/confirm_joint_v2.py','requirements-3d.lock.txt','docs/JOINT_V2_CONFIRMATION_PROTOCOL.md'})
    signatures={name:file_hash(ROOT/name) for name in names}
    metadata=dict(status='running',scene_key_version=3,files_sha256=signatures,config_sha256=digest_json(config),
        scope=config['scope'],expected_episodes=len(config['scenes'])*len(config['methods']),
        cpu_only=True,primary_metric='joint_auc_05cm',joint_auc='checkpoint trapezoid, held after early stop')
    write_json(output/'config.json',config);write_json(output/'run_metadata.json',metadata)
    with zipfile.ZipFile(output/'sources.zip','w',zipfile.ZIP_DEFLATED) as archive:
        for name in names:archive.write(ROOT/name,name)
    records=[];jobs=[];scene_keys=set()
    for entry in config['scenes']:
        scene=scene_key(config,entry)
        if scene in scene_keys:raise ValueError('duplicate scene configuration')
        scene_keys.add(scene)
        world=create_scene(config,entry);evaluator=ReconstructionEvaluator(world,config['reference_samples'])
        np.savez_compressed(output/(scene+'_reference.npz'),points=evaluator.reference,classes=evaluator.classes,
            reachable=world.reachable,occupancy=world.occupancy,vertices=np.asarray(world.mesh.vertices),triangles=np.asarray(world.mesh.triangles))
        for method in config['methods']:
            jobs.append((config,entry,scene,method,output))
    # Independent CPU experiments may run concurrently; scheduling does not
    # alter simulated action budgets. Wall latencies are under concurrent load.
    np.random.default_rng(331).shuffle(jobs)
    with ProcessPoolExecutor(max_workers=config.get('workers',1)) as pool:
        futures=[pool.submit(episode_task,*job) for job in jobs]
        for future in as_completed(futures):
            result=future.result();records.append(result)
            with (output/'episodes.jsonl').open('a') as stream:stream.write(json.dumps(result,allow_nan=False)+'\n')
            print(result['scene'],result['method'],'C',round(result['coverage_2d'],3),'F1',round(result['f1_05cm'],3),'J-AUC',round(result['joint_auc_05cm'],4),
                'collision',result['collisions'],'sec',round(result['wall_time_s'],1),flush=True)
    assert all(file_hash(ROOT/name)==sha for name,sha in signatures.items()),'sources changed during run'
    metadata.update(status='complete',episodes=len(records),peak_worker_rss_mib=max(r['worker_peak_rss_mib'] for r in records),workers=config.get('workers',1))
    write_json(output/'run_metadata.json',metadata)
    write_json(output/'summary.json',{method:{key:float(np.mean([r[key] for r in records if r['method']==method]))
        for key in ('coverage_auc','coverage_2d','joint_auc_05cm','f1_05cm','surface_error_mean_m','collisions','wall_time_s')} for method in config['methods']})


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--config',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    args=p.parse_args();run(json.loads(args.config.read_text()),args.output)
