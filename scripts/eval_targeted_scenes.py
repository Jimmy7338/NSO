#!/usr/bin/env python3
"""Prospective scene-adaptation experiments, unchanged existing CPU policies."""
import csv
from datetime import datetime,timezone
import hashlib
import json
from pathlib import Path
import resource
import sys
import time
import zipfile
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from env.targeted_scenes import generate_scene,FAMILIES,CONDITIONS
from env.grid_exploration import GridConfig,GridExplorationEnv
from nso.grid_mechanisms import MechanismPolicy
from utils.cpu_protocol import geometry_hash,file_hash,digest_json,runtime_versions

CASES={'geometry':('gain_control','aligned'),
       'semantic_aligned':('gain_semantic','aligned'),
       'semantic_shuffled':('gain_semantic','shuffled'),
       'semantic_constant':('gain_semantic','constant'),
       'semantic_noisy':('gain_semantic','noisy'),
       'combined_aligned':('gain_semantic_structure','aligned'),
       'combined_shuffled':('gain_semantic_structure','shuffled')}


def write_json(path,value):path.write_text(json.dumps(value,indent=2,allow_nan=False,ensure_ascii=False)+'\n')


def run_episode(scene,config,case,entrance,output):
    output.mkdir()
    method,condition=CASES[case]
    ec=GridConfig(**config['environment'])
    env=GridExplorationEnv(scene.occupancy,ec,scene.semantic_fields[condition])
    start=scene.entrances[entrance]
    obs=env.reset(start=start,heading=1 if entrance==0 else 3)
    policy=MechanismPolicy(ec,method,config['max_candidates'],config['mechanisms'])
    assets=env.evaluation_assets();reference=assets['reachable']
    room_ids=[r['room_id'] for r in scene.rooms if r['productive']]
    productive=np.isin(scene.room_labels,room_ids)&reference
    rows=[];masks=[];poses=[];durations=[]
    def record(observation,action,decision_ms=0):
        metrics=env.evaluation_metrics();known=observation.belief!=-1
        metrics.update(action=action,row=observation.position[0],col=observation.position[1],heading=observation.heading,
            decision_ms=decision_ms,
            productive_area_coverage=float(np.count_nonzero(known&productive)/productive.sum()),
            productive_landmarks_seen=sum(known[tuple(p['position'])] for p in scene.landmarks if p['productive']),
            all_landmarks_seen=sum(known[tuple(p['position'])] for p in scene.landmarks))
        # Convert NumPy integer counts before strict JSON serialization.
        metrics['productive_landmarks_seen']=int(metrics['productive_landmarks_seen'])
        metrics['all_landmarks_seen']=int(metrics['all_landmarks_seen'])
        rows.append(metrics);masks.append(np.packbits(observation.visible.ravel()));poses.append((*observation.position,observation.heading))
    record(obs,'reset');started=time.perf_counter()
    while True:
        t=time.perf_counter();decision=policy.act(obs);duration=(time.perf_counter()-t)*1000;durations.append(duration)
        obs,done=env.step(decision.action);policy.observe_outcome(obs,done)
        record(obs,decision.action,duration)
        if done:break
    elapsed=time.perf_counter()-started
    with (output/'steps.csv').open('w') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    for name,records in [('candidates.jsonl',policy.selection_audit),('goal_attempts.jsonl',policy.attempts)]:
        with (output/name).open('w') as f:
            for r in records:f.write(json.dumps(r,allow_nan=False)+'\n')
    np.savez_compressed(output/'observations.npz',**assets,visible_packed=np.stack(masks),poses=np.asarray(poses),
         final_belief=obs.belief,final_semantic_belief=policy.semantic,
         semantic_field=scene.semantic_fields[condition],room_labels=scene.room_labels,productive_mask=productive)
    curve=np.interp(np.arange(ec.max_steps+1),np.arange(len(rows)),[r['coverage_ratio'] for r in rows])
    result=dict(rows[-1],**policy.diagnostics(),case=case,method=method,semantic_condition=condition,
        schema='targeted_scenes_v1',status='complete',coverage_auc=float(np.trapz(curve)/ec.max_steps),
        episode_wall_time_s=elapsed,decision_ms_median=float(np.median(durations)),decision_ms_p95=float(np.percentile(durations,95)),
        initial_pose=list(poses[0]),artifact_dir=output.name,ground_truth_sha256=file_digest(scene.occupancy),
        canonical_geometry_sha256=geometry_hash(scene.occupancy),semantic_sha256=file_digest(scene.semantic_fields[condition]),
        productive_landmark_count=len(room_ids))
    write_json(output/'episode.json',result)
    return result


def file_digest(array):return hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest()


def run(config,output):
    if config.get('schema')!='targeted_scenes_v1':raise ValueError('wrong scenario config schema')
    GridConfig(**config['environment'])
    if not config['cases'] or len(set(config['cases']))!=len(config['cases']) or set(config['cases'])-CASES.keys():raise ValueError('invalid cases')
    if len(set((m['family'],m['size'],m['seed']) for m in config['maps']))!=len(config['maps']):raise ValueError('duplicate scene specification')
    scenes=[];seen=set()
    for entry in config['maps']:
        scene=generate_scene(entry['family'],entry['size'],entry['seed'],config['environment']['resolution_m'])
        value=geometry_hash(scene.occupancy)
        if value in seen:raise ValueError('duplicate geometry modulo rotation/reflection')
        seen.add(value);scenes.append((entry,scene))
    output=Path(output);output.mkdir(parents=True,exist_ok=False)
    names=['env/targeted_scenes.py','env/grid_exploration.py','env/grid_semantics.py','nso/grid_mechanisms.py',
           'nso/grid_topology.py','nso/frontier_policy.py','utils/grid_geometry.py','utils/paper_eval.py',
           'utils/cpu_protocol.py','scripts/eval_targeted_scenes.py','scripts/analyze_targeted_scenes.py']
    signatures={n:file_hash(ROOT/n) for n in names}
    metadata=dict(schema='targeted_scenes_v1',status='running',created_at=datetime.now(timezone.utc).isoformat(),
                  expected_episodes=len(scenes)*len(config['cases']),config_sha256=digest_json(config),
                  files_sha256=signatures,runtime=runtime_versions(),neural_training=False,
                  policy_changes=False,primary_metric='coverage_auc',
                  primary_contrasts=['semantic_aligned minus geometry','combined_aligned minus geometry',
                                     'semantic_aligned minus semantic_shuffled','combined_aligned minus combined_shuffled'],
                  scope='prospectively specified synthetic scenario-adaptation study, after prior negative suite',
                  limitations=['category/room-size correlation is deliberately supplied by generator',
                               'one entrance per scene, no claim of real visual recognition or original full neural NSO',
                               'same-family size/seed variants are correlated; uncertainty summaries exploratory'])
    write_json(output/'config.json',config);write_json(output/'run_metadata.json',metadata)
    with zipfile.ZipFile(output/'sources.zip','w',zipfile.ZIP_DEFLATED) as z:
        for name in names:z.write(ROOT/name,name)
    for entry,scene in scenes:
        map_id=f"{entry['family']}_{entry['size']}_seed{entry['seed']}"
        write_json(output/f'{map_id}.scene.json',dict(scene.descriptors,rooms=scene.rooms,landmarks=scene.landmarks,
                   entrances=scene.entrances,canonical_geometry_sha256=geometry_hash(scene.occupancy)))
        np.savez_compressed(output/f'{map_id}.scene.npz',occupancy=scene.occupancy,room_labels=scene.room_labels,**scene.semantic_fields)
        entrance=entry['entrance']
        for case in config['cases']:
            name=f'{map_id}_{case}'
            try:
                result=run_episode(scene,config,case,entrance,output/name)
                result.update(map_id=map_id,family=entry['family'],size=entry['size'],map_seed=entry['seed'],episode_key=name)
                write_json(output/name/'episode.json',result)
            except Exception as exc:
                import traceback
                (output/f'{name}.error.txt').write_text(traceback.format_exc())
                result=dict(status='failed',episode_key=name,map_id=map_id,case=case,error=str(exc))
            with (output/'episodes.jsonl').open('a') as f:f.write(json.dumps(result,allow_nan=False)+'\n')
            print(f"{name}: {result['status']}"+(f" coverage={result['coverage_ratio']:.3f}, time={result['episode_wall_time_s']:.2f}s" if result['status']=='complete' else ''),flush=True)
    if signatures!={n:file_hash(ROOT/n) for n in names}:raise RuntimeError('scenario experiment source changed during run')
    records=[json.loads(s) for s in (output/'episodes.jsonl').read_text().splitlines()]
    metadata.update(status='complete' if all(r['status']=='complete' for r in records) else 'completed_with_failures',
                    failed_episodes=sum(r['status']!='complete' for r in records),frozen_sources_verified=True,
                    peak_process_rss_mib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024)
    write_json(output/'run_metadata.json',metadata)
    if metadata['failed_episodes']:return 1
    from scripts.analyze_targeted_scenes import analyze
    analyze(output)
    return 0

if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--config',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();sys.exit(run(json.loads(a.config.read_text()),a.output))
