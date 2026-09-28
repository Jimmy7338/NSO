#!/usr/bin/env python3
"""Secondary local diagnostic using already captured counterfactual frames.

Common ROI is the union of GEOMETRY-only estimated object extents. Reference
sampling remains uniform area; no semantic importance weight is introduced.
This post-hoc diagnostic cannot replace the frozen whole-scene primary metric.
"""
import os
os.environ.setdefault('OMP_NUM_THREADS','1');os.environ.setdefault('OPENBLAS_NUM_THREADS','1')
import argparse,json,sys
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor,as_completed
import numpy as np
import open3d as o3d
from scipy.stats import spearmanr
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from env.virtual3d_v2 import VirtualConfigV2
from nso.semantic_completion_v3 import SemanticHistoryMapperV3,ObjectCompletionModel
from utils.rgbd_contract import RGBDFrame,PlanarScan
from utils.reconstruction_metrics import ray_scene


def task(directory,row):
    source=Path(row['source']);config=json.loads((source/'config.json').read_text())
    name=row['scene'];layout,seed,step=name.split('_');seed=int(seed);step=int(step[4:])
    results=[json.loads(s) for s in (source/'episodes.jsonl').read_text().splitlines()]
    result=next(r for r in results if r['method']=='full_v2' and r['seed']==seed and r['layout']==layout)
    entry=next(e for e in config['scenes'] if e['seed']==seed and e['layout']==layout)
    c=VirtualConfigV2(**(config['environment']|entry.get('environment',{})))
    folder=source/result['artifact_dir'];shape=np.load(folder/'maps.npz')['final_belief'].shape
    frames=[RGBDFrame.load(folder/'frames'/f'{i:04d}.npz') for i in range(step+1)]
    scans=[PlanarScan.load(folder/'scans'/f'{i:04d}.npz') for i in range(step+1)]
    def mapped():
        m=SemanticHistoryMapperV3(shape,c,c.truncation_m)
        for f,s in zip(frames,scans):m.update(f,s)
        return m
    mapper=mapped();model=ObjectCompletionModel(mapper,False)
    ref=np.load(source/(result['scene']+'_reference.npz'))['points'];roi=np.zeros(len(ref),bool)
    for obj in model.objects:
        roi|=(np.abs(ref-obj['center'])<=obj['dims']/2+.15).all(axis=1)
    points=o3d.core.Tensor(ref[roi].astype(np.float32))
    before=ray_scene(mapper.mesh()).compute_distance(points,nthreads=1).numpy()<=.05
    outcomes=json.loads((directory/name/'outcomes.json').read_text())
    for outcome in outcomes:
        m=mapped()
        for i in range(1,5):m.update(RGBDFrame.load(directory/name/f"candidate{outcome['candidate']:03d}_frame{i}.npz"))
        after=ray_scene(m.mesh()).compute_distance(points,nthreads=1).numpy()<=.05
        outcome['common_object_recall_gain']=float(after.mean()-before.mean())
    result=dict(scene=name,common_object_reference_points=int(roi.sum()),scorers={})
    for scorer in row['scorers']:
        best=max(outcomes,key=lambda x:(x[scorer],-x['candidate']))
        corr=spearmanr([x[scorer] for x in outcomes],[x['common_object_recall_gain'] for x in outcomes]).statistic
        result['scorers'][scorer]=dict(recall_gain=best['common_object_recall_gain'],
            ranking=float(corr) if np.isfinite(corr) else None)
    (directory/name/'local_outcomes.json').write_text(json.dumps(outcomes,indent=2)+'\n')
    return result

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('directory',type=Path);a=p.parse_args()
    rows=json.loads((a.directory/'summary.json').read_text());results=[]
    with ProcessPoolExecutor(max_workers=2) as pool:
        futures=[pool.submit(task,a.directory,row) for row in rows]
        for f in as_completed(futures):
            result=f.result();results.append(result);print(json.dumps(result),flush=True)
    (a.directory/'local_summary.json').write_text(json.dumps(results,indent=2)+'\n')
