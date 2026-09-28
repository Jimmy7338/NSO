#!/usr/bin/env python3
"""Measure production v2 signals against counterfactual mesh gains.

Frozen inspection data is already viewed development data. Camera-pose injection
only adapts its continuous endpoint poses to the production scoring function;
neither scorer receives the evaluator / truth. Shared sensor object masks match
the local evaluation scope, as in inspection_v1.
"""
import os
os.environ.setdefault('OMP_NUM_THREADS','1');os.environ.setdefault('OPENBLAS_NUM_THREADS','1')
import json
from pathlib import Path
import sys
from unittest.mock import patch
import numpy as np
import open3d as o3d
from scipy.stats import spearmanr
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from env.virtual3d import VirtualConfig
from env.grid_exploration import GridConfig
from nso.mapping3d_v2 import QualityMapperV2
from nso.joint_planner_v2 import JointPlannerV2
from utils.inspection_benchmark import TargetEvaluator,angular_proxy
from utils.reconstruction_metrics import ray_scene
from utils.rgbd_contract import RGBDFrame
from utils.cpu_protocol import file_hash


def run(source,output):
    output.mkdir(parents=True,exist_ok=False);records=[]
    config=json.loads((source/'config.json').read_text());c=VirtualConfig(voxel_m=.03)
    for seed in config['seeds']:
        for geometry in config['geometries']:
            scene=f'{geometry}_{seed}';ref=np.load(source/scene/'reference.npz')
            evaluator=TargetEvaluator.__new__(TargetEvaluator);evaluator.reference=ref['points'];evaluator.roi=ref['roi']
            evaluator.truth=ray_scene(o3d.io.read_triangle_mesh(str(source/scene/'truth.ply')));evaluator.prediction_samples=60000
            for noise in ('iid_01','iid_03','fixed_01'):
                group=source/scene/noise
                prefix=[RGBDFrame.load(p) for p in sorted((group/'prefix').glob('*.npz'))]
                baseline_mapper=QualityMapperV2((30,40),c,.12)
                for f in prefix:baseline_mapper.update(f)
                baseline_mesh=baseline_mapper.mesh();support=evaluator.distances(baseline_mesh)[1]<=.05
                baseline=evaluator.evaluate(baseline_mesh,support)
                q=baseline_mapper.quality_evidence();mask=q['label']>1;q={k:v[mask] for k,v in q.items()}
                policy=JointPlannerV2(GridConfig(),c,semantic=False);policy.mapper=baseline_mapper;policy.quality=q;policy.ray=ray_scene(baseline_mesh)
                for branch in ('repeat_fresh','near','far','side','opposite'):
                    mapper=QualityMapperV2((30,40),c,.12)
                    for f in prefix:mapper.update(f)
                    future=[RGBDFrame.load(p) for p in sorted((group/branch/'frames').glob('*.npz'))]
                    pose=future[0].world_from_camera;cell=mapper.grid_cell(pose[:3,3])
                    with patch('nso.joint_planner_v2.camera_pose',return_value=pose):
                        gain,completion,precision=policy._quality_gain(cell,0)
                    legacy=angular_proxy(baseline_mapper,pose,cell)['angular']
                    for index,frame in enumerate(future,1):
                        mapper.update(frame)
                        if index not in (1,12):continue
                        metrics=evaluator.evaluate(mapper.mesh(),support)
                        records.append(dict(seed=seed,geometry=geometry,noise=noise,branch=branch,frames=index,
                            completion=completion,precision=precision,combined=float(gain.sum()),legacy_angular=legacy,
                            f1_gain=metrics['f1_05cm']-baseline['f1_05cm'],
                            fixed_support_improvement=baseline['fixed_support_distance_m']-metrics['fixed_support_distance_m']))
            print(scene,'done',flush=True)
    rankings=[]
    for frames in (1,12):
        for objective in ('f1_gain','fixed_support_improvement'):
            for signal in ('legacy_angular','completion','precision','combined'):
                correlations=[]
                for seed in config['seeds']:
                    for geometry in config['geometries']:
                        for noise in ('iid_01','iid_03','fixed_01'):
                            rows=[r for r in records if (r['seed'],r['geometry'],r['noise'],r['frames'])==(seed,geometry,noise,frames)]
                            x=[r[signal] for r in rows];y=[r[objective] for r in rows]
                            if np.ptp(x)>0 and np.ptp(y)>0:correlations.append(float(spearmanr(x,y).statistic))
                rankings.append(dict(frames=frames,objective=objective,signal=signal,mean_spearman=float(np.mean(correlations)),groups=len(correlations)))
    for name,data in [('measurements.json',records),('ranking.json',rankings)]:
        (output/name).write_text(json.dumps(data,indent=2)+'\n')
    (output/'metadata.json').write_text(json.dumps(dict(scope='development diagnostic, no held-out efficacy claim',
        mapper_sha256=file_hash(ROOT/'nso/mapping3d_v2.py'),policy_sha256=file_hash(ROOT/'nso/joint_planner_v2.py'),
        script_sha256=file_hash(Path(__file__)),source=str(source.resolve()),observed_object_mask_shared=True),indent=2)+'\n')
    print(json.dumps(rankings))


if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('--source',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();run(a.source,a.output)
