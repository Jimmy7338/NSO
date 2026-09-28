#!/usr/bin/env python3
"""Exploratory resolution sensitivity using frozen, identical RGB-D inputs.

This follow-up was chosen AFTER reading inspection_v1 results; it is not part
of that preregistered matrix. It does not tune or replace the default backend.
"""
import os
os.environ.setdefault('OMP_NUM_THREADS','2')
os.environ.setdefault('OPENBLAS_NUM_THREADS','1')
import argparse
import json
from pathlib import Path
import sys
import time
import numpy as np
import open3d as o3d
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from env.virtual3d import VirtualConfig
from nso.mapping3d import SensorMapper
from utils.inspection_benchmark import TargetEvaluator
from utils.reconstruction_metrics import ray_scene
from utils.rgbd_contract import RGBDFrame
from utils.cpu_protocol import file_hash


def run(source,output):
    output.mkdir(parents=True,exist_ok=False)
    metadata=json.loads((source/'run_metadata.json').read_text())
    config=json.loads((source/'config.json').read_text())
    for name in ('nso/mapping3d.py','utils/inspection_benchmark.py','utils/rgbd_contract.py'):
        assert file_hash(ROOT/name)==metadata['files_sha256'][name]
    rows=[]
    started=time.perf_counter()
    for seed in config['validation_seeds']:
        for geometry in config['geometries']:
            scene=f'{geometry}_{seed}'
            group=source/scene/'iid_01'
            target=TargetEvaluator.__new__(TargetEvaluator)
            ref=np.load(source/scene/'reference.npz')
            target.reference,target.roi=ref['points'],ref['roi']
            target.truth=ray_scene(o3d.io.read_triangle_mesh(str(source/scene/'truth.ply')))
            # 10x prediction samples to check small P/mean-error estimates.
            target.prediction_samples=240000
            prefix=[RGBDFrame.load(p) for p in sorted((group/'prefix').glob('*.npz'))]
            for voxel in (.06,.03):
                c=VirtualConfig(voxel_m=voxel,depth_sigma_m=0,dropout=0)
                baseline_mapper=SensorMapper((30,40),c)
                for frame in prefix:baseline_mapper.update(frame)
                base_mesh=baseline_mapper.mesh()
                support=target.distances(base_mesh)[1]<=.05
                baseline=target.evaluate(base_mesh,support)
                for branch in ('repeat_fresh','near','side','opposite'):
                    mapper=SensorMapper((30,40),c)
                    for frame in prefix:mapper.update(frame)
                    for path in sorted((group/branch/'frames').glob('*.npz')):
                        mapper.update(RGBDFrame.load(path))
                    mesh=mapper.mesh()
                    metrics=target.evaluate(mesh,support)
                    artifact=f'{scene}_{round(voxel*100):02d}cm_{branch}.ply'
                    o3d.io.write_triangle_mesh(str(output/artifact),mesh)
                    rows.append(dict(scene=scene,seed=seed,geometry=geometry,voxel_m=voxel,
                        truncation_m=4*voxel,branch=branch,baseline=baseline,metrics=metrics,
                        delta={k:float(v-baseline[k]) for k,v in metrics.items() if v is not None and baseline[k] is not None},mesh=artifact))
    (output/'measurements.json').write_text(json.dumps(rows,indent=2)+'\n')
    result=dict(status='complete',scope='post-hoc engineering sensitivity; voxel and truncation jointly changed',
                source=str(source.resolve()),source_manifest_sha256=file_hash(source/'artifacts_sha256.json'),
                script_sha256=file_hash(Path(__file__)),comparisons=len(rows),wall_time_s=time.perf_counter()-started,
                prediction_samples=240000,default_backend_changed=False)
    (output/'metadata.json').write_text(json.dumps(result,indent=2)+'\n')
    # Save the exact script alongside dependent source hashes retained upstream.
    (output/'diagnostic_source.py').write_bytes(Path(__file__).read_bytes())
    lines=['# 三维后端分辨率诊断（事后探索）','',
        '输入为主实验验证种子 103/104 的相同 1 cm 独立噪声深度帧，4 帧前缀＋12 帧追加。比较 6 cm 和 3 cm 体素；截断距离随之从 24 cm 变为 12 cm，因此这是后端尺度的联合敏感性，不是单独体素的因果消融。',
        '', '每个网格按面积采样 240000 点，是主实验的十倍；目标区域筛选和固定真值保持相同。支持面在各自后端前缀内固定，跨后端应优先比较绝对完整率／误差，不能把不同支持面的改善直接当作相同点集对照。',
        '', '|物体|体素/cm|动作|最终 F1@5cm|最终 P@5cm|最终表面误差/mm|ΔF1|', '|---|---:|---|---:|---:|---:|---:|']
    for geometry in config['geometries']:
        for voxel in (.06,.03):
            for branch in ('repeat_fresh','near','side','opposite'):
                subset=[r for r in rows if r['geometry']==geometry and r['voxel_m']==voxel and r['branch']==branch]
                avg=lambda key:np.mean([r['metrics'][key] for r in subset])
                delta=np.mean([r['delta']['f1_05cm'] for r in subset])
                lines.append(f'|{geometry}|{voxel*100:.0f}|{branch}|{avg("f1_05cm"):.4f}|{avg("p_05cm"):.4f}|{avg("surface_error_m")*1000:.3f}|{delta:+.4f}|')
    lines += ['', '本诊断在查看主实验结果后追加，没有独立确认集，也没有模拟位姿噪声。未替换默认后端；不把更细体素视为已验证最优参数。']
    (output/'results.md').write_text('\n'.join(lines)+'\n')
    print(json.dumps(result))


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--source',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    run(args.source,args.output)
