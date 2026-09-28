#!/usr/bin/env python3
"""Independently verify saved meshes/coverage and render a shareable report."""
import os
os.environ.setdefault('OMP_NUM_THREADS','2')
import csv
import hashlib
import json
from pathlib import Path
import sys
import zipfile
import numpy as np
import open3d as o3d
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from utils.reconstruction_metrics import ReconstructionEvaluator,ray_scene,surface_samples
from utils.cpu_protocol import digest_json


def analyze(source):
    source=Path(source);config=json.loads((source/'config.json').read_text())
    metadata=json.loads((source/'run_metadata.json').read_text())
    assert metadata['status']=='complete'
    assert metadata['config_sha256']==digest_json(config)
    with zipfile.ZipFile(source/'sources.zip') as archive:
        for name,value in metadata['files_sha256'].items():assert hashlib.sha256(archive.read(name)).hexdigest()==value
    records=[json.loads(s) for s in (source/'episodes.jsonl').read_text().splitlines()]
    assert len(records)==metadata['expected_episodes']
    names=[f"{s['layout']}_seed{s['seed']}_{s.get('semantic_condition','aligned')}" for s in config['scenes']]
    expected={(scene,method) for scene in names for method in config['methods']}
    assert len(expected)==len(records) and {(r['scene'],r['method']) for r in records}==expected
    mesh_checks=0;scene_curves={}
    for name in names:
        truth=o3d.io.read_triangle_mesh(str(source/(name+'_truth.ply')))
        evaluator=ReconstructionEvaluator.__new__(ReconstructionEvaluator);evaluator.truth=ray_scene(truth)
        with np.load(source/(name+'_reference.npz')) as data:
            evaluator.reference=data['points'];evaluator.classes=data['classes'];reference=data['reachable'].copy()
        pairs=[]
        for record in [r for r in records if r['scene']==name]:
            path=source/record['artifact_dir'];metrics=[json.loads(s) for s in (path/'metrics.jsonl').read_text().splitlines()]
            assert len(list((path/'frames').glob('*.npz')))==record['frames']==record['steps']+1
            assert len(list((path/'scans').glob('*.npz')))==record['frames']
            with np.load(path/'map_observations.npz') as data:
                masks=np.unpackbits(data['known_packed'],axis=1)[:,:reference.size].reshape(-1,*reference.shape).astype(bool)
                coverage=np.count_nonzero(masks&reference,axis=(1,2))/reference.sum()
                pairs.append(tuple(data['poses'][0]))
            with (path/'steps.csv').open() as f:steps=list(csv.DictReader(f))
            np.testing.assert_allclose(coverage,[float(s['coverage_2d']) for s in steps],atol=1e-12,rtol=0)
            assert abs(record['coverage_2d']-coverage[-1])<1e-12
            for metric in metrics:
                mesh=o3d.io.read_triangle_mesh(str(path/'meshes'/f"{metric['step']:05d}.ply"))
                actual=evaluator.evaluate(mesh,float(coverage[metric['step']]),tuple(config['thresholds_m']))
                for key in ('precision_05cm','recall_05cm','f1_05cm','joint_05cm','f1_10cm','joint_10cm'):
                    assert abs(actual[key]-metric[key])<1e-9,(path,key)
                mesh_checks+=1
            budget=config['environment']['max_steps']
            approximate=np.trapz(np.interp(np.arange(budget+1),[m['step'] for m in metrics],[m['joint_05cm'] for m in metrics]))/budget
            assert abs(approximate-record['joint_auc_05cm'])<1e-12
            scene_curves[(name,record['method'])]=metrics
        assert len(set(pairs))==1
    # Semantics must not change navigation/reconstruction of nonsemantic controls.
    paired_controls=0
    for name in names:
        if not name.endswith('_aligned'):continue
        shuffled=name[:-len('aligned')]+'shuffled'
        if shuffled not in names:continue
        for method in ('coverage','geometric_quality'):
            a=source/(name+'_'+method);b=source/(shuffled+'_'+method)
            with np.load(a/'map_observations.npz') as x,np.load(b/'map_observations.npz') as y:
                np.testing.assert_array_equal(x['poses'],y['poses'])
            for key in ('f1_05cm','joint_05cm','joint_auc_05cm'):
                ra=next(r for r in records if r['artifact_dir']==a.name);rb=next(r for r in records if r['artifact_dir']==b.name)
                assert ra[key]==rb[key]
            paired_controls+=1
    grouped={}
    for condition in ('aligned','shuffled'):
        selected=[r for r in records if r['scene'].endswith('_'+condition)]
        if not selected:continue
        grouped[condition]={m:{k:float(np.mean([r[k] for r in selected if r['method']==m])) for k in
            ('coverage_2d','precision_05cm','recall_05cm','f1_05cm','joint_05cm','joint_auc_05cm','surface_error_mean_m')}
            for m in config['methods']}
    report=dict(episodes=len(records),meshes_recomputed=mesh_checks,source_archive_verified=True,
        coverage_recomputed=True,joint_auc_recomputed=True,nonsemantic_control_pairs_verified=paired_controls,
        collisions=sum(r['collisions'] for r in records),frames=sum(r['frames'] for r in records),
        loop_seconds=sum(r['wall_time_s'] for r in records),peak_rss_mib=metadata['peak_rss_mib'],
        independent_geometry_count=len({r['truth_sha256'] for r in records}),
        engineering_pipeline='passed',effectiveness='exploratory; no significance or real-robot claim',means=grouped)
    (source/'verification.json').write_text(json.dumps(report,indent=2,ensure_ascii=False)+'\n')
    lines=['# CPU 三维虚拟实验与重建结果','',
        f"{len(records)} 回合，{report['independent_geometry_count']} 个独立几何实例；语义对照共享几何，不能重复计作新场景。",
        '这是流程与机制小试：真实射线深度＋平面雷达、完美模拟里程计、TSDF 重建。没有模拟 ZED 双目匹配或实车动力学。','',
        '| 标签条件 | 方法 | 二维覆盖 | 三维准确率 | 三维完整率 | F1 | 覆盖×F1 | 联合 AUC* |',
        '|---|---|---:|---:|---:|---:|---:|---:|']
    for condition,methods in grouped.items():
        for method,v in methods.items():
            lines.append('| '+condition+' | '+method+' | '+' | '.join(f'{v[k]:.4f}' for k in
                ('coverage_2d','precision_05cm','recall_05cm','f1_05cm','joint_05cm','joint_auc_05cm'))+' |')
    lines+=['','*主表阈值 5 cm；10 cm 结果保留在逐回合数据中。AUC 为每 20 步检查点的梯形近似，不是逐帧精确积分。',
        '准确率与完整率使用面积均匀表面采样；完整率分母是策略无关的固定可观测真实表面集，未重建部分不会被排除。',
        'quality_proxy 是观测方向不足的启发式，不是已校准的三维误差；不能把补看收益全部归因于语义。',
        '所有方法同一重建后端。静态语义、几何补看、语义辅助补看均保留，无默认优胜者。','',
        '## 文件','',
        '- `*_truth.ply`：真实模型，只用于模拟器／评价器。',
        '- 每回合 `reconstruction.ply`：TSDF 重建网格；`frames/` 与 `scans/`：可回放原始观测。',
        '- `verification.json`：覆盖与全部检查点网格指标重算。',
        '- `overview.png`、`viewer.html`：图表和可旋转三维模型（选择固定的第一个正确语义场景）。']
    (source/'results.md').write_text('\n'.join(lines)+'\n')
    chosen=next(n for n in names if n.endswith('_aligned'));method='semantic_quality'
    fig,axes=plt.subplots(1,3,figsize=(14,4),layout='constrained')
    with np.load(source/(chosen+'_'+method)/'frames/00000.npz') as d:
        axes[0].imshow(d['depth_m'],cmap='viridis',vmin=0,vmax=4);axes[0].set_title('Simulated depth / metres');axes[0].axis('off')
    for m in config['methods']:
        values=scene_curves[(chosen,m)]
        axes[1].plot([v['step'] for v in values],[v['f1_05cm'] for v in values],label=m)
        axes[2].plot([v['step'] for v in values],[v['joint_05cm'] for v in values],label=m)
    axes[1].set_title('3D F1 at 5 cm');axes[2].set_title('2D coverage x 3D F1')
    for ax in axes[1:]:ax.set_xlabel('Primitive actions');ax.set_ylim(0,1);ax.grid(alpha=.2)
    axes[2].legend(fontsize=8);fig.suptitle(chosen+' (illustrative case, see all results)')
    fig.savefig(source/'overview.png',dpi=160);plt.close(fig)
    import plotly.graph_objects as go
    viewer=go.Figure()
    for title,file in [('真实场景',source/(chosen+'_truth.ply')),('三维重建',source/(chosen+'_'+method)/'reconstruction.ply')]:
        mesh=o3d.io.read_triangle_mesh(str(file));points,_=surface_samples(mesh,14000,123)
        viewer.add_trace(go.Scatter3d(x=points[:,0],y=points[:,1],z=points[:,2],mode='markers',name=title,
            marker=dict(size=1.5,color=points[:,2],colorscale='Viridis',opacity=.7),visible=True if title=='三维重建' else 'legendonly'))
    viewer.update_layout(title='三维虚拟实验：点击图例切换真实场景／重建（点为网格表面采样）',
                         scene=dict(aspectmode='data',xaxis_title='x / m',yaxis_title='y / m',zaxis_title='z / m'))
    viewer.write_html(str(source/'viewer.html'),include_plotlyjs=True,full_html=True)
    print(json.dumps({k:v for k,v in report.items() if k!='means'},indent=2))


if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('source');a=p.parse_args();analyze(a.source)
