#!/usr/bin/env python3
"""Independent coverage/mesh/replay checks and paired seed-block intervals."""
import os
os.environ.setdefault('OMP_NUM_THREADS','2');os.environ.setdefault('OPENBLAS_NUM_THREADS','1')
import argparse
import hashlib
import json
from pathlib import Path
import sys
import zipfile
import numpy as np
import open3d as o3d
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from utils.virtual_scene_factory import scene_configuration,scene_key
from nso.mapping3d_v2 import QualityMapperV2
from nso.camera_mapping_v2 import CameraQualityMapperV2
from nso.semantic_completion_v3 import SemanticHistoryMapperV3
from utils.rgbd_contract import RGBDFrame,PlanarScan
from utils.reconstruction_metrics import ReconstructionEvaluator,ray_scene
from utils.cpu_protocol import digest_json


def mesh_from_npz(path):
    data=np.load(path,allow_pickle=False)
    mesh=o3d.geometry.TriangleMesh();mesh.vertices=o3d.utility.Vector3dVector(data['vertices']);mesh.triangles=o3d.utility.Vector3iVector(data['triangles'])
    return mesh


def analyze(output,replay=False):
    config=json.loads((output/'config.json').read_text());meta=json.loads((output/'run_metadata.json').read_text())
    assert meta['status']=='complete' and meta['config_sha256']==digest_json(config)
    with zipfile.ZipFile(output/'sources.zip') as archive:
        for name,sha in meta['files_sha256'].items():assert hashlib.sha256(archive.read(name)).hexdigest()==sha
    records=[json.loads(line) for line in (output/'episodes.jsonl').read_text().splitlines()]
    assert len(records)==meta['expected_episodes']
    expected=set();scene_configs={};control_keys={}
    for entry in config['scenes']:
        _,c=scene_configuration(config,entry)
        scene=scene_key(config,entry,meta.get('scene_key_version',1))
        scene_configs[scene]=c
        control_keys[scene]=scene_key(config,entry|{'semantic_condition':'aligned'},meta.get('scene_key_version',1))
        for method in config['methods']:expected.add((scene,method))
    actual=[(r['scene'],r['method']) for r in records]
    assert len(expected)==len(actual)==len(set(actual)) and set(actual)==expected,'experiment matrix mismatch'
    checked=0;replayed=0;paired_controls=0
    if any(method.endswith('_v4') for method in config['methods']) and 'full_method' not in config:
        raise ValueError('V4 matrix must explicitly identify full_method')
    if 'full_method' in config and config['full_method'] not in config['methods']:
        raise ValueError('full_method is not in the experiment matrix')
    for result in records:
        episode=output/result['artifact_dir'];ref=np.load(output/(result['scene']+'_reference.npz'),allow_pickle=False)
        evaluator=ReconstructionEvaluator.__new__(ReconstructionEvaluator)
        evaluator.reference=ref['points'];evaluator.classes=ref['classes'];evaluator.truth=ray_scene(mesh_from_npz(output/(result['scene']+'_reference.npz')))
        metrics=[json.loads(line) for line in (episode/'metrics.jsonl').read_text().splitlines()]
        steps=[json.loads(line) for line in (episode/'steps.jsonl').read_text().splitlines()]
        maps=np.load(episode/'maps.npz',allow_pickle=False);reachable=ref['reachable']
        masks=np.unpackbits(maps['known_packed'],axis=1)[:,:reachable.size].reshape(-1,*reachable.shape).astype(bool)
        coverage=np.sum(masks&reachable,axis=(1,2))/reachable.sum()
        np.testing.assert_array_equal([r['step'] for r in steps],np.arange(len(steps)))
        np.testing.assert_allclose(coverage,[r['coverage'] for r in steps],rtol=0,atol=1e-12)
        np.testing.assert_allclose(result['coverage_2d'],coverage[-1],rtol=0,atol=1e-12)
        assert result['step']==steps[-1]['step']==metrics[-1]['step']
        for checkpoint in metrics:
            np.testing.assert_allclose(checkpoint['coverage_2d'],coverage[checkpoint['step']],rtol=0,atol=1e-12)
        budget=scene_configs[result['scene']].max_steps
        coverage_auc=np.trapz(np.interp(np.arange(budget+1),np.arange(len(steps)),coverage))/budget
        np.testing.assert_allclose(result['coverage_auc'],coverage_auc,rtol=0,atol=1e-12)
        for threshold in config['thresholds_m']:
            tag=f'{round(threshold*100):02d}cm'
            auc=np.trapz(np.interp(np.arange(budget+1),[r['step'] for r in metrics],[r[f'joint_{tag}'] for r in metrics]))/budget
            np.testing.assert_allclose(auc,result[f'joint_auc_{tag}'],rtol=0,atol=1e-12)
        mesh=mesh_from_npz(episode/'final_mesh.npz')
        actual=evaluator.evaluate(mesh,float(coverage[-1]),config['thresholds_m'])
        for key,value in actual.items():
            if value is not None:np.testing.assert_allclose(value,result[key],rtol=0,atol=1e-9)
        checked+=1
        if replay:
            c=scene_configs[result['scene']]
            cls=CameraQualityMapperV2 if config.get('camera_coverage',False) else QualityMapperV2
            if config.get('object_hypotheses',False):cls=SemanticHistoryMapperV3
            mapper=cls(reachable.shape,c,c.truncation_m)
            checkpoints={r['step']:r for r in metrics}
            for path in sorted((episode/'frames').glob('*.npz')):
                frame=RGBDFrame.load(path);scan=PlanarScan.load(episode/'scans'/path.name)
                mapper.update(frame,scan)
                index=int(path.stem)
                if index in checkpoints:
                    current=evaluator.evaluate(mapper.mesh(),coverage[index],config['thresholds_m'])
                    for key,value in current.items():
                        if value is not None:np.testing.assert_allclose(value,checkpoints[index][key],rtol=0,atol=1e-9)
                    replayed+=1
            np.testing.assert_array_equal(np.asarray(mapper.mesh().vertices),np.asarray(mesh.vertices))
            np.testing.assert_array_equal(np.asarray(mapper.mesh().triangles),np.asarray(mesh.triangles))
        print('verified',result['artifact_dir'],flush=True)
    # Same-seed pairing; layout/noise conditions are repeated measures.
    comparisons=[]
    full_method=config.get('full_method','full_v2')
    for baseline in config['methods']:
        if baseline==full_method:continue
        differences=[]
        for seed in sorted({r['seed'] for r in records}):
            full={r['scene']:r for r in records if r['seed']==seed and r['method']==full_method}
            base={r['scene']:r for r in records if r['seed']==seed and r['method']==baseline}
            if set(full)!=set(base) or not full:continue
            differences.append({key:float(np.mean([full[s][key]-base[s][key] for s in full])) for key in
                ('joint_auc_05cm','coverage_2d','f1_05cm','surface_error_mean_m')})
        if not differences:continue
        rng=np.random.default_rng(9981);n=len(differences);indices=rng.integers(n,size=(20000,n))
        row=dict(baseline=baseline,seed_blocks=n)
        for key in differences[0]:
            values=np.array([r[key] for r in differences]);bootstrap=values[indices].mean(axis=1)
            row[key]=dict(mean=float(values.mean()),ci95=list(map(float,np.quantile(bootstrap,[.025,.975]))) if n>=2 else None,
                          positive_blocks=int(np.count_nonzero(values>0)))
        comparisons.append(row)
    for method in ('coverage','legacy_geometry','geometry_v2','geometry_v4'):
        for a in records:
            if a['method']!=method or a['semantic_condition']!='aligned':continue
            matches=[b for b in records if b['method']==method and b['semantic_condition']!='aligned'
                     and control_keys[b['scene']]==control_keys[a['scene']]]
            for b in matches:
                x=np.load(output/a['artifact_dir']/'maps.npz');y=np.load(output/b['artifact_dir']/'maps.npz')
                np.testing.assert_array_equal(x['poses'],y['poses']);paired_controls+=1
    verification=dict(status='passed',episodes=checked,replayed_checkpoints=replayed,nonsemantic_control_pairs=paired_controls,paired_comparisons=comparisons,
        scope=config['scope'],no_robot_or_full_slam_claim=True)
    verification['analysis_source_sha256']={name:hashlib.sha256((ROOT/name).read_bytes()).hexdigest()
        for name in ('scripts/analyze_joint_v2.py','utils/virtual_scene_factory.py')}
    verification['analysis_sources_match_run_archive']=all(meta['files_sha256'].get(name)==sha
        for name,sha in verification['analysis_source_sha256'].items())
    (output/'verification.json').write_text(json.dumps(verification,indent=2,ensure_ascii=False)+'\n')
    lines=['# 联合规划闭环对照','',f"范围：{config['scope']}。{len(records)} 回合；按种子配对，布局／噪声条件不当作独立样本。",'',
        '|方法|覆盖 AUC|最终覆盖|联合 AUC@5cm|最终 F1@5cm|表面误差/mm|碰撞|','|---|---:|---:|---:|---:|---:|---:|']
    for method in config['methods']:
        rows=[r for r in records if r['method']==method];avg=lambda k:np.mean([r[k] for r in rows])
        lines.append(f'|{method}|{avg("coverage_auc"):.4f}|{avg("coverage_2d"):.4f}|{avg("joint_auc_05cm"):.4f}|{avg("f1_05cm"):.4f}|{avg("surface_error_mean_m")*1000:.3f}|{sum(r["collisions"] for r in rows)}|')
    lines+=['','|完整方法 − 对照|Δ联合 AUC|配对种子 bootstrap 95% 区间|','|---|---:|---|']
    for r in comparisons:
        value=r['joint_auc_05cm'];ci=value['ci95']
        interval=f'[{ci[0]:+.5f}, {ci[1]:+.5f}]' if ci is not None else '不估计：只有一个种子块'
        lines.append(f'|{r["baseline"]}|{value["mean"]:+.5f}|{interval}|')
    lines+=['','区间按固定协议比较报告；开发集区间不构成独立确认。语义、路线、完整方法优势必须分别看对应消融，不能仅靠比纯覆盖更好推断。',
        '',f'复核：{checked} 个最终网格与逐步二维覆盖已重算，{replayed} 个检查点经过 RGB-D／雷达回放复算。',
        '', '在线质量是观测信息构成的有界代理，未宣称校准为真实 F1。语义来源按场景配置区分模拟可见标签与理想 RGB 人工标识，均不验证自然图像开放词汇识别；位姿扰动为有界输入误差，导航使用离散真实位姿，不是完整 SLAM 估计器。']
    (output/'results.md').write_text('\n'.join(lines)+'\n')


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('output',type=Path);p.add_argument('--replay',action='store_true')
    args=p.parse_args();analyze(args.output,args.replay)
