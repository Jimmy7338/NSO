#!/usr/bin/env python3
"""Recompute saved-mesh metrics and report all predeclared interventions."""
import os
os.environ.setdefault('OMP_NUM_THREADS','2')
os.environ.setdefault('OPENBLAS_NUM_THREADS','1')
import argparse
import hashlib
import json
from pathlib import Path
import sys
import zipfile
import numpy as np
import open3d as o3d
from scipy.stats import spearmanr
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from utils.inspection_benchmark import TargetEvaluator
from utils.reconstruction_metrics import ray_scene
from utils.cpu_protocol import file_hash,digest_json


def mean(rows,key,section='delta'):
    return float(np.mean([r[section][key] for r in rows]))


def analyze(output):
    config = json.loads((output/'config.json').read_text())
    metadata = json.loads((output/'run_metadata.json').read_text())
    assert metadata['status']=='complete'
    assert digest_json(config)==metadata['config_sha256']
    artifacts = json.loads((output/'artifacts_sha256.json').read_text())
    for name,sha in artifacts.items():
        assert file_hash(output/name)==sha,name
    with zipfile.ZipFile(output/'sources.zip') as archive:
        for name,sha in metadata['files_sha256'].items():
            assert hashlib.sha256(archive.read(name)).hexdigest()==sha,name
    rows = [json.loads(line) for line in (output/'measurements.jsonl').read_text().splitlines()]
    keys = {(r['seed'],r['geometry'],r['noise'],r['branch'],r['checkpoint']) for r in rows}
    expected = {(s,g,n,b,k) for s in config['seeds'] for g in config['geometries']
                for n in config['noise_models'] for b in config['branches'] for k in config['checkpoints']}
    assert keys==expected and len(keys)==len(rows)
    checked = 0
    evaluators = {}
    checked_baselines = set()
    for row in rows:
        scene = row['scene']
        if scene not in evaluators:
            # Rebuild evaluator from archived truth/reference, not simulator.
            reference = np.load(output/scene/'reference.npz',allow_pickle=False)
            evaluator = TargetEvaluator.__new__(TargetEvaluator)
            evaluator.reference = reference['points']
            evaluator.roi = reference['roi']
            evaluator.truth = ray_scene(o3d.io.read_triangle_mesh(str(output/scene/'truth.ply')))
            evaluator.prediction_samples = config['prediction_samples']
            evaluators[scene] = evaluator
        evaluator = evaluators[scene]
        group = output/scene/row['noise']
        support = np.load(group/'support.npz',allow_pickle=False)['mask']
        mesh = o3d.io.read_triangle_mesh(str(output/row['mesh']))
        measured = evaluator.evaluate(mesh,support)
        baseline = json.loads((group/'baseline.json').read_text())
        if str(group) not in checked_baselines:
            baseline_mesh = o3d.io.read_triangle_mesh(str(group/'baseline.ply'))
            _,baseline_distances = evaluator.distances(baseline_mesh)
            np.testing.assert_array_equal(support,baseline_distances<=.05)
            reconstructed_baseline = evaluator.evaluate(baseline_mesh,support)
            for key,value in reconstructed_baseline.items():
                if value is not None:
                    np.testing.assert_allclose(value,baseline[key],atol=1e-9,rtol=0)
            checked_baselines.add(str(group))
        for key,value in measured.items():
            if value is not None:
                np.testing.assert_allclose(value,row['metrics'][key],rtol=0,atol=1e-9,err_msg=row['mesh']+key)
        for key,value in row['delta'].items():
            np.testing.assert_allclose(value,measured[key]-baseline[key],atol=1e-9,rtol=0)
            if row['branch']=='no_update':
                np.testing.assert_allclose(value,0.,atol=1e-12)
        checked += 1
    final = [r for r in rows if r['checkpoint']==max(config['checkpoints'])]
    summaries = []
    for split in ('development','validation','all'):
        for noise in config['noise_models']:
            for branch in config['branches']:
                subset = [r for r in final if r['noise']==noise and r['branch']==branch and (split=='all' or r['split']==split)]
                summaries.append(dict(split=split,noise=noise,branch=branch,n=len(subset),
                    f1_gain=mean(subset,'f1_05cm'),recall_gain=mean(subset,'r_05cm'),precision_gain=mean(subset,'p_05cm'),
                    surface_error_improvement_mm=-1000*mean(subset,'surface_error_m'),
                    fixed_support_improvement_mm=-1000*mean(subset,'fixed_support_distance_m')))
    # Rank *within* each identical-prefix problem. Geometry/scene difficulty
    # cannot manufacture a pooled correlation. No-update/duplicate are controls.
    ranks = []
    candidate_branches = ('repeat_fresh','near','far','side','opposite')
    for seed in config['seeds']:
        for geometry in config['geometries']:
            for noise in config['noise_models']:
                for checkpoint in config['checkpoints']:
                    group = sorted([r for r in rows if (r['seed'],r['geometry'],r['noise'],r['checkpoint'])==(seed,geometry,noise,checkpoint) and r['branch'] in candidate_branches],key=lambda r:r['branch'])
                    for objective in ('f1_05cm','fixed_support_distance_m'):
                        gains = np.array([r['delta'][objective] for r in group]) * (-1 if objective=='fixed_support_distance_m' else 1)
                        for name in ('angular','semantic','ungated'):
                            # Initial forecast is the deployable decision signal;
                            # cumulative proxy is logged only for later diagnosis.
                            scores = np.array([r['initial_proxy'][name] for r in group])
                            corr = float(spearmanr(scores,gains).statistic) if np.ptp(scores)>0 and np.ptp(gains)>0 else None
                            chosen = int(np.argmax(scores))
                            ranks.append(dict(seed=seed,geometry=geometry,noise=noise,checkpoint=checkpoint,
                                split=group[0]['split'],objective=objective,proxy=name,spearman=corr,
                                chosen_branch=group[chosen]['branch'],best_branch=group[int(np.argmax(gains))]['branch'],
                                regret=float(gains.max()-gains[chosen]),chosen_gain=float(gains[chosen])))
    verification = dict(status='passed',checked_mesh_checkpoints=checked,checked_baselines=len(checked_baselines),artifact_hashes=len(artifacts),
        matrix_complete=True,no_update_zero_gain=True,archived_sources_verified=True,
        geometry_instances=len(config['seeds'])*len(config['geometries']),
        independent_seed_blocks=len(config['seeds']),noise_conditions_are_paired_not_independent=True,
        navigation_claim=False,semantic_increment_claim=False)
    for name,data in [('summary.json',summaries),('proxy_ranking.json',ranks),('verification.json',verification)]:
        (output/name).write_text(json.dumps(data,indent=2,ensure_ascii=False,allow_nan=False)+'\n')
    labels = dict(no_update='不更新',duplicate_last='重复同一帧',repeat_fresh='原地新帧',near='靠近',far='远离',side='侧向45°',opposite='背面180°')
    lines = ['# 重复观测与质量奖励：受控机制实验','',
        f"冻结协议 `{config['protocol']}`；{verification['geometry_instances']} 个几何实例、4 个种子块、4 种噪声、7 个分支、{checked} 个网格检查点。每个分支共享相同 4 帧前缀，增加 1/4/12 次观测。",
        '', '**范围**：静态物体的端点观测干预；不是机器人导航回合。目标位置只用于搭建受控实验与评价，代理分数仅使用已观测数据。视线朝向采用连续偏航扩展，因此检验的是原奖励公式，不是原离散控制器的完整行为。',
        '', '为使奖励与目标物体评价范围一致，排序使用传感器已观测物体标签（>1）筛出的表面，对角度和语义版本应用相同筛选；没有使用真实包围框或未来帧。全场景角度分数另存为 `scene_angular`，不与目标局部质量直接作预测相关。此设置使用现成合成分割，不能证明不依赖语义的物体发现能力。单一物体类别权重只是常数，因此本实验不应据其排名宣称语义增益。',
        '', '主评价为固定、与分支无关的可观测目标表面，按面积采样；阈值同时保留 1/2/5/10 cm。P 来自目标包围区域内重建面到真实目标的距离，R 来自固定参考点到重建面的距离。',
        '', '“固定支持面”是相同前缀中已在 5 cm 内重建的参考点，后续对这些相同点计算到重建面的平均距离（截断 10 cm），同时报告保留率；它控制评价表面的变化，但仍是几何距离诊断，不是定位精度。表面平均误差的采样支持随重建变化，必须与此指标一同看。',
        '', '所有收益是相对各自相同前缀的变化；误差改善为正表示距离下降。不更新分支仍分配同样等待时长。其余分支增加同样帧数；移动路径距离另记，时间仅记录含净转角的下界，没有模拟途中采集与全部转弯，不能用这些结果宣称相同总任务时间下的规划优势。',
        '', '## 全部几何实例，追加 12 帧','']
    for noise in config['noise_models']:
        lines += [f'### {noise}', '', '|干预|ΔF1@5cm|ΔR@5cm|ΔP@5cm|表面误差改善/mm|固定支持面改善/mm|', '|---|---:|---:|---:|---:|---:|']
        for r in summaries:
            if r['split']=='all' and r['noise']==noise:
                lines.append(f"|{labels[r['branch']]}|{r['f1_gain']:+.4f}|{r['recall_gain']:+.4f}|{r['precision_gain']:+.4f}|{r['surface_error_improvement_mm']:+.3f}|{r['fixed_support_improvement_mm']:+.3f}|")
        lines += ['']
    lines += ['## 未用于调整公式的验证种子：初始奖励与实际收益排序','',
              '种子 101/102 为开发分组，103/104 为验证分组；本阶段未拟合模型或选择权重。下表在每个相同前缀的五个实际候选内计算相关，再报告均值；噪声条件与两类物体共用种子，不能当作独立大样本。','',
              '|预测目标（追加12帧）|代理|平均 Spearman|平均选点遗憾|', '|---|---|---:|---:|']
    for objective in ('f1_05cm','fixed_support_distance_m'):
        for proxy in ('angular','semantic','ungated'):
            subset = [r for r in ranks if r['split']=='validation' and r['checkpoint']==12 and r['objective']==objective and r['proxy']==proxy]
            correlations = [r['spearman'] for r in subset if r['spearman'] is not None]
            lines.append(f"|{objective}|{proxy}|{np.mean(correlations):+.3f}|{np.mean([r['regret'] for r in subset]):.6f}|")
    lines += ['', '遗憾 = 此组实际最大收益 − 代理选择的候选收益；固定支持面目标的单位为米。并列分数按候选名称排序确定，不使用真实收益破平局。`ungated` 移除二维遮挡门控，仅作诊断。',
        '', '## 复核和边界','',
        f"所有 {checked} 个保存网格的指标已重算，完整矩阵、文件哈希、源码快照、不更新零收益均通过。完整逐例数据见 `measurements.jsonl`，排序失败例见 `proxy_ranking.json`，分组均值见 `summary.json`。",
        '', '独立噪声为逐像素独立高斯（1/3 cm）；固定误差为传感器像素上持续不变的 1 cm 高斯样本，不是实车标定模型。完美位姿、无丢帧、6 cm TSDF、96×72 深度图；当前结果不代表真实 ZED、位姿漂移或大规模 SLAM。',
        '', '复杂几何与类别仍绑定，因此本实验只能判断补看机制和旧奖励的预测能力，不能单独证明语义优于几何。验证分组的原始数据现已公开；后续调参后必须用新种子做最终评估。']
    (output/'results.md').write_text('\n'.join(lines)+'\n')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes = plt.subplots(1,3,figsize=(15,4.5))
    branches = ['repeat_fresh','near','far','side','opposite']
    colors = ['#3976af','#d0833c']
    for i,geometry in enumerate(('box','shelf')):
        subset = [r for r in final if r['geometry']==geometry and r['noise']=='iid_01']
        for ax,key,scale,title in zip(axes,['f1_05cm','r_05cm','fixed_support_distance_m'],[1,1,-1000],['F1 gain @ 5 cm','Completeness gain @ 5 cm','Fixed-support improvement (mm)']):
            vals = [mean([r for r in subset if r['branch']==b],key)*scale for b in branches]
            ax.bar(np.arange(5)+(i-.5)*.36,vals,.36,label=geometry,color=colors[i])
            ax.set_xticks(np.arange(5),['Repeat','Near','Far','Side','Back'],rotation=20)
            ax.set_title(title)
            ax.axhline(0,color='black',linewidth=.7)
    axes[0].legend()
    fig.suptitle('Controlled inspection: 12 extra frames, IID depth noise 1 cm; 4 seeds per geometry')
    fig.tight_layout()
    fig.savefig(output/'mechanism.png',dpi=160)
    plt.close(fig)
    print(json.dumps(verification,ensure_ascii=False))


if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('output',type=Path)
    analyze(parser.parse_args().output)
