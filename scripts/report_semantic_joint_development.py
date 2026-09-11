#!/usr/bin/env python3
"""Collect ALL completed v2/v3 development iterations without selecting wins."""
import argparse,json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def run(root,output):
    output.mkdir(parents=True,exist_ok=False)
    rows=[];stages=[]
    for folder in sorted(root.glob('joint_v[23]_*')):
        if not (folder/'run_metadata.json').exists():continue
        meta=json.loads((folder/'run_metadata.json').read_text())
        if meta['status']!='complete':continue
        config=json.loads((folder/'config.json').read_text())
        episodes=[json.loads(line) for line in (folder/'episodes.jsonl').read_text().splitlines()]
        full={r['scene']:r for r in episodes if r['method']=='full_v2'}
        geo={r['scene']:r for r in episodes if r['method']=='geometry_v2'}
        stat=dict(run=folder.name,episodes=len(episodes),seeds=sorted({r['seed'] for r in episodes}),methods={})
        for method in config['methods']:
            selected=[r for r in episodes if r['method']==method]
            stat['methods'][method]={k:float(np.mean([r[k] for r in selected])) for k in
                ('joint_auc_05cm','coverage_2d','f1_05cm','surface_error_mean_m','collisions')}
        if set(full)==set(geo) and full:
            stat['semantic_relative_auc_gain']=float(np.mean([r['joint_auc_05cm'] for r in full.values()])/np.mean([r['joint_auc_05cm'] for r in geo.values()])-1)
            stat['semantic_positive_scenes']=sum(full[s]['joint_auc_05cm']>geo[s]['joint_auc_05cm']+1e-9 for s in full)
            stat['paired_scenes']=len(full)
        stages.append(stat)
        for r in episodes:rows.append(dict(run=folder.name,**r))
    (output/'all_completed_runs.json').write_text(json.dumps(stages,indent=2)+'\n')
    (output/'all_episodes.json').write_text(json.dumps(rows,indent=2)+'\n')
    lines=['# 本轮联合规划与语义开发全记录','',
        '这里只汇总开发实验。重复使用种子和场景不能增加独立样本数，表中相对变化不能作为显著性证明。保留所有完整运行，包含退化版本。',
        '', '|开发运行|回合数|几何种子|完整方法联合 AUC|去语义联合 AUC|语义相对增量|','|---|---:|---|---:|---:|---:|']
    for s in stages:
        f=s['methods'].get('full_v2');g=s['methods'].get('geometry_v2')
        if not f or not g:continue
        lines.append(f"|[{s['run']}](../{s['run']}/results.md)|{s['episodes']}|{s['seeds']}|{f['joint_auc_05cm']:.5f}|{g['joint_auc_05cm']:.5f}|{s['semantic_relative_auc_gain']*100:+.3f}%|")
    lines+=['','## 最新版本逐场景结果','',
        '|场景条件|布局／种子|完整联合 AUC|去语义联合 AUC|Δ联合 AUC|Δ最终覆盖/百分点|','|---|---|---:|---:|---:|---:|']
    fig,axes=plt.subplots(1,2,figsize=(11,4),sharey=True)
    for ax,condition in zip(axes,('occluded','generic')):
        folder=f'joint_v3_channels_{condition}_development_20260910'
        posterior=[r for r in rows if r['run']=='joint_v3_posterior_development_20260910' and r.get('occluded_objects')==(condition=='occluded')]
        subset=posterior or [r for r in rows if r['run']==folder]
        full={r['scene']:r for r in subset if r['method']=='full_v2'}
        geo={r['scene']:r for r in subset if r['method']=='geometry_v2'}
        keys=sorted(full)
        for key in keys:
            f,g=full[key],geo[key]
            lines.append(f"|{condition}|{f['layout']}／{f['seed']}|{f['joint_auc_05cm']:.5f}|{g['joint_auc_05cm']:.5f}|{f['joint_auc_05cm']-g['joint_auc_05cm']:+.5f}|{(f['coverage_2d']-g['coverage_2d'])*100:+.2f}|")
        x=np.arange(len(keys));width=.36
        ax.bar(x-width/2,[geo[k]['joint_auc_05cm'] for k in keys],width,label='Geometry only',color='#a8b3c5')
        ax.bar(x+width/2,[full[k]['joint_auc_05cm'] for k in keys],width,label='Semantic full',color='#367db0')
        ax.set_xticks(x,[f"{full[k]['layout']}\n{full[k]['seed']}" for k in keys]);ax.set_title(condition+' objects')
        ax.set_ylim(0,.75);ax.grid(axis='y',alpha=.2);ax.set_axisbelow(True)
    axes[0].set_ylabel('AUC(2D coverage × F1@5cm)');axes[1].legend(frameon=False)
    fig.suptitle('Development only — matched backend, measured-shape fit and action budget');fig.tight_layout()
    fig.savefig(output/'semantic_ablation.png',dpi=180);fig.savefig(output/'semantic_ablation.pdf');plt.close(fig)
    current=[r for r in rows if r['run']=='joint_v3_posterior_development_20260910']
    if current:
        full={r['scene']:r for r in current if r['method']=='full_v2'}
        lines+=['','## 完整方案与同后端对照','',
            '|对照|完整方法联合 AUC 相对变化|最终覆盖差异/百分点|最终 F1 差异/百分点|','|---|---:|---:|---:|']
        for method in ('coverage','legacy_geometry','geometry_v2','no_hypotheses_v3'):
            base={r['scene']:r for r in current if r['method']==method}
            assert set(base)==set(full)
            avg=lambda group,key:np.mean([r[key] for r in group.values()])
            lines.append(f"|{method}|{100*(avg(full,'joint_auc_05cm')/avg(base,'joint_auc_05cm')-1):+.3f}%|{100*(avg(full,'coverage_2d')-avg(base,'coverage_2d')):+.3f}|{100*(avg(full,'f1_05cm')-avg(base,'f1_05cm')):+.3f}|")
    if current:
        lines+=['','## 布局分组（开发结果）','',
            '|布局|完整方法 − 纯覆盖：联合 AUC 相对变化|最终覆盖差异/百分点|完整方法 − 去物体假设：联合 AUC 相对变化|','|---|---:|---:|---:|']
        for layout in ('rooms','warehouse'):
            groups={m:[r for r in current if r['layout']==layout and r['method']==m] for m in ('full_v2','coverage','no_hypotheses_v3')}
            avg=lambda m,k:np.mean([r[k] for r in groups[m]])
            lines.append(f"|{layout}|{100*(avg('full_v2','joint_auc_05cm')/avg('coverage','joint_auc_05cm')-1):+.3f}%|{100*(avg('full_v2','coverage_2d')-avg('coverage','coverage_2d')):+.3f}|{100*(avg('full_v2','joint_auc_05cm')/avg('no_hypotheses_v3','joint_auc_05cm')-1):+.3f}%|")
    label_rows=[r for r in rows if r['run']=='joint_v3_labels_development_20260910']
    if label_rows and current:
        lines+=['','## 标签干预（相同最新实现）','',
            '|物体条件|标签|平均联合 AUC|平均最终覆盖|','|---|---|---:|---:|']
        for opaque in (False,True):
            for condition in ('aligned','shuffled','absent'):
                group=[r for r in current+label_rows if r.get('occluded_objects')==opaque and r['semantic_condition']==condition and r['method']=='full_v2']
                lines.append(f"|{'带背板' if opaque else '开放'}|{condition}|{np.mean([r['joint_auc_05cm'] for r in group]):.5f}|{np.mean([r['coverage_2d'] for r in group]):.5f}|")
    objectness=[r for r in rows if r['run']=='joint_v3_objectness_development_20260910']
    if objectness and current:
        key=lambda r:(r['layout'],r['seed'],r['occluded_objects'])
        baseline={key(r):r for r in objectness}
        full={key(r):r for r in current if r['method']=='full_v2'}
        assert set(baseline)==set(full)
        delta=np.mean([full[k]['joint_auc_05cm']-baseline[k]['joint_auc_05cm'] for k in full])
        relative=np.mean([full[k]['joint_auc_05cm'] for k in full])/np.mean([baseline[k]['joint_auc_05cm'] for k in full])-1
        lines+=['','## 细类别信息与二值前景语义的区分','',
            f'完整方法相对仅物体／背景语义对照：Δ联合 AUC = {delta:+.6f}，相对变化 {relative*100:+.3f}%。两者共享实测几何似然与物体候选。二值对照关闭类别形状先验和类别近看奖励，保留物体／背景区分。',
            '', '缺失标签与去语义方法的全部 8 组轨迹、二维已知掩码和最终网格完全一致；16 组标签干预的场景真值与初始非语义观测一致。见标签干预目录的 label_intervention_audit.json。']
    lines+=['','![语义消融](semantic_ablation.png)','',
        '源码、配置、逐帧深度、雷达、候选评分和最后网格分别保存在各实验目录。独立确认种子 301—308 尚未使用。当前开发结果不能声明完整语义方案已通过；语义增量仍是必须通过的核心判据。']
    (output/'results.md').write_text('\n'.join(lines)+'\n')
    print('completed development runs:',len(stages),'episodes:',len(rows))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,default=Path('eval_results'));p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();run(a.root,a.output)
