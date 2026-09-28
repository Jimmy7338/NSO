#!/usr/bin/env python3
"""Audit and summarize matched geometry/semantic interventions, retaining all cases."""
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap


def interval(values):
    values=np.asarray(values,float);rng=np.random.default_rng(20260911)
    sampled=values[rng.integers(len(values),size=(10000,len(values)))].mean(1)
    return dict(mean=float(values.mean()),ci95=np.percentile(sampled,[2.5,97.5]).tolist())


def analyze(folder):
    folder=Path(folder);config=json.loads((folder/'config.json').read_text());meta=json.loads((folder/'run_metadata.json').read_text())
    if meta['status']!='complete' or not meta['frozen_sources_verified']:raise ValueError('incomplete scenario suite')
    records=[json.loads(s) for s in (folder/'episodes.jsonl').read_text().splitlines()]
    expected={(f"{m['family']}_{m['size']}_seed{m['seed']}",case) for m in config['maps'] for case in config['cases']}
    keys=[(r['map_id'],r['case']) for r in records]
    if len(keys)!=len(expected) or set(keys)!=expected:raise ValueError('incomplete/duplicate scenario matrix')
    signatures={};curves=defaultdict(list);verified=0
    for r in records:
        path=folder/r['artifact_dir']
        with np.load(path/'observations.npz') as d:
            world=d['occupancy'];h,w=world.shape
            pair=(r['ground_truth_sha256'],tuple(r['initial_pose']))
            if r['map_id'] in signatures and signatures[r['map_id']]!=pair:raise ValueError('unpaired physical geometry/start')
            signatures[r['map_id']]=pair
            masks=np.unpackbits(d['visible_packed'],axis=1)[:,:h*w].reshape(-1,h,w)
            known=np.maximum.accumulate(masks,axis=0)
            np.testing.assert_array_equal(known[-1],d['final_belief']!=-1)
            curve=np.count_nonzero(known&d['reachable'],axis=(1,2))/d['reachable'].sum()
            full=np.interp(np.arange(config['environment']['max_steps']+1),np.arange(len(curve)),curve)
            if abs(full[-1]-r['coverage_ratio'])>1e-12 or abs(np.trapz(full)/config['environment']['max_steps']-r['coverage_auc'])>1e-12:raise ValueError('physical coverage mismatch')
            if abs(np.count_nonzero(known[-1]&d['productive_mask'])/d['productive_mask'].sum()-r['productive_area_coverage'])>1e-12:raise ValueError('productive area metric mismatch')
            seen=masks[:-1].any(0);memory=d['final_semantic_belief']
            if np.any(memory[~seen]):raise ValueError('hidden semantic leakage')
            if r['semantic_updates']:np.testing.assert_array_equal(memory[seen],d['semantic_field'][seen])
            curves[(r['case'],r['size'])].append(full)
            verified+=1
    fields=['coverage_auc','coverage_ratio','explored_area_m2','productive_area_coverage','productive_landmarks_seen','revisit_ratio','collisions','episode_wall_time_s']
    report=dict(schema='targeted_scenes_summary_v1',episodes=len(records),maps=len(signatures),
                scope=meta['scope'],audit=dict(episodes_reconstructed=verified,matched_geometry_start=True,semantic_visibility=True),
                primary_metric='coverage_auc',means={},by_family_size={},contrasts={},limitations=meta['limitations'])
    for case in config['cases']:
        rows=[r for r in records if r['case']==case]
        report['means'][case]={k:float(np.mean([r[k] for r in rows])) for k in fields}
        for family,size in sorted({(r['family'],r['size']) for r in rows}):
            selected=[r for r in rows if r['family']==family and r['size']==size]
            report['by_family_size'].setdefault(f'{family}_{size}',{})[case]={k:float(np.mean([r[k] for r in selected])) for k in fields}
    pairs=[('geometry','semantic_aligned'),('geometry','combined_aligned'),('semantic_shuffled','semantic_aligned'),
           ('semantic_constant','semantic_aligned'),('combined_shuffled','combined_aligned'),('semantic_aligned','combined_aligned'),
           ('geometry','semantic_noisy'),('semantic_noisy','semantic_aligned')]
    for a,b in pairs:
        if a not in config['cases'] or b not in config['cases']:continue
        left={r['map_id']:r for r in records if r['case']==a};right={r['map_id']:r for r in records if r['case']==b}
        diff={k:interval([right[m][k]-left[m][k] for m in left]) for k in ('coverage_auc','coverage_ratio','productive_area_coverage')}
        diff['auc_wins_ties_losses']=[sum(right[m]['coverage_auc']>left[m]['coverage_auc']+1e-12 for m in left),
             sum(abs(right[m]['coverage_auc']-left[m]['coverage_auc'])<=1e-12 for m in left),
             sum(right[m]['coverage_auc']<left[m]['coverage_auc']-1e-12 for m in left)]
        # Families/sizes reuse seeds: expose seed blocks rather than treating
        # all geometry variants as independent evidence of generalization.
        by_seed=defaultdict(list)
        for m in left:by_seed[left[m]['map_seed']].append(right[m]['coverage_auc']-left[m]['coverage_auc'])
        diff['auc_by_seed_block']={str(k):float(np.mean(v)) for k,v in by_seed.items()}
        report['contrasts'][f'{b} minus {a}']=diff
    (folder/'summary.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    sizes=sorted({r['size'] for r in records});fig,axes=plt.subplots(1,len(sizes),figsize=(6*len(sizes),4.5),squeeze=False)
    for size,ax in zip(sizes,axes[0]):
        for case in config['cases']:
            x=np.asarray(curves[(case,size)])*100
            ax.plot(x.mean(0),label=case)
        ax.set(title=f'{size} cells, {size*config["environment"]["resolution_m"]:.0f} m extent',xlabel='Atomic actions',ylabel='Physical coverage (%)',ylim=(0,100));ax.grid(alpha=.2)
    axes[0,-1].legend(fontsize=8);fig.tight_layout();fig.savefig(folder/'coverage_by_scale.png',dpi=160);plt.close(fig)
    # Fixed example: first seed at each size, no selection by performance.
    scene_files=sorted(folder.glob('*.scene.json'));families=list(dict.fromkeys(m['family'] for m in config['maps']))
    fig,axes=plt.subplots(len(families),len(sizes),figsize=(6*len(sizes),5*len(families)),squeeze=False)
    for i,family in enumerate(families):
        for j,size in enumerate(sizes):
            entry=next(m for m in config['maps'] if m['family']==family and m['size']==size)
            mid=f'{family}_{size}_seed{entry["seed"]}';scene=json.loads((folder/f'{mid}.scene.json').read_text())
            with np.load(folder/f'{mid}.scene.npz') as d:
                ax=axes[i,j];ax.imshow(d['occupancy'],cmap=ListedColormap(['#f2f4f5','#384655']),vmin=0,vmax=1)
                for room in scene['rooms']:
                    r,c=room['entrance'];ax.scatter(c,r,c='#e78b24' if room['productive'] else '#3598c5',s=22,marker='s')
                for case,color in [('geometry','#687c92'),('semantic_aligned','#cb3b3b')]:
                    if case not in config['cases']:continue
                    with np.load(folder/f'{mid}_{case}'/'observations.npz') as e:
                        poses=e['poses'];ax.plot(poses[:,1],poses[:,0],label=case,color=color,lw=.85,alpha=.9)
                ax.set_title(f'{family} | {scene["extent_m"]:.0f} m | {scene["room_count"]} rooms\nreachable area {scene["reachable_area_m2"]:.0f} m²')
                ax.set_xticks([]);ax.set_yticks([])
    axes[0,0].legend(fontsize=8);fig.suptitle('Evaluation-only full layouts | orange: functional room cues, blue: utility cues')
    fig.tight_layout(rect=(0,0,1,.97));fig.savefig(folder/'scene_examples.png',dpi=140);plt.close(fig)
    lines=['# 大规模功能语义场景实验结果','',f"共 {len(signatures)} 张地图、{len(records)} 回合。面积覆盖仍为主要任务，功能区域覆盖仅为次要指标。",'',
           '| 条件 | 覆盖 AUC | 最终覆盖率 | 功能区域覆盖率 |','|---|---:|---:|---:|']
    for case,v in report['means'].items():lines.append(f"| {case} | {v['coverage_auc']:.4f} | {v['coverage_ratio']:.2%} | {v['productive_area_coverage']:.2%} |")
    lines.extend(['','全部物理覆盖与 AUC 从逐步可见掩码重建；同图各条件起点完全配对。语义与房间功能/规模的关联由生成器明确提供，不能解释为真实视觉识别效果。','',
                  '逐场景/尺度结果及完整配对区间见 summary.json；区间为探索性地图重采样，模板相似性及多重对比限制外推。'])
    (folder/'results.md').write_text('\n'.join(lines)+'\n')
    print(json.dumps(dict(episodes=verified,means=report['means'],contrasts=report['contrasts']),indent=2))
    return report

if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('folder',type=Path);analyze(p.parse_args().folder)
