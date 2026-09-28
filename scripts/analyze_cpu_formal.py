#!/usr/bin/env python3
"""Prespecified paired map summaries and independent archive reconstruction."""
import csv
import hashlib
import json
from pathlib import Path
import sys
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from utils.cpu_protocol import geometry_hash,digest_json


def bootstrap(values):
    values=np.asarray(values,float)
    rng=np.random.default_rng(20260910)
    samples=values[rng.integers(0,len(values),(10000,len(values)))].mean(1)
    return dict(mean=float(values.mean()),ci95=list(map(float,np.percentile(samples,[2.5,97.5]))))


def analyze(folder):
    folder=Path(folder)
    config=json.loads((folder/'config.json').read_text())
    protocol=json.loads((folder/'frozen_protocol.json').read_text())
    meta=json.loads((folder/'run_metadata.json').read_text())
    if not meta.get('frozen_protocol_verified_at_end') or meta['status']!='complete':
        raise ValueError('formal run did not complete with frozen code')
    if digest_json(config)!=protocol['config_sha256']:raise ValueError('archive config changed')
    episodes=[json.loads(s) for s in (folder/'episodes.jsonl').read_text().splitlines()]
    expected={(f"{m['layout']}_seed{m['seed']}",s,k) for m in config['maps'] for s in config['start_seeds'] for k in config['methods']}
    keys=[(e['map_id'],e['start_seed'],e['method']) for e in episodes]
    if len(keys)!=len(expected) or set(keys)!=expected or any(e['status']!='complete' for e in episodes):
        raise ValueError('incomplete/duplicate/unexpected formal matrix')
    if len(episodes)!=protocol['expected_episodes']:raise ValueError('protocol count mismatch')
    fields=['coverage_auc','coverage_ratio','revisit_ratio','path_length_m','collisions']
    bymap={}
    for method in config['methods']:
        bymap[method]={}
        for name in protocol['maps']:
            runs=[e for e in episodes if e['method']==method and e['map_id']==name]
            bymap[method][name]={k:float(np.mean([e[k] for e in runs])) for k in fields}
    paired={}
    pairs=[('gain_control','gain_semantic_structure'),('gain_control','gain_semantic'),
           ('gain_control','gain_structure'),('gain_semantic_structure','gain_semantic_structure_rpn'),
           ('gain_semantic_structure','gain_semantic_structure_budget_soft'),
           ('gain_semantic_structure_budget_soft','gain_semantic_structure_rpn'),
           ('nearest_frontier','gain_per_cost'),('gain_per_cost','gain_control')]
    for a,b in pairs:
        paired[f'{b} minus {a}']={k:bootstrap([bymap[b][m][k]-bymap[a][m][k] for m in protocol['maps']]) for k in fields}
    assets={};attempts_checked=0
    for e in episodes:
        path=folder/e['artifact_dir']
        with np.load(path/'observations.npz') as d:
            h,w=d['map_shape'];world=d['occupancy']
            if geometry_hash(world)!=protocol['maps'][e['map_id']]:raise ValueError('archived map mismatch')
            key=(e['map_id'],e['start_seed'])
            signature=(e['ground_truth_sha256'],tuple(e['initial_pose']),e['semantic_sha256'])
            if key in assets and assets[key]!=signature:raise ValueError('unpaired starting assets')
            assets[key]=signature
            masks=np.unpackbits(d['visible_packed'],axis=1)[:,:h*w].reshape(-1,h,w)
            known=np.maximum.accumulate(masks,axis=0)
            np.testing.assert_array_equal(known[-1],d['final_belief']!=-1)
            coverage=np.count_nonzero(known & d['reachable'],axis=(1,2))/d['reachable'].sum()
            curve=np.interp(np.arange(config['environment']['max_steps']+1),np.arange(len(coverage)),coverage)
            if abs(e['coverage_ratio']-coverage[-1])>1e-12 or abs(e['coverage_auc']-np.trapz(curve)/config['environment']['max_steps'])>1e-12:
                raise ValueError('coverage/AUC failed independent reconstruction')
            if (path/'goal_attempts.jsonl').exists():
                rows=[json.loads(s) for s in (path/'goal_attempts.jsonl').read_text().splitlines()]
                if len(rows)!=e['goal_changes']:raise ValueError('missing goal outcomes')
                for row in rows:
                    end=d['poses'][row['end_step']]
                    if row['controller_success']==1 and list(end)!=row['goal']+[row['goal_heading']]:raise ValueError('false success label')
                    if 'censored' in row['status'] and row['controller_success'] is not None:raise ValueError('censored label misuse')
                    attempts_checked+=1
                if e['semantic_updates']:
                    seen=masks[:-1].any(0);memory=d['final_semantic_belief']
                    np.testing.assert_array_equal(memory[seen],d['synthetic_semantic_ground_truth'][seen])
                    if np.any(memory[~seen]):raise ValueError('unseen semantic leakage')
        if e['method'].endswith('_rpn'):
            if not e['rpn_frozen_verified'] or e['rpn_sha256']!=hashlib.sha256((folder/'rpn_frozen.npz').read_bytes()).hexdigest():
                raise ValueError('RPN fingerprint mismatch')
    report=dict(schema='cpu_formal_analysis_v1',map_count=len(protocol['maps']),episode_count=len(episodes),
                primary_metric=protocol['primary_metric'],primary_contrast=protocol['primary_contrast'],
                statistics=protocol['statistics'],paired_differences=paired,
                means={m:{k:float(np.mean([r[k] for r in values.values()])) for k in fields} for m,values in bymap.items()},
                audit=dict(episodes_reconstructed=len(episodes),goal_attempts_checked=attempts_checked,
                           canonical_geometry_verified=True,paired_assets_verified=True,semantic_visibility_verified=True,frozen_rpn_verified=True),
                limitations=protocol['limitations'])
    (folder/'formal_summary.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    with (folder/'map_means.csv').open('w') as f:
        writer=csv.DictWriter(f,fieldnames=['method','map_id']+fields);writer.writeheader()
        for method,values in bymap.items():
            for name,row in values.items():writer.writerow(dict(method=method,map_id=name,**row))
    lines=['# 冻结 CPU 二维对比结果','',f"{len(protocol['maps'])} 张地图 × {len(config['start_seeds'])} 个起点 × {len(config['methods'])} 种方法，共 {len(episodes)} 回合。",
           '','| 方法 | 覆盖 AUC | 最终覆盖率 | 重复移动比例 |','|---|---:|---:|---:|']
    for m,r in report['means'].items():lines.append(f"| {m} | {r['coverage_auc']:.4f} | {r['coverage_ratio']:.2%} | {r['revisit_ratio']:.2%} |")
    lines.extend(['','主要对比：语义加结构减预算对照。'])
    primary=paired['gain_semantic_structure minus gain_control']['coverage_auc']
    lines.append(f"覆盖 AUC 差值 {primary['mean']:+.4f}，按地图配对 bootstrap 95% 区间 [{primary['ci95'][0]:+.4f}, {primary['ci95'][1]:+.4f}]。")
    lines.extend(['','先对起点平均，再对地图等权平均；程序生成地图数量有限，区间只描述本测试套件的不确定性。次要比较未作多重比较校正，不据此挑选显著结果。',
                  '','全部回合的覆盖率及 AUC 已由可见掩码重建，目标结果和冻结权重已核验。二维合成语义与完美里程计不能证明真实视觉感知或三维 SLAM 效果。'])
    (folder/'paper_results.md').write_text('\n'.join(lines)+'\n')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    names=list(paired);means=[paired[n]['coverage_auc']['mean'] for n in names]
    low=[v-paired[n]['coverage_auc']['ci95'][0] for n,v in zip(names,means)]
    high=[paired[n]['coverage_auc']['ci95'][1]-v for n,v in zip(names,means)]
    fig,ax=plt.subplots(figsize=(11,5))
    ax.errorbar(means,np.arange(len(names)),xerr=[low,high],fmt='o',capsize=3)
    ax.axvline(0,color='gray',linestyle='--');ax.set_yticks(np.arange(len(names)),names,fontsize=8)
    ax.set(xlabel='Paired coverage AUC difference (95% map bootstrap interval)',title='Frozen CPU suite | secondary contrasts exploratory')
    fig.tight_layout();fig.savefig(folder/'paired_auc.png',dpi=160);plt.close(fig)
    print(json.dumps(dict(episodes=len(episodes),primary=primary,audit=report['audit']),indent=2))
    return report

if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('folder',type=Path);analyze(p.parse_args().folder)
