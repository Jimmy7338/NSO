#!/usr/bin/env python3
"""Predeclared seed-block tests and conservative claim gates for frozen v2."""
import argparse
import itertools
import json
from pathlib import Path
import numpy as np


def paired_summary(full,base,key):
    seeds=sorted({r['seed'] for r in full.values()})
    differences=np.array([np.mean([full[s][key]-base[s][key] for s in full if full[s]['seed']==seed]) for seed in seeds])
    rng=np.random.default_rng(19731)
    bootstrap=differences[rng.integers(len(seeds),size=(50000,len(seeds)))].mean(axis=1)
    signs=np.array(list(itertools.product((-1,1),repeat=len(seeds))))
    null=(signs*differences).mean(axis=1)
    return dict(mean=float(differences.mean()),ci95=list(map(float,np.quantile(bootstrap,[.025,.975]))),
        ci98_333=list(map(float,np.quantile(bootstrap,[.05/6,1-.05/6]))),
        sign_flip_two_sided_p=float(np.mean(np.abs(null)>=abs(differences.mean())-1e-12)),
        positive_seed_blocks=int(np.count_nonzero(differences>0)),seed_blocks=len(seeds),differences=differences.tolist())


def confirm(path):
    config=json.loads((path/'config.json').read_text())
    if config['scope']!='independent frozen confirmation':raise ValueError('not a confirmation run')
    audit=json.loads((path/'verification.json').read_text())
    if audit['status']!='passed' or not audit['replayed_checkpoints']:raise ValueError('replay audit required')
    rows=[json.loads(l) for l in (path/'episodes.jsonl').read_text().splitlines()]
    assert set(r['seed'] for r in rows)==set(range(301,309))
    assert all(r['semantic_condition']=='aligned' and r['depth_sigma_m']==.01 and r['pose_noise_m']==0 for r in rows)
    full={r['scene']:r for r in rows if r['method']=='full_v2'}
    assert {(r['seed'],r['layout'],r.get('occluded_objects')) for r in full.values()}=={(seed,layout,opaque) for seed in range(301,309) for layout in ('rooms','warehouse') for opaque in (False,True)}
    comparisons={}
    for method in config['methods']:
        if method=='full_v2':continue
        base={r['scene']:r for r in rows if r['method']==method}
        assert set(full)==set(base) and len(base)==32
        comparisons[method]={key:paired_summary(full,base,key) for key in
            ('joint_auc_05cm','coverage_2d','f1_05cm','surface_error_mean_m')}
        comparisons[method]['joint_auc_relative_gain']=float(np.mean([r['joint_auc_05cm'] for r in full.values()])/np.mean([r['joint_auc_05cm'] for r in base.values()])-1)
    superiority=lambda method:comparisons[method]['joint_auc_05cm']['ci98_333'][0]>0
    coverage_ok=lambda method:comparisons[method]['coverage_2d']['mean']>=-.02
    claims=dict(joint_advantage_vs_coverage=superiority('coverage') and coverage_ok('coverage'),
        joint_advantage_vs_legacy_geometry=superiority('legacy_geometry') and coverage_ok('legacy_geometry'),
        semantic_increment=superiority('geometry_v2') and coverage_ok('geometry_v2'),
        zero_collisions_full=all(r['collisions']==0 for r in full.values()),
        lower_surface_error_vs_coverage=comparisons['coverage']['surface_error_mean_m']['ci95'][1]<0,
        no_real_robot_or_slam_estimator_claim=True)
    for method,name in [('no_route_v2','region_route_increment'),('no_camera_v2','camera_coverage_increment'),('no_hypotheses_v3','object_hypotheses_increment')]:
        if method in comparisons:claims[name]=comparisons[method]['joint_auc_05cm']['ci95'][0]>0
    claims['complete_semantic_scheme_supported']=all(claims.get(k,False) for k in
        ('joint_advantage_vs_coverage','joint_advantage_vs_legacy_geometry','semantic_increment','zero_collisions_full','region_route_increment','camera_coverage_increment','object_hypotheses_increment'))
    result=dict(status='complete',comparisons=comparisons,claims=claims,
        statistical_scope='8 paired seed blocks, two layouts and two occlusion conditions per seed; primary family Bonferroni 98.333% intervals',
        claim_scope='synthetic CPU virtual conditions only; no general SOTA/real robot/SLAM claim')
    (path/'confirmation.json').write_text(json.dumps(result,indent=2,ensure_ascii=False)+'\n')
    lines=['# 冻结独立确认结论','', '8 个新种子块、每块两种布局与两种遮挡条件。三个主比较采用 Bonferroni 校正的 98.333% 区间；下面的结论只适用于本次虚拟条件。','',
        '|完整方法 − 对照|联合 AUC 相对变化|Δ联合 AUC|98.333% 区间|最终覆盖差异/百分点|','|---|---:|---:|---|---:|']
    for method,data in comparisons.items():
        a=data['joint_auc_05cm'];lo,hi=a['ci98_333']
        lines.append(f'|{method}|{data["joint_auc_relative_gain"]*100:+.2f}%|{a["mean"]:+.5f}|[{lo:+.5f}, {hi:+.5f}]|{data["coverage_2d"]["mean"]*100:+.2f}|')
    lines+=['','## 逐项判据','']
    for key,value in claims.items():lines.append(f'- {key}: **{value}**')
    lines+=['','任何未通过项都必须保留。整套方案优于旧对照不等于语义模块有效；只要语义增量未通过，就不能写成“已证明完整语义方案优势”。',
        '', '已查看的确认集不能继续用于调参后重新声明独立确认。误差指标的采样支持随重建变化；F1 改善不能自动表述为表面精度提高。']
    (path/'confirmation.md').write_text('\n'.join(lines)+'\n')
    print(json.dumps(claims,ensure_ascii=False))


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('output',type=Path)
    confirm(parser.parse_args().output)
