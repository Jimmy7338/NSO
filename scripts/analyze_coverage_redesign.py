#!/usr/bin/env python3
"""Reconstruct all development evidence and report failures without selection."""
import csv
import hashlib
import json
from pathlib import Path
import sys
import zipfile
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.ndimage import label
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from utils.cpu_protocol import digest_json, geometry_hash, verify_protocol
from utils.grid_geometry import inflated_obstacles


def analyze(sources, output):
    output = Path(output); output.mkdir(parents=True, exist_ok=False)
    runs, all_rows = [], []
    old_config = json.loads((ROOT/'configs/cpu/formal_v1.json').read_text())
    verify_protocol(old_config, ROOT)
    targeted = json.loads((ROOT/'eval_results/cpu_targeted_scenes_v1_20260910/run_metadata.json').read_text())
    for name, expected in targeted['files_sha256'].items():
        assert hashlib.sha256((ROOT/name).read_bytes()).hexdigest() == expected, name
    for source in map(Path, sources):
        config = json.loads((source/'config.json').read_text())
        metadata = json.loads((source/'run_metadata.json').read_text())
        assert metadata['status'] == 'complete'
        assert digest_json(config) == metadata['config_sha256']
        with zipfile.ZipFile(source/'sources.zip') as archive:
            for name, expected in metadata['files_sha256'].items():
                assert hashlib.sha256(archive.read(name)).hexdigest() == expected, name
        records = [json.loads(s) for s in (source/'episodes.jsonl').read_text().splitlines()]
        expected = {(m['id'], method) for m in config['maps'] for method in config['methods']}
        assert len(records) == len(expected)
        assert {(r['map_id'], r['method']) for r in records} == expected
        pairs = {}
        for record in records:
            path = source/record['artifact_dir']
            with (path/'steps.csv').open() as f:
                steps = list(csv.DictReader(f))
            assert len(steps) == record['steps']+1
            env_config = config['environments'][record['suite']]
            with np.load(path/'observations.npz') as data:
                world, reference, poses = data['occupancy'], data['reachable'], data['poses']
                blocked = inflated_obstacles(world, env_config['robot_radius_m']/env_config['resolution_m'])
                components, _ = label(~blocked)
                np.testing.assert_array_equal(reference, components == components[tuple(poses[0,:2])])
                assert components[tuple(poses[0,:2])] != 0
                known = np.zeros_like(world, bool)
                curve = []
                for packed in data['visible_packed']:
                    known |= np.unpackbits(packed)[:world.size].reshape(world.shape).astype(bool)
                    curve.append(float(np.count_nonzero(known & reference)/reference.sum()))
                np.testing.assert_allclose(curve, [float(s['coverage_ratio']) for s in steps], atol=1e-12, rtol=0)
                np.testing.assert_array_equal(known, data['final_belief'] != -1)
                assert not np.any(blocked[tuple(poses[:,:2].T)])
                budget = env_config['max_steps']
                full = np.interp(np.arange(budget+1), np.arange(len(curve)), curve)
                assert abs(np.trapz(full)/budget-record['coverage_auc']) < 1e-12
                assert abs(curve[-1]-record['coverage_ratio']) < 1e-12
                assert geometry_hash(world) == record['map_hash']
                pair = (record['map_hash'], tuple(poses[0]), budget)
                if record['map_id'] in pairs:
                    assert pairs[record['map_id']] == pair
                pairs[record['map_id']] = pair
            all_rows.append(dict(run=source.name, **record))
        runs.append(dict(run=source.name, episodes=len(records),
                         summary=json.loads((source/'summary.json').read_text()),
                         loop_seconds=sum(r['wall_time_s'] for r in records),
                         peak_rss_mib=metadata['peak_rss_mib']))
    flat = []
    for run in runs:
        for suite, methods in run['summary']['means'].items():
            for method, metrics in methods.items():
                flat.append(dict(run=run['run'], suite=suite, method=method, **metrics))
    with (output/'all_variant_means.csv').open('w') as f:
        w = csv.DictWriter(f, fieldnames=list(flat[0])); w.writeheader(); w.writerows(flat)
    screen = {}
    for method in ('route32', 'route64'):
        checks = {}
        for suite in ('generic', 'targeted'):
            latest = runs[-1]['summary']['means'][suite]
            original = runs[0]['summary']['means'][suite]
            checks[suite] = (latest[method]['coverage_auc'] >= original['geometry']['coverage_auc']*1.05
                and latest[method]['coverage_ratio'] >= original['geometry']['coverage_ratio']-.02
                and latest[method]['coverage_auc'] >= max(original[m]['coverage_auc'] for m in
                                                          ('geometry','nearest','geometry_cost')))
        checks['safety_latency'] = all(r['collisions']==0 and r['decision_ms_p95']<200 for r in all_rows
                                      if r['run']==runs[-1]['run'] and r['method']==method)
        screen[method] = dict(checks=checks, passed=all(checks.values()))
    passed = any(v['passed'] for v in screen.values())
    report = dict(status='development_screen_passed_needs_holdout' if passed else 'development_screen_not_passed',
        development_screen=screen, episodes_audited=len(all_rows),
        source_archives_verified=True, prior_formal_sources_unchanged=True,
        prior_targeted_sources_unchanged=True, same_world_start_budget=True,
        coverage_and_auc_reconstructed=True, collisions=sum(r['collisions'] for r in all_rows),
        loop_seconds=sum(r['loop_seconds'] for r in runs),
        peak_rss_mib=max(r['peak_rss_mib'] for r in runs),
        evidence_scope='six previously seen development maps; repeated baselines are not independent evidence',
        heldout_validation_started=False, runs=runs)
    (output/'audit.json').write_text(json.dumps(report,indent=2,ensure_ascii=False)+'\n')
    lines = ['# 方案修订可行性检验：保留全部开发结果', '',
        f'共 {len(all_rows)} 个完整开发回合，使用同一批 6 张已见地图；不是独立正式测试。',
        '所有回合（包括早停和失败原型）均保留；重复基线不增加独立样本数。',
        ('结论：开发均值达到初筛条件，需另做独立验证；不能写正式正向结论。' if passed else
         '结论：尚未通过进入独立验证的开发筛选，不批准扩展训练或写正向结论。'), '',
        '| 轮次 | 场景组 | 条件 | AUC | 最终覆盖 |', '|---|---|---|---:|---:|']
    for row in flat:
        lines.append(f"| {row['run'].replace('cpu_coverage_v2_','').replace('_20260910','')} | {row['suite']} | {row['method']} | {row['coverage_auc']:.5f} | {row['coverage_ratio']:.2%} |")
    lines += ['', '## 如何解释', '',
        '- 第一轮检验代价、固定时限和安全观测；不支持只修这些环节便可使原组合有效。',
        '- 第二轮修复新原型的候选抽样错误；恢复了合理行为，但收益不稳定。',
        '- 第三轮加入朝向状态最短路及两个视点的边际覆盖；短时域误作可达性边界导致早停。',
        '- 第四轮加入长距离转移回退；这是修复第三轮自己的缺陷，不能当作对旧系统的独立优势。',
        '- 新原型仍未实现完整的区域访问序列，不能以这些结果断言完整分层方案已有效或必然无效。',
        '- 本轮没有增加语义训练；原语义／结构组合的负证据继续有效。', '',
        '## 复现', '',
        '每个运行目录含原始配置、逐动作观测、候选日志、源码压缩包及哈希。',
        '旧开发版本应从相应 sources.zip 解压到单独目录后运行归档配置；当前代码是第四轮版本。',
        '最终版本可使用 configs/cpu/coverage_v2_transfer_dev.json 复现，输出目录必须不存在。',
        '完整方案、文献与验收条件见 docs/PLANNER_REDESIGN_AND_FEASIBILITY.md。']
    (output/'results.md').write_text('\n'.join(lines)+'\n')
    fig, axes = plt.subplots(1,2,figsize=(11,4),layout='constrained')
    chosen = [(0,'geometry','Geometry'),(0,'combined','Old combined'),
              (0,'geometry_cost','Cost corrected'),(1,'projected_single','Safe viewpoints'),
              (3,'route32','Two-view + transfer')]
    for ax,suite in zip(axes,['generic','targeted']):
        values = [runs[i]['summary']['means'][suite][m]['coverage_auc'] for i,m,_ in chosen]
        ax.bar(np.arange(len(values)),values,color=['#557ea0','#b66360','#829e8c','#829e8c','#829e8c'])
        ax.set_xticks(np.arange(len(values)),[n for _,_,n in chosen],rotation=20,ha='right')
        ax.set_ylabel('Coverage AUC (higher is better)')
        ax.set_title(suite+' / 3 development maps')
        for i,v in enumerate(values): ax.text(i,v+.005,f'{v:.3f}',ha='center',fontsize=9)
        ax.set_ylim(0,max(values)*1.2)
    fig.suptitle('Development evidence only — held-out validation not performed')
    fig.savefig(output/'development_comparison.png',dpi=160);plt.close(fig)
    print(json.dumps({k:v for k,v in report.items() if k!='runs'},indent=2))


if __name__ == '__main__':
    import argparse
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--sources',nargs='+',required=True)
    p.add_argument('--output',required=True)
    a=p.parse_args();analyze(a.sources,a.output)
