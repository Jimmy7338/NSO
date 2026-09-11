#!/usr/bin/env python3
"""Combine verified development matrices without pretending repeats are samples."""
import argparse
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def report(main, ablation, output):
    records = {}
    runs = []
    repeat_pairs = 0
    for directory in (main, ablation):
        meta = json.loads((directory/'run_metadata.json').read_text())
        verification = json.loads((directory/'verification.json').read_text())
        assert meta['status'] == 'complete' and verification['status'] == 'passed'
        rows = [json.loads(line) for line in (directory/'episodes.jsonl').read_text().splitlines()]
        assert len(rows) == meta['expected_episodes']
        runs.append(dict(directory=str(directory.relative_to(ROOT)), episodes=len(rows),
                         replayed_checkpoints=verification['replayed_checkpoints']))
        for row in rows:
            key = row['scene'], row['method']
            if key in records:
                old, old_dir = records[key]
                for metric in ('coverage_auc','coverage_2d','joint_auc_05cm','f1_05cm','surface_error_mean_m'):
                    np.testing.assert_allclose(old[metric], row[metric], rtol=0, atol=1e-12)
                for filename, keys in [('maps.npz', ('poses','known_packed')),
                                       ('final_mesh.npz', ('vertices','triangles'))]:
                    with np.load(old_dir/old['artifact_dir']/filename) as a, np.load(directory/row['artifact_dir']/filename) as b:
                        for name in keys:
                            np.testing.assert_array_equal(a[name], b[name])
                repeat_pairs += 1
            else:
                records[key] = row, directory
    methods = ('coverage','full_v2','geometry_v4','objectness_v4','no_hypotheses_v4',
               'no_feedback_v4','no_guard_v4','full_v4')
    metrics = ('coverage_auc','coverage_2d','joint_auc_05cm','f1_05cm','surface_error_mean_m')
    averages = {method: {metric: float(np.mean([r[0][metric] for key, r in records.items()
                                               if key[1] == method])) for metric in metrics}
                for method in methods}
    full = averages['full_v4']
    contrasts = {method: dict(joint_auc_relative=full['joint_auc_05cm']/averages[method]['joint_auc_05cm']-1,
                             coverage_difference_pp=100*(full['coverage_2d']-averages[method]['coverage_2d']))
                 for method in methods if method != 'full_v4'}
    feedback = sum(row['guard_diagnostics']['feedback_rejections'] for (scene, method), (row, _) in records.items()
                   if method == 'full_v4')
    feedback_equivalent_pairs = 0
    for (scene, method), (row, directory) in records.items():
        if method != 'full_v4':continue
        other, other_directory = records[(scene, 'no_feedback_v4')]
        for filename, keys in [('maps.npz', ('poses','known_packed')),
                               ('final_mesh.npz', ('vertices','triangles'))]:
            with np.load(directory/row['artifact_dir']/filename) as a, np.load(other_directory/other['artifact_dir']/filename) as b:
                for name in keys:np.testing.assert_array_equal(a[name], b[name])
        feedback_equivalent_pairs += 1
    state = dict(scope='development only; two independent geometry seed blocks', runs=runs,
                 unique_scene_method_pairs=len(records), deterministic_repeat_pairs=repeat_pairs,
                 averages=averages, contrasts=contrasts, semantic_required=True,
                 fine_category_increment_positive_development=contrasts['objectness_v4']['joint_auc_relative'] > 0,
                 semantic_increment_positive_development=contrasts['geometry_v4']['joint_auc_relative'] > 0,
                 coverage_loss_within_two_pp_development=contrasts['coverage']['coverage_difference_pp'] >= -2,
                 object_hypothesis_net_gain_positive_development=contrasts['no_hypotheses_v4']['joint_auc_relative'] > 0,
                 feedback_rejections_full=feedback, independent_confirmation_run=False,
                 identical_full_without_feedback_pairs=feedback_equivalent_pairs,
                 complete_four_module_system_proven=False, mainstream_baseline_advantage_proven=False,
                 ready_for_independent_confirmation=False,
                 reference_definition='frozen composite-box triangle-area samples with reachable visibility filtering',
                 unique_physical_exterior_sensitivity_resolved=False)
    output.mkdir(parents=True, exist_ok=True)
    (output/'development_readiness.json').write_text(json.dumps(state, ensure_ascii=False, indent=2)+'\n')
    lines = ['# V4 覆盖保护与语义开发复核', '',
             f'两批有效运行共 {sum(r["episodes"] for r in runs)} 回合，{len(records)} 个不重复场景—方法组合；'
             f'{repeat_pairs} 个完整方法重复对照的轨迹、覆盖掩码、最终网格与核心指标完全一致。只有两个开发几何种子，未运行独立确认。', '',
             '|方法|联合 AUC@5cm|终点覆盖/%|终点 F1/%|平均误差/mm|',
             '|---|---:|---:|---:|---:|']
    for method in methods:
        row = averages[method]
        lines.append(f'|{method}|{row["joint_auc_05cm"]:.6f}|{100*row["coverage_2d"]:.3f}|'
                     f'{100*row["f1_05cm"]:.3f}|{1000*row["surface_error_mean_m"]:.3f}|')
    lines += ['', '|完整 V4 相对对照|联合 AUC 相对变化/%|终点覆盖差/百分点|',
              '|---|---:|---:|']
    for method, row in contrasts.items():
        lines.append(f'|{method}|{100*row["joint_auc_relative"]:+.3f}|{row["coverage_difference_pp"]:+.3f}|')
    lines += ['', f'完整 V4 的实测反馈拒绝触发 {feedback} 次。没有触发不能证明反馈模块有效；'
              '去反馈的同轨迹结果应保留，不能把覆盖保护的收益转记为反馈学习收益。',
              '', '去物体假设同时改变其候选生成和收益预测，评估的是该模块整体净作用；'
              '细类别信息的增量还需看同候选形状库的 geometry_v4 与二值 objectness_v4。',
              '', '两批均已复算终点网格、逐步覆盖、覆盖 AUC 与联合 AUC；'
              f'其中 {sum(r["replayed_checkpoints"] for r in runs)} 个检查点完成原始 RGB-D／雷达回放。'
              '语义仍为合成可见标签，离散导航使用真实格点位姿；不是完整 ANS、真实开放词汇或实车效果。',
              '', '历史三维参考使用冻结的组合箱体网格及可达视点可见性过滤。重叠共面可能影响表面采样权重；'
              '唯一实体外表面参考的敏感性复核仍需完成，新设施网格修正没有追溯改写本表。回放一致不等于排除该指标偏差。',
              '', '另有一批 40 回合因并行场景文件变化未通过严格源码冻结，单独标记 failed_source_mutation，'
              '不纳入本表或确认。这里的消融来自原始源码快照隔离重跑。完整记录未被替换。']
    (output/'results.md').write_text('\n'.join(lines)+'\n')
    print(json.dumps({k: v for k, v in state.items() if k not in ('averages','contrasts')}, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--main', type=Path, default=ROOT/'eval_results/joint_v4_feedback_development_20260910')
    parser.add_argument('--ablation', type=Path, default=ROOT/'eval_results/joint_v4_ablation_verified_20260910')
    parser.add_argument('--output', type=Path, default=ROOT/'eval_results/semantic_joint_v4_development_review_20260910')
    args = parser.parse_args()
    report(args.main, args.ablation, args.output)
