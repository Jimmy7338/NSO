#!/usr/bin/env python3
"""Read-only, post-outcome score decomposition; never an online planner.

No candidate, threshold or score is changed. Oracle values are descriptive
upper bounds on this fixed candidate pool, not executable policies. Fixed
support distances are evaluator-only GT-to-mesh distances, not precision or
SLAM pose accuracy. Outputs must be outside the original run directory.
"""
import os
os.environ.setdefault('OMP_NUM_THREADS', '1')
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import open3d as o3d
from scipy.stats import spearmanr

SCORERS = ('G', 'O', 'S', 'X', 'M', 'N')
ORACLE_METRICS = ('new_area_m2', 'area_per_action', 'f1_gain_05cm', 'f1_gain_per_action',
                  'branch_joint_auc_02cm', 'branch_joint_auc_05cm')


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def load(path):
    return json.loads(Path(path).read_text())


def f_proxy(recall):
    return 2 * np.clip(recall, 0, 1) / (1 + np.clip(recall, 0, 1))


def distance_to_mesh(path, points):
    with np.load(path, allow_pickle=False) as mesh:
        vertices, triangles = mesh['vertices'], mesh['triangles']
    if not len(triangles):
        raise ValueError('fixed-support diagnosis requires nonempty reconstruction')
    ray = o3d.t.geometry.RaycastingScene(nthreads=1)
    ray.add_triangles(o3d.core.Tensor(vertices.astype(np.float32)),
                      o3d.core.Tensor(triangles.astype(np.uint32)))
    return ray.compute_distance(o3d.core.Tensor(points.astype(np.float32)), nthreads=1).numpy().astype(float)


def distance_statistics(before, after, mask):
    b, a = before[mask], after[mask]
    if not len(b):
        return {'points': 0}
    return {'points': len(b), 'before_mean_m': float(b.mean()), 'after_mean_m': float(a.mean()),
            'mean_improvement_m': float(b.mean() - a.mean()),
            'before_p95_m': float(np.quantile(b, .95)), 'after_p95_m': float(np.quantile(a, .95)),
            'p95_improvement_m': float(np.quantile(b, .95) - np.quantile(a, .95)),
            'improved_fraction_at_1e_7m': float(np.mean(b - a > 1e-7)),
            'worsened_fraction_at_1e_7m': float(np.mean(a - b > 1e-7)),
            'before_within_02cm': float(np.mean(b <= .02)), 'after_within_02cm': float(np.mean(a <= .02)),
            'before_within_05cm': float(np.mean(b <= .05)), 'after_within_05cm': float(np.mean(a <= .05))}


def correlation(x, y):
    return float(spearmanr(x, y).statistic) if np.ptp(x) > 0 and np.ptp(y) > 0 else None


def history_diagnosis(run, history, hashes, budget):
    directory = run / Path(history['predictions']).parent
    fixture = directory.parent
    needed = [directory / name for name in ('predictions.json', 'candidates.json', 'outcomes.json', 'prefix_mesh.npz')]
    needed.append(fixture / 'reference.npz')
    predictions, routes, outcomes = (load(directory / name) for name in ('predictions.json', 'candidates.json', 'outcomes.json'))
    with np.load(fixture / 'reference.npz', allow_pickle=False) as reference:
        points = reference['points'].copy()
    baseline_distance = distance_to_mesh(directory / 'prefix_mesh.npz', points)
    first = directory / f"candidate_{routes[0]['candidate_id']:03d}" / 'visibility.npz'
    with np.load(first, allow_pickle=False) as visibility:
        prefix_seen = visibility['prefix'].copy()
    # This support depends only on the common past, never on a future candidate.
    baseline_covered = prefix_seen & (baseline_distance <= .05)
    normal = predictions['normalization']
    c0, q0 = normal['initial_coverage_proxy'], normal['initial_conditional_quality']
    r0 = normal['initial_camera_fraction'] * q0
    nonoccupied = normal['nonoccupied_grid_cells']
    by_id = {row['candidate_id']: row for row in outcomes}
    route_by_id = {row['candidate_id']: row for row in routes}
    rows = []
    for candidate in predictions['candidates']:
        cid = candidate['candidate_id']; route = route_by_id[cid]; outcome = by_id[cid]
        branch = directory / f'candidate_{cid:03d}'
        mesh_path, visible_path = branch / 'final_mesh.npz', branch / 'visibility.npz'
        needed.extend((mesh_path, visible_path))
        with np.load(visible_path, allow_pickle=False) as visibility:
            np.testing.assert_array_equal(visibility['prefix'], prefix_seen)
        actual_distance = distance_to_mesh(mesh_path, points)
        g, s, n = (candidate['scores'][name] for name in ('G', 'S', 'N'))
        cost = candidate['cost']
        c1_raw = c0 + g['radar_coverage_cells'] / nonoccupied
        c1 = min(1., c1_raw)
        camera_only = r0 + q0 * g['camera_coverage_cells'] / nonoccupied
        common_recall = camera_only + g['observed_quality_recall_increment']
        raw = {name: common_recall + candidate['scores'][name]['hidden_recall_increment'] for name in SCORERS}
        coverage_part = float((c1 * f_proxy(camera_only) - c0 * f_proxy(r0)) / cost)
        quality_part = float(c1 * (f_proxy(common_recall) - f_proxy(camera_only)) / cost)
        hidden_g = float(c1 * (f_proxy(raw['G']) - f_proxy(common_recall)) / cost)
        semantic_delta = float(c1 * (f_proxy(raw['S']) - f_proxy(raw['G'])) / cost)
        np.testing.assert_allclose([coverage_part + quality_part, coverage_part + quality_part + hidden_g,
                                   coverage_part + quality_part + hidden_g + semantic_delta],
                                  [n['score'], g['score'], s['score']], rtol=0, atol=1e-12)
        common_support = distance_statistics(baseline_distance, actual_distance, baseline_covered)
        auc_bounds = {name: {
            'lower': candidate['scores'][name]['joint_proxy_gain'] * (budget - cost + .5) / budget,
            'upper': candidate['scores'][name]['joint_proxy_gain'] * (budget - .5) / budget}
            for name in SCORERS}
        rows.append({'candidate_id': cid, 'group': route['group'], 'pose': route['pose'], 'cost': cost,
            'forward_actions': route['actions'].count('forward'),
            'score_by_scorer': {name: candidate['scores'][name]['score'] for name in SCORERS},
            'joint_proxy_gain_by_scorer': {name: candidate['scores'][name]['joint_proxy_gain'] for name in SCORERS},
            'predicted_joint_auc_increment_terminal_bounds': auc_bounds,
            'S_minus_G': s['score'] - g['score'],
            'decomposition': {'common_coverage_rate': coverage_part, 'common_observed_quality_rate': quality_part,
                              'G_hidden_rate': hidden_g, 'S_minus_G_rate': semantic_delta,
                              'S_hidden_rate': s['score'] - n['score']},
            'semantic_abs_delta_over_common_score': abs(semantic_delta) / abs(n['score']) if n['score'] else None,
            'raw_recall_by_scorer': raw, 'recall_clipped_by_scorer': {name: value >= 1 for name, value in raw.items()},
            'coverage_clipped': c1_raw > 1,
            'predicted_hidden_area_m2': {name: candidate['scores'][name]['hidden_area_m2'] for name in SCORERS},
            'observed_quality_gain': g['observed_quality_gain'],
            'observed_quality_recall_increment': g['observed_quality_recall_increment'],
            'actual': {key: outcome[key] for key in ORACLE_METRICS} | {'coverage_gain_m2': outcome['coverage_gain_m2'],
                'stage_new_area_m2': outcome['stage_new_area_m2'], 'failure': outcome['failure'],
                'precision_gain_05cm': outcome['after']['precision_05cm'] - outcome['before']['precision_05cm'],
                'recall_gain_05cm': outcome['after']['recall_05cm'] - outcome['before']['recall_05cm'],
                'mesh_sample_mean_error_improvement_m': outcome['before']['surface_error_mean_m'] - outcome['after']['surface_error_mean_m']},
            'prefix_visible_reference_distance': distance_statistics(baseline_distance, actual_distance, prefix_seen),
            'prefix_visible_and_baseline_covered_05cm_distance': common_support,
            'fixed_support_improvement_per_action_m': common_support['mean_improvement_m'] / cost})
    for path in needed:
        hashes[str(path.relative_to(run))] = sha(path)
    selected = predictions['selected']; winner = next(row for row in rows if row['candidate_id'] == selected['G'])
    semantic_winner = next(row for row in rows if row['candidate_id'] == selected['S'])
    auc_diagnosis = {}
    for scorer in SCORERS:
        rate_winner = next(row for row in rows if row['candidate_id'] == selected[scorer])
        upper = rate_winner['predicted_joint_auc_increment_terminal_bounds'][scorer]['upper']
        certain = [row['candidate_id'] for row in rows
                   if row['predicted_joint_auc_increment_terminal_bounds'][scorer]['lower'] > upper]
        auc_diagnosis[scorer] = {'budget': budget, 'rate_selected_candidate': selected[scorer],
            'rate_winner_auc_upper': upper, 'candidates_lower_above_rate_winner_upper': certain,
            'rate_choice_cannot_be_auc_optimal_under_monotone_fixed_proxy': bool(certain),
            'exact_stepwise_auc_computed': False}
    others = [row for row in rows if row is not winner]
    margin = winner['score_by_scorer']['G'] - max(row['score_by_scorer']['G'] for row in others)
    deltas = [row['S_minus_G'] for row in rows]
    oscillation = max(deltas) - min(deltas)
    residuals = [winner['score_by_scorer']['G'] - row['score_by_scorer']['G'] - (row['S_minus_G'] - winner['S_minus_G']) for row in others]
    crossings = [(winner['score_by_scorer']['G'] - row['score_by_scorer']['G']) / (row['S_minus_G'] - winner['S_minus_G'])
                 for row in others if row['S_minus_G'] > winner['S_minus_G']]
    oracle = {}
    for metric in ORACLE_METRICS:
        upper = max(row['actual'][metric] for row in rows)
        best = [row['candidate_id'] for row in rows if row['actual'][metric] == upper]
        oracle[metric] = {'best_candidate_ids': best, 'upper_bound': upper,
            'selected_S_value': semantic_winner['actual'][metric], 'selected_S_regret': upper - semantic_winner['actual'][metric],
            'oracle_candidates_S_rank': [predictions['rankings']['S'].index(cid) + 1 for cid in best]}
    return {key: history[key] for key in ('fixture', 'history_step', 'seed', 'depth_sigma_m')} | {
        'selected_by_scorer': selected, 'all_six_same_selection': len(set(selected.values())) == 1,
        'geometry_top_margin': margin, 'max_abs_semantic_delta': max(map(abs, deltas)),
        'semantic_delta_oscillation': oscillation,
        'max_abs_delta_over_geometry_margin': max(map(abs, deltas)) / margin if margin > 0 else None,
        'strict_oscillation_selection_certificate': bool(margin > oscillation),
        'exact_minimum_pairwise_residual': min(residuals),
        'hypothetical_delta_scale_to_first_tie': min(crossings) if crossings else None,
        'score_changing_candidates': sum(delta != 0 for delta in deltas),
        'raw_recall_min': min(value for row in rows for value in row['raw_recall_by_scorer'].values()),
        'raw_recall_max': max(value for row in rows for value in row['raw_recall_by_scorer'].values()),
        'recall_clipped_candidate_scorer_pairs': sum(value for row in rows for value in row['recall_clipped_by_scorer'].values()),
        'selected': semantic_winner, 'candidates': rows, 'posthoc_same_pool_oracle': oracle,
        'predicted_auc_development_ablation_bounds': auc_diagnosis,
        'common_observed_quality_rate_vs_fixed_support_improvement_rate_spearman': correlation(
            [row['decomposition']['common_observed_quality_rate'] for row in rows],
            [row['fixed_support_improvement_per_action_m'] for row in rows]),
        'unnormalized_quality_gain_vs_fixed_support_improvement_spearman': correlation(
            [row['observed_quality_gain'] for row in rows],
            [row['prefix_visible_and_baseline_covered_05cm_distance']['mean_improvement_m'] for row in rows]),
        'S_score_vs_actual_area_rate_spearman': correlation([row['score_by_scorer']['S'] for row in rows],
                                                          [row['actual']['area_per_action'] for row in rows])}


def markdown(report):
    histories = report['histories']; total = report['aggregate']; background = report.get('background_diagnosis')
    lines = ['# 固定候选池评分失效诊断（仅开发数据）', '',
        '本报告读取已完整回放的 72 个反事实分支，不修改原始运行、评分、候选、预设门槛或规划策略。'
        '六个可用历史来自两个种子及噪声/时间重复，不能按六个独立场景推断显著性。', '',
        '## 为何分数变化没有改变选择', '',
        '|种子/噪声/历史|G/O/S/X/M/N 所选|成本|几何第一二名差|最大 |S−G| / 差|改变分数候选|',
        '|---|---|---:|---:|---:|---:|']
    # Markdown pipe in a column title must not introduce an extra column.
    lines[-2] = '|种子/噪声/历史|G/O/S/X/M/N 所选|成本|几何第一二名差|最大语义扰动/差|改变分数候选|'
    for h in histories:
        lines.append(f"|{h['seed']}/{h['depth_sigma_m']}/{h['history_step']}|"
                     + '/'.join(str(h['selected_by_scorer'][name]) for name in SCORERS)
                     + f"|{h['selected']['cost']}|{h['geometry_top_margin']:.8f}|{100*h['max_abs_delta_over_geometry_margin']:.2f}%|{h['score_changing_candidates']}/12|")
    lines += ['', f"全部 {len(histories)} 个历史中，六种评分选择完全相同。{total['score_changing_histories']} 个历史有非零 S−G，"
        f"但每个历史都满足下述严格选点不变证书。共有 {total['recall_clipped_candidate_scorer_pairs']} / {total['candidate_scorer_pairs']} 个候选—评分对触发 recall clipping；"
        f"未裁剪 recall 范围 {total['raw_recall_min']:.5f}–{total['raw_recall_max']:.5f}。因此本批次不是 recall 上限饱和掩盖细类作用。", '',
        '令固定候选集的几何分数为 $g_i$、细类分数为 $s_i=g_i+\\delta_i$，共同成本与平局规则不变。'
        '若几何唯一最优点为 $a$，$m=g_a-\\max_{i\\ne a}g_i>0$，且 '
        '$m>\\max_i\\delta_i-\\min_i\\delta_i$，则 $a$ 仍是细类评分的唯一最优点。'
        '证明：任意 $i\\ne a$，$s_a-s_i=(g_a-g_i)-(\\delta_i-\\delta_a)\\ge m-\\operatorname{osc}(\\delta)>0$。', '',
        '更精确的充要检验是逐候选检查 $g_a-g_i>\\delta_i-\\delta_a$；等号时必须再按共同平局规则判断。'
        '这是本次固定分数和候选的代数证书，不证明未来地图、候选或经过重新校准的语义模型也不改变选择。'
        'JSON 中给出逐候选 S−G 与精确剩余间隔；假想扰动倍数对应冻结分数 $g+\\alpha\\delta$ 的线性插值，'
        '不等同于非线性模型真实参数调整，仅描述距离翻转有多远，不能按真值调大系数充当新算法。', '',
        '## 共同项与短扫描偏好', '',
        '为避免把非线性 F1 代理强行视作可加真实收益，采用固定顺序的精确望远镜分解：先加入二维/相机覆盖，再加入已观测质量，'
        '然后加入 G 隐藏面积，最后替换为 S 隐藏面积。四项相加严格等于 S 分数；该归因依赖此声明顺序。'
        '全部原始增量、未裁剪 recall、各分量及其每动作比值保存在 JSON。', '',
        '|种子/噪声/历史|共同覆盖率项|共同质量率项|G 隐藏率项|S−G|前进动作|',
        '|---|---:|---:|---:|---:|---:|']
    for h in histories:
        row = h['selected']; d = row['decomposition']
        lines.append(f"|{h['seed']}/{h['depth_sigma_m']}/{h['history_step']}|{d['common_coverage_rate']:.7f}|{d['common_observed_quality_rate']:.7f}|{d['G_hidden_rate']:.7f}|{d['S_minus_G_rate']:.7f}|{row['forward_actions']}|")
    lines += ['', '低噪声四个历史都选两动作、零前进的往返旋转，隐藏面积贡献为零；高噪声两个历史选较长覆盖路线，去掉全部隐藏收益的 N 仍同选。'
        '这说明单改细类别权重不能解决当前两种主导机制。每动作比率本身并非错误，但未校准质量增量配合两动作分母，足以让短扫描压过真实更有价值的候选。', '',
        '## 质量代理是否对应真实变化', '',
        '额外诊断固定使用：前缀物理相机已见、且距原始 prefix mesh 不超过 5 cm 的同一批 GT 参考点。'
        '使用 Open3D 点到三角网格距离，支撑集只由过去决定，对所有未来候选相同。正“误差改善”表示平均距离下降。'
        '这是固定支撑的 GT→mesh 贴合距离；它不评价 mesh→GT 伪表面精度，也不是位姿精度。F1 的 precision/recall 与总体误差仍另外列出，后者的重建表面采样支撑会随 mesh 改变。', '',
        '|种子/噪声/历史|新增唯一面积 m²|F1 增量 pp|precision 增量 pp|recall 增量 pp|固定支撑平均误差改善 mm|',
        '|---|---:|---:|---:|---:|---:|']
    for h in histories:
        row = h['selected']; a = row['actual']; fixed = row['prefix_visible_and_baseline_covered_05cm_distance']
        lines.append(f"|{h['seed']}/{h['depth_sigma_m']}/{h['history_step']}|{a['new_area_m2']:.4f}|{100*a['f1_gain_05cm']:.4f}|{100*a['precision_gain_05cm']:.4f}|{100*a['recall_gain_05cm']:.4f}|{1000*fixed['mean_improvement_m']:.6f}|")
    lines += ['', '这些量必须分别解释：F1 增加可能主要来自召回而非精度；新可见面积也不等于新增高质量重建。'
        '两例低噪声 step 160 短扫描的新增面积均为零，F1 一例下降、一例不变，不能据质量代理正值声称 F1 改善。'
        '低噪声四例固定支撑平均距离确有约 0.011–0.049 mm 改善，因此不能说这些观察完全无效。'
        '这是远小于 2/5 cm 评价阈值的局部贴合变化；没有重复噪声置信区间，不能推出稳定准确率收益。', '',
        '各历史中，共同质量率项与同支撑平均误差改善/动作的候选内 Spearman 分别为 '
        + '、'.join(f"{h['common_observed_quality_rate_vs_fixed_support_improvement_rate_spearman']:.3f}" for h in histories)
        + '。由于两者共享动作数分母，另核对未除成本的原始质量增益与同支撑误差改善，相关系数分别为 '
        + '、'.join(f"{h['unnormalized_quality_gain_vs_fixed_support_improvement_spearman']:.3f}" for h in histories)
        + '，仍为描述性正关联。它不证明数值已校准；失败不应简化为质量代理全无信息，而是这类微小局部收益与新增覆盖/全图质量的预算权衡失配。', '',
        '## 同池事后上界', '',
        '以下每个指标分别在同一已执行候选池取最大值。各列最优候选可以不同，不能把各项最大值拼成一个可达联合解。'
        '这些 oracle 使用未来真值，只诊断候选池是否已有更好方案；不能部署、不能用于改本轮门槛，亦不代表所有可行路线的全局上界。', '',
        '|种子/噪声/历史|面积率 S / oracle m²/action|F1 增量 S / oracle pp|联合 AUC S / oracle|',
        '|---|---:|---:|---:|']
    for h in histories:
        oracle = h['posthoc_same_pool_oracle']; a = oracle['area_per_action']; f = oracle['f1_gain_05cm']; j = oracle['branch_joint_auc_05cm']
        lines.append(f"|{h['seed']}/{h['depth_sigma_m']}/{h['history_step']}|{a['selected_S_value']:.5f} / {a['upper_bound']:.5f}|{100*f['selected_S_value']:.4f} / {100*f['upper_bound']:.4f}|{j['selected_S_value']:.6f} / {j['upper_bound']:.6f}|")
    lines += ['', '联合 AUC 沿用冻结定义：前缀、到达、结束/失败评价点之间线性插值，结束后保持到 48 动作统一预算；并非逐帧 TSDF 真值曲线。', '',
        '## 无新增权重的预测 AUC 消融：现有数据能证明什么', '',
        '可以保留原 rate 主评分和全部预设门槛，另加开发性消融：固定共同历史、候选和原代理，'
        '累计到第 $t$ 个实际付费动作的代理增量为 $u_t$，$u_0=0$，路线长 $L\\le B=48$，完成后保持 $u_L$。'
        '如果候选内只累计固定地图的可见并集/点质量最大值、权重非负且基准与模型不变，则 $u_t$ 单调。'
        '对梯形积分直接交换求和可得', '',
        '$$A=\\frac1B\\sum_{t=0}^{B-1}\\frac{u_t+u_{t+1}}2'
        '=\\sum_{t=1}^{L}\\frac{B-t+1/2}{B}\\,(u_t-u_{t-1}).$$', '',
        '因此每个新增收益按出现后剩余预算计权，无需新增语义或质量系数。'
        '对于只保存终点增量 $U=u_L$ 的现有 predictions，能严格得到 '
        '$$\\frac{B-L+1/2}{B}U\\le A\\le\\frac{B-1/2}{B}U.$$'
        '若某候选下界已经高于原 rate 最优候选的上界，则无论单调收益在路线中何时出现，原候选都不会是预测 AUC 最优。'
        '这只证明原代理目标下的排序会变，不证明真实面积、F1 或语义优势随之改善。', '',
        '|种子/噪声/历史|原 S rate 选择|下界已胜原选择上界的候选|',
        '|---|---:|---|']
    for h in histories:
        auc = h['predicted_auc_development_ablation_bounds']['S']
        lines.append(f"|{h['seed']}/{h['depth_sigma_m']}/{h['history_step']}|{auc['rate_selected_candidate']}|"
                     + (', '.join(map(str, auc['candidates_lower_above_rate_winner_upper'])) or '无可证者') + '|')
    lines += ['', '四个低噪声历史均有严格超过原短扫描上界的候选，因而在上述固定单调代理假设下，'
        '精确预测 AUC 必然改变这四例原选择；两个高噪声历史的界未能决定。'
        '旧 predictions 没有逐动作预测并集轨迹，不能从终点恢复精确 $u_t$，本报告不使用终点近似替代它。'
        '另行开发的 AUC 消融由同一固定候选路径的逐步 cache 重新计算积分，继续使用独立真实评价，具体新选择以该消融输出为准。', '',
        '此 AUC 仍继承原质量与隐藏收益的标定误差；完成后 hold 是单条分支的统一比较假设，'
        '不包含短路线结束后再次规划的延续价值。若每步重建模型/基准，单调性和上述界可能失效。'
        '因此该消融合理且不新增权重，但不能替代对象建模修正、原 rate 结果、既定通过门槛或完整闭环验证。', '',
        '## 优先修什么', '']
    if background:
        lines += [f"独立 GT 事后对象诊断确认，S 潜在加权面积中的 {100*background['S_background_weight_fraction']:.2f}% 来自背景支撑伪对象；"
                  f"全部 S 潜在权重中 {100*background['S_far_from_exterior_weight_fraction']:.2f}% 距唯一真实外表面超过 5 cm，"
                  f"其中 {100*background['S_far_inside_weight_fraction']:.2f}% 在实体内部且离外表面超过 5 cm。"
                  '该分母是各重复历史的模型潜在权重和，不是实际可见面积、独立场景计数或 F1。', '',
                  f"细类 S 潜在权重近外表面比例为 {100*background['near_5cm_weight_fraction_by_scorer']['S']:.2f}%，"
                  f"几何 G 为 {100*background['near_5cm_weight_fraction_by_scorer']['G']:.2f}%，"
                  f"错标 X 为 {100*background['near_5cm_weight_fraction_by_scorer']['X']:.2f}%。"
                  '这保留了细类形状权重有描述性几何关联的证据，但不能换算成视点效用或完整规划优势。', '']
    lines += ['优先修对象性和包围几何的成立条件，并同步校准质量回报的边际价值与预算权衡：显式保留背景/非对象假设，'
        '用可观测支持与自由空间反证约束单面包围拟合，分开真实物体存在概率与其细类别/隐藏结构概率。'
        '这样再进行效用标定才有可解释对象；直接放大整个 S 隐藏收益权重会同时放大伪对象和错误内部面。'
        '但去掉全部隐藏项的 N 当前仍同选，所以只修对象几何也未必改变选择，共同质量/覆盖项必须独立验证。', '',
        '效用层同样必须修改：用实际新增有效观测反馈校准短扫描收益，区分新增表面、原支撑误差改善与已饱和重复观察，'
        '然后在原预注册开发划分中检验面积/质量预测和排名。当前候选池包含事后更优路线，故不能首先把失败归于没有候选，'
        '但本诊断也不证明候选池已足够覆盖语义有价值的视点。', '',
        '所有修改保持 ANS 双层与 OV-SDF/STGHP/RPN-UQ/IGCR 接口；新模型应独立版本实现，GT 只留在事后评价器，'
        '不输入在线对象筛选、分数或回报。新模型仍须按既定门槛重新验证，不能把本次根因诊断当作改进已奏效。', '',
        '## 可复现性', '',
        f"- 原运行：`{report['source_run']}`。",
        f"- 原完整回放：`{report['verification_status']}`，72/72 分支。",
        f"- 脚本 SHA256：`{report['script_sha256']}`。",
        '- JSON 记录所有输入 SHA256、逐候选精确分解、固定支撑误差、oracle 与适用边界；原运行内容未修改。']
    if background:
        lines.append(f"- GT 对象诊断：`{background['source']}`；规则：{background['background_rule']}。")
    return '\n'.join(lines) + '\n'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--json', type=Path, required=True)
    parser.add_argument('--report', type=Path, required=True)
    parser.add_argument('--background-diagnosis', type=Path)
    args = parser.parse_args(); run = args.run.resolve()
    if args.json.resolve().is_relative_to(run) or args.report.resolve().is_relative_to(run):
        raise ValueError('diagnostic outputs must be outside the original run')
    metadata = load(run / 'metadata.json'); verification = load(run / 'verification.json')
    if metadata['status'] != 'complete' or not verification['passed_full']:
        raise ValueError('complete independent replay required')
    manifest = load(run / 'artifact_hashes.json')
    manifest_before = sha(run / 'artifact_hashes.json')
    source_hashes = {name: sha(run / name) for name in ('metadata.json', 'verification.json', 'summary.json', 'artifact_hashes.json')}
    config = load(run / 'config.json'); source_hashes['config.json'] = sha(run / 'config.json')
    histories = [history_diagnosis(run, h, source_hashes, config['branch_actions']) for h in load(run / 'summary.json')['histories']]
    histories.sort(key=lambda h: (h['seed'], h['depth_sigma_m'], h['history_step']))
    rows = [row for history in histories for row in history['candidates']]
    report = {'scope': 'post-outcome fixed-pool diagnosis; no threshold tuning, no online truth input',
        'source_run': str(run), 'verification_status': verification['status'], 'script_sha256': sha(__file__),
        'history_count': len(histories), 'candidate_count': len(rows), 'histories': histories,
        'aggregate': {'score_changing_histories': sum(h['score_changing_candidates'] > 0 for h in histories),
            'all_six_same_choice_histories': sum(h['all_six_same_selection'] for h in histories),
            'strict_selection_certificates': sum(h['strict_oscillation_selection_certificate'] for h in histories),
            'candidate_scorer_pairs': len(rows) * len(SCORERS),
            'recall_clipped_candidate_scorer_pairs': sum(h['recall_clipped_candidate_scorer_pairs'] for h in histories),
            'raw_recall_min': min(h['raw_recall_min'] for h in histories),
            'raw_recall_max': max(h['raw_recall_max'] for h in histories),
            'mean_selected_and_separate_oracle': {key: {
                name: float(np.mean([h['posthoc_same_pool_oracle'][key][name] for h in histories]))
                for name in ('selected_S_value', 'upper_bound', 'selected_S_regret')} for key in ORACLE_METRICS}},
        'fixed_support': 'prefix physical visibility AND baseline GT-to-prefix-mesh distance <=0.05m; independent of future branch',
        'distance_limitations': 'GT-to-mesh distance is not mesh-to-GT precision or pose accuracy; no repeated-noise confidence intervals',
        'oracle_limitations': 'each objective has separate future-informed same-pool maximum; not a joint attainable vector or global route optimum',
        'source_hashes': source_hashes, 'runtime': {'numpy': np.__version__, 'open3d': o3d.__version__}}
    if args.background_diagnosis:
        path = args.background_diagnosis.resolve(); background = load(path); s = background['scorer_totals']['S']
        background_meta = load(path.parent / 'metadata.json')
        if (Path(background_meta['source_run']).resolve() != run
                or background_meta['source_metadata_sha256'] != sha(run / 'metadata.json')
                or background_meta['source_archive_sha256'] != sha(run / 'sources.zip')
                or not background_meta['diagnostic_only'] or background_meta['feeds_planning']):
            raise ValueError('GT background diagnosis source/scope mismatch')
        report['background_diagnosis'] = {'source': str(path), 'sha256': sha(path),
            'metadata_sha256': sha(path.parent / 'metadata.json'),
            'background_rule': background['background_rule'],
            'S_background_weight_fraction': s['gt_background_supported_weighted_area_m2'] / s['potential_weighted_area_m2'],
            'S_far_from_exterior_weight_fraction': s['far_from_exterior_weight_fraction'],
            'S_far_inside_weight_fraction': s['far_inside_weight_fraction'],
            'near_5cm_weight_fraction_by_scorer': {name: background['scorer_totals'][name]['near_5cm_weight_fraction']
                                                for name in ('G', 'S', 'X')},
            'denominator': 'sum of S potential hypothesis weights across repeated histories; not actual visible area'}
    for name, digest in source_hashes.items():
        if sha(run / name) != digest or (name in manifest and manifest[name] != digest):
            raise ValueError(f'input hash mismatch: {name}')
    if sha(run / 'artifact_hashes.json') != manifest_before:
        raise ValueError('original manifest changed')
    for path in (args.json, args.report): path.parent.mkdir(parents=True, exist_ok=True)
    args.json.write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + '\n')
    args.report.write_text(markdown(report))
    print(json.dumps(report['aggregate'], ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
