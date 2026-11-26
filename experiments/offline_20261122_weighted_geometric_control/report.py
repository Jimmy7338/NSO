#!/usr/bin/env python3
"""Summarize registered paired outcomes without replacing missing clocks."""

import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt

from study import ROOT, WEIGHTS, PROTOCOL, freeze_path, read, sha


MEASUREMENT = ROOT / 'measurement/results/holdout'
DECISION = ROOT / 'decision'
RESULTS = ROOT / 'results'
METRICS = ('J', 'C_free', 'F1_scene_05cm')
DISPLAY = {'G': 'Original G', 'G_route': 'G* (route fit)', 'G_perf': 'G* (map fit)',
           'S': 'Correct category S', 'X': 'Swapped category X', 'S_perf': 'S with G* map weights'}
STYLES = (('#405167', 'o', ':'), ('#6E5687', '^', '--'), ('#0072B2', 's', '-'),
          ('#D55E00', 'D', '-'), ('#73879A', 'X', '-.'), ('#009E73', 'v', '--'))


def measured(row):
    return row.get('status') == 'completed_execution'


def pair_rows(episodes, protocol):
    data = []
    for layout in protocol['heldout_layouts']:
        comparisons = protocol['evaluation']['primary_performance_contrasts'] + protocol['evaluation']['secondary_performance_contrasts']
        for contrast in comparisons:
            a_name, b_name = contrast.split(' - ')
            left, right = (episodes[(layout, arm)] for arm in (a_name, b_name))
            common = {'layout': layout, 'contrast': contrast, 'arm_left': a_name, 'arm_right': b_name,
                      'left_status': left['status'], 'right_status': right['status']}
            if not all(map(measured, (left, right))):
                data.append(common | {'status': 'terminal_unavailable'})
                continue
            metrics = {f'delta_{name}_pp': 100 * (left[name] - right[name]) for name in METRICS}
            data.append(common | metrics | {'status': 'measured', 'left_paid': left['actual_paid'],
                                          'right_paid': right['actual_paid'], 'left_returns': left['returned'],
                                          'right_returns': right['returned'], 'left_collisions': left['collisions'],
                                          'right_collisions': right['collisions'],
                                          'maintains_return': left['returned'] or not right['returned'],
                                          'no_excess_collisions': left['collisions'] <= right['collisions'],
                                          'within_cap': max(left['actual_paid'], right['actual_paid']) <= protocol['native_budget']})
    return data


def passes_safety(row):
    return row['status'] == 'measured' and all(row[key] for key in ('within_cap','maintains_return','no_excess_collisions'))


def adjudicate(pairs, decision, episodes):
    # Compare in fractions. Report percentage-point equivalents separately.
    map_pairs = [row for row in pairs if row['contrast'] == 'G_perf - S']
    cate_pairs = [row for row in pairs if row['contrast'] == 'S_perf - G_perf']
    assert len(map_pairs) == len(cate_pairs) == 4

    def diff(row, name):
        layout = row['layout']
        return episodes[layout, row['arm_left']][name] - episodes[layout, row['arm_right']][name]

    map_ok = [row['layout'] for row in map_pairs if passes_safety(row) and diff(row, 'J') >= -.005
              and diff(row, 'C_free') >= -.010 and diff(row, 'F1_scene_05cm') >= -.010]
    category_good = [row['layout'] for row in cate_pairs if passes_safety(row) and diff(row, 'J') > .005]
    return {'independent_buildings': 4, 'route_replication_gate': decision['route_replication_engineering_gate'],
            'route_multi_candidate_macro': decision['macro_G_route_multi_route'],
            'route_all_local_macro': decision['macro_G_route_all_local'],
            'map_noninferiority_gate': len(map_ok) == 4, 'map_noninferiority_eligible_buildings': map_ok,
            'category_increment_gate': len(category_good) >= 3 and all(
                passes_safety(row) and diff(row, 'J') >= -.005 for row in cate_pairs),
            'category_increment_above_margin_buildings': category_good}


def render(samples, protocol):
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':9,'pdf.fonttype':42})
    fig, axes = plt.subplots(2, 2, figsize=(8.65, 5.65), sharex=True, sharey=True)
    fig.subplots_adjust(left=.095, right=.975, bottom=.20, top=.90, wspace=.18, hspace=.33)
    legend = []
    registered = set(protocol['capture_paid'])
    for ax, layout in zip(axes.flat, protocol['heldout_layouts'], strict=True):
        ax.set_title(layout.removeprefix('paired_').replace('_', ' ').title(), loc='left', weight='semibold')
        ax.grid(axis='y', alpha=.24)
        for arm, (color, marker, style) in zip(DISPLAY, STYLES, strict=True):
            observations = {r['actual_paid']: r for r in samples if r['layout'] == layout and r['arm'] == arm
                            and r['status'] == 'captured' and r['checkpoint'] != 'terminal'}
            assert set(observations) <= registered
            x = protocol['capture_paid']
            y = [100 * observations[p]['J'] if p in observations else float('nan') for p in x]
            line, = ax.plot(x, y, color=color, marker=marker, linestyle=style, markersize=4.0,
                            linewidth=1.15, alpha=.88, label=DISPLAY[arm])
            if ax is axes[0,0]:
                legend.append(line)
        ax.set(xticks=protocol['capture_paid'], xlim=(-3, 195))
        ax.spines[['right','top']].set_visible(False)
    fig.supxlabel('Charged actions including return', y=.125, fontsize=9)
    fig.supylabel('Joint map quality J (%)', x=.025, fontsize=9)
    fig.legend(handles=legend, loc='lower center', bbox_to_anchor=(.5,.033), ncol=3, frameon=False)
    fig.savefig(RESULTS / 'equal_budget_paired_quality.pdf', bbox_inches='tight')
    fig.savefig(RESULTS / 'equal_budget_paired_quality.png', dpi=300, bbox_inches='tight')
    plt.close(fig)


def manuscript(gates, pairs, weights, episodes, decision):
    def values(contrast):
        numbers = [row.get('delta_J_pp') for row in pairs if row['contrast'] == contrast]
        return '、'.join('未取得' if n is None else ('0.000' if abs(n) < .0005 else f'{n:+.3f}') for n in numbers)

    def pct(rate):
        return '未取得' if rate is None else f'{rate:.1%}'

    route_fit, fitting = weights['decision'], weights['performance']
    def decision_rows(filename):
        path = DECISION / filename
        assert sha(path) == decision['output_files_sha256'][filename]
        with path.open(newline='') as stream:
            return list(csv.DictReader(stream))

    by_building = [row for row in decision_rows('heldout_imitation_by_building.csv')
                   if row['compared_to_S'] == 'G_route']
    multi = '、'.join(f"{r['multi_route_match']}/{r['multi_route_count']}" if int(r['multi_route_count']) else '无可比较状态'
                     for r in by_building)
    actual = [row for row in decision_rows('heldout_actual_first_differences.csv')
              if row['arm_left'] == 'G_route' and row['arm_right'] == 'S']
    def first_action(row):
        if row.get('first_action_diff'):
            return f"第 {row['first_action_diff']} 步"
        if row.get('exact_through_both_terminals') == 'True':
            return '全程一致'
        return '未在共有动作中观测到'

    first = '、'.join(map(first_action, actual))
    registered_endpoints = [row for row in episodes.values() if measured(row)]
    native_endpoints = [row for row in registered_endpoints if row['arm'] == row['physical_source_arm']]
    returned = sum(row['returned'] for row in native_endpoints)
    collisions = sum(row['collisions'] for row in native_endpoints)
    return f'''# 论文结果段落

我们在两栋开发建筑上进行 20 轮闭环仿真，并分别按局部路线复现率和终点地图质量选择几何权重。两组权重在四栋新建筑运行前冻结。路线拟合的未知可见面积系数和额外动作代价分别为 {route_fit['unknown_weight']:.2f} 和 {route_fit['action_cost']:.2f}，地图拟合系数分别为 {fitting['unknown_weight']:.2f} 和 {fitting['action_cost']:.2f}。每轮上限为 192 个计费动作，包含返航。所有策略共用中性全局目标规则、局部候选生成、观测与融合。类别对照通过局部预计表面积引入结构先验，类别由仿真提供。

沿原 S 在四栋留出建筑中实际获得的状态，我们从同一合法候选池重算各策略的当前路线。路线拟合 G* 在全部局部选路中的建筑等权匹配率为 {pct(gates['route_all_local_macro'])}。限于至少有两条合法路线的状态，链式 17、环式 18、链式 19、环式 20 的匹配数依次为 {multi}，建筑等权匹配率为 {pct(gates['route_multi_candidate_macro'])}，{'达到' if gates['route_replication_gate'] else '未达到'}预注册判据。在各自闭环执行时，两者的首次动作分歧按同样顺序记录为{first}。

按同样的建筑顺序，地图拟合 G* 相对原 S 的终点联合质量 $J$（正确自由空间覆盖率与整场景 F1@5 cm 的乘积）变化为 {values('G_perf - S')} 个百分点。{len(gates['map_noninferiority_eligible_buildings'])} 栋建筑满足预设容差，{'达到' if gates['map_noninferiority_gate'] else '未达到'}四栋均需通过的地图不劣判据。固定地图拟合权重后，将中性面积换为正确类别面积，$J$ 的变化为 {values('S_perf - G_perf')} 个百分点，{'达到' if gates['category_increment_gate'] else '未达到'}额外价值判据。原 S 相对未加权中性 G 的变化为 {values('S - G')} 个百分点，相对交换类别先验 X 的变化为 {values('S - X')} 个百分点。24 个登记条目中有 {len(registered_endpoints)} 个取得终点，对应 {len(native_endpoints)} 次实际执行，其中 {returned} 次返航，共记录 {collisions} 次模拟碰撞。独立比较样本为四栋建筑。工程判据的结论限于所测布局、评分权重族和 192 步预算。
'''


def main():
    protocol, weights = read(PROTOCOL), read(WEIGHTS)
    measurements, samples = read(MEASUREMENT/'episodes.json'), read(MEASUREMENT/'checkpoints.json')
    decision = read(DECISION/'heldout_decision_audit.json')
    assert measurements['run_freeze_sha256'] == sha(freeze_path('holdout'))
    assert samples['run_freeze_sha256'] == measurements['run_freeze_sha256']
    assert decision['frozen_weight_file_sha256'] == sha(WEIGHTS)
    rows = measurements['rows']
    episodes = {(row['layout'],row['arm']): row for row in rows}
    assert len(episodes) == len(rows) == 24
    for row in rows:
        if measured(row):
            assert abs(row['J'] - row['C_free'] * row['F1_scene_05cm']) <= 1e-10
    paired = pair_rows(episodes, protocol)
    gates = adjudicate(paired, decision, episodes)
    RESULTS.mkdir(exist_ok=True)
    with (RESULTS/'paired_buildings.csv').open('w', newline='') as stream:
        fields = list(dict.fromkeys(k for row in paired for k in row))
        writer = csv.DictWriter(stream, fields)
        writer.writeheader()
        writer.writerows(paired)
    (RESULTS/'engineering_verdict.json').write_text(json.dumps(gates, indent=2, allow_nan=False)+'\n')
    (RESULTS/'manuscript_zh.md').write_text(manuscript(gates, paired, weights, episodes, decision))
    render(samples['rows'], protocol)
    print(gates)


if __name__ == '__main__':
    main()
