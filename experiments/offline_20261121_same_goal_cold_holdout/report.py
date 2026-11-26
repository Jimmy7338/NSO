#!/usr/bin/env python3
"""Render the registered cold-start comparison from independently evaluated endpoints."""

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use('Agg')
from matplotlib import pyplot as plt


HERE = Path(__file__).resolve().parent
COLORS = {'fullG': '#0072B2', 'fullS': '#D55E00', 'originalS': '#009E73', 'F_strict': '#6A578F'}
LABELS = {'fullG': 'Expanded-pool G', 'fullS': 'Expanded-pool S',
          'originalS': 'Original S', 'F_strict': 'Nearest reachable frontier'}
DISPLAY = {'ordinary_chain_13': 'Chain 13', 'ordinary_cycle_14': 'Cycle 14',
           'ordinary_chain_15': 'Chain 15', 'ordinary_cycle_16': 'Cycle 16'}
COMPARATORS = ('fullG', 'originalS', 'F_strict')


def complete_endpoints(rows, protocol):
    keys = {(layout, mode) for layout in protocol['layouts'] for mode in protocol['modes']}
    indexed = {(row['layout'], row['mode']): row for row in rows}
    if len(rows) != len(keys) or indexed.keys() != keys:
        raise ValueError('The endpoint source does not contain exactly one result per registered run.')
    for key, row in indexed.items():
        if row.get('J') is None:
            if row['status'] == 'completed_execution':
                raise ValueError(f'Completed episode lacks a measured endpoint for {key}.')
            continue
        if abs(row['J'] - row['C_free'] * row['F1_scene_05cm']) > 1e-10:
            raise ValueError(f'Inconsistent primary metric for {key}.')
    return indexed


def j_points(row):
    value = row.get('J')
    return None if value is None else 100 * value


def paired_points(semantic, comparator):
    category, control = j_points(semantic), j_points(comparator)
    return None if category is None or control is None else category - control


def render_figure(indexed, protocol, output):
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 9, 'pdf.fonttype': 42,
                         'svg.fonttype': 'none', 'axes.spines.top': False, 'axes.spines.right': False})
    figure, (absolute, paired) = plt.subplots(1, 2, figsize=(8.35, 3.35),
        gridspec_kw={'width_ratios': (1.07, 1)}, sharey=True)
    figure.subplots_adjust(left=.12, right=.98, bottom=.23, top=.86, wspace=.27)
    y_positions = list(range(len(protocol['layouts'])))
    absolute.set_yticks(y_positions, [DISPLAY[name] for name in protocol['layouts']])
    absolute.invert_yaxis()
    missing_any = False
    for y, layout in zip(y_positions, protocol['layouts'], strict=True):
        neutral = j_points(indexed[(layout, 'fullG')])
        category = j_points(indexed[(layout, 'fullS')])
        if neutral is not None and category is not None:
            absolute.plot((neutral, category), (y - .15, y - .05), color='#91A2AB', linewidth=1.1, zorder=1)
        for index, mode in enumerate(protocol['modes']):
            value = j_points(indexed[(layout, mode)])
            if value is None:
                missing_any = True
                continue
            absolute.scatter(value, y + (index - 1.5) * .10, color=COLORS[mode], s=28, zorder=3)
        for index, mode in enumerate(COMPARATORS):
            offset = (index - 1) * .17
            delta = paired_points(indexed[(layout, 'fullS')], indexed[(layout, mode)])
            if delta is not None:
                paired.scatter(delta, y + offset, color=COLORS[mode], s=28, zorder=3)
    absolute.set(title='Actual-stop joint map quality', xlabel='Joint map quality J (%)')
    paired.set(title='Expanded-pool S relative to each comparator', xlabel='Paired difference in J (percentage points)')
    paired.axvline(0, color='#57616B', linewidth=.8)
    paired.axvline(.10, color=COLORS['fullG'], alpha=.60, linewidth=.8, linestyle='--')
    paired.set_xlim(min(paired.get_xlim()[0], -.12), max(paired.get_xlim()[1], .16))
    for axes in (absolute, paired):
        axes.grid(True, axis='y', color='#DDE4E9', linewidth=.65)
        axes.tick_params(axis='y', length=0)
    for mode in protocol['modes']:
        absolute.scatter([], [], color=COLORS[mode], s=28, label=LABELS[mode])
    handles, labels = absolute.get_legend_handles_labels()
    figure.legend(handles, labels, loc='lower center', bbox_to_anchor=(.5, .04), ncol=4,
                  frameon=False, columnspacing=1.2, handletextpad=.38)
    if missing_any:
        figure.suptitle('Missing endpoints are omitted. See the episode ledger.', fontsize=9)
    for suffix in ('pdf', 'svg'):
        figure.savefig(output / f'terminal_J_by_building.{suffix}', bbox_inches='tight')
    plt.close(figure)


def format_points(value, suffix='', signed=False, missing='未取得'):
    if value is None:
        return missing
    return format(value, '+.3f' if signed else '.3f') + suffix


def results_table(indexed, protocol):
    lines = ['| 建筑 | G 终点 J | S 终点 J | 原 S 终点 J | 严格前沿终点 J | S−G | S−原 S | S−严格前沿 |',
             '|---|---:|---:|---:|---:|---:|---:|---:|']
    for layout in protocol['layouts']:
        levels = [format_points(j_points(indexed[(layout, mode)]), '%') for mode in protocol['modes']]
        deltas = [format_points(paired_points(indexed[(layout, 'fullS')], indexed[(layout, mode)]), signed=True)
                  for mode in COMPARATORS]
        lines.append('| ' + ' | '.join([DISPLAY[layout], *levels, *deltas]) + ' |')
    return '\n'.join(lines) + '\n'


def mechanism_table(indexed, pair_rows, protocol):
    pairs = {row['layout']: row for row in pair_rows}
    headings = ['建筑', '第一个不同动作／计费步', '原始传感首次差异步', '完整轨迹及网格',
                '评估候选 G／S', '通过准入的备选 G／S', '选用备选的决策次数 G／S',
                '计费步骤规划总耗时 G／S（秒）', '设施表面净增量 S−G（m²）']
    rows = [headings, ['---'] * len(headings)]
    for layout in protocol['layouts']:
        pair = pairs[layout]
        g, s = [indexed[(layout, mode)] for mode in ('fullG', 'fullS')]
        commands = pair.get('first_command_difference')
        sensor = pair.get('first_sensor_content_difference')
        if pair['status'] == 'failed_or_missing_arm':
            trace = '配对未完成'
        elif pair['identical_charged_action_and_packet_trace'] and pair['identical_terminal_mesh']:
            trace = '一致'
        else:
            trace = '有差异'

        def measured(field, spec=''):
            values = [m.get(field, 0 if m.get('local_route_choices') == 0 else None) for m in (g, s)]
            return '／'.join(format(value, spec) if value is not None else '—' for value in values)

        net = None if g.get('object_net_m2') is None or s.get('object_net_m2') is None \
            else s['object_net_m2'] - g['object_net_m2']
        rows.append([
            DISPLAY[layout],
            f'{commands["fullG"]["action"]}／{commands["fullS"]["action"]}／{commands["paid"]}' if commands else '—',
            str(sensor['paid']) if sensor else '—', trace, measured('assessed_candidates_total'),
            measured('admitted_alternates_total'), measured('selected_alternates_total'),
            measured('decision_wall_s', '.1f'), format_points(net, signed=True, missing='—'),
        ])
    table = '\n'.join('| ' + ' | '.join(row) + ' |' for row in rows)
    return (
        '# 主对照的选路、观测与计算开销\n\n'
        + table
        + '\n\n准入备选数不含原生路线。候选数是本轮实际执行策略各次全池评估的合计。'
          '规划总耗时汇总产生已计费动作的步骤，记录于共享仿真工作站。'
          '设施表面净增量按各臂与自身首帧比较，再在两臂间作差。因果归因见逐设备和候选审计。\n'
    )


def manuscript_result(indexed, protocol, gate):
    layouts = protocol['layouts']
    primary = [(indexed[(layout, 'fullS')], indexed[(layout, 'fullG')]) for layout in layouts]
    measured_pairs = [(s, g) for s, g in primary if paired_points(s, g) is not None]
    threshold = protocol['evaluation']['practical_gate']['threshold_delta_J_fraction_strict_gt']
    positive = sum(s['J'] - g['J'] > threshold for s, g in measured_pairs)
    returns_lost = sum(g['returned'] and not s['returned'] for s, g in measured_pairs)
    more_collisions = sum(s['collisions'] > g['collisions'] for s, g in measured_pairs)
    within_budget = sum(row.get('actual_paid', float('inf')) <= protocol['native_budget'] for row in indexed.values())
    if positive != gate['count_qualifying_buildings']:
        raise ValueError('Displayed result disagrees with the evaluator gate.')
    passed = gate['pass']

    def comparison(mode, missing):
        return ', '.join(format_points(paired_points(indexed[(layout, 'fullS')], indexed[(layout, mode)]),
                                       signed=True, missing=missing) for layout in layouts)

    baselines_zh = (f'相对原 S 的逐建筑差值为 {comparison("originalS", "未取得")} 个百分点，'
                    f'相对严格最近可达前沿为 {comparison("F_strict", "未取得")} 个百分点。')
    baselines_en = (f'The paired differences against original S were {comparison("originalS", "unavailable")} '
                    f'percentage points, and against nearest reachable frontier '
                    f'{comparison("F_strict", "unavailable")} percentage points.')
    zh = (
        '在两座链式与两座环式留出建筑上，四种策略各从共同首帧启动，动作预算为 512 步。'
        '扩池 G 与 S 在共同状态使用同一中性全局目标规则及同一备选路径池，仅局部表面收益的结构先验不同。'
        '仿真提供当前可见设施的理想类别标签。主指标为实际停止时正确自由空间覆盖率与整场景表面 F1@5 cm 的乘积。'
        f'按链式 13、环式 14、链式 15、环式 16 的顺序，S−G 为 {comparison("fullG", "未取得")} 个百分点。'
        f'有 {positive}/{len(layouts)} 座建筑超过 +{100*threshold:.2f} 个百分点。'
        f'在 {len(measured_pairs)} 组有终点的配对中，S 在 {returns_lost} 座建筑失去 G 已实现的返航，'
        f'在 {more_collisions} 座建筑增加碰撞。'
        f'{within_budget}/{len(indexed)} 轮在登记预算内停止。'
        f'{baselines_zh} 此批次{"达到" if passed else "未达到"}预先登记的实用门槛。建筑是独立比较单位。'
    )
    en = (
        'For each of the two chain and two cycle held-out buildings, four policies started from a common first '
        'observation and shared a cap of 512 charged actions. At a common acquired state, expanded-pool G and S used the same '
        'neutral global goal rule and set of local routes. They differed only in the structural prior used to score '
        'local surface gain. The simulator provided ideal labels for visible fixtures. The primary endpoint was the '
        'product of correct free-space coverage and whole-scene surface F1 at 5 cm at actual termination. '
        f'For Chain 13, Cycle 14, Chain 15, and Cycle 16, respectively, S minus G was '
        f'{comparison("fullG", "unavailable")} percentage points. '
        f'{positive} of {len(layouts)} buildings exceeded +{100*threshold:.2f} percentage points. '
        f'Of the {len(measured_pairs)} complete pairs, S lost {returns_lost} completed G returns and had more '
        f'collisions in {more_collisions} buildings. '
        f'{within_budget} of {len(indexed)} episodes stopped within the action cap. {baselines_en} '
        f'The comparison {"met" if passed else "fell short of"} the registered practical criterion. '
        'The building was the independent unit of comparison.'
    )
    return f'# 论文结果段落\n\n{zh}\n\n# Manuscript results\n\n{en}\n'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--episodes', type=Path, default=HERE / 'measurement/results/episodes.json')
    parser.add_argument('--output', type=Path, default=HERE / 'results')
    arguments = parser.parse_args()
    protocol = json.loads((HERE / 'PROTOCOL.json').read_text())
    episode_payload = json.loads(arguments.episodes.read_text())
    gate = json.loads((arguments.episodes.parent / 'engineering_gate.json').read_text())['engineering_gate']
    pairs = json.loads((arguments.episodes.parent / 'pair_causal_audit.json').read_text())['rows']
    indexed = complete_endpoints(episode_payload['rows'], protocol)
    arguments.output.mkdir(exist_ok=True, parents=True)
    render_figure(indexed, protocol, arguments.output)
    (arguments.output / 'terminal_J_table_zh.md').write_text(results_table(indexed, protocol))
    (arguments.output / 'route_mechanism_zh.md').write_text(mechanism_table(indexed, pairs, protocol))
    (arguments.output / 'manuscript_results_zh_en.md').write_text(manuscript_result(indexed, protocol, gate))


if __name__ == '__main__':
    main()
