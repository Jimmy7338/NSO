#!/usr/bin/env python3
"""Write the manuscript paragraph from the sealed two-case measurements."""

import math

from study import ARMS, FREEZE, MEASUREMENT, PROTOCOL, ROOT, read, sha


LABELS = {'paired_chain_17': '链式 17', 'paired_cycle_20': '环式 20'}
FACILITIES = {'open31': '开放货架', 'closed118': '封闭柜体'}
ACTIONS = {'forward': '前进', 'left': '左转', 'right': '右转', 'backward': '后退'}


def area(checkpoint, target):
    return checkpoint['facilities'][target]['gross_new_m2']


def signed(value):
    return f'{value:+.4f}' if abs(value) >= .00005 else '0.0000'


def route_outcome(checkpoint, target):
    event = checkpoint.get('route_event')
    if not event:
        return '无有效路线事件'
    word = '到达' if event['status'] == 'arrived' else '中止'
    if event['reason'] == 'next_footprint_not_known_safe':
        word = '因下一步足迹未获安全确认而中止'
    return f"第 {event['actual_paid']} 步{word}时为 {area(checkpoint, target):.4f} m²"


def case_paragraph(layout, case):
    name, target = LABELS[layout], case['primary_target']
    a, b = [case['arms'][arm] for arm in ARMS]
    difference = case['pair'].get('first_actual_action_difference')
    if difference is None:
        motion = '两臂的共同行驶时段内未观测到指令分歧。'
    else:
        left, right = [ACTIONS.get(difference[f'{arm}_action'], difference[f'{arm}_action']) for arm in ARMS]
        actual_pose = case['pair']['first_actual_pose_difference']
        motion = f"第 {difference['paid']} 步的实际指令首次分歧，类别路线{left}、几何路线{right}。"
        if actual_pose is not None:
            motion += f"第 {actual_pose['paid']} 步首次出现位置或朝向差异。"
    route = f"{name} 以{FACILITIES[target]}为目标设施。{motion}"
    events = [branch.get('route_end', {}).get('route_event', {}) for branch in (a,b)]
    shared_arrival = (all(event.get('status') == 'arrived' for event in events)
                      and events[0]['actual_paid'] == events[1]['actual_paid'])
    if shared_arrival:
        route += f"两条路线均在第 {events[0]['actual_paid']} 步到达同一目标。"
    else:
        route += (f"类别路线{route_outcome(a.get('route_end', {}), target)}，"
                  f"几何路线{route_outcome(b.get('route_end', {}), target)}。")
    if 'equal_clock' not in case['pair']:
        equal = '共同计费时刻缺少至少一臂的实际网格，配对面积记为缺失。'
        equal_delta = None
    else:
        shared = case['shared_route_evaluation_paid']
        am = area(a['equal_clock'], target)
        bm = area(b['equal_clock'], target)
        equal_delta = am - bm
        partition = case['pair']['equal_clock'][target]
        common, only_a, only_b = [partition[key]['area_m2']
                                  for key in ('common_new','category_only_new','geometry_only_new')]
        assert math.isclose(am, common+only_a, abs_tol=1e-12) and math.isclose(bm, common+only_b, abs_tol=1e-12)
        equal = (f"在共同第 {shared} 步，类别与几何路线分别新增 {am:.4f} 和 {bm:.4f} m²，"
                 f"差值为 {signed(equal_delta)} m²。两臂共同新增 {common:.4f} m²，各自独有 {only_a:.4f} 和 {only_b:.4f} m²。")
        earlier = [(name, shared - measurement['route_end']['receipt']['actual_paid'])
                   for name, measurement in (('类别臂', a), ('几何臂', b))]
        for name, extra_paid in earlier:
            if extra_paid:
                equal += f'该时刻的{name}结果包含初始路线结束后的 {extra_paid} 个 G_perf 动作。'
    finals = [item.get('terminal', {}) for item in (a, b)]
    if not all(item.get('status') == 'measured' for item in finals):
        return route + equal + '至少一臂没有可测终点，无法核验后续保留。'
    ca, cb = [item['receipt']['actual_paid'] for item in finals]
    ta, tb = [area(item, target) for item in finals]
    location = f'共同第 {ca} 步终点' if ca == cb else f'各自第 {ca} 和 {cb} 步终点'
    close = (f"在{location}，两臂分别保有相对分叉新恢复的 {ta:.4f} 和 {tb:.4f} m²，"
             f"终点差值为 {signed(ta-tb)} m²。")
    if 'retention' in case['pair']:
        cohort = case['pair']['retention'][target]['category']
        only = cohort['own_only_at_terminal']['area_m2']
        both = cohort['both_at_terminal']['area_m2']
        source = case['pair']['equal_clock'][target]['category_only_new']['area_m2']
        if source > 0:
            close += f"共同步时类别臂独有的表面中，{only:.4f} m² 在终点仍独有，{both:.4f} m² 最终被两臂都恢复。"
    return route + equal + close


def main():
    protocol = read(PROTOCOL)
    measurement = read(ROOT/'evaluation/results/measured.json')
    assert measurement['measurement_protocol_sha256'] == sha(MEASUREMENT)
    assert measurement['runtime_freeze_sha256'] == sha(FREEZE)
    cases = measurement['cases']
    assert list(cases) == [entry['layout'] for entry in protocol['cases']]
    introduction = ('我们选取两处已观察到 S 与几何评分偏好分歧的仿真状态。'
                    '每对从同一原生状态、融合地图和目标开始，分别指定一次类别路线与同代价几何路线。'
                    '随后两臂以相同的 G_perf 权重独立采集和融合观测，192 个计费动作包含公共前缀与返航。'
                    '链式 17 的几何对照在 34 条合法路线中排第 12，环式 20 的对照在 55 条合法路线中排第 1。'
                    '新增面积以分叉前的融合网格为基准，使用固定的可评设施表面和 5 cm 匹配容差。')
    paragraphs = ['# 单次选路干预的论文结果段', introduction]
    paragraphs.extend(case_paragraph(layout, case) for layout, case in cases.items())
    paragraphs.append('这两处干预用于解释个别选路的观察后果。样本来自已经出现分歧的状态，归因范围限于这两条路线对、仿真传感器和所测预算。')
    out = ROOT/'results/manuscript_zh.md'
    out.parent.mkdir(exist_ok=True)
    out.write_text('\n\n'.join(paragraphs)+'\n')
    print(out)


if __name__ == '__main__':
    main()
