#!/usr/bin/env python3
"""Generate the paper paragraph from the complete three-arm evaluation."""

from study import FREEZE, ROOT, read, sha


ARMS = ('category','geometry_equal_cost','geometry_top')


def signed(value):
    return '0.0000' if abs(value)<.00005 else f'{value:+.4f}'


def new_area(checkpoint):
    return checkpoint['facility']['gross_new_m2']


def event_name(arm):
    event = arm['route']
    return f"第 {event['actual_paid']} 步{'到达' if event['status']=='arrived' else '中止'}"


def effect(comparison, phase):
    paired = comparison.get(f'{phase}_surface')
    return None if paired is None else paired.get('category_minus_comparator_new_m2')


def main():
    data = read(ROOT/'evaluation/results/measured.json')
    declared = data.get('runtime_freeze_sha256') or data.get('run_freeze_sha256') or data.get('freeze_sha256')
    assert declared == sha(FREEZE)
    category,equal,top = (data['arms'][arm] for arm in ARMS)
    compared = data['comparisons']['category_minus_top']
    secondary = data['comparisons']['category_minus_equal_cost']
    geometry_event = event_name(top)
    category_event = event_name(category)
    own_top = new_area(top['route_event'])
    own_category = new_area(category['route_event'])
    lead = ('在链式 17 的已知分歧状态开展一例事后配对检验。三臂从原 S 第 105 步的同一融合地图、位姿和目标出发，'
            '此前执行的类别路线与等代价几何路线在 34 条合法候选中分别排第 13 和第 12，名义去程均为 67 步。'
            '新增闭环按 G_perf 的原评分和平局规则选出第 1 名原生路线，名义去程为 63 步。'
            '使用固定的开放货架可评表面和 5 cm 点到网格距离容差，量测相对于分叉网格的新增面积。')
    first = compared.get('first_action_difference')
    if first:
        lead += f"类别路线与首选路线在第 {first['paid']} 步首次发出不同指令。"
    lead += (f"类别路线{category_event}，新增 {own_category:.4f} m² 货架表面。"
             f"几何首选{geometry_event}，新增 {own_top:.4f} m²。")
    main_effect = effect(compared,'common')
    if main_effect is None:
        lead += '共同计费时刻缺少至少一臂的有效网格。'
    else:
        common = compared['common_paid']
        lead += f"在共同第 {common} 步，类别路线相对首选路线的新增面积差为 {signed(main_effect)} m²。"
        for name, arm in (('类别臂',category),('首选臂',top)):
            continuation = common-arm['route']['actual_paid']
            if continuation:
                lead += f"该时刻的{name}已经执行 {continuation} 步后续 G_perf。"
    final = effect(compared,'terminal')
    final_eq = effect(secondary,'terminal')
    clocks = [data['arms'][arm]['terminal']['receipt']['actual_paid'] for arm in ARMS]
    end_label = f'第 {clocks[0]} 步' if len(set(clocks))==1 else '各臂实际终点'
    if final is not None and final_eq is not None:
        net = category['terminal']['facility']['net_m2']-top['terminal']['facility']['net_m2']
        lead += (f"{end_label}类别路线相对首选与等代价几何路线的新增面积差分别为 {signed(final)} 和 {signed(final_eq)} m²。"
                 f"计入分叉时已恢复表面的后续失配后，相对首选的净增面积差为 {signed(net)} m²。")
    retention = compared.get('retention',{}).get('category',{})
    if retention:
        own = retention['own_only_at_terminal']['area_m2']
        shared = retention['both_at_terminal']['area_m2']
        lead += f"共同步由类别臂独有的实测表面在终点有 {own:.4f} m² 仍独有，{shared:.4f} m² 由两臂共同恢复。"
    lead += '该比较只反映这一仿真状态、目标和预算下的选路结果。'
    out = ROOT/'results/manuscript_zh.md'
    out.parent.mkdir(exist_ok=True)
    out.write_text('# 链式 17 首选几何路线对照\n\n'+lead+'\n',encoding='utf-8')
    print(out)


if __name__=='__main__':
    main()
