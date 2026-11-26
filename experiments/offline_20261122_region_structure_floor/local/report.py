#!/usr/bin/env python3
"""Paper-ready editable trajectory and matched reference-surface figure."""

import argparse
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
matplotlib.rcParams.update({'svg.fonttype':'none','pdf.fonttype':42})
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np

from run import ARMS, HERE, PROTOCOL, SCENES, read


COLORS={'G':'#566879','D':'#9F4307','S':'#146E88','X':'#856D9A'}
LINE={'G':':','D':'--','S':'-','X':'-.'}


def points_xy(cells,height,res):
    if not cells:
        return np.empty((0,2))
    arr=np.asarray(cells)
    return np.column_stack((arr[:,1]+.5,height-arr[:,0]-.5))*res


def plot(analysis,out):
    plan=read(PROTOCOL)
    names=[n for n in plan['placements'] if n in analysis['arms']]
    height_in=3.05*len(names)+1.0
    fig,axes=plt.subplots(len(names),2,figsize=(10.2,height_in),squeeze=False)
    fig.subplots_adjust(left=.09,right=.96,bottom=1.0/height_in,top=1-.42/height_in,hspace=.52,wspace=.28)
    label={'east_visible':'East, exposed','east_screened':'East, screened',
           'west_visible':'West, exposed','west_screened':'West, screened'}
    for index,layout in enumerate(names):
        track,shelf=axes[index]
        stem=SCENES/'prepared'/layout/'initial_state.json'
        initial=read(stem)
        height,width=initial['shape']
        r=initial['grid_resolution_m']
        preflight=read(SCENES/'evaluator_references'/layout/'optical_preflight.json')
        with np.load(SCENES/'evaluator_references'/layout/'evaluator_truth.npz',allow_pickle=False) as ref:
            occ=ref['occupancy']
        track.imshow(occ, cmap='Greys',alpha=.23,vmin=0,vmax=1,origin='upper',extent=(0,width*r,0,height*r),interpolation='nearest')
        tracks=analysis['arms'][layout]
        fork_marked=False
        for arm in ARMS:
            val=tracks.get(arm,{})
            if val.get('status')!='verified_native_episode':
                continue
            steps=read(Path(val['steps_path']))
            prefix=val['trigger'].get('paid_before',0)
            trigger_max=min(analysis['common_paid'],len(steps))
            if prefix is None or prefix>=trigger_max:
                continue
            sites=[row['position'] for row in steps[:trigger_max]]
            prefix_pose=initial['anchor'][:2] if prefix==0 else sites[prefix-1]
            xy=points_xy([prefix_pose]+sites[prefix:],height,r)
            track.plot(xy[:,0],xy[:,1],LINE[arm],color=COLORS[arm],linewidth=1.5,alpha=.90)
            if not fork_marked:
                track.scatter(*xy[0],marker='*',c='#111111',s=40,zorder=5)
                fork_marked=True
        hits=preflight['visible_shelf_samples']
        track.set_title(f"{label[layout]} — side-view hits {hits:,}/10,000",loc='left',fontsize=11)
        track.set_xlim(.0,width*r);track.set_ylim(.0,height*r)
        track.set_aspect('equal');track.set_xlabel('x (m)');track.set_ylabel('y (m)')
        mask_receipt=analysis['surface_masks'].get(layout)
        if not mask_receipt:
            shelf.text(.5,.5,'No verified native endpoints',ha='center',va='center',transform=shelf.transAxes)
            continue
        with np.load(mask_receipt['path'],allow_pickle=False) as mask:
            xyz=mask['reference_shelf_xyz_m']
            plotted=xyz[:,[0,2]]
            shelf.scatter(*plotted.T,s=1,c='#BBBBBB',alpha=.35)
            usable={'S_common_shelf','D_common_shelf','S_fork_shelf','S_terminal_shelf','D_terminal_shelf'}<=set(mask.files)
            if usable:
                before=mask['S_fork_shelf']
                l,rnew=mask['S_common_shelf']&~before,mask['D_common_shelf']&~before
                for points,color,sz in [(l&rnew,'#667481',2),(l&~rnew,COLORS['S'],4),(rnew&~l,COLORS['D'],4)]:
                    shelf.scatter(*plotted[points].T,s=sz,c=color,alpha=.8)
                retained=((l&~rnew)&mask['S_terminal_shelf']&~mask['D_terminal_shelf'])|((rnew&~l)&mask['D_terminal_shelf']&~mask['S_terminal_shelf'])
                shelf.scatter(*plotted[retained].T,s=9,facecolor='none',edgecolor='#222222',linewidth=.45)
        caption=(f'Shelf surfaces at action {analysis["common_paid"]}' if usable
                 else f'Action {analysis["common_paid"]} comparison unavailable')
        shelf.set_title(caption,loc='center',fontsize=10)
        shelf.set_xticks(np.round(np.quantile(plotted[:,0],(.04,.96)),2))
        shelf.set_xlabel('x (m)');shelf.set_ylabel('z (m)');shelf.set_aspect('equal')
    handles=[Line2D([0],[0],color=COLORS[arm],linestyle=LINE[arm],label={'G':'neutral G_perf','D':'direct inspection','S':'correct structure','X':'swapped category'}[arm]) for arm in ARMS]
    handles.extend([Line2D([0],[0],color='#111111',marker='*',linestyle='None',label='shared fork'),
                    Line2D([0],[0],color='#667481',marker='o',linestyle='None',label='newly represented in both S and D'),
                    Line2D([0],[0],color='#222222',marker='o',markerfacecolor='none',linestyle='None',label='exclusive through terminal')])
    fig.legend(handles=handles,loc='lower center',bbox_to_anchor=(.5,.07/height_in),ncol=4,frameon=False,fontsize=8)
    fig.savefig(out/'local_inspection_trajectories_and_surfaces.pdf',dpi=240)
    fig.savefig(out/'local_inspection_trajectories_and_surfaces.svg',dpi=240)
    plt.close(fig)


def manuscripts(data,out):
    names=list(data['arms'])
    cases={}
    for name in names:
        c=data['comparisons'].get(name,{}).get('S_minus_D',{})
        value=c.get('common',{}).get('open31')
        terminal=c.get('terminal',{}).get('open31')
        if value and terminal:
            cases[name]=(value,terminal,c)
    episodes=[v for record in data['arms'].values() for v in record.values() if v.get('status')=='verified_native_episode']
    observed=len(episodes)
    arrivals=sum(v['route_event']['status']=='arrived' for v in episodes)
    returns=sum(v['returned'] for v in episodes)
    collisions=sum(v['collisions'] for v in episodes)
    both_available=len(cases)
    clocks=[f'{name} {early["net_delta_m2"]:+.4f}/{terminal["net_delta_m2"]:+.4f} m²'
            for name,(early,terminal,_) in cases.items()]
    pairs='、'.join(clocks)
    pairs_en=', '.join(clocks)
    case_zh=(f'有效配对中，类别条件结构推断相对直接巡检的净新增货架表面积在共同第 96 步和终点分别为 {pairs}。'
             if pairs else '共同第 96 步无可计算的类别条件结构推断与直接巡检配对差。')
    case_en=(f'The net newly represented shelf area for category-conditioned planning minus direct inspection at action 96 and at the actual terminal was {pairs_en}.'
             if pairs_en else 'The measurements did not yield a computable paired contrast at action 96.')
    (out/'manuscript_zh.md').write_text(
        '# 同目标路线的局部巡检对照\n\n'
        '我们在两个独立的合成房间布局中固定四处设施摆位，分别保留可通行的侧窗观察与放置遮挡板的情形。'
        '分叉时，四臂使用同一批观测、同一个全局目标及同一完整合法路线池，随后分别采集与融合深度。'
        '直接巡检与类别条件结构推断共享仿真可见理想类别。直接巡检按已测点的视角欠覆盖、距离和入射角评估各路线。'
        f'验收 {observed}/{len(names)*4} 轮完整回执，有效局部配对 {both_available}/{len(names)} 处。'
        +case_zh+f'共 {arrivals} 轮按初选路线到达，{returns} 轮实际返航，记录 {collisions} 次模拟碰撞。'
        '预算 192 个计费动作包含转向和返航。同一合成父布局的两处摆位为相关对照。\n')
    (out/'manuscript_en.md').write_text(
        '# Matched same-goal inspection\n\n'
        'We fixed four placements in two synthetic parent layouts with an accessible side-view position, '
        'and compared the exposed condition with an opaque screen. The four policies shared the semantic observation input, '
        'global goal and full legal route pool at the choice point. They then collected and fused depth independently. '
        'The direct-inspection baseline used the same ideal visible category labels and ranked routes by the angular undercoverage, '
        'range and incidence of previously measured fixture surfels. '
        'The category-conditioned scorer predicted surface visibility using a structural posterior. '
        f'{observed}/{len(names)*4} episodes are verified and {both_available}/{len(names)} placements yield computable focal contrasts. '
        +case_en+f' The selected branch reached its goal in {arrivals} episodes. '
        f'Across verified episodes, {returns} returned and {collisions} simulated collisions were recorded. '
        'All charged turns and return actions counted against the 192-action budget. '
        'Placements within each parent were paired controls.\n')

    specified={
        'east_visible':'东侧无遮挡',
        'east_screened':'东侧遮挡',
        'west_visible':'西侧无遮挡',
        'west_screened':'西侧遮挡',
    }
    table=[
        '# 同目标巡检配对结果',
        '',
        '面积为相对共同分叉网格的净新增货架表面，单位 m²。每格依次列共同第 96 步与实际终点。首次选路干预之后的重规划均使用中性几何。',
        '',
        '| 摆位 | 中性几何 G | 直接巡检 D | 正确结构 S | 错置结构 X | S−D | S−X |',
        '|---|---:|---:|---:|---:|---:|---:|',
    ]
    for name in names:
        arms = data['arms'][name]
        contrasts = data['comparisons'].get(name,{})
        vals = [specified[name]]
        for arm in ARMS:
            areas = [arms[arm].get(phase,{}).get('open31',{}).get('net_m2') for phase in ('common','terminal')]
            vals.append(' / '.join('—' if area is None else f'{area:.4f}' for area in areas))
        for pair in ('S_minus_D','S_minus_X'):
            areas = [contrasts.get(pair,{}).get(phase,{}).get('open31',{}).get('net_delta_m2') for phase in ('common','terminal')]
            vals.append(' / '.join('—' if area is None else f'{area:+.4f}' for area in areas))
        table.append('| ' + ' | '.join(vals) + ' |')
    table.extend([
        '',
        '第 96 步的独有面积比较相对共同分叉新增的参考表面。末列保留到实际终点，且另一臂在终点仍未恢复同一区域。',
        '',
        '| 摆位 | S/D 同一初选路线 | 首次执行分歧步 | 全程路程 S / D（m） | 第 96 步独有面积 S / D（m²） | 到终点仍独有 S / D（m²） |',
        '|---|---|---:|---:|---:|---:|',
    ])
    for name in names:
        contrast=data['comparisons'].get(name,{}).get('S_minus_D')
        if not contrast:
            table.append(f'| {specified[name]} | — | — | — | — | — |')
            continue
        difference=contrast['first_actual_difference']
        clock=str(difference['paid']) if difference else '全程相同'
        same='是' if contrast['same_chosen_route'] else '否'
        arms=data['arms'][name]
        travel=f"{arms['S']['path_length_m']:.2f} / {arms['D']['path_length_m']:.2f}"
        if name not in cases:
            table.append(f'| {specified[name]} | {same} | {clock} | {travel} | — | — |')
            continue
        early,_,_=cases[name]
        retained=early['retention_m2']
        table.append(f"| {specified[name]} | {same} | {clock} | {travel} | {early['left_only_new_m2']:.4f} / "
                     f"{early['right_only_new_m2']:.4f} | {retained['left_only']['left']:.4f} / "
                     f"{retained['right_only']['right']:.4f} |")
    table.extend(['',f'核验 {observed}/{len(names)*4} 轮，初选路线到达 {arrivals} 轮，'
                     f'返航 {returns} 轮，模拟碰撞 {collisions} 次。四处摆位来自两个合成父布局。',''])
    interrupted_path=HERE/'evidence_checks'/'aborted_checkpoint_diagnostics.json'
    if interrupted_path.exists() and 'east_screened' in names:
        interrupted=read(interrupted_path)
        table.extend([
            '## 东侧遮挡中止前快照',
            '',
            'G 和 S 初选同一路线。两臂各两次都在第 180 步后于共用 Open3D 模板光线场景构建中崩溃。'
            '四次的共同分叉、第 96 和 144 步传感数据链与实际重建网格哈希一致。'
            '以下只复算留存快照，G/S 的全程动作账与终点缺测。',
            '',
            '| 计费步 | G/S 净新增面积（m²） | D（m²） | X（m²） |',
            '|---|---:|---:|---:|',
        ])
        for paid in (96,144):
            stage=f'paid_{paid}'
            same=interrupted['same_native_G_and_S'][stage]['shelf']['net_m2']
            d,x=[interrupted['successfully_completed_arms'][arm][stage]['net_m2'] for arm in ('D','X')]
            table.append(f'| {paid} | {same:.4f} | {d:.4f} | {x:.4f} |')
        table.append('')
    early_path=HERE/'evidence_checks'/'early_stop_48.json'
    if early_path.exists() and 'west_screened' in names:
        early=read(early_path)['independently_verified_arms']
        areas=[early[arm]['shelf']['net_m2'] for arm in ARMS]
        table.extend([
            '## 西侧遮挡提前停止',
            '',
            'G/S/X 同选，均因实时占据图上的返航锚点失去连通性在第 61 步退出。'
            'D 实际于第 46 步到达目标，运行至第 192 步并返航。'
            '第 96 步只对实际到达该时刻的 D 计量。原计划同步采集的第 48 步可作描述性比较。',
            '',
            '| 计费步 | G 净增（m²） | D 净增（m²） | S 净增（m²） | X 净增（m²） |',
            '|---|---:|---:|---:|---:|',
            '| 48 | ' + ' | '.join(f'{value:.4f}' for value in areas) + ' |',
            '',
            '主表中的西侧遮挡终点按各自实停时刻报告。S−D 的终点面积差包含共同时间窗之后的额外动作。',
            '',
        ])
    (out/'paired_result.md').write_text('\n'.join(table))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--results',type=Path,default=HERE/'results')
    args=parser.parse_args()
    data=read(args.results/'measured.json')
    plot(data,args.results)
    manuscripts(data,args.results)


if __name__=='__main__':
    main()
