#!/usr/bin/env python3
"""Render the held-out paired brief and scientific action-time figure."""

import argparse
import csv
from pathlib import Path
from statistics import mean, median

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

from study import ROOT, read, sha


LABELS = {
    "ind26_vertical_spine": ("直线主走廊", "Straight spine"),
    "ind26_bent_corridor": ("折线走廊", "Bent corridor"),
    "ind26_offset_ring": ("错位环道", "Offset ring"),
    "ind26_double_loop": ("双环", "Double loop"),
}
ARMS=("region_S","combined_S","F_strict")
STYLE={"region_S":("#747c88","--","Original region + S"),
       "combined_S":("#136b82","-","Revised region + S"),
       "F_strict":("#a85b21",":","Strict nearest reachable frontier")}


def csv_rows(path):
    with path.open(newline="") as stream:
        return list(csv.DictReader(stream))


def number(row, key):
    return float(row[key]) if row.get(key) not in ("",None) else None


def percent(x, signed=False):
    return "缺测" if x is None else f"{100*x:+.2f}" if signed else f"{100*x:.2f}"


def difference(a,b):
    return None if a is None or b is None else a-b


def actual_return(row):
    return row.get("returned") in ("True","true","1",True)


def plot(curves, checkpoints, out, layouts):
    matplotlib.rcParams.update({"font.family":"DejaVu Sans","font.size":8.4,
                               "svg.fonttype":"none","pdf.fonttype":42})
    fig, axes=plt.subplots(2,4,figsize=(11.8,5.15),sharex=True,sharey="row",layout="constrained")
    for column,layout in enumerate(layouts):
        cover, joint = axes[:,column]
        cover.set_title(LABELS[layout][1],loc="left",fontsize=9.5,pad=7)
        cover.axhline(95,color="#222b35",linestyle=(0,(3,3)),lw=.65,zorder=0)
        for arm in ARMS:
            chosen=sorted((r for r in curves if (r["layout"],r["arm"])==(layout,arm)),key=lambda r:int(r["paid"]))
            captured=[r for r in checkpoints if (r["layout"],r["arm"])==(layout,arm)
                      and r["status"]=="captured" and r.get("J")]
            color,line,_=STYLE[arm]
            cover.plot([int(r["paid"]) for r in chosen],[100*number(r,"C_free") for r in chosen],
                       c=color,ls=line,lw=1.32)
            points={int(r["actual_paid"]):100*number(r,"J") for r in captured}
            ticks=sorted(points)
            if ticks:
                joint.plot(ticks,[points[x] for x in ticks],color=color,ls=line,lw=.85,
                           marker="o",ms=3.2,zorder=2)
                ends=[int(r["actual_paid"]) for r in captured if r["checkpoint"]=="terminal"]
                if ends:
                    x=ends[0]
                    joint.scatter([x],[points[x]],edgecolors=color,facecolors="none",marker="s",
                                  linewidths=.9,s=34,zorder=3)
        for ax in (cover,joint):
            ax.axvline(96,color="#a4b0bd",linestyle="--",lw=.70,zorder=0)
            ax.set_xlim(-3,196)
            ax.set_xticks([0,48,96,144,192])
            ax.set_ylim(-2,104)
            ax.grid(axis="y",color="#e3e8ee",lw=.55)
            ax.spines[["top","right"]].set_visible(False)
        joint.set_xlabel("Paid actions")
    axes[0,0].set_ylabel("Correct free-space cover C (%)")
    axes[1,0].set_ylabel("Joint map quality J (%)")
    handles=[Line2D([],[],color=c,ls=ls,label=name,lw=1.5) for c,ls,name in STYLE.values()]
    handles.append(Line2D([],[],color="#242e38",ls="",marker="s",mfc="none",label="Actual terminal"))
    fig.legend(handles=handles,loc="outside upper center",ncol=4,frameon=False)
    for suffix in ("pdf","svg","png"):
        fig.savefig(out/f"coverage_joint_quality.{suffix}",dpi=230)
    plt.close(fig)


def action_label(row):
    n=number(row,"A95_paid")
    if n is not None:
        return str(int(n))
    final=number(row,"C_free")
    if row["status"]!="completed_execution":
        return "执行或测量缺失"
    return f"预算内未达（{percent(final)}%）"


def clock_label(row):
    paid=number(row,"actual_paid")
    return str(int(paid)) if paid is not None else "缺测"


def seconds(row,key):
    time=number(row,key)
    return f"{time:.2f}" if time is not None else "缺测"


def metric_panel(rows, when):
    table=[
        "| 楼层 | 策略 | 实际动作 | C % | 精确率 % | 召回率 % | F1@5 cm % | J % |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for r in rows:
        if r["checkpoint"]==when:
            table.append(f"| {LABELS[r['layout']][0]} | {r['arm']} | {clock_label(r) if r['status']=='captured' else '缺测'} | " +
                         " | ".join(percent(number(r,k)) for k in ("C_free","p_scene_05cm","r_scene_05cm","F1_scene_05cm","J"))+" |")
    return table


def write_results(episodes, checkpoints, out, layouts):
    keys={(r["layout"],r["arm"]):r for r in episodes}
    score={(r["layout"],r["arm"],r["checkpoint"]):r for r in checkpoints}
    expected={(layout,arm) for layout in layouts for arm in ARMS}
    if len(episodes)!=12 or keys.keys()!=expected:
        raise ValueError("The report expects the complete frozen four-floor matrix.")
    results={}
    for layout in layouts:
        repaired=keys[layout,"combined_S"]
        frontier=keys[layout,"F_strict"]
        original=keys[layout,"region_S"]
        a,b,c=[number(row,"A95_paid") for row in (repaired,frontier,original)]
        timing={"revised_minus_frontier":difference(a,b),"revised_minus_original":difference(a,c)}
        joints=[number(score.get((layout,arm,"paid_96"),{}),"J") for arm in ("combined_S","F_strict","region_S")]
        trip=[actual_return(row) for row in (repaired,frontier,original)]
        recalls=[number(row,"r_scene_05cm") if returned else None
                 for row,returned in zip((repaired,frontier,original),trip,strict=True)]
        results[layout]={"a95":timing,"j96":difference(joints[0],joints[1]),
                         "j96_vs_original":difference(joints[0],joints[2]),
                         "recall_at_return":difference(recalls[0],recalls[1]),
                         "recall_at_return_vs_original":difference(recalls[0],recalls[2])}
    primary=[x["a95"]["revised_minus_frontier"] for x in results.values()
             if x["a95"]["revised_minus_frontier"] is not None]
    win=sum(x<0 for x in primary)
    tied=sum(x==0 for x in primary)
    loss=sum(x>0 for x in primary)
    attained={a:sum(number(keys[l,a],"A95_paid") is not None for l in layouts) for a in ARMS}
    early={a:sum((n:=number(keys[l,a],"A95_paid")) is not None and n<=96 for l in layouts) for a in ARMS}
    recalls=[x["recall_at_return"] for x in results.values() if x["recall_at_return"] is not None]
    earlier_cost=[x["recall_at_return"] for x in results.values() if
                 x["a95"]["revised_minus_frontier"] is not None and x["a95"]["revised_minus_frontier"]<0
                 and x["recall_at_return"] is not None]
    early_j=[x["j96"] for x in results.values() if x["j96"] is not None]
    early_f=[difference(number(score.get((l,"combined_S","paid_96"),{}),"F1_scene_05cm"),
                       number(score.get((l,"F_strict","paid_96"),{}),"F1_scene_05cm")) for l in layouts]
    available_f=[x for x in early_f if x is not None]
    elapsed={a:[number(keys[l,a],"bootstrap_and_task_wall_s") for l in layouts
                if number(keys[l,a],"bootstrap_and_task_wall_s") is not None] for a in ARMS}
    controls=["| 楼层 | 原区域 A95 | 修订区域 A95 | 严格前沿 A95 | 修订−前沿（步） | "
              "第 96 步 J 差（百分点） | 返航召回差（百分点） | 返航动作 修订／前沿 |",
              "| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |"]
    for name,x in results.items():
        original,repaired,frontier=(keys[name,arm] for arm in ARMS)
        delta=x["a95"]["revised_minus_frontier"]
        actions="/".join(clock_label(row) if actual_return(row) else "未返航" for row in (repaired,frontier))
        delta_text="—" if delta is None else f"{int(delta):+d}"
        controls.append(f"| {LABELS[name][0]} | {action_label(original)} | {action_label(repaired)} | "
                        f"{action_label(frontier)} | {delta_text} | {percent(x['j96'],True)} | "
                        f"{percent(x['recall_at_return'],True)} | {actions} |")
    statement=[
        "# 独立四楼验证简报",
        "",
        f"**提前达到高覆盖能否复现。** 全部四楼均进入分母。修订区域达标 {attained['combined_S']}/4，严格前沿 "
        f"{attained['F_strict']}/4，原区域 {attained['region_S']}/4。两法都达标的 {len(primary)}/4 楼中，"
        f"修订比前沿早／同／晚 {win}／{tied}／{loss}。在第 96 步及以前，修订和前沿分别有 "
        f"{early['combined_S']}/4、{early['F_strict']}/4 达到 95%。",
        "",
        *controls,
        "",
        f"**返航表面质量。** 修订与前沿均有返航评价的 {len(recalls)}/4 楼中，修订的表面召回有 "
        f"{sum(x<0 for x in recalls)} 楼较低。在修订确实省步且两法都返航的 {len(earlier_cost)} 楼中，"
        f"有 {sum(x<0 for x in earlier_cost)} 楼呈现较低召回。表内同时给出每法实际返航动作数。"
        + (f"第 96 步 J 在可配对的 {len(early_j)}/4 楼中，修订−前沿平均为 {100*mean(early_j):+.2f} 个百分点。"
           if early_j else ""),
        "",
        (f"第 96 步表面 F1 有 {sum(x>0 for x in available_f)}/{len(available_f)} 楼提高，"
         f"{sum(x<0 for x in available_f)} 楼降低。" if available_f else ""),
        "",
        f"12 轮中 {sum(actual_return(x) for x in episodes)}/12 返航，碰撞累计 "
        f"{sum(int(number(x,'collisions') or 0) for x in episodes)} 次。全套逐动作覆盖、轨迹、保护事件、"
        "真实测量路径和运行时间均随表保留。回合结束前没发生的检查点记为缺测。"
        + (f"并行仿真的中位总耗时为修订 {median(elapsed['combined_S']):.1f} 秒、前沿 "
           f"{median(elapsed['F_strict']):.1f} 秒。" if elapsed["combined_S"] and elapsed["F_strict"] else ""),
        "",
        "已知两楼的原始复测单列。所有数值均来自固定预算、准确仿真位姿和继承传感、融合与评价实现。"
        "四楼是比较单位。",
        "",
        "[四楼详细指标](TABLES.md) · [覆盖与三维质量图](coverage_joint_quality.pdf) · [已知两楼](../known/README.md)",
        "",
    ]
    (out/"BRIEF.md").write_text("\n".join(statement))
    more=["# 四楼完整指标", "", *controls, ""]
    for event in ("paid_48","paid_96","paid_144","terminal"):
        title=f"第 {event.split('_')[1]} 步" if event.startswith("paid_") else "各轮实际终点"
        more.extend([f"## {title}","",*metric_panel(checkpoints,event),""])
    more.extend(["## 修订区域与原区域", "",
        "| 楼层 | 修订−原区域 A95（步） | 第 96 步 J 差（百分点） | 返航召回差（百分点） |",
        "| --- | ---: | ---: | ---: |"])
    for name,x in results.items():
        delta=x["a95"]["revised_minus_original"]
        delta_text="—" if delta is None else f"{int(delta):+d}"
        more.append(f"| {LABELS[name][0]} | {delta_text} | {percent(x['j96_vs_original'],True)} | {percent(x['recall_at_return_vs_original'],True)} |")
    more.extend(["","相关绝对量及所有终点动作数见上表。","",
        "## 每轮运行开销与终止","",
        "| 楼层 | 策略 | 行程 m | 返航 | 碰撞 | 原生终止原因 | 决策合计 s | 单步决策 P95 s | 启动与运行 s |",
        "| --- | --- | ---: | --- | ---: | --- | ---: | ---: | ---: |"])
    for r in episodes:
        trip="是" if actual_return(r) else "否"
        distance=number(r,"path_length_m")
        more.append(f"| {LABELS[r['layout']][0]} | {r['arm']} | "
                    f"{'缺测' if distance is None else f'{distance:.1f}'} | {trip} | {r.get('collisions') or '缺测'} | "
                    f"{r.get('stop_reason') or '缺测'} | {seconds(r,'charged_decision_wall_s')} | "
                    f"{seconds(r,'decision_wall_p95_s')} | {seconds(r,'bootstrap_and_task_wall_s')} |")
    more.extend(["","计时来自本次并行仿真。逐次保护事件见[原始保护记录](../results/formal/guard_events.csv)。",""])
    (out/"TABLES.md").write_text("\n".join(more))
    (out/"CAPTION.md").write_text(
        "# Figure caption\n\nCorrect coverage of reachable free space (top) and joint reconstruction quality "
        "(bottom) in four prospectively registered simulated floors. Curves terminate at the last actually "
        "executed action. The horizontal guide marks the 95% coverage target and the vertical guide marks action 96. "
        "J equals correct free-space cover multiplied by full-scene surface F1 at 5 cm. Symbols are native "
        "mesh evaluations at captured action checkpoints. Connecting segments guide the eye and squares mark "
        "each measured terminal map at its actual clock.\n")


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results",type=Path,default=ROOT/"results/formal")
    args=parser.parse_args()
    ep, checkpoints, curves=[csv_rows(args.results/name) for name in ("episodes.csv","checkpoints.csv","coverage_curve.csv")]
    paper=ROOT/"paper"
    paper.mkdir(exist_ok=True)
    layouts=read(ROOT/"PROTOCOL.json")["layouts"]
    write_results(ep,checkpoints,paper,layouts)
    plot(curves,checkpoints,paper,layouts)
    print("Paper artifacts based on",sha(args.results/"summary.json"))


if __name__=="__main__":
    main()
