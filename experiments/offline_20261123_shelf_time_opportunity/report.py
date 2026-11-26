#!/usr/bin/env python3
"""Build the development opportunity table, figure and submission briefing."""

import argparse
import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np

from study import ROOT, read, sha, verify


LABELS = {"increment_open":"开发：开敞","increment_screen":"开发：遮挡","shelf_01_screen":"已知场景复核：遮挡"}
FIGURE_LABELS = ("Development, open","Development, screened","Known diagnostic, screened")


def value(number, signed=False):
    if number is None:
        return "—"
    return f"{number:+.4f}" if signed else f"{number:.4f}"


def table(summary):
    rows=["| 摆位 | 第 96 步已测／冻结 | D | 原 S | 已测最高 | 最高−D | 最高路线 |",
          "| --- | ---: | ---: | ---: | ---: | ---: | --- |"]
    for name, details in summary["placements"].items():
        a=details["clocks"]["paid_96"]
        names=", ".join(a["best_route_ids"]) or "—"
        rows.append(f"| {LABELS[name]} | {a['n_measured']}/{a['n_frozen']} | {value(a['D_m2'])} | "
                    f"{value(a['S_m2'])} | {value(a['best_measured_m2'])} | {value(a['best_minus_D_m2'],True)} | {names} |")
    return "\n".join(rows)


def late_followthrough(result, summary, paper):
    records=[]
    for name,case in summary["placements"].items():
        earlier=case["clocks"]["paid_144"]
        terminal=case["clocks"]["terminal"]
        direct=case["D_route_id"]
        leader=earlier["best_route_ids"][0] if earlier["best_minus_D_m2"] and earlier["best_route_ids"] else direct
        with np.load(result/f"{name}_masks.npz",allow_pickle=False) as masks:
            weight=masks["reference_shelf_weights_m2"]
            view=masks[f"{leader}_paid_144_shelf"]
            early=view & ~masks[f"{direct}_paid_144_shelf"] & ~masks["fork_shelf"]
            lead_return,d_return=masks[f"{leader}_terminal_shelf"],masks[f"{direct}_terminal_shelf"]
            retained=early & lead_return & ~d_return
            return_delta=float(weight[lead_return & ~d_return].sum()-weight[d_return & ~lead_return].sum())
            records.append({"placement":name,"A144_leader":leader,
                "A144_leader_minus_D_m2":earlier["best_minus_D_m2"],
                "same_A144_route_at_return_minus_D_m2":return_delta,
                "A144_exclusive_m2":float(weight[early].sum()),
                "A144_exclusive_still_exclusive_at_return_m2":float(weight[retained].sum()),
                "return_best_route_ids":"/".join(terminal["best_route_ids"]),
                "return_best_minus_D_m2":terminal["best_minus_D_m2"]})
    with (paper/"late_followthrough.csv").open("w",newline="") as stream:
        writer=csv.DictWriter(stream,fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)
    return records


def plot(summary, out):
    matplotlib.rcParams.update({"font.size":8.5,"svg.fonttype":"none","pdf.fonttype":42})
    fig,axs=plt.subplots(1,3,figsize=(9.2,2.8),sharex=True,sharey=True,layout="constrained")
    keys=("paid_16","paid_24","paid_48","paid_96","paid_144","paid_192")
    times=np.array([16,24,48,96,144,192])
    for ax,(name,case),label in zip(axs,summary["placements"].items(),FIGURE_LABELS,strict=True):
        outcomes=[case["clocks"][key] for key in keys]
        best=[r["best_minus_D_m2"] if r["best_minus_D_m2"] is not None else np.nan for r in outcomes]
        old=[r["S_m2"]-r["D_m2"] if r["S_m2"] is not None and r["D_m2"] is not None else np.nan for r in outcomes]
        ax.axhline(0,color="#475569",lw=.75,zorder=0)
        ax.axvline(96,color="#a4b0bd",lw=.8,ls="--",zorder=0)
        ax.scatter(times,best,s=23,marker="D",c="#126c6a",label="Best measured route")
        ax.scatter(times,old,s=23,marker="o",edgecolors="#ae4d24",facecolors="none",label="Original S")
        for x,r,y in zip(times,outcomes,best,strict=True):
            if np.isfinite(y) and r["n_measured"]<r["n_frozen"]:
                ax.annotate(f"{r['n_measured']}/{r['n_frozen']}",(x,y),xytext=(0,6),textcoords="offset points",
                            color="#475569",fontsize=6,ha="center")
        ax.set_title(label,loc="left",pad=8,fontsize=9)
        ax.set_xticks((16,24,48,96,144,192))
        ax.set_xlabel("Paid task action")
        ax.grid(axis="y",color="#edf1f5",lw=.6)
        ax.spines[["top","right"]].set_visible(False)
    axs[0].set_ylabel("Net shelf area relative to D (m²)")
    fig.legend(handles=[Line2D([],[],color="#126c6a",ls="",marker="D",label="Best measured route at each clock"),
                       Line2D([],[],color="#ae4d24",ls="",marker="o",mfc="none",label="Original S")],
               loc="outside upper center",ncol=2,frameon=False)
    for ext in ("pdf","svg","png"):
        fig.savefig(out/f"development_route_opportunities.{ext}",dpi=220)
    plt.close(fig)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage",type=Path,default=ROOT/"recovery/results")
    args=parser.parse_args()
    summary=read(args.stage/"summary.json")
    verify({str(args.stage/k):v for k,v in summary["result_files_sha256"].items()})
    prior=read(ROOT.parent/"shelf_occlusion_pairs_20261122/results/formal/summary.json")["statistics"]
    old_sg=prior["screen_S_minus_G_96_m2"]
    staged=ROOT/"paper"
    staged.mkdir(exist_ok=True)
    gains=len(summary["positive_distinct_placements"])
    late=late_followthrough(args.stage,summary,staged)
    reason={"stage_II":"按门槛进入时段评分开发。","stop_stage_I":"按门槛结束本轮，不追加完整探索。",
            "review_technical_failure":"技术失败需要完成复核，暂未作停走判断。"}[summary["gate"]]
    follow_rows=[f"| {LABELS[p]} | {x['route_id'] or '—'} | {value(x['terminal_minus_D_m2'],True)} | "
                 f"{value(x['early_exclusive_m2'])} | {value(x['still_exclusive_at_return_m2'])} |"
                 for p,case in summary["placements"].items() for x in (case["A96_best_followed_to_return"],)
                 if case["clocks"]["paid_96"]["any_measured_above_D"]]
    comparison="\n".join([
        "# 路线机会与首选对照",
        "",
        "面积单位为 m²。净新增按共同初始融合网格与固定参考点的 5 cm 匹配计算。最高值来自冻结后实际执行并测到该时点的路线，路线身份按原始候选 ID 给出。",
        "",
        table(summary),
        "",
        "## 第 96 步领先路线的返航结果",
        "",
        *(["| 摆位 | 第 96 步路线 | 实际返航净面积差 | 第 96 步独占面积 | 返航仍独占 |",
           "| --- | --- | ---: | ---: | ---: |",*follow_rows] if follow_rows else
           ["三个摆位的第 96 步首名均为 D，无领先备选的独占面积需要跟踪。"]),
        "",
        "## 较晚时钟的配对追踪",
        "",
        "| 摆位 | 第 144 步首名−D | 同一首名路线返航−D | 返航时所有路线最高−D |",
        "| --- | ---: | ---: | ---: |",
        *[f"| {LABELS[x['placement']]} | {value(x['A144_leader_minus_D_m2'],True)} | "
          f"{value(x['same_A144_route_at_return_minus_D_m2'],True)} | "
          f"{value(x['return_best_minus_D_m2'],True)} |" for x in late],
        "",
        "最后一列允许在同一返航时钟比较不同路线。第 144 步首名和返航首名可能不同。独占面积见[逐路线追踪](late_followthrough.csv)。",
        "",
        f"第 96 步有 {gains}/3 个摆位达到数值改善门槛。{reason}",
        "",
    ])
    (staged/"OPPORTUNITY_TABLE.md").write_text(comparison)
    paragraphs=[
        "# 货架路线机会与投稿决策",
        "",
        f"开发补测固定三个已知摆位，每个摆位执行 D、原 S 和两条与其等成本的几何备选。第 96 步有 **{gains}/3** 个摆位的已测路线优于 D。{reason}",
        "",
        table(summary),
        "",
        "**路线机会。** 旧六个遮挡摆位中，原 S 在第 96 步相对 D 平均 −0.24130 m²。只在两条原轨迹间事后择优，平均上限为 +0.00830 m²。旧 13 条去程样本在共同第 16 步已全部复核，两个摆位的 D 均达到已测最高值。",
        "",
        f"**类别先验。** 旧的类别与中性几何评分配对在有遮挡条件下均值为 {old_sg['mean']:+.3f} m² "
        f"（{old_sg['win']} 胜、{old_sg['tie']} 平、{old_sg['loss']} 负）。本阶段冻结路线用于验证可兑现机会，"
        "原 S 与 D 的差值包含评分规则差异。新机制的类别独立贡献需要在同一时间评分下切换先验。",
        "",
        "**返航和范围。** 已测备选在个别较晚时钟可以领先，何时领先及同一路线返航能保留多少见配对表。统计以摆位为单位，保留保护中断和真实缺测。三个摆位均为已知仿真场景，采用原仿真传感器、建图器和保护，类别为可见仿真标签。",
        "",
        "[路线与返航配对表](OPPORTUNITY_TABLE.md) · [逐时钟图](development_route_opportunities.pdf)",
        "",
    ]
    (staged/"BRIEF.md").write_text("\n".join(paragraphs))
    plot(summary,staged)
    print("Development paper outputs verified against",sha(args.stage/"summary.json"))


if __name__=="__main__":
    main()
