#!/usr/bin/env python3
"""Render publication-sized diagrams from documentation and saved CSVs only.

No scientific module is imported and no experiment or score is recomputed.
The four saved CSVs are checked against their existing manifest before plotting.
Existing output directories are refused. Run from any directory with Python 3.
"""
import argparse
import csv
import hashlib
import json
import math
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "docs/thesis/figures/virtual_paper_20260928"
DEFAULT_OUTPUT = ROOT / "docs/thesis/figures/virtual_method_20260928"
COLORS = {"G": "#0072B2", "S": "#D55E00", "SWAP": "#009E73", "VISTA": "#CC79A7"}
ABLATION_COLORS = {"G": COLORS["G"], "S": COLORS["S"], "swapped": COLORS["SWAP"], "swapped_no_feedback": COLORS["VISTA"]}
ABLATION_LABELS = {"G": "G", "S": "S", "swapped": "X", "swapped_no_feedback": "Xnf"}
METHODS = ("G", "S", "SWAP", "VISTA")
LABELS = {"G": "G", "S": "S", "SWAP": "SWAP-I", "VISTA": "VISTA-I"}
CONDITIONS = tuple((p, h) for p in ("P00", "P01") for h in (0, 1))


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def relative(path):
    return str(path.resolve().relative_to(ROOT))


def checked_inputs():
    paths = [
        ROOT / "docs/thesis/V38_IMPLEMENTED_METHODS_20260920.md",
        ROOT / "nso/cpu_four_modules_v35.py",
        ROOT / "nso/observation_belief_v35.py",
        ROOT / "nso/online_planner_v35.py",
        Path(__file__).resolve(),
        SOURCE / "manifest.json",
    ]
    manifest = json.loads((SOURCE / "manifest.json").read_text())
    rows = {}
    for name in ("v36_measurements.csv", "v39_measurements.csv", "v39_method_means.csv", "paired_effects.csv"):
        path = SOURCE / name
        if digest(path) != manifest["output_sha256"][name]:
            raise ValueError(f"Saved presentation source changed: {path}")
        paths.append(path)
        with path.open(newline="", encoding="utf-8") as stream:
            rows[name] = list(csv.DictReader(stream))
    old, comparison = rows["v36_measurements.csv"], rows["v39_measurements.csv"]
    expected = {(*condition, method) for condition in CONDITIONS for method in METHODS}
    lookup = {(r["parent"], int(r["hypothesis"]), r["method"]): r for r in comparison}
    if len(comparison) != 16 or set(lookup) != expected or len(old) != 8:
        raise ValueError("Require all four saved conditions and all declared methods")
    if {(r["parent"], int(r["hypothesis"]), r["method"]) for r in old} != {
        (*condition, method) for condition in CONDITIONS for method in ("G", "S")
    }:
        raise ValueError("Require complete saved G/S confirmation")
    for row in old:
        matching = lookup[(row["parent"], int(row["hypothesis"]), row["method"])]
        for metric in ("C_map", "F5", "J5"):
            if not math.isclose(float(row[metric]), float(matching[metric]), rel_tol=0, abs_tol=1e-15):
                raise ValueError("Saved confirmation/comparison parity mismatch")
    paired = rows["paired_effects.csv"]
    if len(paired) != 4 or sum(float(r["delta_J5"]) == 0 for r in paired) != 2:
        raise ValueError("Require all two positive and two zero paired effects")
    for row in rows["v39_method_means.csv"]:
        for metric in ("C_map", "F5", "J5"):
            observed = sum(float(lookup[(*condition, row["method"])][metric]) for condition in CONDITIONS) / 4
            if not math.isclose(observed, float(row[metric]), rel_tol=0, abs_tol=1e-15):
                raise ValueError("Saved mean does not match complete four-condition input")
    old_figures = ROOT / "docs/thesis/figures/v38"
    old_manifest_path = old_figures / "manifest.json"
    old_manifest = json.loads(old_manifest_path.read_text())
    online_csv = old_figures / "online_measurements.csv"
    if digest(online_csv) != old_manifest["outputs"][online_csv.name]:
        raise ValueError("Saved V38 online measurements changed")
    timeline_path = ROOT / "docs/thesis/V38_CAUSAL_TIMELINE_20260920.json"
    if digest(timeline_path) != old_manifest["source_sha256"][relative(timeline_path)]:
        raise ValueError("Saved causal timeline changed")
    paths.extend((old_manifest_path, online_csv, timeline_path))
    with online_csv.open(newline="", encoding="utf-8") as stream:
        all_online = list(csv.DictReader(stream))
    rows["v35_measurements"] = [r for r in all_online if r["cohort"] == "v35_online"]
    expected_ablation = {(h, m, t) for h in (0, 1) for m in ABLATION_COLORS for t in ("02cm", "05cm", "10cm")}
    observed_ablation = {(int(r["hypothesis"]), r["method"], r["threshold"]) for r in rows["v35_measurements"]}
    if len(rows["v35_measurements"]) != 24 or observed_ablation != expected_ablation:
        raise ValueError("Require all V35 conditions and all three saved thresholds")
    if any(r["parent"] != "P00" for r in rows["v35_measurements"]):
        raise ValueError("Unexpected development parent")
    timeline = json.loads(timeline_path.read_text())
    rows["v35_timeline"] = [s for s in timeline["series"]
                            if s["phase"] == "V35" and s["parent"] == "P00" and s["hypothesis"] == 1]
    if len(rows["v35_timeline"]) != 4 or {s["mode_in_source"] for s in rows["v35_timeline"]} != set(ABLATION_COLORS):
        raise ValueError("Require all four V35 P00/h1 timeline series")
    for series in rows["v35_timeline"]:
        if [r["observation_after_executed_action"] for r in series["rows"]] != list(range(43)):
            raise ValueError("Require complete observation indices 0 through 42")
        for key in ("source_controller", "source_trace", "source_result"):
            path = ROOT / series[key]
            if digest(path) != timeline["sources"][series[key]]["sha256"]:
                raise ValueError(f"Saved timeline input changed: {path}")
            paths.append(path)
    return rows, {relative(p): digest(p) for p in paths}


def configure():
    os.environ.setdefault("MPLBACKEND", "Agg")
    os.environ.setdefault("MPLCONFIGDIR", "/tmp/nso_virtual_method_20260928_mpl")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({
        "font.family": "Liberation Sans", "font.size": 9,
        "axes.labelsize": 9, "axes.titlesize": 10, "legend.fontsize": 9,
        "axes.linewidth": .65, "xtick.labelsize": 9, "ytick.labelsize": 9,
        "pdf.fonttype": 42, "ps.fonttype": 42, "svg.fonttype": "none",
        "svg.hashsalt": "nso-virtual-method-20260928", "savefig.facecolor": "white",
        "mathtext.fontset": "dejavusans",
    })
    return matplotlib, plt


def save(fig, output, name):
    for extension in ("pdf", "svg", "png"):
        metadata = {"Creator": "NSO saved-evidence publication renderer"} if extension == "pdf" else None
        if extension == "pdf":
            metadata.update(CreationDate=None, ModDate=None)
        fig.savefig(output / f"{name}.{extension}", dpi=220, metadata=metadata)


def draw_method(plt, output):
    from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
    fig = plt.figure(figsize=(7.05, 4.65), facecolor="white")
    ax = fig.add_axes((.018, .065, .964, .925))
    ax.set(xlim=(.5, 13.5), ylim=(.1, 7.65))
    ax.axis("off")
    ink, muted = "#253444", "#65717C"

    def arrow(points, color=ink, dashed=False):
        from matplotlib.path import Path as MplPath
        path = MplPath(points, [MplPath.MOVETO] + [MplPath.LINETO] * (len(points)-1))
        patch = FancyArrowPatch(path=path, arrowstyle="-|>", mutation_scale=9,
                                linewidth=.9, color=color, linestyle=(0, (3, 2)) if dashed else "-",
                                zorder=2)
        ax.add_patch(patch)

    def box(x, y, width, height, title, body="", edge=ink, fill="white"):
        ax.add_patch(FancyBboxPatch((x, y), width, height, boxstyle="round,pad=0.02,rounding_size=0.06",
                                  linewidth=.9, edgecolor=edge, facecolor=fill, zorder=3))
        if body:
            ax.text(x+width/2, y+height*.76, title, ha="center", va="center",
                    fontsize=9.5, fontweight="bold", color=ink, zorder=4)
            ax.text(x+width/2, y+height*.36, body, ha="center", va="center",
                    fontsize=9, linespacing=1.35, color=ink, zorder=4)
        else:
            ax.text(x+width/2, y+height/2, title, ha="center", va="center", fontsize=9, color=ink, zorder=4)

    xs, width = (1, 4.1, 7.2, 10.3), 2.7
    centers = [x + width/2 for x in xs]
    box(1, 6.67, 12, .57,
        "Public templates · safe pose graph · category–structure relation", edge=muted, fill="#F3F5F7")
    ax.plot([centers[0], centers[-1]], [6.08, 6.08], color=muted, lw=.8, ls=(0, (3, 2)), zorder=1)
    ax.plot([7, 7], [6.67, 6.08], color=muted, lw=.8, ls=(0, (3, 2)), zorder=1)
    for center in centers:
        arrow([(center, 6.08), (center, 5.48)], color=muted, dashed=True)
    labels = (
        ("OV-SDF", "Map / class ledger\nSupported prior", COLORS["S"], "#FCF1EB"),
        ("IGCR", "Measured residuals\nBelief correction", COLORS["SWAP"], "#EFF8F4"),
        ("STGHP", "Finite-budget planner\nNext atomic action", COLORS["G"], "#EDF5FA"),
        ("RPN-UQ", "Legal edge\nReturn-budget guard", muted, "#F3F5F7"),
    )
    for x, (title, body, edge, fill) in zip(xs, labels):
        box(x, 4.25, width, 1.23, title, body, edge, fill)
    for x in xs[:-1]:
        arrow([(x+width+.03, 4.86), (x+3.1-.03, 4.86)])

    # Only already executed measurements reach the online ledger and correction.
    arrow([(centers[0], 2.85), (centers[0], 4.22)])
    arrow([(centers[0], 3.61), (centers[1], 3.61), (centers[1], 4.22)])
    ax.plot(centers[0], 3.61, marker="o", ms=2.4, color=ink)
    box(xs[0], 1.5, width, 1.35, "Paid observation", "RGB-D · scan · pose", fill="#FAFBFC")
    box(xs[1], 1.5, width, 1.35, "Measured mapping", "CPU TSDF\n2D occupancy", fill="#FAFBFC")
    arrow([(xs[0]+width+.03, 2.18), (xs[1]-.03, 2.18)])

    # Reference data enter only this sealed-output evaluation branch.
    ax.add_patch(FancyBboxPatch((6.97, 1.24), 3.16, 2.80,
                               boxstyle="round,pad=0.02,rounding_size=0.08",
                               edgecolor="#9A9FA5", facecolor="#FBFBFB", lw=.8,
                               linestyle=(0, (3, 2)), zorder=0))
    ax.text(centers[2], 3.91, "Offline only", ha="center", va="center", fontsize=9, color=muted)
    box(xs[2], 3.14, width, .5, "Reference geometry", edge=muted, fill="#F0F1F2")
    box(xs[2], 1.5, width, 1.35, "Evaluation", "Sealed map + mesh\n" + r"$C_{\rm map}$, $F_1$, $J_5$", fill="white")
    arrow([(centers[2], 3.13), (centers[2], 2.88)], color=muted, dashed=True)
    arrow([(xs[1]+width+.03, 2.18), (xs[2]-.03, 2.18)])

    box(xs[3], 1.5, width, 1.35, "Execute", "1 m forward\n90° turn", fill="#FAFBFC")
    arrow([(centers[3], 4.22), (centers[3], 2.88)])
    arrow([(centers[3], 1.48), (centers[3], .72), (centers[0], .72), (centers[0], 1.47)])
    ax.text(7, .91, "Next paid observation", fontsize=9, ha="center", va="center", color=ink,
            bbox={"facecolor": "white", "edgecolor": "none", "pad": 1.0})
    fig.text(.5, .035, "CPU observation planning · measured geometry feedback · budgeted execution",
             ha="center", va="center", fontsize=9, color=muted)
    save(fig, output, "method_flow")
    plt.close(fig)


def draw_publication(plt, output, rows):
    from matplotlib.lines import Line2D
    lookup = {(r["parent"], int(r["hypothesis"]), r["method"]): r for r in rows["v39_measurements.csv"]}
    means = {r["method"]: r for r in rows["v39_method_means.csv"]}
    fig, axs = plt.subplots(2, 2, figsize=(7.05, 5.35), gridspec_kw={"height_ratios": (1.14, 1)})
    fig.subplots_adjust(left=.09, right=.982, bottom=.09, top=.885, wspace=.31, hspace=.51)
    labels = [f"{p}\nh{h}" for p, h in CONDITIONS]
    markers = {"G": "o", "S": "s", "SWAP": "^", "VISTA": "D"}

    def style(ax, ylabel):
        ax.spines[["top", "right"]].set_visible(False)
        ax.set_axisbelow(True)
        ax.grid(axis="y", color="#E3E6E9", linewidth=.6)
        ax.set(xlim=(-.45, 3.45), ylim=(0, 1.025), ylabel=ylabel)
        ax.set_yticks((0, .25, .5, .75, 1))
        ax.set_xticks(range(4), labels)
        ax.tick_params(axis="x", length=0, pad=4)

    ax = axs[0, 0]
    style(ax, r"Joint score $J_5$")
    ax.set_title("A  Category-information effect", loc="left", pad=8)
    for index, condition in enumerate(CONDITIONS):
        g, s = (float(lookup[(*condition, m)]["J5"]) for m in ("G", "S"))
        ax.plot((index-.10, index+.10), (g, s), color="#87929C", lw=1.0)
        for method, offset, value in (("G", -.10, g), ("S", .10, s)):
            ax.scatter(index+offset, value, s=29, color=COLORS[method], marker=markers[method],
                       edgecolor="white", linewidth=.45, zorder=3)
        ax.text(index, max(g, s)+.07, "0" if s == g else f"+{s-g:.3f}",
                ha="center", va="bottom", fontsize=9, color="#354250")
    ax.text(.02, .04, "Two gains · two ties", transform=ax.transAxes, fontsize=9, color="#53616E")

    ax = axs[0, 1]
    style(ax, r"Joint score $J_5$")
    ax.set_title("B  CPU mechanism adaptations", loc="left", pad=8)
    offsets = {"G": -.21, "S": -.07, "SWAP": .07, "VISTA": .21}
    for method in METHODS:
        ax.scatter([i+offsets[method] for i in range(4)],
                   [float(lookup[(*c, method)]["J5"]) for c in CONDITIONS],
                   s=26, color=COLORS[method], marker=markers[method], edgecolor="white", linewidth=.45, zorder=3)

    for ax, metric, title, ylabel in (
        (axs[1, 0], "C_map", "C  Coverage", r"Coverage $C_{\rm map}$"),
        (axs[1, 1], "F5", "D  Surface reconstruction", r"Surface $F_1$ at 5 cm"),
    ):
        ax.spines[["top", "right"]].set_visible(False)
        ax.set_axisbelow(True)
        ax.grid(axis="y", color="#E3E6E9", linewidth=.6)
        ax.set(ylim=(0, 1.06), ylabel=ylabel, xlim=(-.6, 3.6))
        ax.set_yticks((0, .25, .5, .75, 1))
        ax.set_xticks(range(4), [LABELS[m] for m in METHODS])
        ax.tick_params(axis="x", length=0, pad=6)
        ax.set_title(title, loc="left", pad=8)
        for index, method in enumerate(METHODS):
            values = [float(lookup[(*c, method)][metric]) for c in CONDITIONS]
            mean = float(means[method][metric])
            ax.bar(index, mean, width=.54, color=COLORS[method], alpha=.16, linewidth=.7,
                   edgecolor=COLORS[method], zorder=2)
            ax.scatter([index+d for d in (-.17, -.055, .055, .17)], values, s=18,
                       color=COLORS[method], marker=markers[method], edgecolor="white", linewidth=.35, zorder=3)
            ax.plot((index-.28, index+.28), (mean, mean), color=COLORS[method], lw=1.4, zorder=4)
            ax.text(index, max(values)+.03, f"{mean:.3f}", ha="center", va="bottom", fontsize=9, color="#354250")
    handles = [Line2D([0], [0], marker=markers[m], color="none", markerfacecolor=COLORS[m],
                      markeredgecolor="white", markersize=6, label=LABELS[m]) for m in METHODS]
    fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(.53, .995), ncol=4,
               frameon=False, handletextpad=.3, columnspacing=1.8)
    save(fig, output, "publication_main")
    plt.close(fig)


def draw_ablation(plt, output, rows):
    from matplotlib.lines import Line2D
    lookup = {(int(r["hypothesis"]), r["method"], r["threshold"]): float(r["joint"])
              for r in rows["v35_measurements"]}
    markers = {"G": "o", "S": "s", "swapped": "^", "swapped_no_feedback": "D"}
    fig, axs = plt.subplots(1, 2, figsize=(7.05, 3.15))
    fig.subplots_adjust(left=.09, right=.98, bottom=.24, top=.77, wspace=.37)
    for ax in axs:
        ax.spines[["top", "right"]].set_visible(False)
        ax.set_axisbelow(True)
        ax.grid(axis="y", color="#E3E6E9", linewidth=.6)
    for index, method in enumerate(ABLATION_COLORS):
        axs[0].scatter([h+(index-1.5)*.10 for h in (0, 1)],
                       [lookup[h, method, "05cm"] for h in (0, 1)], s=30,
                       color=ABLATION_COLORS[method], marker=markers[method],
                       edgecolor="white", linewidth=.4, zorder=3)
    axs[0].set(xlim=(-.45, 1.45), ylim=(0, 1), ylabel=r"Joint score $J_5$")
    axs[0].set_yticks((0, .25, .5, .75, 1))
    axs[0].set_xticks((0, 1), ("P00 / h0", "P00 / h1"))
    axs[0].tick_params(axis="x", length=0, pad=6)
    axs[0].set_title("A  Complete development comparison", loc="left", pad=10)
    for h, marker, color, line in ((0, "o", "#4C5967", "-"), (1, "^", "#9098A0", "--")):
        values = [lookup[h, "swapped", t] - lookup[h, "swapped_no_feedback", t]
                  for t in ("02cm", "05cm", "10cm")]
        axs[1].plot(range(3), values, color=color, marker=marker, linestyle=line,
                    markersize=4.8, lw=1.2, label=f"h{h}")
        if h == 1:
            if not values[0] < 0:
                raise ValueError("The saved strict-threshold negative effect disappeared")
            axs[1].annotate(f"{values[0]:+.4f}", (0, values[0]), xytext=(.37, -.009),
                            fontsize=9, color="#354250", arrowprops={"arrowstyle": "-", "lw": .6})
    axs[1].axhline(0, color="#66717D", linewidth=.7)
    axs[1].set(xlim=(-.14, 2.14), ylim=(-.012, .028), ylabel=r"Correction effect $J_X-J_{Xnf}$")
    axs[1].set_yticks((-.01, 0, .01, .02))
    axs[1].set_xticks(range(3), ("2 cm", "5 cm\n(primary)", "10 cm"))
    axs[1].tick_params(axis="x", length=0, pad=6)
    axs[1].set_title("B  Correction across thresholds", loc="left", pad=10)
    axs[1].legend(frameon=False, loc="upper left", ncol=2, handlelength=1.7, columnspacing=1.2)
    handles = [Line2D([0], [0], marker=markers[m], color="none", markerfacecolor=ABLATION_COLORS[m],
                      markeredgecolor="white", markersize=6, label=ABLATION_LABELS[m]) for m in ABLATION_COLORS]
    fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(.5, .993), ncol=4,
               frameon=False, handletextpad=.3, columnspacing=2)
    save(fig, output, "publication_ablation")
    plt.close(fig)


def draw_timeline(plt, output, rows):
    fig, axs = plt.subplots(1, 2, figsize=(7.05, 3.05), sharey=True)
    fig.subplots_adjust(left=.09, right=.98, bottom=.21, top=.76, wspace=.20)
    series = {s["mode_in_source"]: s["rows"] for s in rows["v35_timeline"]}
    for ax, modes, title, event in (
        (axs[0], ("G", "S"), "A  Early category-informed choice", 18),
        (axs[1], ("swapped", "swapped_no_feedback"), "B  Correcting a misleading prior", 28),
    ):
        ax.spines[["top", "right"]].set_visible(False)
        ax.set_axisbelow(True)
        ax.grid(axis="y", color="#E3E6E9", linewidth=.6)
        for method in modes:
            values = series[method]
            ax.step([r["observation_after_executed_action"] for r in values],
                    [r["posterior_h1_after_observation"] for r in values], where="post",
                    color=ABLATION_COLORS[method], lw=1.5,
                    linestyle="--" if method in ("G", "swapped_no_feedback") else "-",
                    label=ABLATION_LABELS[method])
        ax.set(xlim=(0, 42), ylim=(-.04, 1.04), xlabel="Executed action count")
        ax.set_xticks((0, 10, 20, 30, 42))
        ax.set_yticks((0, .25, .5, .75, 1))
        ax.set_title(title, loc="left", pad=10)
        ax.axvline(event, color="#808A94", linewidth=.7, linestyle=":")
        ax.text(event+1.1, .36, f"Observe {event}\nAct {event+1}", fontsize=9,
                va="center", color="#53616E")
    axs[0].set_ylabel(r"Belief weight $p(h=1)$")
    handles, labels = [], []
    for ax in axs:
        h, lab = ax.get_legend_handles_labels()
        handles.extend(h)
        labels.extend(lab)
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(.5, .992), ncol=4,
               frameon=False, handlelength=2, handletextpad=.5, columnspacing=2)
    save(fig, output, "publication_timeline")
    plt.close(fig)


def run(output):
    output = output.resolve()
    if output.exists():
        raise FileExistsError(f"Output directory already exists: {output}")
    rows, input_sha256 = checked_inputs()
    output.mkdir(parents=True)
    matplotlib, plt = configure()
    draw_method(plt, output)
    draw_publication(plt, output, rows)
    draw_ablation(plt, output, rows)
    draw_timeline(plt, output, rows)
    caption = """# Publication-sized method and evidence figures

## Method flow / 方法信息流

**English.** CPU implementation of category-conditioned active observation.
Public templates, a common safe pose graph and the category–structure relation
support the four interfaces. Only already executed RGB-D, scan and pose packets
enter the online ledger and measured belief correction. The finite-budget
planner chooses the next atomic action; a legal-edge and return-budget guard
precedes execution. Actual depth and scan data build the TSDF and occupancy map;
public template surfaces are not fused into the measured mesh. Reference geometry
enters only the offline evaluator after outputs have been sealed. Dashed upper
arrows denote public knowledge, solid arrows executed information and control.
This diagram describes the CPU atomic-action controller used in the controlled
virtual experiments. The four interfaces separate observation representation,
belief correction, action planning and execution safeguards.

**中文。** 类别条件主动观测的 CPU 信息流。四模块共享公开模板、安全姿态图及类别—结构关系，
仅使用已执行动作取得的 RGB-D、扫描和位姿更新账本与实测构型权重。有限预算规划器选择
下一原子动作，经图边与返航预算检查后执行。实际深度与扫描构建 TSDF 和占据地图，
公开模板不补入预测网格；参考几何仅进入封存后的离线评价。上方虚线表示公开知识，
实线表示实际观测与控制信息。本图对应受控虚拟实验所用的 CPU 原子动作控制器，
四个接口分别承担观测表示、信念纠正、动作规划与执行约束检查。

## Publication main evidence / 主结果

**English.** Saved controlled virtual results, replotted at publication width.
(A) All four G/S paired conditions from two seen parent layouts; labels show
absolute differences in J5 and retain both ties. (B) The same four conditions
under four CPU mechanisms. SWAP-I and VISTA-I are mechanism adaptations, not
original complete systems. (C,D) Coverage and surface F1 separately: small
markers show all four conditions and horizontal colored lines show means;
light bars extend from zero to the mean and are not uncertainty intervals.
G/S in B–D reproduce A and do not add independent samples. Mean coverage is
identical across all four methods. No confidence interval or significance claim
is made; parent layouts are seen, related design layouts.

**中文。** 按正文宽度重排的已保存虚拟结果。（A）两个已见父布局、各两个构型的完整 G/S
配对，数字为 J5 绝对差，保留两组零差。（B）共同四条件下的 CPU 机制比较；SWAP-I、VISTA-I
为机制适配，非原作者完整系统。（C、D）分别展示覆盖与表面 F1；小点保留全部条件，
彩色横线为均值，浅色柱从零延伸至均值，不表示置信区间。B–D 中的 G/S 与 A 复用相同
证据，不增加独立样本，四种方法平均覆盖完全相同。图中不作统计显著性或未见布局泛化声明。

## Misleading-prior correction / 错误先验纠错

**English.** Complete saved development results on P00. (A) G, S, X (swapped
class), and Xnf (swapped class without correction) for both configurations.
(B) X minus Xnf at all three distance thresholds. Xnf disables both measured
geometric correction and anticipated future correction, so the contrast measures
the overall correction policy. The h1 negative effect at 2 cm is retained.
Neutral line colors distinguish configurations, not methods. These development
tasks are separate from the confirmation cohort and are not pooled with it.

**中文。** P00 开发批次的完整错误先验对照。（A）两个构型下 G、S、错误类别 X 及 Xnf
的全部终点。（B）2、5、10 cm 下 X−Xnf；Xnf 同时关闭实测几何更新和未来纠错预期，
比较对象为整体纠错政策。保留 h1 在 2 cm 下的负值，中性灰线及不同符号表示构型，
不表示方法。该开发批次不与确认批次混池。

## Recorded belief timeline / 已保存信念时间线

**English.** Complete saved P00/h1 development timelines for all four methods,
including observations 0–42. (A) G/S; (B) X/Xnf. Weights at index t follow the
observation obtained after executing t paid actions and precede action t+1.
Thus observing 18 precedes action 19, and observing 28 precedes action 29;
the horizontal axis is action count, not physical time. Curves are unsmoothed
post-update steps of uncalibrated belief weights. The true h1 designation is
used only for offline interpretation. This mechanism illustration adds no
independent tasks or counterfactual terminal outcomes.

**中文。** P00/h1 开发批次四方法的完整已保存时间线，保留观测 0–42。（A）G/S；
（B）X/Xnf。索引 t 的权重在执行 t 次付费动作并消费观测后产生，随后选择动作 t+1；
因此观测 18 对应下一动作 19，观测 28 对应下一动作 29。横轴为动作计数而非秒，
曲线为未平滑、未校准的更新后构型权重，真实 h1 标识仅用于离线解释。
该图不增加独立任务或反事实终点。

All four figures are 7.05 inches wide with 9 pt body labels and external captions.
Method illustration and saved-data presentation only: zero new experiments,
sensor packets, planning calls, TSDF integrations or surface evaluations.

Reproduce: `python3 scripts/plot_virtual_paper_method_20260928.py --output NEW_DIRECTORY`.
Inputs are checked against the saved source manifest. The new manifest hashes
the implementation descriptions, plot script, saved CSVs and generated outputs.
"""
    (output / "captions.md").write_text(caption, encoding="utf-8")
    for name, expected in input_sha256.items():
        if digest(ROOT / name) != expected:
            raise ValueError(f"Input changed while rendering: {name}")
    manifest = {
        "scope": "Method illustration and saved-evidence publication layout only",
        "input_sha256": input_sha256,
        "output_sha256": {p.name: digest(p) for p in sorted(output.iterdir()) if p.is_file()},
        "figures": ["method_flow", "publication_main", "publication_ablation", "publication_timeline"], "figure_width_inches": 7.05,
        "body_font_points": 9, "matplotlib_version": matplotlib.__version__,
        "complete_conditions": [{"parent": p, "hypothesis": h} for p, h in CONDITIONS],
        "source_rows": {name: len(value) for name, value in rows.items()},
        "all_conditions_retained": True, "paired_G_S_positive": 2, "paired_G_S_ties": 2,
        "ablation_complete_configurations": 2, "ablation_saved_thresholds_cm": [2, 5, 10],
        "ablation_h1_2cm_negative_retained": True, "timeline_complete_series": 4,
        "timeline_observations_per_series": 43, "timeline_smoothed": False,
        "method_colors": {**{m: COLORS[m] for m in ("G", "S")}, "X": ABLATION_COLORS["swapped"], "Xnf": ABLATION_COLORS["swapped_no_feedback"]},
        "reuses_previous_evidence": True, "new_independent_evidence": False,
        "confidence_intervals": False, "original_system_ranking": False,
        "online_ground_truth_access": False, "learned_ANS_policy_evaluated": False,
        "new_experiments": 0, "new_sensor_packets": 0, "new_planner_calls": 0,
        "new_TSDF_integrations": 0, "new_surface_evaluations": 0,
        "output_size_limit_bytes": 2 * 1024 * 1024,
    }
    (output / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2)+"\n")
    size = sum(p.stat().st_size for p in output.iterdir() if p.is_file())
    if size >= 2 * 1024 * 1024:
        raise ValueError(f"Figure package exceeds 2 MiB: {size}")
    print(json.dumps({"output": str(output), "bytes": size, "figures": manifest["figures"]}, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    run(parser.parse_args().output)
