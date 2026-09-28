#!/usr/bin/env python3
"""Plot the frozen V11.1 independent efficacy result and stress boundary."""

import argparse
import hashlib
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def load(path: Path):
    return json.loads(path.read_text())


def sha256(path: Path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main(efficacy: Path, analysis: Path, stress: Path, destination: Path):
    efficacy_data = load(efficacy)
    analysis_data = load(analysis)
    stress_data = load(stress)
    if efficacy_data["status"] != "passed" or not all(efficacy_data["gates"].values()):
        raise ValueError("the complete passed efficacy result is required")
    if analysis_data["status"] != "passed" or stress_data["status"] != "passed":
        raise ValueError("the paired analysis and completed stress result are required")

    methods = (
        "learned_semantic_feedback",
        "learned_geometry",
        "fixed_semantic",
        "fixed_geometry",
        "learned_swapped",
    )
    labels = ("Semantic\nMLP", "Geometry\nMLP", "Fixed\nsemantic", "Fixed\ngeometry", "Swapped\nsemantic")
    colors = ("#176B87", "#4C78A8", "#F2A541", "#9AA0A6", "#C65D57")
    means = [efficacy_data["by_condition"][method]["mean_joint_gain"] for method in methods]

    comparisons = ("learned_geometry", "fixed_semantic", "fixed_geometry", "learned_swapped")
    comparison_labels = ("Geometry MLP", "Fixed semantic", "Fixed geometry", "Swapped semantic")
    contexts = tuple(f"F{i:02d}" for i in range(8))

    figure, axes = plt.subplots(1, 3, figsize=(13.2, 4.25), constrained_layout=True)

    ax = axes[0]
    bars = ax.bar(np.arange(len(methods)), means, color=colors, width=0.72)
    ax.set_xticks(np.arange(len(methods)), labels)
    ax.set_ylabel(r"Mean $\Delta(A_{2D}F_{1,5\mathrm{cm}})$")
    ax.set_title("(a) Frozen efficacy, 96 branches")
    ax.set_ylim(0, max(means) * 1.22)
    for bar, value in zip(bars, means):
        ax.text(bar.get_x() + bar.get_width() / 2, value + 0.025, f"{value:.3f}",
                ha="center", va="bottom", fontsize=8)

    ax = axes[1]
    for index, (comparison, label) in enumerate(zip(comparisons, comparison_labels)):
        row = analysis_data["comparisons"][comparison]
        values = [row["context_differences"][context]["joint_gain"] for context in contexts]
        jitter = np.linspace(-0.10, 0.10, len(values))
        ax.scatter(np.full(len(values), index) + jitter, values, color="#176B87", s=25,
                   alpha=0.82, zorder=3)
        low, high = row["bootstrap_95_percentile_interval"]
        mean = row["mean_difference"]
        ax.errorbar(index, mean, yerr=[[mean - low], [high - mean]], fmt="D",
                    color="#222222", capsize=4, markersize=5, zorder=4)
    ax.axhline(0, color="#555555", linewidth=1)
    ax.set_xticks(np.arange(len(comparisons)), comparison_labels, rotation=20, ha="right")
    ax.set_ylabel("Semantic minus baseline")
    ax.set_title("(b) Eight independent parent contexts")
    ax.text(0.02, 0.97, "dots: parent means\ndiamonds: mean and 95% grouped bootstrap CI",
            transform=ax.transAxes, ha="left", va="top", fontsize=8)

    ax = axes[2]
    stress_methods = ("learned_semantic_feedback", "learned_geometry", "learned_swapped")
    stress_labels = ("Semantic\nMLP", "Geometry\nMLP", "Swapped\nsemantic")
    stress_means = [stress_data["by_condition"][method]["mean_joint_gain"] for method in stress_methods]
    stress_bars = ax.bar(np.arange(3), stress_means,
                         color=(colors[0], colors[1], colors[4]), width=0.65)
    ax.set_xticks(np.arange(3), stress_labels)
    ax.set_ylabel(r"Mean $\Delta(A_{2D}F_{1,5\mathrm{cm}})$")
    ax.set_title("(c) Coverage stress, 40 branches")
    ax.set_ylim(min(stress_means) - 0.025, max(stress_means) + 0.025)
    for bar, value in zip(stress_bars, stress_means):
        ax.text(bar.get_x() + bar.get_width() / 2, value + 0.003, f"{value:.3f}",
                ha="center", va="bottom", fontsize=8)
    difference = stress_data["comparisons"]["semantic_vs_learned_geometry"]["mean_difference"]
    ax.text(0.5, 0.05, f"semantic - geometry = {difference:+.4f}\n0 win / 7 tie / 1 loss",
            transform=ax.transAxes, ha="center", va="bottom", fontsize=9)

    for ax in axes:
        ax.grid(axis="y", alpha=0.22)
        ax.set_axisbelow(True)
        ax.spines[["top", "right"]].set_visible(False)

    destination.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(destination.with_suffix(".png"), dpi=220)
    figure.savefig(destination.with_suffix(".pdf"))
    plt.close(figure)

    record = {
        "schema_version": "semantic_gain_v11_1_confirmation_figure/1",
        "status": "passed_efficacy_with_negative_coverage_stress_boundary",
        "source_sha256": sha256(Path(__file__)),
        "inputs": {
            str(efficacy): sha256(efficacy),
            str(analysis): sha256(analysis),
            str(stress): sha256(stress),
        },
        "efficacy_means": dict(zip(methods, means)),
        "stress_means": dict(zip(stress_methods, stress_means)),
        "sample_unit": analysis_data["sample_unit"],
        "scope": "prospectively frozen artificial two-asset mechanism; no natural-semantic, external-baseline, or robot claim",
    }
    destination.with_suffix(".json").write_text(
        json.dumps(record, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    )
    destination.with_suffix(".caption.md").write_text(
        "V11.1 独立人工机制确认。左：F00--F07 的96条 efficacy 分支主指标均值；"
        "中：语义模型相对四个对照的八个父上下文配对差，误差线为按父上下文整组重采样的95%区间；"
        "右：F08--F11覆盖压力层中语义与几何、置换语义近乎相同。结果支持基础覆盖后的三维补看收益，"
        "不支持覆盖主导阶段的语义增量。\n"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--efficacy", type=Path, required=True)
    parser.add_argument("--analysis", type=Path, required=True)
    parser.add_argument("--stress", type=Path, required=True)
    parser.add_argument("--destination", type=Path, required=True)
    arguments = parser.parse_args()
    main(arguments.efficacy, arguments.analysis, arguments.stress, arguments.destination)
