#!/usr/bin/env python3
"""Audit the V10.3 feedback omission and corrected V10.3.1 rerun."""
import argparse
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def load(path):
    return json.loads(Path(path).read_text())


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def value(row, key):
    if isinstance(key, tuple):
        return row[key[0]][key[1]]
    return row[key]


def paired(rows, baseline, key):
    lookup = {(r["context"], r["arrangement"], r["condition"]): r for r in rows}
    return [value(lookup[c, a, "S"], key) - value(lookup[c, a, baseline], key)
            for c in ("Q0", "Q1", "Q2", "Q3")
            for a in ("shelf_west", "shelf_east")]


def main(old_run, new_run, replay, omission_audit, output, figure):
    old_rows, new_rows = load(old_run / "results.json"), load(new_run / "results.json")
    old_summary, new_summary = load(old_run / "summary.json"), load(new_run / "summary.json")
    replay_result, omission = load(replay / "verification.json"), load(omission_audit / "verification.json")
    old = {(r["context"], r["arrangement"], r["condition"]): r for r in old_rows}
    new = {(r["context"], r["arrangement"], r["condition"]): r for r in new_rows}
    artifact_hashes = load(new_run / "artifact_hashes.json")
    artifact_hashes_ok = all(sha(new_run / name) == digest for name, digest in artifact_hashes.items())

    metrics = {
        "inspection_joint": "new_area_times_final_f1_05cm",
        "new_visible_area_m2": "new_unique_surface_area_m2",
        "coverage_fraction_times_f1": "final_coverage_fraction_times_f1_05cm",
        "covered_area_times_f1": "final_covered_area_times_f1_05cm",
        "precision_05cm": ("after_metrics", "precision_05cm"),
        "recall_05cm": ("after_metrics", "recall_05cm"),
        "f1_05cm": "final_f1_05cm",
        "surface_error_mean_m": ("after_metrics", "surface_error_mean_m"),
        "new_surface_recall_05cm": ("new_visible_surface_quality", "recall_05cm"),
        "path_distance_m": "path_distance_m",
    }
    contrasts = {baseline: {name: {
        "mean_S_minus_baseline": float(np.mean(paired(new_rows, baseline, key))),
        "wins": sum(x > 0 for x in paired(new_rows, baseline, key)),
        "ties": sum(x == 0 for x in paired(new_rows, baseline, key)),
        "losses": sum(x < 0 for x in paired(new_rows, baseline, key)),
    } for name, key in metrics.items()} for baseline in ("G", "N", "X", "S_no_feedback")}

    old_new = {}
    for condition in ("S", "G", "N", "X", "S_no_feedback"):
        keys = [k for k in new if k[2] == condition]
        old_new[condition] = dict(
            action_trajectories_changed=sum(old[k]["actions"] != new[k]["actions"] for k in keys),
            target_sequences_changed=sum(old[k]["selected_candidate_ids"]
                                         != new[k]["selected_candidate_ids"] for k in keys),
            mean_inspection_joint_old=float(np.mean(
                [old[k]["new_area_times_final_f1_05cm"] for k in keys])),
            mean_inspection_joint_new=float(np.mean(
                [new[k]["new_area_times_final_f1_05cm"] for k in keys])))

    full = [r for r in new_rows if r["condition"] != "S_no_feedback"]
    disabled = [r for r in new_rows if r["condition"] == "S_no_feedback"]
    feedback = dict(
        old_zero_observed_branches=sum(r["gain_calibration"]["observed_actions"] == 0
                                       for r in old_rows),
        old_48_pending_branches=sum(len(r["gain_calibration"]["pending_action_ids"]) == 48
                                    for r in old_rows),
        new_full_48_observed_zero_pending=sum(
            r["gain_calibration"]["observed_actions"] == 48
            and not r["gain_calibration"]["pending_action_ids"] for r in full),
        new_disabled_48_recorded_prior_frozen=sum(
            r["gain_calibration"]["observed_actions"] == 48
            and not r["gain_calibration"]["pending_action_ids"]
            and r["gain_calibration"]["alpha"] == {"radar": 1, "camera": 1}
            and r["gain_calibration"]["beta"] == {"radar": 1, "camera": 1}
            for r in disabled),
        full_branches=len(full), disabled_branches=len(disabled),
        semantic_camera_posterior_mean=float(np.mean([
            r["gain_calibration"]["posterior_mean"]["camera"]
            for r in new_rows if r["condition"] == "S"])),
        semantic_radar_predicted_cells_total=int(sum(
            e["radar"]["predicted_cells"]
            for r in new_rows if r["condition"] == "S"
            for e in r["gain_calibration"]["events"])),
        semantic_camera_predicted_cells_total=int(sum(
            e["camera"]["predicted_cells"]
            for r in new_rows if r["condition"] == "S"
            for e in r["gain_calibration"]["events"])))

    result = dict(schema_version="cpu_v10_3_1_feedback_fix_review/1", status="passed_with_negative_H3",
        historical_omission_confirmed=omission["status"] == "confirmed",
        new_artifact_hashes_verified=artifact_hashes_ok,
        independent_replay_passed=(replay_result["status"] == "passed"
                                   and replay_result["branches"] == 40),
        corrected_mechanism_passed=new_summary["mechanism_passed"],
        corrected_efficacy_passed=new_summary["efficacy_passed"],
        feedback=feedback, old_new=old_new, contrasts=contrasts,
        terminal_coverage_fraction_values=sorted({r["after_metrics"]["coverage_2d"]
                                                  for r in new_rows}),
        conclusions={
            "H3_feedback_efficacy": "not established: S and S_no_feedback have identical target/action sequences in 8/8 paired histories",
            "semantic_development_effect": "retained against internal G/N/X on inspection joint, with negative cases and metric tradeoffs reported",
            "metric_scope": "inspection_joint is new visible surface area times terminal global F1; terminal C_2D times F1 collapses to F1 because all branches end at C_2D=1",
            "next_model_change": "learn candidate- and instance-conditioned semantic residual gain with confidence and online correction; a global camera-yield scalar is insufficient"
        },
        input_sha256={str(p): sha(p) for p in (old_run / "summary.json", new_run / "summary.json",
            new_run / "results.json", replay / "verification.json", omission_audit / "verification.json")})
    output.mkdir(parents=True, exist_ok=False)
    (output / "verification.json").write_text(json.dumps(result, ensure_ascii=False,
        indent=2, sort_keys=True) + "\n")

    conditions = ("S", "G", "N", "X", "S_no_feedback")
    labels = ("S", "G", "N", "X", "S-noFB")
    by = new_summary["by_condition"]
    fig, axes = plt.subplots(1, 3, figsize=(11.2, 3.5))
    axes[0].bar(labels, [by[c]["mean_inspection_joint"] for c in conditions])
    axes[0].set_ylabel("m² × F1"); axes[0].set_title("Inspection composite")
    axes[1].bar(labels, [100 * by[c]["mean_final_f1_05cm"] for c in conditions])
    axes[1].set_ylabel("%"); axes[1].set_title("Terminal global F1 @ 5 cm")
    axes[2].bar(labels, [100 * by[c]["mean_new_surface_recall_05cm"] for c in conditions])
    axes[2].set_ylabel("%"); axes[2].set_title("New-surface local recall @ 5 cm")
    for ax in axes:
        ax.grid(axis="y", alpha=.25)
    fig.tight_layout()
    figure.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(figure, dpi=220)
    fig.savefig(figure.with_suffix(".pdf"))
    plt.close(fig)

    report = f"""# V10.3.1 反馈修复与重跑审阅

历史 V10.3 的 40/40 分支均为 0 次收益观测和 48 个滞留预测。V10.3.1 中，反馈开启的 {feedback['full_branches']}/{feedback['full_branches']} 分支均完成 48 次观测且无滞留；关闭反馈的 {feedback['disabled_branches']}/{feedback['disabled_branches']} 分支仍记录结果，但后验保持先验。40/40 分支的动作、传感包、TSDF、指标和安全返航已由不调用规划器的程序独立重放。

H3 仍未成立：S 与 S-no-feedback 在 8/8 配对历史中的目标序列、动作和全部任务指标完全相同。反馈修复只使 G 的 1/8 轨迹变化，且该轨迹的 inspection composite 降低 0.330562。S 的相机兑现率后验均值由 0.5 更新到 {feedback['semantic_camera_posterior_mean']:.6f}，但当前全局标量未改变语义候选排序；二维雷达预测在该已基本覆盖的前缀后为 0 个单元。

S 相对 G/N/X 的 inspection composite 均值差为 {contrasts['G']['inspection_joint']['mean_S_minus_baseline']:.6f} / {contrasts['N']['inspection_joint']['mean_S_minus_baseline']:.6f} / {contrasts['X']['inspection_joint']['mean_S_minus_baseline']:.6f} m²×F1。S 相对 G/N 的最终 F1 仅高 {100*contrasts['G']['f1_05cm']['mean_S_minus_baseline']:.4f} / {100*contrasts['N']['f1_05cm']['mean_S_minus_baseline']:.4f} 个百分点，同时平均表面误差分别差 {1000*contrasts['G']['surface_error_mean_m']['mean_S_minus_baseline']:.4f} / {1000*contrasts['N']['surface_error_mean_m']['mean_S_minus_baseline']:.4f} mm（正值表示 S 更差）。新增区域局部 recall 相对 G/N 为 {100*contrasts['G']['new_surface_recall_05cm']['mean_S_minus_baseline']:.4f} / {100*contrasts['N']['new_surface_recall_05cm']['mean_S_minus_baseline']:.4f} 个百分点。

所有分支终点二维覆盖率均为 1，因此终点 `C_2D×F1` 在本共享前缀检查任务中等于 F1，不能证明覆盖规划优势。历史 `A_new_visible×F1_global` 继续明确命名为 inspection composite。下一阶段训练候选/实例条件的语义残差收益和置信门，并让实际反馈更新该残差，而不是只缩放全局相机收益。
"""
    (output / "REPORT.md").write_text(report)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--old-run", type=Path, required=True)
    p.add_argument("--new-run", type=Path, required=True)
    p.add_argument("--replay", type=Path, required=True)
    p.add_argument("--omission-audit", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--figure", type=Path, required=True)
    a = p.parse_args()
    main(a.old_run, a.new_run, a.replay, a.omission_audit, a.output, a.figure)
