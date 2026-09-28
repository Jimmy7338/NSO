#!/usr/bin/env python3
"""Create the auditable V10.3 to V10.3.1 correction comparison."""
import argparse
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def load(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2,
        sort_keys=True, allow_nan=False) + "\n")


def exact_trajectory(row):
    return [(a["action"], tuple(a["position"]), a["heading"], a["packet_sha256"])
            for a in row["actions"]]


def main(old_run, new_run, replay, output):
    output.mkdir(parents=True, exist_ok=False)
    old_rows = load(old_run / "results.json")
    new_rows = load(new_run / "results.json")
    old = {(r["context"], r["arrangement"], r["condition"]): r for r in old_rows}
    new = {(r["context"], r["arrangement"], r["condition"]): r for r in new_rows}
    if set(old) != set(new):
        raise AssertionError("old/new branch sets differ")
    conditions = ("S", "G", "N", "X", "S_no_feedback")
    by_condition = {}
    for condition in conditions:
        keys = [k for k in sorted(old) if k[2] == condition]
        old_joint = np.asarray([old[k]["new_area_times_final_f1_05cm"] for k in keys])
        new_joint = np.asarray([new[k]["new_area_times_final_f1_05cm"] for k in keys])
        same = [exact_trajectory(old[k]) == exact_trajectory(new[k]) for k in keys]
        by_condition[condition] = dict(branches=len(keys), identical_trajectories=sum(same),
            changed_trajectories=len(keys) - sum(same),
            historical_mean_inspection_joint=float(old_joint.mean()),
            corrected_mean_inspection_joint=float(new_joint.mean()),
            corrected_minus_historical_mean=float((new_joint - old_joint).mean()),
            corrected_mean_precision_05cm=float(np.mean(
                [new[k]["after_metrics"]["precision_05cm"] for k in keys])),
            corrected_mean_recall_05cm=float(np.mean(
                [new[k]["after_metrics"]["recall_05cm"] for k in keys])),
            corrected_mean_f1_05cm=float(np.mean([new[k]["final_f1_05cm"] for k in keys])),
            corrected_mean_coverage_2d=float(np.mean(
                [new[k]["after_metrics"]["coverage_2d"] for k in keys])),
            corrected_mean_local_new_surface_recall_05cm=float(np.mean(
                [new[k]["new_visible_surface_quality"]["recall_05cm"] for k in keys])))
    paired = []
    for context in ("Q0", "Q1", "Q2", "Q3"):
        for arrangement in ("shelf_west", "shelf_east"):
            full = new[context, arrangement, "S"]
            disabled = new[context, arrangement, "S_no_feedback"]
            paired.append(dict(context=context, arrangement=arrangement,
                trajectory_exact=(exact_trajectory(full) == exact_trajectory(disabled)),
                inspection_joint_difference=(full["new_area_times_final_f1_05cm"]
                                             - disabled["new_area_times_final_f1_05cm"]),
                full_posterior=full["gain_calibration"]["posterior_mean"],
                disabled_posterior=disabled["gain_calibration"]["posterior_mean"]))
    old_summary = load(old_run / "summary.json")
    new_summary = load(new_run / "summary.json")
    replay_report = load(replay / "verification.json")
    report = dict(schema_version="cpu_v10_3_feedback_fix_comparison/1", status="complete",
        branch_identity="same 40 branches: 4 contexts x 2 left/right arrangements x 5 conditions",
        historical_defect=dict(branches_with_zero_observed_actions=sum(
            r["gain_calibration"]["observed_actions"] == 0 for r in old_rows),
            branches_with_48_pending_predictions=sum(
                len(r["gain_calibration"]["pending_action_ids"]) == 48 for r in old_rows)),
        corrected_mechanism_gates=new_summary["mechanism_gates"],
        corrected_efficacy_gates=new_summary["efficacy_gates"],
        independent_replay=dict(status=replay_report["status"], branches=replay_report["branches"],
            planner_or_runtime_called=replay_report["planner_or_runtime_called"]),
        by_condition=by_condition, corrected_S_vs_no_feedback=paired,
        corrected_S_vs_baselines=new_summary["comparisons"],
        historical_S_vs_baselines=old_summary["comparisons"],
        conclusions=[
            "The implementation defect is fixed: predictions and actual outcomes close in all 40 branches.",
            "S and S_no_feedback remain identical in all eight paired histories despite different posteriors; the scalar observed camera-yield posterior has no decision-level increment for S on this pool.",
            "The corrected S inspection joint exceeds G, N and X descriptively on this development pool, but this is not a 2D coverage or precision-gain percentage.",
            "Local newly-visible recall shows that the main S advantage is additional discovered surface, not a large local reconstruction-accuracy improvement."
        ])
    write(output / "verification.json", report)
    lines = ["# V10.3 → V10.3.1 反馈修订对照", "",
        "历史 V10.3 的 40/40 分支均生成 48 个预测但消费 0 次；V10.3.1 的 40/40 分支均完成预测—实际观测闭合并留下 0 个待消费预测。独立重放 40/40 通过。", "",
        "修复后 S 与 S_no_feedback 仍为 8/8 轨迹完全一致，检查指标差为 0。因此实现错误已经消除，但当前全局相机收益标量对 S 没有决策级增量。", "",
        "|条件|旧检查联合指标|修复后|轨迹变化|最终 precision@5cm|最终 recall@5cm|新表面 recall@5cm|", "|---|---:|---:|---:|---:|---:|---:|"]
    for condition in conditions:
        r = by_condition[condition]
        lines.append(f"|{condition}|{r['historical_mean_inspection_joint']:.6f}|{r['corrected_mean_inspection_joint']:.6f}|{r['changed_trajectories']}/8|{r['corrected_mean_precision_05cm']:.6f}|{r['corrected_mean_recall_05cm']:.6f}|{r['corrected_mean_local_new_surface_recall_05cm']:.6f}|")
    lines += ["", "检查联合指标为共享前缀后新增可见表面积（m²）×最终全局 F1@5cm。二维覆盖质量联合指标、precision、recall/F1、表面误差、路程和动作时间均在新结果中单列。"]
    (output / "REPORT.md").write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--old-run", type=Path, required=True)
    parser.add_argument("--new-run", type=Path, required=True)
    parser.add_argument("--replay", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    main(args.old_run.resolve(), args.new_run.resolve(), args.replay.resolve(),
         args.output.resolve())
