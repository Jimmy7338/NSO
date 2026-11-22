#!/usr/bin/env python3
"""Join exact forks to the old factual traces and summarize new building pairs."""

import csv
from math import isclose
from pathlib import Path

from runner import PREVIOUS, ROOT, read, sha, write


def csv_rows(path):
    with path.open(newline="") as stream:
        return list(csv.DictReader(stream))


def fmt(number, scale=1):
    return "—" if number is None else f"{float(number)*scale:.2f}"


def dispatches():
    original = PREVIOUS / "results/formal"
    current = ROOT / "results/development"
    cases = read(ROOT/"DEVELOPMENT_FREEZE.json")["jobs"]
    old = csv_rows(original/"decisions.csv")
    new = csv_rows(current/"decisions.csv")
    old_curve = csv_rows(original/"coverage_curve.csv")
    new_curve = csv_rows(current/"coverage_curve.csv")
    evaluated = {(r["layout"],r["arm"]):r for r in read(current/"summary.json")["episodes"]}
    curves = {(row["layout"],row["arm"],int(row["paid"])):float(row["C_free"]) for row in old_curve+new_curve}
    index = {(row["layout"],row["arm"],row["selection_call_id"]):row for row in old+new}
    outcomes = []
    for case in cases:
        row = evaluated[case["layout"],case["arm"]]
        branch = case["branch"]
        uid,paid = branch["state_uid"],branch["paid_before"]
        native_id = str(branch["selection_call_id"])
        layout,arm = row["layout"],row["arm"]
        first = index[layout,"region_S",native_id]
        modified = {"layout":layout,"arm":arm,"state_uid":uid,"common_paid":paid,
                    "status":row["status"],"original_target":first["target_pose"],
                    "original_target_arrived_paid":first["arrived_paid"],
                    "expected_target":branch["expected_pose"]}
        if row["status"] == "completed_execution":
            candidate = index[layout,arm,native_id]
            intervention = read(Path(row["run_directory"])/"matched_intervention.json")
            previous_calls = read(Path(branch["source_run"])/"global_choices.json")
            captured = next(r["outputs"] for r in previous_calls if r["call_id"]==int(native_id))
            scores = {tuple(route["pose"]):score for route,score in zip(
                captured["candidates"],captured["score_audit"],strict=True)}
            for offer in intervention["ranking"]:
                reference = scores[tuple(offer["pose"])]
                for field,key in (("native_outbound_coverage_m2","coverage_only_proxy"),
                                  ("native_outbound_measured_quality_m2","quality_total_proxy")):
                    assert isclose(offer[field],reference[key],rel_tol=1e-12,abs_tol=1e-12)
            modified |= {"actual_target":candidate["target_pose"],"actual_outcome":candidate["actual_outcome"],
                         "actual_dispatch_paid":candidate["first_dispatched_paid"],
                         "actual_arrival_paid":candidate["arrived_paid"],"actual_guard":candidate["first_guard_reason"],
                         "pool_native_measured_scores_exact_match":True, "shared_observation_before_cut":True}
            for delta in (24,48):
                clock=paid+delta
                observed=curves.get((layout,arm,clock))
                baseline=curves.get((layout,"region_S",clock))
                modified[f"C_free_difference_after_{delta}_paid"]=(
                    observed-baseline if observed is not None and baseline is not None else None)
        outcomes.append(modified)
    return outcomes


def main():
    dest=ROOT/"results"
    dev,formal=(read(dest/scope/"summary.json") for scope in ("development","formal"))
    for scope,summary in (("development",dev),("formal",formal)):
        assert summary["status"]=="evaluated" and summary["pending_trials"]==0
        assert summary["run_freeze_sha256"]==sha(ROOT/f"{scope.upper()}_FREEZE.json")
    literal = sum(r["status"]=="completed_execution" for r in dev["episodes"])
    branch_accounting={
        "new_branch_assignments":len(dev["episodes"]), "new_verified_endpoints":literal,
        "archived_comparators":len({r["layout"] for r in dev["episodes"]}),
        "frozen_evaluator_aggregated_endpoint_count":dev["complete_endpoints"],
        "frozen_evaluator_note":"For development only, the aggregate endpoint counter includes the three matched archived regional controls inserted for contrast construction. The episodes array and phase receipt index only the new branches. Map evaluation and raw outputs are unchanged.",
        "frozen_evaluator_summary_sha256":sha(dest/"development/summary.json"),
    }
    ledger=dest/"development/count_audit.json"
    if ledger.exists():
        assert read(ledger)==branch_accounting
    else:
        write(ledger,branch_accounting)
    outcomes=dispatches()
    forks=dest/"forks.json"
    if forks.exists():
        assert read(forks)==outcomes
    else:
        write(forks,outcomes)
    result=["# 门后区域归属与沿途收益验证", "",
            "只对原区域评分实施两项独立变更。门前位姿的未访问和剩余前沿按已识别门后区域计算。"
            "用原生全去程的覆盖和已测表面质量替换端点可见未知面积项。候选、足迹、回程预留及局部 S 续行使用原规则。", "",
            "## 同前缀单次干预", "",
            f"三处原案例分别执行仅更正归属、仅恢复沿途收益的首选。新增 {literal} 轮可核验分支，"
            "另配三个历史对照。两项合用在这三处均与后者选同一合法目标，因此相同目标的证据来自同一分支。"
            "表中覆盖差按共同实际动作时钟计算。开发期总计字段包含了三个历史对照，逐回合记录只列新增分支。"
            " [计数核验](development/count_audit.json)", "",
            "| 布局 | 干预 | 公共前缀步 | 原目标实到步 | 新目标实到步 | +24 步覆盖差（百分点） | +48 步覆盖差（百分点） |",
            "|---|---|---:|---:|---:|---:|---:|"]
    names={"attribution_once":"仅门后归属","outbound_once":"仅沿途收益"}
    for r in outcomes:
        fields=[r["layout"],names[r["arm"]],str(r["common_paid"]),r.get("original_target_arrived_paid") or "—",
                r.get("actual_arrival_paid") or "—",
                *(fmt(r.get(f"C_free_difference_after_{n}_paid"),100) for n in (24,48))]
        result.append("| "+" | ".join(fields)+" |")
    result.extend(["","## 独立楼层","",
                   f"{formal['frozen_trials']} 个预登记回合中有 {formal['complete_endpoints']} 个可核验终点。",
                   "各臂从同一楼层首帧独立执行，192 个计费动作含返航。J 等于正确自由区覆盖与 F1@5 cm 的乘积。","",
                   "| 楼层 | 策略 | 实付步 | 覆盖 % | F1 % | J % | A95 | 重复路程（m） | 返航 | 碰撞 |",
                   "|---|---|---:|---:|---:|---:|---:|---:|---|---:|"])
    for r in formal["episodes"]:
        cells=[r["layout"],r["arm"],str(r.get("actual_paid","—")),
               *(fmt(r.get(k),100) for k in ("C_free","F1_scene_05cm","J")),
               str(r.get("A95_paid") or "—"),fmt(r.get("repeat_distance_m")),
               {True:"是",False:"否"}.get(r.get("returned"),"—"),str(r.get("collisions","—"))]
        result.append("| "+" | ".join(cells)+" |")
    paired={r["layout"]:{} for r in formal["episodes"]}
    for r in formal["contrasts"]:
        paired[r["layout"]][r["contrast"]]=r
    result.extend(["","| 楼层 | J 修订相对原区域（百分点） | J 修订相对严格前沿（百分点） | A95 修订相对严格前沿（步） |",
                   "|---|---:|---:|---:|"])
    for layout, comparisons in paired.items():
        v1=comparisons["repair_vs_region_v1"]
        strict=comparisons["repair_vs_strict"]
        result.append("| "+" | ".join((layout,fmt(v1.get("delta_J"),100),
                                       fmt(strict.get("delta_J"),100),fmt(strict.get("delta_A95_paid"))))+" |")
    result.extend(["","| 楼层 | 策略 | 首次跨房步 | 实际进入房间数 | 仿真墙钟（秒） |",
                   "|---|---|---:|---:|---:|"])
    for r in formal["episodes"]:
        if r["arm"] in ("combined_S","F_strict"):
            result.append("| "+" | ".join((r["layout"],r["arm"],str(r.get("first_new_room_paid") or "—"),
                                           str(r.get("rooms_visited","—")),fmt(r.get("elapsed_s"))))+" |")
    result.extend(["","覆盖按传感器正确识别的可达自由格计量。实际进入房间数以小车轨迹是否跨入为准。",
                   "墙钟时间来自本次并行仿真执行。"])
    result.extend(["","两栋独立楼层的差值用于描述性比较。未执行的动作和缺失终点保留为空。",
                   "[固定候选池逐次审计](../../access_audit/README.md) · [分支核验](forks.json) · [留出配对值](formal/contrasts.csv)",""])
    (dest/"RESULTS.md").write_text("\n".join(result))


if __name__=="__main__":
    main()
