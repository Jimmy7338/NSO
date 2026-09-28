#!/usr/bin/env python3
"""Independent selection/metric arithmetic and a bounded V8.1 development figure.

Uses frozen scores, never oracle-selected candidates. This is not the independent
physical sensor/TSDF replay; it checks action ledgers, visibility unions, PR/F1
arithmetic and checkpoint integration without importing production evaluators.
"""
import argparse
import csv
import hashlib
import json
import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
METHODS = ("S", "G", "O", "N", "X", "M")
HISTORIES = tuple((p, a) for p in ("P0", "P1") for a in ("shelf_west", "shelf_east"))


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path, data):
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n")


def close(actual, recorded, name, errors, tolerance=1e-10):
    error = abs(float(actual) - float(recorded))
    errors[name] = max(errors.get(name, 0.), error)
    if error > tolerance:
        raise ValueError(f"{name}: independent arithmetic mismatch {actual} != {recorded}")


def analyze(run, audit):
    audit.mkdir(parents=True, exist_ok=False)
    manifest = json.loads((run / "artifact_hashes.json").read_text())
    seal = json.loads((run / "pre_execution_seal.json").read_text())
    inputs = {name: sha(run / name) for name in ("artifact_hashes.json", "pre_execution_seal.json")}
    errors = {}

    def read(name, decision=False):
        path = run / name
        identity = sha(path)
        if manifest[name] != identity or (decision and seal["decision_assets"][name] != identity):
            raise ValueError(f"frozen asset differs: {name}")
        inputs[name] = identity
        if path.suffix == ".npz":
            with np.load(path, allow_pickle=False) as archive:
                return {k: archive[k].copy() for k in archive.files}
        return json.loads(path.read_text())

    # Recover every selection from the same frozen score/cost/id ordering before
    # opening any outcome in this arithmetic routine. Scores are not refitted.
    selections, pools = {}, {}
    for parent, arrangement in HISTORIES:
        base = f"{parent}/{arrangement}"
        predictions = read(f"{base}/predictions.json", decision=True)
        routes = read(f"{base}/candidates.json", decision=True)
        if [r["candidate_id"] for r in routes] != list(range(6)):
            raise ValueError("the complete ordered common six-candidate pool is required")
        if predictions["trained"] or predictions["calibrated"]:
            raise ValueError("the V8.1 figure expects the fixed hand-declared prior")
        ids = {}
        for method in METHODS:
            scores = predictions["scores"][method]
            if len(scores) != 6 or not np.isfinite(scores).all():
                raise ValueError("nonfinite or incomplete score vector")
            selected = min(range(6), key=lambda i: (-scores[i], routes[i]["cost"], routes[i]["candidate_id"]))
            if selected != predictions["selected_candidate_ids"][method]:
                raise ValueError("sealed selection disagrees with its declared tie rule")
            ids[method] = selected
        np.testing.assert_array_equal(predictions["scores"]["M"], predictions["scores"]["G"])
        selections[base], pools[base] = ids, routes
    write_json(audit / "selection_from_frozen_predictions.json", dict(
        selections=selections, tie_rule="score descending, planned paid cost ascending, candidate_id ascending",
        source="pre_execution_seal.decision_assets", outcome_based_selection=False))

    checked, selected_rows = [], []
    for parent, arrangement in HISTORIES:
        base = f"{parent}/{arrangement}"
        reference = read(f"{base}/reference.npz")
        weights = reference["weights"]
        fixture = read(f"{base}/fixture.json", decision=True)
        if fixture["config"]["pose_noise_m"] != 0.:
            raise ValueError("pose condition differs from the declared exact-pose scope")
        by_id = {}
        for route in pools[base]:
            candidate_id = route["candidate_id"]
            folder = f"{base}/candidate_{candidate_id:03d}"
            outcome = read(f"{folder}/outcome.json")
            actions = read(f"{folder}/actions.json")
            metrics = read(f"{folder}/metrics.json")
            visibility = read(f"{folder}/visibility.npz")
            paid = len(actions)
            if paid != outcome["paid_actions"] or paid != route["cost"] or paid <= 0:
                raise ValueError("actual paid ledger differs from the completed route")
            if [a["action_index"] for a in actions] != list(range(1, paid + 1)):
                raise ValueError("paid ledger is not consecutive")
            if [a["action"] for a in actions] != route["actions"]:
                raise ValueError("paid actions differ from the frozen candidate")
            if any(a["collision"] for a in actions) or outcome["failure"] is not None:
                raise ValueError("a failure must be retained, not plotted as a completed route")
            if not outcome["full_original_route_completed"] or not outcome["returned_to_anchor"]:
                raise ValueError("the current all-completed scope no longer applies")
            union = visibility["outbound"] | visibility["endpoint"] | visibility["return"]
            np.testing.assert_array_equal(union, visibility["union"])
            new = union & ~visibility["prefix"]
            area = math.fsum(float(w) for w in weights[new])
            close(area, outcome["new_area_m2"], "new_area_m2", errors)
            rate = area / paid
            close(rate, outcome["area_per_action"], "area_per_action", errors)
            for label, values in (("before", outcome["before"]), ("after", outcome["after"])):
                p, r = values["precision_05cm"], values["recall_05cm"]
                f1 = 0. if p + r == 0 else 2 * p * r / (p + r)
                close(f1, values["f1_05cm"], f"{label}_F1_from_PR", errors)
                close(values["coverage_2d"], 1., "coverage_2d_equals_1", errors)
            gain = outcome["after"]["f1_05cm"] - outcome["before"]["f1_05cm"]
            close(gain, outcome["f1_gain_05cm"], "f1_gain_05cm", errors)
            close(gain / paid, outcome["f1_gain_per_action"], "f1_gain_per_action", errors)
            if metrics[0]["action_index"] != 0 or metrics[-1]["action_index"] != paid:
                raise ValueError("the metric trace must include prefix and final paid action")
            # Piecewise-linear integration plus a constant tail is independently
            # evaluated analytically, not by the production interpolation path.
            auc_integral = math.fsum((b["action_index"] - a["action_index"]) *
                (a["joint_05cm"] + b["joint_05cm"]) / 2 for a, b in zip(metrics, metrics[1:]))
            auc_integral += (48 - paid) * metrics[-1]["joint_05cm"]
            auc = auc_integral / 48
            close(auc, outcome["branch_joint_auc_05cm"], "branch_joint_auc_05cm", errors)
            row = dict(history=base, parent=parent, arrangement=arrangement,
                candidate_id=candidate_id, role=route["group"], new_area_m2=area,
                area_per_action=rate, f1_gain_05cm=gain, end_f1_05cm=outcome["after"]["f1_05cm"],
                prefix_f1_05cm=outcome["before"]["f1_05cm"], paid_actions=paid,
                prefix_paid_actions=150, task_paid_actions=150+paid,
                f1_gain_per_action=gain/paid, branch_joint_auc_05cm=auc, coverage_2d=1.)
            checked.append(row); by_id[candidate_id] = row
        for method in METHODS:
            selected_rows.append(dict(method=method, **by_id[selections[base][method]]))
    if len(checked) != 24 or sum(row["paid_actions"] for row in checked) != 752:
        raise ValueError("the fixed complete physical branch ledger changed")
    metrics_to_mean = ("area_per_action", "new_area_m2", "f1_gain_05cm", "end_f1_05cm", "paid_actions",
                       "task_paid_actions", "f1_gain_per_action", "branch_joint_auc_05cm")
    mean = {method: {key: math.fsum(row[key] for row in selected_rows if row["method"] == method) / 4
                     for key in metrics_to_mean} for method in METHODS}
    differences = {method: dict(
        area_rate_relative_gain=mean["S"]["area_per_action"] / mean[method]["area_per_action"] - 1,
        f1_gain_difference=mean["S"]["f1_gain_05cm"] - mean[method]["f1_gain_05cm"],
        end_f1_difference=mean["S"]["end_f1_05cm"] - mean[method]["end_f1_05cm"],
        branch_joint_auc_difference=mean["S"]["branch_joint_auc_05cm"] - mean[method]["branch_joint_auc_05cm"])
        for method in ("G", "O", "N", "X", "M")}
    verification = run / "verification.json"
    physical_status = json.loads(verification.read_text())["status"] if verification.exists() else "pending_independent_physical_replay"
    if verification.exists():
        inputs["verification.json"] = sha(verification)
    result = dict(status="passed_independent_selection_and_metric_arithmetic", physical_replay_status=physical_status,
        physical_replay_performed_by_this_script=False, physical_branches_checked=24, physical_paid_actions=752,
        parents=2, histories=4, equal_weight_per_history=True, no_confidence_interval_or_independence_claim=True,
        reported_mean_rate_is_mean_of_per_history_ratios_not_ratio_of_sums=True,
        branch_primary_excludes_common_prefix_but_prefix_cost_is_reported=150,
        all_completed_return=True, all_collision_counts_zero=True,
        selections=selections, method_means=mean, S_minus_comparators=differences,
        selected_history_rows=selected_rows, all_candidate_rows=checked, maximum_absolute_check_errors=errors,
        F1_guard_threshold_S_minus_G_O=-.001,
        F1_guard_passed=all(differences[m]["f1_gain_difference"] >= -.001 for m in ("G", "O")),
        interpretation="area-rate improvement cannot override the predeclared F1 guard; no overall progression pass",
        limits=["two development parent layouts, four dependent histories", "perfect input pose",
                "prefix already has C=1; late-stage refinement only", "S is a hand-declared uncalibrated asset prior",
                "not a full ANS system or a mainstream-method comparison"],
        input_hashes=inputs, script_sha256=sha(Path(__file__)))
    write_json(audit / "independent_metrics.json", result)
    with (audit / "selected_histories.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(selected_rows[0])); writer.writeheader(); writer.writerows(selected_rows)
    return result


def plot(result, output):
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10, "pdf.fonttype": 42})
    figure, axes = plt.subplots(1, 2, figsize=(10.8, 6.4))
    figure.subplots_adjust(left=.08, right=.97, top=.77, bottom=.29, wspace=.27)
    figure.suptitle("Higher surface efficiency, lower reconstruction gain", y=.965, fontsize=16)
    delta = result["S_minus_comparators"]["G"]
    figure.text(.5, .90,
        f"S versus G/O: area rate {delta['area_rate_relative_gain']*100:+.2f}%  |  "
        f"F1 gain {delta['f1_gain_difference']*100:+.3f} percentage points",
        ha="center", fontsize=11)
    figure.text(.5, .85, "Fixed choices from sealed scores • 2 development parents / 4 paired histories • no confidence intervals",
                ha="center", fontsize=9, color="#4c5560")
    colors = ("#2475b8", "#2475b8", "#d08423", "#d08423")
    markers = ("o", "s", "o", "s")
    offsets = (-.15, -.05, .05, .15)
    for axis, key, scale, title, ylabel in zip(axes,
        ("area_per_action", "f1_gain_05cm"), (1., 100.),
        ("Uniform new visible surface / action", "Reconstruction F1@5cm gain"),
        ("Area rate (m² / paid action)", "F1 increment (percentage points)")):
        axis.axvspan(-.45, .45, color="#eaf3fb", zorder=0)
        axis.axhline(0, color="#85909b", linewidth=.8, zorder=1)
        for hi, history in enumerate(HISTORIES):
            name = "/".join(history)
            values = [next(row[key]*scale for row in result["selected_history_rows"]
                           if row["method"] == m and row["history"] == name) for m in METHODS]
            axis.plot(np.arange(6)+offsets[hi], values, color=colors[hi], alpha=.18, linewidth=.8, zorder=2)
            axis.scatter(np.arange(6)+offsets[hi], values, color=colors[hi], marker=markers[hi],
                         s=40, edgecolor="white", linewidth=.5, zorder=3)
        for mi, method in enumerate(METHODS):
            mean = result["method_means"][method][key]*scale
            axis.plot([mi-.23, mi+.23], [mean, mean], color="#172432", linewidth=2.5, zorder=4)
        axis.set_xticks(range(6), METHODS); axis.set_xlim(-.5, 5.5)
        axis.set_title(title, fontsize=11, pad=12); axis.set_ylabel(ylabel)
        axis.grid(axis="y", alpha=.2); axis.spines[["top", "right"]].set_visible(False)
    handles = [Line2D([], [], color=colors[i], marker=markers[i], linestyle="none", markersize=6,
                      label=f"{p}: shelf {'west' if a=='shelf_west' else 'east'}") for i, (p, a) in enumerate(HISTORIES)]
    handles.append(Line2D([], [], color="#172432", linewidth=2.5, label="Equal-history mean"))
    figure.legend(handles=handles, loc="lower center", bbox_to_anchor=(.5,.178), ncol=5, frameon=False, fontsize=8.5)
    figure.text(.5,.14, "S: correct class prior   G: geometry   O: objectness   N: measured-only   X: swapped classes   M: missing classes",
                ha="center", fontsize=8.5)
    figure.text(.5,.084, "The predeclared F1 guard fails (allowed S − G/O ≥ −0.100 percentage points).",
                ha="center", fontsize=10, fontweight="bold", color="#9a3737")
    figure.text(.5,.035, "Perfect pose • prefix C = 1 • hand-declared S prior • late-stage refinement in an artificial asset-marker world.\n"
                "Not a full ANS evaluation, natural-semantic validation, or evidence of superiority over mainstream methods.",
                ha="center", fontsize=8.5, color="#4c5560", linespacing=1.5)
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output.with_suffix(".png"), dpi=170)
    figure.savefig(output.with_suffix(".pdf"))
    plt.close(figure)
    write_json(output.with_suffix(".json"), dict(
        figure_selection="all four fixed histories and all six required frozen scorers; no CI",
        script_sha256=sha(Path(__file__)), method_means=result["method_means"],
        physical_replay_status_at_generation=result["physical_replay_status"],
        outputs={output.with_suffix(s).name: dict(sha256=sha(output.with_suffix(s)), bytes=output.with_suffix(s).stat().st_size)
                 for s in (".png", ".pdf")}))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, default=ROOT / "eval_results/competition_v8_1_execution_20260911")
    parser.add_argument("--audit-output", type=Path, default=ROOT / "audit_results/competition_v8_1_selected_metrics_20260911")
    parser.add_argument("--output-prefix", type=Path, default=ROOT / "docs/research/figures/competition_v8_1_selected_results")
    args = parser.parse_args()
    result = analyze(args.run, args.audit_output)
    plot(result, args.output_prefix)
    print(json.dumps(dict(status=result["status"], method_means=result["method_means"],
                         S_minus_G=result["S_minus_comparators"]["G"], physical_replay_status=result["physical_replay_status"]), ensure_ascii=False))
