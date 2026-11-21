#!/usr/bin/env python3
"""Rebuild draft manuscript tables from independently evaluated frozen evidence."""
from pathlib import Path
import csv
import hashlib
import json

HERE = Path(__file__).resolve().parent
BASE = HERE.parent

def read(p):
    return json.loads(p.read_text())

def resource(p):
    return {"path": str(p), "sha256": hashlib.sha256(p.read_bytes()).hexdigest()}

def csv_write(name, rows):
    with (HERE / name).open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)

def main():
    summary = read(BASE / "audit/summary.json")
    aggregate = []
    for split, groups in [("train", summary["training_rows"])] + [
        (s, summary["splits"][s]["rows"]) for s in ("validation", "heldout")
    ]:
        for method, r in groups.items():
            aggregate.append(dict(split=split, method=method, **r))
    csv_write("depth_calibration_aggregate.csv", aggregate)
    labels = {"train": "Train (12)", "validation": "Validation (6)", "heldout": "Held out (6)"}
    methods = {"posterior": "Loss update", "prior_only": "Conditioned prior", "fixed_0.65": "Fixed 0.65 m"}
    table = [r"\begin{table}[tbp]", r"\centering\small", r"\setlength{\tabcolsep}{3.5pt}",
        r"\caption{Independent conditional-depth evaluation. All-case CRPS averages the four conditions within each object, then objects. The remaining columns use the predeclared front--side and around-object subset, with two dependent conditions per object. Errors and widths are in metres; $n$ in the split label counts objects. Fallback counts for all conditions are 5/48, 1/24 and 1/24. Fixed-size intervals are point distributions, not confidence estimates.}",
        r"\label{tab:depth-calibration-aggregate}", r"\begin{tabular}{llrrrrr}", r"\toprule",
        r"Split ($n$) & Distribution & All CRPS & Subset CRPS & Median MAE & 95\% count & Width \\", r"\midrule"]
    last = None
    for r in aggregate:
        if last is not None and last != r["split"]:
            table.append(r"\midrule")
        table.append(f"{labels[r['split']] if last != r['split'] else ''} & {methods[r['method']]} & "
            f"{r['all_condition_crps_m']:.4f} & {r['informative_crps_m']:.4f} & "
            f"{r['informative_median_mae_m']:.4f} & {r['informative_interval95_covered']}/{r['informative_interval95_denominator']} & "
            f"{r['informative_interval95_width_m']:.4f} \\\\")
        last = r["split"]
    table += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    (HERE / "depth_calibration_aggregate.tex").write_text("\n".join(table) + "\n")

    conditions = []
    all_conditions = []
    for split in ("validation", "heldout"):
        data = summary["splits"][split]["by_condition"]
        for cond in ("front", "front_side", "around", "occluded"):
            post = data["posterior"][cond]
            interval = post["intervals"]["0.95"]
            conditions.append(dict(split=split, condition=cond,
                posterior_crps_m=post["crps_m"], prior_crps_m=data["prior_only"][cond]["crps_m"],
                fixed_crps_m=data["fixed_0.65"][cond]["crps_m"],
                posterior_median_mae_m=post["median_absolute_error_m"],
                interval95_count=interval["covered_cases"], denominator=interval["case_denominator"],
                interval95_width_m=interval["equal_object_mean_width_m"]))
            for method in methods:
                r = data[method][cond]
                row = dict(split=split, condition=cond, method=method,
                    crps_m=r["crps_m"], median_mae_m=r["median_absolute_error_m"],
                    mean_estimate_mae_m=r["mean_absolute_error_m"], objects=r["objects"], cases=r["cases"],
                    outside_support_cases=r["outside_support_cases"])
                for level, val in r["intervals"].items():
                    for key in ("covered_cases", "case_denominator", "equal_object_mean_width_m"):
                        row[f"interval_{level}_{key}"] = val[key]
                all_conditions.append(row)
    csv_write("depth_calibration_conditions.csv", conditions)
    csv_write("depth_calibration_conditions_all_metrics.csv", all_conditions)
    table = [r"\begin{table}[tbp]", r"\centering\small", r"\setlength{\tabcolsep}{4pt}",
        r"\caption{All four observation conditions on untouched evaluation objects. Each row contains six objects. CRPS and posterior median absolute error (MAE) are in metres. The 95\% count and width describe the loss-updated depth distribution, with declared failed cases retained through the public-prior fallback.}",
        r"\label{tab:depth-calibration-conditions}", r"\begin{tabular}{llrrrrrr}", r"\toprule",
        r"Split & Condition & Fixed & Prior & Update & MAE & 95\% count & Width \\", r"\midrule"]
    c_labels = {"front": "Front", "front_side": "Front + side", "around": "Around object", "occluded": "Occluded"}
    last = None
    for r in conditions:
        if last is not None and last != r["split"]:
            table.append(r"\midrule")
        table.append(f"{labels[r['split']] if last != r['split'] else ''} & {c_labels[r['condition']]} & "
            f"{r['fixed_crps_m']:.4f} & {r['prior_crps_m']:.4f} & {r['posterior_crps_m']:.4f} & "
            f"{r['posterior_median_mae_m']:.4f} & {r['interval95_count']}/{r['denominator']} & "
            f"{r['interval95_width_m']:.4f} \\\\")
        last = r["split"]
    table += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    (HERE / "depth_calibration_conditions.tex").write_text("\n".join(table) + "\n")

    training = read(BASE / "audit/calibration_attempt_001/training_scores.json")
    chosen = next(s for s in training["scales"] if s["sigma_m"] == summary["sigma_m"])
    raw = [("train", chosen)] + [(s, read(BASE / f"audit/evaluation_attempt_001/{s}.json")) for s in ("validation", "heldout")]
    cases = []
    for split, source in raw:
        for method, rows in source["rows"].items():
            for r in rows:
                m = r["metrics"]
                row = dict(split=split, method=method, case_id=r["case_id"], object_id=r["object_id"],
                    condition=r["condition"], crps_m=m["crps_m"], median_m=m["median_m"],
                    median_absolute_error_m=m["median_absolute_error_m"],
                    projected_true_extent_m=r["target"]["projected_true_extent_m"],
                    axis_error_deg=r["target"]["error_to_true_depth_axis_deg"],
                    swapped_width_depth_axis=r["target"]["swapped_width_depth_axis"],
                    fallback=r["evaluator_prior_fallback"], failure_reason=r["failure_reason"] or "",
                    truth_outside_particle_support=m["truth_outside_particle_support"])
                for level, val in m["intervals"].items():
                    for key in ("lower_m", "upper_m", "width_m", "covered"):
                        row[f"interval_{level}_{key}"] = val[key]
                cases.append(row)
    csv_write("depth_calibration_all_cases.csv", cases)
    csv_write("depth_calibration_scale_grid.csv", summary["training_scale_scores"])
    sources = [BASE / "audit/summary.json", BASE / "audit/calibration_attempt_001/training_scores.json",
        BASE / "audit/calibration_attempt_001/selected_scale.json", BASE / "audit/evaluation_attempt_001/validation.json",
        BASE / "audit/evaluation_attempt_001/heldout.json", BASE / "method/METHOD_SPEC.md", BASE / "ARCHITECTURE_DELTA.json",
        BASE / "generator/dataset_manifest.json", BASE / "figure/validation.json", BASE / "figure/calibration_distribution.pdf"]
    outputs = sorted(HERE.glob("depth_calibration_*"))
    receipt = dict(status="draft_tables_from_completed_independent_calibration_only", sigma_m=summary["sigma_m"],
        independent_objects=24, cases=96, per_case_metric_rows=len(cases), aggregate_rows=len(aggregate),
        evaluation_condition_rows=len(conditions), scientific_sources=[resource(p) for p in sources],
        build_script=resource(Path(__file__).resolve()), outputs=[resource(p) for p in outputs],
        manuscript_policy_results_included=False, formal_manuscript_modified=False)
    (HERE / "table_receipt.json").write_text(json.dumps(receipt, indent=2, allow_nan=False) + "\n")
    print(json.dumps({k: receipt[k] for k in ("status", "per_case_metric_rows", "aggregate_rows", "evaluation_condition_rows")}))

if __name__ == "__main__":
    main()
