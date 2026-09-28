#!/usr/bin/env python3
"""Parent-context paired statistics for the V11.1 independent confirmation."""
import argparse
import json
import math
from pathlib import Path

import numpy as np


def write(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2,
                               sort_keys=True, allow_nan=False) + "\n")


def compare(rows, baseline):
    primary = "learned_semantic_feedback"
    contexts = sorted({r["context"] for r in rows})
    metrics = ("joint_gain", "inspection_joint", "path_distance_m", "action_time_s")
    endpoint = ("coverage_2d", "f1_05cm", "precision_05cm", "recall_05cm",
                "surface_error_mean_m", "covered_area_m2", "area_times_f1_05cm")
    context_differences = {}
    for context in contexts:
        left = [r for r in rows if r["context"] == context and r["condition"] == primary]
        right = [r for r in rows if r["context"] == context and r["condition"] == baseline]
        if len(left) != 2 or len(right) != 2:
            raise ValueError("each parent context requires two arrangements per condition")
        values = {name: float(np.mean([r[name] for r in left])
                                    - np.mean([r[name] for r in right])) for name in metrics}
        values.update({"terminal_" + name: float(np.mean([r["after_metrics"][name]
            for r in left]) - np.mean([r["after_metrics"][name] for r in right]))
            for name in endpoint})
        context_differences[context] = values
    diff = np.asarray([context_differences[c]["joint_gain"] for c in contexts])
    nonzero = int(np.count_nonzero(diff))
    wins = int(np.count_nonzero(diff > 0))
    sign_p = (sum(math.comb(nonzero, k) for k in range(wins, nonzero + 1)) / 2**nonzero
              if nonzero else 1.)
    primary_mean = np.mean([r["joint_gain"] for r in rows if r["condition"] == primary])
    baseline_mean = np.mean([r["joint_gain"] for r in rows if r["condition"] == baseline])
    rng = np.random.default_rng(11102)
    boot = np.asarray([np.mean(diff[rng.integers(0, len(diff), len(diff))])
                       for _ in range(20000)])
    aggregate_components = {name: float(np.mean([context_differences[c][name]
        for c in contexts])) for name in context_differences[contexts[0]]}
    return {"baseline": baseline, "independent_parent_contexts": len(contexts),
        "parent_wins": wins, "parent_ties": int(np.count_nonzero(diff == 0)),
        "parent_losses": int(np.count_nonzero(diff < 0)),
        "primary_mean": float(primary_mean), "baseline_mean": float(baseline_mean),
        "mean_difference": float(diff.mean()),
        "relative_to_baseline": float(diff.mean() / baseline_mean),
        "bootstrap_95_percentile_interval": [float(np.quantile(boot, .025)),
                                              float(np.quantile(boot, .975))],
        "exact_one_sided_sign_p": float(sign_p),
        "mean_component_differences": aggregate_components,
        "context_differences": context_differences}


def main(run, replay, output):
    verification = json.loads((replay / "verification.json").read_text())
    if verification["status"] != "passed" or verification["branches"] != 96:
        raise ValueError("96-branch independent physical replay is required")
    rows = json.loads((run / "results.json").read_text())
    baselines = ["learned_geometry", "fixed_semantic", "fixed_geometry", "learned_swapped"]
    report = {"schema_version": "semantic_gain_v11_1_confirmation_analysis/1",
        "status": "passed", "independent_confirmation": True,
        "sample_unit": "eight independent parent contexts; two arrangements averaged within parent",
        "primary_metric": "delta(covered_2d_area_m2 * global_3d_f1_at_5cm)",
        "comparisons": {name: compare(rows, name) for name in baselines}}
    output.mkdir(parents=True, exist_ok=False)
    write(output / "analysis.json", report)
    lines = ["# V11.1 independent confirmation analysis", "",
        "The sample unit is the independently generated parent context. The two category arrangements are averaged within each parent.", "",
        "| Baseline | Mean difference | Relative | Parent W/T/L | Exact one-sided sign p | 95% context bootstrap |",
        "|---|---:|---:|---:|---:|---:|"]
    for name in baselines:
        row = report["comparisons"][name]
        lo, hi = row["bootstrap_95_percentile_interval"]
        lines.append(f"| {name} | {row['mean_difference']:.6f} | "
            f"{100*row['relative_to_baseline']:.2f}% | {row['parent_wins']}/"
            f"{row['parent_ties']}/{row['parent_losses']} | {row['exact_one_sided_sign_p']:.6f} | "
            f"[{lo:.6f}, {hi:.6f}] |")
    (output / "REPORT.md").write_text("\n".join(lines) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--replay", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    main(args.run.resolve(), args.replay.resolve(), args.output.resolve())
