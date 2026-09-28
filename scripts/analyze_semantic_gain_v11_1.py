#!/usr/bin/env python3
"""Create paired development statistics for the independently replayed V11.1 run."""
import argparse
import json
import math
from pathlib import Path

import numpy as np


def write(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2,
                               sort_keys=True, allow_nan=False) + "\n")


def comparison(rows, baseline):
    lookup = {(r["context"], r["arrangement"], r["condition"]): r for r in rows}
    keys = sorted((r["context"], r["arrangement"]) for r in rows
                  if r["condition"] == "learned_semantic_feedback")
    primary = np.asarray([lookup[c, a, "learned_semantic_feedback"]["joint_gain"]
                          for c, a in keys])
    control = np.asarray([lookup[c, a, baseline]["joint_gain"] for c, a in keys])
    diff = primary - control
    nonzero = int(np.count_nonzero(diff))
    wins = int(np.count_nonzero(diff > 0))
    sign_p = (sum(math.comb(nonzero, k) for k in range(wins, nonzero + 1))
              / 2**nonzero if nonzero else 1.0)
    rng = np.random.default_rng(1101)
    boot = np.asarray([np.mean(diff[rng.integers(0, len(diff), len(diff))])
                       for _ in range(20000)])
    return {"baseline": baseline, "pairs": len(diff), "wins": wins,
        "ties": int(np.count_nonzero(diff == 0)), "losses": int(np.count_nonzero(diff < 0)),
        "primary_mean": float(primary.mean()), "baseline_mean": float(control.mean()),
        "mean_difference": float(diff.mean()),
        "relative_to_baseline": float(diff.mean() / control.mean()),
        "median_difference": float(np.median(diff)),
        "bootstrap_95_percentile_interval": [float(np.quantile(boot, .025)),
                                              float(np.quantile(boot, .975))],
        "exact_one_sided_sign_p": float(sign_p), "differences": diff.tolist()}


def main(run, replay, output):
    verification = json.loads((replay / "verification.json").read_text())
    if verification["status"] != "passed" or verification["branches"] != 56:
        raise ValueError("V11.1 independent replay must pass 56 branches")
    rows = json.loads((run / "results.json").read_text())
    output.mkdir(parents=True, exist_ok=False)
    baselines = ["learned_geometry", "fixed_semantic", "fixed_geometry",
                 "learned_swapped", "learned_semantic_no_feedback"]
    report = {"schema_version": "semantic_gain_v11_1_paired_analysis/1",
        "status": "passed", "development_only": True,
        "primary_metric": "delta(covered_2d_area_m2 * global_3d_f1_at_5cm)",
        "multiple_testing_note": "descriptive development inference; no independent-set p-value yet",
        "comparisons": {name: comparison(rows, name) for name in baselines}}
    write(output / "analysis.json", report)
    lines = ["# V11.1 paired development analysis", "",
        "All 56 physical branches passed independent replay. These statistics describe the already-viewed development domain and are not independent confirmation.", "",
        "| Baseline | Mean difference | Relative | W/T/L | Exact one-sided sign p | Bootstrap 95% interval |",
        "|---|---:|---:|---:|---:|---:|"]
    for name in baselines:
        row = report["comparisons"][name]
        lo, hi = row["bootstrap_95_percentile_interval"]
        lines.append(f"| {name} | {row['mean_difference']:.6f} | {100*row['relative_to_baseline']:.2f}% | "
                     f"{row['wins']}/{row['ties']}/{row['losses']} | {row['exact_one_sided_sign_p']:.6f} | "
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
