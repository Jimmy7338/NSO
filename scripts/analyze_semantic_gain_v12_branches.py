#!/usr/bin/env python3
"""Audit strict V12 development branches; never converts them into confirmation."""
import argparse
import json
from pathlib import Path

import numpy as np


def read(path):
    return json.loads(Path(path).read_text())


def main(inputs, output):
    contexts = []
    for directory in inputs:
        if (directory / "INVALIDATED_DO_NOT_USE.md").exists():
            raise ValueError(f"refusing invalidated branch directory: {directory}")
        summary = read(directory / "summary.json")
        outcomes = read(directory / "outcomes.json")
        features = {int(row["candidate_id"]): row for row in read(directory / "features.json")}
        predictions = read(directory / "predictions.json")
        if (not summary["complete_context"] or not summary["all_executed_safe"]
                or len(outcomes) != summary["candidate_count"]):
            raise ValueError(f"incomplete or unsafe strict context: {directory}")
        by_id = {int(row["candidate_id"]): row for row in outcomes}
        if len(by_id) != len(outcomes) or set(by_id) != set(features):
            raise ValueError("candidate outcomes and features do not align")
        ranked = sorted(outcomes, key=lambda row: -row["joint_gain_per_action"])
        fixed = {}
        for mode, candidate_id in predictions["selected_candidate_ids"].items():
            row = by_id[int(candidate_id)]
            fixed[mode] = {"candidate_id": int(candidate_id), "group": row["group"],
                "utility": row["joint_gain_per_action"], "true_rank": ranked.index(row) + 1}
        target_rows = []
        for row in outcomes:
            if row["asset_index"] is None:
                continue
            signed_target = float(features[row["candidate_id"]]["semantic"][12])
            sign = 0 if signed_target == 0 else (1 if signed_target > 0 else -1)
            target_rows.append({"candidate_id": row["candidate_id"], "group": row["group"],
                "class_sign": sign, "utility": row["joint_gain_per_action"],
                "coverage_gain_m2": row["coverage_gain_m2"],
                "f1_gain_05cm": row["f1_gain_05cm"]})
        category = {}
        for role in ("entry", "deep"):
            category[role] = {}
            for sign, name in ((-1, "simple"), (1, "complex")):
                selected = [row for row in target_rows if role in row["group"]
                            and row["class_sign"] == sign]
                category[role][name] = {key: float(np.mean([row[key] for row in selected]))
                    for key in ("utility", "coverage_gain_m2", "f1_gain_05cm")}
                category[role][name]["count"] = len(selected)
            simple = category[role]["simple"]["utility"]
            complex_value = category[role]["complex"]["utility"]
            category[role]["complex_vs_simple_relative_utility"] = (
                (complex_value - simple) / simple if simple else None)
        contexts.append({"context": summary["context"], "directory": str(directory),
            "candidates": len(outcomes), "fallback_safe_returns": sum(
                row["returned_after_route_denial"] for row in outcomes),
            "best": [{"candidate_id": row["candidate_id"], "group": row["group"],
                      "utility": row["joint_gain_per_action"]} for row in ranked[:5]],
            "fixed_selections": fixed, "category_diagnostics": category})
    result = {"schema_version": "semantic_gain_v12_strict_branch_analysis/1",
        "status": "development_diagnostic_complete", "contexts": contexts,
        "eligible_for_development_training": True,
        "confirmation": False,
        "causal_semantic_gain_proven": False,
        "reason": "category diagnostics remain confounded by route location and coverage until multiple randomized contexts and label interventions are evaluated"}
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists():
        raise FileExistsError(output)
    output.write_text(json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", action="append", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    main([path.resolve() for path in args.input], args.output.resolve())
