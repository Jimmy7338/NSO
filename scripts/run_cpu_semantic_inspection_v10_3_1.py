#!/usr/bin/env python3
"""Rerun the frozen V10.3 design after closing observed-gain feedback."""
import os
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.run_cpu_semantic_inspection_v10_2 import execute, write_json
from utils.cpu_protocol import file_hash


CONDITIONS = ("S", "G", "N", "X", "S_no_feedback")
CONTEXTS = ("Q0", "Q1", "Q2", "Q3")
ARRANGEMENTS = ("shelf_west", "shelf_east")


def mean(rows, condition, field):
    return float(np.mean([r[field] for r in rows if r["condition"] == condition]))


def nested_mean(rows, condition, parent, field):
    return float(np.mean([r[parent][field] for r in rows if r["condition"] == condition]))


def comparison(rows, primary, baseline):
    lookup = {(r["context"], r["arrangement"], r["condition"]): r for r in rows}
    pairs = sorted((context, arrangement) for context, arrangement, condition in lookup
                   if condition == primary)
    diffs = [lookup[c, a, primary]["new_area_times_final_f1_05cm"]
             - lookup[c, a, baseline]["new_area_times_final_f1_05cm"] for c, a in pairs]
    base_mean = float(np.mean([lookup[c, a, baseline]["new_area_times_final_f1_05cm"]
                              for c, a in pairs]))
    return dict(mean_difference=float(np.mean(diffs)),
        relative_mean_difference=float(np.mean(diffs)) / base_mean,
        wins=sum(v > 0 for v in diffs), ties=sum(v == 0 for v in diffs),
        losses=sum(v < 0 for v in diffs), differences=diffs)


def main(prepared, output, smoke=False):
    output.mkdir(parents=True, exist_ok=False)
    protocol = ROOT / "configs/virtual3d/cpu_four_module_v10_3_1_feedback_fix_protocol.json"
    sources = [Path(__file__).resolve(), protocol, ROOT / "nso/cpu_four_modules_v10.py",
        ROOT / "nso/observed_gain_calibration_v10_1.py", ROOT / "nso/hierarchical_options_v10.py",
        ROOT / "nso/runtime_integration.py", ROOT / "nso/observed_runtime_mapper_v10.py",
        ROOT / "scripts/run_cpu_semantic_inspection_v10_2.py"]
    frozen = {str(p.relative_to(ROOT)): file_hash(p) for p in sources}
    selected_conditions = ("S", "S_no_feedback") if smoke else CONDITIONS
    selected_contexts = ("Q0",) if smoke else CONTEXTS
    selected_arrangements = ("shelf_west",) if smoke else ARRANGEMENTS
    manifest = dict(schema_version="cpu_semantic_inspection_v10_3_1/1", status="running",
        created_utc=datetime.now(timezone.utc).isoformat(), smoke=bool(smoke),
        protocol=json.loads(protocol.read_text()), source_sha256=frozen,
        worlds_constructed_before_source_freeze=0, raw_sensor_files_saved=False)
    write_json(output / "manifest.json", manifest)
    rows = []
    for condition in selected_conditions:
        for context in selected_contexts:
            for arrangement in selected_arrangements:
                row = execute(prepared, context, arrangement, condition,
                              revision="v10_3_1", novelty_floor=.25)
                rows.append(row)
                write_json(output / "partial.json", rows)
                print(condition, context, arrangement,
                      row["new_area_times_final_f1_05cm"],
                      row["gain_calibration"]["posterior_mean"], flush=True)

    by_condition = {}
    for condition in selected_conditions:
        by_condition[condition] = dict(
            mean_inspection_joint=mean(rows, condition, "new_area_times_final_f1_05cm"),
            mean_new_visible_area_m2=mean(rows, condition, "new_unique_surface_area_m2"),
            mean_final_coverage_fraction_times_f1=mean(
                rows, condition, "final_coverage_fraction_times_f1_05cm"),
            mean_final_covered_area_times_f1=mean(
                rows, condition, "final_covered_area_times_f1_05cm"),
            mean_final_precision_05cm=nested_mean(rows, condition, "after_metrics", "precision_05cm"),
            mean_final_recall_05cm=nested_mean(rows, condition, "after_metrics", "recall_05cm"),
            mean_final_f1_05cm=mean(rows, condition, "final_f1_05cm"),
            mean_surface_error_m=nested_mean(rows, condition, "after_metrics", "surface_error_mean_m"),
            mean_new_surface_recall_05cm=nested_mean(
                rows, condition, "new_visible_surface_quality", "recall_05cm"),
            mean_path_distance_m=mean(rows, condition, "path_distance_m"),
            mean_action_time_s=mean(rows, condition, "action_time_s"))

    comparisons = ({"S_vs_" + baseline: comparison(rows, "S", baseline)
                    for baseline in ("G", "N", "X", "S_no_feedback")}
                   if not smoke else {"S_vs_S_no_feedback": comparison(
                       rows, "S", "S_no_feedback")})
    feedback_full = [r for r in rows if r["condition"] != "S_no_feedback"]
    feedback_disabled = [r for r in rows if r["condition"] == "S_no_feedback"]
    mechanism_gates = dict(
        all_safe=all(not r["failed"] and r["collisions"] == 0 and r["returned_to_anchor"]
                     for r in rows),
        all_four_interfaces=all(r["modules_called"] == ["IGCR", "OV-SDF", "RPN-UQ", "STGHP"]
                                for r in rows),
        feedback_full_closed=all(r["gain_calibration"]["observed_actions"] == 48
            and not r["gain_calibration"]["pending_action_ids"]
            and r["gain_calibration"]["posterior_mean"] != {"radar": .5, "camera": .5}
            for r in feedback_full),
        feedback_disabled_recorded_and_frozen=all(
            r["gain_calibration"]["observed_actions"] == 48
            and not r["gain_calibration"]["pending_action_ids"]
            and r["gain_calibration"]["alpha"] == {"radar": 1, "camera": 1}
            and r["gain_calibration"]["beta"] == {"radar": 1, "camera": 1}
            for r in feedback_disabled))
    efficacy_gates = ({"S_mean_exceeds_G_N_X": all(
        comparisons["S_vs_" + b]["mean_difference"] > 0 for b in ("G", "N", "X")),
        "S_mean_exceeds_no_feedback": comparisons["S_vs_S_no_feedback"]["mean_difference"] > 0}
        if not smoke else {"S_mean_exceeds_no_feedback":
                           comparisons["S_vs_S_no_feedback"]["mean_difference"] > 0})
    result = dict(by_condition=by_condition, comparisons=comparisons,
        mechanism_gates=mechanism_gates, mechanism_passed=all(mechanism_gates.values()),
        efficacy_gates=efficacy_gates, efficacy_passed=all(efficacy_gates.values()),
        branches=len(rows), smoke=bool(smoke))
    write_json(output / "results.json", rows)
    write_json(output / "summary.json", result)
    manifest["status"] = "complete"
    write_json(output / "manifest.json", manifest)
    if any(file_hash(ROOT / name) != digest for name, digest in frozen.items()):
        raise RuntimeError("source changed during V10.3.1 execution")
    write_json(output / "artifact_hashes.json", {str(p.relative_to(output)): file_hash(p)
        for p in sorted(output.rglob("*")) if p.is_file() and p.name != "artifact_hashes.json"})
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--prepared", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    main(args.prepared.resolve(), args.output.resolve(), args.smoke)
