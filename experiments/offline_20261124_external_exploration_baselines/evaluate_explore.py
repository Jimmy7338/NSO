#!/usr/bin/env python3
"""Measure the isolated Explore Lite baseline against the frozen four-floor series."""

import argparse
import csv
import importlib.util
import json
import math
import os
from pathlib import Path
from statistics import mean, median

for variable in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[variable] = "1"

import numpy as np
import open3d as o3d


ROOT = Path(__file__).resolve().parent
BASELINE = ROOT.parent / "region_independent_floors_20261123"
FROZEN_RESULTS = BASELINE / "results/formal"
PROTOCOL = ROOT / "EXPLORE_PROTOCOL.json"
ARM = "ExploreLite_selection"
CHECKPOINTS = ("start", "paid_48", "paid_96", "paid_144", "paid_192", "terminal")
METRICS = ("C_free", "p_scene_05cm", "r_scene_05cm", "F1_scene_05cm", "J")
SOURCE = BASELINE / "evaluation/raw_qa.py"
SPEC = importlib.util.spec_from_file_location("published_independent_raw_evaluator", SOURCE)
raw = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(raw)


def require(condition, message):
    if not condition:
        raise ValueError(message)


def indexed(rows, keys):
    result = {tuple(row[k] for k in keys): row for row in rows}
    require(len(result) == len(rows), f"Duplicate keys {keys} in published source")
    return result


def reference_for(layout, archived):
    location = BASELINE / "scenes/evaluator_references" / layout
    receipt = raw.read(location / "RECEIPT.json")
    pinned = archived["reference_files_sha256"]
    for name in ("RECEIPT.json", "evaluator_truth.npz", "fixed_reference.npz", "source_mesh.ply"):
        path = location / name
        require(raw.sha(path) == pinned[str(path)], f"Frozen evaluation input changed: {path}")
    with np.load(location / "evaluator_truth.npz", allow_pickle=False) as inputs:
        reachable = inputs["reachable"]
    with np.load(location / "fixed_reference.npz", allow_pickle=False) as inputs:
        samples = inputs["all"]
    require(int(reachable.sum()) == receipt["reachable_cells"] and len(samples) > 0, "Bad frozen reference")
    scene = raw.ray(o3d.io.read_triangle_mesh(str(location / "source_mesh.ply")))
    return reachable, samples, scene


def measure(layout, run, official, official_audit, arm=ARM, protocol=PROTOCOL,
            receipt_name="explore_lite_receipt.json"):
    receipt = raw.read(run / receipt_name)
    manifest_file = run / "manifest.json"
    manifest = raw.read(manifest_file)
    require(receipt["arm"] == arm and receipt["layout"] == manifest["layout"] == layout,
            "External treatment identity does not match the registered episode")
    require(receipt["native_manifest_sha256"] == raw.sha(manifest_file)
            and receipt["external_protocol_sha256"] == raw.sha(protocol),
            "External receipt does not bind the exact manifest and current protocol")
    calls = receipt["selection_calls"]
    require(calls, "External selection was never called")
    require(manifest["arm"] == "F_strict" and manifest["actual_closed_loop"]
            and not manifest["evaluator_inputs_read_by_actor"], "Shared execution contract changed")
    archived = Path(official["run_directory"])
    archived_manifest = raw.read(archived / "manifest.json")
    shared = ("initial_packet_sha256", "initial_belief_sha256", "initial_pose",
              "initial_owner_sha256", "initial_grid_resolution_m", "native_budget",
              "runner_freeze_sha256", "actual_scene_seed", "frozen_cold_input_sha256")
    require(all(manifest[key] == archived_manifest[key] for key in shared),
            "Common initial state or shared execution freeze mismatch")
    require(raw.read(run / "initial_raw_rgbd_scan_sha256.json")
            == raw.read(archived / "initial_raw_rgbd_scan_sha256.json"),
            "Cold RGB-D and scan data changed")
    require(raw.read(run / "start/checkpoint.json") == raw.read(archived / "start/checkpoint.json"),
            "Cold sensor-derived grid or initial reconstruction changed")
    for name in ("steps.json", "observed_belief_trace.npz", "frontier_events.json"):
        require(raw.sha(run / name) == manifest["files_sha256"][name], f"Native trace changed: {name}")
    steps = raw.read(run / "steps.json")
    paid = len(steps)
    require(paid == manifest["paid_actions"] and 0 <= paid <= manifest["native_budget"] == 192
            and manifest["observed_native_termination"]["final_action_id"] == paid,
            "Executed action count differs from manifest or terminal receipt")
    require(all(0 <= call["paid"] < 192 for call in calls), "Invalid selection clock")
    require(all(step["step"] == clock and step["action"] in ("left", "right", "forward")
                and step["guard_allowed"] and (step["reserved_return_cost"] is None
                                               or clock+step["reserved_return_cost"] <= 192)
                for clock, step in enumerate(steps, 1)), "Invalid native action or invalid return reservation")
    end_pose = [*steps[-1]["position"], steps[-1]["heading"]] if steps else manifest["initial_pose"]
    collisions = sum(step["collision"] for step in steps)
    returned = end_pose == manifest["initial_pose"]
    if manifest["final_pose"] is not None:
        require(end_pose == manifest["final_pose"], "Native end pose mismatched")
    require(collisions == manifest["collisions"]
            and returned == manifest["returned"] == manifest["observed_native_termination"]["returned_to_anchor"],
            "Collision or return receipt mismatched")
    guards = raw.read(run / "frontier_events.json")
    summarized_guards = [e for e in guards if e["event"] in ("guard_rejection", "return_guard_rejection")]
    timing = [step["decision_wall_s"] for step in steps]
    require(math.isclose(sum(timing), manifest["charged_decision_wall_s"], abs_tol=1e-8), "Timing mismatched")
    reachable, fixed_points, source_scene = reference_for(layout, official_audit)
    with np.load(run / "observed_belief_trace.npz", allow_pickle=False) as inputs:
        require(list(inputs["paid"]) == list(range(paid+1)), "Coverage trace has skipped action clocks")
        beliefs = inputs["belief"]
    total = int(reachable.sum())
    count = np.count_nonzero(beliefs[:, reachable] == 0, axis=1).tolist()
    a95 = next((t for t, cells in enumerate(count) if 20*cells >= 19*total), None)
    coverage = [{"layout": layout, "paid": clock, "correct_free_cells": cells, "reachable_cells": total,
                 "C_free": cells/total} for clock, cells in enumerate(count)]
    checkpoints = []
    for label in CHECKPOINTS:
        expected = 0 if label == "start" else paid if label == "terminal" else int(label.split("_")[-1])
        row = {"layout": layout, "checkpoint": label, "evaluation_paid": None, "requested_paid": expected,
               "status": "not_reached", **dict.fromkeys(METRICS)}
        if expected <= paid:
            directory = run / label
            snap = raw.read(directory / "checkpoint.json")
            require(snap["actual_paid"] == expected and snap["status"] == "captured",
                    f"Checkpoint is missing or belongs to wrong clock: {label}")
            require(all(raw.sha(directory / key) == value for key, value in snap["files_sha256"].items()),
                    f"Checkpoint content changed: {label}")
            with np.load(directory / "belief.npz", allow_pickle=False) as stored:
                require(np.array_equal(beliefs[expected], stored["belief"]), f"Checkpoint grid mismatch: {label}")
            p, r, f1 = raw.direct_metrics(source_scene, fixed_points, directory / "native_mesh.ply")
            c = count[expected]/total
            row.update(zip(METRICS, (c, p, r, f1, c*f1), strict=True))
            row.update(evaluation_paid=expected, status="captured")
        checkpoints.append(row)
    ep = {"layout": layout, "arm": arm, "status": "completed_execution",
          "actual_paid": paid, "A95_paid": a95, "A95_status": "reached" if a95 is not None else "not_reached",
          "actual_return": returned, "collisions": collisions, "guard_rejections": len(summarized_guards),
          "stop_reason": manifest["observed_native_termination"]["reason"],
          "failed": manifest["observed_native_termination"]["failed"],
          "duration_s": manifest["bootstrap_and_task_wall_s"],
          "charged_decision_s": manifest["charged_decision_wall_s"],
          "decision_p50_s": median(timing) if timing else None,
          "external_selections": sum(call["chosen"] is not None for call in calls),
          "external_calls": len(calls), "path_length_m": manifest["path_length_m"],
          **{metric: checkpoints[-1][metric] for metric in METRICS}}
    return ep, checkpoints, coverage


def paired(ep, checkpoints, previous_episode, previous_points):
    layout = ep["layout"]
    pairs = []
    for point in checkpoints:
        label = point["checkpoint"]
        incumbent = previous_points[(layout, "combined_S", label)]
        row = {"layout": layout, "checkpoint": label, "explore_actual_paid": point["evaluation_paid"],
               "nso_actual_paid": incumbent["actual_paid"] or None, "explore_status": point["status"],
               "nso_status": incumbent["status"], "explore_A95": ep.get("A95_paid"),
               "nso_A95": int(previous_episode["A95_paid"]) if previous_episode["A95_paid"] else None}
        row["nso_actions_saved_if_both_hit"] = (row["explore_A95"]-row["nso_A95"]
                                                 if row["explore_A95"] is not None
                                                 and row["nso_A95"] is not None else None)
        for metric in METRICS:
            explore_value = point[metric]
            nso_value = float(incumbent[metric]) if incumbent[metric] else None
            row[f"explore_{metric}"] = explore_value
            row[f"nso_{metric}"] = nso_value
            row[f"nso_minus_explore_{metric}_pp"] = (
                100*(nso_value-explore_value) if explore_value is not None and nso_value is not None else None)
        pairs.append(row)
    return pairs


def write_csv(path, rows):
    names = list(dict.fromkeys(field for row in rows for field in row))
    with path.open("x", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=names)
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", type=Path, default=ROOT / "execution/explore")
    parser.add_argument("--output", type=Path, default=ROOT / "results/explore")
    parser.add_argument("--arm", default=ARM)
    parser.add_argument("--protocol", type=Path, default=PROTOCOL)
    parser.add_argument("--receipt", default="explore_lite_receipt.json")
    args = parser.parse_args()
    protocol = raw.read(args.protocol)
    require(Path(args.receipt).name == args.receipt, "Receipt must be a filename")
    archived_audit = raw.read(FROZEN_RESULTS / "measurement_audit.json")
    require(raw.sha(BASELINE / "FORMAL_FREEZE.json") == archived_audit["run_freeze_sha256"],
            "Published runner freeze changed")
    for name in ("episodes.csv", "checkpoints.csv"):
        require(raw.sha(FROZEN_RESULTS / name) == archived_audit["results_sha256"][name],
                f"Published data changed: {name}")
    old_episodes = indexed(raw.load_csv(FROZEN_RESULTS / "episodes.csv"), ("layout", "arm"))
    old_points = indexed(raw.load_csv(FROZEN_RESULTS / "checkpoints.csv"), ("layout", "arm", "checkpoint"))
    old_audit = indexed(archived_audit["recorded_arm_receipts"], ("layout", "arm"))
    episodes, points, coverage, pairs = [], [], [], []
    for layout in protocol["layouts"]:
        incumbent = old_episodes[(layout, "combined_S")]
        run = args.runs / layout
        completion_path = args.runs / f"{layout}.exit.json"
        completion = raw.read(completion_path) if completion_path.is_file() else {}
        if not (run / args.receipt).is_file() or completion.get("exit_code", 0) != 0:
            episode = {"layout": layout, "arm": args.arm,
                       "status": "missing" if not completion else "technical_failure",
                       "exit_code": completion.get("exit_code"), "completion_path": str(completion_path)}
            captured = [{"layout": layout, "checkpoint": point, "evaluation_paid": None,
                         "status": episode["status"], **dict.fromkeys(METRICS)} for point in CHECKPOINTS]
            curve = []
        else:
            episode, captured, curve = measure(layout, run, incumbent, old_audit[(layout, "combined_S")],
                                                args.arm, args.protocol, args.receipt)
        episodes.append(episode)
        points.extend(captured)
        coverage.extend(curve)
        pairs.extend(paired(episode, captured, incumbent, old_points))
    mid = [row["nso_minus_explore_J_pp"] for row in pairs
           if row["checkpoint"] == "paid_96" and row["nso_minus_explore_J_pp"] is not None]
    complete = [episode for episode in episodes if episode["status"] == "completed_execution"]
    summary = {"arm": args.arm, "registered_floors": len(protocol["layouts"]), "complete_floors": len(complete),
               "prototype_scope": protocol["scope"], "upstream_commit": protocol["upstream_commit"],
               "nso_A96_J_win": sum(diff > 0 for diff in mid), "nso_A96_J_loss": sum(diff < 0 for diff in mid),
               "nso_A96_J_tie": sum(diff == 0 for diff in mid), "paired_A96_J_floors": len(mid),
               "mean_nso_minus_explore_A96_J_pp": mean(mid) if mid else None,
               "inference": "Descriptive paired comparisons on four fixed floors. Individual actions are not independent replicates.",
               "timing": "Wall-clock values were observed in different simulation launches, for descriptive use only.",
               "metric": "Correctly observed reachable free-space fraction times surface F1 within 5 cm",
               "external_protocol_sha256": raw.sha(args.protocol), "raw_measurement_code_sha256": raw.sha(SOURCE),
               "published_measurement_audit_sha256": raw.sha(FROZEN_RESULTS / "measurement_audit.json"),
               "cases": episodes}
    args.output.mkdir(parents=True, exist_ok=False)
    for name, rows in (("episodes", episodes), ("checkpoints", points), ("coverage", coverage), ("pairs", pairs)):
        write_csv(args.output / (name+".csv"), rows)
    with (args.output / "summary.json").open("x") as stream:
        json.dump(summary, stream, indent=2, allow_nan=False)
        stream.write("\n")
    print(json.dumps({key: value for key, value in summary.items() if key != "cases"}, indent=2))


if __name__ == "__main__":
    main()
