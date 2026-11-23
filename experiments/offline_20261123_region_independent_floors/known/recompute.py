#!/usr/bin/env python3
"""Read-only native snapshot and coverage audit of the two published floors."""

import argparse
from collections import defaultdict
import csv
import hashlib
import io
import json
import math
import os
from pathlib import Path
import sys

for variable in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[variable] = "1"
sys.dont_write_bytecode = True

import numpy as np

HERE = Path(__file__).resolve().parent
ANALYSES = HERE.parents[1]
SOURCE = ANALYSES / "region_global_revision_20261122/next/global_attribution"
PUBLISHED = "experiments/offline_20261122_region_global_revision/next/global_attribution"
BASE = "f482722f6cbfc3135358080b08f27274f54cb535"
REMOTE_BLOBS = {
    "FORMAL_PROTOCOL.json": "48e43841f58eaf3eacf97359ecd0bd2eae3d9252",
    "FORMAL_FREEZE.json": "d539411836f1ee93ece735b75d75f3406d72ae79",
    "results/RESULTS.md": "507688096a64eef370ee200bade0b5d3f0084ca5",
    "results/formal/summary.json": "61e9946ec39c57732b87e6e77fb0baa04c775ab8",
    "results/formal/coverage_curve.csv": "1d7020be64ee897e5c8bf9e6b89853b38edd36c8",
    "results/formal/checkpoints.csv": "73134644758bfcbab20f76d5e13d0b1530551ff7",
}

KNOWN = ("attr26_corridor_crosspass", "attr26_loop_bypass")
ARMS = ("combined_S", "F_strict", "region_S")
CLOCKS = ("start", "paid_48", "paid_96", "paid_144", "paid_192", "terminal")
METRICS = ("C_free", "p_scene_05cm", "r_scene_05cm", "F1_scene_05cm", "J")


def read(path):
    return json.loads(path.read_text())


def csv_rows(path):
    with path.open(newline="") as stream:
        return list(csv.DictReader(stream))


def require(condition, description):
    if not condition:
        raise ValueError(description)


def same_number(actual, expected, description):
    require(math.isclose(actual, float(expected), rel_tol=0, abs_tol=1e-12),
            f"{description}: new {actual}, archived {expected}")


def sha(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def blob(path):
    raw = path.read_bytes()
    return hashlib.sha1(f"blob {len(raw)}\0".encode() + raw).hexdigest()


def convert_csv(rows):
    stream = io.StringIO(newline="")
    fields = list(dict.fromkeys(field for row in rows for field in row))
    writer = csv.DictWriter(stream, fields)
    writer.writeheader()
    writer.writerows(rows)
    return stream.getvalue()


def compare_run(layout, arm, worker, published, checkpoints, archived_curve, budget, freeze_sha,
                initial, reference, check, weighted, metric):
    directory = Path(worker["run_path"])
    manifest_path = directory / "manifest.json"
    require(worker["exit_code"] == 0 and worker["freeze_sha256"] == freeze_sha
            and directory == Path(published["run_directory"]) and Path(worker["result_path"]) == manifest_path,
            f"Wrong closed-loop archived attempt: {layout}, {arm}")
    manifest, manifest_sha = weighted.engine.verify_manifest(directory, "manifest.json", worker["result_sha256"])
    require(manifest_sha == published["source_manifest_sha256"]
            and (manifest["layout"], manifest["arm"], manifest["native_budget"]) == (layout, arm, budget)
            and manifest["runner_freeze_sha256"] == freeze_sha
            and manifest["initial_packet_sha256"] == initial["initial_packet_sha256"]
            and manifest["actual_closed_loop"] and not manifest["forced_route_or_goal"]
            and not manifest["evaluator_inputs_read_by_actor"],
            f"Command/context mismatch: {directory}")
    trace = read(directory / "steps.json")
    trace_hash = weighted.native.verify_trace(trace, initial, budget, manifest)
    weighted.native.verify_action_guard(trace, budget)
    require(set(manifest["checkpoints"]) == set(CLOCKS), f"Checkpoint schedule drift: {directory}")
    with np.load(directory / "observed_belief_trace.npz", allow_pickle=False) as data:
        paid = data["paid"]
        beliefs = data["belief"]
    reachable = reference["reachable"]
    denominator = int(reachable.sum())
    require(denominator > 0 and np.array_equal(paid, np.arange(len(trace) + 1))
            and beliefs.shape == (len(trace) + 1, *reachable.shape)
            and len(archived_curve) == len(trace) + 1, f"Coverage trace identity drift: {directory}")
    counts = np.count_nonzero((beliefs == 0) & reachable[None, :, :], axis=(1, 2))
    a95 = next((i for i, n in enumerate(counts) if 100 * int(n) >= 95 * denominator), None)
    require(a95 == published.get("A95_paid"), f"First actual >=95% coverage differs: {directory}")
    curve = []
    for i, n in enumerate(counts):
        old = archived_curve[i]
        require((i, int(n), denominator) == (int(old["paid"]), int(old["correct_free_cells"]),
                                               int(old["reachable_cells"])), f"Coverage divergence at {i}: {directory}")
        measured = int(n) / denominator
        same_number(measured, old["C_free"], f"C_free at paid {i}")
        curve.append({"layout": layout, "arm": arm, "paid": i, "C_free": measured,
                      "correct_free_cells": int(n), "reachable_cells": denominator})

    state = dict(directory=directory, manifest=manifest, trace=trace, initial=initial, belief=beliefs)
    evidence = {}
    records = []
    for name in CLOCKS:
        scheduled = len(trace) if name == "terminal" else 0 if name == "start" else int(name.removeprefix("paid_"))
        old = checkpoints[layout, arm, name]
        if scheduled > len(trace):
            outcome, _ = weighted.native.evaluate_snapshot(directory, name, manifest, initial, trace, reference, metric)
            require(outcome["status"] == old["status"] == "unavailable", f"Actual run padded at {name}: {directory}")
            records.append({"layout": layout, "arm": arm, "checkpoint": name, "status": "unavailable",
                            "actual_paid": None, "requested_paid": scheduled,
                            "source_checkpoint_sha256": outcome["source_checkpoint_sha256"]})
            continue
        observation, _ = check.verify_measurement(weighted, metric, reference, state, name, scheduled)
        receipt = observation["receipt"]
        require(int(old["actual_paid"]) == scheduled and old["status"] == "captured"
                and old["source_checkpoint_sha256"] == observation["checkpoint_sha256"],
                f"Published snapshot mismatch: {directory}/{name}")
        for key in METRICS:
            same_number(observation[key], old[key], f"{directory}/{name} {key}")
        same_number(observation["C_free"], curve[scheduled]["C_free"], f"{directory}/{name} trace coverage")
        same_number(observation["J"], observation["C_free"] * observation["F1_scene_05cm"],
                    f"{directory}/{name} joint")
        item = {"layout": layout, "arm": arm, "checkpoint": name, "status": "captured",
                "actual_paid": scheduled, "requested_paid": scheduled,
                **{key: observation[key] for key in METRICS},
                "correct_free_cells": int(counts[scheduled]), "reachable_cells": denominator,
                "source_checkpoint_sha256": observation["checkpoint_sha256"],
                "native_mesh_sha256": observation["mesh_sha256"],
                "native_mesh_path": str(Path(observation["mesh_path"]).relative_to(ANALYSES)),
                "native_belief_path": str((directory / name / "belief.npz").relative_to(ANALYSES))}
        records.append(item)
        evidence[name] = item
        if name == "terminal":
            require(receipt["packet_chain_sha256"] == trace_hash, f"Terminal packet chain mismatch: {directory}")
            closure = weighted.native.close_record(receipt, read(directory / "runtime_summary.json"), trace, initial)
            for key in ("actual_paid", "returned", "collisions", "stop_reason", "terminal_pose"):
                require(closure[key] == published[key], f"Actual closure differs in {key}: {directory}")

    summary = {"layout": layout, "arm": arm, "A95_paid": a95,
               "A95_status": "reached" if a95 is not None else "not_reached_within_budget",
               "actual_paid": closure["actual_paid"], "returned": closure["returned"],
               "collisions": closure["collisions"], "stop_reason": closure["stop_reason"],
               "terminal_pose": closure["terminal_pose"], "trajectory_path": str((directory/"steps.json").relative_to(ANALYSES)),
               "observed_belief_trace_path": str((directory/"observed_belief_trace.npz").relative_to(ANALYSES)),
               "manifest_sha256": manifest_sha, "steps_sha256": manifest["files_sha256"]["steps.json"],
               "belief_trace_sha256": manifest["files_sha256"]["observed_belief_trace.npz"],
               "initial_packet_sha256": manifest["initial_packet_sha256"],
               "source_reference_receipt_sha256": reference["source_reference_receipt_sha256"],
               "checkpoints": evidence}
    require(summary["source_reference_receipt_sha256"] == published["source_reference_receipt_sha256"],
            f"Reference receipt disagreement: {directory}")
    return summary, records, curve


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--verify", action="store_true", help="Recompute and compare the existing outputs without writing.")
    args = cli.parse_args()
    sys.path.insert(0, str(ANALYSES/"shelf_occlusion_pairs_20261122/measurement"))
    from historical import checks
    check, weighted, metric = checks()

    formal = SOURCE/"results/formal"
    freeze_path = SOURCE/"FORMAL_FREEZE.json"
    phase_path = SOURCE/"execution/formal/phase_receipt.json"
    protocol, freeze, phase, summary = map(read, (SOURCE/"FORMAL_PROTOCOL.json", freeze_path,
                                                   phase_path, formal/"summary.json"))
    freeze_sha = sha(freeze_path)
    require(freeze_sha == phase["freeze_sha256"] == summary["run_freeze_sha256"]
            and sha(phase_path) == summary["phase_receipt_sha256"]
            and sha(SOURCE/"FORMAL_PROTOCOL.json") == freeze["protocol_sha256"] == summary["protocol_sha256"],
            "Known-floor publication, protocol and frozen execution do not bind.")
    require(tuple(protocol["layouts"]) == KNOWN and set(protocol["arms"]) == set(ARMS)
            and protocol["budget"] == freeze["budget"] == 192, "Known-floor registered matrix has changed.")
    for key, digest in REMOTE_BLOBS.items():
        require(blob(SOURCE/key) == digest, f"Source bytes differ from GitHub at {BASE}: {key}")
    file_counts = {key: len(freeze[key]) for key in ("actor_files", "evaluator_files", "actor_initial_files")}
    for key in file_counts:
        for path, digest in freeze[key].items():
            require(sha(Path(path)) == digest, f"Frozen source drift: {path}")
    scene_root = Path(freeze["scene_root"])
    require(scene_root == SOURCE/"scenes", "Scene source changed.")
    weighted.SCENES = scene_root
    checkpoint_lookup = {(r["layout"],r["arm"],r["checkpoint"]):r for r in csv_rows(formal/"checkpoints.csv")}
    coverage_lookup = defaultdict(list)
    for row in csv_rows(formal/"coverage_curve.csv"):
        coverage_lookup[row["layout"],row["arm"]].append(row)
    run_rows = {(r["layout"],r["arm"]):r for r in phase["episodes"]}
    published_run_rows = {(r["layout"],r["arm"]):r for r in summary["episodes"]}
    matrix = {(layout, arm) for layout in KNOWN for arm in ARMS}
    require(matrix == run_rows.keys() == published_run_rows.keys(), "The archived matrix is incomplete.")
    rows, all_snapshots, all_curves = {}, [], []
    for layout in KNOWN:
        initial = read(scene_root/"prepared"/layout/"initial_state.json")
        reference = weighted.load_reference(layout, initial)
        for arm in ARMS:
            k = layout, arm
            row, snapshots, curve = compare_run(*k,run_rows[k],published_run_rows[k],checkpoint_lookup,
                                                coverage_lookup[k],192,freeze_sha,initial,reference,check,weighted,metric)
            rows[k] = row
            all_snapshots.extend(snapshots)
            all_curves.extend(curve)

    pairs = []
    for layout, expected_a95, expected_j96_pp in zip(KNOWN, [(58,112),(81,108)], [10.34,12.41], strict=True):
        revised, frontier = (rows[layout, arm] for arm in ("combined_S","F_strict"))
        revised_96, frontier_96 = (row["checkpoints"]["paid_96"] for row in (revised,frontier))
        revised_ret, frontier_ret = (row["checkpoints"]["terminal"] for row in (revised,frontier))
        actual_pair = (revised["A95_paid"], frontier["A95_paid"])
        j96_pp = 100 * (revised_96["J"] - frontier_96["J"])
        require(actual_pair == expected_a95 and round(j96_pp, 2) == expected_j96_pp,
                f"Previously reported main comparison failed replication on {layout}")
        pairs.append({"layout":layout,"revised_A95":actual_pair[0],"strict_A95":actual_pair[1],
                      "revised_minus_strict_A95":actual_pair[0] - actual_pair[1],
                      "revised_J96":revised_96["J"],"strict_J96":frontier_96["J"],
                      "revised_minus_strict_J96_pp":j96_pp,
                      "revised_return_paid":revised["actual_paid"],"strict_return_paid":frontier["actual_paid"],
                      "revised_return_J":revised_ret["J"],"strict_return_J":frontier_ret["J"],
                      "revised_minus_strict_return_J_pp":100 * (revised_ret["J"] - frontier_ret["J"]),
                      "revised_return_R":revised_ret["r_scene_05cm"],"strict_return_R":frontier_ret["r_scene_05cm"],
                      "revised_minus_strict_return_R_pp":100 * (revised_ret["r_scene_05cm"] - frontier_ret["r_scene_05cm"])})

    audit = {"status":"reproduced_from_archived_native_trajectories_and_meshes",
             "layout_scope":"previously known corridor and loop only",
             "source_base_commit":BASE,"source_repo_directory":PUBLISHED,"pinned_remote_git_blobs":REMOTE_BLOBS,
             "source_freeze_sha256":freeze_sha,"phase_receipt_sha256":sha(phase_path),
             "source_files_sha256":{key:sha(SOURCE/key) for key in REMOTE_BLOBS},
             "frozen_files_rehashed":file_counts,"actual_episodes":len(rows),
             "source_reference_match_m":0.05,"primary_pairs":pairs,"runs":list(rows.values()),
             "source_evaluator_sha256":sha(Path(check.__file__)),
             "matched_coverage_rows":len(all_curves),"remeasured_meshes":sum(r["status"]=="captured" for r in all_snapshots)}
    output = {"PAIRS.csv":convert_csv(pairs),"checkpoints.csv":convert_csv(all_snapshots),
              "coverage_curve.csv":convert_csv(all_curves),"audit.json":json.dumps(audit,indent=2,allow_nan=False)+"\n"}
    for filename, content in output.items():
        destination = HERE/filename
        if args.verify:
            require(destination.read_bytes() == content.encode(), f"Remeasured product changed: {filename}")
        else:
            with destination.open("x", newline="") as stream:
                stream.write(content)
    print(json.dumps({"main_pairs":pairs,"verified_frozen_files":file_counts,
                      "remeasured_meshes":audit["remeasured_meshes"]},indent=2))


if __name__ == "__main__":
    main()
