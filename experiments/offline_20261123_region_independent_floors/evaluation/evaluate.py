#!/usr/bin/env python3
"""Evaluate the frozen twelve-run matrix and independently audit clocks and pairing."""

import argparse
from collections import defaultdict
import csv
from itertools import combinations
import json
import math
import os
from pathlib import Path
import statistics
import sys

for variable in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[variable] = "1"

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
ANALYSES = ROOT.parent
ARCHIVE = ANALYSES / "region_global_revision_20261122/next/global_attribution"
ORIGINAL = ANALYSES / "region_structure_floor_20261122"
sys.path.insert(0, str(ORIGINAL))
from regression import import_file, read, sha


ARMS = ("region_S", "combined_S", "F_strict")
CAP = 192
CHECKPOINTS = (("start", 0), ("paid_48", 48), ("paid_96", 96),
               ("paid_144", 144), ("paid_192", 192), ("terminal", None))
METRICS = ("C_free", "p_scene_05cm", "r_scene_05cm", "F1_scene_05cm", "J")
CONTRASTS = (("revised_vs_strict", "combined_S", "F_strict"),
             ("revised_vs_original", "combined_S", "region_S"))
FORWARD = ((-1, 0), (0, 1), (1, 0), (0, -1))
BASE_COMMIT = "f482722f6cbfc3135358080b08f27274f54cb535"


def require(condition, detail):
    if not condition:
        raise ValueError(detail)


def agree(actual, expected, detail, tolerance=1e-12):
    require(math.isclose(actual, expected, abs_tol=tolerance, rel_tol=0), detail)


def write_rows(path, rows, leading=()):
    fields = list(dict.fromkeys([*leading, *(key for row in rows for key in row)]))
    with path.open("x", newline="") as stream:
        writer = csv.DictWriter(stream, fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: json.dumps(v, ensure_ascii=False, allow_nan=False, sort_keys=True)
                             if isinstance(v, (list, dict)) else v for k, v in row.items()})


def write_json(path, document):
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("x") as stream:
        json.dump(document, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")
    temporary.replace(path)


def input_contract(native, root, known_smoke):
    native.ROOT = root
    frozen, spec, digest, jobs, receipts, phase_path = native.validate_inputs("formal", False)
    layouts = spec["layouts"]
    require(len(layouts) == (2 if known_smoke else 4) and len(set(layouts)) == len(layouts),
            "Scope is exactly four new layouts (or two known smoke-check layouts).")
    require(tuple(spec["arms"]) == ARMS and len(jobs) == 3 * len(layouts) and set(jobs) == set(receipts),
            "The finished receipt must cover all scheduled policies exactly once.")
    require(frozen["budget"] == CAP and frozen["capture_paid"] == [p for _, p in CHECKPOINTS if p is not None],
            "Mismatch in native charged budget or audit checkpoints.")
    require(frozen["local_auxiliary_weights"] == {"unknown_weight": .25, "action_cost": .2},
            "Unexpected earlier local score coefficients.")
    if not known_smoke:
        registered = read(root/"PROTOCOL.json")
        require(registered["base_commit"] == BASE_COMMIT and registered["layouts"] == layouts,
                "New schedule does not match the originally registered base or floors.")
    require(not any("evaluator_references" in Path(p).parts or "evaluator_truth" in Path(p).stem
                    for p in frozen["actor_files"]),
            "Independent scoring reference appears among actor inputs.")
    for (layout, arm), job in jobs.items():
        strict = arm == "F_strict"
        repair = {"target": True, "outbound": True} if arm == "combined_S" else None
        require(job["prior"] == (None if strict else "S") and job["regional_enabled"] == (not strict)
                and job["literal"] and job.get("repair") == repair and not job.get("branch"),
                f"Unexpected policy intervention in {layout}/{arm}")
    return frozen, spec, digest, jobs, receipts, phase_path


def checked_worker(root, worker, digest, frozen_unix_s):
    directory = Path(worker["run_path"])
    expected = root / "execution/formal" / worker["layout"] / worker["arm"]
    require(directory.name == "run" and directory.parent.name.startswith("attempt_")
            and directory.parent.parent.resolve() == expected.resolve(), f"Unregistered path: {directory}")
    launch = directory.parent / "launch.json"
    exit_file = directory.parent / "exit.json"
    log = directory.parent / "worker.log"
    started = read(launch)
    require(all(worker.get(k) == v for k, v in started.items()) and read(exit_file) == worker,
            f"Launch/exit evidence diverged from phase receipt: {directory}")
    require(started["freeze_sha256"] == digest and started["started_unix_s"] >= frozen_unix_s
            and worker["observed_exit_unix_s"] >= started["started_unix_s"]
            and sha(log) == worker["worker_log_sha256"], f"Unbound frozen attempt: {directory}")
    require(worker["result_path"] is None or Path(worker["result_path"]).parent == directory,
            f"Worker result does not belong to its attempt: {directory}")
    return directory, {"launch_path": str(launch), "launch_sha256": sha(launch), "exit_path": str(exit_file),
                       "exit_sha256": sha(exit_file), "worker_log_path": str(log),
                       "worker_log_sha256": worker["worker_log_sha256"], "pid": worker["pid"],
                       "worker_wall_s": worker["observed_exit_unix_s"] - started["started_unix_s"]}


def verify_primitives(trace, initial, context):
    pose = tuple(initial)
    n_turns = n_forward = n_logged_return = collisions = 0
    distance = 0.0
    for expected, row in enumerate(trace, 1):
        action = row["action"]
        require(row["step"] == expected and action in ("forward", "left", "right")
                and (row["phase_before_action"] in ("outbound", "return")),
                f"Unaccounted paid primitive {context} clock={expected}")
        row0, col0, heading = pose
        after = (*row["position"], row["heading"])
        require(0 <= heading < 4 and 0 <= after[2] < 4 and row["guard_allowed"] is True,
                f"Uncertified primitive at {context} clock={expected}")
        if action == "forward":
            dr, dc = FORWARD[heading]
            expect = pose if row["collision"] else (row0 + dr, col0 + dc, heading)
            n_forward += 1
            distance += not row["collision"]
        else:
            expect = (row0, col0, (heading + (1 if action == "right" else -1)) % 4)
            n_turns += 1
        require(after == expect, f"Observed transition disagrees with native primitive: {context}, {expected}")
        collisions += bool(row["collision"])
        n_logged_return += row["phase_before_action"] == "return"
        pose = after
    return {"charged_forward_actions": n_forward, "charged_turn_actions": n_turns,
            "logged_return_phase_actions": n_logged_return, "translation_cells": int(distance),
            "collision_actions": collisions}


def coverage_check(rows, trace, directory, reference, first):
    """Cross-check the inherited vectorized estimator by independently indexing true cells."""
    reachable = reference["reachable"]
    true_cells = np.flatnonzero(reachable)
    total = len(true_cells)
    require(total and len(rows) == len(trace) + 1, f"Incomplete paid coverage trace: {directory}")
    with np.load(directory / "observed_belief_trace.npz", allow_pickle=False) as state:
        ticks, belief = state["paid"], state["belief"]
    require(np.array_equal(ticks, np.arange(len(trace)+1))
            and belief.shape == (len(ticks), *reachable.shape), f"Measured occupancy is not aligned: {directory}")
    hits = []
    threshold = (95 * total + 99) // 100
    for k, (estimate, row) in enumerate(zip(belief, rows, strict=True)):
        count = int((estimate.flat[true_cells] == 0).sum())
        require(row["paid"] == k and row["correct_free_cells"] == count and row["reachable_cells"] == total,
                f"Correct-free recount differs at {directory}, {k}")
        agree(row["C_free"], count / total, f"Area fraction mismatch at {directory}, {k}")
        if count >= threshold:
            hits.append(k)
    require(first == (hits[0] if hits else None), f"Wrong first 95 percent crossing: {directory}")
    return {"reachable_cells": total, "threshold_integer_cells": threshold, "independent_steps_recounted": len(ticks)}


def enrich_cloud(row, path, reference, coverage, episode, use_for_comparison):
    row = row.copy()
    name, status = row["checkpoint"], row["status"]
    row["checkpoint_receipt_path"] = str(path/"checkpoint.json") if path else None
    row["fused_mesh_path"] = str(path/"native_mesh.ply") if path and status == "captured" else None
    row["observed_belief_path"] = str(path/"belief.npz") if path and status == "captured" else None
    row["source_reference_path"] = str(reference)
    row["eligible_for_paired_comparison"] = bool(use_for_comparison and status == "captured")
    row["checkpoint_note"] = ("captured" if status == "captured" and use_for_comparison else
                              "partial_capture_from_incomplete_technical_attempt" if status == "captured" else status)
    if status == "captured" and use_for_comparison:
        tick = row["actual_paid"]
        require(coverage[tick]["correct_free_cells"] == row["correct_free_cells"]
                and coverage[tick]["reachable_cells"] == row["reachable_cells"], f"Checkpoint mismatch: {path}")
        precision, recall = row["p_scene_05cm"], row["r_scene_05cm"]
        expect = 2*precision*recall/(precision+recall) if precision+recall else 0
        require(all(math.isfinite(row[k]) and 0 <= row[k] <= 1 for k in METRICS),
                f"Invalid native map metric: {path}")
        agree(row["F1_scene_05cm"], expect, f"F1 does not match precision/recall: {path}")
        agree(row["J"], row["C_free"] * row["F1_scene_05cm"], f"Joint score mismatch: {path}")
        if name == "terminal":
            require(tick == episode["actual_paid"], f"Actual terminal clock drift: {path}")
    if status == "unavailable":
        require(row["requested_paid"] > episode["actual_paid"], f"Missing a checkpoint before stop: {path}")
        row.update(checkpoint_note="native_terminal_before_checkpoint", actual_paid=None)
    return row


def guards(directory, case, cap, actual):
    observations = []
    for name in ("audit.json", "regional_guards.json", "frontier_events.json"):
        file = directory / name
        if not file.is_file():
            continue
        for index, event in enumerate(read(file)):
            kind = event["event"]
            if not any(x in kind for x in ("guard_", "rejection", "rejected", "blocked")) and "motion" not in event:
                continue
            paid = event.get("paid_before", event.get("paid", event.get("frame_id")))
            paid = int(paid.removeprefix("frame-")) if isinstance(paid, str) else paid
            require(paid is None or 0 <= paid <= actual <= cap,
                    f"Guard evidence clocks outside paid trace: {file}/{index}")
            assess = event.get("assessment", {})
            observations.append(case | {"record_kind": "native_event", "source_file": str(file),
                                 "source_row_index": index, "event": kind, "paid": paid,
                                 "reason": assess.get("reason", event.get("reason")),
                                 "phase": event.get("phase"), "action": assess.get("action", event.get("motion")),
                                 "event_data": event})
    return observations


def check_policy_trace(job, directory, manifest):
    strict = job["arm"] == "F_strict"
    expected = (None,None) if strict else ("G","S")
    require((manifest["global_mode"],manifest["local_mode"]) == expected,
            f"Native execution motors used unexpected selector modes: {directory}")
    region = read(directory/"regional_choices.json")
    require(not (strict and region), f"Strict frontier unexpectedly acquired region proposals: {directory}")
    repair_count = 0
    for event in region:
        ranking = event["ranking"]
        repair = ranking.get("attribution_path_repair")
        required = job["arm"] == "combined_S" and bool(ranking.get("ranking"))
        require(bool(repair) == required, f"Wrong regional score branch at {directory}, {event['paid_before']}")
        if repair:
            require(repair["score_repair"] == {"target":True,"outbound":True}
                    and repair["actual_paid"] == event["paid_before"]
                    and repair["ranking"] == ranking["ranking"],
                    f"Revised region score receipt is inconsistent: {directory}")
            repair_count += 1
    return repair_count


def check_success(ev, job, directory, episode, parts, binding, scene_root):
    case = {"layout": job["layout"], "arm": job["arm"]}
    base = scene_root / "evaluator_references" / job["layout"]
    initial = read(scene_root / "prepared" / job["layout"] / "initial_state.json")
    reference = ev.load_reference(job["layout"], initial)
    trace, raw = binding["trace"], binding["raw"]
    manifest = read(directory/"manifest.json")
    require(episode["actual_paid"] == manifest["paid_actions"] == len(trace) <= CAP
            and metadata_identity(initial, raw, binding["start"], manifest),
            f"Cold identity or budget mismatch: {directory}")
    episode["recorded_combined_score_repairs"] = check_policy_trace(job,directory,manifest)
    primitives = verify_primitives(trace, initial["anchor"], directory)
    require(primitives["charged_forward_actions"] + primitives["charged_turn_actions"] == len(trace)
            and primitives["collision_actions"] == episode["collisions"], f"Motion count drift: {directory}")
    agree(episode["path_length_m"], primitives["translation_cells"] * initial["grid_resolution_m"],
          f"Independent odometry length differs: {directory}", tolerance=1e-7)
    decision_samples = [row["decision_wall_s"] for row in trace]
    decision_sum = math.fsum(decision_samples)
    if "charged_decision_wall_s" in manifest:
        agree(decision_sum, manifest["charged_decision_wall_s"],
              f"Actual charged planning time differs: {directory}", tolerance=1e-7)
    coverage = parts["coverage_curve"]
    recount = coverage_check(coverage, trace, directory, reference, episode["A95_paid"])
    episode.update(primitives | recount)
    native_guards = guards(directory, case, CAP, len(trace))
    episode.update({"A95_status": "reached" if episode["A95_paid"] is not None else "not_reached_within_budget",
                    "native_stop_evaluated": True, "charged_decision_wall_s": decision_sum,
                    "native_guard_event_rows": len(native_guards), "terminal_requested_budget": CAP,
                    "returned_to_original_heading": episode["returned"],
                    "trajectory_path": str(directory/"steps.json"), "coverage_trace_path": str(directory/"observed_belief_trace.npz"),
                    "manifest_path": str(directory/"manifest.json"), "runtime_summary_path": str(directory/"runtime_summary.json"),
                    "reference_receipt_path": str(base/"RECEIPT.json"),
                    "reference_samples_path": str(base/"fixed_reference.npz"), "reference_free_grid_path": str(base/"evaluator_truth.npz")})
    parts["checkpoints"] = [enrich_cloud(r, directory/r["checkpoint"], base/"RECEIPT.json", coverage,
                                         episode, True) for r in parts["checkpoints"]]
    native_guards.append(case | {"record_kind": "actual_terminal", "source_file": str(directory/"runtime_summary.json"),
                         "event": "native_termination", "paid": len(trace), "reason": episode["stop_reason"],
                         "event_data": episode["termination"]})
    parts["guard_events"] = native_guards
    return {"terminal_trace_path": str(directory/"steps.json"), "terminal_trace_sha256": sha(directory/"steps.json"),
            "coverage_trace_path": str(directory/"observed_belief_trace.npz"),
            "coverage_trace_sha256": sha(directory/"observed_belief_trace.npz"),
            "manifest_sha256": episode["source_manifest_sha256"], "elapsed_s": episode["elapsed_s"],
            "cold": {"raw": raw, "mesh_fields": binding["start"]["native_mesh_fields"],
                     "start_belief_sha256": binding["start"]["belief_sha256"]},
            "coverage_recount": recount, "successful_return": episode["returned"], "actual_paid": len(trace),
            "reference_files_sha256": {str(p):sha(p) for p in sorted(base.iterdir()) if p.is_file()}}


def metadata_identity(initial, raw, start, manifest):
    return (start["actual_paid"] == 0 and start["pose"] == initial["anchor"]
            and initial["initial_packet_sha256"] == raw["packet_sha256"] == start["packet_sha256"] == manifest["initial_packet_sha256"]
            and start["belief_sha256"] == manifest["initial_belief_sha256"])


def check_failure(job, episode, parts, directory, scene_root):
    episode.update(A95_status="technical_failure", native_stop_evaluated=False,
                   manifest_path=None, trajectory_path=None, returned=None)
    by_checkpoint = {r["checkpoint"]: r for r in parts.get("checkpoints", [])}
    rows = []
    ref = scene_root/"evaluator_references"/job["layout"]/"RECEIPT.json"
    for name, paid in CHECKPOINTS:
        row = by_checkpoint.get(name, {"layout":job["layout"], "arm":job["arm"], "checkpoint":name,
                                       "status":"technical_missing", "requested_paid":paid, "actual_paid":None})
        rows.append(enrich_cloud(row, directory/name if name in by_checkpoint else None, ref, [], episode, False))
    parts["checkpoints"] = rows
    return {"technical_failure": True, "attempt_path": str(directory.parent),
            "partial_checkpoints_not_used_for_pairing": sorted(by_checkpoint)}


def a95_state(left, right):
    a, b = left["A95_status"], right["A95_status"]
    if "technical_failure" in (a, b):
        return "not_comparable_technical_failure", None
    if a != b:
        return ("left_threshold_only" if a == "reached" else "right_threshold_only"), None
    if a != "reached":
        return "neither_reached_within_budget", None
    saved = right["A95_paid"] - left["A95_paid"]
    return ("left_fewer_actions" if saved > 0 else "right_fewer_actions" if saved < 0 else "equal_actions"), saved


def pair_rows(layouts, episodes, checkpoints):
    pairs, clock_pairs, detail = [], [], {}
    table = {(r["layout"], r["arm"], r["checkpoint"]):r for r in checkpoints}
    for contrast, left, right in CONTRASTS:
        trial = []
        for layout in layouts:
            a, b = episodes[layout,left], episodes[layout,right]
            identity = {"layout":layout,"contrast":contrast,"left_arm":left,"right_arm":right}
            delta = {}
            for label, requested in CHECKPOINTS:
                x, y = table[layout,left,label], table[layout,right,label]
                measured = x["eligible_for_paired_comparison"] and y["eligible_for_paired_comparison"]
                change = {f"delta_{k}_pp":100*(x[k]-y[k]) if measured else None for k in METRICS}
                delta[label] = row = identity | {"checkpoint":label,"requested_paid":requested,
                    "left_actual_paid":x.get("actual_paid") if x["eligible_for_paired_comparison"] else None,
                    "right_actual_paid":y.get("actual_paid") if y["eligible_for_paired_comparison"] else None,
                    "left_checkpoint_status":x["checkpoint_note"],"right_checkpoint_status":y["checkpoint_note"],
                    "paired_available":measured,"equal_actual_paid":bool(measured and x["actual_paid"]==y["actual_paid"]),
                    "both_returned_on_terminal_pair":(bool(measured and a["returned"] and b["returned"])
                                                       if label=="terminal" else None)
                } | {f"{side}_{k}":v.get(k) if v["eligible_for_paired_comparison"] else None
                     for side,v in (("left",x),("right",y)) for k in METRICS} | change
                clock_pairs.append(row)
            threshold, saving = a95_state(a,b)
            pair = identity | {"A95_outcome":threshold,"saved_actions_left_relative_to_right":saving,
                "revised_actions_saved":saving,"left_A95_paid":a.get("A95_paid"),"right_A95_paid":b.get("A95_paid"),
                "left_A95_status":a["A95_status"],"right_A95_status":b["A95_status"],
                "left_terminal_coverage":a.get("C_free"),"right_terminal_coverage":b.get("C_free"),
                "left_returned":a.get("returned"),"right_returned":b.get("returned"),
                "left_actual_terminal_paid":a.get("actual_paid"),"right_actual_terminal_paid":b.get("actual_paid"),
                "left_terminal_reason":a.get("stop_reason"),"right_terminal_reason":b.get("stop_reason"),
                "left_collisions":a.get("collisions"),"right_collisions":b.get("collisions"),
                "terminal_equal_actual_paid":delta["terminal"]["equal_actual_paid"],
                "both_returned":delta["terminal"]["both_returned_on_terminal_pair"],
                **{f"A96_{k}":v for k,v in delta["paid_96"].items() if k.startswith("delta_")},
                **{f"return_{k}":v if delta["terminal"]["both_returned_on_terminal_pair"] else None
                   for k,v in delta["terminal"].items() if k.startswith("delta_")}}
            pairs.append(pair)
            trial.append(pair)
        detail[contrast] = paired_aggregate(trial, [x for x in clock_pairs if x["contrast"]==contrast])
    return pairs, clock_pairs, detail


def summarize_values(values):
    samples = [x for x in values if x is not None]
    return {"n":len(samples),"mean":statistics.fmean(samples) if samples else None,
            "median":statistics.median(samples) if samples else None,"by_floor":values}


def paired_aggregate(arms, clocks):
    floors = len(arms)
    available = [r for r in clocks if r["checkpoint"]=="paid_96"]
    terminal = [r for r in clocks if r["checkpoint"]=="terminal"]
    saved = [r["revised_actions_saved"] for r in arms]
    eligible = [r for r in arms if r["revised_actions_saved"] is not None
                and r["revised_actions_saved"] > 0 and r["both_returned"]]
    return {"scheduled_floor_count":floors,
            "left_A95_reached_count":sum(r["left_A95_status"]=="reached" for r in arms),
            "right_A95_reached_count":sum(r["right_A95_status"]=="reached" for r in arms),
            "both_A95_reached_count":sum(x is not None for x in saved),
            "left_faster_both_reached_count":sum(x is not None and x > 0 for x in saved),
            "right_faster_both_reached_count":sum(x is not None and x < 0 for x in saved),
            "equal_both_reached_count":saved.count(0),
            "threshold_outcomes":{s:sum(x["A95_outcome"]==s for x in arms)
                                  for s in sorted({x["A95_outcome"] for x in arms})},
            "saved_actions_among_both_reached":summarize_values(saved),
            "A96_paired":{k:summarize_values([x[f"delta_{k}_pp"] for x in available]) for k in METRICS},
            "actual_return_paired":{k:summarize_values([x[f"delta_{k}_pp"] if x["both_returned_on_terminal_pair"] else None
                                                       for x in terminal]) for k in METRICS},
            "faster_and_both_returned_floors":[r["layout"] for r in eligible],
            "faster_and_lower_return_recall_floors":[r["layout"] for r in eligible
                                                    if r["return_delta_r_scene_05cm_pp"] < -1e-10]}


def evaluate(root, output, known_smoke):
    require(not output.exists(), f"Use a new evaluation output location: {output}")
    native = import_file("old_regional_floor_evaluation", ORIGINAL/"evaluate.py")
    frozen, spec, digest, jobs, receipts, phase_path = input_contract(native, root, known_smoke)
    scene_root = Path(frozen["scene_root"])
    ev = import_file("old_regional_weighted_for_new_floors", native.WEIGHTED_METRIC)
    ev.SCENES = scene_root
    world = import_file("new_frozen_floor_truth", scene_root/"scene.py")
    metric = ev.files.load_metric(read(ev.files.SCENES/"freeze.json"), read(ev.files.OLD_EVAL/"freeze.json"))
    output.mkdir(parents=True)
    episodes, bindings, runs, tables = {}, {}, {}, defaultdict(list)
    for layout in spec["layouts"]:
        for arm in ARMS:
            key = (layout,arm)
            worker, job = receipts[key], jobs[key]
            directory, transport = checked_worker(root, worker, digest, frozen["frozen_unix_s"])
            value, parts, binding = native.measure_episode(ev,directory,job,worker,frozen,digest,scene_root,world,metric)
            value.update(transport)
            if value["status"] == "completed_execution":
                proof = check_success(ev, job, directory, value, parts, binding, scene_root)
                bindings[key] = binding
            else:
                proof = check_failure(job, value, parts, directory, scene_root)
                bindings[key] = None
            episodes[key] = value
            runs[key] = {"layout":layout,"arm":arm} | transport | proof
            for title, rows in parts.items():
                tables[title].extend(rows)
            print(f"{layout} {arm}: {value['status']}, A95={value.get('A95_paid')}, "
                  f"final={value.get('actual_paid')}, return={value.get('returned')}",flush=True)
    for layout in spec["layouts"]:
        group = [runs[layout,arm] for arm in ARMS if "cold" in runs[layout,arm]]
        require(not group or all(r["cold"] == group[0]["cold"] and
                                  r["reference_files_sha256"] == group[0]["reference_files_sha256"] for r in group),
                f"Paired arms do not bind exact common cold state/reference: {layout}")
    tables["episodes"] = list(episodes.values())
    tables["first_divergence"] = [native.pair_divergence(ev, layout, names, bindings)
                                  for layout in spec["layouts"] for names in combinations(ARMS,2)]
    tables["pairs"], tables["paired_checkpoints"], aggregates = pair_rows(spec["layouts"],episodes,tables["checkpoints"])
    for name, rows in tables.items():
        write_rows(output/f"{name}.csv",rows, leading=("layout", "arm") if name!="pairs" and name!="paired_checkpoints" else ("layout","contrast"))
    result = {"schema":"nso_new_paired_floor_metrics_v1","phase":"formal",
              "scope":"known_archive_smoke" if known_smoke else "four_new_frozen_floors",
              "base_commit":BASE_COMMIT if not known_smoke else None,
              "run_freeze_sha256":digest,"protocol_sha256":frozen["protocol_sha256"],
              "phase_receipt_sha256":sha(phase_path),"preregistered_budget":CAP,"independent_unit":"floor",
              "registered_floors":spec["layouts"],"arms":list(ARMS),"frozen_trials":len(jobs),
              "completed_native_runs":sum(e["status"]=="completed_execution" for e in episodes.values()),
              "technical_failures":sum(e["status"]=="technical_failure" for e in episodes.values()),
              "per_arm":{a:{"scheduled":len(spec["layouts"]),
                            "native_completed":sum(e["status"]=="completed_execution" for (_, arm),e in episodes.items() if arm==a),
                            "actual_returns":sum(bool(e.get("returned")) for (_, arm),e in episodes.items() if arm==a),
                            "collisions_observed":sum(e.get("collisions",0) for (_, arm),e in episodes.items() if arm==a),
                            "A95_reached":sum(e["A95_status"]=="reached" for (_, arm),e in episodes.items() if arm==a)}
                         for a in ARMS},
              "episodes":tables["episodes"],"pairs":tables["pairs"],"contrast_aggregates":aggregates,
              "results_sha256":{p.name:sha(p) for p in sorted(output.glob("*.csv"))}}
    write_json(output/"summary.json",result)
    audit = {"schema":"nso_new_floor_independent_raw_measurement_v1",
             "summary_sha256":sha(output/"summary.json"),"results_sha256":result["results_sha256"],
             "run_freeze_sha256":digest,"phase_receipt_sha256":sha(phase_path),
             "new_floor_evaluator_path":str(Path(__file__).resolve()),"new_floor_evaluator_sha256":sha(Path(__file__)),
             "inherited_region_evaluator_sha256":sha(ORIGINAL/"evaluate.py"),
             "inherited_weighted_evaluator_sha256":sha(native.WEIGHTED_METRIC),
             "published_global_attribution_evaluator_sha256":sha(ARCHIVE/"measure.py"),
             "recorded_arm_receipts":[runs[key] for key in jobs]}
    write_json(output/"measurement_audit.json",audit)
    print(f"Verified {result['completed_native_runs']}/{result['frozen_trials']} native episodes and "
          f"{len(tables['coverage_curve'])} paid coverage states to {output}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path)
    parser.add_argument("--known-smoke", action="store_true",
                        help="Verify the new measuring wrapper on two archived floors only; requires a new --out.")
    args = parser.parse_args()
    if args.known_smoke and args.out is None:
        parser.error("--known-smoke requires a separate --out")
    root = ARCHIVE if args.known_smoke else ROOT
    evaluate(root, (args.out or root/"results/formal").resolve(), args.known_smoke)


if __name__ == "__main__":
    main()
