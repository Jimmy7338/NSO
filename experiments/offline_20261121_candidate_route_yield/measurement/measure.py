"""Evaluate fixed candidate-route snapshots offline against the archived scene reference."""

import argparse
import csv
import hashlib
import importlib.util
import json
import math
import os
import sys
from pathlib import Path

for variable in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(variable, "1")

import numpy as np
import open3d as o3d

STUDY = Path(__file__).resolve().parents[1]
ARCHIVE = STUDY.parent / "semantic_large_scene_20261121"
SCENES = ARCHIVE / "mechanism/scenes"
HISTORY = ARCHIVE / "mechanism/runner"
OLD_EVAL = ARCHIVE / "evaluation"
NATIVE_METRIC = STUDY.parent / "semantic_scene_redesign_20261119/exploration_evaluation/surface_reference.py"
GROUPS = {"shelf": "open31", "cabinet": "closed118"}
THRESHOLD_M = 0.05
MODES = ("G", "S", "X")
CHECKPOINT_FIELDS = (
    "status", "actual_paid", "requested_paid", "C_free", "correct_free_cells", "reachable_cells",
    "F1_scene_05cm", "p_scene_05cm", "r_scene_05cm", "J", "delta_C_free", "delta_F1_scene_05cm",
    "delta_J", "object_gross_m2", "object_lost_m2", "object_net_m2",
    "shelf_gross_m2", "shelf_lost_m2", "shelf_net_m2",
    "cabinet_gross_m2", "cabinet_lost_m2", "cabinet_net_m2",
)
RECON_METRIC = STUDY.parent / (
    "reproduction_20261104/simulation_publication_stage/experiments/"
    "unknown_exploration_20261104/source/utils/reconstruction_metrics.py"
)
NATIVE_EXECUTION_FREEZE = STUDY.parent / "paired_unseen_gain_20261120/execution/freeze.json"


def read_json(path):
    return json.loads(Path(path).read_text())


def sha256(path):
    with Path(path).open("rb") as file:
        return hashlib.file_digest(file, "sha256").hexdigest()


def canonical_sha256(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode()).hexdigest()


def chain_sha256(previous, packet):
    return hashlib.sha256(bytes.fromhex(previous) + bytes.fromhex(packet)).hexdigest()


def verify(path, checksum):
    if sha256(path) != checksum:
        raise ValueError(f"Provenance mismatch in {path}")


def verify_bound(path, mapping):
    checksum = mapping[str(path)]
    verify(path, checksum)
    return checksum


def read_belief(path):
    with np.load(path, allow_pickle=False) as data:
        return data["belief"], [*map(int, data["position"]), int(data["heading"])]


def read_mesh(path):
    mesh = o3d.io.read_triangle_mesh(str(path))
    vertices = np.asarray(mesh.vertices)
    if not np.isfinite(vertices).all():
        raise ValueError(f"Nonfinite vertex in {path}")
    return mesh


def load_reference(layout, scene_freeze, runner_freeze):
    reference_dir = SCENES / "evaluator_references" / layout
    receipt_path = reference_dir / "RECEIPT.json"
    samples_path = reference_dir / "fixed_reference.npz"
    source_path = reference_dir / "source_mesh.ply"
    reachable_path = SCENES / "prepared" / layout / "evaluator_truth.npz"
    initial_path = HISTORY / "layouts" / layout / "prepared/initial_state.json"
    files_sha256 = {
        str(path): verify_bound(path, scene_freeze["files_sha256"])
        for path in (receipt_path, samples_path, source_path, reachable_path)
    }
    files_sha256[str(initial_path)] = verify_bound(initial_path, runner_freeze["files_sha256"])
    receipt = read_json(receipt_path)
    if (receipt["layout"] != layout or receipt["reference_npz_sha256"] != files_sha256[str(samples_path)]
            or receipt["source_mesh_sha256"] != files_sha256[str(source_path)]):
        raise ValueError(f"Reference receipt disagrees with old freeze: {layout}")
    verify(reachable_path, receipt["source_inputs_sha256"][str(reachable_path)])
    initial = read_json(initial_path)
    with np.load(samples_path, allow_pickle=False) as samples_file:
        samples = {}
        weights = {}
        for label, key in {"scene": "all", **GROUPS}.items():
            points = np.asarray(samples_file[key], dtype=np.float32)
            metadata = receipt["metadata"][key]
            weight = metadata["area_per_sample_m2"]
            if points.shape != (metadata["visible_samples"], 3) or not len(points):
                raise ValueError(f"Unexpected fixed reference dimensions for {layout}: {key}")
            if not np.isfinite(points).all() or not math.isclose(
                    weight * metadata["drawn_samples"], metadata["sampled_source_area_m2"]):
                raise ValueError(f"Invalid sampled reference measure for {layout}: {key}")
            samples[label] = points
            weights[label] = weight
    with np.load(reachable_path, allow_pickle=False) as truth_file:
        reachable = truth_file["reachable"]
    if len(initial["shape"]) != 2 or list(reachable.shape) != initial["shape"]:
        raise ValueError(f"Initial pose and reachable grid disagree: {layout}")
    if receipt["reachable_cells"] != int(reachable.sum()):
        raise ValueError(f"Fixed reference denominator inconsistent with receipt: {layout}")
    source_mesh = read_mesh(source_path)
    if not len(source_mesh.triangles):
        raise ValueError(f"Missing source triangles: {source_path}")
    return {
        "samples": samples, "weights": weights, "source_mesh": source_mesh, "reachable": reachable,
        "initial": initial, "files_sha256": files_sha256,
        "receipt_visible_area_m2": {label: weights[label] * len(samples[label]) for label in GROUPS},
    }


def load_archive(layout, runner_freeze_sha):
    attempt = HISTORY / "layouts" / layout / "runs/G/attempt_001"
    archive_manifest_path = attempt / "artifact_manifest.json"
    manifest = read_json(archive_manifest_path)
    if manifest["freeze_sha256"] != runner_freeze_sha:
        raise ValueError(f"Archive runtime freeze hash mismatch: {layout}")
    for name in ("intervention.json", "steps.json", "initial_evaluation.json", "choices.json"):
        verify(attempt / name, manifest["files_sha256"][name])
    intervention = read_json(attempt / "intervention.json")["intervention"]
    if intervention["paid_before"] != 0 or intervention["mode"] != "G" or not intervention["qualified"]:
        raise ValueError(f"Archived decision is not the starting intervention: {layout}")
    archive_steps = read_json(attempt / "steps.json")
    selected_id = intervention["chosen_candidate_id"]
    candidate_summaries = {item["candidate_id"]: item for item in intervention["candidate_summaries"]}
    if set(candidate_summaries) != {0, 1, 2, 3}:
        raise ValueError(f"Archived candidate IDs changed: {layout}")
    return {
        "intervention": intervention, "candidate_summaries": candidate_summaries, "archive_steps": archive_steps,
        "selected_id": selected_id, "prefix_summary": read_json(attempt / "initial_evaluation.json"),
        "first_choice": read_json(attempt / "choices.json")[0],
        "manifest_sha256": sha256(archive_manifest_path),
        "intervention_sha256": manifest["files_sha256"]["intervention.json"],
        "choices_sha256": manifest["files_sha256"]["choices.json"],
        "steps_sha256": manifest["files_sha256"]["steps.json"],
    }


def measure_geometry(mesh, belief, reference, metric):
    free, total = metric.count_free(belief, reference["reachable"])
    samples = reference["samples"]
    surfaces = metric.evaluate_surface(
        mesh, reference["source_mesh"],
        {"all": samples["scene"], "open31": samples["shelf"], "closed115_fronts": samples["cabinet"]},
        thresholds=(THRESHOLD_M,),
    )
    predicted = metric.ray_scene(mesh) if surfaces["predicted_surface_samples"] else None
    represented = {
        name: predicted.compute_distance(o3d.core.Tensor(samples[name]), nthreads=1).numpy() <= THRESHOLD_M
        if predicted is not None else np.zeros(len(samples[name]), dtype=bool)
        for name in GROUPS
    }
    coverage = free / total
    f1 = float(surfaces["f1_05cm"])
    return {
        "C_free": coverage, "correct_free_cells": free, "reachable_cells": total, "F1_scene_05cm": f1,
        "p_scene_05cm": float(surfaces["precision_05cm"]), "r_scene_05cm": float(surfaces["recall_05cm"]),
        "J": coverage * f1, "native_mesh_triangles": surfaces["predicted_mesh_triangles"],
        "sampled_predicted_surface": surfaces["predicted_surface_samples"],
    }, represented


def compare_geometry(current_metrics, present, baseline_metrics, baseline_present, weights):
    result = {
        "delta_" + field: current_metrics[field] - baseline_metrics[field]
        for field in ("C_free", "F1_scene_05cm", "J")
    }
    objects = {}
    for group, mask_after in present.items():
        mask_before = baseline_present[group]
        added = int(np.count_nonzero(mask_after & ~mask_before))
        lost = int(np.count_nonzero(mask_before & ~mask_after))
        measure = weights[group]
        row = {
            "gross_samples": added, "lost_samples": lost,
            "represented_samples": int(mask_after.sum()), "fixed_samples": len(mask_after),
            "sample_area_m2": measure, "gross_m2": added * measure, "lost_m2": lost * measure,
            "net_m2": (added - lost) * measure,
            "represented_m2": int(mask_after.sum()) * measure,
            "recall_05cm": float(mask_after.mean()),
        }
        result.update({f"{group}_{field}": row[field] for field in ("gross_m2", "lost_m2", "net_m2")})
        objects[group] = row
    result.update({
        f"object_{field}": sum(row[field] for row in objects.values())
        for field in ("gross_m2", "lost_m2", "net_m2", "represented_m2")
    })
    return result | {"surface_by_object": objects}


def load_metric(scene_freeze, evaluation_freeze):
    verify_bound(NATIVE_METRIC, scene_freeze["files_sha256"])
    verify_bound(OLD_EVAL / "measure.py", evaluation_freeze["files_sha256"])
    legacy_file = OLD_EVAL / "measure.py"
    legacy_spec = importlib.util.spec_from_file_location("archived_nso_measure", legacy_file)
    legacy = importlib.util.module_from_spec(legacy_spec)
    legacy_spec.loader.exec_module(legacy)
    from surface_reference import ray_scene
    from types import SimpleNamespace

    return SimpleNamespace(count_free=legacy.count_free, evaluate_surface=legacy.evaluate_surface, ray_scene=ray_scene)


def write_json(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, allow_nan=False, indent=2) + "\n")
    temporary.replace(path)


def write_csv(path, rows, fields):
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", newline="") as file:
        writer = csv.DictWriter(file, fields)
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def verified_run(run, archived, reference, candidate_id, equal_paid, bindings):
    manifest_path = run / "manifest.json"
    manifest = read_json(manifest_path)
    if (manifest["status"] != "completed_execution" or manifest["candidate_id"] != candidate_id
            or manifest["layout"] != reference["initial"]["layout"] or manifest["evaluator_inputs_read"]):
        raise ValueError(f"Candidate branch identity or evaluator independence failed: {run}")
    if (manifest["source_protocol_sha256"] != bindings["protocol"]
            or manifest["runner_sha256"] != bindings["actor"]
            or manifest["source_commit"] != bindings["source_commit"]
            or not manifest["neutral_g_continuation"] or manifest["native_budget"] != bindings["budget"]):
        raise ValueError(f"Candidate branch does not match frozen execution contract: {run}")
    files = manifest["files_sha256"]
    for name, digest in files.items():
        verify(run / name, digest)
    for name in ("first_anchor.json", "steps.json", "audit.json"):
        if name not in files:
            raise ValueError(f"Missing branch evidence in {run}: {name}")
    first_anchor = read_json(run / "first_anchor.json")
    steps = read_json(run / "steps.json")
    if [step["step"] for step in steps] != list(range(1, len(steps) + 1)):
        raise ValueError(f"Non-contiguous actual paid trace: {run}")
    if manifest["paid_actions"] != len(steps) or manifest["collisions"] != sum(bool(s["collision"]) for s in steps):
        raise ValueError(f"Candidate manifest counts disagree with actual trace: {run}")
    initial_packet = reference["initial"]["initial_packet_sha256"]
    initial_chain = rolling_chain = chain_sha256("0" * 64, initial_packet)
    final_packet = initial_packet
    for step in steps:
        final_packet = step["packet_sha256"]
        rolling_chain = chain_sha256(rolling_chain, final_packet)
        if rolling_chain != step["packet_chain_sha256"]:
            raise ValueError(f"Broken packet chain at step {step['step']}: {run}")
    if (manifest["initial_packet_sha256"] != initial_packet
            or manifest["final_packet_chain_sha256"] != rolling_chain
            or manifest["final_packet_sha256"] != final_packet):
        raise ValueError(f"Candidate manifest packet provenance disagrees with actual trace: {run}")
    option = first_anchor["executed_option"]
    if int(option["candidate_id"]) != candidate_id:
        raise ValueError(f"Wrong bound candidate for {run}")
    old_intervention = archived["intervention"]
    first_choice = archived["first_choice"]
    old_options = first_choice["outputs"]["candidates"]
    if (
        first_anchor["native_options"] != old_options
        or first_anchor["scores"] != first_choice["outputs"]["scores"]
        or first_anchor["first_record_sha256"] != canonical_sha256(first_choice)
        or canonical_sha256(first_anchor["native_options"]) != old_intervention["candidate_set_sha256"]
        or canonical_sha256(first_anchor["scores"]) != old_intervention["score_set_sha256"]
        or first_anchor["belief_sha256"] != old_intervention["belief_sha256"]
        or first_anchor["packet_sha256"] != initial_packet
        or first_anchor["initial_owner_sha256"] != reference["initial"]["initial_owner_sha256"]
        or first_anchor["candidate_summaries"] != old_intervention["candidate_summaries"]
    ):
        raise ValueError(f"First decision or fixed source state differs from archive: {run}")
    summary = archived["candidate_summaries"][candidate_id]
    native_option = old_options[candidate_id]
    native_keys = ("candidate_id", "pose", "outbound_actions", "outbound_states", "return_actions", "return_states")
    if any(option[key] != native_option[key] for key in native_keys):
        raise ValueError(f"Bound option changed original route: {run}")
    if (len(option["outbound_actions"]) != summary["outbound_cost"]
            or len(option["return_actions"]) != summary["reserved_return_cost"]):
        raise ValueError(f"Bound native option and archived costs disagree: {run}")
    if candidate_id == archived["selected_id"]:
        historical_option = archived["intervention"]["chosen_native_option"]
        for key in ("pose", "outbound_actions", "outbound_states", "return_actions", "return_states"):
            if option[key] != historical_option[key]:
                raise ValueError(f"Archived selected branch changed {key}: {run}")
        if len(steps) > len(archived["archive_steps"]):
            raise ValueError(f"Short branch exceeds original logged trace: {run}")
        trace_keys = (
            "step", "action", "position", "heading", "collision", "packet_sha256", "nonsemantic_packet_sha256",
            "owner_sha256", "active_option_id", "phase_before_action",
        )
        for actual, old in zip(steps, archived["archive_steps"], strict=False):
            if any(actual[key] != old[key] for key in trace_keys):
                raise ValueError(f"Original branch packet/trajectory diverged at step {actual['step']}: {run}")
    actual_arrival = None
    for index, step in enumerate(steps):
        pose = [*step["position"], step["heading"]]
        if (step["collision"] or step["active_option_id"] != option["option_id"]
                or step["active_candidate_id"] != option["candidate_id"]
                or step["phase_before_action"] != "outbound"
                or index >= len(option["outbound_actions"])
                or step["action"] != option["outbound_actions"][index]
                or pose != option["outbound_states"][index + 1]):
            break
        if pose == option["pose"]:
            actual_arrival = step["step"]
            break
    labels = {"start": 0, "arrival": actual_arrival, f"paid_{equal_paid}": equal_paid}
    checkpoints = {}
    for label, expected in labels.items():
        receipt_relative = f"{label}/checkpoint.json"
        if receipt_relative not in files:
            raise ValueError(f"Missing fixed checkpoint receipt in {run}: {label}")
        receipt = read_json(run / receipt_relative)
        if receipt["status"] == "unavailable":
            if label == "start":
                raise ValueError(f"No verified common prefix for {run}")
            if label != "arrival" and expected <= len(steps):
                raise ValueError(f"Missing recorded paid checkpoint {label} despite actual paid trace: {run}")
            if label == "arrival" and actual_arrival is not None:
                raise ValueError(f"Logged branch reached the native target but no arrival was captured: {run}")
            checkpoints[label] = receipt
            continue
        if receipt["status"] != "captured" or expected is None or receipt["actual_paid"] != expected:
            raise ValueError(f"Wrong actual paid checkpoint {label} in {run}, expected {expected}")
        if not 0 <= expected <= len(steps):
            raise ValueError(f"Checkpoint appears beyond paid trace: {run}")
        actual_step = steps[expected - 1] if expected else None
        base = reference["initial"]
        expected_pose = ([*actual_step["position"], actual_step["heading"]] if actual_step is not None
                         else base["anchor"])
        expected_packet = (actual_step["packet_sha256"] if actual_step is not None
                           else base["initial_packet_sha256"])
        expected_chain = actual_step["packet_chain_sha256"] if actual_step is not None else initial_chain
        if (receipt["pose"] != expected_pose or receipt["packet_sha256"] != expected_packet
                or receipt["packet_chain_sha256"] != expected_chain):
            raise ValueError(f"Checkpoint {label} pose or packet does not match its paid trace in {run}")
        for name in ("belief.npz", "native_mesh.ply"):
            relative = f"{label}/{name}"
            verify(run / relative, receipt["files_sha256"][name])
            if files[relative] != receipt["files_sha256"][name]:
                raise ValueError(f"Run and checkpoint hashes disagree: {run / relative}")
        checkpoints[label] = receipt
    return manifest, checkpoints, steps, option, {
        "run_manifest_path": str(manifest_path), "run_manifest_sha256": sha256(manifest_path),
        "steps_sha256": files["steps.json"], "first_anchor_sha256": files["first_anchor.json"],
        "audit_sha256": files["audit.json"],
        "checkpoint_sha256": {label: files[f"{label}/checkpoint.json"] for label in checkpoints},
    }


def evaluate_candidate(run, archive, reference, metric, layout, parent, cid, equal_paid, prefix, bindings):
    failure_path = run / "failure.json"
    if failure_path.exists() and not (run / "manifest.json").exists():
        failure = read_json(failure_path)
        if (failure["status"] != "failed" or failure["layout"] != layout or failure["candidate_id"] != cid):
            raise ValueError(f"Candidate execution failure identity does not match its directory: {run}")
        row = archive["candidate_summaries"][cid] | {
            "layout": layout, "parent": parent, "archive_selected": cid == archive["selected_id"],
            "status": "failed", "arrival_status": "unavailable", "arrival": "execution_failed",
            "termination": failure["error"], "stop": "execution_failed", "common_paid": equal_paid,
            "checkpoints": {
                label: {"status": "unavailable", "outcome": "execution_failed"}
                for label in ("start", "arrival", f"paid_{equal_paid}")
            },
            "source_hashes": {"failure_path": str(failure_path), "failure_sha256": sha256(failure_path)},
        }
        return row, prefix
    manifest, receipts, steps, option, hashes = verified_run(run, archive, reference, cid, equal_paid, bindings)
    start_paths = receipts["start"]["files_sha256"]
    if prefix is None:
        belief, pose = read_belief(run / "start/belief.npz")
        raw_belief_sha256 = hashlib.sha256(np.ascontiguousarray(belief).tobytes()).hexdigest()
        if raw_belief_sha256 != archive["intervention"]["belief_sha256"]:
            raise ValueError(f"Measured common initial belief does not match archived decision: {layout}")
        if pose != receipts["start"]["pose"]:
            raise ValueError(f"Recorded prefix pose differs from its belief export: {layout}")
        base_metrics, base_present = measure_geometry(read_mesh(run / "start/native_mesh.ply"), belief, reference, metric)
        if base_metrics["correct_free_cells"] != archive["prefix_summary"]["correctly_free_cells"]:
            raise ValueError(f"Recomputed initial coverage differs from frozen prefix: {layout}")
        prefix = (start_paths, base_metrics, base_present)
    base_hashes, base_metrics, base_present = prefix
    if base_hashes != start_paths:
        raise ValueError(f"Candidate {cid} for {layout} did not start from the exact same mesh and belief")
    measurements = {}
    for label, receipt in receipts.items():
        if receipt["status"] == "unavailable":
            measurements[label] = receipt
            continue
        if label == "start":
            measured, represented = base_metrics, base_present
        else:
            belief, pose = read_belief(run / label / "belief.npz")
            if pose != receipt["pose"]:
                raise ValueError(f"Recorded checkpoint pose differs from its belief export: {run} {label}")
            if hashlib.sha256(np.ascontiguousarray(belief).tobytes()).hexdigest() != receipt["belief_sha256"]:
                raise ValueError(f"Recorded checkpoint belief differs from raw-state digest: {run} {label}")
            measured, represented = measure_geometry(
                read_mesh(run / label / "native_mesh.ply"), belief, reference, metric,
            )
        delta = compare_geometry(measured, represented, base_metrics, base_present, reference["weights"])
        measurements[label] = receipt | measured | delta
    summary = archive["candidate_summaries"][cid]
    row = summary | {
        "layout": layout, "parent": parent, "archive_selected": cid == archive["selected_id"],
        "target_pose": option["pose"], "native_option_id": option["option_id"], "status": manifest["status"],
        "actual_paid": len(steps), "arrival_status": receipts["arrival"]["status"], "arrival": manifest.get("arrival"),
        "termination": manifest["observed_native_termination"], "stop": manifest["stop"],
        "paid_at_arrival_or_interruption": manifest["paid_at_arrival_or_interruption"],
        "actual_collisions": sum(bool(s["collision"]) for s in steps),
        "common_paid": equal_paid, "checkpoints": measurements, "source_hashes": hashes,
        "recorded_option": {key: option[key] for key in ("pose", "outbound_actions", "return_actions")},
    }
    return row, prefix


def csv_row(row):
    output = {
        key: row.get(key) for key in (
            "layout", "parent", "candidate_id", "side", "archive_selected", "outbound_cost", "reserved_return_cost",
            "cost", "status", "arrival_status", "arrival", "actual_paid", "paid_at_arrival_or_interruption",
            "actual_collisions", "common_paid", "common_score", "stop",
        )
    }
    for mode in MODES:
        output[f"score_{mode}"] = row["scores"][mode]
        output[f"object_area_{mode}_m2"] = row["template_area_m2"][mode]
        output[f"rank_{mode}"] = row["ranks"][mode]
        if mode != "G":
            output[f"category_{mode}_minus_G"] = row["scores"][mode] - row["scores"]["G"]
    output["termination"] = json.dumps(row["termination"], ensure_ascii=False, separators=(",", ":"))
    for prefix, label in (("start", "start"), ("arrival", "arrival"), ("matched", f"paid_{row['common_paid']}")):
        checkpoint = row["checkpoints"][label]
        output.update({f"{prefix}_{field}": checkpoint.get(field) for field in CHECKPOINT_FIELDS})
        output[f"{prefix}_outcome"] = checkpoint.get("outcome")
    return output


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, default=STUDY / "PROTOCOL.json")
    parser.add_argument("--runs", type=Path, default=STUDY / "runner/runs")
    parser.add_argument("--out", type=Path, default=STUDY / "measurement/results")
    return parser.parse_args()


def main(args):
    protocol = read_json(args.protocol)
    if (not protocol["declared_before_opening_new_endpoint_metrics"]
            or protocol["previous_local_archive"] != str(ARCHIVE)):
        raise ValueError("A locked and matching candidate experiment protocol is required")
    actor_freeze_path = args.runs.parent / "freeze.json"
    actor_bindings = read_json(actor_freeze_path)["files_sha256"]
    protocol_sha = verify_bound(args.protocol, actor_bindings)
    actor_sha = verify_bound(args.runs.parent / "execute.py", actor_bindings)
    bindings = {"protocol": protocol_sha, "actor": actor_sha, "source_commit": protocol["source_commit"],
                "budget": protocol["native_budget_including_return"]}
    original_scene_freeze = SCENES / "freeze.json"
    original_runner_freeze = HISTORY / "freeze.json"
    original_evaluation_freeze = OLD_EVAL / "freeze.json"
    eval_freeze = read_json(original_evaluation_freeze)
    scene_freeze_sha = verify_bound(original_scene_freeze, eval_freeze["files_sha256"])
    runner_freeze_sha = verify_bound(original_runner_freeze, eval_freeze["files_sha256"])
    runner_freeze = read_json(original_runner_freeze)
    if verify_bound(original_scene_freeze, runner_freeze["files_sha256"]) != scene_freeze_sha:
        raise ValueError("Disagreement on old scene freeze hash between evaluation and runner")
    scene_freeze = read_json(original_scene_freeze)
    verify_bound(NATIVE_EXECUTION_FREEZE, runner_freeze["files_sha256"])
    recon_metric_sha = verify_bound(RECON_METRIC, read_json(NATIVE_EXECUTION_FREEZE)["files_sha256"])
    metric = load_metric(scene_freeze, eval_freeze)
    if protocol["layouts_in_order"] != scene_freeze["layout_order"]:
        raise ValueError("Candidate experiment does not match the fixed scene set and order")
    layouts, rows = {}, []
    for layout in protocol["layouts_in_order"]:
        prefix = None
        archive = load_archive(layout, runner_freeze_sha)
        reference = load_reference(layout, scene_freeze, runner_freeze)
        parent = scene_freeze["layout_specs"][layout]["parent"]
        equal_paid = protocol["common_paid_checkpoints_by_parent"][parent]
        if (archive["selected_id"] != protocol["original_candidate"][layout]
                or reference["initial"]["anchor"] != archive["intervention"]["current_pose"]
                or reference["initial"]["initial_packet_sha256"] != archive["intervention"]["packet_sha256"]):
            raise ValueError(f"Archival candidate intervention and frozen prefix disagree in {layout}")
        for cid in protocol["candidates_in_order"]:
            summary = archive["candidate_summaries"][cid]
            if [summary["outbound_cost"], summary["reserved_return_cost"]] != protocol[
                    "actual_annotated_outbound_return_costs"][parent][cid]:
                raise ValueError(f"Archived native costs differ from frozen experiment in {layout} candidate {cid}")
            run = args.runs / layout / f"candidate_{cid}"
            row, prefix = evaluate_candidate(
                run, archive, reference, metric, layout, parent, cid, equal_paid, prefix, bindings,
            )
            rows.append(row)
        layouts[layout] = {
            "old_manifest_sha256": archive["manifest_sha256"], "old_intervention_sha256": archive["intervention_sha256"],
            "old_steps_sha256": archive["steps_sha256"], "old_choices_sha256": archive["choices_sha256"],
            "reference_inputs_sha256": reference["files_sha256"],
            "reference_visible_object_area_m2": reference["receipt_visible_area_m2"],
        }
    csv_rows = list(map(csv_row, rows))
    args.out.mkdir(parents=True, exist_ok=True)
    write_csv(args.out / "candidate_rows.csv", csv_rows, list(csv_rows[0]))
    write_json(args.out / "candidate_rows.json", {
        "status": "measured_read_only", "actor_reused_or_executed_by_measurement": False,
        "protocol_sha256": protocol_sha, "evaluator_sha256": sha256(__file__),
        "branch_runner_freeze_sha256": sha256(actor_freeze_path),
        "branch_runner_sha256": actor_sha, "native_reconstruction_metric_sha256": recon_metric_sha,
        "old_scene_freeze_sha256": scene_freeze_sha, "old_runner_freeze_sha256": runner_freeze_sha,
        "old_evaluation_freeze_sha256": sha256(original_evaluation_freeze),
        "native_metric_sha256": sha256(NATIVE_METRIC), "threshold_m": THRESHOLD_M,
        "comparison_bands": {key: protocol["metrics"][key] for key in (
            "numeric_score_tie", "numeric_J_tie", "material_object_area_m2", "material_J_fraction",
        )},
        "layouts": layouts, "rows": rows,
    })


if __name__ == "__main__":
    main(parse_args())
