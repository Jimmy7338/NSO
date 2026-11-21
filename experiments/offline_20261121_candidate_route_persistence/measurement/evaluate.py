"""Read-only native mapping evaluation of the frozen route persistence study."""

import argparse
import hashlib
import importlib.util
import math
from pathlib import Path

STUDY = Path(__file__).resolve().parents[1]
PRIOR = STUDY.parent / "candidate_route_yield_20261121"
PRIOR_MEASURE = PRIOR / "measurement/measure.py"


def load_prior_module():
    spec = importlib.util.spec_from_file_location("nso_candidate_yield_measure", PRIOR_MEASURE)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


prior = load_prior_module()
from numpy import ascontiguousarray, count_nonzero, load, savez_compressed, zeros
from open3d.core import Tensor

FIXED_LABELS = ("scene", "shelf", "cabinet", "correct_free")
ARCHIVE_HISTORY = prior.HISTORY
SHORT_ROOT = PRIOR / "runner/runs"
RIGHT_BRANCH = PRIOR / "runner/branch_b/runs/warehouse_right"
SHORT_TABLE = PRIOR / "measurement/results/candidate_rows.json"
BRANCH_TABLE = PRIOR / "measurement/results/branch_b_continuation.json"
TRACE_FIELDS = (
    "step", "action", "position", "heading", "collision", "packet_sha256", "nonsemantic_packet_sha256", "owner_sha256",
    "active_option_id", "selected_target_pose", "phase_before_action",
)
METRIC_FIELDS = (
    "C_free", "correct_free_cells", "reachable_cells", "F1_scene_05cm", "p_scene_05cm", "r_scene_05cm", "J",
    "delta_C_free", "delta_F1_scene_05cm", "delta_J",
    "object_gross_m2", "object_lost_m2", "object_net_m2", "object_represented_m2",
    "shelf_gross_m2", "shelf_lost_m2", "shelf_net_m2", "shelf_represented_m2",
    "cabinet_gross_m2", "cabinet_lost_m2", "cabinet_net_m2", "cabinet_represented_m2",
    "scene_gross_m2", "scene_lost_m2", "scene_net_m2", "scene_represented_m2",
)
SHORT_OUTPUTS = {
    "warehouse": {0: "start", 30: "paid_30"},
    "gallery": {0: "start", 38: "arrival", 58: "paid_58"},
}


def check_trace(trace, initial_packet, cap, manifest):
    if [step["step"] for step in trace] != list(range(1, len(trace) + 1)) or len(trace) > cap:
        raise ValueError("Run does not contain a contiguous trace under the frozen budget")
    if manifest["paid_actions"] != len(trace) or manifest["collisions"] != sum(bool(s["collision"]) for s in trace):
        raise ValueError("Run counts disagree with actual trace")
    rolling = prior.chain_sha256("0" * 64, initial_packet)
    for step in trace:
        rolling = prior.chain_sha256(rolling, step["packet_sha256"])
        if "packet_chain_sha256" in step and rolling != step["packet_chain_sha256"]:
            raise ValueError(f"Packet chain disagrees at paid action {step['step']}")
    if "final_packet_chain_sha256" in manifest and manifest["final_packet_chain_sha256"] != rolling:
        raise ValueError("Terminal packet-chain digest disagrees with trace")
    return rolling


def verify_exact_prefix(actual, recorded, label):
    if len(actual) < len(recorded):
        raise ValueError(f"Actual trace ended before the archived observed prefix for {label}")
    for step, archived in zip(actual, recorded, strict=False):
        if any(step[key] != archived[key] for key in TRACE_FIELDS):
            raise ValueError(f"Changed prefix at paid action {step['step']} for {label}")


def evaluate_verified_snapshot(mesh_path, belief_path, receipt, reference, metric):
    belief, pose = prior.read_belief(belief_path)
    if pose != receipt["pose"]:
        raise ValueError(f"Belief pose and receipt differ: {belief_path}")
    belief_sha256 = hashlib.sha256(ascontiguousarray(belief).tobytes()).hexdigest()
    if "belief_sha256" in receipt and belief_sha256 != receipt["belief_sha256"]:
        raise ValueError(f"Belief state and raw-state checksum differ: {belief_path}")
    native_mesh = prior.read_mesh(mesh_path)
    measured, represented = prior.measure_geometry(native_mesh, belief, reference, metric)
    points = reference["samples"]["scene"]
    if measured["sampled_predicted_surface"]:
        ray = metric.ray_scene(native_mesh)
        represented["scene"] = ray.compute_distance(Tensor(points), nthreads=1).numpy() <= prior.THRESHOLD_M
    else:
        represented["scene"] = zeros(len(points), dtype=bool)
    represented["correct_free"] = (belief == 0) & reference["reachable"]
    if (not math.isclose(float(represented["scene"].mean()), measured["r_scene_05cm"], abs_tol=1e-12, rel_tol=0)
            or int(count_nonzero(represented["correct_free"])) != measured["correct_free_cells"]):
        raise ValueError(f"Fixed surface or free-cell raster does not reproduce the independently measured result: {mesh_path}")
    return measured, represented


def sample_changes(before, after, reference):
    weights = reference["weights"] | {"correct_free": 1}
    changes = {}
    for label in FIXED_LABELS:
        mask_before, mask_after = before[label], after[label]
        added = int(count_nonzero(~mask_before & mask_after))
        lost = int(count_nonzero(mask_before & ~mask_after))
        counts = {
            "before": int(count_nonzero(mask_before)), "after": int(count_nonzero(mask_after)),
            "retained": int(count_nonzero(mask_before & mask_after)), "gross": added, "lost": lost, "net": added - lost,
        }
        if counts["after"] != counts["retained"] + added or counts["before"] != counts["retained"] + lost:
            raise ValueError(f"Correspondence between masks does not conserve counts: {label}")
        unit = "cells" if label == "correct_free" else "m2"
        changes[label] = {f"{key}_{unit}": value * weights[label] for key, value in counts.items()}
        changes[label]["fixed_count"] = (int(reference["reachable"].sum()) if label == "correct_free"
                                         else len(reference["samples"][label]))
        changes[label]["sample_counts"] = counts
    changes["object"] = {
        name: changes["shelf"][name] + changes["cabinet"][name]
        for name in ("before_m2", "after_m2", "retained_m2", "gross_m2", "lost_m2", "net_m2")
    }
    return changes


def augment_measurement(measured, represented, baseline, base_presence, reference):
    changes = sample_changes(base_presence, represented, reference)
    differences = {f"delta_{key}": measured[key] - baseline[key] for key in ("C_free", "F1_scene_05cm", "J")}
    for group in ("scene", "shelf", "cabinet", "object"):
        differences.update({f"{group}_{key}_m2": changes[group][f"{key}_m2"] for key in ("gross", "lost", "net")})
        differences[f"{group}_represented_m2"] = changes[group]["after_m2"]
    return measured | differences


def interval_measurement(start, end, reference):
    row_before, masks_before = start
    row_after, masks_after = end
    return {
        "layout": row_after["layout"], "candidate_id": row_after["candidate_id"],
        "from_checkpoint": row_before["checkpoint"], "to_checkpoint": row_after["checkpoint"],
        "from_paid": row_before["actual_paid"], "to_paid": row_after["actual_paid"],
        **{f"delta_{field}": row_after[field] - row_before[field] for field in ("C_free", "F1_scene_05cm", "J")},
        "changes": sample_changes(masks_before, masks_after, reference),
    }


def write_masks(path, masks):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.stem + ".tmp.npz")
    savez_compressed(temporary, **{label: masks[label] for label in FIXED_LABELS})
    temporary.replace(path)
    return prior.sha256(path)


def verify_manifest(directory, filename, binding=None):
    manifest_path = directory / filename
    digest = prior.sha256(manifest_path)
    if binding is not None:
        prior.verify(manifest_path, binding)
    manifest = prior.read_json(manifest_path)
    for name, checksum in manifest["files_sha256"].items():
        prior.verify(directory / name, checksum)
    return manifest, digest


def snapshot(directory, name, manifest, origin):
    receipt_key = f"{name}/checkpoint.json"
    receipt = prior.read_json(directory / receipt_key)
    prior.verify(directory / receipt_key, manifest["files_sha256"][receipt_key])
    paths = {
        filename: directory / name / filename for filename in ("native_mesh.ply", "belief.npz")
        if filename in receipt["files_sha256"]
    }
    for filename, path in paths.items():
        prior.verify(path, receipt["files_sha256"][filename])
        if receipt["files_sha256"][filename] != manifest["files_sha256"][f"{name}/{filename}"]:
            raise ValueError(f"Checkpoint checksum was changed after sealing {path}")
    return receipt, paths, origin, manifest["files_sha256"][receipt_key]


def common_short(layout, parent, candidate, earlier):
    path = SHORT_ROOT / layout / f"candidate_{candidate}"
    manifest, _ = verify_manifest(path, "manifest.json", earlier["source_hashes"]["run_manifest_sha256"])
    steps = prior.read_json(path / "steps.json")
    captures = {}
    for paid, name in SHORT_OUTPUTS[parent].items():
        receipt, paths, origin, digest = snapshot(path, name, manifest, "prior_short_exact")
        if receipt["actual_paid"] == paid:
            if digest != earlier["source_hashes"]["checkpoint_sha256"][name]:
                raise ValueError(f"Prior measurement is unrelated to checkpoint {path / name}")
            captures[paid] = (receipt, paths, origin, digest, earlier["checkpoints"][name])
    return manifest, steps, captures


def links_for_new_route(layout, candidate, new_context, initial, short, short_steps, protocol):
    run = STUDY / "runner/runs" / layout / f"candidate_{candidate}"
    if (run / "failure.json").exists() and not (run / "manifest.json").exists():
        error = prior.read_json(run / "failure.json")
        if error["layout"] != layout or error["candidate_id"] != candidate or error["status"] != "failed":
            raise ValueError(f"Invalid failed-route receipt in {run}")
        return {"status": "failed", "failure": error, "source_manifest_sha256": prior.sha256(run / "failure.json")}, [], {}
    manifest, digest = verify_manifest(run, "manifest.json")
    if (manifest["status"] != "completed_execution" or manifest["layout"] != layout
            or manifest["candidate_id"] != candidate or manifest["source_commit"] != protocol["source_commit"]
            or manifest["source_protocol_sha256"] != new_context["protocol_sha"]
            or manifest["runner_freeze_sha256"] != new_context["runner_freeze_sha"]
            or manifest["evaluator_inputs_read_by_actor"] or not manifest["neutral_g_continuation"]):
        raise ValueError(f"Run does not match the frozen continuation manifest: {run}")
    trace = prior.read_json(run / "steps.json")
    verify_exact_prefix(trace, short_steps, run)
    recovery = prior.read_json(run / "checkpoint_recovery.json")
    old_run = SHORT_ROOT / layout / f"candidate_{candidate}"
    if (recovery["prior_steps_sha256"] != short["files_sha256"]["steps.json"]
            or recovery["prior_manifest_sha256"] != prior.sha256(old_run / "manifest.json")
            or recovery["matched_short_prefix_actions"] != len(short_steps)
            or not recovery["before_first_new_action"]):
        raise ValueError(f"Checkpoint recovery receipt is not the exact charged prefix: {run}")
    for name, recovered in recovery["snapshot_references"].items():
        source, _, _, digest_before = snapshot(old_run, name, short, "prior_short_exact")
        fields = ("actual_paid", "packet_sha256", "packet_chain_sha256", "pose", "belief_sha256", "native_mesh_fields")
        if recovered["prior_checkpoint_sha256"] != digest_before or any(recovered[k] != source[k] for k in fields):
            raise ValueError(f"Continuation checkpoint recovery does not match original {name}: {run}")
    prior.verify(run / "checkpoint_recovery.json", manifest["checkpoint_recovery_file_sha256"])
    manifest["source_manifest_sha256"] = digest
    manifest["belief_trace"] = str(run / "observed_belief_trace.npz")
    checkpoints = {
        name: (*snapshot(run, name, manifest, "new_continuation"), None)
        for name in ["terminal", *(f"paid_{n}" for n in (*protocol["optional_diagnostic_checkpoints"], 128))]
    }
    if checkpoints["terminal"][0]["packet_chain_sha256"] != check_trace(
        trace, initial["initial_packet_sha256"], protocol["native_total_paid_cap"], manifest,
    ):
        raise ValueError(f"Native terminal lacks exact trace chain: {run}")
    return manifest, trace, checkpoints


def reuse_right_branch(layout, candidate, initial, short_steps, protocol, frozen):
    branch = RIGHT_BRANCH / f"candidate_{candidate}"
    binding = frozen["files_sha256"][str(branch / "manifest.json")]
    manifest, digest = verify_manifest(branch, "manifest.json", binding)
    trace = prior.read_json(branch / "steps.json")
    verify_exact_prefix(trace, short_steps, branch)
    terminal = snapshot(branch, "terminal", manifest, "previous_branch_B")
    if terminal[0]["packet_chain_sha256"] != check_trace(
        trace, initial["initial_packet_sha256"], protocol["native_total_paid_cap"], manifest,
    ):
        raise ValueError(f"Archived Branch B terminal lacks exact trace chain: {branch}")
    previous = next(row for row in prior.read_json(BRANCH_TABLE)["rows"] if row["candidate_id"] == candidate)
    if previous["manifests_sha256"]["terminal"] != digest:
        raise ValueError(f"Archived Branch B independent result is unrelated to its source manifest: {branch}")
    return manifest | {"source_manifest_sha256": digest}, trace, {"terminal": (*terminal, previous["terminal"])}


def reuse_native_archive(layout, candidate, initial, short_steps, protocol, frozen):
    run = ARCHIVE_HISTORY / "layouts" / layout / "runs/G/attempt_001"
    binding = frozen["files_sha256"][str(run / "artifact_manifest.json")]
    old_manifest, digest = verify_manifest(run, "artifact_manifest.json", binding)
    manifest = prior.read_json(run / "result.json")
    trace = prior.read_json(run / "steps.json")
    verify_exact_prefix(trace, short_steps, run)
    chain = check_trace(trace, initial["initial_packet_sha256"], protocol["native_total_paid_cap"], manifest)
    if (manifest["effective_args"]["cpu_score_mode"] != "G"
            or prior.read_json(run / "intervention.json")["intervention"]["chosen_candidate_id"] != candidate):
        raise ValueError(f"Original archive cannot supply the declared comparator: {run}")
    receipt = {
        "status": "captured", "actual_paid": manifest["paid_actions"], "requested_paid": manifest["paid_action_budget"],
        "pose": manifest["final_pose"], "packet_sha256": trace[-1]["packet_sha256"], "packet_chain_sha256": chain,
        "outcome": "original_archived_native_terminal",
        "files_sha256": {filename: old_manifest["files_sha256"][filename]
                         for filename in ("final_tsdf_mesh.ply", "final_belief.npz")},
    }
    paths = {"native_mesh.ply": run / "final_tsdf_mesh.ply", "belief.npz": run / "final_belief.npz"}
    targets = {"terminal": (receipt, paths, "original_archived", None, None)}
    manifest = manifest | {"source_manifest_sha256": digest, "returned": manifest["termination"]["returned_to_anchor"]}
    if layout.startswith("gallery"):
        recovery = STUDY / "measurement/archive_recovery" / layout / f"candidate_{candidate}"
        archival, recovery_hash = verify_manifest(recovery, "manifest.json")
        if (archival["source_manifest_sha256"] != digest or archival["terminal_packet_chain_sha256"] != chain
                or archival["terminal_mesh_sha256"] != old_manifest["files_sha256"]["final_tsdf_mesh.ply"]
                or archival["source_protocol_sha256"] != frozen["protocol_sha256"]
                or archival["archived_actions_replayed"] != len(trace)
                or any(archival[key] for key in ("new_policy_actions", "planning_calls", "scoring_calls"))):
            raise ValueError(f"Mismatched mapping-only archive recovery: {recovery}")
        for name in archival["captured"]:
            if name.startswith("paid_"):
                targets[name] = (*snapshot(recovery, name, archival, "exact_archived_sensor_recovery"), None)
        manifest["recovered_archive_manifest_sha256"] = recovery_hash
        manifest["belief_trace"] = str(recovery / "observed_belief_trace.npz")
    return manifest, trace, targets


def verify_checkpoint_against_trace(receipt, requested, trace, initial):
    paid = receipt["actual_paid"]
    if not 0 <= paid <= len(trace) or (isinstance(requested, int) and requested != paid):
        raise ValueError(f"Endpoint measured after {paid} cannot fulfill the intended checkpoint {requested}")
    pose = [*trace[paid - 1]["position"], trace[paid - 1]["heading"]] if paid else initial["anchor"]
    packet = trace[paid - 1]["packet_sha256"] if paid else initial["initial_packet_sha256"]
    if receipt["pose"] != pose or receipt["packet_sha256"] != packet:
        raise ValueError(f"Checkpoint pose or packet differs from actual paid trace at {paid}")


def compare_remeasurement(new, old, label):
    for key in ("C_free", "F1_scene_05cm", "J", "object_gross_m2", "object_lost_m2", "object_net_m2"):
        if not math.isclose(new[key], old[key], abs_tol=1e-12, rel_tol=0):
            raise ValueError(f"Remeasurement of {key} differs from exact frozen snapshot: {label}")


def per_action_curve(manifest, trace, checkpoints, reference):
    if "belief_trace" not in manifest:
        if trace and all("correctly_free_cells" in step for step in trace):
            start = checkpoints[0]["correct_free_cells"]
            return [(0, start, None, None)] + [
                (step["step"], step["correctly_free_cells"], None, None) for step in trace
            ]
        return [(row["actual_paid"], row["correct_free_cells"], None, None) for row in checkpoints
                if row["status"] == "captured"]
    with load(manifest["belief_trace"], allow_pickle=False) as data:
        paid_values, belief = data["paid"], data["belief"]
    if list(paid_values) != list(range(len(trace) + 1)) or belief.shape[1:] != reference["reachable"].shape:
        raise ValueError("Per-action saved occupancy arrays do not align with the paid trace")
    prior_mask = None
    rows = []
    for paid, grid in enumerate(belief):
        present = (grid == 0) & reference["reachable"]
        free = int(count_nonzero(present))
        if paid and "correctly_free_cells" in trace[paid - 1] and trace[paid - 1]["correctly_free_cells"] != free:
            raise ValueError(f"Recomputed archived correct-free curve differs at action {paid}")
        gross = int(count_nonzero(present & ~prior_mask)) if prior_mask is not None else 0
        lost = int(count_nonzero(prior_mask & ~present)) if prior_mask is not None else 0
        rows.append((paid, free, gross, lost))
        prior_mask = present
    for row in checkpoints:
        if row["status"] == "captured" and rows[row["actual_paid"]][1] != row["correct_free_cells"]:
            raise ValueError(f"Per-action truth curve cannot reproduce paid checkpoint {row['actual_paid']}")
    return rows


def measure_route(layout, candidate, parent, source, earlier, context, reference, metric, protocol, output):
    manifest_short, trace_short, captures = common_short(layout, parent, candidate, earlier)
    frozen = context["frozen"]
    initial = reference["initial"]
    if source == "new":
        manifest, trace, extra = links_for_new_route(
            layout, candidate, context, initial, manifest_short, trace_short, protocol,
        )
    elif layout == "warehouse_right":
        manifest, trace, extra = reuse_right_branch(layout, candidate, initial, trace_short, protocol, frozen)
    else:
        manifest, trace, extra = reuse_native_archive(layout, candidate, initial, trace_short, protocol, frozen)
    route_failed = manifest["status"] == "failed"
    if route_failed:
        trace = trace_short
    cap = protocol["native_total_paid_cap"]
    wanted = {
        0, *protocol["optional_diagnostic_checkpoints"],
        *(v for v in protocol["required_checkpoints"][parent] if isinstance(v, int)),
    }
    rows, intervals = [], []
    previous_observed = None
    for key in [*sorted(wanted), "terminal"]:
        title = "start" if key == 0 else (f"paid_{key}" if isinstance(key, int) else key)
        checkpoint = captures[key] if key in captures else extra.get(title)
        row = {"layout": layout, "parent": parent, "candidate_id": candidate, "checkpoint": title,
               "run_kind": "new_continuation" if source == "new" else "archived_reuse"}
        if checkpoint is None:
            if route_failed:
                condition = "execution_failed_before_terminal_evidence"
            elif isinstance(key, int) and key > manifest["paid_actions"]:
                condition = "native_stop_before_checkpoint"
            else:
                condition = "no_archived_snapshot"
            rows.append(row | {
                "status": "unavailable", "requested_paid": cap if key == "terminal" else key,
                "actual_paid": None, "condition": condition,
            })
            continue
        receipt, paths, origin, digest, previous = checkpoint
        if receipt["status"] == "unavailable":
            rows.append(row | {
                "status": "unavailable", "requested_paid": receipt["requested_paid"], "actual_paid": None,
                "condition": receipt["outcome"], "source_checkpoint_sha256": digest,
            })
            continue
        verify_checkpoint_against_trace(receipt, key, trace, initial)
        measured, masks = evaluate_verified_snapshot(paths["native_mesh.ply"], paths["belief.npz"], receipt, reference, metric)
        if key == 0:
            baseline, base_masks = measured, masks
        augmented = augment_measurement(measured, masks, baseline, base_masks, reference)
        if previous is not None:
            compare_remeasurement(augmented, previous, (layout, candidate, key))
        result_relative_file = Path("presence") / layout / f"candidate_{candidate}" / (title + ".npz")
        row |= augmented | {
            "status": "captured", "condition": origin, "actual_paid": receipt["actual_paid"],
            "requested_paid": receipt["requested_paid"], "pose": receipt["pose"],
            "packet_sha256": receipt["packet_sha256"], "packet_chain_sha256": receipt["packet_chain_sha256"],
            "source_checkpoint_sha256": digest, "source_snapshot_sha256": receipt["files_sha256"],
            "presence_path": str(result_relative_file), "presence_sha256": write_masks(output / result_relative_file, masks),
        }
        rows.append(row)
        if previous_observed is not None and previous_observed[0]["actual_paid"] != row["actual_paid"]:
            intervals.append(interval_measurement(previous_observed, (row, masks), reference))
        previous_observed = (row, masks)

    termination = manifest.get("termination", {})
    terminal = rows[-1] | {
        "returned": manifest.get("returned"), "collisions": manifest.get("collisions"),
        "stop_reason": termination.get("reason", "execution_failure" if route_failed else None),
        "failed": bool(termination.get("failed", route_failed)), "path_length_m": manifest.get("path_length_m"),
        "source_manifest_sha256": manifest.get("source_manifest_sha256"),
        "recovered_archive_manifest_sha256": manifest.get("recovered_archive_manifest_sha256"),
    }
    if terminal["status"] == "captured" and (
        terminal["actual_paid"] != manifest["paid_actions"]
        or terminal["actual_paid"] != termination["final_action_id"]
        or terminal["returned"] != (terminal["pose"] == initial["anchor"])
    ):
        raise ValueError(f"Terminal state/paid count does not match native closure for {(layout, candidate)}")
    curve = per_action_curve(manifest, trace, rows, reference) if not route_failed else []
    free_total = int(reference["reachable"].sum())
    actions = [
        {"layout": layout, "candidate_id": candidate, "paid": paid, "correct_free_cells": free,
         "reachable_cells": free_total, "C_free": free / free_total,
         "gross_correct_free_cells": gross, "lost_correct_free_cells": lost}
        for paid, free, gross, lost in curve
    ]
    terminal["has_true_coverage_per_action"] = bool(curve and len(curve) == len(trace) + 1)
    for threshold in (0.95, 1.0):
        field = f"first_verified_C_free_{int(threshold * 100)}_paid"
        terminal[field] = next((row["paid"] for row in actions if row["C_free"] >= threshold), None)
    return rows, intervals, terminal, actions


def flatten_interval(row):
    values = {key: value for key, value in row.items() if key != "changes"}
    for group, changes in row["changes"].items():
        for field, value in changes.items():
            if field != "sample_counts":
                values[f"{group}_{field}"] = value
    return values


def write_tables(output, name, rows, metadata, fields=None):
    if fields is None:
        fields = list(dict.fromkeys(key for row in rows for key in row))
    prior.write_csv(output / f"{name}.csv", [{key: row.get(key) for key in fields} for row in rows], fields)
    prior.write_json(output / f"{name}.json", metadata | {"rows": rows})


def main(args):
    protocol_path = STUDY / "PROTOCOL.json"
    frozen_path = STUDY / "PROTOCOL_FREEZE.json"
    new_runner_freeze_path = STUDY / "runner/freeze.json"
    protocol = prior.read_json(protocol_path)
    frozen = prior.read_json(frozen_path)
    new_runner = prior.read_json(new_runner_freeze_path)
    bound_sha = prior.verify_bound(protocol_path, frozen["files_sha256"])
    if (not protocol["frozen_before_new_terminal_outcomes"] or protocol["source_local"] != str(PRIOR)
            or frozen["source_commit"] != protocol["source_commit"]
            or prior.verify_bound(protocol_path, new_runner["files_sha256"]) != bound_sha):
        raise ValueError("Input freeze differs from the frozen continuation protocol")
    files_sha256 = {str(x): prior.verify_bound(x, frozen["files_sha256"]) for x in (
        PRIOR_MEASURE, SHORT_TABLE, BRANCH_TABLE, prior.SCENES / "freeze.json",
        prior.HISTORY / "freeze.json", prior.OLD_EVAL / "freeze.json",
    )}
    old_scene = prior.read_json(prior.SCENES / "freeze.json")
    old_runner = prior.read_json(prior.HISTORY / "freeze.json")
    old_eval = prior.read_json(prior.OLD_EVAL / "freeze.json")
    prior.verify_bound(prior.NATIVE_EXECUTION_FREEZE, old_runner["files_sha256"])
    reconstruction_sha = prior.verify_bound(
        prior.RECON_METRIC, prior.read_json(prior.NATIVE_EXECUTION_FREEZE)["files_sha256"],
    )
    continued_actor_sha = prior.verify_bound(STUDY / "runner/continue_route.py", new_runner["files_sha256"])
    metric = prior.load_metric(old_scene, old_eval)
    context = {"frozen": frozen, "protocol_sha": bound_sha, "runner_freeze_sha": prior.sha256(new_runner_freeze_path)}
    earlier = {(row["layout"], row["candidate_id"]): row for row in prior.read_json(SHORT_TABLE)["rows"]}
    checkpoints, terminal, intervals, curves, layout_metadata = [], [], [], [], {}
    args.out.mkdir(parents=True, exist_ok=True)
    for layout in protocol["layouts_in_order"]:
        parent = old_scene["layout_specs"][layout]["parent"]
        reference = prior.load_reference(layout, old_scene, old_runner)
        layout_metadata[layout] = {"reference_inputs_sha256": reference["files_sha256"],
                                   "sample_weights_m2": reference["weights"],
                                   "reachable_cells": int(reference["reachable"].sum())}
        new_ids = protocol["new_routes"].get(layout, [])
        archive_ids = protocol["archived_full_routes"].get(layout, [])
        for cid in sorted(new_ids + archive_ids):
            source = "new" if cid in new_ids else "archived"
            cuts, changes, close, actions = measure_route(
                layout, cid, parent, source, earlier[layout, cid], context, reference, metric, protocol, args.out,
            )
            checkpoints.extend(cuts)
            intervals.extend(changes)
            terminal.append(close)
            curves.extend(actions)
    metadata = {
        "status": "measured_read_only", "input_sha256": files_sha256 | {
            "protocol": bound_sha, "frozen_inputs": prior.sha256(frozen_path),
            "new_runner_freeze": context["runner_freeze_sha"], "evaluator": prior.sha256(__file__),
            "continued_actor": continued_actor_sha, "native_reconstruction_metric": reconstruction_sha,
        }, "layouts": layout_metadata, "surface_threshold_m": prior.THRESHOLD_M,
    }
    common_fields = (
        "layout", "parent", "candidate_id", "checkpoint", "run_kind", "status", "condition", "actual_paid",
        "requested_paid", *METRIC_FIELDS, "presence_path", "presence_sha256",
    )
    write_tables(args.out, "checkpoints", checkpoints, metadata, common_fields)
    write_tables(args.out, "terminal", terminal, metadata, (*common_fields, "returned", "collisions", "stop_reason",
                  "failed", "path_length_m", "source_manifest_sha256", "has_true_coverage_per_action",
                  "first_verified_C_free_95_paid", "first_verified_C_free_100_paid"))
    write_tables(args.out, "coverage_curve", curves, metadata)
    prior.write_json(args.out / "interval_geometry.json", metadata | {"rows": intervals})
    flat = list(map(flatten_interval, intervals))
    prior.write_csv(args.out / "interval_geometry.csv", flat, list(flat[0]))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=STUDY / "measurement/results")
    main(parser.parse_args())
