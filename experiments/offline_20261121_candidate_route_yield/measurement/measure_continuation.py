"""Remeasure only the predeclared two-route Branch B continuation."""

import argparse
import hashlib
import math
from pathlib import Path

from measure import (
    HISTORY, OLD_EVAL, SCENES, STUDY, chain_sha256, compare_geometry, load_archive, load_metric, load_reference,
    measure_geometry, read_belief, read_json, read_mesh, sha256, verify, verify_bound, write_csv, write_json,
)

import numpy as np

SHORT_ROOT = STUDY / "runner/runs"
BRANCH_ROOT = STUDY / "runner/branch_b"
SHORT_TABLE = STUDY / "measurement/results/candidate_rows.json"
GATE_FILE = STUDY / "decision.json"
TRACE_FIELDS = (
    "step", "action", "position", "heading", "collision", "packet_sha256", "nonsemantic_packet_sha256",
    "owner_sha256", "active_option_id", "selected_target_pose", "phase_before_action",
)
METRICS = (
    "C_free", "correct_free_cells", "F1_scene_05cm", "p_scene_05cm", "r_scene_05cm", "J",
    "object_net_m2", "object_gross_m2", "object_lost_m2", "object_represented_m2",
    "shelf_net_m2", "cabinet_net_m2",
)


def read_snapshot(path, receipt, reference, metric):
    for filename, checksum in receipt["files_sha256"].items():
        verify(path / filename, checksum)
    belief, pose = read_belief(path / "belief.npz")
    belief_sha256 = hashlib.sha256(np.ascontiguousarray(belief).tobytes()).hexdigest()
    if pose != receipt["pose"] or belief_sha256 != receipt["belief_sha256"]:
        raise ValueError(f"Belief content and receipt do not align at {path}")
    measured, represented = measure_geometry(read_mesh(path / "native_mesh.ply"), belief, reference, metric)
    return measured, represented


def compare_surface_retention(base, mid, final, weights):
    result = {}
    for name, terminal in final.items():
        start, early, weight = base[name], mid[name], weights[name]
        early_added = ~start & early
        result[name] = {
            "added_through_short_retained_m2": int(np.count_nonzero(early_added & terminal)) * weight,
            "added_through_short_lost_m2": int(np.count_nonzero(early_added & ~terminal)) * weight,
            "new_relative_to_both_at_terminal_m2": int(np.count_nonzero(~start & ~early & terminal)) * weight,
            "baseline_only_surface_recaptured_m2": int(np.count_nonzero(start & ~early & terminal)) * weight,
        }
    result["combined"] = {
        key: sum(group[key] for group in result.values())
        for key in result["shelf"]
    }
    return result


def verified_prefix(short_run, measured_row, trace, prefix_paid):
    verify(short_run / "manifest.json", measured_row["source_hashes"]["run_manifest_sha256"])
    verify(short_run / "steps.json", measured_row["source_hashes"]["steps_sha256"])
    short_trace = read_json(short_run / "steps.json")
    if len(short_trace) != prefix_paid or len(trace) < prefix_paid:
        raise ValueError(f"Continuation does not include the measured prefix: {short_run}")
    for earlier, later in zip(short_trace, trace, strict=False):
        if any(earlier[key] != later[key] for key in TRACE_FIELDS):
            raise ValueError(f"Continuation split before paid action {earlier['step']} in {short_run}")
    return short_trace


def verify_short_metric(measured, earlier, run):
    for key in ("C_free", "F1_scene_05cm", "J"):
        if not math.isclose(measured[key], earlier[key], rel_tol=0, abs_tol=1e-12):
            raise ValueError(f"Short checkpoint {key} differs in {run}")


def evaluate_candidate(run, short_run, row, archive, reference, metric, prefix_paid, bounds):
    cap = bounds["budget"]
    manifest_path = run / "manifest.json"
    receipt_path = run / "terminal/checkpoint.json"
    manifest = read_json(manifest_path)
    if (manifest["status"] != "completed_execution" or manifest["branch_freeze_sha256"] != bounds["freeze_sha256"]
            or manifest["native_budget"] != cap or manifest["source_commit"] != bounds["source_commit"]
            or manifest["source_protocol_sha256"] != bounds["protocol_sha256"]
            or manifest["matched_short_prefix_actions"] != prefix_paid
            or manifest["evaluator_inputs_read_by_actor"]
            or manifest["reused_exact_archive"] != row["archive_selected"]):
        raise ValueError(f"Continuation does not match the frozen Branch B contract: {run}")
    check_files = (
        "steps.json", "first_anchor.json", "runtime_summary.json", "checkpoint_recovery.json",
        "terminal/checkpoint.json", "terminal/native_mesh.ply", "terminal/belief.npz",
    )
    for name in check_files:
        verify(run / name, manifest["files_sha256"][name])
    receipt = read_json(receipt_path)
    if receipt != manifest["checkpoints"]["terminal"]:
        raise ValueError(f"Captured terminal receipt and manifest disagree: {run}")
    steps = read_json(run / "steps.json")
    verified_prefix(short_run, row, steps, prefix_paid)
    verify(run / "first_anchor.json", row["source_hashes"]["first_anchor_sha256"])
    termination = read_json(run / "runtime_summary.json")["termination"]
    if [step["step"] for step in steps] != list(range(1, len(steps) + 1)):
        raise ValueError(f"Non-contiguous paid trace after continuation: {run}")
    if (len(steps) > cap or receipt["actual_paid"] != len(steps) or manifest["paid_actions"] != len(steps)
            or termination["final_action_id"] != len(steps) or termination != manifest["termination"]):
        raise ValueError(f"Branch B terminal does not correspond to the declared budget: {run}")
    if manifest["layout"] != row["layout"] or manifest["candidate_id"] != row["candidate_id"]:
        raise ValueError(f"Branch B run identity mismatch: {run}")
    if (receipt["pose"] != [*steps[-1]["position"], steps[-1]["heading"]]
            or receipt["packet_sha256"] != steps[-1]["packet_sha256"]
            or manifest["returned"] != (receipt["pose"] == reference["initial"]["anchor"])
            or manifest["collisions"] != sum(bool(s["collision"]) for s in steps)):
        raise ValueError(f"Terminal receipt differs from last actual observation: {run}")
    rolling_chain = chain_sha256("0" * 64, reference["initial"]["initial_packet_sha256"])
    for step in steps:
        rolling_chain = chain_sha256(rolling_chain, step["packet_sha256"])
        if rolling_chain != step["packet_chain_sha256"]:
            raise ValueError(f"Broken continuation packet chain at paid step {step['step']}: {run}")
    if rolling_chain != receipt["packet_chain_sha256"] or rolling_chain != manifest["final_packet_chain_sha256"]:
        raise ValueError(f"Terminal packet digests disagree: {run}")
    base_row = row["checkpoints"]["start"]
    matched_row = row["checkpoints"][f"paid_{prefix_paid}"]
    recovery = read_json(run / "checkpoint_recovery.json")
    if (
        recovery["paid"] != prefix_paid
        or recovery["packet_chain_sha256"] != matched_row["packet_chain_sha256"]
        or recovery["short_checkpoint_file_sha256"] != row["source_hashes"]["checkpoint_sha256"][f"paid_{prefix_paid}"]
        or recovery["belief_sha256"] != matched_row["belief_sha256"]
        or recovery["native_mesh_fields"] != matched_row["native_mesh_fields"]
        or recovery["short_snapshot_files_sha256"] != matched_row["files_sha256"]
    ):
        raise ValueError(f"Branch B continuation does not share its measured short checkpoint: {run}")
    if row["archive_selected"]:
        if not recovery["original_trace_prefix_fields_identical"]:
            raise ValueError(f"Reused historical prefix differs from the fixed experiment: {run}")
        if len(steps) != len(archive["archive_steps"]):
            raise ValueError(f"Archived continuation changed the count of executed actions: {run}")
        for actual, earlier in zip(steps, archive["archive_steps"], strict=True):
            if any(actual[key] != earlier[key] for key in TRACE_FIELDS):
                raise ValueError(f"Archived continuation changed at paid action {actual['step']}: {run}")
        source_path = HISTORY / "layouts" / row["layout"] / "runs/G/attempt_001"
        old_manifest_path = source_path / "artifact_manifest.json"
        verify(old_manifest_path, archive["manifest_sha256"])
        old_files = read_json(old_manifest_path)["files_sha256"]
        for old, new in (("final_tsdf_mesh.ply", "native_mesh.ply"), ("final_belief.npz", "belief.npz")):
            verify(source_path / old, old_files[old])
            verify(run / "terminal" / new, old_files[old])
    else:
        if not all(recovery[field] for field in (
            "all_step_fields_identical", "native_mesh_fields_matched", "belief_matched",
        )):
            raise ValueError(f"Branch B alternate has no exact recovered checkpoint: {run}")
    base, base_points = read_snapshot(short_run / "start", base_row, reference, metric)
    matched, matched_points = read_snapshot(short_run / f"paid_{prefix_paid}", matched_row, reference, metric)
    terminal, terminal_points = read_snapshot(run / "terminal", receipt, reference, metric)
    verify_short_metric(base, base_row, short_run)
    verify_short_metric(matched, matched_row, short_run)
    from_start = compare_geometry(terminal, terminal_points, base, base_points, reference["weights"])
    from_short = compare_geometry(terminal, terminal_points, matched, matched_points, reference["weights"])
    retained = compare_surface_retention(base_points, matched_points, terminal_points, reference["weights"])
    return {
        "layout": row["layout"], "candidate_id": row["candidate_id"], "archive_reused": row["archive_selected"],
        "actual_paid": len(steps), "requested_cap": cap, "actual_collisions": sum(bool(s["collision"]) for s in steps),
        "returned_to_anchor": receipt["pose"] == reference["initial"]["anchor"],
        "terminal_pose": receipt["pose"], "terminal_receipt": receipt, "status": manifest["status"],
        "stop": termination["reason"], "termination": termination, "path_length_m": manifest["path_length_m"],
        "matched_short": {key: matched_row[key] for key in METRICS},
        "terminal": terminal | from_start, "change_after_matched_short": from_short,
        "retention": retained, "checked_prefix_packets": prefix_paid,
        "manifests_sha256": {"short": row["source_hashes"]["run_manifest_sha256"], "terminal": sha256(manifest_path)},
    }


def csv_row(row):
    output = {key: row[key] for key in (
        "layout", "candidate_id", "archive_reused", "actual_paid", "requested_cap", "actual_collisions",
        "returned_to_anchor", "status", "stop", "path_length_m",
    )}
    for label in ("matched_short", "terminal"):
        output.update({f"{label}_{key}": row[label][key] for key in METRICS})
    change = row["change_after_matched_short"]
    output.update({f"after_short_{key}": change[key] for key in (
        "delta_C_free", "delta_F1_scene_05cm", "delta_J", "object_net_m2", "object_gross_m2", "object_lost_m2",
    )})
    output.update(row["retention"]["combined"])
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=STUDY / "measurement/results")
    args = parser.parse_args()
    protocol = read_json(STUDY / "PROTOCOL.json")
    gate = read_json(GATE_FILE)
    if gate["branch"] != "B" or not gate["continuation_pair"]:
        raise ValueError("Only the activated continuation pair may be evaluated")
    verify(STUDY / "PROTOCOL.json", gate["input_sha256"]["protocol"])
    verify(SHORT_TABLE, gate["input_sha256"]["measurement"])
    short = read_json(SHORT_TABLE)
    selected = gate["continuation_pair"]
    layout, pair = selected["layout"], selected["candidate_ids"]
    branch_freeze_path = BRANCH_ROOT / "freeze.json"
    branch_freeze = read_json(branch_freeze_path)
    if branch_freeze["layout"] != layout or branch_freeze["candidates"] != pair:
        raise ValueError("Branch B frozen execution pair differs from the recorded gate decision")
    verify_bound(GATE_FILE, branch_freeze["files_sha256"])
    verify_bound(BRANCH_ROOT / "continue_pair.py", branch_freeze["files_sha256"])
    rows = {row["candidate_id"]: row for row in short["rows"] if row["layout"] == layout}
    old_eval = read_json(OLD_EVAL / "freeze.json")
    old_runner_sha = verify_bound(HISTORY / "freeze.json", old_eval["files_sha256"])
    verify_bound(SCENES / "freeze.json", old_eval["files_sha256"])
    old_runner = read_json(HISTORY / "freeze.json")
    old_scene = read_json(SCENES / "freeze.json")
    reference = load_reference(layout, old_scene, old_runner)
    archive = load_archive(layout, old_runner_sha)
    metric = load_metric(old_scene, old_eval)
    cap = protocol["native_budget_including_return"]
    prefix_paid = protocol["common_paid_checkpoints_by_parent"][rows[pair[0]]["parent"]]
    if (branch_freeze["common_paid"] != prefix_paid or branch_freeze["total_native_budget"] != cap
            or not branch_freeze["neutral_g_continuation"]):
        raise ValueError("Branch B checkpoint and continuation policy differ from declared conditions")
    bounds = {
        "budget": cap, "source_commit": protocol["source_commit"], "freeze_sha256": sha256(branch_freeze_path),
        "protocol_sha256": gate["input_sha256"]["protocol"],
    }
    endpoints = []
    for cid in pair:
        relative = Path(layout) / f"candidate_{cid}"
        endpoint = evaluate_candidate(
            BRANCH_ROOT / "runs" / relative, SHORT_ROOT / relative, rows[cid], archive, reference, metric,
            prefix_paid, bounds,
        )
        endpoints.append(endpoint)
    same_paid = endpoints[0]["actual_paid"] == endpoints[1]["actual_paid"]
    result = {
        "status": "measured_read_only", "source_b_decision_sha256": sha256(GATE_FILE),
        "source_short_measurements_sha256": sha256(SHORT_TABLE), "source_evaluator_sha256": sha256(__file__),
        "branch_freeze_sha256": bounds["freeze_sha256"],
        "layout": layout, "candidate_ids": pair, "common_short_paid": prefix_paid, "declared_cap": cap,
        "common_paid_terminal_comparison": same_paid, "rows": endpoints,
        "first_candidate_minus_second_at_common_short": {
            key: endpoints[0]["matched_short"][key] - endpoints[1]["matched_short"][key] for key in METRICS
        },
        "first_candidate_minus_second_at_terminal": {
            key: endpoints[0]["terminal"][key] - endpoints[1]["terminal"][key] for key in METRICS
        } if same_paid else None,
    }
    args.out.mkdir(parents=True, exist_ok=True)
    csv_rows = list(map(csv_row, endpoints))
    write_csv(args.out / "branch_b_continuation.csv", csv_rows, list(csv_rows[0]))
    write_json(args.out / "branch_b_continuation.json", result)


if __name__ == "__main__":
    main()
