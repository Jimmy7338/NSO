"""Read-only independent cold-start occupancy, reconstruction, and causal evaluation."""

import argparse
from collections import Counter
import importlib.util
from math import isclose, isfinite
from pathlib import Path

from freeze_inputs import HOLDOUT, OUTPUT, SOURCE, STUDY, read, require, sha

RAW_SENSOR_FIELDS = ("depth_values_sha256", "rgb_values_sha256", "scan_ranges_sha256")


def source_metric():
    path = SOURCE / "measurement/evaluate.py"
    spec = importlib.util.spec_from_file_location("nso_native_frozen_evaluator", path)
    native = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(native)
    return native


native = source_metric()
engine, files = native.engine, native.files
METRICS = native.TERMINAL_METRICS
TIME_FIELDS = (
    "elapsed_s", "decision_wall_s", "decision_wall_s_per_paid_action", "decision_wall_s_p50", "decision_wall_s_p95",
    "native_global_choices", "global_candidates_total", "local_route_choices", "assessed_candidates_total",
    "admitted_alternates_total", "rejected_local_candidates_total", "assessed_candidates_per_choice_max",
    "admitted_alternates_per_choice_max", "selected_alternates_total", "different_hypothetical_winners",
    "frontier_selection_candidate_count_status",
)


def first_difference(g, s, fields):
    for nr, (a, b) in enumerate(zip(g, s, strict=False), 1):
        require(a["step"] == b["step"] == nr, "Paired actual clocks are not consecutive")
        changed = [key for key in fields if a[key] != b[key]]
        if changed:
            return {"paid": nr, "different_fields": changed, "fullG": {k: a[k] for k in fields},
                    "fullS": {k: b[k] for k in fields}}
    return None


def full_pool_counts(path, manifest, choices, shape):
    originals = read(path / "local_choices.json")
    require(len(originals) == len(choices) == manifest["local_route_choices"], "Local event count drift")
    if manifest["arm"] not in ("fullG", "fullS"):
        require(not originals and not manifest["all_distinct_local_candidates_assessed"],
                "Local pool enabled in original comparator")
        return {}, []
    require(manifest["all_distinct_local_candidates_assessed"] and manifest["closed_loop_full_pool"],
            "Exhaustive chooser did not identify itself in sealed actor manifest")
    counters = Counter()
    individual = []
    for event, normalized in zip(originals, choices, strict=True):
        core = event["choice_audit"]
        proof = event["full_pool"]
        rows = core["candidates"]
        assessed = len(rows)
        admitted = sum(bool(row["admitted"]) for row in rows)
        require(proof["enabled"] and proof["unassessed_after_cap"] == 0
                and proof["original_small_pool_cap"] == 4
                and proof["exhaustive_cap_upper_bound"] == 4 * shape[0] * shape[1] + 1
                and proof["original_distinct_alternates"] == core["proposal_diagnostics"]["distinct_alternate_traces"]
                == assessed - 1
                and proof["full_candidate_pool_sha256"] == core["candidate_pool_sha256"]
                and proof["full_score_sha256"] == core["score_table_sha256"]
                and admitted == core["admitted_candidate_count"]
                and assessed == core["assessed_candidate_count"]
                and len({tuple(row["outbound_actions"]) for row in rows}) == assessed
                and event["paid_before"] == normalized["paid_before"],
                f"Candidate records are not the full distinct pool: {path}, paid {event['paid_before']}")
        counters.update(assessed_candidates_total=assessed, admitted_alternates_total=admitted - 1,
                        rejected_local_candidates_total=assessed - admitted,
                        selected_alternates_total=int(core["selected_route_id"] != core["original_route_id"]),
                        different_hypothetical_winners=int(
                            core["hypothetical_selected_route_id"]["G"] != core["hypothetical_selected_route_id"]["S"]
                        ))
        individual.append({"paid_before": event["paid_before"], "goal_pose": core["native_goal_pose"],
                           "assessed_candidates": assessed, "admitted_alternates": admitted - 1})
    return {
        **counters, "assessed_candidates_per_choice_max": max((x["assessed_candidates"] for x in individual), default=0),
        "admitted_alternates_per_choice_max": max((x["admitted_alternates"] for x in individual), default=0),
    }, individual


def recorded_failure(path, layout, mode, cause):
    failed = path / "failure.json"
    error = read(failed) if failed.is_file() else None
    heartbeat = read(path / "heartbeat.json") if (path / "heartbeat.json").is_file() else None
    return {
        "layout": layout, "mode": mode, "status": "failed_or_interrupted",
        "failed": True, "failure": error,
        "stop_reason": cause, "last_recorded_paid_not_a_terminal": None if heartbeat is None else heartbeat.get("actual_paid"),
        "failure_sha256": sha(failed) if failed.is_file() else None,
    }, [], [], None


def evaluate_episode(layout, mode, protocol, runner_freeze, binding, reference, metric, route_config):
    directory = STUDY / "runner/runs" / layout / mode
    manifest_path = directory / "manifest.json"
    if not manifest_path.exists():
        return recorded_failure(directory, layout, mode, "no_sealed_native_terminal")
    manifest = read(manifest_path)
    arm_source = {"fullG": ("G", "G"), "fullS": ("G", "S"), "originalS": ("S", None), "F_strict": (None, None)}
    require(manifest["status"] == "completed_execution" and manifest["layout"] == layout and manifest["arm"] == mode
            and (manifest["global_mode"], manifest["local_mode"]) == arm_source[mode]
            and manifest["source_commit"] == protocol["source_commit"]
            and manifest["source_protocol_sha256"] == binding["protocol_sha256"]
            and manifest["runner_freeze_sha256"] == binding["freeze_sha256"]
            and manifest["cold_started_at_paid_zero"] and not manifest["evaluator_inputs_read_by_actor"]
            and manifest["published_original_goal_selector_unswitched"]
            and manifest["actual_scene_seed"] == protocol["scene_seed"],
            f"Sealed held-out actor changed the registered starting state, seed or policy: {directory}")
    initial_path = STUDY / "runner/cold_inputs" / f"{layout}.json"
    require(sha(initial_path) == manifest["frozen_cold_input_sha256"] == runner_freeze["files_sha256"][str(initial_path)],
            f"Run has a different frozen cold-start receipt: {directory}")
    expected = {"start", "terminal", *(f"paid_{p}" for p in protocol["capture_paid"] if p > 0)}
    require(set(manifest["checkpoints"]) == expected, f"Scheduled capture set changed: {directory}")
    cap = protocol["native_budget"]
    row, seen, raw = native.evaluate_run(directory, "holdout", layout, mode, cap, reference, metric, binding, route_config)
    snapshots = {point["checkpoint"]: point for point in seen}
    trace = raw["trace"]
    initial_raw = read(directory / "initial_raw_rgbd_scan_sha256.json")
    require(initial_raw["step"] == 0 and initial_raw["packet_sha256"] == raw["start"]["packet_sha256"]
            and all(initial_raw[k] for k in RAW_SENSOR_FIELDS)
            and all(all(step.get(k) for k in RAW_SENSOR_FIELDS) for step in trace),
            f"Missing measured raw sensor values from a native action: {directory}")
    require(snapshots["start"]["actual_paid"] == 0 and snapshots["terminal"]["actual_paid"] == len(trace),
            f"Cold or terminal measured snapshot is indexed by a different clock: {directory}")
    for value in protocol["capture_paid"][1:]:
        checkpoint = snapshots[f"paid_{value}"]
        if value <= len(trace):
            require(checkpoint["status"] == "captured" and checkpoint["actual_paid"] == value,
                    f"Registered clock {value} was paid without a stored measurement: {directory}")
        else:
            require(checkpoint["status"] == "unavailable", f"Artificially padded checkpoint {value}: {directory}")
    paid_times = [item["decision_wall_s"] for item in trace]
    require(all(isfinite(t) and t >= 0 for t in paid_times) and manifest["elapsed_s"] >= 0,
            f"Invalid logged computation times: {directory}")
    full_counts, per_decision = full_pool_counts(directory, manifest, raw["choices"], reference["reachable"].shape)
    global_events = read(directory / "global_choices.json")
    global_count = sum(len(record["outputs"]["candidates"]) for record in global_events)
    row |= {
        "elapsed_s": manifest["elapsed_s"], "decision_wall_s": sum(paid_times),
        "decision_wall_s_per_paid_action": (sum(paid_times) / len(paid_times)) if paid_times else None,
        "decision_wall_s_p50": float(files.np.percentile(paid_times, 50)) if paid_times else None,
        "decision_wall_s_p95": float(files.np.percentile(paid_times, 95)) if paid_times else None,
        "native_global_choices": manifest["native_global_choices"],
        "global_candidates_total": global_count if mode != "F_strict" else None,
        "frontier_selection_candidate_count_status": ("not_logged_by_archive_interface" if mode == "F_strict" else None),
        "local_candidate_counts": per_decision,
        **full_counts,
    }
    coverage = [
        {"layout": layout, "mode": mode, "paid": paid, "correct_free_cells": free,
         "reachable_cells": row["reachable_cells"], "C_free": free / row["reachable_cells"],
         "gross_correct_free_cells": gross, "lost_correct_free_cells": lost}
        for paid, free, gross, lost in engine.per_action_curve(
            {"belief_trace": str(directory / "observed_belief_trace.npz")}, trace, seen, reference,
        )
    ]
    raw["device_hashes_at_start"] = {key: initial_raw[key] for key in RAW_SENSOR_FIELDS}
    return row, seen, coverage, raw


def audit_pair(layout, g, s, budget, route_config):
    if not g or not s:
        return {"layout": layout, "status": "failed_or_missing_arm", "verified_actual_split": False}
    proof = native.verify_category_pair(layout, g, s, budget, *route_config)
    commands = first_difference(g["trace"], s["trace"], ("action",))
    poses = first_difference(g["trace"], s["trace"], ("position", "heading"))
    device = first_difference(g["trace"], s["trace"], RAW_SENSOR_FIELDS)
    packets = first_difference(g["trace"], s["trace"], ("packet_sha256",))
    paid = commands["paid"] if commands is not None else None
    matched_prior = paid is not None and native.same_paid_prefix(g["trace"], s["trace"], paid - 1) and (
        first_difference(g["trace"][:paid - 1], s["trace"][:paid - 1], RAW_SENSOR_FIELDS) is None
    )
    causal = (bool(proof.get("causal_paid_action")) and proof["causal_paid_action"] == paid
              and poses is not None and poses["paid"] == paid and matched_prior)
    require(g["device_hashes_at_start"] == s["device_hashes_at_start"], "Cold paired raw acquisition is different")
    return proof | {
        "verified_actual_split": bool(causal), "first_command_difference": commands, "first_pose_difference": poses,
        "first_sensor_content_difference": device, "first_packet_difference": packets,
        "measured_inputs_identical_until_first_different_command": matched_prior,
        "command_difference_attribution_unresolved": commands is not None and not causal,
        "actual_paid_G": len(g["trace"]), "actual_paid_S": len(s["trace"]),
    }


def compare_outcomes(outcomes, layouts, comparators, cap, threshold):
    indexed = {(row["layout"], row["mode"]): row for row in outcomes}
    comparisons = []
    for layout in layouts:
        s = indexed[layout, "fullS"]
        for other in comparators:
            g = indexed[layout, other]
            row = {"layout": layout, "comparison": f"fullS - {other}",
                   "fullS_status": s["status"], "comparator_status": g["status"]}
            if any(m["status"] != "completed_execution" for m in (g, s)):
                comparisons.append(row | {"status": "unavailable"})
                continue
            deltas = {f"delta_{key}": s[key] - g[key] for key in (
                "J", "C_free", "F1_scene_05cm", "object_net_m2", "shelf_net_m2", "cabinet_net_m2",
                "actual_paid", "collisions", "decision_wall_s", "elapsed_s",
            )}
            outcome = deltas["delta_J"]
            material = "gain" if outcome > threshold else ("loss" if outcome < -threshold else "within_band")
            comparisons.append(row | deltas | {
                "status": "measured", "fullS_actual_paid": s["actual_paid"], "comparator_actual_paid": g["actual_paid"],
                "within_budget_both": s["actual_paid"] <= cap and g["actual_paid"] <= cap,
                "returned": s["returned"], "comparator_returned": g["returned"],
                "returned_deteriorated_against_comparator": g["returned"] and not s["returned"],
                "collisions_increased_against_comparator": deltas["delta_collisions"] > 0,
                "material_J_gain": material == "gain", "material_J_classification": material,
                **{f"delta_{key}_pp": 100 * deltas[f"delta_{key}"] for key in ("J", "C_free", "F1_scene_05cm")},
            })
    return comparisons


def paired_csv(row):
    compact = {k: row.get(k) for k in (
        "layout", "status", "verified_actual_split", "same_effective_runtime", "common_start", "causal_paid_action",
        "actual_paid_G", "actual_paid_S", "command_difference_attribution_unresolved",
    )}
    for key in ("first_command_difference", "first_pose_difference", "first_sensor_content_difference",
                "first_packet_difference"):
        change = row.get(key)
        compact[f"{key}_paid"] = None if change is None else change["paid"]
    if row.get("first_command_difference"):
        compact |= {mode + "_first_actual_action": row["first_command_difference"][mode]["action"]
                    for mode in ("fullG", "fullS")}
    return compact


def audited_gate(contrasts, outcomes, protocol, phase):
    criteria = protocol["evaluation"]["practical_gate"]
    primary = [row for row in contrasts if row["comparison"] == "fullS - fullG"]
    qualifying = [row["layout"] for row in primary if row.get("material_J_gain")]
    all_observed = all(row["status"] == "measured" for row in primary)
    returned = all_observed and not any(row["returned_deteriorated_against_comparator"] for row in primary)
    collisions = all_observed and not any(row["collisions_increased_against_comparator"] for row in primary)
    complete = phase == "all" and len(outcomes) == len(protocol["layouts"]) * len(protocol["modes"])
    within_budget = complete and all(row["status"] == "completed_execution" and
                                     row["actual_paid"] <= protocol["native_budget"] for row in outcomes)
    return {
        "status": "evaluated" if complete else "pending_baseline_batch",
        "pass": bool(returned and collisions and within_budget and len(qualifying) >= criteria["minimum_buildings"])
        if complete else None,
        "qualifying_buildings": qualifying, "count_qualifying_buildings": len(qualifying),
        "maintained_all_achieved_fullG_returns": returned, "no_increased_collisions_against_fullG": collisions,
        "all_episodes_within_budget": within_budget if complete else None,
        "registered_buildings": len(protocol["layouts"]),
        "mode": "terminal_native_stop", "threshold_delta_J_fraction_strict_gt": criteria["threshold_delta_J_fraction_strict_gt"],
    }


def supervisor_receipts(protocol, runner_freeze_path, phase):
    scopes = ("primary", "baselines") if phase == "all" else ("primary",)
    receipts = {}
    for index, section in enumerate(scopes):
        path = STUDY / "runner" / (section + "_supervisor_receipt.json")
        receipt = read(path)
        expected = {(layout, mode) for layout in protocol["layouts"] for mode in protocol["batches"][index]}
        jobs = {(job["layout"], job["arm"]) for job in receipt["jobs"]}
        require(receipt["status"] in ("completed", "recorded_failures")
                and receipt["runner_freeze_sha256"] == sha(runner_freeze_path)
                and jobs == expected and len(jobs) == len(receipt["jobs"]),
                f"Sealed supervisor docket omits, duplicates or changes expected jobs: {section}")
        for job in receipt["jobs"]:
            log = STUDY / "runner/logs" / f"{job['layout']}_{job['arm']}.log"
            files.verify(log, job["log_sha256"])
            actual = STUDY / "runner/runs" / job["layout"] / job["arm"]
            require((actual / "manifest.json").is_file() if job["returncode"] == 0 else not (actual / "manifest.json").exists(),
                    f"Sealed subprocess exit disagrees with presence of final receipt: {actual}")
        receipts[section] = sha(path)
    return receipts


def main(args):
    protocol_path = STUDY / "PROTOCOL.json"
    data_freeze_path = OUTPUT / "EVALUATOR_FREEZE.json"
    runner_freeze_path = STUDY / "runner/holdout_freeze.json"
    protocol = read(protocol_path)
    data_freeze, runner_freeze = read(data_freeze_path), read(runner_freeze_path)
    require(data_freeze["status"] == "frozen_evaluator_reference_inputs"
            and data_freeze["source_commit"] == runner_freeze["source_commit"] == protocol["source_commit"]
            and runner_freeze["status"] == "frozen_before_any_heldout_actor"
            and runner_freeze["registered_capture_paid"] == protocol["capture_paid"]
            and runner_freeze["capture_paid"] == protocol["capture_paid"][1:]
            and runner_freeze["independent_evaluator_data_freeze_sha256"] == protocol["source_evaluator_freeze_sha256"]
            and runner_freeze.get("independent_evaluator_code_and_reference_sha256", sha(data_freeze_path))
            == sha(data_freeze_path)
            and not data_freeze["heldout_policy_outcomes_read_for_freeze"]
            and not any("evaluator_references" in p or "evaluator_truth" in p or p.endswith("EVALUATOR_FREEZE.json")
                        for p in runner_freeze["files_sha256"]),
            "Actor/reference code boundary or scheduled action clocks changed")
    for path, digest in (data_freeze["inputs_sha256"] | runner_freeze["files_sha256"]).items():
        files.verify(path, digest)
    require(data_freeze["inputs_sha256"][str(Path(__file__).resolve())] == sha(Path(__file__).resolve()),
            "Evaluator wrapper was not frozen before independent cold holdout")
    config = protocol["frozen_online_fields"]
    route_config = (config["max_extra_outbound_actions"], config["min_unknown_union_ratio"])
    prior_eval = read(files.OLD_EVAL / "freeze.json")
    prior_scene = read(files.SCENES / "freeze.json")
    metric = files.load_metric(prior_scene, prior_eval)
    binding = {"protocol_sha256": sha(protocol_path), "freeze_sha256": sha(runner_freeze_path)}
    supervised = supervisor_receipts(protocol, runner_freeze_path, args.phase)
    source = read(HOLDOUT / "evaluator_freeze.json")
    layouts = protocol["layouts"]
    arms = protocol["modes"] if args.phase == "all" else protocol["batches"][0]
    outcomes, checkpoints, curves, pairs = [], [], [], []
    for layout in layouts:
        reference = native.holdout_reference(layout, source)
        raw = {}
        for arm in arms:
            result, measured, coverage, trajectory = evaluate_episode(
                layout, arm, protocol, runner_freeze, binding, reference, metric, route_config,
            )
            outcomes.append(result)
            checkpoints.extend(measured)
            curves.extend(coverage)
            raw[arm] = trajectory
        restored = [trajectory for trajectory in raw.values() if trajectory]
        if restored:
            initial = restored[0]
            require(all(native.exact_common_observation(r["start"], initial["start"])
                        and r["device_hashes_at_start"] == initial["device_hashes_at_start"] for r in restored),
                    f"Comparator used different cold initial measured sensor or mesh state: {layout}")
        pairs.append(audit_pair(layout, raw["fullG"], raw["fullS"], protocol["native_budget"], route_config))
    comparisons = compare_outcomes(outcomes, layouts, [arm for arm in arms if arm != "fullS"], protocol["native_budget"],
                                   protocol["evaluation"]["practical_gate"]["threshold_delta_J_fraction_strict_gt"])
    gate = audited_gate(comparisons, outcomes, protocol, args.phase)
    out = args.out or OUTPUT / ("results" if args.phase == "all" else "primary_results")
    out.mkdir(parents=True, exist_ok=True)
    metadata = {
        "status": "measured_read_only", "phase": args.phase, "source_commit": protocol["source_commit"],
        "protocol_sha256": binding["protocol_sha256"], "runner_freeze_sha256": binding["freeze_sha256"],
        "evaluator_freeze_sha256": sha(data_freeze_path), "supervisor_receipts_sha256": supervised,
        "layouts": data_freeze["layouts"], "independent_unit": "building",
        "stop_clock_not_imputed": True, "local_time_scope": "Decision wall time on charged steps; "
                     "shared-host simulator timings, not deployment latency and no stop-loop cost imputed.",
    }
    native.write_tables(out, "episodes", outcomes, metadata, (
        "layout", "mode", "status", "actual_paid", "native_budget", "returned", "collisions", "failed", "stop_reason",
        "last_recorded_paid_not_a_terminal", "path_length_m", "first_C_free_95_paid", "first_C_free_100_paid", *METRICS,
        *TIME_FIELDS, "source_manifest_sha256", "failure_sha256",
    ))
    native.write_tables(out, "checkpoints", checkpoints, metadata, (
        "layout", "mode", "checkpoint", "status", "actual_paid", "requested_paid", "outcome", *METRICS,
        "source_checkpoint_sha256",
    ))
    native.write_tables(out, "coverage_curve", curves, metadata)
    native.write_tables(out, "contrasts", comparisons, metadata)
    compact = list(map(paired_csv, pairs))
    files.write_csv(out / "pair_causal_audit.csv", compact, list(compact[0]))
    files.write_json(out / "pair_causal_audit.json", metadata | {"rows": pairs})
    files.write_json(out / "engineering_gate.json", metadata | {"engineering_gate": gate, "input_evidence_sha256": {
        name: sha(out / name) for name in ("episodes.json", "checkpoints.json", "contrasts.json", "pair_causal_audit.json")
    }})
    print(gate)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=("primary", "all"), default="all")
    parser.add_argument("--out", type=Path)
    main(parser.parse_args())
