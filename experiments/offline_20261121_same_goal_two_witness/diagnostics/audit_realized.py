"""Independently bind each sealed witness run to its observed actions and sensors."""

import argparse
import csv
from pathlib import Path

import numpy as np

from source_mechanism import (
    ANALYSES, OUT, SOURCE_KEYS, array_sha, load, old_run, one_between, require, sha, trace_pose, written,
)

ROOT = OUT.parent
MODES = ("G", "S")
ACQUISITION = ("depth_values_sha256", "rgb_values_sha256", "scan_ranges_sha256")
PACKET = (*ACQUISITION, "frame_sha256", "packet_sha256", "nonsemantic_packet_sha256", "owner_sha256", "packet_chain_sha256")


def read_sealed(path, expected, inputs):
    digest = sha(path)
    require(digest == expected, f"Input hash fails source seal: {path}")
    inputs[str(path)] = digest
    return load(path)


def write_csv(path, rows):
    columns = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", newline="") as stream:
        output = csv.DictWriter(stream, columns, lineterminator="\n")
        output.writeheader()
        output.writerows(rows)


def audit_run(layout, mode, case, protocol, protocol_path, run_freeze, inputs):
    run = ROOT / "runner/runs" / layout / mode
    if not (run / "manifest.json").exists():
        failure = run / "failure.json"
        require(failure.exists(), f"Run has not sealed or documented failure: {run}")
        inputs[str(failure)] = sha(failure)
        return {"layout": layout, "mode": mode, "status": "retained_execution_failure",
                "reason": load(failure)["error"]}, None
    manifest, checks = load(run / "manifest.json"), {}
    inputs[str(run / "manifest.json")] = sha(run / "manifest.json")
    require(
        manifest["status"] == "completed_execution" and manifest["layout"] == layout and manifest["arm"] == mode
        and manifest["source_commit"] == protocol["source_commit"]
        and manifest["source_protocol_sha256"] == sha(protocol_path) and manifest["runner_freeze_sha256"] == sha(run_freeze)
        and manifest["actual_closed_loop"] and not manifest["evaluator_inputs_read_by_actor"]
        and not manifest["forced_goal"] and not manifest["forced_route"]
        and manifest["global_mode"] == manifest["subsequent_local_mode"] == "G"
        and manifest["subsequent_local_alternate_cap"] == 4,
        f"Manifest does not describe the authorized one-choice experiment: {run}",
    )
    for name in ("intervention.json", "local_choices.json", "global_choices.json", "audit.json", "steps.json"):
        checks[name] = read_sealed(run / name, manifest["files_sha256"][name], inputs)
    prefix, call_id = case["paid_before"], case["global_call_id"]
    decision, local, global_, runtime, steps = (checks[name] for name in (
        "intervention.json", "local_choices.json", "global_choices.json", "audit.json", "steps.json",
    ))
    require(
        manifest["replayed_archived_prefix_actions"] == prefix
        and manifest["actual_new_actions"] == len(steps) - prefix
        and manifest["native_budget"] == protocol["total_paid_action_budget"]
        and manifest["paid_actions"] == len(steps) <= manifest["native_budget"]
        and [row["step"] for row in steps] == list(range(1, len(steps) + 1))
        and all(row["charged_new_continuation_action"] == (row["step"] > prefix) for row in steps),
        f"Paid trace or archive prefix counts mismatch: {run}",
    )
    source = old_run(layout, "newG", inputs)
    require(all(all(old[key] == new[key] for key in SOURCE_KEYS)
                for old, new in zip(source["steps.json"][:prefix], steps)),
            f"Restored prefix action, calibrated frame, packet or measured state diverges: {run}")
    require(decision["prior_source_state_id"] == case["state_id"] and decision["paid_before"] == prefix
            and decision["global_call_id"] == call_id and decision["expanded_pool_and_scores_equal_to_offline_source"]
            and decision["native_selected_option_unchanged"] and decision["prior_prefix_reproduced_before_selection"]
            and not decision["forced_goal"] and not decision["forced_route"],
            f"No verified expanded action choice: {run}")
    rng = manifest["restored_rng_identity"]
    require(rng == decision["restored_rng_identity"] and rng["scene_seed"] == 20261121
            and rng["seed_components_when_depth_noise_enabled"] == ["scene_seed", "paid_frame_id", 739]
            and rng["deterministic_pose_geometry"] and rng["depth_noise_sigma_m"] == rng["depth_dropout_fraction"] == 0
            and rng["stateful_generator_retained_by_simulator"] is False
            and rng["source_packet_chain_required_through_paid"] == prefix,
            f"Simulator seed/noise contract differs from deterministic archived source: {run}")
    source_state = ANALYSES / "same_goal_pool_audit_20261121" / case["source_state"]
    require(decision["witness_file"] == str(source_state) and decision["witness_sha256"] == sha(source_state),
            f"Witness provenance is not the archived offline score state: {run}")
    inputs[str(source_state)] = sha(source_state)
    old_expanded = load(source_state)["expanded"]
    old_summary = old_expanded["summary"]
    winner = old_summary["expanded_hypothetical_winners"][mode]
    chosen = one_between(old_expanded["candidates"], lambda row: row["route_id"] == winner,
                         f"{mode} original offline expanded winner")
    require(decision["chosen_route_id"] == winner and decision["selected_outbound_actions"] == chosen["outbound_actions"]
            and decision["selected_outbound_states"] == chosen["outbound_states"]
            and decision["native_target"] == chosen["goal_pose"] == old_summary["native_goal_pose"]
            and decision["expanded_candidate_pool_sha256"] == old_summary["expanded_candidate_pool_sha256"]
            and decision["expanded_score_table_sha256"] == old_summary["expanded_score_table_sha256"]
            and decision["actual_return_actions"] == decision["original_return_actions"] == old_summary["native_return_actions"],
            f"Executed sidecar was not emitted by the frozen expanded selector: {run}")
    treatment = one_between(local, lambda row: row["intervention"], "single local intervention")
    require(all(row["choice_audit"]["selected_mode"] == (mode if row["intervention"] else "G")
                and (row["intervention"] or row["choice_audit"]["admitted_candidate_count"] <= 5)
                for row in local)
            and all(row["outputs"]["mode"] == "G" for row in global_)
            and treatment["selection_call_id"] == call_id
            and treatment["active_outbound_sha256"] == chosen["outbound_sha256"]
            and treatment["native_return_states"] == treatment["rebound_return_states"] == old_summary["native_return_states"],
            "Found a second treatment, changed native goal mode or nonneutral follow-on choice")
    initial_choice = one_between(global_, lambda row: row["call_id"] == call_id, "native fixed-global target call")
    require(initial_choice["outputs"]["selected"]["pose"] == decision["native_target"], f"Global pose mismatch at {run}")
    native_guard = {row["next_action_id"]: row for row in runtime if row["event"] == "paid_action_authorized"}
    require(len(native_guard) == len(steps) and all(
        native_guard[step["step"]]["assessment"]["allowed"] and step["guard_allowed"]
        and native_guard[step["step"]]["assessment"]["action"] == step["action"]
        and native_guard[step["step"]]["assessment"]["reserved_return_cost"] == step["reserved_return_cost"]
        and native_guard[step["step"]]["option_id"] == step["active_option_id"]
        and (step["collision"] or native_guard[step["step"]]["assessment"]["next_pose"] == trace_pose(step))
        for step in steps
    ), f"Actual actions violate the archived observation and native guard accounting: {run}")
    before, after = decision["pre_call_native_guard"], decision["post_call_native_guard"]
    released = steps[prefix] if len(steps) > prefix else None
    if released:
        require(after and after[-1]["action_id"] == prefix and after[-1]["call_id"] > call_id
                and after[-1]["outputs"]["allowed"] and after[-1]["outputs"]["action"] == released["action"]
                and (released["collision"] or after[-1]["outputs"]["next_pose"] == trace_pose(released)),
                f"The native guard did not authorize the first released action: {run}")
    if layout == "warehouse_right":
        refusal = before[-1]
        replan = one_between(decision["runtime_same_frame_replan_events"], lambda row: row["frame_id"] == f"frame-{prefix}",
                             "same-frame native replan")
        require(refusal["action_id"] == prefix and refusal["call_id"] < call_id
                and (not after or call_id < after[-1]["call_id"])
                and not refusal["outputs"]["allowed"] and refusal["outputs"]["reason"] == "next_footprint_not_known_safe"
                and refusal["outputs"]["action"] == "forward" and refusal["outputs"]["next_pose"] == [33,38,3]
                and refusal["executed_option_id"] == replan["rejected_option_id"] == steps[prefix-1]["active_option_id"]
                and replan["selected_option_id"] == decision["native_option_id"],
                f"Warehouse intervention missed the original dynamic guard/replanning sequence: {run}")
    else:
        require(not before and not decision["runtime_same_frame_replan_events"]
                and steps[prefix-1]["matched_target_pose_after_step"],
                f"Gallery intervention failed the normal completion-to-new-choice sequence: {run}")
    bound = [row for row in steps if row["bound_witness_option"]]
    require(all(
        row["active_outbound_sha256"] == chosen["outbound_sha256"] and row["route_action_index_zero_based"] == n
        and row["action"] == chosen["outbound_actions"][n]
        and (row["collision"] or trace_pose(row) == chosen["outbound_states"][n+1])
        for n, row in enumerate(bound)
    ), f"Bound option actual commanded state sequence does not follow selected sidecar: {run}")
    arrivals = [row["step"] for row in bound if row["matched_target_pose_after_step"] and not row["collision"]]
    require(decision["actual_option_arrival_paid"] == (arrivals[0] if arrivals else None)
            and decision["charged_witness_option_actions"] == len(bound)
            and manifest["collisions"] == sum(row["collision"] for row in steps)
            and manifest["returned"] == (manifest["final_pose"] == manifest["initial_pose"]),
            f"Return, collision, arrival or paid trajectory is not independently corroborated: {run}")
    belief_path = run / "observed_belief_trace.npz"
    require(sha(belief_path) == manifest["files_sha256"][belief_path.name], "Observed-grid ledger changed")
    inputs[str(belief_path)] = sha(belief_path)
    with np.load(belief_path, allow_pickle=False) as data:
        require(list(data["paid"]) == list(range(len(steps) + 1))
                and array_sha(data["belief"][prefix]) == decision["measured_prefix"]["belief_sha256"],
                f"Frozen witness map does not match known observed-path history: {run}")
    row = {
        "layout": layout, "mode": mode, "status": manifest["status"], "archive_prefix_paid": prefix,
        "terminal_paid": len(steps), "new_paid_actions": len(steps)-prefix, "returned": manifest["returned"],
        "collisions": manifest["collisions"], "native_stop": manifest["observed_native_termination"]["reason"],
        "chosen_route_id": winner, "nominal_outbound_cost": chosen["outbound_cost"], "bound_actions_actual": len(bound),
        "actual_witness_arrival_paid": decision["actual_option_arrival_paid"],
        "first_charged_action": released["action"] if released else None,
        "first_accepted_successor_return_cost": after[-1]["outputs"]["reserved_return_cost"] if after else None,
        "source_archive_prefix_matches_exact": True, "single_expanded_selection_verified": True,
        "native_rejection_before_expanded_warehouse_selection": bool(before),
        "native_rejection_call_id": before[-1]["call_id"] if before else None, "native_global_call_id": call_id,
        "first_stock_authorization_after_call": after[-1]["call_id"] if after else None,
    }
    return row, {"steps": steps, "decision": decision, "selected": chosen}


def different_clock(a, b, field):
    return next((x["step"] for x, y in zip(a, b) if x[field] != y[field]), None)


def audit_pair(layout, case, data):
    g, s = data
    base = {"layout": layout, "common_prefix_paid": case["paid_before"], "nominal_first_difference_paid": case["expected_first_nominal_split"]["paid"]}
    if not g or not s:
        return base | {"status": "incomplete_pair"}, []
    gs, ss = g["steps"], s["steps"]
    cut = case["paid_before"]
    states = list(zip(gs[cut:cut+case.get("verify_common_first_outbound_observations",0)],
                      ss[cut:cut+case.get("verify_common_first_outbound_observations",0)]))
    sensor_rows = []
    for a, b in states:
        require(a["step"] == b["step"], "Paired sensor array checkpoint uses different clocks")
        sensor_rows.append({"layout": layout, "paid": a["step"], "same_action": a["action"] == b["action"],
                            "same_measured_pose": trace_pose(a) == trace_pose(b),
                            **{field+"_matches": a[field] == b[field] for field in PACKET}})
    require(all(all(a[key] == b[key] for key in (*SOURCE_KEYS, *PACKET)) for a, b in zip(gs[:cut], ss[:cut])),
            f"Paired arms do not share the actual archive observations up to branchpoint {layout}")
    action = different_clock(gs, ss, "action")
    pose = next((a["step"] for a,b in zip(gs,ss) if trace_pose(a)!=trace_pose(b)), None)
    sensor = next((a["step"] for a,b in zip(gs,ss) if any(a[key]!=b[key] for key in ACQUISITION)), None)
    packet = different_clock(gs, ss, "packet_sha256")
    closure = {}
    same_branch_context = (g["decision"]["measured_prefix"] == s["decision"]["measured_prefix"]
                           and g["decision"]["native_target"] == s["decision"]["native_target"]
                           and g["decision"]["restored_rng_identity"] == s["decision"]["restored_rng_identity"]
                           and g["decision"]["expanded_candidate_pool_sha256"] == s["decision"]["expanded_candidate_pool_sha256"]
                           and g["decision"]["expanded_score_table_sha256"] == s["decision"]["expanded_score_table_sha256"])
    require(same_branch_context, "G/S were compared with unmatched observed state, goal, candidate and score data")
    if action is not None:
        ga, sa = gs[action-1], ss[action-1]
        pre_ga, pre_sa = gs[action-2], ss[action-2]
        closure = {
            "G_first_different_action": ga["action"], "S_first_different_action": sa["action"],
            "shared_pre_pose_at_action_split": trace_pose(pre_ga) if trace_pose(pre_ga) == trace_pose(pre_sa) else None,
            "G_measured_post_pose": trace_pose(ga), "S_measured_post_pose": trace_pose(sa),
            "both_witness_bound_at_actual_split": ga["bound_witness_option"] and sa["bound_witness_option"],
            "paid_action_split_matches_nominal_pair": case["expected_first_nominal_split"] ==
                {"paid": action, "G": ga["action"], "S": sa["action"]},
        }
    return base | {
        "status": "paired_complete", "same_acquired_start_and_expanded_comparison": same_branch_context,
        "first_paid_action_difference": action, "first_measured_pose_difference": pose,
        "first_raw_depth_rgb_or_scan_value_difference": sensor, "first_packet_digest_difference": packet,
        **{"first_"+key+"_difference": different_clock(gs, ss, key) for key in ACQUISITION},
        "expected_equal_postchoice_sensor_observations": case.get("verify_common_first_outbound_observations",0),
        "verified_equal_postchoice_sensor_observations": sum(all(val for key,val in row.items()
            if key.endswith("_matches") or key.startswith("same_")) for row in sensor_rows),
        **closure,
    }, sensor_rows


def main(output):
    protocol_path = ROOT / "PROTOCOL.json"
    run_freeze = ROOT / "runner/freeze.json"
    frozen_runner = load(run_freeze)
    protocol = load(protocol_path)
    actor = ROOT / "runner/continue_witness.py"
    require(sha(actor) == frozen_runner["files_sha256"][str(actor)], "Actor source differs from frozen checksum")
    inputs = {str(p): sha(p) for p in (Path(__file__), protocol_path, run_freeze, actor)}
    arms, pairs, common_sensors = [], [], []
    for layout in protocol["layouts_in_order"]:
        case = protocol["pairs"][layout]
        data = []
        for mode in MODES:
            row, run = audit_run(layout, mode, case, protocol, protocol_path, run_freeze, inputs)
            arms.append(row)
            data.append(run)
        row, sensors = audit_pair(layout, case, data)
        pairs.append(row)
        common_sensors.extend(sensors)
    output.mkdir(parents=True, exist_ok=True)
    pages = {"realized_arm_audit.csv": arms, "realized_pair_mechanism.csv": pairs,
             "gallery_common_sensor_checks.csv": common_sensors}
    for name, rows in pages.items():
        write_csv(output / name, rows)
    written(output / "realized_mechanism_receipt.json", {
        "status": "read_only_witness_action_sensor_and_native_guard_audit", "actor_started": False,
        "per_arm": arms, "per_pair": pairs,
        "inputs_sha256": dict(sorted(inputs.items())),
        "outputs_sha256": {name: sha(output / name) for name in pages},
    })


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=OUT)
    main(parser.parse_args().output)
