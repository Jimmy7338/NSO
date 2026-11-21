"""Read-only world, execution and native-reconstruction checks for same-goal routes."""

import argparse
import hashlib
import importlib.util
import json
import math
from pathlib import Path

STUDY = Path(__file__).resolve().parents[1]
PERSISTENCE = STUDY.parent / "candidate_route_persistence_20261121"
YIELD = STUDY.parent / "candidate_route_yield_20261121"
PERSISTENCE_EVAL = PERSISTENCE / "measurement/evaluate.py"


def read_persistence_evaluator():
    spec = importlib.util.spec_from_file_location("nso_persistence_evaluator", PERSISTENCE_EVAL)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


engine = read_persistence_evaluator()
files = engine.prior
TERMINAL_METRICS = (
    "C_free", "correct_free_cells", "reachable_cells", "F1_scene_05cm", "p_scene_05cm", "r_scene_05cm", "J",
    "delta_C_free", "delta_F1_scene_05cm", "delta_J", "object_gross_m2", "object_lost_m2", "object_net_m2",
    "shelf_net_m2", "cabinet_net_m2", "scene_gross_m2", "scene_lost_m2", "scene_net_m2",
)
TRACE_ACTION_FIELDS = ("action", "position", "heading")
OBSERVATION_FIELDS = ("pose", "packet_sha256", "belief_sha256", "packet_chain_sha256", "native_mesh_fields")


def exact_common_observation(a, b):
    return all(a.get(field) == b.get(field) and a.get(field) is not None for field in OBSERVATION_FIELDS)


def within_tolerance(a, b, tolerance=1e-9):
    return math.isclose(a, b, abs_tol=tolerance, rel_tol=0)


def canonical_checksum(value):
    data = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(data.encode()).hexdigest()


def verify_new_file(directory, filename, manifest):
    path = directory / filename
    files.verify(path, manifest["files_sha256"][filename])
    return path


def verify_trace(trace, initial, cap, manifest):
    checked = engine.check_trace(trace, initial["initial_packet_sha256"], cap, manifest)
    if "initial_packet_sha256" in manifest and manifest["initial_packet_sha256"] != initial["initial_packet_sha256"]:
        raise ValueError("Run received a different initial packet")
    return checked


def evaluate_snapshot(directory, name, manifest, initial, trace, reference, native_metric):
    receipt, paths, _, digest = engine.snapshot(directory, name, manifest, "native_observed")
    if receipt["status"] == "unavailable":
        return receipt | {"source_checkpoint_sha256": digest}, None
    expected = name if name == "terminal" else 0 if name == "start" else int(name.removeprefix("paid_"))
    engine.verify_checkpoint_against_trace(receipt, expected, trace, initial)
    measured, masks = engine.evaluate_verified_snapshot(
        paths["native_mesh.ply"], paths["belief.npz"], receipt, reference, native_metric,
    )
    return receipt | measured | {"source_checkpoint_sha256": digest}, masks


def charged_first_difference(trace_a, trace_b, start_paid):
    """Return action and endpoint difference at the first shared-clock divergence."""
    for action_a, action_b in zip(trace_a[start_paid:], trace_b[start_paid:], strict=False):
        if action_a["step"] != action_b["step"]:
            raise ValueError("Trace clocks disagree at same index")
        if any(action_a[key] != action_b[key] for key in TRACE_ACTION_FIELDS):
            return {
                "paid": action_a["step"], "actions": [x["action"] for x in (action_a, action_b)],
                "post_poses": [[*x["position"], x["heading"]] for x in (action_a, action_b)],
                "actual_action_changed": action_a["action"] != action_b["action"],
                "observed_pose_changed": (
                    action_a["position"] != action_b["position"] or action_a["heading"] != action_b["heading"]
                ),
            }
    return None


def first_pose_match(trace, target, paid_after=0):
    for step in trace[paid_after:]:
        pose = [*step["position"], step["heading"]]
        if pose == list(target) and not step["collision"]:
            return step["step"]
    return None


def planned_actions_through_paid(candidate, trace, choice, through_paid):
    """Check that the selector's path caused the charged actions through divergence."""
    paid_before = choice["paid_before"]
    expected_actions = candidate["outbound_actions"]
    expected_states = candidate["outbound_states"]
    completed = through_paid - paid_before
    if not 0 < completed <= len(expected_actions) or len(expected_states) != len(expected_actions) + 1:
        return False
    for index, step in enumerate(trace[paid_before:through_paid]):
        if (step["action"] != expected_actions[index]
                or [*step["position"], step["heading"]] != list(expected_states[index + 1])
                or step["collision"] or not step["guard_allowed"]
                or step["active_option_id"] != choice["active_option_id"]
                or step["local_route_id"] != candidate["route_id"]
                or step["active_outbound_sha256"] != choice["chosen_outbound_sha256"]
                or step["route_action_index_zero_based"] != index):
            return False
    return len(trace) >= through_paid


def same_paid_prefix(a, b, paid_before):
    if len(a) < paid_before or len(b) < paid_before:
        return False
    common_fields = ("step", "action", "position", "heading", "packet_sha256", "collision")
    return all(all(left[key] == right[key] for key in common_fields)
               for left, right in zip(a[:paid_before], b[:paid_before], strict=True))


def verify_candidate_nominal(route, native, goal, reserve_max, gain_fraction=.95):
    """Independently inspect the frozen admissibility fields for an alternate."""
    outbound = route["outbound_actions"]
    states = route["outbound_states"]
    if len(states) != len(outbound) + 1 or list(states[-1]) != list(goal):
        raise ValueError("Candidate route does not attain the same pose, including heading")
    if len(outbound) != route["outbound_cost"] or len(outbound) > len(native["outbound_actions"]) + reserve_max:
        raise ValueError("Admitted alternate exceeds the additional action budget")
    if route["reserved_return_cost"] != native["reserved_return_cost"]:
        raise ValueError("Admitted alternate replaced the original return reserve")
    if route["unknown_union_cells"] < gain_fraction * native["unknown_union_cells"] - 1e-9:
        raise ValueError("Admitted alternate fails the neutral predicted coverage requirement")
    if route["common_score"] != 0.0:
        raise ValueError("An additive fee changed the eligible route score")
    planned = {key: route[key] for key in ("outbound_actions", "outbound_states")}
    if canonical_checksum(planned) != route["outbound_sha256"]:
        raise ValueError("Candidate action list or anticipated poses do not match the stamped trace")
    if route["route_id"] != native["route_id"] and (
        not route["novel_view_tags"] or route["waypoint_pose"] not in states
    ):
        raise ValueError("An admitted alternate has no new view sector or one-waypoint trace")


def verified_route_choice(choice, trace, mode, cap, reserve_max, gain_fraction):
    rows = choice["candidates"]
    candidates = {r["route_id"]: r for r in rows}
    if (len(candidates) != len(rows)
            or canonical_checksum([{key: value for key, value in row.items() if key != "scores"} for row in rows])
            != choice["candidate_pool_sha256"]
            or canonical_checksum([row["scores"] for row in rows]) != choice["score_table_sha256"]):
        raise ValueError("Candidate source pool/score checksum changed")
    if not candidates:
        if choice["selected_route_id"] is not None:
            raise ValueError("Recorded winner has no original candidate data")
        return candidates
    native = candidates[choice["original_route_id"]]
    paid_before = choice["paid_before"]
    if not 0 <= paid_before <= len(trace):
        raise ValueError("Choice was recorded outside its charged-action clock")
    pose = None
    packet_sha = None
    if paid_before:
        row = trace[paid_before - 1]
        pose = [*row["position"], row["heading"]]
        packet_sha = row["packet_sha256"]
    for row in candidates.values():
        if row["admitted"]:
            verify_candidate_nominal(row, native, choice["native_goal_pose"], reserve_max, gain_fraction)
            if pose is not None and (list(row["outbound_states"][0]) != pose
                                     or choice["packet_sha256"] != packet_sha):
                raise ValueError("Local route starts from a different actual observation")
            if paid_before + row["outbound_cost"] + row["reserved_return_cost"] > cap:
                raise ValueError("Admitted route and its reserved return exceed episode cap")
    admitted = [row for row in candidates.values() if row["admitted"]]
    if len({tuple(r["outbound_actions"]) for r in admitted}) != len(admitted):
        raise ValueError("Identical actuator commands were admitted twice")
    for score_mode in ("G", "S"):
        score_winner = candidates[choice["hypothetical_selected_route_id"][score_mode]]
        maximum = max(r["scores"][score_mode] for r in admitted)
        tied = [r for r in admitted if within_tolerance(r["scores"][score_mode], maximum)]
        winner = min(tied, key=lambda r: (r["outbound_cost"], r["route_id"] != native["route_id"], r["outbound_sha256"]))
        if score_winner != winner:
            raise ValueError("Recorded local winner did not maximize its declared score")
    if (choice["selected_mode"] != mode or
            choice["selected_route_id"] != choice["hypothetical_selected_route_id"][mode]):
        raise ValueError("Enacted choice differs from its declared sole scoring mode")
    selected = candidates[choice["selected_route_id"]]
    if (selected["outbound_actions"] != choice["selected_outbound_actions"]
            or selected["outbound_states"] != choice["selected_outbound_states"]
            or native["outbound_actions"] != choice["native_outbound_actions"]
            or native["outbound_states"] != choice["native_outbound_states"]
            or choice["native_return_actions"] != choice["rebound_return_actions"]
            or choice["native_return_states"] != choice["rebound_return_states"]
            or len(choice["native_return_actions"]) != native["reserved_return_cost"]):
        raise ValueError("Local actuator binding differs from its candidate or the native reserved return")
    return candidates


def verify_category_pair(layout, neutral, semantic, cap, reserve_max, gain_fraction):
    trace_g = neutral["trace"]
    trace_s = semantic["trace"]
    difference = charged_first_difference(trace_g, trace_s, 0)
    default = {
        "layout": layout, "neutral_mode": neutral["mode"], "semantic_mode": semantic["mode"],
        "status": "checked_no_attributed_difference", "common_start": exact_common_observation(
            neutral["start"], semantic["start"],
        ), "first_observed_difference": difference, "causal_paid_action": None, "paired_events": [],
        "same_effective_runtime": neutral["effective_runtime"] == semantic["effective_runtime"],
        "identical_charged_action_and_packet_trace": (
            len(trace_g) == len(trace_s) and same_paid_prefix(trace_g, trace_s, len(trace_g))
        ),
        "identical_terminal_mesh": (
            neutral["terminal"]["files_sha256"]["native_mesh.ply"] == semantic["terminal"]["files_sha256"]["native_mesh.ply"]
        ),
    }
    if not default["common_start"]:
        return default | {"status": "different_frozen_start"}
    if not default["same_effective_runtime"]:
        return default | {"status": "different_effective_runtime"}
    choices_g = neutral["choices"]
    choices_s = semantic["choices"]
    pair_rows = []
    for g, s in zip(choices_g, choices_s, strict=False):
        paid = g["paid_before"]
        if paid != s["paid_before"]:
            break
        if difference and paid >= difference["paid"]:
            break
        same_pool = g["candidates"] == s["candidates"] and g["candidate_pool_sha256"] == s["candidate_pool_sha256"]
        same_observation = same_paid_prefix(trace_g, trace_s, paid) and all(
            g[key] == s[key] for key in ("packet_sha256", "belief_sha256", "native_outbound_sha256", "native_goal_pose")
        )
        same_global = g["global_event"]["inputs"] == s["global_event"]["inputs"] and (
            g["global_event"]["outputs"] == s["global_event"]["outputs"]
        )
        if not (same_observation and same_pool and same_global):
            pair_rows.append({"paid_before": paid, "common_observation_goal": same_observation,
                              "same_route_pool_and_scores": same_pool, "same_global_selection": same_global,
                              "status": "comparison_context_changed"})
            break
        candidates_g = verified_route_choice(g, trace_g, "G", cap, reserve_max, gain_fraction)
        candidates_s = verified_route_choice(s, trace_s, "S", cap, reserve_max, gain_fraction)
        selected = [g["selected_route_id"], s["selected_route_id"]]
        distinct = g["chosen_outbound_sha256"] != s["chosen_outbound_sha256"]
        detail = {
            "paid_before": paid, "goal_pose": g["native_goal_pose"],
            "common_observation_goal": True, "same_route_pool_and_scores": True, "same_global_selection": True,
            "packet_sha256": g["packet_sha256"], "belief_sha256": g["belief_sha256"],
            "native_outbound_sha256": g["native_outbound_sha256"], "candidate_pool_sha256": g["candidate_pool_sha256"],
            "legal_alternates": sum(bool(r["admitted"]) for r in candidates_g.values()) - 1,
            "selected_route_ids": selected, "selected_mode_winners_different": distinct,
            "selected_scores": [candidates_g[selected[0]]["scores"], candidates_s[selected[1]]["scores"]],
            "status": "common_choice" if not distinct else "different_plan_no_execution_proof",
        }
        if distinct and difference and difference["actual_action_changed"] and same_paid_prefix(
            trace_g, trace_s, difference["paid"] - 1,
        ):
            paid_difference = difference["paid"]
            bound_g = planned_actions_through_paid(candidates_g[selected[0]], trace_g, g, paid_difference)
            bound_s = planned_actions_through_paid(candidates_s[selected[1]], trace_s, s, paid_difference)
            if bound_g and bound_s:
                detail["status"] = "different_category_choice_executed"
                detail["applied_routes_match_each_actual_prefix"] = True
                detail["actual_difference"] = difference
                return default | {"status": detail["status"], "causal_paid_action": paid_difference,
                                  "paired_events": [*pair_rows, detail]}
        pair_rows.append(detail)
    return default | {"paired_events": pair_rows}


def close_record(receipt, summary, trace, initial):
    termination = summary.get("termination")
    if termination is None:
        raise ValueError("Episode terminal receipt has no native stop outcome")
    expected_pose = [*trace[-1]["position"], trace[-1]["heading"]] if trace else initial["anchor"]
    closed_at_anchor = expected_pose == initial["anchor"]
    if (receipt["actual_paid"] != len(trace) or receipt["pose"] != expected_pose
            or termination["final_action_id"] != len(trace) or termination["returned_to_anchor"] != closed_at_anchor):
        raise ValueError("Native closure time or actual pose differs from measured final state")
    return {
        "actual_paid": len(trace), "returned": closed_at_anchor,
        "collisions": sum(bool(row["collision"]) for row in trace),
        "stop_reason": termination["reason"], "failed": bool(termination.get("failed")),
        "terminal_pose": expected_pose, "termination": termination,
        "path_length_m": trace[-1]["path_length_m"] if trace else 0.0,
    }


def verify_logged_choices(directory, manifest, trace, cap, route_config):
    global_path = verify_new_file(directory, "global_choices.json", manifest)
    local_path = verify_new_file(directory, "local_choices.json", manifest)
    global_events = files.read_json(global_path)
    locals_raw = files.read_json(local_path)
    event_by_id = {event["call_id"]: event for event in global_events}
    if len(event_by_id) != len(global_events) or any(
        event["outputs"]["mode"] != manifest["global_mode"] for event in global_events
    ):
        raise ValueError(f"Recorded global selector changed modes: {directory}")
    output = []
    for raw in locals_raw:
        decision = raw["choice_audit"]
        target = raw["same_global_target"]
        global_event = event_by_id[raw["selection_call_id"]]
        native = global_event["outputs"]["selected"]
        if (
            raw["global_mode"] != manifest["global_mode"] or raw["local_mode"] != manifest["local_mode"]
            or target != native["pose"] or target != decision["native_goal_pose"]
            or native["outbound_actions"] != raw["native_outbound_actions"]
            or native["outbound_states"] != raw["native_outbound_states"]
            or native["return_actions"] != raw["native_return_actions"]
            or native["return_states"] != raw["native_return_states"]
            or native["option_id"] != raw["active_option_id"]
            or raw["packet_sha256"] != decision["packet_sha256"]
            or raw["observed_belief_sha256"] != decision["belief_sha256"]
            or raw["active_local_route_id"] != decision["selected_route_id"]
            or raw["active_outbound_sha256"] != decision["chosen_outbound_sha256"]
            or raw["remaining_budget"] != cap - raw["paid_before"]
            or decision["paid_before"] != raw["paid_before"]
        ):
            raise ValueError(f"Local scoring or binding record altered the native global target: {directory}")
        normalized = decision | {
            "paid_before": raw["paid_before"], "active_option_id": raw["active_option_id"], "global_event": global_event,
            **{key: raw[key] for key in (
                "selected_outbound_actions", "selected_outbound_states", "native_outbound_actions",
                "native_outbound_states", "native_return_actions", "rebound_return_actions",
                "native_return_states", "rebound_return_states",
            )},
        }
        verified_route_choice(normalized, trace, raw["local_mode"], cap, *route_config)
        output.append(normalized)
    if (len(output) != manifest["local_route_choices"] or len(global_events) != manifest["native_global_choices"]):
        raise ValueError(f"Choice table and run counts disagree: {directory}")
    return output


def enumerate_arrivals(choices, trace):
    arrivals = []
    for event in choices:
        goal = event["native_goal_pose"]
        after = event["paid_before"]
        bound = event["active_option_id"]
        end = next((step for step in trace[after:] if step["active_option_id"] == bound
                    and [*step["position"], step["heading"]] == goal and not step["collision"]), None)
        arrivals.append({
            "issued_at_paid": after, "selected_goal_pose": goal, "option_id": bound,
            "route_id": event["selected_route_id"], "reached_under_same_option": end is not None,
            "first_reached_paid": end["step"] if end is not None else None,
            "final_target_appeared_in_trace_even_if_replanned": first_pose_match(trace, goal, after),
        })
    return arrivals


def verify_action_guard(trace, cap):
    for step in trace:
        if not step["guard_allowed"]:
            raise ValueError(f"Actuator applied a guard-rejected action at step {step['step']}")
        reserved = step["reserved_return_cost"]
        if reserved is not None and step["step"] + reserved > cap:
            raise ValueError(f"Applied action compromises the reserved return at step {step['step']}")


def evaluate_run(directory, stage, layout, arm, cap, ref, native_metric, run_binding, route_config):
    initial = ref["initial"]
    if (directory / "failure.json").exists() and not (directory / "manifest.json").exists():
        failure = files.read_json(directory / "failure.json")
        if (failure["stage"], failure["layout"], failure["arm"]) != (stage, layout, arm):
            raise ValueError(f"Run fail record has wrong identity: {directory}")
        return {"stage": stage, "layout": layout, "mode": arm, "status": "failed", "failed": True,
                "error": failure["error"], "source_manifest_sha256": files.sha256(directory / "failure.json")}, [], []
    manifest, digest = engine.verify_manifest(directory, "manifest.json")
    if (manifest["stage"] != stage or manifest["layout"] != layout or manifest["arm"] != arm
            or not manifest["actual_closed_loop"] or manifest["forced_route_or_goal"]
            or manifest["evaluator_inputs_read_by_actor"] or manifest["native_budget"] != cap):
        raise ValueError(f"Run manifest failed autonomy or study membership check: {directory}")
    attempt = files.read_json(directory / "attempt.json")
    if (attempt["runner_freeze_sha256"] != run_binding["freeze_sha256"]
            or attempt["source_protocol_sha256"] != run_binding["protocol_sha256"]
            or attempt["initial_owner_sha256"] != initial["initial_owner_sha256"]
            or attempt["initial_pose"] != initial["anchor"]):
        raise ValueError(f"Attempt did not use the exact frozen start and runtime inputs: {directory}")
    trace = files.read_json(directory / "steps.json")
    summary = files.read_json(directory / "runtime_summary.json")
    terminal_chain = verify_trace(trace, initial, cap, manifest)
    verify_action_guard(trace, cap)
    choices = verify_logged_choices(directory, manifest, trace, cap, route_config)
    snapshots, rows = {}, []
    for name in manifest["checkpoints"]:
        receipt, masks = evaluate_snapshot(directory, name, manifest, initial, trace, ref, native_metric)
        if receipt["status"] == "captured":
            if name == "start":
                baseline, base_masks = receipt, masks
                if receipt["belief_sha256"] != manifest["initial_belief_sha256"]:
                    raise ValueError(f"Native start map is different from the declared start in {directory}")
            receipt = engine.augment_measurement(receipt, masks, baseline, base_masks, ref)
        snapshots[name] = receipt
        rows.append({"stage": stage, "layout": layout, "mode": arm, "checkpoint": name, **receipt})
    if snapshots["terminal"]["packet_chain_sha256"] != terminal_chain:
        raise ValueError(f"Captured terminal mesh does not match the actual charged terminal: {directory}")
    terminal = close_record(snapshots["terminal"], summary, trace, initial)
    if (terminal["returned"] != manifest["returned"] or terminal["terminal_pose"] != manifest["final_pose"]
            or terminal["termination"] != manifest["observed_native_termination"]):
        raise ValueError(f"Computed terminal outcome differs from run manifest: {directory}")
    arrivals = enumerate_arrivals(choices, trace)
    alternate_counts = [
        sum(r["admitted"] and r["route_id"] != choice["original_route_id"] for r in choice["candidates"])
        for choice in choices
    ]
    score_shifts = [max(
        (abs(r["scores"]["S"] - r["scores"]["G"]) for r in choice["candidates"] if r["admitted"]),
        default=0.0,
    ) for choice in choices]
    diagnostics = {}
    for choice in choices:
        for reason, count in choice["proposal_diagnostics"]["exclusions"].items():
            diagnostics[reason] = diagnostics.get(reason, 0) + count
    row = {"stage": stage, "layout": layout, "mode": arm, "status": manifest["status"],
           "run_kind": "native_autonomous_simulator", "source_manifest_sha256": digest,
           "native_budget": cap, "candidate_arrivals": arrivals, "local_route_choices": len(choices),
           "legal_alternative_decisions": sum(count > 0 for count in alternate_counts),
           "legal_alternate_options": sum(alternate_counts), "proposal_exclusion_counts": diagnostics,
           "decisions_with_effective_score_shift": sum(shift > 1e-9 for shift in score_shifts),
           "maximum_absolute_score_shift_m2": max(score_shifts, default=0.0),
           "different_hypothetical_winners": sum(
               choice["hypothetical_selected_route_id"]["G"] != choice["hypothetical_selected_route_id"]["S"]
               for choice in choices
           ),
           "different_from_native_choices": sum(
               choice["selected_route_id"] != choice["original_route_id"] for choice in choices
           ),
           "arrived": any(r["reached_under_same_option"] for r in arrivals) if choices else None,
           "first_selected_target_reached_paid": arrivals[0]["first_reached_paid"] if arrivals else None,
           **{key: snapshots["terminal"][key] for key in TERMINAL_METRICS}, **terminal}
    metric_trace = engine.per_action_curve(
        {"belief_trace": str(directory / "observed_belief_trace.npz")}, trace, rows, ref,
    )
    reachable = snapshots["terminal"]["reachable_cells"]
    for threshold in (0.95, 1.0):
        row[f"first_C_free_{round(threshold * 100)}_paid"] = next(
            (paid for paid, free, *_ in metric_trace if free / reachable >= threshold), None,
        )
    return row, rows, {"mode": arm, "trace": trace, "start": snapshots["start"], "terminal": snapshots["terminal"],
                       "choices": choices, "effective_runtime": attempt["effective_runtime"]}


def write_tables(output, name, rows, metadata, fields=None):
    if fields is None:
        fields = list(dict.fromkeys(key for row in rows for key in row))
    files.write_csv(output / f"{name}.csv", [{key: row.get(key) for key in fields} for row in rows], fields)
    files.write_json(output / f"{name}.json", metadata | {"rows": rows})


def holdout_reference(layout, frozen):
    base = STUDY / "holdout"
    reference = base / "evaluator_references" / layout
    paths = {
        "receipt": reference / "RECEIPT.json", "fixed": reference / "fixed_reference.npz",
        "mesh": reference / "source_mesh.ply", "initial": base / "prepared" / layout / "initial_state.json",
        "reachable": reference / "evaluator_truth.npz",
    }
    bindings = {key: files.verify_bound(path, frozen["files_sha256"]) for key, path in paths.items()}
    receipt = files.read_json(paths["receipt"])
    if (receipt["reference_npz_sha256"] != bindings["fixed"]
            or receipt["source_mesh_sha256"] != bindings["mesh"]):
        raise ValueError(f"Held-out geometry receipts disagree: {layout}")
    with engine.load(paths["reachable"], allow_pickle=False) as truth:
        reachable = truth["reachable"]
    samples, weights = {}, {}
    with engine.load(paths["fixed"], allow_pickle=False) as fixed:
        for name, key in {"scene": "all", **files.GROUPS}.items():
            samples[name] = fixed[key]
            weights[name] = receipt["metadata"][key]["area_per_sample_m2"]
            if samples[name].shape != (receipt["metadata"][key]["visible_samples"], 3):
                raise ValueError(f"Held-out fixed surface denominator changed: {layout} {key}")
    initial = files.read_json(paths["initial"])
    if list(reachable.shape) != initial["shape"] or int(reachable.sum()) != receipt["reachable_cells"]:
        raise ValueError(f"Held-out free-cell denominator changed: {layout}")
    return {
        "initial": initial, "reachable": reachable, "source_mesh": files.read_mesh(paths["mesh"]),
        "samples": samples, "weights": weights,
        "files_sha256": {str(path): bindings[key] for key, path in paths.items()},
    }


def development_gate(rows, pairs, cap, metadata, metrics_path, pairs_path):
    complete = len(rows) == 8 and all(row["status"] == "completed_execution" for row in rows)
    all_safe = complete and all(
        row["returned"] and not row["failed"] and row["actual_paid"] <= cap and row["collisions"] == 0 for row in rows
    )
    has_alternates = any(row.get("legal_alternative_decisions", 0) > 0 for row in rows)
    actual_category_effect = any(row["status"] == "different_category_choice_executed" for row in pairs)
    return {
        "stage": "development", "advance_holdout": all_safe and has_alternates and actual_category_effect,
        "all_eight_complete": complete, "all_eight_native_safe_return": all_safe,
        "admitted_non_native_route": has_alternates, "category_caused_executed_difference": actual_category_effect,
        "causal_witnesses": [{
            "layout": row["layout"], "causal_paid_action": row["causal_paid_action"],
            "paired_events": [event for event in row["paired_events"]
                              if event["status"] == "different_category_choice_executed"],
        } for row in pairs if row["status"] == "different_category_choice_executed"],
        "runner_freeze_sha256": metadata["runner_freeze_sha256"],
        "protocol_sha256": metadata["protocol_sha256"], "episodes_sha256": files.sha256(metrics_path),
        "actual_choice_audit_sha256": files.sha256(pairs_path),
    }


def holdout_contrasts(rows, cap, threshold=.001):
    by_layout = {}
    for row in rows:
        by_layout.setdefault(row["layout"], {})[row["mode"]] = row
    comparisons = []
    for layout, modes in by_layout.items():
        semantic = modes["newS"]
        for comparison in ("newG", "originalS", "F_strict"):
            baseline = modes[comparison]
            record = {"layout": layout, "comparison": f"newS - {comparison}"}
            verified = all(c["status"] == "completed_execution" for c in (semantic, baseline))
            if not verified:
                comparisons.append(record | {"status": "unavailable"})
                continue
            differences = {
                f"delta_{key}": semantic[key] - baseline[key] for key in (
                    "J", "C_free", "F1_scene_05cm", "actual_paid", "collisions",
                )
            }
            comparisons.append(record | differences | {
                "status": "measured", "material_J_gain": differences["delta_J"] > threshold,
                "returned": semantic["returned"], "comparator_returned": baseline["returned"],
                "within_budget_both": semantic["actual_paid"] <= cap and baseline["actual_paid"] <= cap,
                "return_or_collision_deteriorated": (
                    (baseline["returned"] and not semantic["returned"]) or differences["delta_collisions"] > 0
                ),
            })
    primary = [row for row in comparisons if row["comparison"] == "newS - newG"]
    gains = sum(bool(row.get("material_J_gain")) for row in primary)
    safe = all(row["status"] == "measured" and not row["return_or_collision_deteriorated"]
               and row["within_budget_both"] for row in primary)
    return comparisons, {
        "layouts": len(primary), "primary_material_wins": gains, "all_safe_against_primary_comparator": safe,
        "paper_engineering_gate": len(primary) == 4 and gains >= 3 and safe,
    }


def main(args):
    stage = args.stage
    protocol_path = STUDY / "PROTOCOL.json"
    freeze_path = STUDY / f"runner/{stage}_freeze.json"
    protocol = files.read_json(protocol_path)
    runner_freeze = files.read_json(freeze_path)
    group = protocol["development" if stage == "development" else "heldout"]
    cap = group["paid_action_budget"]
    expected_schedule = [(layout, arm) for layout in group["layouts"] for arm in group["modes"]]
    if (not protocol["declared_before_developer_policy_runs"]
            or runner_freeze["status"] != "frozen_before_any_new_paid_episode"
            or [(job["layout"], job["arm"]) for job in runner_freeze["schedule"]] != expected_schedule
            or protocol["source_local"] != str(PERSISTENCE)):
        raise ValueError("Fixed actor schedule and declared experiment do not match")
    for path, digest in runner_freeze["files_sha256"].items():
        files.verify(path, digest)
    run_binding = {"protocol_sha256": files.sha256(protocol_path), "freeze_sha256": files.sha256(freeze_path)}
    old_scene = files.read_json(files.SCENES / "freeze.json")
    old_runner = files.read_json(files.HISTORY / "freeze.json")
    old_eval = files.read_json(files.OLD_EVAL / "freeze.json")
    native_metric = files.load_metric(old_scene, old_eval)
    files.verify_bound(files.NATIVE_EXECUTION_FREEZE, old_runner["files_sha256"])
    recon_sha256 = files.verify_bound(
        files.RECON_METRIC, files.read_json(files.NATIVE_EXECUTION_FREEZE)["files_sha256"],
    )
    geometry_freeze = files.read_json(STUDY / "holdout/evaluator_freeze.json") if stage == "holdout" else None
    route_protocol = protocol["local_route"]
    route_config = (route_protocol["max_extra_outbound_actions"], route_protocol["min_unknown_union_ratio"])
    rows, checkpoints, pairs = [], [], []
    metadata = {"stage": stage, "protocol_sha256": run_binding["protocol_sha256"],
                "runner_freeze_sha256": run_binding["freeze_sha256"], "evaluator_sha256": files.sha256(__file__),
                "native_reconstruction_metric_sha256": recon_sha256, "layouts": {}}
    for layout in group["layouts"]:
        reference = files.load_reference(layout, old_scene, old_runner) if stage == "development" else (
            holdout_reference(layout, geometry_freeze)
        )
        metadata["layouts"][layout] = {"reference_inputs_sha256": reference["files_sha256"],
                                       "reachable_cells": int(reference["reachable"].sum())}
        raw = {}
        for arm in group["modes"]:
            run = STUDY / "runner/runs" / stage / layout / arm
            row, snapshots, source = evaluate_run(
                run, stage, layout, arm, cap, reference, native_metric, run_binding, route_config,
            )
            rows.append(row)
            checkpoints.extend(snapshots)
            raw[arm] = source
        if raw["newG"] and raw["newS"]:
            pairs.append(verify_category_pair(layout, raw["newG"], raw["newS"], cap, *route_config))
        else:
            pairs.append({"layout": layout, "status": "failed_run", "causal_paid_action": None, "paired_events": []})
    output = args.out or STUDY / "measurement/results" / stage
    output.mkdir(parents=True, exist_ok=True)
    write_tables(output, "episodes", rows, metadata, (
        "stage", "layout", "mode", "status", "actual_paid", "native_budget", "returned", "collisions", "failed",
        "stop_reason", "path_length_m", "legal_alternative_decisions", "local_route_choices", "arrived",
        "legal_alternate_options", "different_hypothetical_winners", "different_from_native_choices",
        "decisions_with_effective_score_shift", "maximum_absolute_score_shift_m2",
        "first_selected_target_reached_paid", "first_C_free_95_paid", "first_C_free_100_paid", *TERMINAL_METRICS,
        "source_manifest_sha256",
    ))
    write_tables(output, "checkpoints", checkpoints, metadata, (
        "stage", "layout", "mode", "checkpoint", "status", "actual_paid", "requested_paid", *TERMINAL_METRICS,
        "source_checkpoint_sha256",
    ))
    write_tables(output, "choice_audit", pairs, metadata, (
        "layout", "neutral_mode", "semantic_mode", "status", "common_start", "causal_paid_action",
        "same_effective_runtime", "identical_charged_action_and_packet_trace", "identical_terminal_mesh",
        "first_observed_difference", "paired_events",
    ))
    if stage == "development":
        decision = development_gate(rows, pairs, cap, metadata, output / "episodes.json", output / "choice_audit.json")
        decision_path = STUDY / "measurement/development_decision.json" if args.out is None else output / "decision.json"
        files.write_json(decision_path, decision)
        print(json.dumps({key: decision[key] for key in (
            "advance_holdout", "all_eight_native_safe_return", "admitted_non_native_route",
            "category_caused_executed_difference",
        )}, ensure_ascii=False))
    else:
        contrasts, gate = holdout_contrasts(rows, cap)
        write_tables(output, "contrasts", contrasts, metadata | {"engineering_gate": gate})


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("development", "holdout"))
    parser.add_argument("--out", type=Path)
    main(parser.parse_args())
