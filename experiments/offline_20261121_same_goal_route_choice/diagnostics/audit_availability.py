"""Read sealed same-goal traces and localize alternative/decision attrition."""

import argparse
from collections import Counter, defaultdict
import csv
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ARMS = ("newG", "newS")
NATIVE = "native"
AVAILABILITY_REASONS = (
    "no_modeled_measured_object_support",
    "no_within_bound_oriented_waypoint",
    "no_additional_estimated_object_azimuth",
    "novel_waypoints_only_duplicate_or_early_goal",
    "novel_route_failed_unknown_union_floor",
    "alternate_admitted",
)


def load(path):
    return json.loads(path.read_text())


def sha(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def canonical(value):
    packed = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(packed.encode()).hexdigest()


def require(condition, message):
    if not condition:
        raise ValueError(message)


def write_csv(path, data):
    columns = list(dict.fromkeys(key for row in data for key in row))
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, columns, lineterminator="\n")
        writer.writeheader()
        writer.writerows(data)


def action_divergence(a, b):
    return next((offset for offset, (x, y) in enumerate(zip(a, b)) if x != y), None)


def attribute(audit, admitted, causes):
    if len(admitted) > 1:
        return "alternate_admitted"
    if not audit["measured_supported_tracks"]:
        return "no_modeled_measured_object_support"
    if not causes["within_cost_waypoint_states"]:
        return "no_within_bound_oriented_waypoint"
    if not causes["estimated_distinct_view_waypoints"]:
        return "no_additional_estimated_object_azimuth"
    if not causes["distinct_alternate_traces"]:
        return "novel_waypoints_only_duplicate_or_early_goal"
    require(causes["exclusions"].get("observable_unknown_union_below_native_floor", 0) > 0,
            "Expected coverage rejection for distinct novel traces without admissions")
    return "novel_route_failed_unknown_union_floor"


def read_run(run, freeze_hash, protocol_hash, args):
    paths = dict()
    manifest_path = run / "manifest.json"
    if not manifest_path.exists():
        failure = run / "failure.json"
        if failure.exists():
            return {"status": "runner_failure", "failure": load(failure)["error"]}, [], [], {str(failure): sha(failure)}
        raise ValueError(f"Run lacks verified sealed manifest or failure record: {run}")
    manifest = load(manifest_path)
    paths[str(manifest_path)] = sha(manifest_path)
    require(manifest["status"] == "completed_execution" and manifest["actual_closed_loop"] is True
            and manifest["forced_route_or_goal"] is False and manifest["evaluator_inputs_read_by_actor"] is False
            and manifest["local_route_only"] and manifest["global_mode"] == "G",
            f"Unverified policy contract: {run}")
    require(manifest["stage"] == args.stage, f"Unexpected stage for {run}")
    records = {}
    for name in ("local_choices.json", "global_choices.json", "steps.json", "attempt.json"):
        p = run / name
        check = sha(p)
        require(check == manifest["files_sha256"][name], f"Provenance mismatch: {p}")
        paths[str(p)] = check
        records[name] = load(p)
    attempt = records["attempt.json"]
    require(attempt["runner_freeze_sha256"] == freeze_hash and attempt["source_protocol_sha256"] == protocol_hash,
            f"Development attempt is not tied to the requested freeze: {run}")
    options, steps = records["local_choices.json"], records["steps.json"]
    require(manifest["paid_actions"] == len(steps) and manifest["local_route_choices"] == len(options)
            and [step["step"] for step in steps] == list(range(1, len(steps) + 1)),
            f"Wrong local choices or action sequence length in {run}")
    global_map = {choice["call_id"]: choice for choice in records["global_choices.json"]}
    return manifest, options, steps, paths | {"__global_map": global_map}


def verify_option(meta, event, charged, global_event, settings):
    audit = event["choice_audit"]
    diag = audit["proposal_diagnostics"]
    require(event["option_id"] == audit["native_option_id"] and event["packet_sha256"] == audit["packet_sha256"]
            and event["observed_belief_sha256"] == audit["belief_sha256"] and event["paid_before"] == audit["paid_before"]
            and event["same_global_target"] == audit["native_goal_pose"]
            and event["local_mode"] == audit["selected_mode"] and event["global_mode"] == "G"
            and global_event["outputs"]["selected"]["pose"] == event["same_global_target"]
            and global_event["outputs"]["selected"]["outbound_actions"] == event["native_outbound_actions"]
            and global_event["outputs"]["selected"]["outbound_states"] == event["native_outbound_states"]
            and global_event["outputs"]["mode"] == "G"
            and event["native_return_states"] == event["rebound_return_states"]
            and event["native_return_actions"] == event["rebound_return_actions"],
            f"Local/global or return reservation mismatch: {meta} / {event['option_id']}")
    candidates = {row["route_id"]: row for row in audit["candidates"]}
    admitted = {cid: data for cid, data in candidates.items() if data["admitted"]}
    require(len(candidates) == audit["assessed_candidate_count"]
            and len(admitted) == audit["admitted_candidate_count"] and NATIVE in admitted
            and len(admitted) <= 1 + settings["max_alternates"],
            f"Candidate admission counts differ from freeze: {meta} / {event['option_id']}")
    pool_digest = canonical([{key: value for key, value in row.items() if key != "scores"} for row in audit["candidates"]])
    require(pool_digest == audit["candidate_pool_sha256"]
            and canonical([row["scores"] for row in audit["candidates"]]) == audit["score_table_sha256"],
            f"Candidate table mismatch: {meta} / {event['option_id']}")
    native = admitted[NATIVE]
    floor = native["unknown_union_cells"] * settings["min_unknown_union_ratio"]
    budget = event["remaining_budget"]
    for route_id, row in candidates.items():
        require(row["outbound_sha256"] == canonical({k: row[k] for k in ("outbound_states", "outbound_actions")})
                and row["goal_pose"] == row["outbound_states"][-1] == event["same_global_target"]
                and row["outbound_states"][0] == event["pose"]
                and row["reserved_return_cost"] == len(event["native_return_actions"])
                and row["outbound_cost"] == len(row["outbound_actions"])
                and row["outbound_cost"] + row["reserved_return_cost"] <= budget,
                f"Invalid sealed candidate hash/endpoints/cost: {meta} / {event['option_id']} / {route_id}")
        if route_id != NATIVE:
            require(row["outbound_cost"] - native["outbound_cost"] <= settings["max_extra_outbound_actions"]
                    and row["novel_view_tags"]
                    and bool(row["unknown_union_cells"] + 1e-12 >= floor) == row["admitted"],
                    f"Invalid alternative vs predeclared cost/view/unknown rules: {meta} / {event['option_id']} / {route_id}")
    require(audit["score_equality_tolerance"] == settings["score_equality_tolerance"],
            f"Unknown score threshold in sealed local audit: {meta} / {event['option_id']}")
    for mode, cid in audit["hypothetical_selected_route_id"].items():
        top = max(row["scores"][mode] for row in admitted.values())
        tie_pool = [row for row in admitted.values() if top - row["scores"][mode] <= settings["score_equality_tolerance"]]
        optimal = min(tie_pool, key=lambda route: (route["outbound_cost"], route["route_id"] != NATIVE, route["outbound_sha256"]))
        require(optimal["route_id"] == cid, f"Reported {mode} choice disagrees with frozen ranking")
    selected = admitted[audit["selected_route_id"]]
    require(audit["selected_route_id"] == audit["hypothetical_selected_route_id"][event["local_mode"]]
            and selected["outbound_sha256"] == audit["chosen_outbound_sha256"] == event["active_outbound_sha256"]
            and native["outbound_sha256"] == audit["native_outbound_sha256"]
            and selected["outbound_actions"] == event["selected_outbound_actions"]
            and selected["outbound_states"] == event["selected_outbound_states"],
            f"Selected local execution copy does not match audit: {meta} / {event['option_id']}")
    for index, step in enumerate(charged):
        require(step["route_action_index_zero_based"] == index and step["guard_allowed"]
                and step["active_outbound_sha256"] == selected["outbound_sha256"]
                and step["action"] == selected["outbound_actions"][index]
                and step["selected_target_pose"] == event["same_global_target"],
                f"Actual authorization/action did not follow selected route: {meta} / {event['option_id']}")
        if not step["collision"]:
            require([*step["position"], step["heading"]] == selected["outbound_states"][index + 1],
                    f"Actual pose diverges from authorized route: {meta} / {step['step']}")
    g, s = (admitted[audit["hypothetical_selected_route_id"][mode]] for mode in ("G", "S"))
    offset_vs_native = action_divergence(selected["outbound_actions"], native["outbound_actions"])
    offset_between_modes = action_divergence(g["outbound_actions"], s["outbound_actions"])
    category_flip = g["route_id"] != s["route_id"]
    adjustments = [row["scores"]["S"] - row["scores"]["G"] for row in admitted.values()]
    max_pair_margin_shift = max(adjustments) - min(adjustments)
    result = {
        **meta, "paid_before": event["paid_before"], "option_id": event["option_id"],
        "global_target_pose": json.dumps(event["same_global_target"]),
        "global_native_sha256": event["global_native_sha256"],
        "observed_belief_sha256": event["observed_belief_sha256"],
        "start_pose": json.dumps(event["pose"]),
        "remaining_budget": event["remaining_budget"],
        "native_outbound": native["outbound_cost"], "selected_outbound": selected["outbound_cost"],
        "reserved_return": native["reserved_return_cost"], "native_unknown_union": native["unknown_union_cells"],
        "selected_unknown_union": selected["unknown_union_cells"],
        "modeled_track_count": len(audit["measured_supported_tracks"]),
        "availability_reason": attribute(audit, admitted, diag), "within_cost_waypoint_states": diag["within_cost_waypoint_states"],
        "distinct_view_waypoints": diag["estimated_distinct_view_waypoints"], "distinct_alternate_traces": diag["distinct_alternate_traces"],
        "duplicates_skipped": diag["duplicate_traces_skipped"], "early_goal_skipped": diag["early_goal_traces_skipped"],
        "coverage_rejected": diag["exclusions"].get("observable_unknown_union_below_native_floor", 0),
        "unchecked_due_to_admitted_pool_limit": diag["exclusions"].get("remaining_candidates_after_admitted_pool_cap", 0),
        "alternates_admitted": len(admitted) - 1, "hypothetical_G": g["route_id"], "hypothetical_S": s["route_id"],
        "selected_route": selected["route_id"], "hypothetical_category_choice_flip": category_flip,
        "max_abs_category_adjustment_m2": max(map(abs, adjustments)),
        "max_abs_category_pair_margin_shift_m2": max_pair_margin_shift,
        "category_flip_first_different_planned_action_one_based": None if offset_between_modes is None else offset_between_modes + 1,
        "selected_differs_from_native": selected["route_id"] != NATIVE,
        "charged_option_actions": len(charged),
        "first_paid_change_from_native_trace": charged[offset_vs_native]["step"]
        if offset_vs_native is not None and len(charged) > offset_vs_native else None,
        "charged_past_hypothetical_G_S_differing_offset": offset_between_modes is not None and len(charged) > offset_between_modes,
        "completed_planned_target": any(step["matched_target_pose_after_step"] for step in charged),
        "packet_sha256": audit["packet_sha256"], "candidate_pool_sha256": audit["candidate_pool_sha256"],
        "native_outbound_sha256": audit["native_outbound_sha256"],
        "score_table_sha256": audit["score_table_sha256"],
    }
    return result


def compare_first_divergence(layout, paired):
    left, right = paired
    status = {"layout": layout, "newG_status": left[0]["status"], "newS_status": right[0]["status"]}
    if "runner_failure" in (left[0]["status"], right[0]["status"]):
        return status | {"first_actual_difference_verified": False, "reason": "incomplete_pair"}
    require(all(left[0][name] == right[0][name] for name in ("initial_packet_sha256", "initial_belief_sha256", "initial_pose")),
            f"Cannot compare pair with unmatched initialization: {layout}")
    step_lists = (left[2], right[2])
    offset = next((i for i, (a, b) in enumerate(zip(*step_lists))
                   if (a["action"], a["packet_sha256"]) != (b["action"], b["packet_sha256"])), None)
    if offset is None:
        return status | {"first_actual_difference_verified": False,
                         "reason": "same_recorded_action_and_observation_trace"
                         if len(left[2]) == len(right[2]) else "different_number_of_actions_without_shared_clock_divergence"}
    g, s = (steps[offset] for steps in step_lists)
    newg, news = (item[1].get(step["active_option_id"]) for item, step in zip(paired, (g,s)))
    detail = status | {"paid": g["step"], "actual_G": g["action"], "actual_S": s["action"],
                       "first_actual_difference_verified": False}
    if newg is None or news is None:
        return detail | {"reason": "active_local_option_missing"}
    common = all(newg[key] == news[key] for key in (
        "paid_before", "packet_sha256", "observed_belief_sha256", "global_target_pose", "global_native_sha256",
        "native_outbound_sha256", "remaining_budget", "candidate_pool_sha256", "score_table_sha256",
    ))
    g_switch, s_switch = (row["category_flip_first_different_planned_action_one_based"] for row in (newg, news))
    execution = (g["action"] != s["action"] and g_switch is not None and g_switch == s_switch
                 and g["route_action_index_zero_based"] == s["route_action_index_zero_based"] == g_switch - 1
                 and newg["selected_route"] == newg["hypothetical_G"] == news["hypothetical_G"]
                 and news["selected_route"] == newg["hypothetical_S"] == news["hypothetical_S"])
    return detail | {
        "first_actual_difference_verified": bool(common and execution),
        "same_observed_option_anchor_and_pool": common,
        "reason": "class_prior_switch_explains_first_charged_divergence" if common and execution else "unattributed_divergence",
        "global_target_pose": newg["global_target_pose"], "choice_paid_before": newg["paid_before"],
        "newG_local_route": newg["selected_route"], "newS_local_route": news["selected_route"],
    }


def run(args):
    protocol_path = ROOT / "PROTOCOL.json"
    settings = load(protocol_path)["local_route"]
    frozen_path = ROOT / "runner" / f"{args.stage}_freeze.json"
    freeze = load(frozen_path)
    protocol_hash, freeze_hash = sha(protocol_path), sha(frozen_path)
    sources = {str(p): sha(p) for p in (Path(__file__), protocol_path, frozen_path)}
    for path in (ROOT / "runner/full_run.py", ROOT / "route_choice.py"):
        require(sha(path) == freeze["files_sha256"][str(path)], "Frozen local integration source changed")
        sources[str(path)] = sha(path)
    per_decision, per_run, inputs = [], [], defaultdict(dict)
    for arm_record in freeze["schedule"]:
        layout, arm = (arm_record[key] for key in ("layout", "arm"))
        if arm not in ARMS:
            continue
        directory = ROOT / "runner/runs" / args.stage / layout / arm
        manifest, options, steps, files = read_run(directory, freeze_hash, protocol_hash, args)
        meta = {"layout": layout, "arm": arm, "status": manifest["status"]}
        counts = Counter()
        option_rows = {}
        global_table = files.pop("__global_map", {})
        if options:
            charged = defaultdict(list)
            for step in steps:
                charged[step["active_option_id"]].append(step)
            for option in options:
                row = verify_option(meta, option, charged[option["option_id"]], global_table[option["selection_call_id"]], settings)
                per_decision.append(row)
                option_rows[option["option_id"]] = row
                counts[row["availability_reason"]] += 1
                counts["causal_flip_in_same_recorded_state"] += row["hypothetical_category_choice_flip"]
                counts["admitted_and_category_changed_pair_margin"] += (
                    row["alternates_admitted"] > 0
                    and row["max_abs_category_pair_margin_shift_m2"] > settings["score_equality_tolerance"]
                )
                counts["selected_alternate"] += row["selected_differs_from_native"]
                counts["actual_paid_change_from_native"] += row["first_paid_change_from_native_trace"] is not None
                counts["flip_not_executed_to_differing_offset"] += (
                    row["hypothetical_category_choice_flip"] and not row["charged_past_hypothetical_G_S_differing_offset"]
                )
        sources.update(files)
        inputs[layout][arm] = (manifest, option_rows, steps)
        per_run.append({**meta, "paid_actions": manifest.get("paid_actions"),
                        "failure": manifest.get("failure"), "local_decisions": len(options),
                        "native_global_calls": len(global_table),
                        "native_global_calls_without_selected_option": sum(
                            event["outputs"]["selected"] is None for event in global_table.values()
                        ),
                        **{cause: counts[cause] for cause in AVAILABILITY_REASONS},
                        **{cause: counts[cause] for cause in (
                            "causal_flip_in_same_recorded_state", "admitted_and_category_changed_pair_margin",
                            "selected_alternate", "actual_paid_change_from_native", "flip_not_executed_to_differing_offset",
                        )}})
    pairs = [compare_first_divergence(layout, [arms[arm] for arm in ARMS]) for layout, arms in inputs.items()]
    args.output.mkdir(parents=True, exist_ok=True)
    pages = {
        f"{args.stage}_availability_decisions.csv": per_decision,
        f"{args.stage}_availability_by_arm.csv": per_run,
        f"{args.stage}_paired_first_actual_effect.csv": pairs,
    }
    for filename, values in pages.items():
        write_csv(args.output / filename, values)
    receipt = {
        "status": "sealed_availability_diagnostic_actor_not_invoked",
        "stage": args.stage,
        "experimental_unit": "paired parent layout; dependent mirrored development placements and per-run decisions are descriptive",
        "per_decision_count": len(per_decision), "ports_or_actor_started": False,
        "pairs_verified_category_first_actual_divergence": sum(row["first_actual_difference_verified"] for row in pairs),
        "sha256_inputs": dict(sorted(sources.items())),
        "sha256_outputs": {name: sha(args.output / name) for name in pages},
    }
    (args.output / f"{args.stage}_availability_receipt.json").write_text(json.dumps(receipt, indent=2) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("development", "holdout"), default="development")
    parser.add_argument("--output", type=Path, default=Path(__file__).resolve().parent)
    run(parser.parse_args())
