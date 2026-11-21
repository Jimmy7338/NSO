"""Verify the two frozen witness mechanisms in archived records without running an actor."""

import argparse
from dataclasses import asdict
from functools import lru_cache
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

ANALYSES = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent
SOURCE_STAGE = ANALYSES / "same_goal_route_choice_20261121/runner/runs/development"
POOL_STUDY = ANALYSES / "same_goal_pool_audit_20261121"
NATIVE_SOURCE = ANALYSES / (
    "reproduction_20261104/simulation_publication_stage/experiments/unknown_exploration_20261104/source"
)
POOL_SOURCE_STATE = {
    "gallery_left:56:230": "gallery_left/states/01_paid_056.json",
    "warehouse_right:142:578": "warehouse_right/states/04_paid_142.json",
}
SOURCE_KEYS = (
    "step", "action", "position", "heading", "collision", "done", "packet_sha256",
    "nonsemantic_packet_sha256", "owner_sha256", "packet_chain_sha256",
    "active_option_id", "selection_call_id", "route_action_index_zero_based",
    "local_route_id", "active_outbound_sha256", "selected_target_pose",
)


def require(condition, message):
    if not condition:
        raise ValueError(message)


def load(path):
    return json.loads(path.read_text())


def sha(path):
    with path.open("rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()


def array_sha(value):
    return hashlib.sha256(np.ascontiguousarray(value).tobytes()).hexdigest()


def trace_pose(step):
    return [*step["position"], step["heading"]]


def written(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


@lru_cache(maxsize=1)
def known_safety_guard():
    sys.path.insert(0, str(NATIVE_SOURCE))
    from nso.execution_guard_v8 import ObservedExecutionGuard

    return ObservedExecutionGuard(0.2, 0.2)


def old_run(layout, arm, inputs):
    root = SOURCE_STAGE / layout / arm
    manifest_path = root / "manifest.json"
    inputs[str(manifest_path)] = sha(manifest_path)
    manifest = load(manifest_path)
    require(manifest["status"] == "completed_execution" and manifest["returned"]
            and manifest["collisions"] == 0 and manifest["paid_actions"] == 256,
            f"Archival source run is not a complete sealed reference: {root}")
    hashes = manifest["files_sha256"]
    files = {}
    for name in ("steps.json", "local_choices.json", "global_choices.json", "audit.json", "observed_belief_trace.npz"):
        path = root / name
        check = sha(path)
        require(check == hashes[name], f"Archived source data changed: {path}")
        inputs[str(path)] = check
        files[name] = path if path.suffix == ".npz" else load(path)
    return files


def one_between(rows, predicate, name):
    matched = [row for row in rows if predicate(row)]
    require(len(matched) == 1, f"Expected exactly one {name}")
    return matched[0]


def audit_source_witness(state_id, inputs):
    paths = POOL_SOURCE_STATE[state_id]
    estimated = POOL_STUDY / "replay/runs" / paths
    manifest_file = estimated.parent.parent / "manifest.json"
    evidence = load(POOL_STUDY / "completion_audit.json")
    inputs[str(POOL_STUDY / "completion_audit.json")] = sha(POOL_STUDY / "completion_audit.json")
    manifest = load(manifest_file)
    require(manifest["files_sha256"][f"states/{estimated.name}"] == sha(estimated), "Expanded-state archived hash mismatch")
    inputs.update({str(estimated): sha(estimated), str(manifest_file): sha(manifest_file)})
    committed = load(estimated)
    require(state_id == committed["state_id"]
            and committed["counterfactual_actions_executed"] == 0
            and committed["used_for_action_selection"] is False
            and evidence["status"] == "verified",
            f"Expected only a sealed offline-scored prior witness: {state_id}")
    layout = committed["layout"]
    paid = committed["paid_before"]
    caller = committed["global_call_id"]
    g, s = (old_run(layout, arm, inputs) for arm in ("newG", "newS"))
    gs, ss = (row["steps.json"] for row in (g, s))
    require(len(gs) == len(ss) == 256 and all(
        all(a[key] == b[key] for key in SOURCE_KEYS) for a, b in zip(gs, ss)
    ), "Development arms did not share the expected source trajectory")
    opt = one_between(g["local_choices.json"], lambda row: row["selection_call_id"] == caller, "archived selected option")
    same_opt = one_between(s["local_choices.json"], lambda row: row["selection_call_id"] == caller, "S archived selected option")
    require(opt["paid_before"] == paid and opt["packet_sha256"] == committed["packet_sha256"]
            and opt["observed_belief_sha256"] == committed["belief_sha256"]
            and opt["global_native_sha256"] == committed["global_native_sha256"],
            f"Option or input digest mismatch at {state_id}")
    require(all(opt[key] == same_opt[key] for key in ("same_global_target", "native_return_states",
                "native_return_actions", "selected_outbound_actions", "selected_outbound_states")),
            "Source local arms differ at this witness")
    pre, authorized = (gs[k - 1] for k in (paid, paid + 1))
    require(trace_pose(pre) == committed["actual_pose"] and opt["pose"] == trace_pose(pre)
            and authorized["selection_call_id"] == caller,
            f"Source action transition does not use expected current witness: {state_id}")
    frame = f"frame-{paid}"
    native_events = [row for row in g["audit.json"] if row.get("frame_id") == frame and row["event"] != "observation"]
    event_classes = [row["event"] for row in native_events]
    global_choice = one_between(native_events,
        lambda row: row["event"] == "global_choice" and row["option"]["selection_call_id"] == caller,
        "source native global action-boundary choice")
    paid_authorization = one_between(native_events,
        lambda row: row["event"] == "paid_action_authorized" and row["selection_call_id"] == caller,
        "source paid action authorization")
    require(global_choice["option"]["pose"] == committed["expanded"]["summary"]["native_goal_pose"]
            and paid_authorization["assessment"]["action"] == authorized["action"]
            and paid_authorization["assessment"]["next_pose"] == trace_pose(authorized),
            "Source authorization does not match archive's goal and actual next action")
    with np.load(g["observed_belief_trace.npz"], allow_pickle=False) as bundle:
        require(list(bundle["paid"]) == list(range(257)), "Too few archived measured belief states")
        belief = bundle["belief"][paid]
    require(array_sha(belief) == opt["observed_belief_sha256"], "Archived belief trace mismatch at witness")
    guard = known_safety_guard()
    trigger = {"condition": "previous_committed_goal_arrival", "previous_target_arrived": pre["matched_target_pose_after_step"]}
    if layout == "warehouse_right":
        rejection = one_between(native_events, lambda row: row["event"] == "exploration_joint_replan_after_rejection",
                                "same-frame rejection-driven replan")
        previous_opt = one_between(g["local_choices.json"],
            lambda row: row["option_id"] == rejection["rejected_option_id"], "previous active option before rejection")
        index = pre["route_action_index_zero_based"] + 1
        old_action = previous_opt["selected_outbound_actions"][index]
        expected_pose = previous_opt["selected_outbound_states"][index + 1]
        check = guard.assess(belief, tuple(pre["position"]), pre["heading"], old_action, opt["remaining_budget"])
        require(not pre["matched_target_pose_after_step"] and rejection["selected_option_id"] == opt["option_id"]
                and pre["active_option_id"] == previous_opt["option_id"] and event_classes == [
                    "global_choice", "exploration_joint_replan_after_rejection", "paid_action_authorized",
                ]
                and not check.allowed and check.reason == rejection["reason"] == "next_footprint_not_known_safe"
                and list(check.target) == expected_pose[:2],
                "Failed to recover original observed-map rejection and same-call replan")
        trigger = {
            "condition": "native_guard_rejection_of_previous_option",
            "reason": check.reason,
            "declined_native_action": old_action,
            "declined_native_endpoint": expected_pose,
            "declined_center_belief": int(belief[check.target]),
            "declined_center_footprint_safe": bool(guard.safe_grid(belief)[check.target]),
            "declined_option_id": previous_opt["option_id"],
            "declined_selection_call": previous_opt["selection_call_id"],
            "first_replacement_call": caller,
            "call_event_order": event_classes,
            "note": "Native runtime appends rejection event after choose_goal but guard decision precedes choose_goal.",
        }
    else:
        require(pre["matched_target_pose_after_step"] and event_classes == ["global_choice", "paid_action_authorized"],
                "Expected ordinary new option after archived earlier goal arrival")
    rounds = {}
    for mode in ("G", "S"):
        winner = committed["expanded"]["summary"]["expanded_hypothetical_winners"][mode]
        candidate = one_between(committed["expanded"]["candidates"], lambda row: row["route_id"] == winner,
                               f"{mode} archived expanded winner")
        initial_action = candidate["outbound_actions"][0]
        checked = guard.assess(belief, tuple(pre["position"]), pre["heading"], initial_action, opt["remaining_budget"])
        _, _, heading = candidate["outbound_states"][1]
        returning = guard.return_plan(belief, checked.target, heading, tuple(committed["expanded"]["summary"]["native_return_anchor"]),
                                      opt["remaining_budget"] - 1)
        require(candidate["admitted"] and checked.allowed and returning.available
                and candidate["outbound_cost"] + candidate["reserved_return_cost"] <= opt["remaining_budget"],
                f"Expanded source route does not fit initial observed guard/cost: {state_id}/{mode}")
        rounds[mode] = {
            "archived_winner_id": winner,
            "archived_outbound_sha256": candidate["outbound_sha256"],
            "original_goal_pose": candidate["goal_pose"],
            "outbound_cost": candidate["outbound_cost"], "reserved_return_cost": candidate["reserved_return_cost"],
            "candidate_initial_action": initial_action,
            "first_command_frozen_guard": asdict(checked), "first_successor_return_frozen_cost": returning.paid_cost,
        }
    return {
        "state_id": state_id, "layout": layout, "paid_before": paid, "native_global_call_id": caller,
        "observed_pose": trace_pose(pre), "packet_sha256": opt["packet_sha256"],
        "belief_sha256": opt["observed_belief_sha256"], "frozen_budget_remaining": opt["remaining_budget"],
        "original_global_goal_pose": opt["same_global_target"], "original_native_outbound_cost": len(opt["native_outbound_actions"]),
        "original_archived_action_at_next_paid_index": authorized["action"], "original_trigger": trigger,
        "archived_expanded_winners_on_original_observed_state": rounds,
        "archived_first_hypothetical_difference": committed["expanded"]["comparisons"]["expanded_G_vs_S"]["first_hypothetical_action_difference"],
    }


def main(out):
    inputs = {}
    protocol_path = OUT.parent / "PROTOCOL.json"
    protocol = load(protocol_path)
    archived_receipt = load(POOL_STUDY / "publication/receipt.json")
    require(protocol["frozen_before_any_new_trajectory"] and protocol["source_commit"] == archived_receipt["head"]
            and Path(protocol["source_local"]) == POOL_STUDY and len(protocol["pairs"]) == len(POOL_SOURCE_STATE),
            "Two-witness protocol does not name the verified archive")
    for path in (Path(__file__), protocol_path, NATIVE_SOURCE / "nso/execution_guard_v8.py", NATIVE_SOURCE / "nso_bridge/exploration_joint_policy.py",
                 ANALYSES / "semantic_scene_redesign_20261119/scene_adapter/replicacad_world.py",
                 ANALYSES / "semantic_large_scene_20261121/mechanism/runner/protocol.json",
                 POOL_STUDY / "publication/receipt.json"):
        inputs[str(path)] = sha(path)
    rows = [audit_source_witness(state_id, inputs) for state_id in POOL_SOURCE_STATE]
    for row in rows:
        pair = protocol["pairs"][row["layout"]]
        split = row["archived_first_hypothetical_difference"]
        winners = row["archived_expanded_winners_on_original_observed_state"]
        require(pair["state_id"] == row["state_id"] and pair["paid_before"] == row["paid_before"]
                and pair["global_call_id"] == row["native_global_call_id"]
                and pair["expected_first_nominal_split"] == {"paid": split["hypothetical_paid_action_id"],
                                                           "G": split["first_action"], "S": split["second_action"]}
                and all(winners[arm]["archived_winner_id"] == pair["archived_expected_winners_for_verification_only"][arm]
                        and winners[arm]["outbound_cost"] == pair["archived_nominal_outbound_for_verification_only"][arm]
                        and winners[arm]["reserved_return_cost"] == pair["native_reserved_return"]
                        for arm in ("G", "S")),
                f"Predeclared witness differs from verified source: {row['state_id']}")
    out.mkdir(parents=True, exist_ok=True)
    written(out / "source_mechanism.json", {
        "status": "archived_source_witness_mechanisms_validated_without_execution",
        "source_commit": protocol["source_commit"],
        "actor_started": False, "states": rows,
        "simulation_rng": "Development source defaults depth_sigma_m=dropout=0. Nonzero depth noise is per-frame SeedSequence(seed,step_count,739); archived exact packet SHA is the operational replay check. Initial camera render and laser are deterministic at the same pose.",
        "source_inputs_sha256": dict(sorted(inputs.items())),
        "scope": "Old issued actions and current observed-map guard are verified; post-branch measurements and future local safety require actual paired experiments.",
    })


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=OUT)
    main(parser.parse_args().output)
