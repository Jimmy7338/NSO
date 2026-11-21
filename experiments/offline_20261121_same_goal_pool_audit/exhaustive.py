"""Enumerate the frozen same-goal generator beyond its original four-alternate cap."""

from functools import lru_cache
import hashlib
import importlib.util
import json
from pathlib import Path

import numpy as np

ARCHIVE = Path(__file__).resolve().parents[1] / "same_goal_route_choice_20261121"
SOURCE_NAME = "route_choice.py"
SOURCE_COMMIT = "02e9c5dcc0da7df61f23d8683fec2dd92c145c1e"
SOURCE_SHA256 = "ec3ccf9780283301cee6a412923e73896186fa0041f8537496ef423932b21fbb"
MODES = ("G", "S")
CAP_REASON = "remaining_candidates_after_admitted_pool_cap"
COVERAGE_REASON = "observable_unknown_union_below_native_floor"
FROZEN_SETTINGS = (
    "max_extra_outbound_actions", "max_alternates", "min_unknown_union_ratio", "azimuth_bins",
    "score_equality_tolerance",
)


def _canonical(value):
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    ).hexdigest()


def _array_sha(value):
    return hashlib.sha256(np.ascontiguousarray(value).tobytes()).hexdigest()


def _require(condition, message):
    if not condition:
        raise ValueError(message)


@lru_cache(maxsize=1)
def frozen_chooser():
    source = ARCHIVE / SOURCE_NAME
    publication = json.loads((ARCHIVE / "publication/receipt.json").read_text())
    freeze = json.loads((ARCHIVE / "runner/development_freeze.json").read_text())
    settings = json.loads((ARCHIVE / "PROTOCOL.json").read_text())["local_route"]
    _require(
        publication["head"] == SOURCE_COMMIT and publication["status"] == "verified_published"
        and hashlib.sha256(source.read_bytes()).hexdigest() == SOURCE_SHA256
        and freeze["files_sha256"][str(source)] == SOURCE_SHA256
        and settings["max_alternates"] == 4,
        "Source publication or original candidate-generator contract differs from the frozen archive",
    )
    spec = importlib.util.spec_from_file_location("frozen_published_four_alternate_chooser", source)
    core = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(core)
    return core, settings


def _ledger_identity(state):
    mapper = state["mapper"]
    return {
        "paid_before": int(state["packet"].action_id),
        "packet_sha256": state["packet"].sha256(),
        "belief_sha256": _array_sha(mapper.belief),
        "mapper_frames": int(mapper.frames),
        "remaining_budget": int(state["ledger"].remaining_budget),
        "native_return_anchor": list(state["return_anchor"]),
    }


def _posterior_identity(cache):
    models = cache["models"]
    _require(models["G"].bank is models["S"].bank, "G/S lack a shared fitted geometric bank")
    shared_fields = ("points", "hypothesis_ids", "history_keep")
    shared = {}
    for field in shared_fields:
        a, b = (_array_sha(getattr(models[mode], field)) for mode in MODES)
        _require(a == b, f"G/S geometric template mismatch for {field}")
        shared[field] = a
    return {
        "model_basis_sha256": shared,
        "per_mode_weights_sha256": {mode: _array_sha(models[mode].weights) for mode in MODES},
        "model_state_sha256": cache["state_sha256"],
    }


def _rank(core, candidates, mode, tolerance):
    remaining = [row for row in candidates if row["admitted"]]
    ranks = {}
    while remaining:
        chosen = core._choose_candidate(remaining, mode, tolerance)
        ranks[chosen["route_id"]] = len(ranks) + 1
        remaining = [row for row in remaining if row["route_id"] != chosen["route_id"]]
    return ranks


def _route_diff(a, b, paid_before):
    """Compare hypothetical command strings at their shared source, retaining both post-poses."""
    actions_a, actions_b = a["outbound_actions"], b["outbound_actions"]
    common = min(len(actions_a), len(actions_b))
    offset = next((i for i in range(common) if actions_a[i] != actions_b[i]), common)
    if offset == common and len(actions_a) == len(actions_b):
        _require(a["outbound_states"] == b["outbound_states"], "Same command trace produced differing states")
        return None
    _require(a["outbound_states"][:offset + 1] == b["outbound_states"][:offset + 1],
             "Hypothetical routes diverged in pose before the first command difference")
    return {
        "relative_action_one_based": offset + 1,
        "hypothetical_paid_action_id": paid_before + offset + 1,
        "shared_pre_pose": a["outbound_states"][offset],
        "first_action": actions_a[offset] if offset < len(actions_a) else None,
        "second_action": actions_b[offset] if offset < len(actions_b) else None,
        "first_predicted_post_pose": a["outbound_states"][offset + 1] if offset < len(actions_a) else None,
        "second_predicted_post_pose": b["outbound_states"][offset + 1] if offset < len(actions_b) else None,
        "executed": False,
    }


def _winner_row(rows, rank, mode):
    winner = next(row for row in rows if rank.get(row["route_id"]) == 1)
    runner_up = next((row for row in rows if rank.get(row["route_id"]) == 2), None)
    return winner, {
        "route_id": winner["route_id"],
        "outbound_sha256": winner["outbound_sha256"],
        "outbound_cost": winner["outbound_cost"],
        "scores_m2": winner["scores"].copy(),
        "runner_up_route_id": None if runner_up is None else runner_up["route_id"],
        "runner_up_score_m2": None if runner_up is None else runner_up["scores"][mode],
        "winner_minus_runner_up_m2": None if runner_up is None else winner["scores"][mode] - runner_up["scores"][mode],
    }


def _verify_native_graph_paths(core, state, native, rows, config):
    safe = state["guard"].safe_grid(state["mapper"].belief)
    return_path = [tuple(pose) for pose in native["return_states"]]
    reserve = native["return_cost"]
    native_start = tuple(native["outbound_states"][0])
    native_goal = tuple(native["pose"])
    budget = state["ledger"].remaining_budget

    def known_safe(pose):
        r, c, heading = pose
        return 0 <= r < safe.shape[0] and 0 <= c < safe.shape[1] and 0 <= heading < 4 and safe[r, c]

    _require(
        return_path[0] == native_goal and return_path[-1] == tuple(state["return_anchor"])
        and core._actions(return_path) == native["return_actions"]
        and reserve == len(native["return_actions"]) and all(map(known_safe, return_path)),
        "Original reserved return path fails current frozen safety graph or primitive validation",
    )
    for row in rows:
        path = [tuple(pose) for pose in row["outbound_states"]]
        cost = row["outbound_cost"]
        _require(
            path[0] == native_start and path[-1] == native_goal and native_goal not in path[:-1]
            and all(map(known_safe, path))
            and core._actions(path) == row["outbound_actions"]
            and cost == len(path) - 1 and row["reserved_return_cost"] == reserve
            and cost <= native["outbound_cost"] + config["max_extra_outbound_actions"]
            and cost + reserve <= budget,
            f"Expanded proposal failed source-map primitives, early-goal, reserve or cost checks: {row['route_id']}",
        )


def enlarge_pool(state, native, route_union_area, config, original_choice_audit):
    """Verify original archive at this acquired state, then score every proposed legal route.

    The method returns a provenance-checked diagnostic record. It deliberately
    discards both chosen option objects and never issues or changes an action.
    """
    core, frozen_config = frozen_chooser()
    _require(all(config[key] == frozen_config[key] for key in FROZEN_SETTINGS), "Requested settings differ from source")
    ledger = _ledger_identity(state)
    native_hash = _canonical(native)
    posterior = _posterior_identity(state["mapper"]._dual_template_state["cache"])
    capped, parity = {}, {}
    for mode in MODES:
        source = original_choice_audit[mode]
        archived = source.get("choice_audit", source)
        _require(
            ledger["packet_sha256"] == archived["packet_sha256"]
            and ledger["belief_sha256"] == archived["belief_sha256"]
            and ledger["paid_before"] == archived["paid_before"]
            and list(native["pose"]) == archived["native_goal_pose"]
            and native["option_id"] == archived["native_option_id"]
            and native["candidate_id"] == archived["native_global_candidate_id"]
            and mode == archived["selected_mode"],
            f"Replay did not restore the same observed packet, belief, and original global option for {mode}",
        )
        _, capped[mode] = core.choose_route(state, native, mode, route_union_area, config)
        differing = sorted(set(capped[mode]) ^ set(archived))
        differing += [key for key in sorted(set(capped[mode]) & set(archived))
                      if _canonical(capped[mode][key]) != _canonical(archived[key])]
        _require(not differing, f"Frozen four-alternate {mode} reproduction mismatch in fields: {', '.join(differing)}")
        parity[mode] = {
            "archived_full_audit_sha256": _canonical(archived),
            "recomputed_full_audit_sha256": _canonical(capped[mode]),
            "original_selected_route_id": capped[mode]["selected_route_id"],
        }
    original = capped["G"]
    _require(
        all(original[key] == capped["S"][key] for key in ("candidate_pool_sha256", "score_table_sha256",
                                                         "native_goal_pose", "hypothetical_selected_route_id")),
        "Archived G/S capped decisions lack a common pool, scoring table, and winner counterfactuals",
    )
    _require(
        original["admitted_candidate_count"] > 1
        and original["hypothetical_selected_route_id"]["G"] == original["hypothetical_selected_route_id"]["S"],
        "Scope is the frozen admitted-alternate common-state sample with shared original winner",
    )

    proposal_count = original["proposal_diagnostics"]["distinct_alternate_traces"]
    # One beyond the known proposal count guarantees that the identical loop
    # evaluates every proposal, even when every one satisfies the coverage floor.
    expanded_config = dict(config, max_alternates=max(config["max_alternates"], proposal_count + 1))
    _, expanded = core.choose_route(state, native, "G", route_union_area, expanded_config)
    records = expanded["candidates"]
    old_records = original["candidates"]
    _require(
        all(original["proposal_diagnostics"][key] == expanded["proposal_diagnostics"][key]
                for key in original["proposal_diagnostics"] if key != "exclusions")
        and expanded["proposal_diagnostics"]["exclusions"].get(CAP_REASON, 0) == 0
        and len(records) == proposal_count + 1
        and _canonical(records[:len(old_records)]) == _canonical(old_records)
        and len(records) - len(old_records) == original["proposal_diagnostics"]["exclusions"].get(CAP_REASON, 0),
        "Expanded generation did not reproduce the original trace prefix or exhaust its cap-skipped tail",
    )
    _require(
        len({row["route_id"] for row in records}) == len({row["outbound_sha256"] for row in records}) == len(records)
        and all(row["goal_pose"] == original["native_goal_pose"] for row in records),
        "Expanded trace identity/endpoint invariant violated",
    )
    _verify_native_graph_paths(core, state, native, records, config)
    _require(
        _canonical(native) == native_hash and _ledger_identity(state) == ledger
        and _posterior_identity(state["mapper"]._dual_template_state["cache"]) == posterior,
        "Offline scoring mutated native option, observed state, action budget, or frozen posterior",
    )
    _require(any(row["admitted"] for row in records), "Native route is missing from expanded score table")
    tolerance = config["score_equality_tolerance"]
    original_rank = {mode: _rank(core, old_records, mode, tolerance) for mode in MODES}
    expanded_rank = {mode: _rank(core, records, mode, tolerance) for mode in MODES}
    old_winners, new_winners, modes = {}, {}, {}
    for each in MODES:
        old, old_evidence = _winner_row(old_records, original_rank[each], each)
        new, new_evidence = _winner_row(records, expanded_rank[each], each)
        _require(
            original["hypothetical_selected_route_id"][each] == old["route_id"]
            and expanded["hypothetical_selected_route_id"][each] == new["route_id"],
            "Mode-specific independent rank disagrees with frozen winner",
        )
        diff = _route_diff(old, new, ledger["paid_before"])
        modes[each] = {
            "original": old_evidence, "expanded": new_evidence,
            "changed_route": diff is not None,
            "expanded_minus_original_score_m2": new["scores"][each] - old["scores"][each],
            "first_hypothetical_action_difference": diff,
        }
        old_winners[each], new_winners[each] = old, new
    g, s = (new_winners[arm] for arm in MODES)
    between_modes = _route_diff(g, s, ledger["paid_before"])
    old_ids = {row["route_id"] for row in old_records}
    tail = records[len(old_records):]
    admitted_tail = sum(row["admitted"] for row in tail)
    tail_excluded = sum(not row["admitted"] for row in tail)
    _require(tail_excluded == sum(row["exclusion_reason"] == COVERAGE_REASON for row in tail),
             "Unrecognized excluded proposal in unchanged generator")
    summary = {
        **ledger,
        "native_option_id": native["option_id"], "native_goal_pose": list(native["pose"]),
        "native_outbound_sha256": original["native_outbound_sha256"],
        "native_return_actions": list(native["return_actions"]), "native_return_states": native["return_states"],
        "native_reserved_return_cost": int(native["return_cost"]),
        "original_max_alternates": config["max_alternates"],
        "score_equality_tolerance_m2": tolerance,
        "original_assessed_candidates": len(old_records),
        "original_admitted_alternates": original["admitted_candidate_count"] - 1,
        "original_coverage_rejected": len(old_records) - original["admitted_candidate_count"],
        "original_skipped_by_pool_cap": len(tail),
        "expanded_assessed_candidates": len(records),
        "expanded_admitted_alternates": expanded["admitted_candidate_count"] - 1,
        "expanded_coverage_rejected": len(records) - expanded["admitted_candidate_count"],
        "newly_assessed_and_admitted": admitted_tail,
        "newly_assessed_and_coverage_rejected": tail_excluded,
        "unassessed_proposals_remaining": 0,
        "original_hypothetical_winners": original["hypothetical_selected_route_id"],
        "expanded_hypothetical_winners": expanded["hypothetical_selected_route_id"],
        "expanded_winners_split_by_prior": between_modes is not None,
        "both_winners_shift_to_same_new_route": between_modes is None and all(modes[m]["changed_route"] for m in MODES),
        "any_winner_shift": any(modes[m]["changed_route"] for m in MODES),
        "expanded_candidate_pool_sha256": expanded["candidate_pool_sha256"],
        "expanded_score_table_sha256": expanded["score_table_sha256"],
    }
    return {
        "status": "archived_four_cap_reproduced_and_original_proposal_tail_scored",
        "source_commit": SOURCE_COMMIT,
        "source_chooser_sha256": SOURCE_SHA256,
        "score_or_action_used_by_actor": False,
        "original_parity": {
            "archived_full_audit_exact": True,
            "original_candidate_pool_sha256": original["candidate_pool_sha256"],
            "original_score_table_sha256": original["score_table_sha256"],
            "archived_arms": parity,
            "original_prefix_exact_in_expanded_pool": True,
            **posterior,
        },
        "expanded_safety_scope": "primitive edges, footprint, cost and native return suffix checked against the restored observed map only",
        "summary": summary,
        "comparisons": {
            "per_mode": modes,
            "expanded_G_vs_S": {
                "first_hypothetical_action_difference": between_modes,
                "G_score_G_winner_minus_S_winner_m2": g["scores"]["G"] - s["scores"]["G"],
                "S_score_S_winner_minus_G_winner_m2": s["scores"]["S"] - g["scores"]["S"],
            },
        },
        "candidates": [{
            **row,
            "original_pool": row["route_id"] in old_ids,
            "skipped_by_original_cap": index >= len(old_records),
            "proposal_index_zero_based": index - 1,
            "original_rank": {mode: original_rank[mode].get(row["route_id"]) for mode in MODES},
            "expanded_rank": {mode: expanded_rank[mode].get(row["route_id"]) for mode in MODES},
        } for index, row in enumerate(records)],
    }
