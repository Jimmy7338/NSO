"""Read sealed offline candidate rows and independently recheck eligibility and choice."""

import argparse
from collections import Counter, deque
import math
from pathlib import Path

import numpy as np
from scipy.ndimage import distance_transform_edt

from archive_states import (
    ARMS, COMMIT, DIRS, SOURCE, STUDY, array_digest, digest, read, require, sha_file, verify, verify_path,
    write_csv, write_json,
)

CAP_REASON = "remaining_candidates_after_admitted_pool_cap"
COVERAGE_REASON = "observable_unknown_union_below_native_floor"
DECORATIONS = ("original_pool", "skipped_by_original_cap", "proposal_index_zero_based", "original_rank", "expanded_rank")
MODES = ("G", "S")
SOURCE_RUNTIME = STUDY.parent / (
    "reproduction_20261104/simulation_publication_stage/experiments/unknown_exploration_20261104/source"
)
SOURCE_BINDING = STUDY.parent / "paired_unseen_gain_20261120/execution/freeze.json"


def shortest_distances(safe, pose, reverse=False):
    start = tuple(pose)
    agenda = deque([start])
    distances = {start: 0}
    height, width = safe.shape
    move_sign = -1 if reverse else 1
    while agenda:
        r, c, h = here = agenda.popleft()
        dr, dc = DIRS[h]
        rr, cc = r + move_sign * dr, c + move_sign * dc
        adjacent = [(r, c, (h + 1) % 4), (r, c, (h - 1) % 4)]
        if 0 <= rr < height and 0 <= cc < width and safe[rr, cc]:
            adjacent.append((rr, cc, h))
        for state in adjacent:
            if state not in distances:
                distances[state] = distances[here] + 1
                agenda.append(state)
    return distances


def safe_scene(belief, start, goal):
    require(belief.ndim == 2 and np.isin(belief, [-1, 0, 1]).all(), "Historical belief is not a ternary grid")
    # The frozen simulator uses a 0.2 m grid and a 0.2 m circular footprint.
    # Match the native circumscribed-cell border convention without instantiating a controller.
    safe = distance_transform_edt(np.pad(belief == 0, 1, constant_values=False))[1:-1, 1:-1] > 1 + math.sqrt(2) / 2
    require(safe[tuple(start[:2])] and safe[tuple(goal[:2])], "Historical endpoint violates the observed-safe graph")
    return safe, shortest_distances(safe, start), shortest_distances(safe, goal, reverse=True)


def classify_proposal(row, native, origin, cfg, safe, forward, backward, original):
    rid = row["route_id"]
    actions, states = row["outbound_actions"], row["outbound_states"]
    label = f"{origin['layout']} paid {origin['paid_before']} {rid}"
    verify_path(actions, states, origin["observed_pose"], origin["goal_pose"], label)
    require(all(0 <= r < safe.shape[0] and 0 <= c < safe.shape[1] and safe[r, c] for r, c, _ in states),
            f"Path leaves the known-safe observed pose graph: {label}")
    require(digest({"outbound_actions": actions, "outbound_states": states}) == row["outbound_sha256"]
            and len(actions) == row["outbound_cost"] and row["reserved_return_cost"] == origin["native_return_cost"]
            and row["goal_pose"] == origin["goal_pose"] and row["common_score"] == 0.,
            f"Route checksum, packed target, cost, reserve, or scoring fee changed: {label}")
    require(row["outbound_cost"] + row["reserved_return_cost"] <= (256 - origin["paid_before"]),
            f"Route and unchanged original return exceed remaining actions: {label}")
    radar, camera, union = (row[x] for x in ("predicted_radar_cells", "predicted_camera_cells", "unknown_union_cells"))
    require(all(isinstance(x, int) and x >= 0 for x in (radar, camera, union))
            and max(radar, camera) <= union <= radar + camera,
            f"Recorded observed unknown-cell coverage is internally inconsistent: {label}")
    if rid == "native":
        require(row["admitted"] and row["waypoint_pose"] is None and not row["novel_view_tags"]
                and forward[tuple(origin["goal_pose"])] == len(actions), f"Original shortest path changed: {label}")
    else:
        via = tuple(row["waypoint_pose"])
        require(row["waypoint_pose"] in states[1:-1] and via in forward and via in backward
                and len(actions) == forward[via] + backward[via]
                and len(actions) <= native["outbound_cost"] + cfg["max_extra_outbound_actions"]
                and origin["goal_pose"] not in states[:-1],
                f"Route is no longer the original one-waypoint shortest trace or extra-action bound: {label}")
        supported = set(original["measured_supported_tracks"])
        old_tags = set(map(tuple, original["native_view_tags"]))
        new_tags = set(map(tuple, row["novel_view_tags"]))
        require(new_tags and not (new_tags & old_tags)
                and all(track in supported and 0 <= azimuth < cfg["azimuth_bins"] for track, azimuth in new_tags),
                f"Recorded novelty does not come from original measured-instance/azimuth context: {label}")
        enough = union + 1e-12 >= cfg["min_unknown_union_ratio"] * native["unknown_union_cells"]
        require(row["admitted"] == enough and row["exclusion_reason"] == (None if enough else COVERAGE_REASON),
                f"Route uses a different class-independent coverage check: {label}")
    require(set(row["scores"]) == (set(MODES) if row["admitted"] else set())
            and all(math.isfinite(v) and v >= 0 for v in row["scores"].values()), f"Invalid recorded scores: {label}")


def rank_candidates(rows, mode, tolerance):
    remaining = [row for row in rows if row["admitted"]]
    ranked = []
    while remaining:
        top = max(row["scores"][mode] for row in remaining)
        eligible = [row for row in remaining if top - row["scores"][mode] <= tolerance]
        selected = min(eligible, key=lambda row: (row["outbound_cost"], row["route_id"] != "native",
                                                    row["outbound_sha256"]))
        remaining.remove(selected)
        ranked.append(selected)
    return ranked


def nominal_difference(before, after, paid):
    old, new = before["outbound_actions"], after["outbound_actions"]
    common = min(len(old), len(new))
    offset = next((index for index in range(common) if old[index] != new[index]), common)
    if offset == common and len(old) == len(new):
        require(before["outbound_states"] == after["outbound_states"], "Identical actions have different nominal poses")
        return None
    require(before["outbound_states"][:offset + 1] == after["outbound_states"][:offset + 1],
            "Nominal poses differ before the first differing action")
    return {
        "relative_action_one_based": offset + 1, "hypothetical_paid_action_id": paid + offset + 1,
        "shared_pre_pose": before["outbound_states"][offset],
        "first_action": old[offset] if offset < len(old) else None,
        "second_action": new[offset] if offset < len(new) else None,
        "first_predicted_post_pose": before["outbound_states"][offset + 1] if offset < len(old) else None,
        "second_predicted_post_pose": after["outbound_states"][offset + 1] if offset < len(new) else None,
        "executed": False,
    }


def numeric_equal(a, b):
    return a is b if a is None or b is None else math.isclose(a, b, abs_tol=1e-12, rel_tol=0)


def verify_scores(expanded, original, rows, old_rows, cfg, paid):
    requested_tie = cfg["score_equality_tolerance"]
    ranks_a = {mode: rank_candidates(old_rows, mode, requested_tie) for mode in MODES}
    ranks_b = {mode: rank_candidates(rows, mode, requested_tie) for mode in MODES}
    for subset, label in ((ranks_a, "original_rank"), (ranks_b, "expanded_rank")):
        rank_maps = {mode: {r["route_id"]: i for i, r in enumerate(subset[mode], 1)} for mode in MODES}
        for row in rows:
            expected = {mode: rank_maps[mode].get(row["route_id"]) for mode in MODES}
            require(row[label] == expected, f"Incorrect deterministic tied ranking {label}: {row['route_id']}")
    compare = expanded["comparisons"]
    summary = expanded["summary"]
    old_winners, new_winners = {}, {}
    metrics = {}
    for mode in MODES:
        old, new = (ranks_a[mode][0], ranks_b[mode][0])
        old_winners[mode], new_winners[mode] = old, new
        detail = compare["per_mode"][mode]
        require(old["route_id"] == original["hypothetical_selected_route_id"][mode]
                == summary["original_hypothetical_winners"][mode]
                and new["route_id"] == summary["expanded_hypothetical_winners"][mode],
                f"Recomputed winner disagrees for {mode}")
        require(nominal_difference(old, new, paid) == detail["first_hypothetical_action_difference"],
                f"Wrong command difference from original to expanded for {mode}")
        shift = new["scores"][mode] - old["scores"][mode]
        require(numeric_equal(shift, detail["expanded_minus_original_score_m2"]), f"Wrong score change for {mode}")
        metrics |= {f"expanded_minus_original_{mode}_m2": shift}
        for phase, rank in (("original", ranks_a[mode]), ("expanded", ranks_b[mode])):
            expected, actual = rank[0], detail[phase]
            second = rank[1] if len(rank) > 1 else None
            margin = None if second is None else expected["scores"][mode] - second["scores"][mode]
            require(actual["route_id"] == expected["route_id"] and actual["outbound_sha256"] == expected["outbound_sha256"]
                    and actual["scores_m2"] == expected["scores"] and numeric_equal(actual["winner_minus_runner_up_m2"], margin)
                    and actual["runner_up_route_id"] == (second["route_id"] if second else None)
                    and numeric_equal(actual["runner_up_score_m2"], second["scores"][mode] if second else None),
                    f"Wrong winner/runner-up scores for {phase}/{mode}")
            metrics[f"{phase}_{mode}_winner"] = expected["route_id"]
            metrics[f"{phase}_{mode}_winner_paid_outbound"] = expected["outbound_cost"]
            metrics[f"{phase}_{mode}_winner_runner_up_margin_m2"] = margin
            metrics |= {f"{phase}_{mode}_winner_score_{score_mode}_m2": expected["scores"][score_mode]
                        for score_mode in MODES}
    g, s = new_winners["G"], new_winners["S"]
    pair_difference = nominal_difference(g, s, paid)
    pair = compare["expanded_G_vs_S"]
    advantages = {
        "G_score_G_winner_minus_S_winner_m2": g["scores"]["G"] - s["scores"]["G"],
        "S_score_S_winner_minus_G_winner_m2": s["scores"]["S"] - g["scores"]["S"],
    }
    require(pair["first_hypothetical_action_difference"] == pair_difference
            and all(numeric_equal(advantage, pair[name]) for name, advantage in advantages.items()),
            "Expanded cross-prior winner margins or nominal action difference disagree")
    changed = {mode: nominal_difference(old_winners[mode], new_winners[mode], paid) is not None
               for mode in MODES}
    require((pair_difference is not None) == summary["expanded_winners_split_by_prior"]
            and summary["both_winners_shift_to_same_new_route"] == (pair_difference is None and all(changed.values()))
            and summary["any_winner_shift"] == any(changed.values()), "Raw nominal action classification disagrees")
    relationship = "category_dependent_nominal_split" if pair_difference else (
        "shared_pool_driven_nominal_change" if all(changed.values()) else "unchanged_nominal_winners"
    )
    return metrics | advantages, relationship, pair_difference, new_winners


def verify_state(raw, origin, original_rows, archived_belief, cfg, frozen_chooser_sha):
    paid = origin["paid_before"]
    origin_fields = ("layout", "paid_before", "global_call_id", "packet_sha256", "belief_sha256")
    require(all(raw[k] == origin[k] for k in origin_fields)
            and raw["source_decision_ordinal"] == origin["ordinal"]
            and raw["actual_pose"] == origin["observed_pose"]
            and raw["state_id"] == f"{origin['layout']}:{paid}:{origin['global_call_id']}",
            "Offline result represents a different recorded state")
    require(raw["original_source_G_SHA"] == origin["original_row_G_sha256"]
            and raw["original_source_S_SHA"] == origin["original_row_S_sha256"]
            and not raw["used_for_action_selection"] and raw["counterfactual_actions_executed"] == 0,
            "Original mode provenance or counterfactual isolation failed")
    prior_g, prior_s = (original_rows[arm][origin["ordinal"]] for arm in ARMS)
    original = raw["original_choice_audit"]
    require(original == prior_g["choice_audit"]
            and digest(prior_g) == origin["original_row_G_sha256"]
            and digest(prior_s) == origin["original_row_S_sha256"]
            and raw["global_native_sha256"] == prior_g["global_native_sha256"],
            "Original capped scoring or full source wrapper does not exactly replay")
    require(archived_belief.ndim == 2 and array_digest(archived_belief) == origin["belief_sha256"],
            "Original frozen grid differs from the offline replay")
    option = raw["original_native_option"]
    require(option["option_id"] == origin["native_option_id"] and option["pose"] == origin["goal_pose"]
            and option["outbound_actions"] == prior_g["native_outbound_actions"]
            and option["outbound_states"] == prior_g["native_outbound_states"]
            and option["return_actions"] == prior_g["native_return_actions"]
            and option["return_states"] == prior_g["native_return_states"],
            "Replayed local choice changes original native goal, path or return")

    expanded = raw["expanded"]
    require(expanded["status"] == "archived_four_cap_reproduced_and_original_proposal_tail_scored"
            and expanded["source_commit"] == COMMIT and expanded["source_chooser_sha256"] == frozen_chooser_sha
            and not expanded["score_or_action_used_by_actor"], "Expansion did not use the sealed chooser offline")
    parity = expanded["original_parity"]
    for mode, prior_row in zip(MODES, (prior_g, prior_s), strict=True):
        sealed = parity["archived_arms"][mode]
        require(sealed["archived_full_audit_sha256"] == sealed["recomputed_full_audit_sha256"]
                == digest(prior_row["choice_audit"]) == origin[f"original_core_{mode}_sha256"]
                and sealed["original_selected_route_id"] == prior_row["choice_audit"]["selected_route_id"],
                f"Original {mode} rerun score/exclusion/pool parity failed")
    require(parity["original_candidate_pool_sha256"] == original["candidate_pool_sha256"]
            == origin["original_pool_sha256"]
            and parity["original_score_table_sha256"] == original["score_table_sha256"] == origin["original_scores_sha256"]
            and parity["archived_full_audit_exact"] and parity["original_prefix_exact_in_expanded_pool"],
            "Original score/pool/exclusion parity failed")
    detail = expanded["summary"]
    ledger = {
        "paid_before": paid, "packet_sha256": origin["packet_sha256"], "belief_sha256": origin["belief_sha256"],
        "mapper_frames": paid + 1, "remaining_budget": 256 - paid, "native_return_anchor": origin["return_anchor"],
        "native_option_id": origin["native_option_id"], "native_goal_pose": origin["goal_pose"],
        "native_outbound_sha256": origin["native_outbound_sha256"], "native_return_actions": option["return_actions"],
        "native_return_states": option["return_states"], "native_reserved_return_cost": origin["native_return_cost"],
        "original_max_alternates": cfg["max_alternates"], "score_equality_tolerance_m2": cfg["score_equality_tolerance"],
    }
    require(all(detail[key] == v for key, v in ledger.items()), "Observed state, charge, goal, or return differs from source")
    old_rows = original["candidates"]
    full = expanded["candidates"]
    full_base = [{k: v for k, v in row.items() if k not in DECORATIONS} for row in full]
    source_cap_tail = original["proposal_diagnostics"]["exclusions"].get(CAP_REASON, 0)
    trace_count = original["proposal_diagnostics"]["distinct_alternate_traces"]
    require(len(full) == trace_count + 1 and len(full) - len(old_rows) == source_cap_tail
            and full_base[:len(old_rows)] == old_rows, "Expanded pool is not the original ledger prefix plus its entire unassessed tail")
    require(detail["expanded_candidate_pool_sha256"] == digest([
        {k: v for k, v in row.items() if k != "scores"} for row in full_base
    ]) and detail["expanded_score_table_sha256"] == digest([row["scores"] for row in full_base]),
            "Expanded candidate or score checksum changed")
    require(len(full) == len({row["route_id"] for row in full})
            == len({row["outbound_sha256"] for row in full})
            == len({tuple(row["outbound_actions"]) for row in full})
            and full[0]["route_id"] == "native", "Expanded pool contains duplicated command sequences or loses native")
    safe, from_start, to_goal = safe_scene(archived_belief, origin["observed_pose"], origin["goal_pose"])
    require(all(safe[r, c] for r, c, _ in option["return_states"]), "Archived return cannot remain in the same observed graph")
    for i, row in enumerate(full):
        require(row["proposal_index_zero_based"] == i - 1 and row["original_pool"] == (i < len(old_rows))
                and row["skipped_by_original_cap"] == (i >= len(old_rows)), "Expanded row mislabels archived membership")
        classify_proposal(row, full[0], origin, cfg, safe, from_start, to_goal, original)
    tail = full[len(old_rows):]
    counters = {
        "original_assessed_candidates": len(old_rows),
        "original_admitted_alternates": sum(row["admitted"] for row in old_rows) - 1,
        "original_coverage_rejected": sum(not row["admitted"] for row in old_rows),
        "original_skipped_by_pool_cap": len(tail),
        "expanded_assessed_candidates": len(full),
        "expanded_admitted_alternates": sum(row["admitted"] for row in full) - 1,
        "expanded_coverage_rejected": sum(not row["admitted"] for row in full),
        "newly_assessed_and_admitted": sum(row["admitted"] for row in tail),
        "newly_assessed_and_coverage_rejected": sum(not row["admitted"] for row in tail),
        "unassessed_proposals_remaining": 0,
    }
    require(all(detail[key] == v for key, v in counters.items()), "Expanded score/admission/rejection counts disagree")
    scores, label, difference, winning_rows = verify_scores(expanded, original, full, old_rows, cfg, paid)
    event = {
        "layout": origin["layout"], "ordinal": raw["ordinal"], "source_decision_ordinal": origin["ordinal"],
        "paid_before": paid, "global_call_id": origin["global_call_id"], "native_option_id": origin["native_option_id"],
        "state_id": raw["state_id"], "observed_pose": raw["actual_pose"], "goal_pose": origin["goal_pose"],
        "packet_sha256": origin["packet_sha256"], "belief_sha256": origin["belief_sha256"],
        "native_outbound_cost": origin["native_outbound_cost"], "native_return_cost": origin["native_return_cost"],
        "original_pool_sha256": origin["original_pool_sha256"], "original_scores_sha256": origin["original_scores_sha256"],
        "expanded_pool_sha256": detail["expanded_candidate_pool_sha256"],
        "expanded_scores_sha256": detail["expanded_score_table_sha256"],
        **counters, **scores, "relationship": label, "counterfactual_actions_executed": 0,
        "first_difference_relative_action_one_based": None, "first_difference_nominal_paid_action": None,
        "shared_pre_action_pose": None, "G_first_nominal_action": None, "S_first_nominal_action": None,
    }
    witness = None
    if difference:
        event.update({
            "first_difference_relative_action_one_based": difference["relative_action_one_based"],
            "first_difference_nominal_paid_action": difference["hypothetical_paid_action_id"],
            "shared_pre_action_pose": difference["shared_pre_pose"],
            "G_first_nominal_action": difference["first_action"] or "end_of_sequence",
            "S_first_nominal_action": difference["second_action"] or "end_of_sequence",
        })
        witness = event | {"first_hypothetical_difference": difference, "nominal_winning_routes": winning_rows}
    return event, witness


def verify_layout(layout, expected, archived, running_freeze, cfg, chooser_sha):
    directory = STUDY / "replay/runs" / layout
    manifest_path = directory / "manifest.json"
    manifest = read(manifest_path)
    require(manifest["status"] == "completed_exact_archived_replay" and manifest["layout"] == layout
            and manifest["source_commit"] == COMMIT and manifest["replay_freeze_sha256"] == sha_file(running_freeze)
            and manifest["replayed_archived_actions"] == 256 and manifest["scored_decision_states"] == len(expected)
            and manifest["counterfactual_actions_executed"] == 0 and not manifest["new_trajectory_generated"]
            and not manifest["heldout_data_read"] and not manifest["truth_used_for_ranking"]
            and manifest["original_guard_verified_per_action"] and manifest["same_trace_and_online_choices_as_both_archives"],
            f"Replay never achieved exact archived stop or went outside frozen scope: {layout}")
    for rel, checksum in manifest["files_sha256"].items():
        verify(directory / rel, checksum)
    required_files = {file for file in manifest["files_sha256"] if file.startswith("states/")}
    require(len(manifest["state_files"]) == len(expected) and set(manifest["state_files"]) == required_files,
            f"Replay state membership omitted or duplicated a state: {layout}")
    original_rows = {}
    belief = None
    for arm in ARMS:
        binding = archived["run_file_bindings"][layout][arm]
        source_directory = Path(binding["manifest_path"]).parent
        prior = read(binding["manifest_path"])
        replay_sha = manifest["source_archives"][arm]
        require(Path(replay_sha["path"]) == source_directory
                and replay_sha["files_sha256"]["manifest.json"] == binding["manifest_sha256"]
                and all(checksum == binding["files_sha256"][name] for name, checksum in replay_sha["files_sha256"].items()
                        if name != "manifest.json"), f"Replay points to a different original run: {layout}/{arm}")
        for name in ("start", "terminal"):
            reconstructed, old = manifest[name], prior["checkpoints"][name]
            require(reconstructed["paid"] == old["actual_paid"]
                    and reconstructed["native_mesh_sha256"] == old["files_sha256"]["native_mesh.ply"]
                    and all(reconstructed[key] == old[key] for key in (
                        "packet_sha256", "belief_sha256", "packet_chain_sha256", "native_mesh_fields",
                    )), f"Replayed archived {name} was not byte-identical: {layout}/{arm}")
        original_rows[arm] = read(source_directory / "local_choices.json")
        if arm == "newG":
            path = source_directory / "observed_belief_trace.npz"
            verify(path, binding["files_sha256"]["observed_belief_trace.npz"])
            with np.load(path, allow_pickle=False) as state:
                belief = state["belief"]
    rows, witnesses = [], []
    for seq, (state, relative) in enumerate(zip(expected, manifest["state_files"], strict=True)):
        raw = read(directory / relative)
        require(raw["ordinal"] == seq, f"Archived replay order is ambiguous: {layout}/{relative}")
        row, witness = verify_state(raw, state, original_rows, belief[state["paid_before"]], cfg, chooser_sha)
        rows.append(row | {"replayed_state_sha256": manifest["files_sha256"][relative]})
        if witness:
            witnesses.append(witness | {"replayed_state_sha256": manifest["files_sha256"][relative]})
    return rows, witnesses, sha_file(manifest_path)


def main(output):
    protocol_path = STUDY / "PROTOCOL.json"
    source_freeze_path = STUDY / "SOURCE_FREEZE.json"
    running_freeze = STUDY / "replay/freeze.json"
    protocol = read(protocol_path)
    frozen = read(source_freeze_path)
    cfg = protocol["frozen_online_fields"]
    require(protocol["frozen_before_expanded_scores"] and protocol["source_commit"] == frozen["source_commit"] == COMMIT
            and protocol["source_local"] == str(SOURCE), "Experiment must use the predeclared published source")
    for path, checksum in frozen["files_sha256"].items():
        verify(path, checksum)
    frozen_replay = read(running_freeze)
    for path, checksum in frozen_replay["files_sha256"].items():
        verify(path, checksum)
    correction_path = STUDY / "replay/execution_correction.json"
    correction_sha = None
    if correction_path.is_file():
        correction = read(correction_path)
        require(not correction["source_protocol_or_scope_changed"]
                and not correction["scoring_or_observations_changed"]
                and correction["corrected_replay_sha256"] == sha_file(STUDY / "replay/replay.py"),
                "Replay restart is unrelated to the sealed terminal serialization correction")
        correction_sha = sha_file(correction_path)
    ledger_path = Path(__file__).resolve().with_name("archived_states.json")
    verify(ledger_path, frozen_replay["files_sha256"][str(ledger_path)])
    archived = read(ledger_path)
    require(cfg == {key: read(SOURCE / "PROTOCOL.json")["local_route"][key] for key in cfg},
            "Offline audit changes a frozen route rule")
    # Bind the reference footprint convention used by safe_scene() and next_pose().
    source_engine = read(SOURCE_BINDING)["files_sha256"]
    for rel in ("nso/execution_guard_v8.py", "utils/grid_geometry.py", "env/virtual3d.py"):
        source_path = SOURCE_RUNTIME / rel
        verify(source_path, source_engine[str(source_path)])
    chooser = SOURCE / "route_choice.py"
    chooser_sha = verify(chooser, archived["source_study_files_sha256"]["route_choice.py"])
    layout_order = protocol["scope"]["layouts_in_order"]
    rows, witnesses = [], []
    manifests = {}
    for layout in layout_order:
        expected = [row for row in archived["rows"] if row["layout"] == layout]
        require(len(expected) == protocol["scope"]["states_per_layout"][layout], "Original state count changed")
        current, nominal, digest_manifest = verify_layout(layout, expected, archived, running_freeze, cfg, chooser_sha)
        rows.extend(current)
        witnesses.extend(nominal)
        manifests[layout] = digest_manifest
    require(len(rows) == protocol["scope"]["common_states"] == 25
            and len({row["state_id"] for row in rows}) == 25, "No complete claim from a smaller or duplicated matrix")
    chronology = sorted(witnesses, key=lambda row: (row["paid_before"], layout_order.index(row["layout"]),
                                                    row["native_option_id"]))
    relationships = dict(Counter(row["relationship"] for row in rows))
    metadata = {
        "status": "verified_cap_only_offline_audit", "source_commit": COMMIT,
        "protocol_sha256": sha_file(protocol_path), "source_freeze_sha256": sha_file(source_freeze_path),
        "replay_freeze_sha256": sha_file(running_freeze), "archived_states_sha256": sha_file(ledger_path),
        "verified_replay_manifests_sha256": manifests, "verification_code_sha256": sha_file(__file__),
        "terminal_serialization_correction_sha256": correction_sha,
        "verified_states": len(rows), "historical_source_actions_replayed": 4 * 256,
        "counterfactual_actions_executed": 0, "new_trajectories": 0, "heldout_runs": 0,
        "relationship_counts": relationships, "earliest_verified_nominal_category_difference": chronology[0] if chronology else None,
    }
    output.mkdir(parents=True, exist_ok=True)
    write_csv(output / "results.csv", rows)
    write_json(output / "results.json", metadata | {"rows": rows, "nominal_category_witnesses": chronology})
    print(f"Verified {len(rows)} exact archived states. Nominal category differences: {len(chronology)}.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=Path(__file__).resolve().parent)
    main(parser.parse_args().out)
