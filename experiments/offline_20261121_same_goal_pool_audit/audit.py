"""Verify every frozen archive, offline tail route and paired result before closing."""

import csv
import hashlib
import json
import math
from pathlib import Path


ROOT = Path(__file__).resolve().parent
MODES = ("G", "S")


def read(path):
    return json.loads(path.read_text())


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def require(condition, issue):
    if not condition:
        raise ValueError(issue)


def verify_mapping(mapping, directory):
    for name, expected in mapping.items():
        require(sha(directory / name) == expected, f"Frozen file changed: {directory / name}")
    return len(mapping)


def rank_winner(rows, mode, tolerance):
    eligible = [row for row in rows if row["admitted"]]
    highest = max(row["scores"][mode] for row in eligible)
    return min((row for row in eligible if highest - row["scores"][mode] <= tolerance),
               key=lambda row: (row["outbound_cost"], row["route_id"] != "native", row["outbound_sha256"]))


def compare_state(state, before, settings):
    expanded = state["expanded"]
    summary, parity = expanded["summary"], expanded["original_parity"]
    exact = (summary["paid_before"], summary["packet_sha256"], summary["belief_sha256"], summary["native_goal_pose"],
             summary["native_option_id"], state["actual_pose"], summary["native_return_anchor"])
    require(exact == (before["paid_before"], before["packet_sha256"], before["belief_sha256"], before["goal_pose"],
                      before["native_option_id"], before["observed_pose"], before["return_anchor"]),
            f"Recreated state differs from the archive: {state['state_id']}")
    require(not state["used_for_action_selection"] and state["counterfactual_actions_executed"] == 0
            and not expanded["score_or_action_used_by_actor"], "An expanded option reached the runtime")
    require(parity["archived_full_audit_exact"] and parity["original_prefix_exact_in_expanded_pool"]
            and parity["original_candidate_pool_sha256"] == before["original_pool_sha256"]
            and parity["original_score_table_sha256"] == before["original_scores_sha256"], "Original scoring parity failed")
    for mode in MODES:
        row = parity["archived_arms"][mode]
        require(row["archived_full_audit_sha256"] == row["recomputed_full_audit_sha256"] == before[f"original_core_{mode}_sha256"],
                f"Original {mode} score/route data were not exactly reproduced")
    rows = expanded["candidates"]
    original_count = summary["original_assessed_candidates"]
    original, extra = rows[:original_count], rows[original_count:]
    require(len(extra) == summary["original_skipped_by_pool_cap"] and summary["unassessed_proposals_remaining"] == 0
            and all(row["skipped_by_original_cap"] for row in extra)
            and len({tuple(row["outbound_actions"]) for row in rows}) == len(rows), "Cap-only suffix coverage or identity invalid")
    reserve = summary["native_reserved_return_cost"]
    require(reserve == before["native_return_cost"] == len(summary["native_return_actions"]), "Native return reservation changed")
    for row in rows:
        expected_admit = row["unknown_union_cells"] + 1e-12 >= settings["min_unknown_union_ratio"] * rows[0]["unknown_union_cells"]
        require(row["admitted"] == expected_admit and row["goal_pose"] == before["goal_pose"]
                and row["outbound_states"][0] == before["observed_pose"] and row["outbound_states"][-1] == before["goal_pose"]
                and row["outbound_cost"] == len(row["outbound_actions"])
                and row["outbound_cost"] <= before["native_outbound_cost"] + settings["max_extra_outbound_actions"]
                and row["reserved_return_cost"] == reserve and row["outbound_cost"] + reserve <= summary["remaining_budget"],
                f"Online eligibility arithmetic changed for {state['state_id']}/{row['route_id']}")
        require((row["admitted"] and set(row["scores"]) == set(MODES) and
                 all(math.isfinite(value) and value >= 0 for value in row["scores"].values())) or
                (not row["admitted"] and not row["scores"]), "Invalid admitted-only score table")
    tolerance = settings["score_equality_tolerance"]
    old = {mode: rank_winner(original, mode, tolerance) for mode in MODES}
    new = {mode: rank_winner(rows, mode, tolerance) for mode in MODES}
    for mode in MODES:
        require(old[mode]["route_id"] == before["original_winners"][mode] and
                new[mode]["route_id"] == summary["expanded_hypothetical_winners"][mode], "Winner/tie order changed")
    g_actions, s_actions = (new[mode]["outbound_actions"] for mode in MODES)
    different = g_actions != s_actions
    same_shift = not different and g_actions != old["G"]["outbound_actions"] and s_actions != old["S"]["outbound_actions"]
    require(different == summary["expanded_winners_split_by_prior"] and
            same_shift == summary["both_winners_shift_to_same_new_route"], "Action-string interpretation does not reproduce")
    if different:
        first = next((i for i, (g, s) in enumerate(zip(g_actions, s_actions)) if g != s), min(len(g_actions), len(s_actions)))
        witness = expanded["comparisons"]["expanded_G_vs_S"]["first_hypothetical_action_difference"]
        require(witness["relative_action_one_based"] == first + 1 and not witness["executed"] and
                witness["hypothetical_paid_action_id"] == before["paid_before"] + first + 1 and
                witness["first_action"] == (g_actions[first] if first < len(g_actions) else None) and
                witness["second_action"] == (s_actions[first] if first < len(s_actions) else None), "Wrong first differing action")
    return {"cap_skipped": len(extra), "new_eligible": sum(row["admitted"] for row in extra),
            "new_rejected": sum(not row["admitted"] for row in extra), "common_pool_only_shift": int(same_shift),
            "potential_category_action_split": int(different), "same_original_choice": int(not different and not same_shift)}


def main():
    protocol = read(ROOT / "PROTOCOL.json")
    source_path = ROOT / "SOURCE_FREEZE.json"
    source = read(source_path)
    source_count = verify_mapping(source["files_sha256"], ROOT)
    require(source["source_commit"] == protocol["source_commit"] and source["archived_development_episodes"] == 8 and
            source["archived_paired_zero_difference"] == 4, "Original source evidence is inconsistent")
    replay = ROOT / "replay"
    receipt = read(replay / "supervisor_receipt.json")
    freeze = replay / "freeze.json"
    input_count = verify_mapping(read(freeze)["files_sha256"], ROOT)
    layouts = protocol["scope"]["layouts_in_order"]
    require(receipt["status"] == "completed" and receipt["replay_freeze_sha256"] == sha(freeze) and
            [job["layout"] for job in receipt["jobs"]] == layouts, "Missing four sealed archival replays")
    before_table = read(ROOT / "verification/archived_states.json")
    before = {(row["layout"], row["paid_before"], row["global_call_id"]): row for row in before_table["rows"]}
    index, reported, raw_paths = [], {}, {}
    actions = 0
    for job in receipt["jobs"]:
        layout = job["layout"]
        require(job["process_returncode"] == 0 and job["log_sha256"] == sha(replay / f"logs/{layout}.log"), "Replay failed")
        path = replay / f"runs/{layout}/manifest.json"
        record = read(path)
        verify_mapping(record["files_sha256"], path.parent)
        require(record["replay_freeze_sha256"] == sha(freeze) and record["original_guard_verified_per_action"] and
                record["same_trace_and_online_choices_as_both_archives"] and not record["new_trajectory_generated"] and
                not record["heldout_data_read"] and not record["truth_used_for_ranking"] and record["counterfactual_actions_executed"] == 0,
                f"Manifest does not prove a read-only archival replay at {layout}")
        actions += record["replayed_archived_actions"]
        require(len(record["state_files"]) == protocol["scope"]["states_per_layout"][layout], "Enlarged state count drifted")
        for name in record["state_files"]:
            raw_path = path.parent / name
            state = read(raw_path)
            identity = layout, state["paid_before"], state["global_call_id"]
            require(identity not in reported and identity in before, "Wrong or duplicate enlarged decision state")
            reported[identity] = compare_state(state, before[identity], protocol["frozen_online_fields"])
            raw_paths[identity] = raw_path
        index.append({"layout": layout, "manifest": str(path.relative_to(ROOT)), "manifest_sha256": sha(path),
                      "state_files_sha256": {name: record["files_sha256"][name] for name in record["state_files"]}})
    require(len(reported) == 25 and reported.keys() == before.keys(), "Frozen historic state set was not fully audited")
    by_first = sorted(reported, key=lambda key: (key[1], layouts.index(key[0]), before[key]["native_option_id"]))
    earliest = next((key for key in by_first if reported[key]["potential_category_action_split"]), None)
    result_path = ROOT / "verification/results.json"
    verified = read(result_path)
    witness = verified["earliest_verified_nominal_category_difference"]
    verified_first = None if witness is None else (witness["layout"], witness["paid_before"], witness["global_call_id"])
    require(verified["status"] == "verified_cap_only_offline_audit" and verified["verified_states"] == len(reported)
            and verified_first == earliest and verified["counterfactual_actions_executed"] == verified["new_trajectories"]
            == verified["heldout_runs"] == 0, "Independent witness or audit scope differs")
    with (ROOT / "verification/results.csv").open(newline="") as stream:
        table = list(csv.DictReader(stream))
    require(len(table) == 25 and {(row["layout"], int(row["paid_before"]), int(row["global_call_id"])) for row in table}
            == reported.keys(), "Required per-state CSV is incomplete")
    totals = {k: sum(row[k] for row in reported.values()) for k in next(iter(reported.values()))}
    labels = {"category_dependent_nominal_split": "potential_category_action_split",
              "shared_pool_driven_nominal_change": "common_pool_only_shift",
              "unchanged_nominal_winners": "same_original_choice"}
    require(all(verified["relationship_counts"].get(label, 0) == totals[count] for label, count in labels.items()),
            "Independent paired action counts disagree")
    correction_path = replay / "execution_correction.json"
    correction = read(correction_path)
    prior_freeze = Path(correction["prior_freeze"])
    partial = sorted((prior_freeze.parent / "runs").glob("*/states/*.json"))
    require(len(partial) == correction["preserved_partial_result_count"] and not correction["source_protocol_or_scope_changed"]
            and not correction["scoring_or_observations_changed"] and all(
                sha(path) == sha(replay / "runs" / path.relative_to(prior_freeze.parent / "runs")) for path in partial
            ), "Terminal verifier correction altered a previously scored state")
    audit = {"status": "verified", "source_commit": protocol["source_commit"], "source_frozen_files_verified": source_count,
             "replay_frozen_files_verified": input_count, "archived_replayed_actions": actions, "states_with_exact_original_parity": len(reported),
             "counterfactual_actions_executed": 0, "earliest_key_layout_paid_global_call": earliest, "totals": totals,
             "per_state_csv_sha256": sha(ROOT / "verification/results.csv"), "independent_result_json_sha256": sha(result_path),
             "execution_correction_receipt_sha256": sha(correction_path), "pre_correction_states_reproduced_exactly": len(partial),
             "manuscript_paragraph_sha256": sha(ROOT / "RESULTS_ZH.md")}
    evidence = {"source_freeze": str(source_path.relative_to(ROOT)), "replay_freeze": str(freeze.relative_to(ROOT)),
                "archived_state_index": "verification/archived_states.json", "raw_replays": index,
                "execution_correction": str(correction_path.relative_to(ROOT)),
                "archived_attempt_freeze_sha256": sha(prior_freeze)}
    if earliest is not None:
        raw = raw_paths[earliest]
        relative = raw.relative_to(ROOT)
        old = before[earliest]
        witness_path = ROOT / "earliest_witness.json"
        witness_path.write_text(json.dumps({"status": "offline_nominal_difference_for_future_validation",
            "source_commit": protocol["source_commit"], "raw_local_state": str(relative),
            "raw_published_state": f"evidence/{earliest[0]}/{raw.name}.gz", "raw_state_sha256": sha(raw),
            "source_manifest_sha256": {mode: old[f"source_manifest_{mode}_sha256"] for mode in MODES},
            "counterfactual_executed": False, "witness": witness}, indent=2, allow_nan=False) + "\n")
        audit["earliest_witness_sha256"] = sha(witness_path)
        evidence["earliest_witness"] = witness_path.name
    for name, value in (("completion_audit.json", audit), ("EVIDENCE_INDEX.json", evidence)):
        (ROOT / name).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


if __name__ == "__main__":
    main()
