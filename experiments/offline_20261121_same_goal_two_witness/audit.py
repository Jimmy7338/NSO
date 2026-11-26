"""Complete the scope, archive, command and measurement audit for the four runs."""

import json
from pathlib import Path

from report import ROOT, flatten_rows, read, sha


ARMS = ("G", "S")
TELEMETRY = ("action", "position", "heading", "packet_sha256", "depth_values_sha256", "rgb_values_sha256", "scan_ranges_sha256")


def require(result, message):
    if not result:
        raise ValueError(message)


def verify(mapping, directory):
    for name, expected in mapping.items():
        require(sha(directory / name) == expected, f"Evidence file changed: {directory / name}")
    return len(mapping)


def actual_first_difference(g, s, prefix, fields):
    return next((left["step"] for left, right in zip(g[prefix:], s[prefix:])
                 if any(left[key] != right[key] for key in fields)), None)


def verify_arm(protocol, freeze, directory, group, mode):
    manifest_path = directory / "manifest.json"
    manifest = read(manifest_path)
    verify(manifest["files_sha256"], directory)
    prefix = group["paid_before"]
    require(manifest["actual_closed_loop"] and not manifest["evaluator_inputs_read_by_actor"] and
            not manifest["forced_goal"] and not manifest["forced_route"] and
            manifest["native_budget"] == protocol["total_paid_action_budget"] and
            manifest["single_intervention_paid_before"] == prefix == manifest["replayed_archived_prefix_actions"] and
            manifest["single_intervention_call_id"] == group["global_call_id"] and
            manifest["source_commit"] == protocol["source_commit"] and manifest["runner_freeze_sha256"] == sha(freeze) and
            manifest["global_mode"] == manifest["subsequent_local_mode"] == "G" and manifest["subsequent_local_alternate_cap"] == 4,
            f"Run modified the registered choice/continuation scope: {directory}")
    intervention = read(directory / "intervention.json")
    require(intervention["prior_prefix_reproduced_before_selection"] and intervention["expanded_pool_and_scores_equal_to_offline_source"] and
            intervention["native_selected_option_unchanged"] and not intervention["forced_goal"] and not intervention["forced_route"] and
            intervention["original_return_actions"] == intervention["actual_return_actions"] and
            len(intervention["actual_return_actions"]) == group["native_reserved_return"] and
            len(intervention["selected_outbound_actions"]) == group["archived_nominal_outbound_for_verification_only"][mode] and
            intervention["chosen_route_id"] == group["archived_expected_winners_for_verification_only"][mode],
            f"Restored witness differs from live selector outputs: {directory}")
    local = read(directory / "local_choices.json")
    require(sum(item["intervention"] for item in local) == 1 and
            all(item["choice_audit"]["selected_mode"] == (mode if item["intervention"] else "G") for item in local),
            f"More than one mode change or wrong neutral restoration: {directory}")
    rng = manifest["restored_rng_identity"]
    require(rng == intervention["restored_rng_identity"] and rng["scene_seed"] == 20261121 and rng["deterministic_pose_geometry"] and
            rng["depth_noise_sigma_m"] == rng["depth_dropout_fraction"] == 0 and
            not rng["stateful_generator_retained_by_simulator"], f"Simulator random identity changed: {directory}")
    trace = read(directory / "steps.json")
    require([item["step"] for item in trace] == list(range(1, manifest["paid_actions"] + 1)) and
            len(trace) - prefix == manifest["actual_new_actions"] and len(trace) <= manifest["native_budget"] and
            all(item["guard_allowed"] and all(key in item for key in TELEMETRY) for item in trace) and
            all(item["charged_new_continuation_action"] == (item["step"] > prefix) for item in trace),
            f"Invalid charged action ledger or missing sensor input at {directory}")
    require(manifest["collisions"] == sum(bool(row["collision"]) for row in trace) and
            manifest["final_packet_chain_sha256"] == trace[-1]["packet_chain_sha256"] and
            manifest["final_pose"] == [*trace[-1]["position"], trace[-1]["heading"]], "Terminal trace does not bind manifest")
    entry = {"manifest": str(manifest_path.relative_to(ROOT)), "manifest_sha256": sha(manifest_path),
             "files_sha256": manifest["files_sha256"]}
    return manifest, intervention, trace, entry


def main():
    protocol_path = ROOT / "PROTOCOL.json"
    protocol = read(protocol_path)
    runner = ROOT / "runner"
    source_path = ROOT / "SOURCE_FREEZE.json"
    source = read(source_path)
    source_count = verify(source["files_sha256"], ROOT)
    freeze_path = runner / "freeze.json"
    freeze, receipt = (read(path) for path in (freeze_path, runner / "supervisor_receipt.json"))
    runner_count = verify(freeze["files_sha256"], ROOT)
    schedule = [(layout, mode) for layout in protocol["layouts_in_order"] for mode in ARMS]
    require([(job["layout"], job["mode"]) for job in receipt["jobs"]] == schedule
            == [(job["layout"], job["mode"]) for job in freeze["schedule"]] and
            receipt["runner_freeze_sha256"] == sha(freeze_path) and
            freeze["status"] == "frozen_before_any_new_trajectory" and
            receipt["status"] in {"completed", "failed"}, "Exactly four sealed continuations are required")
    require(source["source_commit"] == protocol["source_commit"] == freeze["source_commit"] and
            set(path.relative_to(runner / "runs").parts for path in (runner / "runs").glob("*/*")) == set(schedule),
            "Wrong predecessor or unregistered actor output")

    measurement = ROOT / "measurement/results"
    pairs = read(measurement / "pair_causal_audit.json")["rows"]
    outcomes = {row["layout"]: row for row in pairs}
    episodes = {(row["layout"], row["mode"]): row for row in read(measurement / "episodes.json")["rows"]}
    require(set(episodes) == set(schedule), "Missing registered terminal or failure")
    current, entries, failures = {}, [], []
    for job in receipt["jobs"]:
        layout, mode = job["layout"], job["mode"]
        path = runner / "runs" / layout / mode
        require(sha(runner / f"logs/{layout}_{mode}.log") == job["log_sha256"], "Actor log changed")
        if job["returncode"] != 0:
            failed = read(path / "failure.json")
            require(episodes[layout, mode]["failed"], f"Interruption suppressed from output: {path}")
            failures.append({"layout": layout, "mode": mode, "receipt": str((path / "failure.json").relative_to(ROOT)),
                             "sha256": sha(path / "failure.json"), "status": failed["status"]})
            continue
        manifest, intervention, trace, entry = verify_arm(protocol, freeze_path, path, protocol["pairs"][layout], mode)
        measured = episodes[layout, mode]
        require((manifest["paid_actions"], manifest["returned"], manifest["collisions"]) ==
                (measured["actual_paid"], measured["returned"], measured["collisions"]), "Terminal metrics altered the action ledger")
        entries.append(entry)
        current[layout, mode] = manifest, intervention, trace

    split_summary = {}
    for layout, group in protocol["pairs"].items():
        if not all((layout, arm) in current for arm in ARMS):
            continue
        (gm, gi, gt), (sm, si, st) = (current[layout, arm] for arm in ARMS)
        prefix = group["paid_before"]
        require([[s[key] for key in TELEMETRY] for s in gt[:prefix]] ==
                [[s[key] for key in TELEMETRY] for s in st[:prefix]] and
                gm["restored_rng_identity"] == sm["restored_rng_identity"] and
                gi["measured_prefix"] == si["measured_prefix"] and
                gi["expanded_candidate_pool_sha256"] == si["expanded_candidate_pool_sha256"] and
                gi["expanded_score_table_sha256"] == si["expanded_score_table_sha256"] and
                gi["native_target"] == si["native_target"], f"Arms do not share the same actual recovered state: {layout}")
        result = outcomes[layout]
        fields = (("first_command_difference", ("action",)), ("first_pose_difference", ("position", "heading")),
                  ("first_packet_difference", ("packet_sha256", "nonsemantic_packet_sha256")),
                  ("first_sensor_content_difference", TELEMETRY[-3:]))
        split = {key: actual_first_difference(gt, st, prefix, keys) for key, keys in fields}
        require(all((result[key] or {}).get("paid") == paid for key, paid in split.items()),
                "Published onset does not match the charged trace")
        if layout == "warehouse_right":
            for intervened in (gi, si):
                denied = intervened["pre_call_native_guard"][-1]
                allowed = intervened["post_call_native_guard"][-1]
                require(denied["call_id"] < group["global_call_id"] < allowed["call_id"] and
                        denied["outputs"]["reason"] == "next_footprint_not_known_safe" and not denied["outputs"]["allowed"] and
                        allowed["outputs"]["allowed"], "Warehouse witness was taken before rather than inside native replan")
            require(result["source_guard_order_verified"], "Independent guard-order evidence is missing")
        split_summary[layout] = split

    results = ROOT / "results"
    report = read(results / "decision.json")
    verify(report["input_sha256"], ROOT)
    verify(report["result_sha256"], results)
    _, retally = flatten_rows(protocol, pairs)
    gate = read(measurement / "engineering_gate.json")
    verify(gate["input_evidence_sha256"], measurement)
    require(retally == report["per_layout"] and
            report["continue_to_separate_zero_start_proposal"] == gate["engineering_gate"]["pass"], "Gate not reproducible")
    final = {"status": "verified", "source_commit": protocol["source_commit"], "frozen_source_files": source_count,
             "frozen_actor_inputs": runner_count, "registered_and_attempted_continuations": len(schedule),
             "completed_continuations": len(entries), "engineering_interruptions": failures,
             "new_actual_paid_actions": sum(row[0]["actual_new_actions"] for row in current.values()),
             "first_actual_onsets": split_summary, "terminal_conditions": retally,
             "continuation_gate": report["continue_to_separate_zero_start_proposal"], "heldout_runs": 0,
             "paired_report_sha256": sha(results / "paired_results.csv"), "paper_paragraph_sha256": sha(ROOT / "RESULTS_ZH.md")}
    index = {"source_freeze": "SOURCE_FREEZE.json", "actor_freeze": "runner/freeze.json", "raw_arms": entries,
             "interruptions": failures, "native_mesh_and_belief_location": str(runner / "runs"),
             "source_mechanism_preflight": "diagnostics/source_mechanism.json"}
    for name, data in (("completion_audit.json", final), ("EVIDENCE_INDEX.json", index)):
        (ROOT / name).write_text(json.dumps(data, indent=2, allow_nan=False) + "\n")


if __name__ == "__main__":
    main()
