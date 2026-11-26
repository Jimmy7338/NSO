"""Independently measure native checkpoints and verify the two archived G/S interventions."""

import argparse
import importlib.util
import math
from pathlib import Path

STUDY = Path(__file__).resolve().parents[1]
PREDECESSOR = STUDY.parent / "same_goal_pool_audit_20261121"
DEVELOPMENT = STUDY.parent / "same_goal_route_choice_20261121"
WITNESS_TABLE = PREDECESSOR / "verification/results.json"
PREDECESSOR_COMMIT = "3855d5d710e659adf665aa61dbc66eafb217c467"
ARMS = ("G", "S")
CAP = 256


def import_path(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


development = import_path("nso_verified_prior_same_goal_metrics", DEVELOPMENT / "measurement/evaluate.py")
engine = development.engine
files = development.files
METRICS = development.TERMINAL_METRICS
TRACE_CLOCK_FIELDS = ("step", "action", "position", "heading", "packet_sha256", "packet_chain_sha256",
                      "frame_sha256", "owner_sha256", "nonsemantic_packet_sha256", "collision", "done")
RAW_SENSOR_FIELDS = ("depth_values_sha256", "rgb_values_sha256", "scan_ranges_sha256")


def require(passed, reason):
    if not passed:
        raise ValueError(reason)


def project(step, fields):
    return {name: step[name] for name in fields}


def first_shared_paid_difference(g, s, start_paid, fields):
    end_paid = min(len(g), len(s))
    for index in range(start_paid, end_paid):
        a, b = g[index], s[index]
        require(a["step"] == b["step"] == index + 1, "Pair has lost the same paid clock")
        difference = [key for key in fields if a.get(key) != b.get(key)]
        if difference:
            return {"paid": index + 1, "different_fields": difference,
                    "G": {key: a.get(key) for key in fields}, "S": {key: b.get(key) for key in fields}}
    return None


def first_actual_reach(trace, witness, exact_option):
    goal = witness["goal_pose"]
    paid_after = witness["paid_before"]
    option = witness["native_option_id"]
    return next((row["step"] for row in trace[paid_after:]
                 if not row["collision"] and [*row["position"], row["heading"]] == goal
                 and (not exact_option or row["active_option_id"] == option)), None)


def original_prefix(layout, reference, witness):
    result = {}
    catalog = files.read_json(DEVELOPMENT / "EVIDENCE_INDEX.json")
    published = {row["manifest"]: row for row in catalog["run_files_rooted_at_each_manifest"]}
    for mode in ARMS:
        relative = f"runner/runs/development/{layout}/new{mode}/manifest.json"
        directory = (DEVELOPMENT / relative).parent
        manifest, digest = engine.verify_manifest(directory, "manifest.json", published[relative]["manifest_sha256"])
        require(manifest["files_sha256"] == published[relative]["files_sha256"],
                f"Published same-goal source record changed at {layout}/{mode}")
        trace = files.read_json(directory / "steps.json")
        development.verify_trace(trace, reference["initial"], CAP, manifest)
        records = files.read_json(directory / "global_choices.json")
        calls = [record for record in records if record["call_id"] == witness["global_call_id"]]
        require(len(calls) == 1 and calls[0]["action_id"] == witness["paid_before"]
                and calls[0]["outputs"]["selected"]["option_id"] == witness["native_option_id"]
                and calls[0]["outputs"]["selected"]["pose"] == witness["goal_pose"],
                f"Ambiguous archived global selection for {layout}/{mode}")
        result[mode] = {"trace": trace, "global_call": calls[0], "manifest": manifest, "manifest_sha256": digest}
    original_paid = witness["paid_before"]
    require(all(project(a, TRACE_CLOCK_FIELDS) == project(b, TRACE_CLOCK_FIELDS)
                for a, b in zip(result["G"]["trace"][:original_paid], result["S"]["trace"][:original_paid], strict=True)),
            f"Witness has no exact common historical observation prefix in {layout}")
    require(result["G"]["global_call"] == result["S"]["global_call"],
            f"Witness archive global goal selection differs within {layout}")
    return result


def verify_historical_prefix(trace, source, witness):
    n = witness["paid_before"]
    require(len(trace) >= n, f"Run ended before archived intervention {witness['layout']}")
    for arm in ARMS:
        for candidate, archived in zip(trace[:n], source[arm]["trace"][:n], strict=True):
            require(all(candidate[key] == value for key, value in archived.items() if key != "decision_wall_s"),
                    f"Replay differs from source {arm} at paid step {candidate['step']}")
    require(trace[n - 1]["packet_sha256"] == witness["packet_sha256"]
            and [*trace[n - 1]["position"], trace[n - 1]["heading"]] == witness["observed_pose"],
            f"Historical prefix ended at different packet/pose for {witness['layout']}")


def path_execution(trace, witness, mode):
    route = witness["nominal_winning_routes"][mode]
    n = witness["paid_before"]
    length = len(route["outbound_actions"])
    matched = 0
    stop = None
    for index, step in enumerate(trace[n:n + length]):
        planned_pose = route["outbound_states"][index + 1]
        actual_pose = [*step["position"], step["heading"]]
        if (step["action"] != route["outbound_actions"][index] or actual_pose != planned_pose
                or step["active_option_id"] != witness["native_option_id"]
                or step["selection_call_id"] != witness["global_call_id"]
                or step["active_outbound_sha256"] != route["outbound_sha256"]
                or step["local_route_id"] != route["route_id"] or step["collision"] or not step["guard_allowed"]):
            stop = {"paid": step["step"], "actual_action": step["action"], "actual_pose": actual_pose,
                    "actual_option_id": step["active_option_id"], "planned_action": route["outbound_actions"][index],
                    "planned_pose": planned_pose}
            break
        matched += 1
    return {
        "route_id": route["route_id"], "nominal_paid_outbound": length, "actual_matching_paid_actions": matched,
        "full_plan_executed": matched == length, "deviation": stop,
        "first_goal_paid_same_option": first_actual_reach(trace, witness, exact_option=True),
        "first_goal_paid_any_option": first_actual_reach(trace, witness, exact_option=False),
    }


def metric_checkpoint(directory, name, manifest, initial, trace, reference, metric, witness):
    receipt, paths, _, digest = engine.snapshot(directory, name, manifest, "native_observed_checkpoint")
    if receipt["status"] == "unavailable":
        return receipt | {"source_checkpoint_sha256": digest}, None
    expected = {"prefix": witness["paid_before"], "start": 0, "arrival": "arrival", "terminal": "terminal"}
    expected = expected[name] if name in expected else int(name.removeprefix("paid_"))
    engine.verify_checkpoint_against_trace(receipt, expected, trace, initial)
    measured, masks = engine.evaluate_verified_snapshot(
        paths["native_mesh.ply"], paths["belief.npz"], receipt, reference, metric,
    )
    return receipt | measured | {"source_checkpoint_sha256": digest}, masks


def event_at(tape, stage, option, **matching):
    selected = [
        (index, event) for index, event in enumerate(tape)
        if event["event"] == stage and all(event.get(key) == value for key, value in matching.items())
        and (option is None or (event.get("option") or {}).get("option_id", event.get("option_id")) == option)
    ]
    require(len(selected) == 1, f"Runtime did not retain one {stage} event: {matching} {option}")
    return selected[0]


def verify_warehouse_order(audit, chosen, witness, prior_guard):
    n = witness["paid_before"]
    call = witness["global_call_id"]
    new = witness["native_option_id"]
    old_option = f"{new.rsplit('-', 1)[0]}-4"
    marker = f"frame-{n}"
    prior_call = prior_guard.get("call", prior_guard)
    guard_output = prior_call["outputs"]
    require(prior_call["method"] == "assess_local_action" and prior_call["call_id"] < call
            and prior_call["action_id"] == n and not guard_output["allowed"]
            and guard_output["reason"] == "next_footprint_not_known_safe",
            "Warehouse branch did not preserve the original denied safety call before selection")
    require(prior_guard["executed_option_id"] == old_option,
            "Warehouse prior denial refers to a different option")
    observation_index, _ = event_at(audit, "observation", None, frame_id=marker, action_id=n)
    global_index, global_event = event_at(audit, "global_choice", new, frame_id=marker)
    replan_index, replan = event_at(audit, "exploration_joint_replan_after_rejection", None, frame_id=marker,
                                   rejected_option_id=old_option, selected_option_id=new)
    require(global_event["option"]["pose"] == witness["goal_pose"]
            and replan["rejected_option_id"] == old_option and replan["selected_option_id"] == new
            and replan["reason"] == guard_output["reason"]
            and observation_index < global_index,
            "Warehouse runtime replan record changed the common native target")
    require(replan_index >= global_index, "Warehouse replan summary should be emitted after choosing its replacement")
    next_authorized = [(index, row) for index, row in enumerate(audit) if row["event"] == "paid_action_authorized"
                       and row["next_action_id"] == n + 1]
    require((not next_authorized and chosen is None)
            or (len(next_authorized) == 1 and chosen is not None and replan_index < next_authorized[0][0]),
            "Warehouse command was issued before its original native guard and replan, or missing from trace")
    return {"verified": True, "denied_guard_call": prior_call["call_id"], "common_global_call": call,
            "charged_action_id": n + 1 if chosen is not None else None,
            "applied_option_id": chosen.get("active_option_id") if chosen is not None else None,
            "rejected_prior_option_id": old_option, "guard_reason": replan["reason"],
            "replan_runtime_summary_index": replan_index,
            "runtime_authorization_index": next_authorized[0][0] if next_authorized else None}


def action_lineage(run, witness, paid):
    trace = run["trace"]
    row = trace[paid - 1]
    pose = [*row["position"], row["heading"]]
    auth_index, auth = event_at(run["audit"], "paid_action_authorized", row["active_option_id"], next_action_id=paid)
    guard = auth["assessment"]
    require(guard["allowed"] and guard["action"] == row["action"]
            and (row["collision"] or guard["next_pose"] == pose) and auth["selection_call_id"] == row["selection_call_id"],
            f"Native guard-authorized motion does not match measured state at paid {paid}")
    witness_choice = witness["native_option_id"]
    native_replans = [
        event for event in run["audit"][:auth_index]
        if event["event"] == "exploration_joint_replan_after_rejection"
        and event.get("rejected_option_id") == witness_choice
        and witness["paid_before"] <= int(event["frame_id"].removeprefix("frame-")) < paid
    ]
    active_witness = (row["active_option_id"] == witness_choice
                      and run["execution"]["actual_matching_paid_actions"] >= paid - witness["paid_before"])
    return {
        "authorized": True, "active_bound_witness": active_witness, "native_replans_from_witness": native_replans,
        "linked_to_treatment": active_witness or bool(native_replans), "actual_option_id": row["active_option_id"],
        "runtime_authorization_index": auth_index, "charged_action_id": paid, "observed_pose_after": pose,
        "collision": row["collision"], "guard_predicted_pose": guard["next_pose"],
    }


def pair_evidence(layout, witness, protocol_pair, pair):
    row = {"layout": layout, "state_id": witness["state_id"], "historical_prefix_paid": witness["paid_before"],
           "verified_actual_split": False, "status": "both_runs_measured"}
    if any(not pair[arm] for arm in ARMS):
        return row | {"status": "failed_or_interrupted_arm"}
    g, s = (pair[arm] for arm in ARMS)
    n = witness["paid_before"]
    same_prefix_mesh = development.exact_common_observation(g["prefix"], s["prefix"])
    require(same_prefix_mesh, f"{layout} arms do not share the same measured prefix state")
    require(g["effective_runtime"] == s["effective_runtime"], "Arms changed an original runtime setting")
    require(g["rng"] == s["rng"] and not g["rng"]["stateful_generator_retained_by_simulator"],
            "Arms do not share the same stateless simulator sensing inputs")
    commands, poses, packets, device = (
        first_shared_paid_difference(g["trace"], s["trace"], n, fields)
        for fields in (("action",), ("position", "heading"), ("packet_sha256", "nonsemantic_packet_sha256"),
                       RAW_SENSOR_FIELDS)
    )
    expected = protocol_pair["expected_first_nominal_split"]
    expected_first_step = expected["paid"]
    before = expected_first_step - 1
    actual_shared_observations = len(g["trace"]) >= before and len(s["trace"]) >= before and (
        first_shared_paid_difference(g["trace"][:before], s["trace"][:before], n, TRACE_CLOCK_FIELDS) is None
    )
    if layout == "gallery_left":
        require(before - n == protocol_pair["verify_common_first_outbound_observations"],
                "Frozen gallery planned split no longer corresponds to nine common prefix observations")
    actual_action_paid = commands["paid"] if commands is not None else None
    common_before_actual = actual_action_paid is not None and (
        first_shared_paid_difference(g["trace"][:actual_action_paid - 1], s["trace"][:actual_action_paid - 1], n,
                                     (*TRACE_CLOCK_FIELDS, *RAW_SENSOR_FIELDS)) is None
    )
    kept_through_actual_difference = commands is not None and all(
        pair[arm]["execution"]["actual_matching_paid_actions"] >= commands["paid"] - n for arm in ARMS
    )
    same_guard_order = all(pair[arm]["guard_order"].get("verified", False) for arm in ARMS) if layout == "warehouse_right" else None
    lineage = {arm: action_lineage(pair[arm], witness, actual_action_paid) for arm in ARMS} if commands is not None else {}
    has_actual_motion_difference = commands is not None and poses is not None and poses["paid"] == actual_action_paid
    actual_proof = (has_actual_motion_difference and common_before_actual
                    and all(lineage[arm]["linked_to_treatment"] and not lineage[arm]["collision"] for arm in ARMS)
                    and (layout != "warehouse_right" or same_guard_order))
    matches_nominal = (bool(actual_proof) and actual_action_paid == expected_first_step
                       and all(commands[arm]["action"] == expected[arm] for arm in ARMS)
                       and actual_shared_observations and kept_through_actual_difference)
    matched_paid = protocol_pair["equal_paid_checkpoint"]
    matched = {arm: pair[arm]["checkpoints"].get(f"paid_{matched_paid}") for arm in ARMS}
    terminal = {arm: pair[arm]["endpoints"] for arm in ARMS}
    def snapshot_comparison(points):
        complete = all(points[arm] and points[arm]["status"] == "captured" for arm in ARMS)
        visible = {
            arm: None if points[arm] is None else {
                k: points[arm].get(k) for k in ("status", "actual_paid", "requested_paid", "C_free", "F1_scene_05cm", "J")
            }
            for arm in ARMS
        }
        if complete:
            visible["delta_S_minus_G"] = {
                metric: points["S"][metric] - points["G"][metric] for metric in ("C_free", "F1_scene_05cm", "J")
            }
        return visible
    terminals = snapshot_comparison(terminal)
    for arm in ARMS:
        terminals[arm] |= {k: pair[arm]["closure"].get(k) for k in ("returned", "collisions", "stop_reason", "failed")}
    return row | {
        "verified_actual_split": bool(actual_proof), "same_measured_prefix": same_prefix_mesh,
        "matches_nominal_split": bool(matches_nominal), "same_rng_identity": True,
        "source_guard_order_verified": same_guard_order, "native_guard_order_G": g["guard_order"],
        "native_guard_order_S": s["guard_order"], "first_nominal_difference": expected,
        "first_command_guard_and_option_lineage": lineage, "common_before_first_actual_command_difference": common_before_actual,
        "motion_difference_attribution_unresolved": commands is not None and not actual_proof,
        "first_command_difference": commands, "first_pose_difference": poses, "first_packet_difference": packets,
        "first_sensor_content_difference": device, "common_after_intervention_through_paid": (
            before if actual_shared_observations else None
        ),
        "recent_bound_trace_both_arms_at_divergence": kept_through_actual_difference,
        "route_execution_G": g["execution"], "route_execution_S": s["execution"],
        "matched_paid": matched_paid, "matched": snapshot_comparison(matched), "terminal": terminals,
    }


def read_witnesses(protocol, source_freeze):
    source = PREDECESSOR / "publication/receipt.json"
    require(files.read_json(source)["head"] == PREDECESSOR_COMMIT, "Wrong source offline witness publication")
    archive = files.read_json(WITNESS_TABLE)
    table = {row["layout"]: row for row in archive["nominal_category_witnesses"]}
    by_layout = {}
    for layout in protocol["layouts_in_order"]:
        config = protocol["pairs"][layout]
        path = PREDECESSOR / config["source_state"]
        run = path.parent.parent
        digest = files.verify_bound(path, source_freeze["files_sha256"])
        verified = files.read_json(run / "manifest.json")
        require(verified["files_sha256"][str(path.relative_to(run))] == digest
                and verified["scored_decision_states"] > 0 and not verified["new_trajectory_generated"],
                f"Witness does not belong to a sealed offline-only replay: {layout}")
        row = table[layout]
        raw = files.read_json(path)
        require(raw["state_id"] == row["state_id"] == config["state_id"]
                and raw["paid_before"] == row["paid_before"] == config["paid_before"]
                and raw["global_call_id"] == row["global_call_id"] == config["global_call_id"]
                and row["replayed_state_sha256"] == digest
                and raw["expanded"]["summary"]["expanded_winners_split_by_prior"]
                and raw["expanded"]["summary"]["expanded_hypothetical_winners"]
                == config["archived_expected_winners_for_verification_only"],
                f"Witness archive, independent candidate audit, and protocol disagree: {layout}")
        by_layout[layout] = (row, raw, path, digest)
    return by_layout


def verify_intervention(directory, trace, witness, raw, source, mode, witness_sha):
    treatment = files.read_json(directory / "intervention.json")
    n = witness["paid_before"]
    historical_before = trace[n - 1]
    route = witness["nominal_winning_routes"][mode]
    standard = {
        "status": "expanded_winner_bound_once", "layout": witness["layout"], "mode": mode,
        "paid_before": n, "global_call_id": witness["global_call_id"], "native_option_id": witness["native_option_id"],
        "native_target": witness["goal_pose"], "native_global_mode": "G", "source_commit": PREDECESSOR_COMMIT,
        "witness_sha256": witness_sha, "prior_source_state_id": witness["state_id"],
        "source_G_SHA": raw["original_source_G_SHA"], "source_S_SHA": raw["original_source_S_SHA"],
        "prior_prefix_reproduced_before_selection": True, "expanded_pool_and_scores_equal_to_offline_source": True,
        "native_selected_option_unchanged": True, "original_neutral_alternate_cap": 4,
        "chosen_route_id": route["route_id"], "selected_outbound_actions": route["outbound_actions"],
        "selected_outbound_states": route["outbound_states"], "expanded_candidate_pool_sha256": witness["expanded_pool_sha256"],
        "expanded_score_table_sha256": witness["expanded_scores_sha256"], "forced_goal": False, "forced_route": False,
        "next_local_decisions": "original_neutral_G_cap4",
    }
    require(all(treatment[key] == value for key, value in standard.items()),
            f"Actual treatment changed the original one-choice plan or its measured state: {directory}")
    restored = treatment["measured_prefix"]
    require(restored["pose"] == witness["observed_pose"] and restored["packet_sha256"] == witness["packet_sha256"]
            and restored["belief_sha256"] == witness["belief_sha256"]
            and restored["packet_chain_sha256"] == historical_before["packet_chain_sha256"],
            f"Intervention was ranked on a different observed prefix: {directory}")
    native = raw["original_native_option"]
    require(treatment["original_return_actions"] == treatment["actual_return_actions"] == native["return_actions"]
            and treatment["actual_enlarged_alternate_cap"] > treatment["exact_deduplicated_proposal_count"]
            and len(route["outbound_actions"]) + len(native["return_actions"]) <= CAP - n,
            f"Outward choice changed native return, cap scope, or budget: {directory}")
    event_stream = files.read_json(directory / "local_choices.json")
    selected = [event for event in event_stream if event["intervention"]]
    require(len(selected) == 1 and all(evt["local_mode"] == ("G" if not evt["intervention"] else mode)
                                      and evt["global_mode"] == "G" for evt in event_stream),
            f"Actual runtime intervened more than once or changed the neutral continuation: {directory}")
    first = selected[0]
    choice = first["choice_audit"]
    decorations = {"original_pool", "skipped_by_original_cap", "proposal_index_zero_based", "original_rank", "expanded_rank"}
    archived = [{k: v for k, v in candidate.items() if k not in decorations} for candidate in raw["expanded"]["candidates"]]
    require(first["paid_before"] == n and first["selection_call_id"] == witness["global_call_id"]
            and first["same_global_target"] == witness["goal_pose"]
            and first["rebound_return_states"] == first["native_return_states"] == native["return_states"]
            and choice["selected_mode"] == mode and choice["selected_route_id"] == route["route_id"]
            and choice["chosen_outbound_sha256"] == route["outbound_sha256"]
            and choice["candidates"] == archived
            and choice["candidate_pool_sha256"] == raw["expanded"]["summary"]["expanded_candidate_pool_sha256"]
            and choice["score_table_sha256"] == raw["expanded"]["summary"]["expanded_score_table_sha256"],
            f"Expanded modes did not score the same original-bound route alternatives: {directory}")
    globals_ = files.read_json(directory / "global_choices.json")
    matches = [ev for ev in globals_ if ev["call_id"] == witness["global_call_id"]]
    require(len(matches) == 1 and matches[0] == source["G"]["global_call"] == source["S"]["global_call"]
            and all(ev["outputs"]["mode"] == "G" for ev in globals_),
            f"Shared native global decision does not match the frozen archived choice: {directory}")
    first_action = trace[n] if len(trace) > n else None
    post_calls = treatment["post_call_native_guard"]
    post = None
    if first_action is not None:
        require(bool(post_calls), f"First charged action has no prior native assessment: {directory}")
        post = post_calls[-1]
        require(post["call_id"] > witness["global_call_id"] and post["action_id"] == n
                and post["outputs"]["allowed"] and post["outputs"]["action"] == first_action["action"]
                == treatment["first_charged_action_after_witness"],
                f"First released action was not recorded with native guard approval: {directory}")
    runtime_audit = files.read_json(directory / "audit.json")
    if witness["layout"] == "warehouse_right":
        denied = treatment["pre_call_native_guard"]
        require(denied and not denied[-1]["outputs"]["allowed"], f"Prior native denial is missing: {directory}")
        order = verify_warehouse_order(runtime_audit, first_action, witness, denied[-1])
    else:
        require(not treatment["pre_call_native_guard"], "Gallery recovery unexpectedly reached old guard branch")
        order = {"verified": None, "native_trigger": "earlier_committed_goal_arrival"}
    order["authorized_guard_call"] = post["call_id"] if post is not None else None
    order["actual_first_action"] = post["outputs"]["action"] if post is not None else None
    return treatment, order


def verify_one_run(layout, mode, protocol_pair, restored, source, reference, metric, bound):
    witness, raw, witness_path, witness_sha = restored
    directory = STUDY / "runner/runs" / layout / mode
    if (directory / "failure.json").exists() and not (directory / "manifest.json").exists():
        path = directory / "failure.json"
        failure = files.read_json(path)
        return {"layout": layout, "mode": mode, "status": "failed", "error": failure["error"], "failed": True,
                "source_manifest_sha256": files.sha256(path)}, [], [], None
    manifest, run_hash = engine.verify_manifest(directory, "manifest.json")
    n = witness["paid_before"]
    require(manifest["source_protocol_sha256"] == bound["protocol_sha"]
            and manifest["runner_freeze_sha256"] == bound["runner_sha"] and manifest["source_commit"] == PREDECESSOR_COMMIT
            and manifest["status"] == "completed_execution" and manifest["layout"] == layout and manifest["arm"] == mode
            and manifest["actual_closed_loop"] and not manifest["evaluator_inputs_read_by_actor"]
            and manifest["native_budget"] == CAP and not manifest["forced_goal"] and not manifest["forced_route"]
            and manifest["global_mode"] == manifest["subsequent_local_mode"] == "G"
            and manifest["subsequent_local_alternate_cap"] == 4
            and manifest["single_intervention_call_id"] == witness["global_call_id"]
            and manifest["single_intervention_paid_before"] == manifest["replayed_archived_prefix_actions"] == n,
            f"Actual run changed predeclared arm, policy, checkpoint scope, or budget: {directory}")
    trace = files.read_json(directory / "steps.json")
    terminal_chain = development.verify_trace(trace, reference["initial"], CAP, manifest)
    development.verify_action_guard(trace, CAP)
    verify_historical_prefix(trace, source, witness)
    require(len(trace) == n + manifest["actual_new_actions"] and all(
        row["charged_new_continuation_action"] == (row["step"] > n) for row in trace
    ), "Archived prefix actions were classified as newly executed")
    require(all(row.get(field) for row in trace for field in RAW_SENSOR_FIELDS),
            "Sensor content hashes missing from actual RGB-D/range trace")
    treatment, guard_order = verify_intervention(directory, trace, witness, raw, source, mode, witness_sha)
    require(Path(treatment["witness_file"]) == witness_path, "Treatment chose a different archived state file")
    rng = manifest["restored_rng_identity"]
    require(rng == treatment["restored_rng_identity"] and rng["scene_seed"] == 20261121
            and rng["source_packet_chain_required_through_paid"] == n and rng["deterministic_pose_geometry"]
            and rng["depth_noise_sigma_m"] == rng["depth_dropout_fraction"] == 0.
            and not rng["stateful_generator_retained_by_simulator"],
            "Current run did not reconstruct the frozen independent sensing inputs")
    prefix, base_masks = metric_checkpoint(directory, "prefix", manifest, reference["initial"], trace, reference, metric, witness)
    require(prefix["belief_sha256"] == witness["belief_sha256"], "Prefix saved map differs from witnessed observed state")
    expect = {"start", "prefix", "terminal", "arrival", *(
        f"paid_{p}" for p in protocol_pair["checkpoints"] if p != n
    )}
    require(set(manifest["checkpoints"]) == expect, "Measured clocks differ from the protocol")
    rows, checkpoints = [], {}
    order = ["start", "prefix", *(f"paid_{p}" for p in protocol_pair["checkpoints"] if p != n), "arrival", "terminal"]
    for name in order:
        if name == "prefix":
            receipt, masks = prefix, base_masks
        else:
            receipt, masks = metric_checkpoint(directory, name, manifest, reference["initial"], trace, reference, metric, witness)
        if receipt["status"] == "captured":
            receipt = engine.augment_measurement(receipt, masks, prefix, base_masks, reference)
        checkpoints[name] = receipt
        rows.append({"layout": layout, "mode": mode, "checkpoint": name, **receipt})
    terminal = checkpoints["terminal"]
    runtime = files.read_json(directory / "runtime_summary.json")
    closure = development.close_record(terminal, runtime, trace, reference["initial"])
    require(terminal["packet_chain_sha256"] == terminal_chain and closure["returned"] == manifest["returned"]
            and closure["collisions"] == manifest["collisions"] and closure["terminal_pose"] == manifest["final_pose"]
            and closure["termination"] == manifest["observed_native_termination"],
            f"Actual native closure differs from runtime execution record: {directory}")
    execution = path_execution(trace, witness, mode)
    measured_arrival = checkpoints["arrival"]["actual_paid"] if checkpoints["arrival"]["status"] == "captured" else None
    require(measured_arrival == execution["first_goal_paid_same_option"]
            == manifest["actual_witness_arrival_paid"] == treatment["actual_option_arrival_paid"],
            f"Nominal arrival was substituted for actual arrival: {directory}")
    curve = engine.per_action_curve(
        {"belief_trace": str(directory / "observed_belief_trace.npz")}, trace, rows, reference,
    )
    total = terminal["reachable_cells"]
    coverage = [
        {"layout": layout, "mode": mode, "paid": paid, "correct_free_cells": free, "reachable_cells": total,
         "C_free": free / total, "gross_correct_free_cells": gross, "lost_correct_free_cells": lost}
        for paid, free, gross, lost in curve
    ]
    result = {
        "layout": layout, "mode": mode, "status": manifest["status"], "state_id": witness["state_id"],
        "source_manifest_sha256": run_hash, "original_archived_prefix": n, "actual_new_actions": manifest["actual_new_actions"],
        "chosen_route_id": treatment["chosen_route_id"], "nominal_arrival_paid": n + execution["nominal_paid_outbound"],
        "goal_paid_same_option": measured_arrival, "goal_paid_any_option": execution["first_goal_paid_any_option"],
        "matched_witness_paid_actions": execution["actual_matching_paid_actions"],
        "full_witness_route_executed": execution["full_plan_executed"],
        "witness_path_deviation": execution["deviation"],
        **{k: terminal[k] for k in METRICS}, **closure,
    }
    return result, rows, coverage, {
        "prefix": prefix, "trace": trace, "effective_runtime": files.read_json(directory / "effective_runtime.json"),
        "guard_order": guard_order, "execution": execution, "checkpoints": checkpoints,
        "rng": rng, "audit": files.read_json(directory / "audit.json"),
        "endpoints": terminal, "closure": closure,
    }


def engineering_gate(pairs, outcomes, protocol):
    checkpoints = protocol["engineering_gate"]
    threshold = checkpoints["required_terminal_J_gain_strictly_greater_than_fraction"]
    individual = {}
    for row in pairs:
        terminal = row.get("terminal", {})
        changes = terminal.get("delta_S_minus_G")
        individual[row["layout"]] = bool(changes is not None and changes["J"] > threshold)
    safe = len(outcomes) == protocol["maximum_continuations"] and all(
        row["status"] == "completed_execution" and row["actual_paid"] <= CAP and not row["failed"]
        and row["returned"] and row["collisions"] == 0 for row in outcomes
    )
    actual = sum(row["verified_actual_split"] for row in pairs)
    nominal = sum(row.get("matches_nominal_split", False) for row in pairs)
    return {
        "pass": safe and actual == checkpoints["required_actual_split_pairs"] and all(individual.values()),
        "verified_actual_split_pairs": actual, "matched_exact_nominal_split_pairs": nominal,
        "all_actual_continuations_safe_returned": safe, "J_terminal_comparison_pass_by_layout": individual,
        "positive_J_threshold_fraction_strict": threshold, "holdout_runs": 0, "from_zero_validation_run": False,
    }


def pair_csv(row):
    details = {"layout": row["layout"], "state_id": row["state_id"], "status": row["status"],
               "verified_actual_split": row["verified_actual_split"],
               "matches_nominal_split": row.get("matches_nominal_split"),
               "historical_prefix_paid": row["historical_prefix_paid"], "matched_paid": row.get("matched_paid"),
               "source_guard_order_verified": row.get("source_guard_order_verified")}
    for key in ("first_command_difference", "first_pose_difference", "first_sensor_content_difference"):
        item = row.get(key)
        details[key + "_paid"] = None if item is None else item["paid"]
        if item is not None and key == "first_command_difference":
            for mode in ARMS:
                details[f"first_command_{mode}"] = item[mode]["action"]
    for when in ("matched", "terminal"):
        if when not in row:
            continue
        data = row[when]
        for mode in ARMS:
            checkpoint = data[mode]
            if checkpoint:
                details.update({f"{when}_{mode}_{k}": v for k, v in checkpoint.items()})
        for metric, delta in data.get("delta_S_minus_G", {}).items():
            details[f"{when}_S_minus_G_{metric}_fraction"] = delta
            details[f"{when}_S_minus_G_{metric}_pp"] = 100.0 * delta
    return details


def main(output):
    protocol_path = STUDY / "PROTOCOL.json"
    source_freeze_path = STUDY / "SOURCE_FREEZE.json"
    runner_freeze_path = STUDY / "runner/freeze.json"
    protocol = files.read_json(protocol_path)
    source_freeze = files.read_json(source_freeze_path)
    runner_freeze = files.read_json(runner_freeze_path)
    require(protocol["frozen_before_any_new_trajectory"]
            and protocol["source_commit"] == source_freeze["source_commit"] == PREDECESSOR_COMMIT
            and protocol["total_paid_action_budget"] == CAP and protocol["modes"] == list(ARMS)
            and protocol["maximum_continuations"] == len(ARMS) * len(protocol["layouts_in_order"]),
            "Invalid frozen layout/arm/time-budget protocol")
    for path, digest in source_freeze["files_sha256"].items():
        files.verify(path, digest)
    for path, digest in runner_freeze["files_sha256"].items():
        files.verify(path, digest)
    bound = {"protocol_sha": files.sha256(protocol_path), "runner_sha": files.sha256(runner_freeze_path)}
    restored = read_witnesses(protocol, source_freeze)
    old_scene = files.read_json(files.SCENES / "freeze.json")
    old_runner = files.read_json(files.HISTORY / "freeze.json")
    old_eval = files.read_json(files.OLD_EVAL / "freeze.json")
    metric = files.load_metric(old_scene, old_eval)
    files.verify_bound(files.NATIVE_EXECUTION_FREEZE, old_runner["files_sha256"])
    recon_sha = files.verify_bound(files.RECON_METRIC, files.read_json(files.NATIVE_EXECUTION_FREEZE)["files_sha256"])

    episodes, checkpoints, coverage, causal = [], [], [], []
    bindings = {}
    for layout in protocol["layouts_in_order"]:
        reference = files.load_reference(layout, old_scene, old_runner)
        config = protocol["pairs"][layout]
        witness = restored[layout][0]
        source = original_prefix(layout, reference, witness)
        bindings[layout] = {"reference_inputs_sha256": reference["files_sha256"],
                            "original_manifest_sha256": {arm: source[arm]["manifest_sha256"] for arm in ARMS},
                            "witness_sha256": restored[layout][-1]}
        pair = {}
        for arm in ARMS:
            run, assessed, curve, origin = verify_one_run(
                layout, arm, config, restored[layout], source, reference, metric, bound,
            )
            episodes.append(run)
            checkpoints.extend(assessed)
            coverage.extend(curve)
            pair[arm] = origin
        causal.append(pair_evidence(layout, witness, config, pair))
    gate = engineering_gate(causal, episodes, protocol)
    metadata = {
        "status": "measured_read_only", "stage": protocol["stage"], "source_commit": PREDECESSOR_COMMIT,
        "protocol_sha256": bound["protocol_sha"], "source_freeze_sha256": files.sha256(source_freeze_path),
        "runner_freeze_sha256": bound["runner_sha"], "evaluator_sha256": files.sha256(__file__),
        "original_native_metric_sha256": recon_sha, "layout_source_bindings": bindings,
        "metric_start_is_historical_prefix": True, "measured_without_sensor_interpolation": True,
        "persistence_of_rng_state": "No persistent simulator generator to restore. Source seed, measured pose/clock, "
                                    "zero depth noise/dropout and archived packet chain verified.",
    }
    output.mkdir(parents=True, exist_ok=True)
    development.write_tables(output, "episodes", episodes, metadata, (
        "layout", "mode", "status", "original_archived_prefix", "actual_new_actions", "actual_paid",
        "returned", "collisions", "failed", "stop_reason", "path_length_m", "chosen_route_id", "nominal_arrival_paid",
        "goal_paid_same_option", "goal_paid_any_option", "matched_witness_paid_actions", "full_witness_route_executed",
        *METRICS, "source_manifest_sha256",
    ))
    development.write_tables(output, "checkpoints", checkpoints, metadata, (
        "layout", "mode", "checkpoint", "status", "actual_paid", "requested_paid", "outcome", *METRICS,
        "source_checkpoint_sha256",
    ))
    development.write_tables(output, "coverage_curve", coverage, metadata)
    flat = list(map(pair_csv, causal))
    fields = list(dict.fromkeys(key for row in flat for key in row))
    files.write_csv(output / "pair_causal_audit.csv", [{key: row.get(key) for key in fields} for row in flat], fields)
    files.write_json(output / "pair_causal_audit.json", metadata | {"engineering_gate": gate, "rows": causal})
    files.write_json(output / "engineering_gate.json", metadata | {
        "engineering_gate": gate, "input_evidence_sha256": {name: files.sha256(output / name) for name in (
            "episodes.json", "checkpoints.json", "pair_causal_audit.json",
        )},
    })
    print(f"Verified {gate['verified_actual_split_pairs']} actual divergence pairs; engineering gate={gate['pass']}.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=STUDY / "measurement/results")
    main(parser.parse_args().out)
