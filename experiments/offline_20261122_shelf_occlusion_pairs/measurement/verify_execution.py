"""Independently bind the observed fork, selected route and executed checkpoints."""

import hashlib
from math import isclose
from pathlib import Path

import numpy as np

from metrics import ARMS, shelf_at_clock, snap_key


CLOCKS = (48, 96, 144, 192)
SENSORS = {"depth_m": "depth_values_sha256", "color_rgb": "rgb_values_sha256", "scan_ranges_m": "scan_ranges_sha256"}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def raw_sha(array):
    return hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest()


def verify_bank(directory, meta, initial, trace, raw, events, bank, trigger, check, weighted):
    freeze, spec = meta["freeze_sha"], meta["spec"]
    digest = weighted.native.canonical_checksum
    require(bank["freeze_sha256"] == freeze and bank["layout"] == initial["layout"],
            f"Wrong registered observed fork {directory}")
    fork_paid = bank["paid_before"]
    observed_to = min(len(trace), spec["max_trigger_paid"]) if fork_paid is None else fork_paid
    prefix_path = directory / "common_prefix.npz"
    require(check.sha(prefix_path) == bank["prefix_observation_sha256"], f"Prefix frame bytes drift: {directory}")
    rows = [raw, *trace[:observed_to]]
    with np.load(prefix_path, allow_pickle=False) as prefix:
        require(np.array_equal(prefix["paid"], np.arange(observed_to + 1)),
                f"Prefix frames have wrong clocks: {directory}")
        poses = [initial["anchor"]] + [[*r["position"], r["heading"]] for r in trace[:observed_to]]
        require(prefix["poses"].tolist() == poses, f"Stored prefix poses differ from actual trace: {directory}")
        for name, field in SENSORS.items():
            require(all(expected[field] == raw_sha(array) for expected, array in zip(rows, prefix[name], strict=True)),
                    f"Raw {name} frame does not match executed prefix: {directory}")
        pixels = np.cumsum(np.count_nonzero(
            (prefix["semantic"] == 2) & (prefix["depth_m"] > 0),
            axis=tuple(range(1, prefix["semantic"].ndim)),
        ))
        image = np.asarray(prefix["color_rgb"][-1]).copy()
    probes = check.read(directory / "trigger_probes.json")
    expected_events = [e for e in events if e["paid_before"] <= observed_to
                       and (fork_paid is None or e["selection_call_id"] <= bank["selection_call_id"])]
    require(len(probes) == len(expected_events), f"Not every eligible-time native selection was probed: {directory}")
    for probe, event in zip(probes, expected_events, strict=True):
        decision = event["choice_audit"]
        paid = event["paid_before"]
        admitted = {r["outbound_sha256"] for r in decision["candidates"] if r["admitted"]}
        valid = bool(pixels[paid] > 0 and len(admitted) >= 2)
        require(probe["paid"] == paid and probe["selection_call_id"] == event["selection_call_id"]
                and probe["seen_valid_shelf_depth_pixels"] == pixels[paid]
                and probe["admitted_distinct_routes"] == len(admitted) and probe["qualified"] == valid
                and probe["candidate_pool_sha256"] == decision["candidate_pool_sha256"],
                f"Fork eligibility is not reproducible from measured labels and legal route pool: {directory} paid={paid}")
    allowed = [p for p in probes if p["qualified"]]
    applied = [e for e in events if "paired_first_decision_arm" in e["choice_audit"]]
    require(bank["status"] == trigger["status"], f"Trigger and fork bank disagree: {directory}")
    if bank["status"] == "not_triggered":
        require(not allowed and not applied and not bank["arms"] and fork_paid is None,
                f"A registered fork was missed or leaked to a no-trigger run: {directory}")
        expected_chain = (rows[-1]["packet_chain_sha256"] if observed_to else
                          weighted.files.chain_sha256("0" * 64, initial["initial_packet_sha256"]))
        require(bank["observed_prefix_packet_chain_sha256"] == expected_chain,
                f"No-trigger prefix packets are not bound to the native trace: {directory}")
        return image, None, None

    require(bank["status"] == "triggered" and 0 <= fork_paid <= spec["max_trigger_paid"]
            and len(allowed) == len(applied) == 1 and allowed[0] == probes[-1],
            f"Run does not have exactly the first qualified fork: {directory}")
    event = applied[0]
    chosen = event["choice_audit"]
    actual_arm = trigger["arm"]
    require(allowed[0]["paid"] == fork_paid == event["paid_before"] == trigger["paid_before"]
            and allowed[0]["selection_call_id"] == event["selection_call_id"] == bank["selection_call_id"]
            and actual_arm == chosen["paired_first_decision_arm"],
            f"Applied mode or clock differs from observed fork: {directory}")
    require(bank["fork_packet_sha256"] == event["packet_sha256"] == rows[-1]["packet_sha256"]
            and bank["remaining_budget"] == spec["native_budget"] - fork_paid
            and bank["initial_packet_sha256"] == raw["packet_sha256"]
            and bank["scene_seed"] == initial["seed"],
            f"Wrong native observed-state or action ledger at fork: {directory}")
    common = {key: value for key, value in bank.items()
              if key not in {"status", "freeze_sha256", "shared_identity_sha256", "arms",
                             "selected_route_hash_has_return", "elapsed_selection_parts_s"}}
    require(digest(common) == bank["shared_identity_sha256"] == trigger["shared_identity_sha256"],
            f"Observed fork identity hash cannot be recomputed: {directory}")
    audit = check.read(directory / "fork_candidates.json")
    table = audit["candidates"]
    original = next(row for row in table if row["route_id"] == chosen["original_route_id"])
    require(digest(table) == bank["candidate_scores_sha256"]
            and digest([{k: v for k, v in row.items() if k != "scores"} for row in table])
            == bank["candidate_pool_sha256"] == trigger["candidate_pool_sha256"],
            f"Candidate or score table changed after the observed fork: {directory}")
    require(event["same_global_target"] == bank["target_pose"] == trigger["target_pose"]
            and set(bank["arms"]) == set(ARMS),
            f"Frozen local score changed global target or missing selectable mode: {directory}")
    eligible = [row for row in table if row["admitted"]]
    by_route = {entry["route_id"]: entry for entry in eligible}
    hidden = {r["route_id"]:r["supported_hidden_m2"] for r in audit["supported_hidden_by_route"]}
    require(set(by_route)==set(hidden), f"Missing independent D+ supported area estimate: {directory}")
    cell_area = initial["grid_resolution_m"] ** 2
    params = spec["local_route"]
    for row in eligible:
        weighted.native.verify_candidate_nominal(row, original, bank["target_pose"],
                                                 params["max_extra_outbound_actions"],
                                                 params["min_unknown_union_ratio"])
        require(fork_paid + row["outbound_cost"] + row["reserved_return_cost"] <= spec["native_budget"],
                f"Possible local choice breaches the native reserved return: {directory}")
        require(isclose(row["scores"]["D+"] - row["scores"]["D"],
                        spec["hidden_scale"] * hidden[row["route_id"]], rel_tol=0, abs_tol=1e-12),
                f"D+ changes more than supported hidden-surface term: {directory}")
    for arm in ARMS:
        record = bank["arms"][arm]
        rankings = []
        for row in eligible:
            area = row["scores"][arm]
            change = (row["unknown_union_cells"] - original["unknown_union_cells"]) * cell_area
            extra = row["outbound_cost"] - original["outbound_cost"]
            utility = area + spec["weights"]["unknown_weight"] * change - spec["weights"]["action_cost"] * extra
            rankings.append(dict(route_id=row["route_id"], area_m2=area, delta_unknown_union_m2=change,
                                 extra_outbound_actions=extra, score_m2=utility))
        require(rankings == audit["rankings"][arm], f"{arm} score no longer matches frozen fixed weights: {directory}")
        best = max(row["score_m2"] for row in rankings)
        alternatives = [by_route[row["route_id"]] for row in rankings
                        if best - row["score_m2"] <= params["score_equality_tolerance"]]
        winner = min(alternatives, key=lambda row: (row["outbound_cost"], row["route_id"] != original["route_id"],
                                                   row["outbound_sha256"]))
        require(record["route_id"] == winner["route_id"] and record["outbound_sha256"] == winner["outbound_sha256"]
                and record["outbound_actions"] == winner["outbound_actions"]
                and record["outbound_states"] == winner["outbound_states"],
                f"{arm} did not autonomously pick its maximal eligible route: {directory}")
        expected = {key: record[key] for key in ("outbound_actions", "outbound_states", "return_actions", "return_states")}
        require(record["route_identity_sha256"] == digest(expected | {"oriented_target": bank["target_pose"]})
                and record["shared_identity_sha256"] == bank["shared_identity_sha256"]
                and record["common_continuation_id"] == bank["continuation_id"]
                and record["return_actions"] == event["native_return_actions"]
                and record["return_states"] == event["native_return_states"],
                f"{arm} did not retain the native certificate or common continuation: {directory}")
    route = bank["arms"][actual_arm]
    require(chosen["selected_route_id"] == trigger["actual_route_id"] == event["active_local_route_id"] == route["route_id"]
            and event["selected_outbound_actions"] == route["outbound_actions"]
            and event["selected_outbound_states"] == route["outbound_states"],
            f"Executable selected branch differs from its sealed route: {directory}")
    actions = 0
    for index, step in enumerate(trace[fork_paid:fork_paid + len(route["outbound_actions"])]):
        if step["active_option_id"] != trigger["option_id"]:
            break
        expect_pose = route["outbound_states"][index + 1]
        require(step["action"] == route["outbound_actions"][index] and step["guard_allowed"]
                and step["local_route_id"] == route["route_id"]
                and step["active_outbound_sha256"] == route["outbound_sha256"]
                and step["route_action_index_zero_based"] == index
                and (step["collision"] or [*step["position"], step["heading"]] == expect_pose),
                f"Bound flag differs from charged selected motion: {directory}, action {step['step']}")
        actions += 1
        if step["collision"] or step["matched_target_pose_after_step"]:
            break
    return image, actions, audit["rankings"]


def verified_run(directory, assigned, meta, initial, reference, check, weighted, metric, repeat):
    token = meta["freeze_sha"]
    manifest, manifest_sha = weighted.engine.verify_manifest(directory, "manifest.json", assigned["manifest_sha256"])
    attempt = check.read(directory / "attempt.json")
    physical_arm = assigned["execution_arm"]
    require(manifest["status"] == "completed_execution" and manifest["layout"] == initial["layout"]
            and manifest["phase"] == meta["phase"] and manifest["native_budget"] == meta["spec"]["native_budget"]
            and manifest["experimental_local_arm"] == physical_arm and manifest["base_commit"] == meta["spec"]["base_commit"]
            and manifest["global_mode"] == "G" and manifest["continuation_policy"] == "G_perf"
            and manifest["actual_closed_loop"] and not manifest["forced_route_or_goal"]
            and not manifest["evaluator_inputs_read_by_actor"]
            and manifest["initial_packet_sha256"] == attempt["initial_packet_sha256"] == initial["initial_packet_sha256"]
            and manifest["initial_owner_sha256"] == attempt["initial_owner_sha256"] == initial["initial_owner_sha256"]
            and manifest["frozen_actor_files_sha256"] == manifest["study_freeze_sha256"] == attempt["runner_freeze_sha256"] == token,
            f"Wrong sealed execution context: {directory}")
    raw = check.read(directory / "initial_raw_rgbd_scan_sha256.json")
    require(raw["step"] == 0 and raw["packet_sha256"] == initial["initial_packet_sha256"],
            f"Cold packet fingerprint differs: {directory}")
    trace = check.read(directory / "steps.json")
    weighted.native.verify_trace(trace, initial, meta["spec"]["native_budget"], manifest)
    weighted.native.verify_action_guard(trace, meta["spec"]["native_budget"])
    with np.load(directory / "observed_belief_trace.npz", allow_pickle=False) as data:
        require(np.array_equal(data["paid"], np.arange(len(trace) + 1)), f"Missing occupancy clock: {directory}")
        belief = data["belief"]
    bank_path = directory / "fork_bank.json"
    bank = check.read(bank_path)
    trigger = check.read(directory / "mechanism_trigger.json")
    require(check.sha(bank_path) == assigned["fork_bank_sha256"]
            and trigger == (manifest["applied_local_policy"] or {"status": "not_triggered", "arm": physical_arm, "paid_before": None}),
            f"Fork or selected assignment changed after sealing: {directory}")
    events = check.read(directory / "local_choices.json")
    globals_ = check.read(directory / "global_choices.json")
    by_id = {row["call_id"]:row for row in globals_}
    require(len(by_id) == len(globals_) == manifest["native_global_choices"]
            and len(events) == manifest["local_route_choices"]
            and all(e["outputs"]["mode"] == "G" for e in globals_),
            f"Recorded global decisions did not retain neutral G: {directory}")
    for event in events:
        global_choice = by_id[event["selection_call_id"]]["outputs"]["selected"]
        require(global_choice["pose"] == event["same_global_target"]
                and global_choice["return_actions"] == event["native_return_actions"] == event["rebound_return_actions"]
                and event["full_pool"]["enabled"] and event["full_pool"]["unassessed_after_cap"] == 0,
                f"Local decision changed the fixed goal, return, or candidate rules: {directory}")
        if "paired_first_decision_arm" not in event["choice_audit"]:
            ranking = event["choice_audit"]["weighted_ranking"]
            require(ranking["prior"] == "G" and all(ranking[k] == v for k, v in meta["spec"]["weights"].items())
                    and all(ranking["local_area_provider_calls"][arm] == 0 for arm in ("S", "X")),
                    f"An additional local decision or continuation departed from common weighted G: {directory}")
    image, actions, rankings = verify_bank(directory, meta, initial, trace, raw, events, bank, trigger, check, weighted)
    triggered = bank["status"] == "triggered"
    expected = {"start": 0, **({"fork": bank["paid_before"]} if triggered else {}),
                **{snap_key(c): c for c in CLOCKS}, "terminal": len(trace)}
    require(set(manifest["checkpoints"]) == set(expected), f"Wrong capture schedule: {directory}")
    snapshots, repeat_rows = {}, []
    runtime = dict(directory=directory, manifest=manifest, trace=trace, initial=initial, belief=belief)
    for name, paid in expected.items():
        recorded = manifest["checkpoints"][name]
        if name.startswith("paid_") and paid > len(trace):
            require(recorded == check.read(directory / name / "checkpoint.json")
                    and recorded == {"status":"unavailable", "actual_paid":len(trace), "requested_paid":paid,
                                     "outcome":"native_stop_before_checkpoint", "files_sha256":{}},
                    f"Later clock was imputed at early stop: {directory}/{name}")
            continue
        geometry, masks = check.verify_measurement(weighted, metric, reference, runtime, name, paid)
        require(geometry["receipt"] == recorded, f"Manifest and bound checkpoint record differ: {directory}/{name}")
        snapshots[name] = (geometry, masks)
        if repeat and name in {"start", "paid_96", "terminal"}:
            again, replayed = check.verify_measurement(weighted, metric, reference, runtime, name, paid)
            require(geometry["mesh_sha256"] == again["mesh_sha256"] and geometry["checkpoint_sha256"] == again["checkpoint_sha256"],
                    f"Repeatability observed different input snapshots: {directory}/{name}")
            start = snapshots["start"][1]["shelf"]
            weights = np.broadcast_to(np.asarray(reference["weights"]["shelf"], dtype=float), start.shape)
            first = shelf_at_clock(start, masks["shelf"], weights)["net_m2"]
            second = shelf_at_clock(start, replayed["shelf"], weights)["net_m2"]
            repeat_rows.append(dict(run=str(directory), checkpoint=name, actual_paid=paid,
                                    checkpoint_sha256=geometry["checkpoint_sha256"], first_net_m2=first,
                                    repeated_net_m2=second, exact_mask_match=np.array_equal(masks["shelf"], replayed["shelf"]),
                                    abs_delta_m2=abs(second-first)))
    terminal = snapshots["terminal"][0]["receipt"]
    closure = weighted.native.close_record(terminal, check.read(directory / "runtime_summary.json"), trace, initial)
    require(all(closure[k] == manifest[k] for k in ("returned", "collisions", "path_length_m")),
            f"Measured trajectory and native terminal outcome disagree: {directory}")
    if triggered:
        fork = snapshots["fork"][0]["receipt"]
        require(fork["belief_sha256"] == bank["belief_sha256"] and fork["native_mesh_fields"] == bank["mesh_fields"]
                and fork["packet_sha256"] == bank["fork_packet_sha256"]
                and fork["packet_chain_sha256"] == bank["prefix_packet_chain_sha256"],
                f"Fork decision and native fused geometry disagree: {directory}")
        route = manifest["route_event"]
        expected_paid = bank["paid_before"] + actions
        require(route["actual_paid"] == expected_paid and route["actions"] == actions
                and route["goal"] == bank["target_pose"], f"Filed route result disagrees with actual actions: {directory}")
        last = trace[expected_paid-1] if actions else None
        if route["status"] == "arrived":
            require(last is not None and not last["collision"] and last["matched_target_pose_after_step"]
                    and [*last["position"], last["heading"]] == route["goal"],
                    f"Filed arrival lacks actual goal pose: {directory}")
        elif route["status"] == "collision":
            require(last is not None and last["collision"], f"Filed collision lacks native trace: {directory}")
        else:
            require(route["status"] in {"guard_cancelled", "abandoned_or_replanned_before_arrival", "terminal_before_arrival"}
                    and (last is None or not (last["collision"] or last["matched_target_pose_after_step"])),
                    f"Incomplete route conflicts with the actual trace: {directory}")
    return {"directory": str(directory), "manifest_sha256": manifest_sha, "manifest":manifest,
            "snapshots": snapshots, "bank": bank, "raw_initial": raw, "trace": trace, "closure": closure,
            "fork_rgb": image if triggered else None, "actual_route_actions": actions, "repeat_rows":repeat_rows,
            "eligible_fork_rankings": rankings}
