#!/usr/bin/env python3
"""Restore a sealed S history and change exactly one native local route."""

import argparse
from dataclasses import asdict
import hashlib
import json
import os
from pathlib import Path
import sys
import time
import traceback

HERE = Path(__file__).resolve().parent
STUDY = HERE.parent
ANALYSES = STUDY.parent
SOURCE = ANALYSES / "weighted_geometric_control_20261122"
SAME_GOAL = ANALYSES / "same_goal_route_choice_20261121"
ANCHORS = STUDY / "source/frozen_anchors.json"
PROTOCOL = STUDY / "PROTOCOL.json"
BUDGET = 192
GLOBAL_FIELDS = ("call_id", "action_id", "frame_id", "input_sha256", "output_sha256", "inputs", "outputs")
INITIAL_FIELDS = ("layout", "scene_id", "anchor", "initial_packet_sha256", "initial_owner_sha256", "stop_budget")
CONTINUATION = {"unknown_weight": 0.25, "action_cost": 0.20}

sys.dont_write_bytecode = True
for variable in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[variable] = "1"
sys.path.insert(0, str(ANALYSES / "same_goal_cold_holdout_20261121/runner"))
sys.path.insert(0, str(SOURCE / "runner"))
from cold_run import full_local_route_adapter, native, raw_sensor_hashes
from execute import array_sha, capture, chain_hash, missing, np, read, sha
from weighted_score import Weights, choose_route, rank_admitted


def check(value, message):
    if not value:
        raise ValueError(message)


def check_equal(actual, expected, label):
    if actual != expected:
        raise ValueError(f"{label} changed")


def same_trace_row(actual, expected, label):
    checks = set(actual) | set(expected)
    checks.discard("decision_wall_s")
    for key in sorted(checks):
        check_equal(actual.get(key), expected.get(key), f"{label} field {key}")


def stable_local(event):
    audit = event["choice_audit"]
    ranked = audit["weighted_ranking"]
    return dict(event, choice_audit=dict(audit, weighted_ranking={
        key: value for key, value in ranked.items() if key != "supplemental_neutral_audit_wall_s"}))


def frozen_launch(args):
    freeze = read(args.freeze)
    check_equal(sha(PROTOCOL), freeze["protocol_sha256"], "protocol digest")
    actor_files = freeze["actor_files_sha256"]
    check(str(ANCHORS) in actor_files, "Frozen source anchor missing from actor seal")
    for filename, expected in actor_files.items():
        path = Path(filename)
        check("evaluator_references" not in path.parts and path.name != "evaluator_truth.npz",
              f"Truth-bearing actor input {path}")
        check_equal(sha(path), expected, f"sealed file {path}")
    row = next(x for x in freeze["schedule"] if x["layout"] == args.layout and x["arm"] == args.arm)
    anchor = read(ANCHORS)["scenes"][args.layout]
    if "route_id" in row:
        check_equal(row["route_id"], anchor["routes"][args.arm]["route_id"], "scheduled route")
    return freeze, anchor


def globals_now(backend):
    return [{key: event[key] for key in GLOBAL_FIELDS}
            for event in backend.calls if event["method"] == "select_topo_target"]


def history_directory(layout):
    return SOURCE / "runner/runs/holdout" / layout / "S"


def verify_archived_geometry(mapper, receipt):
    mesh = mapper.mesh()
    fields = {f"{field}_sha256": array_sha(getattr(mesh, field)) for field in ("vertices", "triangles")}
    check_equal(fields, receipt["native_mesh_fields"], f"archived fused mesh at paid {receipt['actual_paid']}")
    return fields


def ready_with_peer(args, identity, write):
    payload = json.dumps(identity, sort_keys=True, separators=(",", ":")).encode()
    ready = dict(identity=identity, fork_identity_sha256=hashlib.sha256(payload).hexdigest())
    write(args.out / "fork_ready.json", ready)
    if args.original_run:
        check_equal(read(args.original_run / "fork_ready.json"), ready, "verified replay fork peer")
        return 0.0
    peer_name = "geometry" if args.arm == "category" else "category"
    peer = args.peer_out or args.out.parent / peer_name
    peer_ready = peer / "fork_ready.json"
    started = time.monotonic()
    while not peer_ready.is_file():
        check(not (peer / "failure.json").exists(), "Peer failed exact fork restoration")
        check(time.monotonic() - started < 1200, "Technical timeout waiting for peer fork receipt")
        time.sleep(0.10)
    check_equal(read(peer_ready), ready, "route pair common reconstructed fork")
    return time.monotonic() - started


def pulse(out, write, paid, route_end, started):
    write(out / "heartbeat.json", dict(actual_paid=paid, native_budget=BUDGET,
          route_event=route_end, elapsed_s=time.monotonic() - started))


def actual_route_end(bound, fault, active_id, action, paid, runtime_state):
    if bound is None:
        return None
    if action is not None and active_id == bound["option_id"]:
        return None
    termination = runtime_state.get("termination") or {}
    reason = next((e["reason"] for e in fault
                   if e["event"] == "exploration_joint_replan_after_rejection"
                   and e["rejected_option_id"] == bound["option_id"]), None)
    if reason is None:
        reason = termination.get("reason")
    if reason is None:
        reason = "native_option_replaced_before_target"
    return dict(status="aborted", actual_paid=int(paid), reason=reason,
                option_id=bound["option_id"], target_pose=bound["pose"])


def retained_ranks(candidates, legacy, state, tolerance):
    native_row = next(row for row in candidates if row["route_id"] == legacy.NATIVE_ID)
    weights = Weights(**CONTINUATION)
    winner, scored = rank_admitted(candidates, native_row, "G", weights,
                                   float(state["mapper"].config.resolution_m) ** 2, tolerance)
    ranking = {item["route_id"]: item for item in scored}
    source = {item["route_id"]: item for item in candidates}
    result = {}
    while ranking:
        best = max(item["score_m2"] for item in ranking.values())
        pool = [key for key in ranking if best - ranking[key]["score_m2"] <= tolerance]
        chosen = min(pool, key=lambda key: (source[key]["outbound_cost"], key != legacy.NATIVE_ID,
                                           source[key]["outbound_sha256"]))
        result[chosen] = dict(rank=len(result) + 1, **ranking.pop(chosen))
    check_equal(next(iter(result)), winner["route_id"], "canonical G_perf winner")
    return result


def run(args, freeze, anchor):
    out = args.out
    layout = args.layout
    prefix_paid = anchor["paid_before"]
    trigger = anchor["selection_call_id"]
    route = anchor["routes"][args.arm]
    nominal_paid = prefix_paid + route["outbound_cost"]
    observed = history_directory(layout)
    check_equal(str(observed), anchor["source_run_dir"], "bound archived S directory")
    for source in (anchor["verified_actor_sources"], anchor["original_initial_files"]):
        for record in source.values():
            check_equal(sha(Path(record["path"])), record["sha256"], f"bound source {record['path']}")
    manifest = read(observed / "manifest.json")
    matching_archived_meshes = {item["actual_paid"]: item for item in manifest["checkpoints"].values()
                               if item["status"] == "captured" and item["actual_paid"] <= prefix_paid}
    source_rows = read(observed / "steps.json")
    source_initial_raw = read(observed / "initial_raw_rgbd_scan_sha256.json")
    source_globals = [event for event in read(observed / "global_choices.json") if event["call_id"] <= trigger]
    source_locals = [event for event in read(observed / "local_choices.json") if event["selection_call_id"] <= trigger]
    with np.load(observed / "observed_belief_trace.npz", allow_pickle=False) as trace:
        np.testing.assert_array_equal(trace["paid"][:prefix_paid + 1], np.arange(prefix_paid + 1))
        archived_beliefs = trace["belief"][:prefix_paid + 1]
    expected = source_locals[-1]
    check_equal(len(source_globals), len(source_locals), "archived global and local prefix event counts")
    check_equal(source_globals[-1]["call_id"], trigger, "archived global trigger")
    check_equal(expected["selection_call_id"], trigger, "archived local trigger")
    for item, live in (("prefix_pose", expected["pose"]), ("packet_sha256", expected["packet_sha256"]),
                       ("belief_sha256", expected["observed_belief_sha256"]),
                       ("goal_pose", expected["same_global_target"]),
                       ("return_anchor", expected["choice_audit"]["return_anchor"]),
                       ("candidate_pool_sha256", expected["choice_audit"]["candidate_pool_sha256"]),
                       ("archived_S_score_table_sha256", expected["choice_audit"]["score_table_sha256"])):
        check_equal(anchor[item], live, f"sealed historical {item}")
    weights = Weights(**CONTINUATION)
    old_protocol = read(SAME_GOAL / "PROTOCOL.json")
    group = old_protocol["heldout"]
    group.update(layouts=list(read(ANCHORS)["scenes"]), paid_action_budget=BUDGET)
    original_read, original_loader = native.read, native.load_source
    native.HOLDOUT = SOURCE / "scenes"
    native.ARM_MODES = {"mechanism": ("G", "S")}
    initial_path = native.HOLDOUT / "prepared" / layout / "initial_state.json"
    initial_full = read(initial_path)

    def filtered_read(path):
        if Path(path) == initial_path:
            return {key: initial_full[key] for key in INITIAL_FIELDS}
        return original_read(path)

    native.read = filtered_read
    old, values, initial, global_mode, _ = native.setup_runtime("holdout", layout, "mechanism", old_protocol)
    world, transform, packet, components, runtime, effective, owner = values
    from nso.cpu_sensor_contract_v10 import digest
    from runtime_binding import write

    backend = components._cpu_backend
    mapper = runtime.states[0]["mapper"]
    initial_pose = [*packet.position, packet.heading]
    initial_raw = raw_sensor_hashes(packet, array_sha)
    check_equal(global_mode, "G", "neutral global policy")
    check_equal(world.seed, initial_full["seed"], "renderer seed")
    check_equal(initial_raw, source_initial_raw, "archived raw initial packet")
    check_equal(initial_pose, manifest["initial_pose"], "archived initial pose")
    check_equal(owner["owner_sha256"], manifest["initial_owner_sha256"], "archived initial observation owner")
    np.testing.assert_array_equal(mapper.belief, archived_beliefs[0])
    check_equal(array_sha(mapper.belief), manifest["initial_belief_sha256"], "archived initial map")
    check_equal(tuple(backend.scenes[0]["return_anchor"]), tuple(anchor["return_anchor"]), "native return anchor")
    verified_archived_meshes = {str(0): verify_archived_geometry(mapper, matching_archived_meshes[0])}

    intervention = None
    local_prefix_count = 0
    synchronized_wall_s = 0.0
    steps = []
    beliefs = [mapper.belief.copy()]
    chain = chain_hash("0" * 64, initial_raw["packet_sha256"])

    def install_local(name, path):
        legacy = original_loader(name, path)
        if Path(path) != native.METHOD:
            return legacy
        policy = legacy.choose_route

        def choose(state, native_option, mode, route_union_area, config):
            nonlocal intervention, local_prefix_count, synchronized_wall_s
            check_equal(mode, "S", "compatibility wrapper argument")
            paid = state["packet"].action_id
            if intervention is not None:
                check(paid >= prefix_paid, "G_perf started before intervention")
                return choose_route(legacy, policy, state, native_option, "G", weights, route_union_area, config)

            check(paid <= prefix_paid, "archived native S prefix passed the sealed intervention state")
            selected, audit = choose_route(legacy, policy, state, native_option, "S", Weights(), route_union_area, config)
            archived = source_locals[local_prefix_count]
            saved = archived["choice_audit"]
            local_prefix_count += 1
            check_equal(native_option["selection_call_id"], archived["selection_call_id"], "historical choice call id")
            check_equal(paid, archived["paid_before"], "historical choice paid clock")
            check_equal(state["packet"].sha256(), archived["packet_sha256"], "historical choice packet")
            check_equal(array_sha(mapper.belief), archived["observed_belief_sha256"], "historical choice map")
            check_equal(audit["candidate_pool_sha256"], saved["candidate_pool_sha256"], "historical candidate pool")
            check_equal(audit["score_table_sha256"], saved["score_table_sha256"], "historical candidate scores")
            check_equal(audit["candidates"], saved["candidates"], "historical admitted and rejected candidate rows")
            check_equal(audit["chosen_outbound_sha256"], saved["chosen_outbound_sha256"], "historical selected S trace")
            if native_option["selection_call_id"] != trigger:
                return selected, audit

            check_equal(paid, prefix_paid, "sealed trigger clock")
            check_equal(native_option["pose"], anchor["goal_pose"], "sealed native goal")
            check_equal(native_option["return_actions"], expected["native_return_actions"], "sealed return actions")
            check_equal(native_option["return_states"], expected["native_return_states"], "sealed return states")
            check_equal(globals_now(backend), source_globals, "full historical global candidate and score history")
            check_equal(local_prefix_count, len(source_locals), "full historical local candidate and score history")
            current = next(row for row in audit["candidates"] if row["route_id"] == route["route_id"])
            check_equal(current, route, "sealed forced-route payload")
            check(current["admitted"], "Chosen intervention row is not in the legal native pool")
            if paid:
                check_equal(chain, source_rows[paid-1]["packet_chain_sha256"], "historical prefix chain")
            changed = legacy._same_goal_option(native_option, current)
            ranks = retained_ranks(audit["candidates"], legacy, state, config["score_equality_tolerance"])
            check_equal(next(iter(ranks)), anchor["G_perf_winner"]["route_id"], "frozen G_perf rank leader")
            for arm, prescribed in anchor["routes"].items():
                check_equal(ranks[prescribed["route_id"]]["rank"], anchor["G_perf_rank"][arm]["rank"], "frozen G_perf route rank")
            fork = checkpoints["fork"]
            check_equal(fork["actual_paid"], paid, "captured common fork clock")
            next_noise = np.random.default_rng(np.random.SeedSequence([world.seed, paid + 1, 739]))
            common = dict(layout=layout, native_budget=BUDGET, actual_paid=paid, selection_call_id=trigger,
                          pose=anchor["prefix_pose"], target_pose=anchor["goal_pose"],
                          return_anchor=anchor["return_anchor"], packet=raw_sensor_hashes(state["packet"], array_sha),
                          packet_chain_sha256=chain, belief_sha256=array_sha(mapper.belief),
                          native_mesh_fields=fork["native_mesh_fields"], mesh_sha256=fork["files_sha256"]["native_mesh.ply"],
                          archived_fused_meshes_verified=verified_archived_meshes,
                          feedback_ledger_sha256=digest(state["ledger"].snapshot()),
                          observed_owner_sha256=source_rows[paid-1]["owner_sha256"] if paid else manifest["initial_owner_sha256"],
                          candidate_pool_sha256=audit["candidate_pool_sha256"],
                          source_score_table_sha256=audit["score_table_sha256"],
                          archived_global_output_sha256=source_globals[-1]["output_sha256"],
                          seed=int(world.seed), next_noise_bit_generator_state=next_noise.bit_generator.state)
            synchronized_wall_s += ready_with_peer(args, common, write)
            intervention = dict(
                applied_once=True, actual_paid=paid, selection_call_id=trigger,
                option_id=native_option["option_id"], target_pose=native_option["pose"],
                return_anchor=anchor["return_anchor"], route_id=current["route_id"],
                outbound_sha256=current["outbound_sha256"], packet_sha256=state["packet"].sha256(),
                packet_chain_sha256=chain, belief_sha256=array_sha(mapper.belief), start_pose=[*state["packet"].position, state["packet"].heading],
                candidate_pool_sha256=audit["candidate_pool_sha256"], archived_S_score_table_sha256=audit["score_table_sha256"],
                nominal_paid=nominal_paid, native_candidate=globals_now(backend)[-1]["outputs"]["selected"].get("group"),
                forced_outbound_actions=current["outbound_actions"], forced_outbound_states=current["outbound_states"],
                return_actions=changed["return_actions"], return_states=changed["return_states"],
                historical_prefix_actions_verified=len(steps), historical_local_choices_verified=local_prefix_count,
                historical_global_choices_verified=len(source_globals), random_seed=int(world.seed),
                random_action_clock=int(world.step_count),
                archived_fused_meshes_verified=verified_archived_meshes,
                continuation_global="G", continuation_local="G_perf", **CONTINUATION,
                selected_routes_G_perf={name: ranks[item["route_id"]] for name, item in anchor["routes"].items()},
                G_perf_first_route_id=next(iter(ranks)), complete_G_perf_rank_table=list(ranks.values()),
            )
            write(out / "intervention.json", intervention)
            # Audit the actually applied route while keeping the archived S and G
            # score-table values available for independent comparison.
            audit.update(selected_route_id=current["route_id"], chosen_outbound_sha256=current["outbound_sha256"],
                         single_route_intervention=True)
            return changed, audit

        legacy.choose_route = choose
        return legacy

    native.load_source = install_local
    local_events = full_local_route_adapter(components, old, "S", old_protocol["local_route"], native.local_route_adapter)
    write(out / "attempt.json", dict(status="started", layout=layout, arm=args.arm, native_budget=BUDGET,
          frozen_at=freeze["frozen_unix_s"], runner_freeze_sha256=sha(args.freeze), initial_pose=initial_pose,
          initial_packet_sha256=initial_raw["packet_sha256"], initial_belief_sha256=array_sha(mapper.belief),
          sensor_config=initial_full["actual_config"], effective_runtime=effective,
          randomness="archived scene seed and paid action index; the native depth renderer uses SeedSequence([seed, step_count, 739])",
          start_unix_s=time.time(), expected_prefix_actions=prefix_paid))
    write(out / "initial_raw_rgbd_scan_sha256.json", initial_raw)
    checkpoints = {}
    if prefix_paid == 0:
        checkpoints["fork"] = capture(out, "fork", mapper, packet, chain, 0, "verified_before_single_route_intervention")
    route_end = None
    bound = None
    active_paths = {}
    choices = {}
    checked_local = 0
    started = time.monotonic()
    clocks = {nominal_paid}
    if args.capture_at is not None:
        clocks.add(args.capture_at)
    if 0 in clocks:
        check(prefix_paid == 0, "A post-fork measurement cannot precede restoration")
        checkpoints["paid_0"] = capture(out, "paid_0", mapper, packet, chain, 0, "common_charged_clock", copy_from="fork")

    while not runtime.states[0]["closed"]:
        start_action = time.monotonic()
        previous_sync = synchronized_wall_s
        first_audit = len(runtime.audit)
        action = runtime.next_local_action(0)
        decision_wall_s = time.monotonic() - start_action - (synchronized_wall_s - previous_sync)
        state = runtime.states[0]
        active = state["option"]
        active_id = None if active is None else active["option_id"]
        for local in local_events[checked_local:]:
            choices[local["option_id"]] = local
            if local["selection_call_id"] < trigger:
                expected_prefix = source_locals[checked_local]
                check_equal(stable_local(local), stable_local(expected_prefix), "exact pre-intervention local event")
            elif local["selection_call_id"] == trigger:
                bound = dict(option_id=local["option_id"], pose=local["same_global_target"])
            else:
                ranking = local["choice_audit"]["weighted_ranking"]
                for key, value in (("prior", "G"), *CONTINUATION.items()):
                    check_equal(ranking[key], value, "G_perf continuation scoring")
                check_equal(ranking["local_area_provider_calls"]["S"], 0, "G_perf semantic area leakage")
                local["local_mode"] = "G"
            checked_local += 1
        if bound is not None and route_end is None:
            route_end = actual_route_end(bound, runtime.audit[first_audit:], active_id, action, packet.action_id, state)
            if route_end is not None:
                checkpoints["route_end"] = capture(out, "route_end", mapper, packet, chain,
                                                     route_end["actual_paid"], route_end["reason"],
                                                     steps[-1] if steps else None)
        if action is None:
            break
        pending = state["pending"]
        check(pending["allowed"], "Native action bypassed its guard")
        phase = state["phase"]
        target = None if active is None else list(active["pose"])
        route_index = None
        if active is not None:
            route_index = active_paths.get(active_id, 0)
            check_equal(action, active["outbound_actions"][route_index], "native bound action")
            active_paths[active_id] = route_index + 1
        picked = choices.get(active_id)
        previous_belief = beliefs[-1]
        raw, collision, done = world.step(action)
        packet = old.packet(world, transform, raw, layout, action, collision, done)
        owner = old.front.associate(mapper, raw, packet.frame_id)
        old.method.observe(mapper, packet.frame)
        runtime.observe(0, world.step_count, None, None, None, sensor_packet=packet)
        paid = packet.action_id
        packed = packet.sha256()
        chain = chain_hash(chain, packed)
        pose = [*world.position, world.heading]
        beliefs.append(mapper.belief.copy())
        row = dict(step=paid, action=action, position=list(world.position), heading=world.heading,
            collision=bool(collision), done=bool(done), packet_sha256=packed, packet_chain_sha256=chain,
            frame_sha256=old.method._frame_hash(raw), nonsemantic_packet_sha256=old.base.nonsemantic_packet_sha(packet),
            owner_sha256=owner["owner_sha256"], active_option_id=active_id,
            active_candidate_id=None if active is None else active["candidate_id"],
            selection_call_id=None if active is None else active["selection_call_id"],
            route_action_index_zero_based=route_index, local_route_id=None if picked is None else picked["active_local_route_id"],
            active_outbound_sha256=None if picked is None else picked["active_outbound_sha256"], selected_target_pose=target,
            phase_before_action=phase, matched_target_pose_after_step=(pose == target) if target is not None else None,
            guard_allowed=True, guard_reason=pending["reason"], reserved_return_cost=pending["reserved_return_cost"],
            observed_known_cells=int(np.count_nonzero(mapper.belief != -1)), observed_free_cells=int(np.count_nonzero(mapper.belief == 0)),
            newly_known_cells=int(np.count_nonzero((previous_belief == -1) & (mapper.belief != -1))),
            path_length_m=world.moves * world.config.resolution_m, decision_wall_s=decision_wall_s)
        row.update({key: value for key, value in raw_sensor_hashes(packet, array_sha).items() if key not in row})
        if paid <= prefix_paid:
            same_trace_row(row, source_rows[paid-1], f"restored S paid step {paid}")
            np.testing.assert_array_equal(mapper.belief, archived_beliefs[paid])
            if paid in matching_archived_meshes:
                verified_archived_meshes[str(paid)] = verify_archived_geometry(mapper, matching_archived_meshes[paid])
            if paid == prefix_paid:
                checkpoints["fork"] = capture(out, "fork", mapper, packet, chain, paid,
                                              "verified_before_single_route_intervention", row)
        steps.append(row)
        if bound is not None and route_end is None and active_id == bound["option_id"]:
            if args.arm == "category":
                same_trace_row(row, source_rows[paid-1], f"archived category route paid step {paid}")
            index = paid - prefix_paid
            check_equal(action, route["outbound_actions"][index-1], "executed part of intervention route")
            if collision or done or pose == bound["pose"]:
                arrived = pose == bound["pose"] and not collision
                if arrived:
                    check_equal(paid, nominal_paid, "achieved route arrival clock")
                reason = "reached_bound_target" if arrived else ("collision" if collision else "sensor_terminal_before_target")
                route_end = dict(status="arrived" if arrived else "aborted", actual_paid=int(paid),
                                 reason=reason, option_id=bound["option_id"], target_pose=bound["pose"])
                checkpoints["route_end"] = capture(out, "route_end", mapper, packet, chain, paid, reason, row)
            if not collision:
                check_equal(pose, route["outbound_states"][index], "actual pose while bound")
        if paid in clocks:
            name = f"paid_{paid}"
            same_clock_source = "route_end" if route_end is not None and route_end["actual_paid"] == paid else None
            checkpoints[name] = capture(out, name, mapper, packet, chain, paid, "charged_clock_checkpoint", row,
                                        copy_from=same_clock_source)
        if paid % 10 == 0:
            pulse(out, write, paid, route_end, started)

    check(intervention is not None and bound is not None, "Verified native intervention not reached")
    summary = runtime.sensor_episode_summary(0)
    termination = summary["termination"]
    check(summary["closed"] and termination is not None, "Native episode did not close")
    check_equal(len(steps), termination["final_action_id"], "native terminal clock")
    check(len(steps) <= BUDGET, "native budget exceeded")
    if route_end is None:
        raise ValueError("Native route completed without arrival or interruption boundary")
    route_end["bound_paid_actions"] = sum(row["active_option_id"] == bound["option_id"] for row in steps)
    route_end["same_target_reached_later"] = next((row["step"] for row in steps if row["step"] > route_end["actual_paid"]
                                                      and [*row["position"], row["heading"]] == bound["pose"]), None)
    final_paid = len(steps)
    terminal_source = next((name for name, recorded in checkpoints.items()
                            if recorded["status"] == "captured" and recorded["actual_paid"] == final_paid), None)
    checkpoints["terminal"] = capture(out, "terminal", mapper, packet, chain, BUDGET, "actual_native_terminal",
                                     steps[-1] if steps else None, copy_from=terminal_source)
    for wanted in clocks:
        name = f"paid_{wanted}"
        if name not in checkpoints:
            check(final_paid < wanted, "Measured diagnostic clock omitted")
            checkpoints[name] = missing(out, name, wanted, final_paid, "native_stop_before_checkpoint")
    write(out / "steps.json", steps)
    write(out / "global_choices.json", globals_now(backend))
    write(out / "local_choices.json", local_events)
    write(out / "runtime_summary.json", summary)
    write(out / "audit.json", runtime.audit)
    np.savez_compressed(out / "observed_belief_trace.npz", belief=np.asarray(beliefs, dtype=np.int8),
                        paid=np.arange(len(beliefs), dtype=np.int16))
    result = dict(status="completed_execution", layout=layout, arm=args.arm, native_budget=BUDGET,
        initial_pose=initial_pose, initial_packet_sha256=initial_raw["packet_sha256"],
        initial_belief_sha256=array_sha(beliefs[0]), final_pose=[*packet.position, packet.heading],
        final_packet_chain_sha256=chain, paid_actions=final_paid, returned=termination["returned_to_anchor"],
        collisions=sum(row["collision"] for row in steps), actual_closed_loop=True,
        global_mode="G", local_continuation_mode="G_perf", **asdict(weights),
        observed_native_termination=termination, route_end=route_end,
        path_length_m=steps[-1]["path_length_m"] if steps else 0.0, evaluator_inputs_read_by_actor=False,
        runner_freeze_sha256=sha(args.freeze), source_protocol_sha256=sha(PROTOCOL),
        checkpoints=checkpoints, diagnostic_requested_paid=sorted(clocks),
        elapsed_s=time.monotonic()-started, fork_synchronization_wall_s=synchronized_wall_s,
        charged_decision_wall_s=sum(row["decision_wall_s"] for row in steps))

    if args.original_run:
        original = args.original_run
        original_manifest = read(original / "manifest.json")
        check_equal(original_manifest["runner_freeze_sha256"], result["runner_freeze_sha256"], "matched replay freeze")
        for name, expected_sha in original_manifest["files_sha256"].items():
            check_equal(sha(original / name), expected_sha, f"original execution file {name}")
        prior_rows = read(original / "steps.json")
        check_equal(len(steps), len(prior_rows), "full original replay trace length")
        for now, archived in zip(steps, prior_rows, strict=True):
            same_trace_row(now, archived, f"verified diagnostic replay paid {now['step']}")
        with np.load(original / "observed_belief_trace.npz", allow_pickle=False) as prior_beliefs:
            np.testing.assert_array_equal(beliefs, prior_beliefs["belief"])
        check_equal(read(original / "intervention.json"), intervention, "verified replay intervention")
        for name in ("fork", "route_end", "terminal"):
            check_equal(checkpoints[name], original_manifest["checkpoints"][name], f"verified replay {name} checkpoint")
        for key in ("route_end", "observed_native_termination", "final_pose", "paid_actions", "path_length_m"):
            check_equal(result[key], original_manifest[key], f"verified replay {key}")
        result.update(original_full_trace_verified=True, verified_against_original_manifest_sha256=sha(original / "manifest.json"),
                      extra_capture_paid=args.capture_at)
    pulse(out, write, final_paid, route_end, started)
    # The periodic heartbeat never enters the immutable manifest because its wall clock is diagnostic.
    result["files_sha256"] = {str(path.relative_to(out)): sha(path) for path in sorted(out.rglob("*"))
                              if path.is_file() and path.name != "heartbeat.json"}
    write(out / "manifest.json", result)
    print(json.dumps(dict(layout=layout, arm=args.arm, paid=final_paid, route_end=route_end,
                          returned=result["returned"], collisions=result["collisions"]), ensure_ascii=False), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("layout")
    parser.add_argument("arm", choices=("category", "geometry"))
    parser.add_argument("--freeze", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--capture-at", type=int)
    parser.add_argument("--original-run", type=Path)
    parser.add_argument("--peer-out", type=Path)
    args = parser.parse_args()
    args.freeze, args.out = args.freeze.resolve(), args.out.resolve()
    check((args.capture_at is None) == (args.original_run is None), "Extra capture requires a sealed original run")
    freeze, anchor = frozen_launch(args)
    if args.capture_at is not None:
        check(anchor["paid_before"] <= args.capture_at <= BUDGET, "Extra clock outside the sealed post-fork interval")
    args.out.mkdir(parents=True, exist_ok=False)
    try:
        run(args, freeze, anchor)
    except BaseException as error:
        failure = dict(status="failed", layout=args.layout, arm=args.arm, error=repr(error), traceback=traceback.format_exc(),
                       runner_freeze_sha256=sha(args.freeze))
        (args.out / "failure.json").write_text(json.dumps(failure, indent=2) + "\n")
        raise


if __name__ == "__main__":
    main()
