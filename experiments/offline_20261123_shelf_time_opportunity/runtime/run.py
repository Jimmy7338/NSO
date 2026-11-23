#!/usr/bin/env python3
"""Replay a sealed observed fork, bind one legal outbound, and run native G closure."""

import argparse
import faulthandler
import gc
import importlib.util
import os
from pathlib import Path
import sys
import traceback

for variable in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[variable] = "1"
sys.dont_write_bytecode = True

HERE = Path(__file__).resolve().parent
ANALYSES = HERE.parents[1]
SHELF = ANALYSES / "shelf_occlusion_pairs_20261122"
INCREMENT = ANALYSES / "region_global_revision_20261122/next/local_increment"
WITNESS = SHELF / "followup/local_witness"
BASE = "3a0f74ff09d2d38aaa6b641392256edd2ea9f4db"
CAPTURE_CLOCKS = (16, 24, 48, 96, 144, 192)
BUDGET = 192


def load_published_actor():
    location = SHELF / "runtime/run.py"
    spec = importlib.util.spec_from_file_location("nso_published_occlusion_actor", location)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


legacy = load_published_actor()
from supported_hidden import compare_routes

native, read, sha, digest, write = legacy.native, legacy.read, legacy.sha, legacy.digest, legacy.write


def require(condition, message):
    if not condition:
        raise ValueError(message)


def verify_files(files):
    for filename, expected in files.items():
        parts = Path(filename).parts
        require("evaluator_references" not in parts and "evaluator_truth" not in parts,
                f"Evaluator-only input claimed by actor: {filename}")
        require(sha(filename) == expected, f"Pinned actor or simulator input changed: {filename}")


def bound_file(name, digest, root):
    path = Path(name).resolve()
    require(path.is_file() and path.is_relative_to(root) and sha(path) == digest,
            f"Frozen archive is missing, outside its archive, or changed: {name}")
    return path


def load_inputs(bank_path, layout, route_id):
    bank = read(bank_path)
    require(bank["native_budget"] == BUDGET and bank["primary_task_paid"] == 96,
            "Sealed developmental bank changed the native action budget or primary clock")
    require(layout in bank["placements"], f"Unknown registered placement {layout}")
    placement = bank["placements"][layout]
    candidates = [row for row in placement["routes"] if row["route_id"] == route_id]
    require(len(candidates) == 1, f"Route is not a unique registered branch: {route_id}")
    expected_route = candidates[0]
    source = placement["source"]
    require(source in {"increment", "shelf"}, f"Unknown published source {source}")
    root = INCREMENT if source == "increment" else SHELF
    bank_file = bound_file(placement["source_bank_path"], placement["source_bank_sha256"], root)
    candidates_file = bound_file(placement["source_candidates_path"], placement["source_candidates_sha256"], root)
    source_bank = read(bank_file)
    source_spec = read(root / "PROTOCOL.json")
    if source == "increment":
        input_freeze = root / "INPUT_FREEZE.json"
        require(source_bank["inputs_sha256"] == sha(input_freeze), "Earlier increment actor seal drifted")
        canonical = source_bank["layouts"][layout]
        source_candidates = canonical["eligible_full_pool"]
        frozen = read(input_freeze)
        scene = root / "scenes"
        cache_name = "simulator_cache_sha256"
    else:
        input_freeze = root / "FORMAL_FREEZE.json"
        require(source_bank["freeze_sha256"] == sha(input_freeze), "Earlier shelf actor seal drifted")
        canonical = source_bank
        candidate_trace = read(candidates_file)
        source_candidates = [row for row in candidate_trace["candidates"] if row["admitted"]]
        frozen = read(input_freeze)
        scene = Path(frozen["scene_root"])
        cache_name = "simulator_only_cache_sha256"
    canonical_row = next(row for row in source_candidates if row["route_id"] == route_id)
    require(placement["candidate_pool_sha256"] == canonical["candidate_pool_sha256"]
            and placement["n_legal_native"] == len(source_candidates)
            and placement["paid_before"] == canonical["paid_before"]
            and placement["target_pose"] == canonical["target_pose"],
            "Registered target, clock or candidate bank differs from published archive")
    for key in ("outbound_sha256", "outbound_cost", "outbound_actions", "outbound_states", "reserved_return_cost"):
        require(expected_route[key] == canonical_row[key], f"Registered route differs in {key}")
    verify_files(frozen["actor_files_sha256"])
    verify_files(frozen[cache_name])
    return placement, expected_route, canonical, source_spec, frozen, scene, input_freeze, candidates_file


def ensure_bound_execution(rows, trigger, route):
    """Verify native motions still match the forced outbound only while its option is active."""
    first = trigger["paid_before"]
    matching = []
    for index, row in enumerate(rows[first:]):
        if row["active_option_id"] != trigger["option_id"]:
            break
        require(index < len(route["outbound_actions"])
                and row["guard_allowed"]
                and row["route_action_index_zero_based"] == index
                and row["local_route_id"] == route["route_id"]
                and row["active_outbound_sha256"] == route["outbound_sha256"]
                and row["action"] == route["outbound_actions"][index]
                and (row["collision"] or [*row["position"], row["heading"]] == route["outbound_states"][index+1]),
                f"Native charged action {row['step']} disagrees with registered outbound")
        matching.append(row)
        if row["collision"] or row["matched_target_pose_after_step"]:
            break
    return len(matching)


def compare_short_witness(placement, route, rows, beliefs, event, initial_raw, fork):
    """Read prior physical outcomes during postflight, after every new decision was made."""
    if not route.get("archived_short"):
        return None
    short = route["archived_short"]
    receipt_path = Path(short.get("run") or short.get("directory") or short.get("run_directory") or short.get("path")).resolve()
    if receipt_path.is_file():
        receipt_path = receipt_path.parent
    require(receipt_path.is_relative_to(WITNESS) or receipt_path.is_relative_to(INCREMENT),
            "Prior short acquired native record is outside the two pinned archives")
    witness = bound_file(short["witness_receipt_path"], short["witness_receipt_sha256"], WITNESS)
    execution = bound_file(short["execution_receipt_path"], short["execution_receipt_sha256"], WITNESS)
    manifest_path = receipt_path / "manifest.json"
    require(sha(manifest_path) == short["manifest_sha256"], "Archived short manifest changed")
    old = read(manifest_path)
    old_steps = receipt_path / "steps.json"
    require(sha(old_steps) == old["files_sha256"]["steps.json"] == short["archived_trace_sha256"],
            "Archived short trace changed")
    old_rows = read(old_steps)
    require(old["layout"] == placement and old["route_id"] == route["route_id"]
            and old["first_packet_sha256"] == initial_raw["packet_sha256"]
            and old["fork_packet_sha256"] == fork["packet_sha256"],
            "Archived short run refers to a different start, route or fork")
    old_event = old["route_end"]
    status = {"native_cancelled": "guard_cancelled", "arrived": "arrived", "collision": "collision",
              "native_terminal": "terminal_before_arrival"}.get(old_event["status"], old_event["status"])
    require(old_event == short["observed_outbound_event"]
            and old_event["actual_paid"] == len(old_rows) == event["actual_paid"]
            and status == event["status"] and old_event["reason"] == event["reason"],
            "Native outbound event did not replay the archived short event")
    for prior, actual in zip(old_rows, rows[:len(old_rows)], strict=True):
        match = {key: actual[key] for key in (
            "step", "packet_sha256", "depth_values_sha256", "rgb_values_sha256", "scan_ranges_sha256",
            "action", "collision", "guard_reason", "reserved_return_cost", "owner_sha256")}
        match.update(pose=[*actual["position"], actual["heading"]], path_m=actual["path_length_m"],
                     option_id=actual["active_option_id"], map_sha256=legacy.array_sha(beliefs[actual["step"]]))
        require(match == prior, f"Archived native observation/action differs at charged action {actual['step']}")
    return {"manifest_path": str(manifest_path), "manifest_sha256": sha(manifest_path),
            "steps_sha256": sha(old_steps), "first_to_last_archived_motion_matches": True,
            "matched_charged_actions": len(old_rows), "native_route_event_matches": True,
            "witness_receipt": str(witness), "execution_receipt": str(execution),
            "old_native_event": old_event}


def execute(bank_path, layout, route_id, out):
    placement, route, canonical, source_spec, frozen, scene, input_freeze, candidates_file = load_inputs(bank_path, layout, route_id)
    source = placement["source"]
    bank_sha = sha(bank_path)
    source_settings = read(legacy.SOURCE_PROTOCOL)
    original_layouts = source_spec.get("layouts", frozen.get("placements"))
    source_settings["heldout"].update(layouts=original_layouts, paid_action_budget=BUDGET)
    source_settings["diagnostic_checkpoints"] = list(CAPTURE_CLOCKS)
    weights = legacy.Weights(unknown_weight=.25, action_cost=.20)
    require(source_spec["weights"] == {"unknown_weight": .25, "action_cost": .20},
            "Source changed the common neutral chooser weights")
    initial = read(scene / "prepared" / layout / "initial_state.json")
    native.HERE, native.HOLDOUT, native.PROTOCOL = HERE, scene, HERE.parent / "PROTOCOL.json"
    native.ARM_MODES = {"G": ("G", "G")}
    original_setup, original_load, original_read, original_bind, original_sha = (
        native.setup_runtime, native.load_source, native.read, native.local_route_adapter, native.sha)
    raw, prefix, probes, collections = [], [], [], []
    prefix_chain = "0" * 64
    seen_depth = 0
    trigger = fork = native_writer = None
    internal_manifest = out / ".native_manifest.json"

    def conceal_published_actor_freeze(path):
        if Path(path) == HERE / "holdout_freeze.json":
            return {"world_files_sha256": frozen["actor_files_sha256"]}
        return original_read(path)

    def identify_episode_freeze(path):
        return bank_sha if Path(path) == HERE / "holdout_freeze.json" else original_sha(path)

    def take_packet(packet):
        nonlocal prefix_chain, seen_depth
        require(packet.action_id == len(raw), "Native action and raw sensor clocks disagree")
        raw.append(legacy.raw_sensor_hashes(packet, legacy.array_sha))
        if trigger is None:
            prefix.append(packet)
            prefix_chain = legacy.chain_hash(prefix_chain, packet.sha256())
            seen_depth += int(legacy.np.count_nonzero((packet.frame.semantic == 2) & (packet.frame.depth_m > 0)))
        return packet

    def setup(*args):
        nonlocal native_writer
        result = original_setup(*args)
        import runtime_binding

        old, (world, _, packet, _, _, _, _), _, mode, _ = result
        require(mode == "G" and world.step_count == 0 and world.seed == initial["seed"]
                and packet.sha256() == initial["initial_packet_sha256"], "Prior native cold state changed")
        take_packet(packet)
        if source == "increment":
            require(raw[0] == canonical["initial_raw"], "Earlier raw initial sensor fields do not replay")
        original_packet = old.packet
        old.packet = lambda *a, **kw: take_packet(original_packet(*a, **kw))
        cache_original = old.method._cache

        def collect_then_cache(mapper, mode):
            cached = mapper._dual_template_state["cache"]
            if cached is None or cached["key"] != (int(mapper.frames), mode):
                collections.append({"mapper_frames": int(mapper.frames), "unreachable_collected": gc.collect()})
            return cache_original(mapper, mode)

        old.method._cache = collect_then_cache
        native_writer = runtime_binding.write
        runtime_binding.write = lambda path, data: native_writer(
            internal_manifest if Path(path) == out / "manifest.json" else path, data)
        return result

    def load(name, path):
        module = original_load(name, path)
        if Path(path) != native.METHOD:
            return module
        frozen_route = module.choose_route

        def choose(state, native_option, mode, area, config):
            import runtime_binding
            nonlocal trigger, fork

            require(mode == "G", "Common continuation changed away from neutral local G")
            require(all(config[key] == value for key,value in legacy.INHERITED_FIELDS.items()),
                    "Native candidate admission controls drifted")
            baseline, neutral = legacy.choose_route(module, frozen_route, state, native_option, "G", weights,
                                                    area, config, runtime_binding.revision.reweight)
            if trigger:
                return baseline, neutral
            packet = state["packet"]
            paid = packet.action_id
            legal = [row for row in neutral["candidates"] if row["admitted"]]
            require(len({row["outbound_sha256"] for row in legal}) == len(legal), "Duplicate eligible traces")
            if not legal or len(legal) < 2:
                probes.append({"paid": paid, "qualifies": False, "admitted": len(legal)})
                return baseline, neutral
            if source == "increment":
                _, scoring = legacy.choose_route(module, frozen_route, state, native_option, "S", weights,
                                                  area, config, runtime_binding.revision.reweight)
                ranking = compare_routes(state, scoring["candidates"], source_spec, weights,
                                         config["score_equality_tolerance"], legacy.rank_direct,
                                         legacy.rank_admitted, runtime_binding.model.REPAIR.classify_history)
                qualified = ranking["direct"]["observed_support_voxels"] > 0
            else:
                qualified = seen_depth > 0
                scoring = ranking = None
            probes.append({"paid": paid, "qualifies": qualified, "admitted": len(legal),
                           "candidate_pool_sha256": neutral["candidate_pool_sha256"]})
            if not qualified:
                require(paid <= placement["paid_before"], "Archived qualifying observed fork was missed")
                return baseline, neutral
            require(paid == placement["paid_before"] and native_option["pose"] == placement["target_pose"]
                    and native_option["selection_call_id"] == canonical["selection_call_id"]
                    and neutral["candidate_pool_sha256"] == placement["candidate_pool_sha256"]
                    and state["ledger"].remaining_budget == BUDGET - paid
                    and len(legal) == placement["n_legal_native"],
                    "First legal observed fork, budget, target or candidate population differs")
            mapper = state["mapper"]
            fork = legacy.capture(out, "fork", mapper, packet, prefix_chain, paid, "shared_observed_decision_prefix")
            prefix_sha = legacy.save_prefix(out, prefix)
            if source == "increment":
                require(scoring["candidates"] == canonical["assessed_candidates"]
                        and all(ranking[key] == canonical[key] for key in
                                ("selected", "rankings", "eligible_full_pool", "direct", "support_tracks"))
                        and packet.sha256() == canonical["packet_sha256"]
                        and prefix_chain == canonical["packet_chain_sha256"]
                        and legacy.array_sha(mapper.belief) == canonical["fork_belief_sha256"]
                        and fork["native_mesh_fields"] == canonical["fork_mesh_fields"]
                        and native_option["option_id"] == canonical["option_id"],
                        "Earlier increment observed state, candidate supports, ranks or mesh did not replay")
                audit = scoring
                evidence = {"candidates": scoring["candidates"], **ranking}
            else:
                selected, audit, ranks, direct, tracks, supports, _ = legacy.decide_fork(
                    frozen_route, module, state, native_option, area, config, neutral, weights,
                    read(INCREMENT / "PROTOCOL.json")["supported_hidden"])
                evidence = {"candidates": audit["candidates"], "rankings": ranks, "track_support": tracks,
                            "supported_hidden_by_route": supports, "direct_observations": direct}
                require(evidence == read(candidates_file)
                        and digest(native_option) == canonical["native_option_sha256"]
                        and selected["D"][0]["route_id"] == canonical["arms"]["D"]["route_id"]
                        and selected["S"][0]["route_id"] == canonical["arms"]["S"]["route_id"]
                        and packet.sha256() == canonical["fork_packet_sha256"]
                        and prefix_chain == canonical["prefix_packet_chain_sha256"]
                        and prefix_sha == canonical["prefix_observation_sha256"]
                        and fork["belief_sha256"] == canonical["belief_sha256"]
                        and fork["native_mesh_fields"] == canonical["mesh_fields"],
                        "Earlier shelf raw state, candidate supports, ranks or mesh did not replay")
            row = next(r for r in audit["candidates"] if r["route_id"] == route_id and r["admitted"])
            bound = module._same_goal_option(native_option, row)
            native.check_local_sidecar(native_option, bound, state)
            route_fields = ("outbound_states", "outbound_actions", "return_states", "return_actions")
            require(all(bound[key] == route[key] for key in route_fields)
                    and row["outbound_sha256"] == route["outbound_sha256"]
                    and digest({k: bound[k] for k in route_fields} | {"oriented_target": bound["pose"]}) ==
                        route["route_identity_sha256"],
                    "Route actions or exact native return certificate differ from frozen branch")
            trigger = dict(status="triggered", arm="sealed_outbound", paid_before=paid,
                           option_id=native_option["option_id"], target_pose=native_option["pose"],
                           selection_call_id=native_option["selection_call_id"], actual_route_id=route_id,
                           actual_outbound_sha256=route["outbound_sha256"],
                           candidate_pool_sha256=audit["candidate_pool_sha256"],
                           packet_sha256=packet.sha256(), prefix_observation_sha256=prefix_sha,
                           identity_sha256=route["route_identity_sha256"], remaining_budget=state["ledger"].remaining_budget,
                           native_option_sha256=digest(native_option))
            native_writer(out / "fork_candidates.json", evidence)
            native_writer(out / "fork_bank.json", {
                "status": "triggered", "layout": layout, "source": source, "paid_before": paid,
                "shared_collision_budget": BUDGET, "target_pose": trigger["target_pose"],
                "route_identity_sha256": route["route_identity_sha256"], "route_id": route_id,
                "candidate_bank_sha256": sha(bank_path), "candidate_pool_sha256": audit["candidate_pool_sha256"],
                "source_bank_path": placement["source_bank_path"],
                "source_bank_sha256": placement["source_bank_sha256"],
                "source_candidates_sha256": placement["source_candidates_sha256"],
                "fork_packet_sha256": packet.sha256(), "prefix_packet_chain_sha256": prefix_chain,
                "prefix_observation_sha256": prefix_sha, "belief_sha256": fork["belief_sha256"],
                "mesh_fields": fork["native_mesh_fields"], "continuation_policy": "G_perf",
                "return_actions": bound["return_actions"], "return_states": bound["return_states"],
                "actual_outbound_sha256": route["outbound_sha256"]})
            return bound, audit | {"selected_mode": "sealed_outbound", "selected_route_id": route_id,
                                    "chosen_outbound_sha256": route["outbound_sha256"]}

        module.choose_route = choose
        return module

    native.setup_runtime, native.read, native.load_source, native.sha = setup, conceal_published_actor_freeze, load, identify_episode_freeze
    native.local_route_adapter = lambda *args: legacy.full_local_route_adapter(*args, original_bind)
    try:
        native.run("holdout", layout, "G", source_settings, out)
    finally:
        if native_writer is not None:
            import runtime_binding
            runtime_binding.write = native_writer
    require(trigger is not None and fork is not None, "Registered observed fork did not happen")
    manifest = read(internal_manifest)
    internal_manifest.unlink()
    rows = read(out / "steps.json")
    require(len(rows) == manifest["paid_actions"] == len(raw) - 1, "Native output lost a sensor/action")
    for row, observation in zip(rows, raw[1:], strict=True):
        require(row["step"] == observation["step"] and row["packet_sha256"] == observation["packet_sha256"],
                "Native action and raw sensor records disagree after execution")
        row.update(observation)
    if source == "increment":
        for old_row, row in zip(canonical["prefix_trace"], rows[:placement["paid_before"]], strict=True):
            require(all(old_row[key] == row[key] for key in ("step", "packet_sha256", "action", "rgb_values_sha256",
                        "depth_values_sha256", "scan_ranges_sha256", "collision", "guard_reason", "owner_sha256")),
                    f"Archived shared prefix differs at native paid {row['step']}")
    event = legacy.route_event(rows, trigger, manifest["observed_native_termination"], read(out / "audit.json"))
    require(event["actions"] == ensure_bound_execution(rows, trigger, route),
            "Native route event and charged bound-action count disagree")
    local = read(out / "local_choices.json")
    require(sum(e["choice_audit"]["selected_mode"] == "sealed_outbound" for e in local) == 1
            and all(e["choice_audit"]["selected_mode"] in ("sealed_outbound", "G") for e in local),
            "More than one local choice used the route intervention")
    with legacy.np.load(out / "observed_belief_trace.npz", allow_pickle=False) as data:
        short_receipt = compare_short_witness(layout, route, rows, data["belief"], event, raw[0], fork)
    write(out / "steps.json", rows)
    write(out / "initial_raw_rgbd_scan_sha256.json", raw[0])
    write(out / "mechanism_trigger.json", trigger)
    write(out / "trigger_probes.json", probes)
    write(out / "cache_lifetime.json", {"mode": "gc before invalidated native template cache", "collections": collections})
    manifest["checkpoints"]["fork"] = fork
    for paid in CAPTURE_CLOCKS:
        name = f"paid_{paid}"
        if name not in manifest["checkpoints"]:
            require(paid > manifest["paid_actions"], f"Missing executed paid checkpoint {paid}")
            manifest["checkpoints"][name] = legacy.missing(out, name, paid, manifest["paid_actions"],
                                                            "native_stop_before_checkpoint")
    write(out / "branch_receipt.json", {
        "mode": "one sealed native outbound, then original neutral G_perf through native stop",
        "bank_sha256": bank_sha, "source": source, "source_actor_freeze_sha256": sha(input_freeze),
        "source_bank_sha256": placement["source_bank_sha256"], "route": route_id,
        "route_identity_sha256": route["route_identity_sha256"], "route_event": event,
        "archived_short_replay": short_receipt, "exact_guard": True,
        "primary_task_paid": 96, "native_budget": BUDGET, "original_global_target_retained": True,
        "evaluator_inputs_read_by_actor": False})
    manifest.update(base_commit=BASE, placement=layout, route_id=route_id, route_event=event,
                    study_freeze_sha256=bank_sha, run_kind="stage1_sealed_same_goal_outbound_closure",
                    applied_local_policy=trigger, continuation_policy="G_perf", forced_route_or_goal=True,
                    forced_same_goal_outbound_only=True, original_global_target_retained=True,
                    initial_owner_sha256=initial["initial_owner_sha256"], source_kind=source,
                    archive_short_replay=short_receipt, runner_source_sha256=sha(__file__),
                    files_sha256={p.relative_to(out).as_posix():sha(p) for p in sorted(out.rglob("*"))
                                  if p.is_file() and p.name != "manifest.json"})
    write(out / "manifest.json", manifest)


def main():
    faulthandler.enable()
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--bank", required=True, type=Path)
    cli.add_argument("--placement", required=True)
    cli.add_argument("--route", required=True)
    cli.add_argument("--out", required=True, type=Path)
    args = cli.parse_args()
    out = args.out.resolve()
    require(out.is_relative_to(HERE.parent) and not out.is_relative_to(HERE),
            "Use a new run output path under the current development study")
    out.mkdir(parents=True, exist_ok=False)
    try:
        execute(args.bank.resolve(), args.placement, args.route, out)
    except BaseException as error:
        write(out / "failure.json", {"status": "technical_failure", "error": repr(error),
                                    "placement": args.placement, "route": args.route,
                                    "traceback": traceback.format_exc()})
        raise


if __name__ == "__main__":
    main()
