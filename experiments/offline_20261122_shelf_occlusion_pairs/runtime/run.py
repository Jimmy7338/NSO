#!/usr/bin/env python3
"""Native full 192-action episode with one observed same-goal local intervention."""

import argparse
import faulthandler
import gc
import hashlib
import json
import os
from pathlib import Path
import sys
import time
import traceback

for variable in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[variable] = "1"
sys.dont_write_bytecode = True

HERE = Path(__file__).resolve().parent
STUDY = HERE.parent
ANALYSES = STUDY.parent
LOCAL = ANALYSES / "region_structure_floor_20261122/local"
INCREMENT = ANALYSES / "region_global_revision_20261122/next/local_increment"
WEIGHTED = ANALYSES / "weighted_geometric_control_20261122/runner"
COLD = ANALYSES / "same_goal_cold_holdout_20261121/runner"
SOURCE_PROTOCOL = ANALYSES / "same_goal_route_choice_20261121/PROTOCOL.json"
ARMS = ("G", "S", "D", "D+")
INHERITED_FIELDS = {"max_extra_outbound_actions": 8, "score_equality_tolerance": 1e-9,
                    "min_unknown_union_ratio": 0.95, "azimuth_bins": 8}
SENSOR_FIELDS = ("depth_m", "color_rgb", "semantic", "intrinsic", "world_from_camera")
sys.path[:0] = [str(LOCAL), str(INCREMENT), str(WEIGHTED), str(COLD)]

import numpy as np
from cold_run import full_local_route_adapter, native, raw_sensor_hashes
from direct_inspection import rank_direct
from execute import array_sha, capture, chain_hash, missing
from supported_hidden import SupportedHidden
from weighted_score import Weights, choose_route, rank_admitted


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def digest(data):
    def as_json(value):
        if isinstance(value, np.ndarray):
            return value.tolist()
        if isinstance(value, np.generic):
            return value.item()
        raise TypeError(type(value))

    return hashlib.sha256(json.dumps(data, sort_keys=True, separators=(",", ":"),
                                    allow_nan=False, default=as_json).encode()).hexdigest()


def write(path, value):
    path = Path(path)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    tmp.replace(path)


def validate_frozen_inputs(path, layout, arm):
    frozen = read(path)
    assert frozen["phase"] in ("development", "formal")
    assert layout in frozen["placements"] and arm in ARMS
    assert frozen["native_budget"] == 192
    assert frozen["capture_paid"] == [0, 48, 96, 144, 192]
    assert frozen["max_trigger_paid"] == 64
    assert frozen["weights"] == {"unknown_weight": .25, "action_cost": .20}
    assert frozen["direct_scale"] == 4 and frozen["hidden_scale"] == 1
    for member in ("actor_files_sha256", "simulator_only_cache_sha256"):
        for source, expected in frozen[member].items():
            assert "evaluator_references" not in Path(source).parts
            assert "evaluator_truth" not in Path(source).parts
            if sha(source) != expected:
                raise ValueError(f"Frozen {member} file differs: {source}")
    scene_root = Path(frozen["scene_root"]).resolve()
    for name in ("scene.py", f"prepared/{layout}/initial_state.json", f"prepared/{layout}/initial_frame.npz",
                 f"prepared/{layout}/initial_scan.npz"):
        assert str(scene_root / name) in frozen["actor_files_sha256"]
    return frozen


def route_event(steps, triggered, termination, audit):
    if not triggered:
        return {"status": "not_triggered", "actual_paid": None, "reason": None}
    held = triggered["option_id"]
    paid = triggered["paid_before"]
    start = steps[paid - 1]["path_length_m"] if paid else 0.
    path = start
    reason = next((event.get("reason") for event in audit
                   if event.get("rejected_option_id") == held
                   and event.get("event") == "exploration_joint_replan_after_rejection"), None)
    for row in steps[paid:]:
        if row["active_option_id"] != held:
            status = "guard_cancelled" if reason else "abandoned_or_replanned_before_arrival"
            break
        paid, path = row["step"], row["path_length_m"]
        if row["collision"] or row["matched_target_pose_after_step"]:
            status = "collision" if row["collision"] else "arrived"
            reason = row["guard_reason"]
            break
    else:
        status = "guard_cancelled" if reason else "terminal_before_arrival"
        reason = reason or termination["reason"]
    return dict(status=status, actual_paid=paid, actions=paid-triggered["paid_before"],
                goal=triggered["target_pose"], path_length_m=path-start, reason=reason)


def save_prefix(out, packets):
    path = out / "common_prefix.npz"
    np.savez_compressed(path, paid=np.asarray([p.action_id for p in packets], dtype=np.int16),
                        poses=np.asarray([[*p.position, p.heading] for p in packets], dtype=np.int16),
                        timestamp_s=np.asarray([p.frame.timestamp_s for p in packets]),
                        scan_ranges_m=np.stack([p.scan.ranges_m for p in packets]),
                        world_from_laser=np.stack([p.scan.world_from_laser for p in packets]),
                        **{key: np.stack([getattr(p.frame, key) for p in packets]) for key in SENSOR_FIELDS})
    return sha(path)


def decide_fork(frozen_route, module, state, option, area, config, neutral, weights, hidden):
    """Evaluate all four ranking rules only after the eligibility test passes."""
    import runtime_binding

    start = time.monotonic()
    _, structure = choose_route(module, frozen_route, state, option, "S", weights,
                                area, config, runtime_binding.revision.reweight)
    s_wall = time.monotonic() - start
    assert structure["candidate_pool_sha256"] == neutral["candidate_pool_sha256"]
    rows = structure["candidates"]
    reference = next(row for row in rows if row["route_id"] == module.NATIVE_ID)
    by_id = {row["route_id"]: row for row in rows}
    for row, prior in zip(rows, neutral["candidates"], strict=True):
        assert {k:v for k,v in row.items() if k != "scores"} == {k:v for k,v in prior.items() if k != "scores"}
        assert row["scores"]["G"] == prior["scores"]["G"]

    start = time.monotonic()
    direct_row, direct = rank_direct(state, rows, weights, 4.0, config["score_equality_tolerance"], rank_admitted)
    d_wall = time.monotonic() - start
    direct_area = {row["route_id"]: row["area_m2"] for row in direct["scored_pool"]}
    start = time.monotonic()
    mapper = state["mapper"]
    cache = mapper._dual_template_state["cache"]
    assert cache["predictor"] == "visibility"
    extension = SupportedHidden(mapper, cache["models"], cache["ray"], cache["gains"]["S"],
                                runtime_binding.model.REPAIR.classify_history, hidden)
    route_support = []
    for row in rows:
        if row["admitted"]:
            value, points = extension.score(row["outbound_states"])
            row["scores"].update(D=direct_area[row["route_id"]], **{"D+": direct_area[row["route_id"]] + value})
            route_support.append({"route_id": row["route_id"], "supported_hidden_m2": value,
                                  "supported_hidden_quadrature": points})
    selected, rankings = {}, {}
    for mode in ARMS:
        winner, rankings[mode] = rank_admitted(rows, reference, mode, weights,
                                               mapper.config.resolution_m**2, config["score_equality_tolerance"])
        route = module._same_goal_option(option, winner)
        selected[mode] = (winner, route)
    hidden_wall = time.monotonic() - start
    assert selected["G"][0]["route_id"] == neutral["selected_route_id"]
    assert selected["S"][0]["route_id"] == structure["selected_route_id"]
    assert selected["D"][0]["route_id"] == direct_row["route_id"]
    assert set(by_id) == {row["route_id"] for row in neutral["candidates"]}
    return selected, structure, rankings, direct, extension.track_audit, route_support, {
        "S": s_wall - structure["weighted_ranking"]["supplemental_neutral_audit_wall_s"],
        "S_joint_pool_rebuild": s_wall, "D": d_wall, "D+_increment_and_four_shared_rankings": hidden_wall,
    }


def execute(layout, arm, out, freeze_path, frozen, canonical_bank):
    scene_root = Path(frozen["scene_root"])
    settings = read(SOURCE_PROTOCOL)
    settings["heldout"].update(layouts=frozen["placements"], paid_action_budget=frozen["native_budget"])
    assert all(settings["local_route"][key] == value for key, value in INHERITED_FIELDS.items())
    settings["diagnostic_checkpoints"] = frozen["capture_paid"][1:]
    weights = Weights(**frozen["weights"])
    hidden = read(INCREMENT / "PROTOCOL.json")["supported_hidden"]
    assert hidden["increment_weight"] == frozen["hidden_scale"]
    source_manifest = out / ".native_manifest.json"
    initial = read(scene_root / "prepared" / layout / "initial_state.json")
    initial_packet_sha = initial["initial_packet_sha256"]
    freeze_sha = sha(freeze_path)
    continuation_id = digest({"code": sha(__file__), "freeze": freeze_sha, "global": "native_G",
                              "continuation": "G_perf", "reference": sha(SOURCE_PROTOCOL),
                              "weights": frozen["weights"], "cache": "gc before invalidated template cache"})
    if canonical_bank is not None:
        assert canonical_bank["layout"] == layout and canonical_bank["freeze_sha256"] == freeze_sha
        assert canonical_bank["status"] == "triggered" and canonical_bank["continuation_id"] == continuation_id

    native.HERE, native.HOLDOUT, native.PROTOCOL = HERE, scene_root, STUDY / "PROTOCOL.json"
    native.ARM_MODES = {name: ("G", "G") for name in ARMS}
    original_setup, original_load, original_read, original_bind, original_sha = (
        native.setup_runtime, native.load_source, native.read, native.local_route_adapter, native.sha)
    raw, prefix, probes, collections = [], [], [], []
    prefix_chain = "0" * 64
    shelf_pixels_seen = 0
    fork = trigger = writer = None

    def consume(packet):
        nonlocal prefix_chain, shelf_pixels_seen
        assert packet.action_id == len(raw)
        raw.append(raw_sensor_hashes(packet, array_sha))
        if trigger is None and packet.action_id <= frozen["max_trigger_paid"]:
            prefix.append(packet)
            prefix_chain = chain_hash(prefix_chain, packet.sha256())
            shelf_pixels_seen += int(np.count_nonzero((packet.frame.semantic == 2) & (packet.frame.depth_m > 0)))
        return packet

    def setup(*args):
        nonlocal writer
        value = original_setup(*args)
        old, (world, _, packet, _, _, _, _), _, global_mode, _ = value
        assert global_mode == "G" and world.step_count == 0 and world.seed == initial["seed"]
        assert packet.sha256() == initial_packet_sha
        consume(packet)
        packet_original = old.packet
        old.packet = lambda *a, **kw: consume(packet_original(*a, **kw))

        import runtime_binding
        cache_original = old.method._cache

        def collect_then_cache(mapper, mode):
            cached = mapper._dual_template_state["cache"]
            if cached is None or cached["key"] != (int(mapper.frames), mode):
                collections.append({"mapper_frames": int(mapper.frames), "unreachable_collected": gc.collect()})
            return cache_original(mapper, mode)

        old.method._cache = collect_then_cache
        writer = runtime_binding.write
        runtime_binding.write = lambda path, data: writer(
            source_manifest if Path(path) == out / "manifest.json" else path, data)
        return value

    def actor_read(path):
        if Path(path) == HERE / "holdout_freeze.json":
            return {"world_files_sha256": frozen["actor_files_sha256"]}
        return original_read(path)

    def actor_sha(path):
        return freeze_sha if Path(path) == HERE / "holdout_freeze.json" else original_sha(path)

    def actor_load(name, path):
        module = original_load(name, path)
        if Path(path) != native.METHOD:
            return module
        frozen_route = module.choose_route

        def choose(state, option, mode, area, config):
            nonlocal fork, trigger
            import runtime_binding
            assert mode == "G"
            started = time.monotonic()
            baseline, neutral = choose_route(module, frozen_route, state, option, "G", weights, area,
                                              config, runtime_binding.revision.reweight)
            neutral_wall = time.monotonic() - started
            packet = state["packet"]
            paid = packet.action_id
            if trigger is not None or paid > frozen["max_trigger_paid"]:
                return baseline, neutral
            legal = [row for row in neutral["candidates"] if row["admitted"]]
            distinct = {row["outbound_sha256"] for row in legal}
            assert len(distinct) == len(legal), "native full candidate bank contains duplicate admitted routes"
            for row in legal:
                assert row["outbound_states"][-1] == option["pose"]
            qualified = shelf_pixels_seen > 0 and len(distinct) >= 2
            probes.append({"paid": paid, "paid_before":paid, "selection_call_id": option["selection_call_id"],
                           "seen_valid_shelf_depth_pixels": shelf_pixels_seen,
                           "observed_target_shelf":shelf_pixels_seen > 0,
                           "distinct_legal_outbound_count":len(distinct), "qualifies":qualified,
                           "admitted_distinct_routes": len(distinct),
                           "candidate_pool_sha256": neutral["candidate_pool_sha256"], "qualified": qualified})
            if not qualified:
                return baseline, neutral

            selected, audit, rankings, direct, support, route_support, timing = decide_fork(
                frozen_route, module, state, option, area, config, neutral, weights, hidden)
            assert selected["G"][1] == baseline
            timing["G"] = neutral_wall
            mapper = state["mapper"]
            fork = capture(out, "fork", mapper, packet, prefix_chain, paid, "shared_observed_decision_prefix")
            observation_sha = save_prefix(out, prefix)
            random_state = {str(clock): np.random.default_rng(np.random.SeedSequence(
                [int(initial["seed"]), clock, 739])).bit_generator.state for clock in (paid,paid+1)}
            common = dict(layout=layout, paid_before=paid, initial_packet_sha256=initial_packet_sha,
                          fork_packet_sha256=packet.sha256(), prefix_packet_chain_sha256=prefix_chain,
                          prefix_observation_sha256=observation_sha, belief_sha256=fork["belief_sha256"],
                          mesh_fields=fork["native_mesh_fields"], remaining_budget=state["ledger"].remaining_budget,
                          native_option_sha256=digest(option), candidate_pool_sha256=audit["candidate_pool_sha256"],
                          candidate_scores_sha256=digest(audit["candidates"]), continuation_id=continuation_id,
                          target_pose=option["pose"], pose=[*packet.position, packet.heading],
                          selection_call_id=option["selection_call_id"], scene_seed=initial["seed"],
                          deterministic_sensor_rng=random_state)
            shared = digest(common)
            arm_identities = {}
            for name, (row, route) in selected.items():
                route_sha = digest({"outbound_states": route["outbound_states"],
                                    "outbound_actions": route["outbound_actions"],
                                    "return_states": route["return_states"],"return_actions": route["return_actions"],
                                    "oriented_target": route["pose"]})
                arm_identities[name] = dict(route_id=row["route_id"], outbound_sha256=row["outbound_sha256"],
                    route_identity_sha256=route_sha, outbound_actions=route["outbound_actions"],
                    outbound_states=route["outbound_states"], return_actions=route["return_actions"],
                    return_states=route["return_states"], common_continuation_id=continuation_id,
                    shared_identity_sha256=shared, ranking_wall_s=timing.get(name,
                    timing["D"]+timing["D+_increment_and_four_shared_rankings"]))
            bank = common | {"status":"triggered", "freeze_sha256":freeze_sha, "shared_identity_sha256":shared,
                             "arms":arm_identities, "selected_route_hash_has_return":True,
                             "elapsed_selection_parts_s":timing}
            if canonical_bank is not None:
                assert shared == canonical_bank["shared_identity_sha256"], "independent replay fork differs"
                for name, row in arm_identities.items():
                    assert {k:v for k,v in row.items() if k != "ranking_wall_s"} == {
                        k:v for k,v in canonical_bank["arms"][name].items() if k != "ranking_wall_s"}
            writer(out / "fork_candidates.json", {"candidates":audit["candidates"], "rankings":rankings,
                   "track_support":support, "supported_hidden_by_route":route_support,
                   "direct_observations":direct})
            writer(out / "fork_bank.json", bank)
            row, route = selected[arm]
            trigger = dict(status="triggered", arm=arm, paid_before=paid, option_id=option["option_id"],
                           target_pose=option["pose"], selection_call_id=option["selection_call_id"],
                           actual_route_id=row["route_id"], actual_outbound_sha256=row["outbound_sha256"],
                           shared_identity_sha256=shared, packet_sha256=packet.sha256(),
                           candidate_pool_sha256=audit["candidate_pool_sha256"],
                           candidate_scores_sha256=digest(audit["candidates"]),
                           remaining_budget=common["remaining_budget"],
                           actual_nominal_go_actions=len(route["outbound_actions"]),
                           native_reserved_return_actions=len(route["return_actions"]))
            writer(out / "mechanism_trigger.json", trigger)
            return route, audit | {"selected_mode":arm,"selected_route_id":row["route_id"],
                                   "chosen_outbound_sha256":row["outbound_sha256"],
                                   "score_table_sha256":module._json_sha([r["scores"] for r in audit["candidates"]]),
                                   "paired_first_decision_arm":arm, "weighted_ranking":{
                                       "prior":arm,"scores":rankings[arm],**frozen["weights"]}}

        module.choose_route = choose
        return module

    native.setup_runtime, native.read, native.load_source, native.sha = setup, actor_read, actor_load, actor_sha
    native.local_route_adapter = lambda *args: full_local_route_adapter(*args, original_bind)
    try:
        native.run("holdout", layout, arm,settings,out)
    finally:
        if writer is not None:
            import runtime_binding
            runtime_binding.write = writer
    manifest = read(source_manifest)
    source_manifest.unlink()
    steps = read(out / "steps.json")
    assert len(steps) == manifest["paid_actions"] == len(raw)-1
    assert raw[0]["packet_sha256"] == manifest["initial_packet_sha256"] == initial_packet_sha
    for row, recorded in zip(steps, raw[1:], strict=True):
        assert row["step"] == recorded["step"] and row["packet_sha256"] == recorded["packet_sha256"]
        row.update(recorded)
    write(out / "steps.json", steps)
    write(out / "initial_raw_rgbd_scan_sha256.json", raw[0])
    write(out / "trigger_probes.json", probes)
    write(out / "cache_lifetime.json", {"mode":"gc.collect before stale native cache rebuild", "collections":collections})
    if trigger is None:
        assert canonical_bank is None
        prefix_sha = save_prefix(out,prefix)
        write(out / "fork_bank.json", dict(status="not_triggered", layout=layout, freeze_sha256=freeze_sha,
              arms=None, paid_before=None, continuation_id=continuation_id, eligible_decisions=0,
              observed_prefix_packet_chain_sha256=prefix_chain, prefix_observation_sha256=prefix_sha,
              reason="no_same_goal_legal_opportunity_with_observed_shelf_through_paid_64"))
        write(out / "mechanism_trigger.json", {"status":"not_triggered","arm":arm,"paid_before":None})
    if fork is not None:
        manifest["checkpoints"]["fork"] = fork
    for paid in frozen["capture_paid"][1:]:
        name = f"paid_{paid}"
        if name not in manifest["checkpoints"]:
            assert paid > manifest["paid_actions"]
            manifest["checkpoints"][name] = missing(out,name,paid,manifest["paid_actions"],
                                                      "native_stop_before_checkpoint")
    manifest.update(run_kind="occlusion_pair_single_local_decision", placement=layout, phase=frozen["phase"],
        experimental_local_arm=arm, frozen_actor_files_sha256=freeze_sha, study_freeze_sha256=freeze_sha,
        base_commit=frozen["base_commit"], continuation_policy="G_perf", continuation_id=continuation_id,
        initial_owner_sha256=initial["initial_owner_sha256"], applied_local_policy=trigger,
        independently_reproduced_g_fork=canonical_bank is not None,
        route_event=route_event(steps,trigger,manifest["observed_native_termination"],read(out / "audit.json")),
        decision_total_wall_s=sum(row["decision_wall_s"] for row in steps),
        files_sha256={p.relative_to(out).as_posix():sha(p) for p in sorted(out.rglob("*"))
                      if p.is_file() and p.name != "manifest.json"})
    write(out / "manifest.json", manifest)


def main():
    faulthandler.enable()
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("placement")
    cli.add_argument("arm", choices=ARMS)
    cli.add_argument("--out",required=True,type=Path)
    cli.add_argument("--freeze",required=True,type=Path)
    cli.add_argument("--fork-bank",type=Path)
    cli.add_argument("--allow-unbound-fallback",action="store_true",
                     help="Independently evaluate this arm if the supervised G actor had a technical failure.")
    args = cli.parse_args()
    frozen = validate_frozen_inputs(args.freeze,args.placement,args.arm)
    assert not (args.arm == "G" and (args.fork_bank or args.allow_unbound_fallback))
    assert not (args.fork_bank and args.allow_unbound_fallback)
    assert args.arm == "G" or args.fork_bank or args.allow_unbound_fallback
    canonical = read(args.fork_bank) if args.fork_bank else None
    out = args.out.resolve()
    out.mkdir(parents=True,exist_ok=False)
    try:
        execute(args.placement,args.arm,out,args.freeze,frozen,canonical)
    except BaseException as error:
        write(out / "failure.json",{"status":"technical_failure","error":repr(error),
                                   "traceback":traceback.format_exc()})
        raise


if __name__ == "__main__":
    main()
