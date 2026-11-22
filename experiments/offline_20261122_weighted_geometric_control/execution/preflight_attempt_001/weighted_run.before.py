#!/usr/bin/env python3
"""Run one declared 192-action cold start through the frozen closed-loop actor."""

import argparse
from dataclasses import asdict
import json
from pathlib import Path
import sys
import time
import traceback

HERE = Path(__file__).resolve().parent
STUDY = HERE.parent
ANALYSES = STUDY.parent
PRIOR = ANALYSES / "same_goal_cold_holdout_20261121"
NEW_SCENES = STUDY / "scenes"
PROTOCOL = STUDY / "PROTOCOL.json"
sys.dont_write_bytecode = True
sys.path.insert(0, str(PRIOR / "runner"))

from cold_run import full_local_route_adapter, native, raw_sensor_hashes
from execute import missing
from weighted_score import Weights, choose_route

ORIGINAL_READ, ORIGINAL_SHA, ORIGINAL_SETUP = native.read, native.sha, native.setup_runtime
ORIGINAL_LOAD, ORIGINAL_BIND = native.load_source, native.local_route_adapter
SOURCE_PROTOCOL = ANALYSES / "same_goal_route_choice_20261121/PROTOCOL.json"
NATIVE_BUDGET = 192
INITIAL_FIELDS = ("layout", "scene_id", "anchor", "initial_packet_sha256", "initial_owner_sha256", "stop_budget")


def read(path):
    return json.loads(Path(path).read_text())


def check_launch(layout, arm, split, prior, weights, freeze_path):
    frozen = read(freeze_path)
    if frozen["split"] != split:
        raise ValueError("The episode must use its declared development or holdout freeze")
    if any(ORIGINAL_SHA(path) != expected for path, expected in frozen["files_sha256"].items()):
        raise ValueError("Actor input or runner source changed since freeze")
    matched = [row for row in frozen["schedule"] if row["layout"] == layout and row["arm"] == arm]
    if len(matched) != 1:
        raise ValueError("Episode not uniquely declared in the fixed run schedule")
    job = matched[0]
    if any(job[key] != value for key, value in dict(prior=prior, **asdict(weights)).items()):
        raise ValueError("Local geometric coefficients or prior differ from the fixed run schedule")
    if split == "holdout" and not frozen.get("weight_freeze_sha256"):
        raise ValueError("A holdout launch requires sealed development weights")
    if any("evaluator_references" in path or "evaluator_truth" in path for path in frozen["files_sha256"]):
        raise ValueError("Truth-bearing file in the actor input list")
    return frozen


def source_run(layout, arm, split, prior, weights, protocol, frozen, freeze_path, out):
    started = time.monotonic()
    actor_initial_path = NEW_SCENES / "prepared" / layout / "initial_state.json"
    actor_initial = read(actor_initial_path)
    filtered_initial = {key: actor_initial[key] for key in INITIAL_FIELDS}
    assert actor_initial["paid_actions"] == actor_initial["policy_select_calls"] == 0
    assert actor_initial["actual_config"]["max_steps"] == NATIVE_BUDGET
    assert actor_initial["stop_budget"] == NATIVE_BUDGET
    assert actor_initial["scene_id"] == "same-goal-holdout-" + layout
    assert actor_initial["scene_role"] == ("known_development" if split == "development" else "fresh_holdout")
    assert ORIGINAL_SHA(actor_initial_path) == frozen["files_sha256"][str(actor_initial_path)]

    raw = []
    writer = None
    runtime_binding = None
    source_manifest = out / ".native_manifest.json"
    pseudo_freeze = HERE / "holdout_freeze.json"
    cohort = list(dict.fromkeys(row["layout"] for row in frozen["schedule"]))
    source = read(SOURCE_PROTOCOL)
    source["heldout"].update(layouts=cohort, paid_action_budget=NATIVE_BUDGET)
    assert int(protocol["native_budget"]) == NATIVE_BUDGET
    assert all(source["local_route"][key] == value for key, value in protocol["frozen_online_fields"].items())
    clocks = protocol["capture_paid"]
    assert clocks == sorted(set(clocks)) and clocks[0] == 0 and clocks[-1] == NATIVE_BUDGET
    source["diagnostic_checkpoints"] = clocks[1:]

    def actor_read(path):
        if Path(path) == actor_initial_path:
            return dict(filtered_initial)
        if Path(path) == pseudo_freeze:
            return frozen
        return ORIGINAL_READ(path)

    def actor_sha(path):
        return ORIGINAL_SHA(freeze_path if Path(path) == pseudo_freeze else path)

    def actor_setup(stage, name, policy, settings):
        nonlocal writer, runtime_binding
        result = ORIGINAL_SETUP(stage, name, policy, settings)
        old, (world, _, first, _, _, _, _), initial, global_mode, local_mode = result
        assert stage == "holdout" and policy == arm and name == layout
        assert global_mode == "G" and local_mode == prior and world.step_count == 0
        assert initial == filtered_initial and world.seed == actor_initial["seed"]
        raw.append(raw_sensor_hashes(first, native.array_sha))
        original_packet = old.packet

        def capture_packet(*args, **kwargs):
            packet = original_packet(*args, **kwargs)
            assert packet.action_id == len(raw)
            raw.append(raw_sensor_hashes(packet, native.array_sha))
            return packet

        old.packet = capture_packet
        import runtime_binding as binding
        runtime_binding = binding
        writer = runtime_binding.write

        def defer_manifest(path, data):
            return writer(source_manifest if Path(path) == out / "manifest.json" else path, data)

        runtime_binding.write = defer_manifest
        return result

    def actor_loader(name, path):
        module = ORIGINAL_LOAD(name, path)
        if Path(path) == native.METHOD:
            policy = module.choose_route

            def policy_with_weights(state, native_option, mode, route_union_area, config):
                assert mode == prior
                return choose_route(module, policy, state, native_option, mode, weights,
                                    route_union_area, config, runtime_binding.revision.reweight)

            module.choose_route = policy_with_weights
        return module

    # Native code is imported in a dedicated worker interpreter and remains
    # unchanged on disk. The adapter replaces just the local selection.
    native.ARM_MODES = {arm: ("G", prior)}
    native.HERE, native.HOLDOUT, native.PROTOCOL = HERE, NEW_SCENES, PROTOCOL
    native.read, native.sha = actor_read, actor_sha
    native.load_source, native.setup_runtime = actor_loader, actor_setup
    native.local_route_adapter = lambda *args: full_local_route_adapter(*args, ORIGINAL_BIND)
    try:
        native.run("holdout", layout, arm, source, out)
    finally:
        if writer is not None:
            runtime_binding.write = writer

    result = read(source_manifest)
    source_manifest.unlink()
    rows = read(out / "steps.json")
    assert len(rows) == result["paid_actions"] == len(raw) - 1 <= NATIVE_BUDGET
    assert raw[0]["packet_sha256"] == result["initial_packet_sha256"]
    for row, sensor in zip(rows, raw[1:], strict=True):
        assert row["step"] == sensor["step"] and row["packet_sha256"] == sensor["packet_sha256"]
        row.update({key: value for key, value in sensor.items()
                    if key.endswith("values_sha256") or key == "scan_ranges_sha256"})
    writer(out / "steps.json", rows)
    writer(out / "initial_raw_rgbd_scan_sha256.json", raw[0])
    for paid in clocks[1:]:
        key = f"paid_{paid}"
        if key not in result["checkpoints"]:
            assert paid > result["paid_actions"]
            result["checkpoints"][key] = missing(out, key, paid, result["paid_actions"], "native_stop_before_checkpoint")
    choices = read(out / "local_choices.json")
    assert all(event["full_pool"]["enabled"] and not event["full_pool"]["unassessed_after_cap"] for event in choices)
    result.update(
        study_split=split, study_prior=prior, **asdict(weights), source_protocol_sha256=ORIGINAL_SHA(PROTOCOL),
        runner_freeze_sha256=ORIGINAL_SHA(freeze_path), published_original_goal_selector_unswitched=True,
        cold_started_at_paid_zero=True, actual_scene_seed=actor_initial["seed"],
        initial_owner_sha256=actor_initial["initial_owner_sha256"],
        frozen_cold_input_sha256=ORIGINAL_SHA(actor_initial_path),
        closed_loop_full_pool=True, all_distinct_local_candidates_assessed=True,
        charged_decision_wall_s=sum(row["decision_wall_s"] for row in rows),
        supplementary_local_neutral_audit_wall_s=sum(
            event["choice_audit"]["weighted_ranking"]["supplemental_neutral_audit_wall_s"] for event in choices),
        bootstrap_and_task_wall_s=time.monotonic() - started,
        files_sha256={str(path.relative_to(out)): ORIGINAL_SHA(path) for path in sorted(out.rglob("*"))
                      if path.is_file() and path.name != "manifest.json"},
    )
    writer(out / "manifest.json", result)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("split", choices=("development", "holdout"))
    parser.add_argument("layout")
    parser.add_argument("arm")
    parser.add_argument("--prior", required=True, choices=("G", "S", "X"))
    parser.add_argument("--unknown-weight", type=float, required=True)
    parser.add_argument("--action-cost", type=float, required=True)
    parser.add_argument("--freeze", type=Path, required=True)
    args = parser.parse_args()
    weights = Weights(args.unknown_weight, args.action_cost)
    freeze_path = args.freeze.resolve()
    frozen = check_launch(args.layout, args.arm, args.split, args.prior, weights, freeze_path)
    out = HERE / "runs" / args.split / args.layout / args.arm
    out.mkdir(parents=True, exist_ok=False)
    try:
        source_run(args.layout, args.arm, args.split, args.prior, weights, read(PROTOCOL), frozen, freeze_path, out)
    except BaseException as error:
        (out / "failure.json").write_text(json.dumps(
            dict(status="failed", study_split=args.split, layout=args.layout, arm=args.arm,
                 prior=args.prior, **asdict(weights), runner_freeze_sha256=ORIGINAL_SHA(freeze_path),
                 error=repr(error), traceback=traceback.format_exc()), indent=2) + "\n")
        raise


if __name__ == "__main__":
    main()
