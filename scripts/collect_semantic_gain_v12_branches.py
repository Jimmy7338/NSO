#!/usr/bin/env python3
"""Collect physical candidate outcomes for one V12 development context."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from env.virtual3d_inspection_v4 import InspectionConfigV4, InspectionWorldV4
from nso.components import NSO_Components
from nso.competition_candidates_v8 import measured_assets
from nso.competition_prior_v8 import score_routes
from nso.cpu_sensor_contract_v10 import GridTransform, json_value
from nso.execution_guard_v8 import ObservedExecutionGuard
from nso.runtime_integration import NSORuntimeIntegration
from nso.observed_runtime_mapper_v10 import ObservedRuntimeMapperV10
from nso.semantic_gain_v12 import candidate_set_features, required_candidate_capacity
from scripts.eval_counterfactual_views import coverage
from scripts.smoke_semantic_gain_v12_candidates import external_prefix_states, packet
from utils.grid_geometry import DIRECTIONS
from utils.reconstruction_metrics import ReconstructionEvaluator


ROOT = Path(__file__).resolve().parents[1]


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path, value):
    Path(path).write_text(json.dumps(json_value(value), ensure_ascii=False, indent=2,
        sort_keys=True, allow_nan=False) + "\n")


def metrics(mapper, world, evaluator):
    fraction = coverage(mapper, world.reachable)
    result = evaluator.evaluate(mapper.mesh(), fraction, thresholds=(.05, .10))
    area = fraction * int(world.reachable.sum()) * world.config.resolution_m ** 2
    result.update(coverage_2d=fraction, covered_area_m2=area,
                  area_times_f1_05cm=area * result["f1_05cm"],
                  area_times_f1_10cm=area * result["f1_10cm"])
    return result


def successor(state, action):
    row, col, heading = state
    if action == "forward":
        dr, dc = DIRECTIONS[heading]
        return int(row + dr), int(col + dc), heading
    return row, col, (heading + (1 if action == "right" else -1)) % 4


def execute_candidate(prefix_packets, shape, config, seed, anchor, route, before, evaluator):
    # Open3D's ScalableTSDFVolume is aliased by Python deepcopy.  A fresh mapper
    # and complete sensor replay are required for physically independent branches.
    mapper = ObservedRuntimeMapperV10(shape, config)
    for prefix_packet in prefix_packets:
        mapper.update(prefix_packet.frame, prefix_packet.scan)
    world = InspectionWorldV4(config, seed=seed, semantic_condition="aligned")
    world.position, world.heading = anchor[:2], anchor[2]
    world.step_count = before["prefix_paid_actions"]
    guard = ObservedExecutionGuard(config.resolution_m, config.robot_radius_m)
    failure = None
    mode = "route"
    route_cursor = 0
    denial_reason = None
    terminal = None
    budget = int(route["cost"])
    while terminal is None:
        current = (*world.position, world.heading)
        paid = int(world.step_count - before["prefix_paid_actions"])
        remaining = budget - paid
        return_now = guard.return_plan(mapper.belief, world.position, world.heading,
                                       anchor, remaining)
        if not return_now.available:
            failure = "return_unavailable:" + return_now.reason
            terminal = failure
            break
        if mode == "route" and route_cursor == len(route["actions"]):
            terminal = "original_route_completed" if current == tuple(anchor) else "route_ended_away"
            failure = None if current == tuple(anchor) else terminal
            break
        if mode == "return" and current == tuple(anchor):
            terminal = "returned_after_route_denial"
            break
        if remaining == 0:
            failure = None if current == tuple(anchor) else "budget_exhausted_away_from_anchor"
            terminal = "budget_exhausted"
            break
        action = route["actions"][route_cursor] if mode == "route" else return_now.actions[0]
        assessment = guard.assess(mapper.belief, world.position, world.heading,
                                  action, remaining)
        denied = not assessment.allowed
        reason = assessment.reason
        if not denied:
            next_state = successor(current, action)
            reserve = guard.return_plan(mapper.belief, next_state[:2], next_state[2],
                                        anchor, remaining - 1)
            denied = not reserve.available
            reason = "successor_return:" + reserve.reason
        if denied:
            if mode == "route":
                mode = "return"
                denial_reason = reason
                continue
            failure = "return_action_denied:" + reason
            terminal = failure
            break
        frame, collision, done = world.step(action)
        scan = world.scan()
        mapper.update(frame, scan)
        if mode == "route":
            route_cursor += 1
        if collision or done:
            failure = "collision" if collision else "unexpected_world_terminal"
            terminal = failure
            break
    returned = (*world.position, world.heading) == tuple(anchor)
    after = metrics(mapper, world, evaluator)
    paid = int(world.step_count - before["prefix_paid_actions"])
    return {
        "candidate_id": route["candidate_id"], "group": route["group"],
        "asset_index": route["asset_index"], "planned_cost": route["cost"],
        "paid_actions": paid, "failure": failure, "collision_count": world.collisions,
        "returned_to_anchor": returned, "terminal": terminal,
        "original_route_completed": terminal == "original_route_completed",
        "route_denial_reason": denial_reason,
        "returned_after_route_denial": terminal == "returned_after_route_denial",
        "before": before["metrics"], "after": after,
        "joint_gain": after["area_times_f1_05cm"] - before["metrics"]["area_times_f1_05cm"],
        "joint_gain_per_action": ((after["area_times_f1_05cm"]
            - before["metrics"]["area_times_f1_05cm"]) / paid if paid else None),
        "coverage_gain_m2": after["covered_area_m2"] - before["metrics"]["covered_area_m2"],
        "f1_gain_05cm": after["f1_05cm"] - before["metrics"]["f1_05cm"],
    }


def main(output, context_id, limit):
    if output.exists():
        raise FileExistsError(f"refusing to overwrite {output}")
    output.mkdir(parents=True)
    scene_protocol_path = ROOT / "configs/virtual3d/semantic_gain_v12_scale_development.json"
    branch_protocol_path = ROOT / "configs/virtual3d/semantic_gain_v12_branch_development.json"
    protocol = json.loads(scene_protocol_path.read_text())
    branch_protocol = json.loads(branch_protocol_path.read_text())
    declared = next(row for row in protocol["contexts"] if row["id"] == context_id)
    sources = ("scripts/collect_semantic_gain_v12_branches.py",
        "scripts/smoke_semantic_gain_v12_candidates.py", "nso/semantic_gain_v12.py",
        "nso/hierarchical_options_v10.py", "nso/cpu_four_modules_v10.py",
        "nso/observed_runtime_mapper_v10.py", "env/virtual3d_inspection_v4.py")
    manifest = {"schema_version": "semantic_gain_v12_branch_collection/1",
        "status": "running", "created_utc": datetime.now(timezone.utc).isoformat(),
        "development_only": True, "context": context_id,
        "scene_protocol_sha256": sha(scene_protocol_path),
        "branch_protocol_sha256": sha(branch_protocol_path),
        "source_sha256": {name: sha(ROOT / name) for name in sources},
        "worlds_constructed_before_source_freeze": 0,
        "limit": limit, "claim_boundary": "development candidate outcomes only"}
    write(output / "manifest.json", manifest)

    settings = {**protocol["shared_conditions"],
                **{key: value for key, value in declared.items() if key not in ("id", "seed")}}
    config = InspectionConfigV4(**settings)
    world = InspectionWorldV4(config, seed=declared["seed"], semantic_condition="aligned")
    transform = GridTransform(tuple(world.shape), world.config.resolution_m)
    states, waypoints, anchor = external_prefix_states(world, anchor_sweep=True)
    from nso.response_candidates_v7 import _actions
    actions = _actions(states)
    packets = [packet(world, transform, None, 0)]
    for index, action in enumerate(actions, 1):
        _, collision, done = world.step(action)
        if collision or done:
            raise RuntimeError(f"invalid prefix action {index}")
        packets.append(packet(world, transform, action, index))

    args = SimpleNamespace(nso_backend="cpu_v10", eval=True, train_global=False,
        use_open_vocab_semantic=True, use_topo_graph=True, use_rpn_uq=True, use_igcr=True,
        cpu_score_mode="S", cpu_disable_feedback=False,
        cpu_max_candidates=branch_protocol["candidate_cap"], cpu_coverage_slots=4,
        cpu_planner_revision="v10_3", cpu_measured_novelty_floor=.25,
        run_id=f"v12-branches-{context_id}")
    components = NSO_Components(args)
    components.initialize("cpu", 1, *world.shape, *world.shape)
    runtime = NSORuntimeIntegration(components, 1, world.shape)
    runtime.start_sensor_episode(0, config=config, transform=transform, packets=packets,
        total_budget=config.max_steps, return_anchor=anchor, paid_prefix_actions=len(actions))
    components.select_topo_target(0, len(actions))
    selection = components._cpu_backend.scenes[0]["last_selection"]
    mapper = runtime.states[0]["mapper"]
    assets = measured_assets(mapper)
    fixed = score_routes(mapper, selection["candidates"],
        components._cpu_backend.scenes[0]["ledger"].planning_camera_poses())
    feature_rows = []
    for route, prediction in zip(selection["candidates"], fixed["audit"]):
        semantic, geometry, audit = candidate_set_features(prediction, route, assets)
        feature_rows.append({"candidate_id": route["candidate_id"], "semantic": semantic,
            "geometry": geometry, "feature_audit": audit})
    evaluator = ReconstructionEvaluator(world,
        count=branch_protocol["reference_surface_samples"],
        seed=branch_protocol["reference_seed"])
    before_metrics = metrics(mapper, world, evaluator)
    before = {"prefix_paid_actions": len(actions), "metrics": before_metrics}
    selected_routes = selection["candidates"] if limit is None else selection["candidates"][:limit]
    outcomes = []
    for route in selected_routes:
        row = execute_candidate(packets, world.shape, config, declared["seed"], anchor,
                                route, before, evaluator)
        outcomes.append(row)
        write(output / "partial_outcomes.json", outcomes)
        print(route["candidate_id"], route["group"], row["joint_gain_per_action"],
              row["failure"], flush=True)

    write(output / "prefix.json", {"paid_actions": len(actions), "frames": len(packets),
        "waypoints": waypoints, "anchor": anchor,
        "anchor_sweep_actions": branch_protocol["anchor_sweep_actions"],
        "metrics": before_metrics})
    write(output / "candidates.json", selection["candidates"])
    write(output / "candidate_audit.json", selection["candidate_audit"])
    write(output / "predictions.json", fixed)
    write(output / "features.json", feature_rows)
    write(output / "outcomes.json", outcomes)
    passed = all(row["failure"] is None and row["collision_count"] == 0
                 and row["returned_to_anchor"] and 0 < row["paid_actions"] <= row["planned_cost"]
                 for row in outcomes)
    write(output / "summary.json", {"status": "passed" if passed else "failed",
        "context": context_id, "candidate_count": len(selection["candidates"]),
        "executed_candidates": len(outcomes), "all_executed_safe": passed,
        "complete_context": len(outcomes) == len(selection["candidates"]),
        "planner_outcomes_generated": len(outcomes)})
    manifest["status"] = "complete"
    write(output / "manifest.json", manifest)
    current = {name: sha(ROOT / name) for name in sources}
    if current != manifest["source_sha256"]:
        raise RuntimeError("source changed during V12 branch collection")
    write(output / "artifact_hashes.json", {str(path.relative_to(output)): sha(path)
        for path in sorted(output.iterdir()) if path.is_file()
        and path.name != "artifact_hashes.json"})
    if not passed:
        raise RuntimeError("retained unsafe or incomplete V12 branch")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--context", default="D12-00")
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()
    main(args.output.resolve(), args.context, args.limit)
