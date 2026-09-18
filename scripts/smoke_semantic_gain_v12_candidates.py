#!/usr/bin/env python3
"""Runtime preflight for multi-asset V12 candidates; executes no candidate."""
import argparse
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from scipy.sparse.csgraph import dijkstra

from env.virtual3d_inspection_v4 import InspectionConfigV4, InspectionWorldV4
from nso.components import NSO_Components
from nso.competition_candidates_v8 import measured_assets
from nso.competition_prior_v8 import score_routes
from nso.cpu_sensor_contract_v10 import GridTransform, SensorPacket, json_value
from nso.response_candidates_v7 import _actions, _path
from nso.route_coverage_v2 import orientation_graph
from nso.runtime_integration import NSORuntimeIntegration
from nso.semantic_gain_v12 import candidate_set_features, required_candidate_capacity


ROOT = Path(__file__).resolve().parents[1]


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path, value):
    Path(path).write_text(json.dumps(json_value(value), ensure_ascii=False, indent=2,
        sort_keys=True, allow_nan=False) + "\n")


def external_prefix_states(world, *, anchor_sweep=False):
    """Shared evaluator-defined front-view tour ending at a corridor anchor."""
    graph, cells, ids = orientation_graph(world.reachable)
    front = sorted([(*item["front_pose"][0], item["front_pose"][1])
                    for item in world.inspection_truth], key=lambda pose: (pose[1], pose[0]))
    anchor = (world.start[0], world.shape[1] // 2, 1)
    if not world.reachable[anchor[:2]]:
        raise RuntimeError("declared corridor anchor is not reachable")
    waypoints = [(*world.start, world.heading), *front, anchor]
    states = []
    for start_pose, goal_pose in zip(waypoints, waypoints[1:]):
        start = int(ids[start_pose[:2]]) * 4 + start_pose[2]
        goal = int(ids[goal_pose[:2]]) * 4 + goal_pose[2]
        distances, predecessor = dijkstra(
            graph, directed=True, indices=start, return_predecessors=True)
        if not np.isfinite(distances[goal]):
            raise RuntimeError("external prefix waypoint is unreachable")
        leg = _path(predecessor, start, goal, cells)
        states.extend(leg if not states else leg[1:])
    if anchor_sweep:
        row, col, heading = states[-1]
        for _ in range(4):
            heading = (heading + 1) % 4
            states.append((row, col, heading))
    return states, waypoints, anchor


def packet(world, transform, action, action_id):
    frame, scan = world.sense(), world.scan()
    return SensorPacket("D12-00", "v12-candidate-smoke", f"frame-{action_id}",
        action_id, frame, scan, tuple(map(int, world.position)), int(world.heading),
        "deterministic_inspection_rgbd_and_planar_scan",
        "simulator_exact_discrete_odometry", action, False, False).validate(
            transform, world.config)


def main(output):
    if output.exists():
        raise FileExistsError(f"refusing to overwrite {output}")
    output.mkdir(parents=True)
    protocol_path = ROOT / "configs/virtual3d/semantic_gain_v12_scale_development.json"
    protocol = json.loads(protocol_path.read_text())
    declared = protocol["contexts"][0]
    settings = {**protocol["shared_conditions"],
                **{key: value for key, value in declared.items() if key not in ("id", "seed")}}
    world = InspectionWorldV4(InspectionConfigV4(**settings), seed=declared["seed"],
                              semantic_condition="aligned")
    transform = GridTransform(tuple(world.shape), world.config.resolution_m)
    states, waypoints, anchor = external_prefix_states(world)
    actions = _actions(states)
    if len(actions) >= world.config.max_steps:
        raise RuntimeError("external prefix consumes the whole task budget")
    packets = [packet(world, transform, None, 0)]
    for index, action in enumerate(actions, 1):
        _, collision, done = world.step(action)
        if collision or done:
            raise RuntimeError(f"invalid external prefix at action {index}")
        packets.append(packet(world, transform, action, index))
    if (*world.position, world.heading) != anchor:
        raise RuntimeError("external prefix did not end at its declared anchor")

    declared_assets = 2 * int(declared["bays_per_side"])
    # Runtime does not know how many measured components will survive the
    # observed-geometry filter until after prefix fusion.  Reserve the declared
    # V12 maximum here; production V12 will compute the same rule dynamically.
    candidate_cap = required_candidate_capacity(24, coverage_slots=4)
    runtime_args = SimpleNamespace(nso_backend="cpu_v10", eval=True, train_global=False,
        use_open_vocab_semantic=True, use_topo_graph=True, use_rpn_uq=True, use_igcr=True,
        cpu_score_mode="S", cpu_disable_feedback=False, cpu_max_candidates=candidate_cap,
        cpu_coverage_slots=4, cpu_planner_revision="v10_3",
        cpu_measured_novelty_floor=.25, run_id="v12-candidate-smoke")
    components = NSO_Components(runtime_args)
    components.initialize("cpu", 1, *world.shape, *world.shape)
    runtime = NSORuntimeIntegration(components, 1, world.shape)
    runtime.start_sensor_episode(0, config=world.config, transform=transform,
        packets=packets, total_budget=world.config.max_steps, return_anchor=anchor,
        paid_prefix_actions=len(actions))
    components.select_topo_target(0, len(actions))
    selection = components._cpu_backend.scenes[0]["last_selection"]
    mapper = runtime.states[0]["mapper"]
    assets = measured_assets(mapper)
    fixed = score_routes(mapper, selection["candidates"],
                         components._cpu_backend.scenes[0]["ledger"].planning_camera_poses())
    feature_rows = []
    for route, prediction in zip(selection["candidates"], fixed["audit"]):
        semantic, geometry, audit = candidate_set_features(prediction, route, assets)
        feature_rows.append({"candidate_id": route["candidate_id"], "group": route["group"],
            "asset_index": route["asset_index"], "semantic": semantic, "geometry": geometry,
            "feature_audit": audit})

    observed_asset_ids = sorted({int(route["asset_index"]) for route in selection["candidates"]
                                 if route["asset_index"] is not None})
    semantic_asset_ids = [index for index, asset in enumerate(assets)
                          if int(asset["marked_points"]) > 0]
    omitted = selection["candidate_audit"]["omitted_by_cap"]
    missing = selection["candidate_audit"]["missing_roles"]
    missing_semantic = [role for role in missing if any(
        role.startswith(f"asset_{index}_") for index in semantic_asset_ids)]
    result = {
        "schema_version": "semantic_gain_v12_candidate_runtime_smoke/1",
        "status": "passed" if (len(semantic_asset_ids) == declared_assets and not omitted
            and not missing_semantic and set(semantic_asset_ids) <= set(observed_asset_ids)) else "failed",
        "scope": "development candidate generation and feature preflight; no candidate executed",
        "context": declared["id"],
        "protocol_sha256": sha(protocol_path),
        "source_sha256": {name: sha(ROOT / name) for name in (
            "nso/semantic_gain_v12.py", "nso/hierarchical_options_v10.py",
            "nso/response_features_v7.py", "nso/cpu_four_modules_v10.py",
            "scripts/smoke_semantic_gain_v12_candidates.py")},
        "external_prefix": {"paid_actions": len(actions), "frames": len(packets),
            "waypoints": waypoints, "return_anchor": anchor,
            "uses_evaluator_reachability": True, "shared_across_future_methods": True},
        "declared_asset_count": declared_assets,
        "measured_asset_count": len(assets),
        "measured_assets": [{key: (value.tolist() if isinstance(value, np.ndarray) else value)
            for key, value in asset.items() if key in ("group", "aabb_center", "observed_span",
                "support_points", "support_m2_proxy", "marked_points", "class_vote")}
            for asset in assets],
        "semantic_asset_indices": semantic_asset_ids,
        "candidate_cap": candidate_cap,
        "candidate_count": len(selection["candidates"]),
        "candidate_roles": [route["group"] for route in selection["candidates"]],
        "represented_asset_indices": observed_asset_ids,
        "missing_roles": missing,
        "missing_semantic_roles": missing_semantic,
        "omitted_by_cap": omitted,
        "remaining_action_budget": world.config.max_steps - len(actions),
        "feature_rows": feature_rows,
        "planner_outcomes_generated": 0,
        "truth_used_for_candidate_selection": False,
        "truth_used_for_external_prefix_only": True,
    }
    write(output / "result.json", result)
    write(output / "artifact_hashes.json", {"result.json": sha(output / "result.json")})
    print(json.dumps({key: result[key] for key in ("status", "measured_asset_count",
        "candidate_cap", "candidate_count", "semantic_asset_indices",
        "represented_asset_indices", "missing_roles", "missing_semantic_roles",
        "omitted_by_cap", "remaining_action_budget")}, indent=2))
    if result["status"] != "passed":
        raise RuntimeError("V12 candidate runtime smoke failed; retained result.json")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    main(args.output.resolve())
