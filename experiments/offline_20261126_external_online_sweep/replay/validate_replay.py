"""Validate archived states with the frozen native and regional route providers."""

import argparse
from collections import Counter
from copy import deepcopy
import hashlib
import json
import sys
from types import FunctionType, SimpleNamespace

import numpy as np
from scipy.ndimage import label
from scipy.sparse.csgraph import dijkstra

from reconstruct_history import ANALYSES, HERE, hash_array, load_history, read


ROOT = HERE.parent
REGIONAL = ANALYSES / "region_structure_floor_20261122"
CROSS = np.array(((0, 1, 0), (1, 1, 1), (0, 1, 0)))
CHECKPOINTS = (17, 38)


def rebind(function, **symbols):
    """Use the unchanged frozen Python body with the new observation graph."""
    bound = FunctionType(function.__code__, function.__globals__ | symbols, function.__name__,
                         function.__defaults__, function.__closure__)
    bound.__kwdefaults__ = function.__kwdefaults__
    return bound


def legacy_graph(native, mapper):
    config = mapper.config
    safe = ~native.inflated_obstacles(mapper.belief != 0, config.robot_radius_m / config.resolution_m)
    return SimpleNamespace(safe=safe, edges=None, orientation_graph=lambda: native.orientation_graph(safe),
                           regions=lambda: label(safe, structure=CROSS))


def configured_source(native, mapper, observed_graph):
    def inflate(obstacle, radius):
        if np.array_equal(obstacle, mapper.belief != 0):
            return ~observed_graph.safe
        return native.inflated_obstacles(obstacle, radius)

    def graph(safe):
        np.testing.assert_array_equal(safe, observed_graph.safe)
        return observed_graph.orientation_graph()

    return SimpleNamespace(inflated_obstacles=inflate, orientation_graph=graph, _path=native._path,
                           _actions=native._actions, camera_pose=native.camera_pose,
                           _fov_support=native._fov_support)


def route_identity(route):
    names = ("pose", "group", "outbound_states", "return_states", "outbound_cost", "return_cost", "cost")
    return {name: route[name] for name in names}


def assert_routes_supported(graph, routes, current, anchor, budget):
    matrix, _, ids = graph.orientation_graph()
    for route in routes:
        out, back = route["outbound_states"], route["return_states"]
        assert tuple(out[0]) == current and out[-1] == route["pose"] and out[-1] == back[0]
        assert tuple(back[-1]) == anchor
        assert len(out) - 1 == route["outbound_cost"] and len(back) - 1 == route["return_cost"]
        assert route["cost"] == len(out) + len(back) - 2 <= budget
        path = out + back[1:]
        for first, second in zip(path, path[1:]):
            a, b = 4 * int(ids[tuple(first[:2])]) + first[2], 4 * int(ids[tuple(second[:2])]) + second[2]
            assert a >= 0 and b >= 0 and matrix[a, b] == 1


def split_routes(native, certify, observed_graph, mapper, packet, anchor, budget, proposals):
    labels, count = observed_graph.regions()
    pose = (*packet.position, packet.heading)
    component, home_component = (int(labels[pos]) for pos in (packet.position, anchor[:2]))
    source = configured_source(native, mapper, observed_graph)
    generate = rebind(native.generate_options, inflated_obstacles=source.inflated_obstacles,
                      orientation_graph=source.orientation_graph)
    routes, candidate_audit = generate(mapper, packet.position, packet.heading, budget, anchor,
        coverage_strategy="total_diverse", coverage_slots=4, attempted_camera_mask=np.zeros(mapper.shape, dtype=bool))
    routes = [r for r in routes if component > 0 and component == home_component == int(labels[tuple(r["pose"][:2])])]
    state = {"mapper": mapper, "packet": packet, "return_anchor": anchor,
             "ledger": SimpleNamespace(remaining_budget=budget),
             "graph": {"regions": labels, "current_region": component, "anchor_region": home_component}}
    added, admission = certify(proposals, routes, state, source)
    assert_routes_supported(observed_graph, [*routes, *added], pose, anchor, budget)
    matrix, _, cells = observed_graph.orientation_graph()
    direct = None
    if cells[packet.position] >= 0:
        start = int(cells[packet.position]) * 4 + packet.heading
        targets = matrix.indices[matrix.indptr[start]:matrix.indptr[start + 1]]
        direct = any(target % 4 == packet.heading for target in targets)
    outbound = []
    if component and home_component:
        start = int(cells[packet.position]) * 4 + packet.heading
        home = int(cells[anchor[:2]]) * 4 + anchor[2]
        go = dijkstra(matrix, directed=True, indices=start)
        back = dijkstra(matrix.T.tocsr(), directed=True, indices=home)
        for item in proposals:
            q = tuple(item["pose"][:2])
            node = int(cells[q]) * 4 + item["pose"][2]
            costs = [int(x[node]) if node >= 0 and np.isfinite(x[node]) else None for x in (go, back)]
            outbound.append({"pose": item["pose"], "start_to_target_actions": costs[0],
                             "target_to_return_anchor_actions": costs[1]})
    return {
        "safe_field_sha256": hash_array(observed_graph.safe),
        "connected_components": int(count), "current_component": component, "anchor_component": home_component,
        "current_component_centres": int(np.count_nonzero(labels == component)) if component else 0,
        "can_continue_in_current_heading": direct, "native_option_status": candidate_audit["status"],
        "native_feasible_oriented_states": candidate_audit["feasible_states"],
        "native_routes": [route_identity(route) for route in routes],
        "certified_regional_routes": [route_identity(route) for route in added],
        "fixed_pool_native_admission": admission,
        "fixed_pool_admission_counts": dict(Counter(row["status"] for row in admission)),
        "fixed_pool_oriented_costs": outbound,
    }


def validate(use_sweep):
    history = load_history(max_paid=max(CHECKPOINTS))
    sys.path.insert(0, str(REGIONAL))
    from nso.observed_runtime_mapper_v10 import ObservedRuntimeMapperV10
    from nso_bridge import exploration_joint_policy as native
    from policy.observed_regions import RegionalPlanner
    from regional_adapter import certify
    if use_sweep:
        sys.path.insert(0, str(ROOT))
        from observed_sweep import SweepEvidence

    run, cold = history["run"], history["cold"]
    regional_calls = {row["paid_before"]: row for row in read(run / "regional_choices.json")}
    historical_native_calls = {row["action_id"]: row["outputs"] for row in read(run / "global_choices.json")}
    anchor = tuple(cold["anchor"])
    budget = cold["stop_budget"]
    config = history["mapper"].config
    mapper = ObservedRuntimeMapperV10(history["mapper"].shape, config, truncation_m=config.truncation_m)
    sweep = SweepEvidence(mapper.shape, config.resolution_m, config.robot_radius_m) if use_sweep else None
    regional = RegionalPlanner()
    data = {
        "schema": "hm3d_online_sweep_fixed_prefix_v1", "layout": cold["layout"],
        "verified_archived_sensor_packets_and_beliefs": len(history["packets"]),
        "path_evidence_source": "same past observed RGB-D, scan, odometry and occupancy only",
        "simulator_motion_executed": False, "new_sensor_pose_queried": False,
        "evaluation_truth_used": False, "formal_episodes_run": 0,
        "inputs_sha256": {
            name: hashlib.sha256((run / name).read_bytes()).hexdigest()
            for name in ("steps.json", "observed_belief_trace.npz", "regional_choices.json", "global_choices.json")
        },
        "checks": [],
    }
    for tick, packet in enumerate(history["packets"]):
        mapper.update(packet.frame, packet.scan)
        if sweep is not None:
            sweep.observe(packet.frame, packet.scan)
        pose = (*packet.position, packet.heading)
        regional.observe(mapper.belief, pose, tick)
        if tick not in regional_calls and tick not in CHECKPOINTS:
            continue
        # Probe p17 without mutating the frozen policy's actual decision history.
        planner = regional if tick in regional_calls else deepcopy(regional)
        proposals = planner.proposals(mapper.belief, pose, tick, config.resolution_m,
            max_depth_m=config.max_depth_m, fov_deg=config.fov_deg, robot_radius_m=config.robot_radius_m)
        recorded = regional_calls.get(tick)
        if recorded:
            assert recorded["start_pose"] == list(pose)
            assert [row["proposal"] for row in recorded["certification"]] == proposals
        if tick not in CHECKPOINTS:
            continue
        original = split_routes(native, certify, legacy_graph(native, mapper), mapper, packet, anchor,
                                budget - tick, proposals)
        if recorded:
            assert original["fixed_pool_native_admission"] == recorded["certification"]
            archived = historical_native_calls[tick]
            assert original["safe_field_sha256"] == archived["candidate_audit"]["safe_sha256"]
            assert original["native_routes"] == [route_identity(route) for route in archived["candidates"]]
        row = {
            "paid": tick, "pose": pose, "packet_sha256": packet.sha256(),
            "observed_belief_sha256": hash_array(mapper.belief),
            "candidate_pool_matches_archived_planner_proposals": True if recorded else None,
            "old_native_routes_and_regional_admission_match_archive": True if recorded else None,
            "baseline_inflated_grid": original,
        }
        if sweep is not None:
            row["adapted_observed_sweep"] = split_routes(native, certify, sweep.build(mapper.belief), mapper,
                                                         packet, anchor, budget - tick, proposals)
        data["checks"].append(row)
    return data


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--legacy-only", action="store_true", help="check the frozen baseline against archived decisions")
    args = parser.parse_args()
    result = validate(use_sweep=not args.legacy_only)
    name = "BASELINE_REPLAY.json" if args.legacy_only else "FIXED_REPLAY.json"
    (HERE / name).write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    for row in result["checks"]:
        print(f"p{row['paid']}: " + "; ".join(f"{k}={v['fixed_pool_admission_counts']}"
              for k, v in row.items() if k in ("baseline_inflated_grid", "adapted_observed_sweep")))
