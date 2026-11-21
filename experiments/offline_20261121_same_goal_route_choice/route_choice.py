"""Optional local route comparison at an unchanged observed NSO goal."""

import hashlib
import json
from collections import Counter
from numbers import Integral

import numpy as np
from scipy.sparse.csgraph import dijkstra

from nso.response_candidates_v7 import _actions, _path
from nso.route_coverage_v2 import orientation_graph
from utils.grid_geometry import DIRECTIONS, supercover_line

MODES = ("G", "S")
NATIVE_ID = "native"


def _json_sha(value):
    packed = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(packed.encode()).hexdigest()


def _grid_sha(data):
    return hashlib.sha256(np.ascontiguousarray(data).tobytes()).hexdigest()


def _modeled_support(mapper, neutral_model):
    """Geometry representatives from actual instance-associated support only."""
    supported = {int(item["support_track"]) for item in neutral_model.objects}
    rows = list(mapper.quality.items())
    if not rows or not supported:
        return {}
    keys = np.asarray([key for key, _ in rows], dtype=np.int64)
    tracks = mapper.instance_support_index.memberships_for_keys(keys)
    points = np.asarray([row["point"] for _, row in rows], dtype=float)
    observations = {}
    for track in sorted(supported):
        measured = points[tracks == track]
        measured = measured[np.isfinite(measured).all(axis=1)]
        if not len(measured):
            continue
        center = np.median(measured, axis=0)
        # Use the center-nearest measured point and coordinate-extreme measured points.
        # No labels, template samples or posterior weights participate here.
        indices = [int(np.argmin(np.sum((measured - center) ** 2, axis=1)))]
        for axis in range(3):
            indices.extend((int(np.argmin(measured[:, axis])), int(np.argmax(measured[:, axis]))))
        observations[track] = center, measured[list(dict.fromkeys(indices))]
    return observations


def _clear_to_measured_cell(belief, start, end):
    if any(not 0 <= p < size for p, size in zip(end, belief.shape)):
        return False
    line = supercover_line(end[0] - start[0], end[1] - start[1])
    # The target is a measured surface and can occupy its own grid cell.
    return all(belief[start[0] + dr, start[1] + dc] != 1 for dr, dc in line[1:-1])


def _view_tags(mapper, states, support, bins):
    """Return (observed track, object-centric azimuth bin) per oriented pose."""
    if not len(states) or not support:
        return {}
    grid = np.asarray(states, dtype=int)
    cfg = mapper.config
    resolution = float(cfg.resolution_m)
    centers = np.column_stack((
        (grid[:, 1] + 0.5) * resolution,
        (mapper.shape[0] - grid[:, 0] - 0.5) * resolution,
        np.full(len(grid), cfg.camera_height_m),
    ))
    directions = np.asarray(DIRECTIONS)
    forward = directions[grid[:, 2]][:, ::-1] * np.array([1, -1])
    right = np.column_stack((forward[:, 1], -forward[:, 0]))
    tangent = np.tan(np.deg2rad(cfg.fov_deg / 2))
    aspect = cfg.height_px / cfg.width_px
    rows = [set() for _ in states]
    occlusion_cache = {}
    for track, (support_center, samples) in support.items():
        relative = samples[None, :, :] - centers[:, None, :]
        depth = np.einsum("ijk,ik->ij", relative[:, :, :2], forward)
        horiz = np.einsum("ijk,ik->ij", relative[:, :, :2], right)
        frustum = (
            (depth > 0.15) & (depth <= cfg.max_depth_m)
            & (np.abs(horiz) <= depth * tangent) & (np.abs(relative[:, :, 2]) <= depth * tangent * aspect)
        )
        sample_cells = [tuple(map(int, mapper.grid_cell(point))) for point in samples]
        relative_xy = centers[:, :2] - support_center[:2]
        angle = np.arctan2(relative_xy[:, 1], relative_xy[:, 0])
        sectors = np.floor((angle + np.pi) * bins / (2 * np.pi)).astype(int) % bins
        for p in np.flatnonzero(frustum.any(axis=1)):
            start = tuple(grid[p, :2])
            for i in np.flatnonzero(frustum[p]):
                pair = start, sample_cells[i]
                if pair not in occlusion_cache:
                    occlusion_cache[pair] = _clear_to_measured_cell(mapper.belief, *pair)
                if occlusion_cache[pair]:
                    rows[p].add((track, int(sectors[p])))
                    break
    return {tuple(map(int, pose)): tags for pose, tags in zip(states, rows) if tags}


def _local_variants(state, native, support, extra_limit, azimuth_bins):
    mapper = state["mapper"]
    known_safe = state["guard"].safe_grid(mapper.belief)
    graph, cells, ids = orientation_graph(known_safe)
    original = tuple(tuple(map(int, pose)) for pose in native["outbound_states"])
    start_pose, goal_pose = original[0], original[-1]

    def vertex(pose):
        i = int(ids[pose[:2]])
        if i < 0:
            raise ValueError("original route lies outside native known-safe orientation graph")
        return 4 * i + pose[2]

    start, goal = vertex(start_pose), vertex(goal_pose)
    outward, forward = dijkstra(graph, directed=True, indices=start, return_predecessors=True)
    inward, backward = dijkstra(graph.T.tocsr(), directed=True, indices=goal, return_predecessors=True)
    reserve = len(native["return_actions"])
    bound = min(int(state["ledger"].remaining_budget) - reserve, len(original) - 1 + extra_limit)
    if not np.isfinite(outward[goal]) or len(original) - 1 > bound or outward[goal] != len(original) - 1:
        raise ValueError("native outbound cost/path disagrees with original options graph")
    if any(int(ids[pose[:2]]) < 0 for pose in original):
        raise ValueError("native outbound route approaches a non-certified grid pose")

    eligible = np.flatnonzero((outward + inward <= bound) & (outward > 0) & (inward > 0))
    poses = {int(i): (*map(int, cells[i // 4]), int(i % 4)) for i in eligible}
    all_states = list(dict.fromkeys((*original[1:], *poses.values())))
    tags = _view_tags(mapper, all_states, support, azimuth_bins)
    native_tags = set().union(*(tags.get(pose, ()) for pose in original[1:]))
    proposals = []
    seen = {original}
    duplicate_count = 0
    early_goal = 0
    ordered = sorted(
        (i for i in poses if tags.get(poses[i], set()) - native_tags),
        key=lambda i: (int(outward[i] + inward[i]), -len(tags[poses[i]] - native_tags), poses[i]),
    )
    for via in ordered:
        a = _path(forward, start, via, cells)
        b = _path(backward, goal, via, cells, reverse=True)
        path = tuple(tuple(map(int, pose)) for pose in (*a, *b[1:]))
        if path in seen:
            duplicate_count += 1
            continue
        seen.add(path)
        if goal_pose in path[:-1]:
            early_goal += 1
            continue
        if path[0] != start_pose or path[-1] != goal_pose or len(path) - 1 != int(outward[via] + inward[via]):
            raise ValueError("alternate failed to preserve native same-pose endpoint or shortest via-waypoint cost")
        route_tags = set().union(*(tags.get(pose, ()) for pose in path[1:]))
        proposed_tags = route_tags - native_tags
        proposals.append((poses[via], path, proposed_tags))
    return original, native_tags, proposals, {
        "within_cost_waypoint_states": int(len(eligible)),
        "estimated_distinct_view_waypoints": len(ordered),
        "duplicate_traces_skipped": duplicate_count,
        "early_goal_traces_skipped": early_goal,
        "distinct_alternate_traces": len(proposals),
    }


def _candidate(route_id, path, actions, waypoint, tags, reserve, coverage):
    full_trace = {"outbound_actions": list(actions), "outbound_states": [list(pose) for pose in path]}
    return {
        "route_id": route_id, "waypoint_pose": None if waypoint is None else list(waypoint),
        **full_trace, "outbound_sha256": _json_sha(full_trace),
        "goal_pose": list(path[-1]), "outbound_cost": len(full_trace["outbound_actions"]),
        "reserved_return_cost": reserve, "admitted": True, "exclusion_reason": None,
        "novel_view_tags": [list(tag) for tag in sorted(tags)],
        "common_score": 0.0, "scores": {}, **coverage,
    }


def _same_goal_option(native, winner):
    if winner["route_id"] == NATIVE_ID:
        return native
    out_path, out_actions = winner["outbound_states"], winner["outbound_actions"]
    outbound_cost = len(out_actions)
    return {
        **native, "outbound_actions": list(out_actions), "outbound_states": [list(pose) for pose in out_path],
        "states": [list(pose) for pose in (*out_path, *native["return_states"][1:])],
        "actions": [*out_actions, *native["return_actions"]],
        "cost": outbound_cost + len(native["return_actions"]), "outbound_cost": outbound_cost,
        "arrival_action": outbound_cost,
    }


def _choose_candidate(rows, mode, tie_tolerance):
    highest = max(row["scores"][mode] for row in rows)
    near_highest = [row for row in rows if highest - row["scores"][mode] <= tie_tolerance]
    return min(near_highest, key=lambda row: (
        row["outbound_cost"], row["route_id"] != NATIVE_ID, row["outbound_sha256"],
    ))


def choose_route(state, native, mode, route_union_area, config):
    """Select just the outbound trace after the common original G selector.

    The scorer uses the existing full-template route union over observed inputs.
    G and S share the route pool, optical/coverage admission, fallback and tie
    rule. Their only scoring difference is the structural class posterior.
    """
    if mode not in MODES:
        raise ValueError("local route scoring supports G or S")
    if any(not isinstance(config[key], Integral) or isinstance(config[key], bool) or config[key] < 1
           for key in ("max_alternates", "azimuth_bins", "max_extra_outbound_actions")):
        raise ValueError("positive integral local-route caps required")
    min_unknown = float(config["min_unknown_union_ratio"])
    tie_tolerance = float(config["score_equality_tolerance"])
    if not 0 <= min_unknown <= 1:
        raise ValueError("unknown-union floor must be a ratio")
    if not np.isfinite(tie_tolerance) or tie_tolerance < 0:
        raise ValueError("expected a finite nonnegative score equality tolerance")
    mapper = state["mapper"]
    packet = state["packet"]
    anchor = tuple(map(int, state["return_anchor"]))
    if native is None:
        return None, None
    if (tuple(native["outbound_states"][0]) != (*packet.position, packet.heading)
            or list(native["outbound_states"][-1]) != list(native["pose"])
            or list(native["return_states"][0]) != list(native["pose"])
            or tuple(native["return_states"][-1]) != anchor):
        raise ValueError("received an invalid native option or return reservation")
    if native["return_cost"] != len(native["return_actions"]) or native["outbound_cost"] != len(native["outbound_actions"]):
        raise ValueError("native route action lists disagree with its frozen costs")
    if native["cost"] != native["outbound_cost"] + native["return_cost"]:
        raise ValueError("native total cost disagrees with its outbound/return steps")
    if native["cost"] > state["ledger"].remaining_budget:
        raise ValueError("native route has no original budget reserve")
    cache = mapper._dual_template_state["cache"]
    if cache is None or int(cache["key"][0]) != mapper.frames:
        raise ValueError("no frozen G/S posterior from the current completed observation")
    neutral, semantic = (cache["models"][arm] for arm in MODES)
    if neutral.bank is not semantic.bank:
        raise ValueError("G/S must share the same acquired geometric template bank")
    support = _modeled_support(mapper, neutral)
    original = tuple(tuple(map(int, pose)) for pose in native["outbound_states"])
    if support:
        original, tags, proposals, diagnostics = _local_variants(
            state, native, support, config["max_extra_outbound_actions"], config["azimuth_bins"],
        )
    else:
        tags, proposals = set(), []
        diagnostics = {"within_cost_waypoint_states": 0, "estimated_distinct_view_waypoints": 0,
                       "duplicate_traces_skipped": 0, "early_goal_traces_skipped": 0, "distinct_alternate_traces": 0}
    unknown = mapper.belief == -1
    attempted = state["gain"].attempted_camera_mask(mapper, state["ledger"].planning_camera_poses())
    per_pose_masks = {}

    def route_coverage(path):
        radar, camera = np.zeros_like(unknown), np.zeros_like(unknown)
        for pose in path[1:]:
            if pose not in per_pose_masks:
                per_pose_masks[pose] = state["gain"].route_masks(mapper, [pose], attempted)
            rr, cc = per_pose_masks[pose]
            radar |= rr
            camera |= cc
        rr, cc = radar & unknown, camera & unknown
        return {
            "unknown_union_cells": int(np.count_nonzero(rr | cc)),
            "predicted_radar_cells": int(rr.sum()), "predicted_camera_cells": int(cc.sum()),
        }

    native_coverage = route_coverage(original)
    reserve = len(native["return_actions"])
    native_row = _candidate(NATIVE_ID, original, native["outbound_actions"], None, set(), reserve, native_coverage)
    records = [native_row]
    admitted = [native_row]
    exclusions = Counter()
    for ordinal, (via, path, novel) in enumerate(proposals):
        if len(admitted) == config["max_alternates"] + 1:
            exclusions["remaining_candidates_after_admitted_pool_cap"] += len(proposals) - ordinal
            break
        row = _candidate(f"via_{via[0]}_{via[1]}_{via[2]}", path, _actions(path), via, novel, reserve, route_coverage(path))
        if row["unknown_union_cells"] + 1e-12 < min_unknown * native_coverage["unknown_union_cells"]:
            row.update(admitted=False, exclusion_reason="observable_unknown_union_below_native_floor")
            exclusions[row["exclusion_reason"]] += 1
        else:
            admitted.append(row)
        records.append(row)
    for row in admitted:
        for arm in MODES:
            expected_m2, _ = route_union_area(cache["models"][arm], row["outbound_states"][1:],
                                             cache["ray"], cache["gains"][arm])
            if not np.isfinite(expected_m2) or expected_m2 < 0:
                raise ValueError("route template scorer returned invalid predicted surface")
            row["scores"][arm] = float(expected_m2)
    winners = {arm: _choose_candidate(admitted, arm, tie_tolerance) for arm in MODES}
    winner = winners[mode]
    chosen = _same_goal_option(native, winner)
    return chosen, {
        "schema_version": "nso_same_goal_observed_local_route_v1",
        "score_equality_tolerance": tie_tolerance,
        "paid_before": int(packet.action_id), "packet_sha256": packet.sha256(),
        "belief_sha256": _grid_sha(mapper.belief), "native_goal_pose": list(native["pose"]),
        "native_global_candidate_id": int(native["candidate_id"]), "native_option_id": native["option_id"],
        "return_anchor": list(anchor), "native_reserved_return_cost": reserve,
        "original_route_id": NATIVE_ID, "selected_mode": mode, "selected_route_id": winner["route_id"],
        "hypothetical_selected_route_id": {arm: row["route_id"] for arm, row in winners.items()},
        "native_outbound_sha256": native_row["outbound_sha256"], "chosen_outbound_sha256": winner["outbound_sha256"],
        "candidate_pool_sha256": _json_sha([{key: value for key, value in row.items() if key != "scores"} for row in records]),
        "score_table_sha256": _json_sha([row["scores"] for row in records]),
        "assessed_candidate_count": len(records), "admitted_candidate_count": len(admitted),
        "measured_supported_tracks": list(support), "native_view_tags": [list(tag) for tag in sorted(tags)],
        "coverage_guard": {"minimum_native_union_ratio": min_unknown, "native_unknown_union_cells": native_coverage["unknown_union_cells"]},
        "proposal_diagnostics": dict(diagnostics, exclusions=dict(exclusions)), "candidates": records,
    }
