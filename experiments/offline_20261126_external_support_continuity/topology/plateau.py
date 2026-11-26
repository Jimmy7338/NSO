"""Read-only route/topology audit for archived HM3D 00770 start 1."""

from collections import deque
from heapq import heappop, heappush
import hashlib
from itertools import combinations
import json
import math
from pathlib import Path
import sys

import matplotlib
matplotlib.use("Agg")
from matplotlib import pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm
import numpy as np
from scipy.ndimage import distance_transform_edt
from shapely.geometry import LineString


OUT = Path(__file__).resolve().parent
ARCHIVE = OUT.parents[1] / "external_support_repair_20261126"
SHORT = "b26_hm3d_00770_s1"
START = (39, 17)
DIRECTIONS = ((-1, 0), (0, 1), (1, 0), (0, -1))
ARMS = ("F_strict", "combined_S")
sys.path.insert(0, str(ARCHIVE / "scenes"))
from scene import make_world
import open3d as o3d
from audit_formal_collisions import local_original_triangles
sys.path.insert(0, str(ARCHIVE / "validation"))
from analyse_hm3d_nav_reference import NAV_FILE, read_detour


def neighbors(cell, edges):
    for h, (dr, dc) in enumerate(DIRECTIONS):
        if edges[cell + (h,)]:
            yield cell[0] + dr, cell[1] + dc


def flood(edges, mask, start=START):
    reachable = np.zeros(mask.shape, dtype=bool)
    frontier = deque([start])
    reachable[start] = True
    while frontier:
        for next_cell in neighbors(frontier.popleft(), edges):
            if mask[next_cell] and not reachable[next_cell]:
                reachable[next_cell] = True
                frontier.append(next_cell)
    return reachable


def mask_dist(mask):
    return distance_transform_edt(np.pad(mask, 1))[1:-1, 1:-1]


def largest_components(edges, mask):
    remaining = mask.copy()
    components = []
    for xy in np.argwhere(mask):
        point = tuple(xy)
        if remaining[point]:
            component = flood(edges, remaining, point)
            remaining &= ~component
            components.append(component)
    return sorted(components, key=lambda mask: -mask.sum())


def least_blocked(edges, inflated_safe, belief, destinations):
    costs = {START: (0, 0, 0)}
    parent = {}
    queue = [((0, 0, 0), START)]
    while queue:
        cost, p = heappop(queue)
        if costs[p] != cost:
            continue
        if destinations[p]:
            path = [p]
            while path[-1] != START:
                path.append(parent[path[-1]])
            return cost, path[::-1]
        for q in neighbors(p, edges):
            proposal = (cost[0] + int(not inflated_safe[q]), cost[1]+1,
                        cost[2] + int(belief[q] == -1))
            if q not in costs or proposal < costs[q]:
                costs[q] = proposal
                parent[q] = p
                heappush(queue, (proposal, q))
    raise ValueError("no path through the physically reachable component")


def hash_values(array):
    return hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest()


def cell_of(world, xyz):
    resolution = world.config.resolution_m
    j, i = np.floor(np.asarray(xyz)[:, :2]/resolution).astype(int).T
    return np.column_stack((world.shape[0] - 1 - i, j))


def replay_laser(world, steps):
    evidence = {}
    seen = set()
    checked = 0
    for tick, cell, heading, step in steps:
        packet = cell, heading
        if packet in seen:
            continue
        seen.add(packet)
        world.position, world.heading, world.step_count = cell, heading, tick
        scan = world.scan()
        if step is not None:
            assert hash_values(scan.ranges_m) == step["scan_ranges_sha256"], (tick, cell)
            checked += 1
        angles = scan.angle_min_rad + np.arange(len(scan.ranges_m))*scan.angle_increment_rad
        hit = (scan.ranges_m > 0) & (scan.ranges_m < scan.range_max_m - .001)
        ranges = scan.ranges_m[hit]
        angles = angles[hit]
        local = np.column_stack((np.cos(angles)*ranges, np.sin(angles)*ranges, np.zeros(len(ranges))))
        endpoints = local @ scan.world_from_laser[:3, :3].T + scan.world_from_laser[:3, 3]
        for cell_hit, xyz, distance in zip(cell_of(world, endpoints), endpoints, ranges, strict=True):
            evidence.setdefault(tuple(cell_hit), []).append((tick, cell, float(distance), xyz.tolist()))
    return evidence, checked


def line_of_sight(world, origins, targets):
    start_xy = np.asarray([world.xy(pos) for pos in origins])
    stop_xy = np.asarray([world.xy(tuple(pos)) for pos in targets])
    length = np.linalg.norm(stop_xy[None] - start_xy[:, None], axis=2)
    direction = (stop_xy[None] - start_xy[:, None])/np.maximum(length[..., None], 1e-9)
    origin_world = np.column_stack((start_xy, np.full(len(start_xy), world.config.laser_height_m)))
    origin_src = origin_world @ world.source_from_world[:3, :3].T + world.source_from_world[:3, 3]
    direction_src = np.dstack((direction, np.zeros(length.shape))) @ world.source_from_world[:3, :3].T
    origins_src = np.broadcast_to(origin_src[:, None, :], direction_src.shape)
    rays = np.concatenate((origins_src, direction_src), axis=2).astype(np.float32)
    hit = world._ray.cast_rays(o3d.core.Tensor(rays), nthreads=1)["t_hit"].numpy()
    in_range = length <= world.config.max_depth_m
    clear = in_range & (hit >= length - .005)
    return in_range.any(axis=0), clear.any(axis=0)


def severance(world, series, edges, far_side, archived_poses, margin):
    prior = None
    connected_far_side = []
    for paid, belief in enumerate(series):
        connected = int((flood(edges, mask_dist(belief != 1) > margin) & far_side).sum())
        connected_far_side.append(connected)
        if connected or prior is not None:
            continue
        assert paid > 0
        new_occupied = [tuple(map(int, cell)) for cell in np.argwhere((belief == 1) & (series[paid-1] != 1))]
        minimal_restoration_sets = []
        if len(new_occupied) <= 12:
            for remove in range(1, len(new_occupied)+1):
                for cells in combinations(new_occupied, remove):
                    ignored = belief != 1
                    for cell in cells:
                        ignored[cell] = True
                    test = flood(edges, mask_dist(ignored) > margin) & far_side
                    if test.any():
                        minimal_restoration_sets.append({"ignored_just_added_cells": cells, "connected_far_side_centers": int(test.sum())})
                if minimal_restoration_sets:
                    break
        scan_hits, _ = replay_laser(world, [archived_poses[paid]])
        frame = world.sense()
        assert hash_values(frame.depth_m) == archived_poses[paid][3]["depth_values_sha256"]
        cloud, _ = frame.points(stride=2)
        body_cloud = cloud[(cloud[:, 2] > .15) & (cloud[:, 2] < 1.2)]
        depth_cells = cell_of(world, body_cloud)
        prior = {"paid": paid, "connected_largest_terminal_unknown_component_one_tick_before": connected_far_side[-2],
                 "verified_archived_scan_ranges_sha256": archived_poses[paid][3]["scan_ranges_sha256"],
                 "verified_archived_depth_m_sha256": archived_poses[paid][3]["depth_values_sha256"],
                 "new_occupied_cells": [{"cell": cell, "verified_native_scan_hits_this_tick": len(scan_hits.get(cell, ())),
                                          "verified_native_rgbd_body_band_hits_this_tick": int(np.all(depth_cells == cell, axis=1).sum())}
                                         for cell in new_occupied], "minimal_offline_ignore_witnesses": minimal_restoration_sets}
    return {"first_closure": prior, "offline_connected_far_side_centers_per_paid": connected_far_side}


def inspect(world, edges, reachable, arm):
    run = ARCHIVE / f"execution/formal/{SHORT}/{arm}/attempt_001/run"
    with np.load(run / "observed_belief_trace.npz", allow_pickle=False) as archive:
        series = archive["belief"]
    last = series[-1]
    path = json.loads((run / "steps.json").read_text())
    archived_poses = [(0, START, 2, None), *[(s["step"], tuple(s["position"]), s["heading"], s) for s in path]]
    centers = sorted({cell for _, cell, _, _ in archived_poses})
    raw_hit, checked = replay_laser(world, archived_poses)
    observed = last == 0
    unknown = reachable & (last == -1)
    d_occ = mask_dist(last != 1)
    margin = world.config.robot_radius_m / world.config.resolution_m + math.sqrt(.5)
    inflated_only_mask = d_occ > margin
    inflated_only_component = flood(edges, inflated_only_mask)
    known_safe = flood(edges, mask_dist(observed) > margin)
    known = flood(edges, observed)
    counterfactual = flood(edges, last != 1)
    assert np.array_equal(known_safe, inflated_only_component)
    components = largest_components(edges, unknown)
    from_early_belief = []
    for early in sorted({0, min(7, len(series)-1), min(37, len(series)-1), *range(48, len(series), 48), len(series)-1}):
        component = flood(edges, mask_dist(series[early] != 1) > margin)
        from_early_belief.append({"paid": early, "reachable_truth_centers_ignoring_unknown": int(component.sum()),
                                 "largest_terminal_unseen_component_centers_reachable": int((component & components[0]).sum())})
    (breaches, pathsteps, unseensteps), route = least_blocked(edges, inflated_only_mask, last, components[0])
    cut = []
    for p in route:
        if inflated_only_mask[p]:
            continue
        nearby = []
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                if (dr or dc) and last[p[0]+dr, p[1]+dc] == 1:
                    target = p[0]+dr, p[1]+dc
                    hit = raw_hit.get(target, [])
                    if hit:
                        nearest = min(hit, key=lambda item: np.linalg.norm(np.asarray(world.xy(p))-item[3][:2]))
                    else:
                        nearest = None
                    nearby.append(dict(cell=target, delta=(dr,dc),
                                       first_marked_occupied_paid=int(np.flatnonzero(series[:, target[0], target[1]] == 1)[0]),
                                       raw_laser_hits=len(hit),
                                       nearest_laser_hit=dict(paid=nearest[0], origin_cell=nearest[1],
                                                              scan_range_m=nearest[2], hit_xyz_m=nearest[3],
                                                              distance_to_route_center_m=float(np.linalg.norm(np.asarray(world.xy(p))-nearest[3][:2]))
                                                              ) if nearest else None))
        cut.append(dict(route_cell=p, belief=int(last[p]), physical_transition_in_repaired_graph=True,
                        observed_occupied_neighbors=nearby))
    raw_laser_cells = np.zeros(last.shape, dtype=bool)
    for cell in raw_hit:
        if all(0 <= i < lim for i, lim in zip(cell, last.shape, strict=True)):
            raw_laser_cells[cell] = True
    lidar_only_component = flood(edges, mask_dist(~raw_laser_cells) > margin)
    unknown_xy = np.argwhere(unknown)
    in_range, clear = line_of_sight(world, centers, unknown_xy)
    safe_unvisited = sorted(set(map(tuple, np.argwhere(known_safe))) - set(centers))
    _, clear_but_unvisited = line_of_sight(world, safe_unvisited, unknown_xy)
    coverage = np.count_nonzero((series == 0) & reachable[None, :, :], axis=(1, 2))
    stats = dict(
        total_motion_reachable_reference_centers=int(reachable.sum()), observed_free=int((reachable & observed).sum()),
        scope="Terminal belief, exact simulated swept graph, and continuous virtual laser rays. No optimization policy invoked.",
        last_change_paid=int(np.flatnonzero(np.diff(coverage))[-1]+1),
        visited_centers=len(centers), replayed_unique_full_pose_scan_hashes_checked=checked,
        final_reachable_centers_unknown=int(unknown.sum()), terminal_occupied_true_centers=int((reachable & (last == 1)).sum()),
        footprint_grid_inflation_threshold_cells=margin,
        reachable_true_centers_if_offline_only_observed_occupied_cell_blocked=int(counterfactual.sum()),
        reachable_true_centers_if_offline_only_observed_occupied_cell_inflated=int(inflated_only_component.sum()),
        reachable_true_centers_if_only_original_laser_hit_cells_inflated=int(lidar_only_component.sum()),
        unknown_true_centers_if_only_original_laser_hit_cells_inflated=int((unknown & lidar_only_component).sum()),
        reachable_true_centers_with_current_known_free_requirement_and_inflation=int(known_safe.sum()),
        reachable_true_centers_if_only_center_known_free=int(known.sum()), unknown_component_sizes=[int(one.sum()) for one in components],
        observed_occupied_inflation_at_earlier_paid_clocks=from_early_belief,
        observed_occupied_inflation_severance=severance(world, series, edges, components[0], archived_poses, margin),
        path_witness_to_largest_unknown_component=dict(physical_cells=route, moves=pathsteps,
            blocked_by_observed_dilation_minimal_cells=breaches, unknown_centers_on_path=unseensteps,
            blocked_route_cells=cut),
        raw_laser_unknown_in_range_any_actual_visited_pose=int(in_range.sum()),
        raw_laser_unknown_clear_continuous_line_any_actual_visited_pose=int(clear.sum()),
        raw_laser_unknown_blocked_continuous_line_from_every_in_range_actual_visited_pose=int((in_range & ~clear).sum()),
        raw_laser_unknown_clear_from_any_unvisited_but_fully_known_safe_true_center=int(clear_but_unvisited.sum()),
        coverage_per_paid=coverage.tolist(),
    )
    return stats, (last, known_safe, centers, route, cut)


def plot(reachable, data):
    fig, axes = plt.subplots(1, 2, figsize=(12, 5.6), layout="constrained")
    colors = ['#f5f2eb', '#CDD8E7', '#efce96', '#5285A7', '#263A4C']
    cmap = ListedColormap(colors)
    norm = BoundaryNorm(np.arange(-.5, 5, 1), cmap.N)
    min_r, min_c = np.argwhere(reachable).min(0)-2
    max_r, max_c = np.argwhere(reachable).max(0)+2
    for axis, arm in zip(axes, ARMS, strict=True):
        belief, safe, visits, route, cuts = data[arm]
        image = np.zeros(reachable.shape, dtype=int)
        image[reachable & (belief == -1)] = 1
        image[reachable & (belief == 0)] = 2
        image[safe] = 3
        image[(belief == 1) & (np.indices(reachable.shape)[0] >= min_r)] = 4
        axis.imshow(image, cmap=cmap, norm=norm, interpolation="nearest")
        rr, cc = zip(*route, strict=True)
        axis.plot(cc, rr, color="#859b26", lw=1.7)
        rr, cc = zip(*visits, strict=True)
        axis.scatter(cc, rr, c="#134b31", s=9, alpha=.85)
        rr, cc = zip(*(cut["route_cell"] for cut in cuts), strict=True)
        axis.scatter(cc, rr, c="#C34244", s=50, ec="white", lw=.6, zorder=3)
        axis.scatter([START[1]], [START[0]], c="#f4d12b", marker="*", edgecolor="#263A4C", s=100, zorder=4)
        axis.set(xlim=(min_c, max_c), ylim=(max_r, min_r), xlabel="grid column", ylabel="grid row", title=arm)
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch
    fig.legend(handles=[Patch(color=colors[1], label="Unknown true reachable"), Patch(color=colors[2], label="Observed free outside online-safe component"),
                        Patch(color=colors[3], label="Online-safe connected"), Patch(color=colors[4], label="Observed occupied"),
                        Line2D([], [], color="#859b26", label="Offline passable path"), Line2D([], [], ls="", marker="o", c="#C34244", label="Path cells rejected online"),
                        Line2D([], [], ls="", marker="o", c="#134b31", label="Visited centers")],
               loc="outside lower center", ncol=3, fontsize=8)
    fig.savefig(OUT / "hm3d_start1_plateau.png", dpi=175)
    plt.close(fig)


def test_corridor_geometry(world, stats):
    official, metadata = read_detour(NAV_FILE, world.world_from_source)
    legacy = ARCHIVE.parent / "ras_20261126/heterogeneous/scenes/precomputed_assets/hm3d_00770.npz"
    with np.load(legacy, allow_pickle=False) as cache:
        original_edges = cache["edge_valid"]
    routes = {arm: stats[arm]["path_witness_to_largest_unknown_component"] for arm in ARMS}
    edges = set()
    for route in routes.values():
        blocked = {item["route_cell"] for item in route["blocked_route_cells"]}
        edges.update((p, q) for p, q in zip(route["physical_cells"], route["physical_cells"][1:])
                     if p in blocked or q in blocked)
    result = []
    for p, q in sorted(edges):
        start, end = np.asarray(world.xy(p)), np.asarray(world.xy(q))
        line = LineString((start, end))
        heading = DIRECTIONS.index((q[0] - p[0], q[1] - p[1]))
        _, _, nearest = local_original_triangles(world._source_scene, world.world_from_source, start, end)
        result.append({"start": p, "end": q, "original_body_clearance_m": nearest["distance_to_centerline_m"],
                       "allowed_in_pre_repair_sweep_graph": bool(original_edges[p + (heading,)]),
                       "official_habitat_navmesh_missing_centerline_m": float(line.difference(official).length)})
    return {"secondary_official_navmesh": metadata, "edges_touching_a_dilated_route_cell": result,
            "meaning": "The study tests original raw body triangles; the Habitat mesh is a separately processed navigation reference from the same scan."}


def main():
    path = ARCHIVE / f"scenes/evaluator_references/{SHORT}/evaluator_truth.npz"
    with np.load(path, allow_pickle=False) as archive:
        reachable, edges = archive["reachable"], archive["edge_valid"]
    world = make_world(SHORT)
    assert np.array_equal(world._edge_valid, edges)
    stats, masks = {}, {}
    for arm in ARMS:
        stats[arm], masks[arm] = inspect(world, edges, reachable, arm)
    regional_dir = ARCHIVE / f"execution/formal/{SHORT}"
    paths = [regional_dir / f"{arm}/attempt_001/run" for arm in ("combined_S", "region_S")]
    with np.load(paths[0] / "observed_belief_trace.npz", allow_pickle=False) as left, np.load(
        paths[1] / "observed_belief_trace.npz", allow_pickle=False,
    ) as right:
        same_belief = np.array_equal(left["belief"], right["belief"]) and np.array_equal(left["paid"], right["paid"])
    trajectories = [[(item["action"], item["position"], item["heading"]) for item in json.loads((d / "steps.json").read_text())]
                    for d in paths]
    assert same_belief and trajectories[0] == trajectories[1]
    plot(reachable, masks)
    result = {"case": SHORT, "source": str(ARCHIVE), "raw_laser_height_m": world.config.laser_height_m,
              "raw_laser_range_m": world.config.max_depth_m, "evidence": stats,
              "region_S_has_identical_archived_stepwise_actions_poses_and_belief_as_combined_S": True,
              "independent_geometry_checks": test_corridor_geometry(world, stats)}
    (OUT / "hm3d_start1_plateau.json").write_text(json.dumps(result, indent=2, allow_nan=False)+"\n")


if __name__ == "__main__":
    main()
