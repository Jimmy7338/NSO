"""Select source floors/start poses from source geometry without executing a policy."""

import argparse
from collections import Counter, deque
import json
from pathlib import Path
import time

import numpy as np
import open3d as o3d
from scipy.ndimage import distance_transform_edt
import trimesh

from scene import HERE, SPECS, TexturedRayRenderer, source_scene, sha
import replicacad_world as geometry
from sources import ADMISSION, SOURCES


RESOLUTION = .20
RADIUS = .20
NEIGHBORS = ((-1, 0), (0, 1), (1, 0), (0, -1))


def room_cores(valid, edges, clearance):
    cores = []
    unvisited = valid & (clearance >= .48)
    for row, col in np.argwhere(unvisited):
        start = int(row), int(col)
        if not unvisited[start]:
            continue
        unvisited[start] = False
        group, queue = [start], deque([start])
        while queue:
            r, c = queue.popleft()
            for i, (dr, dc) in enumerate(NEIGHBORS):
                dst = r + dr, c + dc
                if edges[r, c, i] and unvisited[dst]:
                    unvisited[dst] = False
                    group.append(dst)
                    queue.append(dst)
        if len(group) >= 50:
            cores.append(sorted(group))
    return sorted(cores, key=lambda cells: (-len(cells), cells[0]))


def grow_regions(valid, edges, groups):
    labels = np.zeros(valid.shape, dtype=np.int16)
    queue = deque()
    for index, cells in enumerate(groups, 1):
        for cell in cells:
            labels[cell] = index
            queue.append(cell)
    openings = Counter()
    while queue:
        r, c = queue.popleft()
        for i, (dr, dc) in enumerate(NEIGHBORS):
            dst = r + dr, c + dc
            if not edges[r, c, i] or not valid[dst]:
                continue
            if not labels[dst]:
                labels[dst] = labels[r, c]
                queue.append(dst)
            elif labels[dst] != labels[r, c] and i in (1, 2):
                openings[tuple(sorted((int(labels[dst]), int(labels[r, c]))))] += 1
    # Count the complete interfaces after labeling, once per unoriented edge.
    openings.clear()
    for r, c in np.argwhere(valid):
        for i in (1, 2):
            dr, dc = NEIGHBORS[i]
            dst = r + dr, c + dc
            if edges[r, c, i] and valid[dst] and labels[dst] != labels[r, c]:
                openings[tuple(sorted((int(labels[dst]), int(labels[r, c]))))] += 1
    return labels, openings


def flood(edges, seed):
    distances = np.full(edges.shape[:2], -1, dtype=np.int32)
    distances[seed] = 0
    queue = deque([seed])
    while queue:
        r, c = queue.popleft()
        for i, (dr, dc) in enumerate(NEIGHBORS):
            dst = r + dr, c + dc
            if edges[r, c, i] and distances[dst] < 0:
                distances[dst] = distances[r, c] + 1
                queue.append(dst)
    return distances


def largest_component(valid, edges):
    remaining = valid.copy()
    winner = np.zeros(valid.shape, dtype=bool)
    components = []
    for r, c in np.argwhere(valid):
        if remaining[r, c]:
            component = flood(edges, (r, c)) >= 0
            remaining &= ~component
            size = int(component.sum())
            components.append(size)
            if size > winner.sum():
                winner = component
    return winner, sorted(components, reverse=True)


def floor_height(source):
    heights, area = [], []
    for name in source.graph.nodes_geometry:
        tf, mesh_name = source.graph[name]
        mesh = source.geometry[mesh_name]
        faces = trimesh.transform_points(mesh.vertices, tf)[mesh.faces]
        normal = np.cross(faces[:, 1] - faces[:, 0], faces[:, 2] - faces[:, 0])
        length = np.linalg.norm(normal, axis=1)
        keep = (length > 1e-10) & (normal[:, 2] >= .96 * length)
        heights.extend(faces[keep, :, 2].mean(axis=1))
        area.extend(length[keep] / 2)
    heights, area = np.asarray(heights), np.asarray(area)
    bins = np.floor(heights / .05).astype(int)
    ranking = sorted(((int(b), float(area[bins == b].sum())) for b in np.unique(bins)),
                     key=lambda entry: (-entry[1], entry[0]))
    picked = bins == ranking[0][0]
    return float(np.average(heights[picked], weights=area[picked])), [[.05 * b, a] for b, a in ranking[:8]]


def grid_coords(shape, cell):
    r, c = cell
    return (c + .5) * RESOLUTION, (shape[0] - r - .5) * RESOLUTION


def grid_from_mesh(scene, transform, shape):
    geometry.FLOOR_BAND_M = tuple(ADMISSION["scan_floor_support_band_m"])
    geometry.FLOOR_NORMAL_MIN = ADMISSION["scan_floor_normal_abs_cosine_minimum"]
    checker = geometry.MeshMotionGeometry(scene, transform)
    valid, occupancy = np.zeros(shape, bool), np.ones(shape, bool)
    edges = np.zeros((*shape, 4), bool)
    for cell in np.ndindex(shape):
        point = grid_coords(shape, cell)
        occupancy[cell] = not checker.valid(point, radius=0)
        valid[cell] = checker.valid(point, radius=RADIUS)
    for r, c in np.argwhere(valid):
        for i in (1, 2):
            dr, dc = NEIGHBORS[i]
            dst = (r + dr, c + dc)
            if dst[0] < shape[0] and dst[1] < shape[1] and valid[dst]:
                if checker.valid(grid_coords(shape, (r, c)), grid_coords(shape, dst), RADIUS):
                    edges[r, c, i] = edges[dst + ((i + 2) % 4,)] = True
    return occupancy, valid, edges, checker.stats


def start_poses(source, transform, shape, cores, clearance, edges):
    first = np.asarray(cores[0])
    def middle(cells):
        location = np.mean(cells, axis=0)
        return min(map(tuple, cells), key=lambda p: (float(np.linalg.norm(p-location)), -clearance[p], p))
    seed = middle(first)
    distances = flood(edges, seed)
    remote = sorted((middle(np.asarray(group)) for group in cores[1:]),
                    key=lambda p: (-distances[p], p))
    if not remote or distances[remote[0]] * RESOLUTION < 3:
        raise ValueError("Separate room starts fail 3 m swept-graph separation")
    renderer = TexturedRayRenderer(source)
    inverse = np.linalg.inv(transform)
    config = geometry.ReplicaCADConfig(max_steps=192)
    starts = []
    for cell in (seed, remote[0]):
        optical = np.array([*grid_coords(shape, cell), config.camera_height_m, 1.])
        origin = (inverse @ optical)[:3]
        directions = np.asarray([[0, 1, 0], [1, 0, 0], [0, -1, 0], [-1, 0, 0]], dtype=np.float32)
        rays = np.concatenate([np.tile(origin, (4, 1)), directions], axis=1).astype(np.float32)
        hit = renderer.scene.cast_rays(o3d.core.Tensor(rays), nthreads=1)["t_hit"].numpy()
        heading = int(np.argmax(np.minimum(hit, config.max_depth_m)))
        pose = inverse @ geometry.camera_pose(cell, heading, config, shape[0])
        f = config.width_px / (2*np.tan(np.deg2rad(config.fov_deg/2)))
        k = np.array([[f, 0, (config.width_px-1)/2], [0, f, (config.height_px-1)/2], [0, 0, 1.]])
        frame = renderer.render(k, pose, config.width_px, config.height_px, max_depth_m=config.max_depth_m)
        fraction = float((frame["depth_m"] > 0).mean())
        if fraction < ADMISSION["minimum_fraction_depth_hits"]:
            raise ValueError(f"Insufficient initial RGB-D depth at frozen geometry pose {cell}: {fraction}")
        starts.append(dict(start=list(map(int, cell)), heading=heading, xy_world_m=list(grid_coords(shape, cell)),
                           xyz_source_m=[float(x) for x in origin], depth_hit_fraction=fraction,
                           initial_cardinal_range_m=[float(x) if np.isfinite(x) else None for x in hit]))
    return starts, float(distances[remote[0]] * RESOLUTION)


def inspect(name):
    began = time.monotonic()
    source = source_scene(name)
    record = SOURCES[name]
    bounds = np.asarray(source.bounds)
    low = np.floor(bounds[0, :2]/RESOLUTION)*RESOLUTION-1
    high = np.ceil(bounds[1, :2]/RESOLUTION)*RESOLUTION+1
    shape = tuple(np.rint((high - low)[::-1] / RESOLUTION).astype(int).tolist())
    z, bins = floor_height(source)
    transform = np.eye(4)
    transform[:3, 3] = (-low[0], -low[1], -z)
    occupancy, valid, edges, stats = grid_from_mesh(source, transform, shape)
    main, sizes = largest_component(valid, edges)
    area = int(main.sum()) * RESOLUTION**2
    clearance = distance_transform_edt(main) * RESOLUTION
    cores = room_cores(main, edges, clearance)
    receipt = {"source_floor": name, **{k: v for k, v in record.items() if k != "path"},
               "source_file": str(record["path"]), "source_file_sha256": sha(record["path"]),
               "meters_per_source_unit": 1.0, "up_axis": "Z", "source_bounds_m": bounds.tolist(),
               "dominant_source_floor_z_m": z, "top_upward_area_bins_m2": bins,
               "world_from_source": transform.tolist(), "shape": list(shape), "capsule_geometry": stats,
               "max_component_cells": int(main.sum()), "component_sizes_cells": sizes,
               "navigable_area_m2": area, "room_core_sizes": [len(g) for g in cores],
               "actor_select_calls": 0, "elapsed_s": time.monotonic() - began}
    if area < 20 or len(cores) < 2:
        return receipt | {"eligible": False, "reason": "main_component_or_multiple_room_requirement_failed"}
    labels, openings = grow_regions(main, edges, cores)
    points = [(a, b, n) for (a, b), n in sorted(openings.items())]
    try:
        proposed, separation = start_poses(source, transform, shape, cores, clearance, edges)
    except ValueError as error:
        return receipt | {"eligible": False, "reason": str(error)}
    cache = HERE / "precomputed_assets"
    cache.mkdir(exist_ok=True)
    artifact = cache / f"{name}.npz"
    if artifact.exists():
        raise FileExistsError(artifact)
    np.savez_compressed(artifact, occupancy=occupancy, robot_valid=valid, edge_valid=edges, room_labels=labels)
    return receipt | dict(eligible=True, starts=proposed, graph_start_separation_m=separation,
                          room_count=len(cores), room_opening_edges=points,
                          room_core_graph_cycle_rank=len(openings)-len(cores)+1,
                          capsule_cache_sha256=sha(artifact))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("floor", choices=list(SOURCES))
    args = parser.parse_args()
    target = HERE / "source_preflights"
    target.mkdir(exist_ok=True)
    outfile = target / f"{args.floor}.json"
    if outfile.exists():
        raise FileExistsError(outfile)
    report = inspect(args.floor)
    with outfile.open("x") as stream:
        json.dump(report, stream, indent=2, allow_nan=False)
        stream.write("\n")
    print(json.dumps({key: report.get(key) for key in ("source_floor", "eligible", "reason", "room_count",
                                                       "navigable_area_m2", "starts")}))
