"""Prepare the acquired cold packet and common fixed floor references before planning."""

import argparse
import dataclasses
import json
from pathlib import Path
import sys
import time

import numpy as np
import open3d as o3d
import trimesh

from scene import ANALYSES, BUDGET, HERE, LAYOUTS, SPECS, build_scene, make_world, sha
from sources import SOURCES


B11 = ANALYSES / "paired_unseen_gain_20261120/execution"
V3_METRICS = ANALYSES / "dual_template_exploration_20261119/audit/surface_metrics.py"
sys.path[0:0] = [str(B11), str(V3_METRICS.parent)]
import runtime_binding as binding
from surface_metrics import ray_scene, surface_samples, unique_sampling_faces, visible_reference
from env.virtual3d import camera_pose


def read(path):
    return json.loads(path.read_text())


def write(path, value):
    with path.open("x") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write("\n")


def initial(layout):
    old = binding.install()
    from nso.cpu_sensor_contract_v10 import GridTransform

    cfg = read(B11 / "protocol.json")
    cfg["policy"]["paid_action_budget"] = BUDGET
    cfg["layouts"] = {name: {"scene_id": f"same-goal-holdout-{name}"} for name in LAYOUTS}
    cfg["arms"] = {mode: {"score_mode": mode, "predictor": "visibility"} for mode in ("G", "S")}
    binding.ARMS = cfg["arms"]
    old.method.ARMS = tuple(cfg["arms"])
    old.method.MODE = {mode: mode for mode in cfg["arms"]}
    old.protocol = lambda: cfg

    def world_packet(name):
        world = make_world(name)
        grid = GridTransform(tuple(world.shape), world.config.resolution_m)
        return world, grid, old.packet(world, grid, world.sense(), name)

    old.world_and_packet = world_packet
    world, _, packet, _, runtime, _, owner = old.bootstrap(layout, "G")
    mapper = runtime.states[0]["mapper"]
    truth = world.evaluator_truth()
    unknown = int(np.count_nonzero(truth["reachable"] & (mapper.belief == -1)))
    if world.step_count != 0 or mapper.current_footprint_conflict or not unknown:
        raise ValueError(f"Cold acquired floor state invalid for {layout} (conflict={mapper.current_footprint_conflict}, unknown={unknown})")
    if packet.frame.semantic.any() or world.sense_visible_instances().any():
        raise ValueError("External scan enriched with unsupported semantic category")
    out = HERE / "prepared" / layout
    out.mkdir(parents=True, exist_ok=False)
    packet.frame.save(out / "initial_frame.npz")
    packet.scan.save(out / "initial_scan.npz")
    ref = HERE / "evaluator_references" / layout
    ref.mkdir(parents=True, exist_ok=False)
    np.savez_compressed(ref / "evaluator_truth.npz", **truth)
    spec = SPECS[layout]
    initial_state = {
        "layout": layout, "scene_id": packet.scene_id, "anchor": [*packet.position, packet.heading],
        "shape": list(world.shape), "stop_budget": BUDGET, "paid_actions": 0, "seed": world.seed,
        "grid_resolution_m": world.config.resolution_m, "actual_config": dataclasses.asdict(world.config),
        "initial_packet_sha256": packet.sha256(), "initial_owner_sha256": owner["owner_sha256"],
        "owner_counts": owner["owner_counts"], "scene_role": spec["scene_role"],
        "source_floor": spec["source_floor"], "start_index": spec["start_index"], "policy_select_calls": 0,
    }
    write(out / "initial_state.json", initial_state)
    write(ref / "cold_start_preflight.json", {
        "initial_unknown_reachable_cells": unknown, "initial_scan_hit_count": int(
            np.count_nonzero(packet.scan.ranges_m < world.config.max_depth_m)),
        "frame_depth_hit_fraction": float(np.count_nonzero(packet.frame.depth_m) / packet.frame.depth_m.size),
        "actor_select_calls": 0, "geometry": world.geometry_receipt,
    })
    return world.config, truth["reachable"]


def mesh_from_source(layout):
    source, _, _ = build_scene(layout)
    points, triangles, offset = [], [], 0
    for node in source.graph.nodes_geometry:
        transform, key = source.graph[node]
        part = source.geometry[key]
        xyz = trimesh.transform_points(part.vertices, transform)
        points.append(xyz)
        triangles.append(np.asarray(part.faces, np.int32) + offset)
        offset += len(xyz)
    return o3d.geometry.TriangleMesh(o3d.utility.Vector3dVector(np.concatenate(points)),
                                     o3d.utility.Vector3iVector(np.concatenate(triangles)))


def reference(floor_layouts, config, reachable):
    began = time.monotonic()
    names = [layout for layout, _ in floor_layouts]
    if not all(np.array_equal(mask, reachable) for _, mask in floor_layouts):
        raise ValueError("Start pair does not share the same swept-navigable floor component")
    floor = SPECS[names[0]]["source_floor"]
    mesh = mesh_from_source(names[0])
    sampling, _, duplicates = unique_sampling_faces(mesh, np.zeros(len(mesh.triangles), dtype=np.int16))
    protocol = HERE / "reference_protocol.json"
    rules = read(protocol)
    seed = rules["base_seed"] + 1000 * list(SOURCES).index(floor)
    lattice = {tuple(map(int, p)) for p in np.argwhere(reachable) if p[0] % 4 == 0 and p[1] % 4 == 0}
    lattice.update(tuple(SPECS[name]["start"]) for name in names)
    poses = np.asarray([camera_pose(cell, heading, config, reachable.shape[0])
                        for cell in sorted(lattice) for heading in range(4)])
    points, _ = surface_samples(sampling, rules["global_area_samples"], seed)
    visible, first, rays = visible_reference(points, ray_scene(mesh), poses, config,
                                             tolerance_m=rules["first_hit_tolerance_m"])
    if not visible.any():
        raise ValueError("No reference surface visible from frozen navigable floor")
    area = sampling.get_surface_area()
    meta = {"all": {"sampled_source_area_m2": area, "drawn_samples": len(points),
                    "visible_samples": int(visible.sum()), "estimated_visible_area_m2": float(area*visible.mean()),
                    "area_per_sample_m2": float(area/len(points)), "seed": seed, "first_hit_rays": rays}}
    first_directory = None
    for layout in names:
        out = HERE / "evaluator_references" / layout
        if first_directory is None:
            np.savez_compressed(out / "fixed_reference.npz", camera_poses=poses,
                                all=points[visible].astype(np.float32), all_first_pose=first[visible])
            if not o3d.io.write_triangle_mesh(str(out / "source_mesh.ply"), mesh):
                raise OSError("Surface reference PLY export failed")
            first_directory = out
        else:
            # Hardlink read-only shared reference, so both starts use exactly the same evaluation points.
            for suffix in ("fixed_reference.npz", "source_mesh.ply"):
                (out / suffix).hardlink_to(first_directory / suffix)
        deps = (Path(__file__), HERE / "scene.py", protocol, HERE / "prepared" / layout / "initial_state.json",
                out / "evaluator_truth.npz", V3_METRICS)
        write(out / "RECEIPT.json", {"status": "fixed_source_only_surface_reference_prepared",
            "layout": layout, "source_floor": floor, "source_license": SOURCES[floor]["license"],
            "reference_npz_sha256": sha(out / "fixed_reference.npz"),
            "source_mesh_sha256": sha(out / "source_mesh.ply"),
            "source_inputs_sha256": {str(path): sha(path) for path in deps}, "metadata": meta,
            "reachable_cells": int(reachable.sum()), "camera_lattice_cells": len(lattice),
            "camera_pose_count": len(poses), "source_triangles": len(mesh.triangles),
            "duplicate_sampling_handling": duplicates, "source_object_records": [],
            "actor_executed": False, "actor_trajectory_read": False,
            "reference_source_is_native_geometry_not_template": True, "elapsed_s": time.monotonic()-began})


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("floor", choices=sorted({spec["source_floor"] for spec in SPECS.values()}))
    args = parser.parse_args()
    floor_layouts, floor_config = [], None
    for layout in LAYOUTS:
        if SPECS[layout]["source_floor"] == args.floor:
            floor_config, reachable_mask = initial(layout)
            floor_layouts.append((layout, reachable_mask))
            print(f"Verified initial packet: {layout}", flush=True)
    reference(floor_layouts, floor_config, floor_layouts[0][1])
    print(f"Verified independent reference: {args.floor}", flush=True)
