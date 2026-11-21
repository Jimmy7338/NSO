"""Freeze source geometry, zero-action sensor input and independent surface references."""

import argparse
import dataclasses
import hashlib
import json
import os
from pathlib import Path
import sys

for variable in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[variable] = "1"

import numpy as np
import open3d as o3d

from scene import ANALYSES, BUDGET, HERE, LAYOUTS, SEED, SPECS, SOURCE, build_scene, make_world


NATIVE_PREPARE = ANALYSES / "semantic_large_scene_20261121/mechanism/scenes"
RUNTIME = ANALYSES / "paired_unseen_gain_20261120/execution"
METRIC = ANALYSES / "dual_template_exploration_20261119/audit/surface_metrics.py"
sys.path[:0] = [str(NATIVE_PREPARE), str(RUNTIME), str(METRIC.parent)]
import prepare_reference as native_reference
import runtime_binding as binding
from surface_metrics import fixed_camera_poses, ray_scene, subset_mesh, surface_samples, unique_sampling_faces, visible_reference


def read(path):
    return json.loads(path.read_text())


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write(path, value):
    with path.open("x") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write("\n")


def initial(layout):
    old = binding.install()
    from nso.cpu_sensor_contract_v10 import GridTransform

    cfg = read(RUNTIME / "protocol.json")
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
    unknown_reachable = int(np.count_nonzero(truth["reachable"] & (mapper.belief == -1)))
    if world.step_count != 0 or mapper.current_footprint_conflict or not unknown_reachable:
        raise ValueError(f"Invalid unobserved cold-start geometry for {layout}")
    out = HERE / "prepared" / layout
    out.mkdir(parents=True, exist_ok=False)
    packet.frame.save(out / "initial_frame.npz")
    packet.scan.save(out / "initial_scan.npz")
    evaluation = HERE / "evaluator_references" / layout
    evaluation.mkdir(parents=True, exist_ok=False)
    np.savez_compressed(evaluation / "evaluator_truth.npz", **truth)
    metadata = {
        "layout": layout, "scene_id": packet.scene_id, "anchor": [*packet.position, packet.heading],
        "shape": list(world.shape), "stop_budget": BUDGET, "paid_actions": 0, "seed": SEED,
        "grid_resolution_m": world.config.resolution_m, "actual_config": dataclasses.asdict(world.config),
        "initial_packet_sha256": packet.sha256(), "initial_owner_sha256": owner["owner_sha256"],
        "owner_counts": owner["owner_counts"],
        "scene_role": "ordinary_heldout_parent",
        "policy_select_calls": 0,
    }
    write(out / "initial_state.json", metadata)
    write(evaluation / "cold_start_preflight.json", {"initial_unknown_reachable_cells": unknown_reachable,
          "actor_select_calls": 0, "geometry": world.geometry_receipt})
    return out, truth["reachable"], world.config, metadata, unknown_reachable


def reference(layout, initial_dir, reachable, config, initial_state):
    protocol = HERE / "reference_protocol.json"
    rules = read(protocol)
    native_reference.build_scene = build_scene
    mesh, labels, records = native_reference.source_mesh(layout)
    source_mesh, source_labels, duplicates = unique_sampling_faces(mesh, labels)
    ray = ray_scene(mesh)
    poses, cells = fixed_camera_poses(reachable, config, initial_state["anchor"][:2])
    values, metadata = {"camera_poses": poses}, {}
    for name, label in (("all", None), ("open31", 31), ("closed118", 118)):
        submesh = source_mesh if label is None else subset_mesh(source_mesh, source_labels == label)
        count = rules["global_area_samples" if label is None else "per_target_area_samples"]
        seed = rules["base_seed"] + 1000 * LAYOUTS.index(layout) + (label or 0)
        points, _ = surface_samples(submesh, count, seed)
        visible, first, rays = visible_reference(points, ray, poses, config, tolerance_m=rules["first_hit_tolerance_m"])
        if not visible.any():
            raise ValueError(f"No reference surface for {layout}/{name}")
        values[name] = points[visible].astype(np.float32)
        values[name + "_first_pose"] = first[visible]
        area = submesh.get_surface_area()
        metadata[name] = {"sampled_source_area_m2": area, "drawn_samples": count, "visible_samples": int(visible.sum()),
            "estimated_visible_area_m2": float(area * visible.mean()), "area_per_sample_m2": float(area / count),
            "seed": seed, "first_hit_rays": rays}
    out = HERE / "evaluator_references" / layout
    np.savez_compressed(out / "fixed_reference.npz", **values)
    if not o3d.io.write_triangle_mesh(str(out / "source_mesh.ply"), mesh):
        raise OSError(f"Failed saving reference mesh at {out}")
    deps = (Path(__file__), HERE / "scene.py", SOURCE, protocol, initial_dir / "initial_state.json",
            out / "evaluator_truth.npz", Path(native_reference.__file__), METRIC)
    write(out / "RECEIPT.json", {"status": "fixed_source_only_surface_reference_prepared", "layout": layout,
        "reference_npz_sha256": sha(out / "fixed_reference.npz"), "source_mesh_sha256": sha(out / "source_mesh.ply"),
        "source_inputs_sha256": {str(path): sha(path) for path in deps}, "metadata": metadata,
        "reachable_cells": int(reachable.sum()), "camera_lattice_cells": cells, "camera_pose_count": len(poses),
        "duplicate_sampling_handling": duplicates, "source_object_records": records, "actor_executed": False,
        "actor_trajectory_read": False, "reference_source_is_native_geometry_not_template": True})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("layout", choices=LAYOUTS)
    args = parser.parse_args()
    directory, reachable, config, state, unknown_reachable = initial(args.layout)
    reference(args.layout, directory, reachable, config, state)
    print(json.dumps({"layout": args.layout, "reachable_cells": int(reachable.sum()),
                      "cold_start_unknown_reachable_cells": unknown_reachable,
                      "shape": SPECS[args.layout]["shape"], "actor_select_calls": 0}), flush=True)


if __name__ == "__main__":
    main()
