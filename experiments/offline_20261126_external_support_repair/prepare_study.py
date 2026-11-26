"""Produce corrected motion caches and references, with no policy invocations."""

import argparse
from collections import deque
import json
from pathlib import Path
import shutil
import sys

import numpy as np

from repair_geometry import RepairedMotionGeometry, native as motion_geometry
from study import OFFLINE, PREVIOUS, ROOT, SCENES, supervisor


sys.path.insert(0, str(SCENES))
from scene import SOURCES, SPECS, make_world, native, source_scene


NATIVE_AUDIT = PREVIOUS / "FORMAL_COLLISION_AUDIT.json"
DIRECTIONS = ((-1, 0), (0, 1), (1, 0), (0, -1))


def prepare_cache(floor, log):
    io = supervisor()
    starts = [r for r in SPECS.values() if r["source_floor"] == floor]
    spec = starts[0]
    shape = tuple(spec["shape"])
    config = native.ReplicaCADConfig(max_steps=192)
    radius, resolution = config.robot_radius_m, config.resolution_m
    geom = RepairedMotionGeometry(source_scene(floor), np.asarray(spec["world_from_source"]))

    def xy(cell):
        row, col = cell
        return ((col + .5) * resolution, (shape[0] - row - .5) * resolution)

    def inside(cell):
        return all(0 <= value < limit for value, limit in zip(cell, shape, strict=True))

    occupancy = np.ones(shape, dtype=bool)
    valid = np.zeros(shape, dtype=bool)
    edges = np.zeros((*shape, 4), dtype=bool)
    for cell in np.ndindex(shape):
        center = xy(cell)
        occupancy[cell] = not geom.valid(center, radius=0)
        if not occupancy[cell]:
            valid[cell] = geom.valid(center, radius=radius)
    for r, c in np.argwhere(valid):
        for direction in (1, 2):
            dr, dc = DIRECTIONS[direction]
            target = (r+dr, c+dc)
            if inside(target) and valid[target] and geom.valid(xy((r, c)), xy(target), radius):
                edges[r, c, direction] = edges[*target, (direction+2) % 4] = True
    old_name = OFFLINE / "precomputed_assets" / f"{floor}.npz"
    with np.load(old_name, allow_pickle=False) as old:
        old_valid = old["robot_valid"]
        old_edges = old["edge_valid"]
        if np.any(old_valid & ~valid) or np.any(old_edges & ~edges):
            raise ValueError(f"Support repair removed old allowed geometry: {floor}")
        rooms = old["room_labels"]
    name = SCENES / "precomputed_assets" / f"{floor}.npz"
    name.parent.mkdir(parents=True, exist_ok=True)
    with name.open("xb") as stream:
        np.savez_compressed(stream, occupancy=occupancy, robot_valid=valid, edge_valid=edges, room_labels=rooms)
    cases = []
    for before in io.read(NATIVE_AUDIT)["failures"]:
        if before["source_floor"] != floor:
            continue
        cell, target, heading = tuple(before["cell"]), tuple(before["target"]), before["heading"]
        old_ok, new_ok = bool(old_edges[*cell, heading]), bool(edges[*cell, heading])
        opposite = bool(edges[*target, (heading+2) % 4])
        verified = (not old_ok and new_ok and opposite and not geom.original.valid(xy(cell), xy(target), radius)
                    and geom.valid(xy(cell), xy(target), radius))
        cases.append({"cell": cell, "target": target, "heading": heading, "before_allowed": old_ok,
                      "after_allowed": new_ok, "reverse_allowed": opposite, "verified": verified})
    cache_summary = {"floor": floor, "old_sha": io.sha(old_name), "repaired_sha": io.sha(name),
                     "old_pose_cells": int(old_valid.sum()), "repaired_pose_cells": int(valid.sum()),
                     "old_directed_edges": int(old_edges.sum()), "repaired_directed_edges": int(edges.sum()),
                     "case_checks": cases, "body_height_band_m": list(motion_geometry.COLLISION_BAND_M),
                     "native_support_stats": geom.stats}
    io.write(log / f"{floor}_cache.json", cache_summary)
    for key, start in SPECS.items():
        if start["source_floor"] == floor:
            prepare_start(key, old_name, name, log)


def prepare_start(layout, old_cache, repaired_cache, log):
    io = supervisor()
    incoming = OFFLINE / "prepared" / layout
    former = OFFLINE / "evaluator_references" / layout
    output = SCENES / "prepared" / layout
    refs = SCENES / "evaluator_references" / layout
    output.mkdir(parents=True, exist_ok=False)
    refs.mkdir(parents=True, exist_ok=False)
    for name in ("initial_state.json", "initial_frame.npz", "initial_scan.npz"):
        (output / name).hardlink_to(incoming / name)
    for name in ("fixed_reference.npz", "source_mesh.ply"):
        (refs / name).hardlink_to(former / name)
    world = make_world(layout)
    cold = io.read(output / "initial_state.json")
    from utils.rgbd_contract import RGBDFrame, PlanarScan
    acquired = world.sense()
    old_frame = RGBDFrame.load(output / "initial_frame.npz")
    old_scan = PlanarScan.load(output / "initial_scan.npz")
    frame_ok = all(np.array_equal(getattr(acquired, name), getattr(old_frame, name))
                   for name in ("depth_m", "color_rgb", "intrinsic", "world_from_camera", "semantic"))
    scan = world.scan()
    scan_ok = all(np.array_equal(getattr(scan, name), getattr(old_scan, name))
                  for name in ("ranges_m", "world_from_laser", "angle_min_rad", "angle_increment_rad", "range_max_m"))
    if not frame_ok or not scan_ok or [*world.start, world.heading] != cold["anchor"]:
        raise ValueError(f"Shared start, calibration or raw observation changed: {layout}")
    truth = world.evaluator_truth()
    with np.load(former / "evaluator_truth.npz", allow_pickle=False) as previous:
        old_mask = previous["reachable"]
    if np.any(old_mask & ~truth["reachable"]):
        raise ValueError(f"Corrected ground support lost old reachable cells: {layout}")
    with (refs / "evaluator_truth.npz").open("xb") as stream:
        np.savez_compressed(stream, **truth)
    receipt = io.read(former / "RECEIPT.json")
    receipt |= {"reachable_cells": int(truth["reachable"].sum()), "original_reachable_cells": int(old_mask.sum()),
                "new_reachable_ledger_sha256": io.sha(refs / "evaluator_truth.npz"),
                "original_reference_receipt_sha256": io.sha(former / "RECEIPT.json"),
                "repaired_capsule_cache_sha256": io.sha(repaired_cache),
                "original_capsule_cache_sha256": io.sha(old_cache),
                "surface_reference_unchanged": True, "renderer_inputs_unchanged": True}
    io.write(refs / "RECEIPT.json", receipt)
    io.write(log / f"{layout}_start.json",
             {"layout": layout, "same_depth_rgb_semantic_camera": frame_ok, "same_planar_scan": scan_ok,
              "anchor": cold["anchor"], "raw_packet_sha256": cold["initial_packet_sha256"],
              "old_reachable": int(old_mask.sum()), "repaired_reachable": int(truth["reachable"].sum())})


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("floor", choices=sorted({spec["source_floor"] for spec in SPECS.values()}))
    args = parser.parse_args()
    log = ROOT / "validation/geometry"
    log.mkdir(parents=True, exist_ok=True)
    prepare_cache(args.floor, log)
