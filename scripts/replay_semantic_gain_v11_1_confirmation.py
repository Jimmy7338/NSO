#!/usr/bin/env python3
"""Independent physical and metric replay of frozen V11.1 confirmation rows.

This verifier does not import or call STGHP, the learned gain model, the option
generator, the four-module backend, or the runtime integration layer.
"""
import os
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

import argparse
import json
from pathlib import Path
import sys

import numpy as np
import open3d as o3d

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from env.virtual3d_confirmation_v11_2 import (create_confirmation_world,
    collect_confirmation_prefix)
from nso.cpu_sensor_contract_v10 import GridTransform, SensorPacket, json_value
from nso.observed_runtime_mapper_v10 import ObservedRuntimeMapperV10
from scripts.eval_competition_v9_1 import restore_evaluator, snapshot_metrics
from utils.counterfactual_surface_visibility import reference_visible, surface_increment
from utils.cpu_protocol import file_hash
from utils.reconstruction_metrics import ray_scene
from utils.rgbd_contract import RGBDFrame, PlanarScan


def write(path, value):
    Path(path).write_text(json.dumps(json_value(value), ensure_ascii=False, indent=2,
        sort_keys=True, allow_nan=False) + "\n")


def exact(actual, expected, name):
    if expected is None:
        if actual is not None:
            raise AssertionError(f"{name}: expected None")
    elif isinstance(expected, (int, float)):
        np.testing.assert_allclose(actual, expected, rtol=0, atol=1e-12,
                                   err_msg=name)
    elif actual != expected:
        raise AssertionError(f"{name}: mismatch")


def replay(prepared, row):
    context, arrangement = row["context"], row["arrangement"]
    world = create_confirmation_world(context, arrangement)
    generated = collect_confirmation_prefix(world)
    folder = prepared / context / arrangement / "prefix"
    records = json.loads((folder / "records.json").read_text())
    mapper = ObservedRuntimeMapperV10(world.shape, world.config)
    for index in range(world.context.prefix_step + 1):
        frame = RGBDFrame.load(folder / "frames" / f"{index:04d}.npz")
        scan = PlanarScan.load(folder / "scans" / f"{index:04d}.npz")
        np.testing.assert_array_equal(frame.depth_m, generated[index]["frame"].depth_m)
        np.testing.assert_array_equal(frame.semantic, generated[index]["frame"].semantic)
        np.testing.assert_array_equal(scan.ranges_m, generated[index]["scan"].ranges_m)
        mapper.update(frame, scan)
    reference = dict(np.load(prepared / context / arrangement / "reference.npz"))
    evaluator = restore_evaluator(world, reference)
    _, before = snapshot_metrics(mapper, world, evaluator, reference, (.05, .10))
    future = np.zeros_like(reference["prefix_seen"])
    transform = GridTransform(tuple(world.shape), world.config.resolution_m)
    for saved in row["actions"]:
        frame, collision, done = world.step(saved["action"])
        scan = world.scan()
        packet = SensorPacket(f"{context}_{arrangement}", f"v10_1-{context}-{arrangement}",
            f"frame-{world.step_count}", int(world.step_count), frame, scan,
            tuple(map(int, world.position)), int(world.heading),
            "deterministic_simulator_rgbd_and_scan", "simulator_exact_discrete_odometry",
            saved["action"], bool(collision), bool(done)).validate(transform, world.config)
        if (packet.sha256() != saved["packet_sha256"]
                or tuple(world.position) != tuple(saved["position"])
                or world.heading != saved["heading"]):
            raise AssertionError("physical action, pose, or packet mismatch")
        mapper.update(frame, scan)
        future |= reference_visible(evaluator.reference, frame, evaluator.truth,
                                    frame.world_from_camera, world.config.max_depth_m)
    mesh, after = snapshot_metrics(mapper, world, evaluator, reference, (.05, .10))
    area = surface_increment(reference["prefix_seen"], future, reference["weights"])
    joint_gain = after["area_times_f1_05cm"] - before["area_times_f1_05cm"]
    inspection_joint = area * after["f1_05cm"]
    exact(area, row["new_visible_surface_area_m2"], "new visible surface area")
    exact(joint_gain, row["joint_gain"], "primary joint gain")
    exact(inspection_joint, row["inspection_joint"], "inspection joint")
    for name, expected in row["before_metrics"].items():
        exact(before[name], expected, f"before.{name}")
    for name, expected in row["after_metrics"].items():
        exact(after[name], expected, f"after.{name}")
    new = future & ~reference["prefix_seen"]
    points = evaluator.reference[new]
    distances = (ray_scene(mesh).compute_distance(o3d.core.Tensor(points.astype(np.float32)),
        nthreads=1).numpy() if len(points) and len(mesh.triangles) else np.asarray([]))
    local = row["new_visible_local_quality"]
    exact(len(points), local["reference_samples"], "local reference samples")
    exact(float(np.mean(distances <= .05)) if len(distances) else 0.,
          local["recall_05cm"], "local recall")
    exact(float(distances.mean()) if len(distances) else None,
          local["completeness_error_mean_m"], "local mean error")
    exact(float(np.percentile(distances, 95)) if len(distances) else None,
          local["completeness_error_p95_m"], "local p95 error")
    exact(sum(saved["action"] == "forward" for saved in row["actions"])
          * world.config.resolution_m, row["path_distance_m"], "path distance")
    exact(len(row["actions"]) * world.config.action_duration_s,
          row["action_time_s"], "action time")
    anchor = (*records[-1]["position"], records[-1]["heading"])
    if (*world.position, world.heading) != anchor or world.collisions:
        raise AssertionError("unsafe terminal state")
    if row["failed"] or row["collisions"] or not row["returned_to_anchor"]:
        raise AssertionError("saved safety claims contradict replay")
    return {"context": context, "arrangement": arrangement,
        "condition": row["condition"], "packets_exact": True, "metrics_exact": True,
        "safe_return": True, "joint_gain": joint_gain,
        "inspection_joint": inspection_joint, "actions": len(row["actions"])}


def main(run, prepared, freeze, output):
    receipt = json.loads((freeze / "receipt.json").read_text())
    for name, digest in receipt["method_source_sha256"].items():
        if file_hash(ROOT / name) != digest:
            raise RuntimeError("execution-freeze source changed")
    if file_hash(prepared / "artifact_hashes.json") != receipt[
            "preparation_artifact_hashes_sha256"]:
        raise RuntimeError("prepared confirmation inventory changed")
    hashes = json.loads((run / "artifact_hashes.json").read_text())
    if any(file_hash(run / name) != digest for name, digest in hashes.items()):
        raise RuntimeError("frozen V11.1 run artifact changed")
    manifest = json.loads((run / "manifest.json").read_text())
    if any(file_hash(ROOT / name) != digest for name, digest in manifest["source_sha256"].items()):
        raise RuntimeError("frozen V11.1 source changed")
    rows = json.loads((run / "results.json").read_text())
    if len(rows) not in (40, 96):
        raise ValueError("unexpected confirmation branch inventory")
    checks = []
    for row in rows:
        checks.append(replay(prepared, row))
        print("replayed", row["condition"], row["context"], row["arrangement"], flush=True)
    summary = json.loads((run / "summary.json").read_text())
    for condition, reported in summary["by_condition"].items():
        values = [c["joint_gain"] for c in checks if c["condition"] == condition]
        exact(float(np.mean(values)), reported["mean_joint_gain"],
              f"summary mean {condition}")
    output.mkdir(parents=True, exist_ok=False)
    report = {"schema_version": "semantic_gain_v11_1_confirmation_independent_replay/1",
        "status": "passed", "verifier_written_after_execution": True,
        "planner_runtime_or_gain_model_called": False, "branches": len(checks),
        "independent_confirmation": True,
        "artifact_hashes_verified": len(hashes),
        "frozen_source_files_verified": len(manifest["source_sha256"]),
        "checks": checks}
    write(output / "verification.json", report)
    (output / "REPORT.md").write_text(
        "# V11.1 confirmation independent physical and metric replay\n\n"
        f"Passed {len(checks)}/{len(checks)} frozen development branches. The post-hoc "
        "verifier did not call the planner, runtime integration, option generator, or learned "
        "gain model. It regenerated every prefix and future sensor packet from saved actions, "
        "rebuilt each TSDF map, recomputed the primary 2D-area-times-3D-F1 gain and all recorded "
        "secondary metrics, and checked collision-free return to the exact prefix anchor.\n")
    print(json.dumps({"status": "passed", "branches": len(checks)}, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--prepared", type=Path, required=True)
    parser.add_argument("--freeze", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    main(args.run.resolve(), args.prepared.resolve(), args.freeze.resolve(),
         args.output.resolve())
