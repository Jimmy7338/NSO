#!/usr/bin/env python3
"""Independent physical, mapping, metric and decision audit for V10.1 outputs.

This verifier never calls the four-module planner or runtime integration.
It was written after the development execution and is therefore an independent
replay check, not a prospectively frozen analysis program.
"""
import os
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from env.virtual3d_competition_v9 import create_competition_world, collect_competition_prefix
from nso.cpu_sensor_contract_v10 import GridTransform, SensorPacket, json_value
from nso.observed_runtime_mapper_v10 import ObservedRuntimeMapperV10
from scripts.eval_competition_v9_1 import restore_evaluator, snapshot_metrics
from utils.counterfactual_surface_visibility import reference_visible, surface_increment
from utils.cpu_protocol import file_hash
from utils.rgbd_contract import RGBDFrame, PlanarScan


def write_json(path, value):
    Path(path).write_text(json.dumps(json_value(value), ensure_ascii=False,
                                     indent=2, sort_keys=True, allow_nan=False) + "\n")


def compare_frame(actual, expected):
    for name in actual.__dataclass_fields__:
        np.testing.assert_array_equal(np.asarray(getattr(actual, name)),
                                      np.asarray(getattr(expected, name)))


def compare_scan(actual, expected):
    for name in actual.__dataclass_fields__:
        np.testing.assert_array_equal(np.asarray(getattr(actual, name)),
                                      np.asarray(getattr(expected, name)))


def compare_saved_map(mapper, path):
    saved = np.load(path)
    np.testing.assert_array_equal(saved["belief"], mapper.belief)
    np.testing.assert_array_equal(saved["camera_seen"], mapper.camera_seen)
    np.testing.assert_array_equal(saved["visible"], mapper.visible)
    keys = sorted(mapper.quality)
    np.testing.assert_array_equal(saved["quality_keys"],
                                  np.asarray(keys, dtype=np.int64).reshape(-1, 3))
    np.testing.assert_array_equal(saved["quality_points"],
        np.asarray([mapper.quality[k]["point"] for k in keys], dtype=float).reshape(-1, 3))
    np.testing.assert_array_equal(saved["quality_normals"],
        np.asarray([mapper.quality[k]["normal"] for k in keys], dtype=float).reshape(-1, 3))
    for field in ("n", "bits", "label", "information", "residual",
                  "best_range", "normal_dispersion"):
        np.testing.assert_array_equal(saved["quality_" + field],
            np.asarray([mapper.quality[k][field] for k in keys]))


def compare_mesh(mapper, path):
    mesh, saved = mapper.mesh(), np.load(path)
    for name in ("vertices", "triangles", "vertex_normals", "vertex_colors"):
        np.testing.assert_array_equal(saved[name], np.asarray(getattr(mesh, name)))


def validate_decisions(folder, mode):
    calls = json.loads((folder / "module_calls.json").read_text())
    if not all(any(c["module"] == module for c in calls)
               for module in ("OV-SDF", "STGHP", "RPN-UQ", "IGCR")):
        raise AssertionError("a four-module interface is absent")
    if any(c["truth_used"] or c["trained"] or c["calibrated"] for c in calls):
        raise AssertionError("policy log claims forbidden truth/training/calibration")
    choices = [c for c in calls if c["method"] == "select_topo_target"]
    chosen_poses = []
    for call in choices:
        out = call["outputs"]
        if out["mode"] != mode or out["planner_revision"] != "v10_1":
            raise AssertionError("wrong internal comparison or planner revision")
        routes, scores, selected = out["candidates"], out["scores"][mode], out["selected"]
        expected = (min(range(len(routes)), key=lambda i: (-scores[i], routes[i]["cost"],
                    routes[i]["candidate_id"])) if routes else None)
        expected = None if expected is None or scores[expected] <= 0 else routes[expected]["candidate_id"]
        actual = None if selected is None else selected["candidate_id"]
        if actual != expected:
            raise AssertionError("logged choice is not the logged score argmax")
        chosen_poses.append(None if selected is None else selected["pose"])
    return dict(calls=len(calls), choices=len(choices), chosen_poses=chosen_poses,
                call_chain_ids_strict=all(c["call_id"] == i + 1 for i, c in enumerate(calls)))


def replay_branch(prepared, folder, context, arrangement, mode):
    result = json.loads((folder / "result.json").read_text())
    actions = json.loads((folder / "actions.json").read_text())
    raw = json.loads((folder / "raw_manifest.json").read_text())
    world = create_competition_world(context, arrangement)
    generated = collect_competition_prefix(world)
    prefix = prepared / context / arrangement / "prefix"
    records = json.loads((prefix / "records.json").read_text())
    mapper = ObservedRuntimeMapperV10(world.shape, world.config)
    for i, row in enumerate(records):
        frame = RGBDFrame.load(prefix / "frames" / f"{i:04d}.npz")
        scan = PlanarScan.load(prefix / "scans" / f"{i:04d}.npz")
        compare_frame(generated[i]["frame"], frame)
        compare_scan(generated[i]["scan"], scan)
        mapper.update(frame, scan)
    reference = dict(np.load(prepared / context / arrangement / "reference.npz"))
    evaluator = restore_evaluator(world, reference)
    _, before = snapshot_metrics(mapper, world, evaluator, reference, (.05, .10))
    future = np.zeros_like(reference["prefix_seen"])
    transform = GridTransform(tuple(world.shape), world.config.resolution_m)
    if len(actions) != len(raw) or len(actions) != 48:
        raise AssertionError("incomplete future action record")
    for index, (action_row, raw_row) in enumerate(zip(actions, raw), 1):
        frame, collision, done = world.step(action_row["action"])
        scan = world.scan()
        saved_frame = RGBDFrame.load(folder / raw_row["frame_path"])
        saved_scan = PlanarScan.load(folder / raw_row["scan_path"])
        compare_frame(frame, saved_frame); compare_scan(scan, saved_scan)
        if file_hash(folder / raw_row["frame_path"]) != raw_row["frame_sha256"]:
            raise AssertionError("frame hash mismatch")
        if file_hash(folder / raw_row["scan_path"]) != raw_row["scan_sha256"]:
            raise AssertionError("scan hash mismatch")
        packet = SensorPacket(f"{context}_{arrangement}", f"v10_1-{context}-{arrangement}",
            f"frame-{world.step_count}", int(world.step_count), saved_frame, saved_scan,
            tuple(map(int, world.position)), int(world.heading),
            "rendered_rgb_marker_and_planar_scan", "simulator_exact_discrete_odometry",
            action_row["action"], bool(collision), bool(done)).validate(transform, world.config)
        if packet.sha256() != raw_row["packet_sha256"]:
            raise AssertionError("packet hash mismatch")
        if (world.step_count != action_row["absolute_action_id"] or collision
                or tuple(world.position) != tuple(action_row["position"])
                or world.heading != action_row["heading"]):
            raise AssertionError("physical trajectory mismatch")
        mapper.update(saved_frame, saved_scan)
        future |= reference_visible(evaluator.reference, saved_frame, evaluator.truth,
                                    saved_frame.world_from_camera, world.config.max_depth_m)
    compare_saved_map(mapper, folder / "final_map.npz")
    compare_mesh(mapper, folder / "final_mesh.npz")
    _, after = snapshot_metrics(mapper, world, evaluator, reference, (.05, .10))
    area = surface_increment(reference["prefix_seen"], future, reference["weights"])
    np.testing.assert_allclose(area, result["new_unique_surface_area_m2"], rtol=0, atol=1e-12)
    for key in before:
        a, b = before[key], result["before"][key]
        if a is None or b is None:
            assert a is b
        else:
            np.testing.assert_allclose(a, b, rtol=0, atol=1e-12)
    for key in after:
        a, b = after[key], result["after"][key]
        if a is None or b is None:
            assert a is b
        else:
            np.testing.assert_allclose(a, b, rtol=0, atol=1e-12)
    np.testing.assert_allclose(area * after["f1_05cm"],
                               result["new_area_times_final_f1_05cm"], rtol=0, atol=1e-12)
    anchor = tuple(records[-1][key] for key in ())
    expected_anchor = (*records[-1]["position"], records[-1]["heading"])
    if (*world.position, world.heading) != expected_anchor or world.collisions != 0:
        raise AssertionError("branch did not return safely")
    decision = validate_decisions(folder, mode)
    return dict(context=context, arrangement=arrangement, mode=mode,
                physical_packets_exact=True, final_map_exact=True, final_mesh_exact=True,
                metrics_exact=True, safe_return=True, **decision)


def main(run, prepared, output):
    expected_hashes = json.loads((run / "artifact_hashes.json").read_text())
    changed = [name for name, digest in expected_hashes.items()
               if file_hash(run / name) != digest]
    if changed:
        raise RuntimeError(f"execution artifacts changed: {changed[:5]}")
    freeze = json.loads((run / "freeze.json").read_text())
    source_changed = [name for name, digest in freeze["source_sha256"].items()
                      if file_hash(ROOT / name) != digest]
    if source_changed:
        raise RuntimeError(f"frozen execution source changed: {source_changed}")
    rows = []
    for context in ("Q0", "Q1", "Q2", "Q3"):
        for arrangement in ("shelf_west", "shelf_east"):
            for mode in ("S", "G", "N"):
                row = replay_branch(prepared, run / context / arrangement / mode,
                                    context, arrangement, mode)
                rows.append(row)
                print("replayed", context, arrangement, mode, flush=True)
    choice_effect = {}
    for context in ("Q0", "Q1", "Q2", "Q3"):
        for arrangement in ("shelf_west", "shelf_east"):
            group = {r["mode"]: r for r in rows if r["context"] == context
                     and r["arrangement"] == arrangement}
            choice_effect[f"{context}/{arrangement}"] = dict(
                S_differs_from_G=group["S"]["chosen_poses"] != group["G"]["chosen_poses"],
                S_differs_from_N=group["S"]["chosen_poses"] != group["N"]["chosen_poses"],
                first_pose={m: group[m]["chosen_poses"][0] for m in ("S", "G", "N")})
    report = dict(schema_version="cpu_semantic_inspection_v10_1_independent_replay/1",
        status="passed", verifier_written_after_execution=True,
        planner_or_runtime_called=False, branches=len(rows), checks=rows,
        choice_effect=choice_effect,
        histories_with_S_choice_different_from_G=sum(v["S_differs_from_G"] for v in choice_effect.values()),
        histories_with_S_choice_different_from_N=sum(v["S_differs_from_N"] for v in choice_effect.values()),
        artifact_hashes_verified=len(expected_hashes), frozen_source_files_verified=len(freeze["source_sha256"]))
    output.mkdir(parents=True, exist_ok=False)
    write_json(output / "verification.json", report)
    (output / "REPORT.md").write_text(
        "# V10.1 independent replay\n\n"
        f"Passed {len(rows)}/24 physical branches. The post-hoc verifier did not call the planner "
        "or runtime. It regenerated every prefix and future sensor packet, reconstructed every map "
        "and mesh, recomputed every primary metric, checked safe return, and verified logged score "
        "argmax decisions. Semantic S changed the executed choice sequence relative to G in "
        f"{report['histories_with_S_choice_different_from_G']}/8 histories and relative to N in "
        f"{report['histories_with_S_choice_different_from_N']}/8.\n")
    print(json.dumps({k: report[k] for k in ("status", "branches",
          "histories_with_S_choice_different_from_G", "histories_with_S_choice_different_from_N")}, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--prepared", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    main(args.run.resolve(), args.prepared.resolve(), args.output.resolve())
