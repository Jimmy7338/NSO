#!/usr/bin/env python3
"""Run the frozen V10.1 post-prefix semantic inspection comparison.

The 150-action prefix is external shared task structure.  S/G/N each receive
the same measured prefix and then execute their own 48-action closed loop.
Simulator truth is used only by physical sensing and the post-run evaluator.
"""
import os
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil
import sys
from types import SimpleNamespace
import zipfile

import numpy as np
import open3d as o3d

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from env.virtual3d_competition_v9 import create_competition_world, collect_competition_prefix
from nso.components import NSO_Components
from nso.runtime_integration import NSORuntimeIntegration
from nso.cpu_sensor_contract_v10 import GridTransform, SensorPacket, json_value
from scripts.eval_competition_v9_1 import restore_evaluator, snapshot_metrics
from utils.counterfactual_surface_visibility import reference_visible, surface_increment
from utils.cpu_protocol import file_hash
from utils.rgbd_contract import RGBDFrame, PlanarScan

# Import transitive policy sources before freezing them.
import nso.cpu_four_modules_v10
import nso.hierarchical_options_v10
import nso.observed_gain_calibration_v10_1


def write_json(path, value):
    Path(path).write_text(json.dumps(json_value(value), ensure_ascii=False,
                                     indent=2, sort_keys=True, allow_nan=False) + "\n")


def save_mesh(path, mesh):
    np.savez_compressed(path, vertices=np.asarray(mesh.vertices),
                        triangles=np.asarray(mesh.triangles),
                        vertex_normals=np.asarray(mesh.vertex_normals),
                        vertex_colors=np.asarray(mesh.vertex_colors))


def map_arrays(mapper):
    keys = sorted(mapper.quality)
    result = dict(belief=mapper.belief, camera_seen=mapper.camera_seen,
                  visible=mapper.visible,
                  quality_keys=np.asarray(keys, dtype=np.int64).reshape(-1, 3),
                  quality_points=np.asarray([mapper.quality[k]["point"] for k in keys],
                                            dtype=float).reshape(-1, 3),
                  quality_normals=np.asarray([mapper.quality[k]["normal"] for k in keys],
                                             dtype=float).reshape(-1, 3))
    for field in ("n", "bits", "label", "information", "residual",
                  "best_range", "normal_dispersion"):
        dtype = np.int64 if field in ("n", "bits", "label") else float
        result["quality_" + field] = np.asarray(
            [mapper.quality[k][field] for k in keys], dtype=dtype)
    return result


def freeze_sources(output, protocol_path):
    paths = {Path(__file__).resolve(), protocol_path,
             ROOT / "configs/virtual3d/competition_v9_contexts.json"}
    for module in list(sys.modules.values()):
        name = getattr(module, "__file__", None)
        if name:
            path = Path(name).resolve()
            if (path.is_relative_to(ROOT) and ".venv" not in str(path.relative_to(ROOT))
                    and path.suffix == ".py"):
                paths.add(path)
    hashes = {str(path.relative_to(ROOT)): file_hash(path) for path in sorted(paths)}
    with zipfile.ZipFile(output / "sources.zip", "w", zipfile.ZIP_DEFLATED) as archive:
        for name in hashes:
            archive.write(ROOT / name, name)
    receipt = dict(schema_version="cpu_semantic_inspection_v10_1_source_freeze/1",
                   frozen_utc=datetime.now(timezone.utc).isoformat(),
                   source_sha256=hashes,
                   archive_sha256=file_hash(output / "sources.zip"),
                   constructed_worlds_before_this_freeze=0)
    write_json(output / "freeze.json", receipt)
    return receipt


def prefix_packets(prepared, context, arrangement, world, transform):
    """Advance the physical world and bind the already frozen packet bytes."""
    folder = prepared / context / arrangement / "prefix"
    records = json.loads((folder / "records.json").read_text())
    generated = collect_competition_prefix(world)
    if len(generated) != 151 or len(records) != 151:
        raise RuntimeError("unexpected prefix length")
    packets = []
    for i, row in enumerate(records):
        frame_path = folder / "frames" / f"{i:04d}.npz"
        scan_path = folder / "scans" / f"{i:04d}.npz"
        frame, scan = RGBDFrame.load(frame_path), PlanarScan.load(scan_path)
        physical = generated[i]
        if (physical["step"] != row["step"] or physical["action"] !=
                (None if i == 0 else row["action"])
                or tuple(world.prefix_actions[:i]) != tuple(r["action"] for r in records[1:i+1])
                or not np.array_equal(physical["frame"].depth_m, frame.depth_m)
                or not np.array_equal(physical["frame"].semantic, frame.semantic)
                or not np.array_equal(physical["scan"].ranges_m, scan.ranges_m)):
            raise RuntimeError(f"physical prefix differs from frozen input at {i}")
        packet = SensorPacket(f"{context}_{arrangement}",
            f"v10_1-{context}-{arrangement}", f"frame-{i}", i, frame, scan,
            tuple(map(int, row["position"])), int(row["heading"]),
            "frozen_v9_rgbd_and_planar_scan", "simulator_exact_discrete_odometry",
            None if i == 0 else row["action"], False, False).validate(transform, world.config)
        packets.append(packet)
    return packets


def run_branch(prepared, output, context, arrangement, mode, protocol):
    folder = output / context / arrangement / mode
    (folder / "raw").mkdir(parents=True)
    world = create_competition_world(context, arrangement)
    transform = GridTransform(tuple(world.shape), world.config.resolution_m)
    packets = prefix_packets(prepared, context, arrangement, world, transform)
    reference = dict(np.load(prepared / context / arrangement / "reference.npz"))
    evaluator = restore_evaluator(world, reference)
    args = SimpleNamespace(nso_backend="cpu_v10", eval=True, train_global=False,
        use_open_vocab_semantic=True, use_topo_graph=True, use_rpn_uq=True, use_igcr=True,
        cpu_score_mode=mode, cpu_disable_feedback=False,
        cpu_max_candidates=protocol["max_candidates"],
        cpu_coverage_slots=protocol["coverage_slots"], cpu_planner_revision="v10_1",
        run_id=output.name + f"-{context}-{arrangement}-{mode}")
    components = NSO_Components(args)
    components.initialize("cpu", 1, *world.shape, *world.shape)
    runtime = NSORuntimeIntegration(components, 1, world.shape)
    anchor = (*packets[-1].position, packets[-1].heading)
    runtime.start_sensor_episode(0, config=world.config, transform=transform,
        packets=packets, total_budget=198, return_anchor=anchor, paid_prefix_actions=150)
    mapper = runtime.states[0]["mapper"]
    before_mesh, before = snapshot_metrics(mapper, world, evaluator, reference, (.05, .10))
    save_mesh(folder / "prefix_mesh.npz", before_mesh)
    future_seen = np.zeros_like(reference["prefix_seen"])
    actions, raw = [], []
    while not runtime.states[0]["closed"]:
        action = runtime.next_local_action(0)
        if action is None:
            break
        frame, collision, done = world.step(action)
        scan = world.scan()
        packet = SensorPacket(f"{context}_{arrangement}",
            f"v10_1-{context}-{arrangement}", f"frame-{world.step_count}",
            int(world.step_count), frame, scan, tuple(map(int, world.position)), int(world.heading),
            "rendered_rgb_marker_and_planar_scan", "simulator_exact_discrete_odometry",
            action, bool(collision), bool(done)).validate(transform, world.config)
        local_index = world.step_count - 150
        frame_path = folder / "raw" / f"frame_{local_index:03d}.npz"
        scan_path = folder / "raw" / f"scan_{local_index:03d}.npz"
        frame.save(frame_path); scan.save(scan_path)
        future_seen |= reference_visible(evaluator.reference, frame, evaluator.truth,
                                         frame.world_from_camera, world.config.max_depth_m)
        result = runtime.observe(0, world.step_count, None, None, None, sensor_packet=packet)
        row = dict(local_action_index=local_index, absolute_action_id=world.step_count,
            action=action, position=list(world.position), heading=world.heading,
            collision=bool(collision), done=bool(done), packet_sha256=packet.sha256(),
            accepted=result["accepted"])
        actions.append(row)
        raw.append(dict(**row, frame_path=str(frame_path.relative_to(folder)),
                        scan_path=str(scan_path.relative_to(folder)),
                        frame_sha256=file_hash(frame_path), scan_sha256=file_hash(scan_path)))
        if world.step_count > 198:
            raise RuntimeError("48-action post-prefix budget exceeded")
    summary = runtime.sensor_episode_summary(0)
    mapper = runtime.states[0]["mapper"]
    mesh, after = snapshot_metrics(mapper, world, evaluator, reference, (.05, .10))
    save_mesh(folder / "final_mesh.npz", mesh)
    np.savez_compressed(folder / "final_map.npz", **map_arrays(mapper))
    np.savez_compressed(folder / "visibility.npz", prefix=reference["prefix_seen"], future=future_seen)
    new_area = surface_increment(reference["prefix_seen"], future_seen, reference["weights"])
    result = dict(context=context, arrangement=arrangement, mode=mode,
        prefix_paid_actions=150, future_paid_actions=len(actions),
        task_paid_actions=150 + len(actions), collision_count=world.collisions,
        unique_future_positions=len({tuple(row["position"]) for row in actions}),
        forward_actions=sum(row["action"] == "forward" for row in actions),
        before=before, after=after, new_unique_surface_area_m2=new_area,
        new_area_times_final_f1_05cm=new_area * after["f1_05cm"],
        new_area_times_final_f1_10cm=new_area * after["f1_10cm"],
        f1_gain_05cm=after["f1_05cm"] - before["f1_05cm"],
        coverage_2d_gain_m2=after["covered_area_m2"] - before["covered_area_m2"],
        returned_to_anchor=bool(summary["termination"]["returned_to_anchor"]),
        failed=bool(summary["termination"]["failed"]),
        termination_reason=summary["termination"]["reason"],
        global_choices=summary["modules"]["plans"], arrived_count=summary["arrived_count"],
        gain_calibration=summary["modules"]["observed_gain_calibration"])
    write_json(folder / "actions.json", actions)
    write_json(folder / "raw_manifest.json", raw)
    write_json(folder / "module_calls.json", components._cpu_backend.calls)
    write_json(folder / "runtime_events.json", runtime.audit)
    write_json(folder / "runtime_summary.json", summary)
    write_json(folder / "result.json", result)
    return result


def aggregate(rows, protocol):
    modes = protocol["semantic_inspection_fixture"]["primary_internal_comparisons"]
    by_mode = {}
    for mode in modes:
        selected = [r for r in rows if r["mode"] == mode]
        by_mode[mode] = {key: float(np.mean([r[key] for r in selected])) for key in (
            "new_unique_surface_area_m2", "new_area_times_final_f1_05cm",
            "f1_gain_05cm", "coverage_2d_gain_m2", "forward_actions")}
        by_mode[mode].update(branches=len(selected), failures=sum(r["failed"] for r in selected),
                             collisions=sum(r["collision_count"] for r in selected))
    paired = {}
    for baseline in ("G", "N"):
        diffs = []
        for context in protocol["semantic_inspection_fixture"]["contexts"]:
            for arrangement in protocol["semantic_inspection_fixture"]["arrangements"]:
                lookup = {(r["context"], r["arrangement"], r["mode"]): r for r in rows}
                s = lookup[context, arrangement, "S"]["new_area_times_final_f1_05cm"]
                b = lookup[context, arrangement, baseline]["new_area_times_final_f1_05cm"]
                diffs.append(s - b)
        paired["S_vs_" + baseline] = dict(mean_difference=float(np.mean(diffs)),
            positive_pairs=sum(v > 0 for v in diffs), zero_pairs=sum(v == 0 for v in diffs),
            negative_pairs=sum(v < 0 for v in diffs), differences=diffs)
    gate = dict(safe=all(not r["failed"] and r["collision_count"] == 0 for r in rows),
        positive_mean_over_G=paired["S_vs_G"]["mean_difference"] > 0,
        positive_mean_over_N=paired["S_vs_N"]["mean_difference"] > 0)
    return dict(by_mode=by_mode, paired_joint_differences=paired, gates=gate,
                passed_numeric_portion=all(gate.values()))


def run(prepared, output):
    if output.exists():
        raise FileExistsError(output)
    if shutil.disk_usage(ROOT).free < 400 * 1024**2:
        raise RuntimeError("at least 400 MiB free is required before this bounded run")
    protocol_path = ROOT / "configs/virtual3d/cpu_four_module_v10_1_development_protocol.json"
    protocol = json.loads(protocol_path.read_text())
    output.mkdir(parents=True)
    freeze = freeze_sources(output, protocol_path)
    write_json(output / "manifest.json", dict(
        schema_version="cpu_semantic_inspection_v10_1_run/1", protocol=protocol,
        prepared_input=str(prepared.resolve()), prepared_manifest_sha256=file_hash(prepared / "artifact_hashes.json"),
        source_sha256=freeze["source_sha256"], source_archive_sha256=freeze["archive_sha256"],
        evaluator_truth_available_to_policy=False,
        prefix_role="externally scripted shared facility coverage task"))
    rows = []
    try:
        fixture = protocol["semantic_inspection_fixture"]
        for context in fixture["contexts"]:
            for arrangement in fixture["arrangements"]:
                for mode in fixture["primary_internal_comparisons"]:
                    row = run_branch(prepared, output, context, arrangement, mode, protocol)
                    rows.append(row)
                    write_json(output / "partial_results.json", rows)
                    print(context, arrangement, mode, row["future_paid_actions"],
                          row["new_area_times_final_f1_05cm"], flush=True)
        summary = aggregate(rows, protocol)
        summary.update(status="complete_pending_independent_replay", branches=len(rows), results=rows)
        write_json(output / "summary.json", summary)
        changed = [name for name, sha in freeze["source_sha256"].items()
                   if file_hash(ROOT / name) != sha]
        if changed:
            raise RuntimeError(f"source changed during execution: {changed}")
    except Exception as exc:
        write_json(output / "failure.json", dict(error_type=type(exc).__name__, reason=str(exc),
                                                  completed_branches=len(rows)))
        raise
    finally:
        write_json(output / "artifact_hashes.json", {
            str(p.relative_to(output)): file_hash(p) for p in sorted(output.rglob("*"))
            if p.is_file() and p.name != "artifact_hashes.json"})


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--prepared", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    run(args.prepared.resolve(), args.output.resolve())
