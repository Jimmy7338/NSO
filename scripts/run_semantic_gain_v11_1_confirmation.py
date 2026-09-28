#!/usr/bin/env python3
"""Run the frozen V11.1 method on sealed, previously unopened F contexts."""
import os
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import open3d as o3d

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from env.virtual3d_confirmation_v11_2 import create_confirmation_world
from nso.components import NSO_Components
from nso.runtime_integration import NSORuntimeIntegration
from nso.cpu_sensor_contract_v10 import GridTransform, SensorPacket, json_value
from scripts.eval_competition_v9_1 import restore_evaluator, snapshot_metrics
from scripts.run_cpu_semantic_inspection_v10_1 import prefix_packets
from utils.counterfactual_surface_visibility import reference_visible, surface_increment
from utils.reconstruction_metrics import ray_scene


CONDITION = {
    "learned_semantic_feedback": ("L", False, 1.),
    "learned_semantic_no_feedback": ("L", True, 1.),
    "learned_geometry": ("K", False, 1.),
    "fixed_semantic": ("S", False, 1.),
    "fixed_geometry": ("G", False, 1.),
    "learned_swapped": ("Y", False, 1.),
    "low_confidence_fallback": ("Q", False, 0.),
}


def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path, value):
    Path(path).write_text(json.dumps(json_value(value), ensure_ascii=False, indent=2,
        sort_keys=True, allow_nan=False) + "\n")


def execute(prepared, model, context, arrangement, condition):
    mode, disable_feedback, confidence = CONDITION[condition]
    world = create_confirmation_world(context, arrangement)
    transform = GridTransform(tuple(world.shape), world.config.resolution_m)
    packets = prefix_packets(prepared, context, arrangement, world, transform)
    reference = dict(np.load(prepared / context / arrangement / "reference.npz"))
    evaluator = restore_evaluator(world, reference)
    args = SimpleNamespace(nso_backend="cpu_v10", eval=True, train_global=False,
        use_open_vocab_semantic=True, use_topo_graph=True, use_rpn_uq=True, use_igcr=True,
        cpu_score_mode=mode, cpu_disable_feedback=disable_feedback,
        cpu_max_candidates=6, cpu_coverage_slots=4, cpu_planner_revision="v11_1",
        cpu_measured_novelty_floor=.25, cpu_semantic_gain_model_path=str(model),
        cpu_semantic_confidence_scale=confidence, cpu_semantic_confidence_threshold=.25,
        run_id=f"v11_1-{context}-{arrangement}-{condition}")
    components = NSO_Components(args); components.initialize("cpu", 1, *world.shape, *world.shape)
    runtime = NSORuntimeIntegration(components, 1, world.shape)
    anchor = (*packets[-1].position, packets[-1].heading)
    prefix_actions = int(world.context.prefix_step)
    runtime.start_sensor_episode(0, config=world.config, transform=transform,
        packets=packets, total_budget=prefix_actions + 48,
        return_anchor=anchor, paid_prefix_actions=prefix_actions)
    _, before = snapshot_metrics(runtime.states[0]["mapper"], world, evaluator, reference, (.05, .10))
    future = np.zeros_like(reference["prefix_seen"]); actions = []
    while not runtime.states[0]["closed"]:
        action = runtime.next_local_action(0)
        if action is None: break
        frame, collision, done = world.step(action); scan = world.scan()
        previous_packet = runtime.states[0]["packet"]
        packet = SensorPacket(previous_packet.scene_id, previous_packet.episode_id,
            f"frame-{world.step_count}", int(world.step_count), frame, scan,
            tuple(map(int, world.position)), int(world.heading),
            "deterministic_simulator_rgbd_and_scan", "simulator_exact_discrete_odometry",
            action, bool(collision), bool(done)).validate(transform, world.config)
        future |= reference_visible(evaluator.reference, frame, evaluator.truth,
                                    frame.world_from_camera, world.config.max_depth_m)
        runtime.observe(0, world.step_count, None, None, None, sensor_packet=packet)
        actions.append(dict(action=action, position=list(world.position), heading=world.heading,
                            packet_sha256=packet.sha256()))
    summary = runtime.sensor_episode_summary(0)
    mesh, after = snapshot_metrics(runtime.states[0]["mapper"], world, evaluator, reference, (.05, .10))
    area = surface_increment(reference["prefix_seen"], future, reference["weights"])
    new = future & ~reference["prefix_seen"]; points = evaluator.reference[new]
    distances = (ray_scene(mesh).compute_distance(o3d.core.Tensor(points.astype(np.float32)),
        nthreads=1).numpy() if len(points) and len(mesh.triangles) else np.asarray([]))
    calls = [c for c in components._cpu_backend.calls if c["method"] == "select_topo_target"]
    modules = summary["modules"]
    return dict(context=context, stratum=world.context.stratum,
        arrangement=arrangement, condition=condition, mode=mode,
        feedback_enabled=not disable_feedback, confidence_scale=confidence,
        actions=actions, future_paid_actions=len(actions),
        selected_candidate_ids=[None if c["outputs"]["selected"] is None else
            c["outputs"]["selected"]["candidate_id"] for c in calls],
        selected_poses=[None if c["outputs"]["selected"] is None else
            c["outputs"]["selected"]["pose"] for c in calls],
        in_distribution_plan_count=sum(all(r["v11_in_distribution"] for r in c["outputs"]["score_audit"])
            for c in calls if c["outputs"]["score_audit"]), total_plan_count=len(calls),
        new_visible_surface_area_m2=area,
        joint_gain=after["area_times_f1_05cm"] - before["area_times_f1_05cm"],
        inspection_joint=area * after["f1_05cm"],
        before_metrics=before, after_metrics=after,
        new_visible_local_quality=dict(reference_samples=len(points),
            recall_05cm=float(np.mean(distances <= .05)) if len(distances) else 0.,
            completeness_error_mean_m=float(distances.mean()) if len(distances) else None,
            completeness_error_p95_m=float(np.percentile(distances, 95)) if len(distances) else None),
        path_distance_m=sum(a["action"] == "forward" for a in actions) * world.config.resolution_m,
        action_time_s=len(actions) * world.config.action_duration_s,
        collisions=world.collisions, returned_to_anchor=summary["termination"]["returned_to_anchor"],
        failed=summary["termination"]["failed"], modules_called=sorted(set(
            c["module"] for c in components._cpu_backend.calls)),
        observed_gain=modules["observed_gain_calibration"],
        conditional_gain=modules["conditional_gain_residual"])


def pair(rows, primary, baseline, metric="joint_gain"):
    lookup = {(r["context"], r["arrangement"], r["condition"]): r for r in rows}
    keys = sorted((r["context"], r["arrangement"]) for r in rows if r["condition"] == primary)
    diffs = [lookup[c, a, primary][metric] - lookup[c, a, baseline][metric] for c, a in keys]
    return dict(mean_difference=float(np.mean(diffs)), wins=sum(x > 0 for x in diffs),
                ties=sum(x == 0 for x in diffs), losses=sum(x < 0 for x in diffs), differences=diffs)


def main(prepared, output, stratum):
    output.mkdir(parents=True, exist_ok=False)
    protocol_path = ROOT / "configs/virtual3d/semantic_gain_v11_1_confirmation2_protocol.json"
    protocol = json.loads(protocol_path.read_text())
    model = ROOT / protocol["model_bundle"]
    sources = [Path(__file__).resolve(), protocol_path, model,
        ROOT / "nso/cpu_four_modules_v10.py", ROOT / "nso/semantic_gain_v11.py",
        ROOT / "nso/runtime_integration.py", ROOT / "nso/hierarchical_options_v10.py"]
    preparation_hashes = prepared / "artifact_hashes.json"
    manifest = dict(schema_version="semantic_gain_v11_1_confirmation/1", status="running",
        created_utc=datetime.now(timezone.utc).isoformat(), protocol=protocol, stratum=stratum,
        source_sha256={str(p.relative_to(ROOT)): sha(p) for p in sources},
        preparation_artifact_hashes_sha256=sha(preparation_hashes),
        confirmation_retraining=False, independent_confirmation=True)
    write(output / "manifest.json", manifest)
    if stratum not in ("efficacy", "coverage_stress"):
        raise ValueError("unknown confirmation stratum")
    conditions = tuple(protocol["efficacy_conditions" if stratum == "efficacy"
                                else "stress_conditions"])
    contexts = tuple(protocol["efficacy_contexts" if stratum == "efficacy"
                              else "coverage_stress_contexts"])
    arrangements = tuple(protocol["arrangements"])
    rows = []
    for condition in conditions:
        for context in contexts:
            for arrangement in arrangements:
                row = execute(prepared, model, context, arrangement, condition)
                rows.append(row); write(output / "partial.json", rows)
                print(condition, context, arrangement, row["joint_gain"],
                      row["conditional_gain"]["observed_actions"], flush=True)
    by = {condition: dict(mean_joint_gain=float(np.mean([r["joint_gain"]
            for r in rows if r["condition"] == condition])),
        mean_inspection_joint=float(np.mean([r["inspection_joint"]
            for r in rows if r["condition"] == condition])),
        mean_coverage_times_f1=float(np.mean([r["after_metrics"]["area_times_f1_05cm"]
            for r in rows if r["condition"] == condition])),
        mean_f1=float(np.mean([r["after_metrics"]["f1_05cm"]
            for r in rows if r["condition"] == condition])),
        mean_precision=float(np.mean([r["after_metrics"]["precision_05cm"]
            for r in rows if r["condition"] == condition])),
        mean_recall=float(np.mean([r["after_metrics"]["recall_05cm"]
            for r in rows if r["condition"] == condition])),
        mean_path_distance_m=float(np.mean([r["path_distance_m"]
            for r in rows if r["condition"] == condition]))) for condition in conditions}
    comparisons = {"semantic_vs_" + baseline: pair(rows, "learned_semantic_feedback", baseline)
        for baseline in conditions if baseline != "learned_semantic_feedback"}
    gates = dict(all_safe_and_four_interfaces=all(not r["failed"] and not r["collisions"]
        and r["returned_to_anchor"] and r["modules_called"] == ["IGCR", "OV-SDF", "RPN-UQ", "STGHP"]
        for r in rows))
    if stratum == "efficacy":
        gates.update(semantic_above_learned_geometry=comparisons[
            "semantic_vs_learned_geometry"]["mean_difference"] > 0,
            semantic_above_fixed_semantic=comparisons[
                "semantic_vs_fixed_semantic"]["mean_difference"] > 0,
            correct_semantics_above_swapped=comparisons[
                "semantic_vs_learned_swapped"]["mean_difference"] > 0,
            no_pairwise_loss_to_geometry=comparisons[
                "semantic_vs_learned_geometry"]["losses"] == 0,
            low_confidence_matches_geometry=all(
                a["actions"] == b["actions"] for a, b in zip(
                    sorted([r for r in rows if r["condition"] == "low_confidence_fallback"],
                           key=lambda x:(x["context"],x["arrangement"])),
                    sorted([r for r in rows if r["condition"] == "learned_geometry"],
                           key=lambda x:(x["context"],x["arrangement"])))),
            )
    summary = dict(schema_version="semantic_gain_v11_1_confirmation_result/1",
        status="passed" if all(gates.values()) else "failed", development_only=False,
        branches=len(rows), by_condition=by, comparisons=comparisons,
        gates=gates, passed=all(gates.values()), stratum=stratum,
        independent_confirmation=True)
    write(output / "results.json", rows); write(output / "summary.json", summary)
    manifest["status"] = "complete"; write(output / "manifest.json", manifest)
    if any(sha(ROOT / name) != value for name, value in manifest["source_sha256"].items()):
        raise RuntimeError("source changed during V11 closed-loop run")
    write(output / "artifact_hashes.json", {str(p.relative_to(output)): sha(p)
        for p in sorted(output.rglob("*")) if p.is_file() and p.name != "artifact_hashes.json"})
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    p = argparse.ArgumentParser(); p.add_argument("--prepared", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--stratum", choices=("efficacy", "coverage_stress"), required=True)
    a = p.parse_args()
    try:
        main(a.prepared.resolve(), a.output.resolve(), a.stratum)
    except Exception as exc:
        manifest_path = a.output.resolve() / "manifest.json"
        if manifest_path.exists():
            manifest = json.loads(manifest_path.read_text())
            manifest["status"] = "failed"
            manifest["error"] = {"type": type(exc).__name__, "message": str(exc)}
            write(manifest_path, manifest)
        raise
