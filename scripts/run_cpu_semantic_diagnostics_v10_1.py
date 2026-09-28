#!/usr/bin/env python3
"""Compact post-hoc diagnostics for semantic correctness and IGCR feedback."""
import os
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from env.virtual3d_competition_v9 import create_competition_world
from nso.components import NSO_Components
from nso.runtime_integration import NSORuntimeIntegration
from nso.cpu_sensor_contract_v10 import GridTransform, SensorPacket, json_value
from scripts.eval_competition_v9_1 import restore_evaluator, snapshot_metrics
from scripts.run_cpu_semantic_inspection_v10_1 import prefix_packets
from utils.counterfactual_surface_visibility import reference_visible, surface_increment
from utils.cpu_protocol import file_hash


def write_json(path, value):
    Path(path).write_text(json.dumps(json_value(value), ensure_ascii=False,
        indent=2, sort_keys=True, allow_nan=False) + "\n")


def branch(prepared, context, arrangement, condition):
    mode, disable_feedback = (("X", False) if condition == "wrong_semantic_X" else ("S", True))
    world = create_competition_world(context, arrangement)
    transform = GridTransform(tuple(world.shape), world.config.resolution_m)
    packets = prefix_packets(prepared, context, arrangement, world, transform)
    reference = dict(np.load(prepared / context / arrangement / "reference.npz"))
    evaluator = restore_evaluator(world, reference)
    args = SimpleNamespace(nso_backend="cpu_v10", eval=True, train_global=False,
        use_open_vocab_semantic=True, use_topo_graph=True, use_rpn_uq=True, use_igcr=True,
        cpu_score_mode=mode, cpu_disable_feedback=disable_feedback,
        cpu_max_candidates=12, cpu_coverage_slots=4, cpu_planner_revision="v10_1",
        run_id=f"v10_1-diagnostic-{context}-{arrangement}-{condition}")
    components = NSO_Components(args); components.initialize("cpu", 1, *world.shape, *world.shape)
    runtime = NSORuntimeIntegration(components, 1, world.shape)
    anchor = (*packets[-1].position, packets[-1].heading)
    runtime.start_sensor_episode(0, config=world.config, transform=transform,
        packets=packets, total_budget=198, return_anchor=anchor, paid_prefix_actions=150)
    _, before = snapshot_metrics(runtime.states[0]["mapper"], world, evaluator, reference, (.05, .10))
    future = np.zeros_like(reference["prefix_seen"]); actions = []
    while not runtime.states[0]["closed"]:
        action = runtime.next_local_action(0)
        if action is None:
            break
        frame, collision, done = world.step(action); scan = world.scan()
        packet = SensorPacket(f"{context}_{arrangement}", f"v10_1-{context}-{arrangement}",
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
    _, after = snapshot_metrics(runtime.states[0]["mapper"], world, evaluator, reference, (.05, .10))
    area = surface_increment(reference["prefix_seen"], future, reference["weights"])
    choices = [c["outputs"]["selected"] for c in components._cpu_backend.calls
               if c["method"] == "select_topo_target"]
    return dict(context=context, arrangement=arrangement, condition=condition, mode=mode,
        feedback_enabled=not disable_feedback, future_paid_actions=len(actions), actions=actions,
        selected_poses=[None if c is None else c["pose"] for c in choices],
        selected_candidate_ids=[None if c is None else c["candidate_id"] for c in choices],
        new_unique_surface_area_m2=area, final_f1_05cm=after["f1_05cm"],
        f1_gain_05cm=after["f1_05cm"] - before["f1_05cm"],
        new_area_times_final_f1_05cm=area * after["f1_05cm"],
        coverage_2d_gain_m2=after["covered_area_m2"] - before["covered_area_m2"],
        returned_to_anchor=summary["termination"]["returned_to_anchor"],
        failed=summary["termination"]["failed"], collisions=world.collisions,
        planner_revision=summary["modules"]["planner_revision"],
        gain_calibration=summary["modules"]["observed_gain_calibration"],
        feedback=summary["modules"]["feedback"])


def main(prepared, primary, output):
    output.mkdir(parents=True, exist_ok=False)
    source = Path(__file__).resolve()
    manifest = dict(schema_version="cpu_semantic_diagnostics_v10_1/1",
        status="running", created_utc=datetime.now(timezone.utc).isoformat(),
        post_hoc_diagnostics=True, source_sha256=file_hash(source),
        primary_run_sha256=file_hash(primary / "summary.json"),
        conditions=["wrong_semantic_X", "no_IGCR_feedback"])
    write_json(output / "manifest.json", manifest)
    rows = []
    for condition in manifest["conditions"]:
        for context in ("Q0", "Q1", "Q2", "Q3"):
            for arrangement in ("shelf_west", "shelf_east"):
                row = branch(prepared, context, arrangement, condition)
                rows.append(row); write_json(output / "partial.json", rows)
                print(condition, context, arrangement,
                      row["new_area_times_final_f1_05cm"], flush=True)
    primary_rows = json.loads((primary / "summary.json").read_text())["results"]
    primary_s = {(r["context"], r["arrangement"]): r for r in primary_rows if r["mode"] == "S"}
    summary = {}
    for condition in manifest["conditions"]:
        selected = [r for r in rows if r["condition"] == condition]
        diffs = [primary_s[r["context"], r["arrangement"]]["new_area_times_final_f1_05cm"]
                 - r["new_area_times_final_f1_05cm"] for r in selected]
        summary[condition] = dict(mean_joint=float(np.mean(
            [r["new_area_times_final_f1_05cm"] for r in selected])),
            primary_S_minus_diagnostic_mean=float(np.mean(diffs)),
            primary_S_better_pairs=sum(v > 0 for v in diffs),
            ties=sum(v == 0 for v in diffs), primary_S_worse_pairs=sum(v < 0 for v in diffs),
            differences=diffs, failures=sum(r["failed"] for r in selected),
            collisions=sum(r["collisions"] for r in selected))
    manifest["status"] = "complete"
    write_json(output / "manifest.json", manifest)
    write_json(output / "results.json", rows)
    write_json(output / "summary.json", summary)
    write_json(output / "artifact_hashes.json", {str(p.relative_to(output)): file_hash(p)
        for p in sorted(output.rglob("*")) if p.is_file() and p.name != "artifact_hashes.json"})
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--prepared", type=Path, required=True)
    parser.add_argument("--primary", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    main(args.prepared.resolve(), args.primary.resolve(), args.output.resolve())
