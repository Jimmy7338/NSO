#!/usr/bin/env python3
"""Re-execute sealed physical trajectories and independently assemble metrics.

Uses the same runtime/backend, so this is physical-and-metric repeatability,
not a second implementation of the planning algorithm.
"""
import os
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
import open3d as o3d

from env.virtual3d_competition_v9 import create_competition_world
from nso.continuation_v13_2 import force_outbound_candidate
from nso.cpu_sensor_contract_v10 import GridTransform
from nso.decision_replay_v13 import decision_state, load_packet
from scripts.collect_semantic_gain_v13_history import packet, sha, start, write
from utils.reconstruction_metrics import ReconstructionEvaluator, ray_scene
from utils.counterfactual_surface_visibility import reference_visible


def metrics(runtime, world, evaluator):
    mapper = runtime.states[0]["mapper"]
    coverage = float(((mapper.belief != -1) & world.reachable).sum() / world.reachable.sum())
    result = evaluator.evaluate(mapper.mesh(), coverage)
    area = coverage * float(world.reachable.sum()) * world.config.resolution_m ** 2
    return dict(coverage_2d=coverage, covered_area_m2=area,
        area_times_f1_05cm=area * result["f1_05cm"], **{k: v for k, v in result.items()
        if k.startswith(("precision_", "recall_", "f1_", "joint_", "surface_error_"))})


def main(source, output):
    manifest = json.loads((source / "manifest.json").read_text())
    if manifest["status"] != "complete":
        raise ValueError("collection is not terminal and complete")
    for name, expected in manifest["source_sha256"].items():
        if sha(ROOT / name) != expected:
            raise ValueError(f"source hash differs: {name}")
    artifact_hashes = json.loads((source / "artifact_hashes.json").read_text())
    for name, expected in artifact_hashes.items():
        if sha(source / name) != expected:
            raise ValueError(f"collection artifact differs: {name}")
    if manifest["protocol"]["first_option_execution"] != "complete_outbound_then_common_geometry_continuation":
        raise ValueError("V13.2 outbound contract required")
    history = ROOT / manifest["protocol"]["history"]
    for name, expected in manifest["history_artifact_hashes"].items():
        if sha(history / name) != expected:
            raise ValueError(f"history artifact differs: {name}")
    config = json.loads((history / "manifest.json").read_text())["protocol"]
    history_packets = [load_packet(p) for p in sorted((history / "packets").glob("*.npz"))]
    checkpoints = {r["action_id"]: r for r in json.loads((history / "checkpoints.json").read_text())}
    rows = json.loads((source / "partial.json").read_text())
    output.mkdir(parents=True, exist_ok=False)
    frozen = {**manifest["source_sha256"], str(Path(__file__).resolve().relative_to(ROOT)): sha(Path(__file__).resolve())}
    write(output / "manifest.json", dict(status="running", source=str(source), source_sha256=frozen))
    checks = []
    try:
        with np.load(source / "reference.npz", allow_pickle=False) as data:
            reference = {k: data[k].copy() for k in data.files}
        for row in rows:
            checkpoint = checkpoints[row["checkpoint_action"]]
            name = f"step_{row['checkpoint_action']:03d}_candidate_{row['candidate_id']:03d}"
            folder = source / name
            world = create_competition_world(config["parent"], config["arrangement"])
            transform = GridTransform(tuple(world.shape), world.config.resolution_m)
            first = packet(world, config, None)
            if first.sha256() != history_packets[0].sha256():
                raise ValueError("initial regenerated packet differs")
            runtime = start(config, world, transform, first)
            for expected in history_packets[1:row["checkpoint_action"] + 1]:
                action = runtime.next_local_action(0)
                if action != expected.action:
                    raise ValueError("replayed history policy action differs")
                frame, collision, done = world.step(action)
                actual = packet(world, config, action, frame, collision, done)
                if actual.sha256() != expected.sha256():
                    raise ValueError("regenerated historical sensor bytes differ")
                runtime.observe(0, actual.action_id, None, None, None, sensor_packet=actual)
            if decision_state(runtime)["sha256"] != checkpoint["state"]["sha256"]:
                raise ValueError("restored decision state differs")
            np.testing.assert_array_equal(reference["vertices"], np.asarray(world.mesh.vertices))
            np.testing.assert_array_equal(reference["triangles"], np.asarray(world.mesh.triangles))
            evaluator = ReconstructionEvaluator.__new__(ReconstructionEvaluator)
            evaluator.config = world.config
            evaluator.truth = ray_scene(world.mesh)
            evaluator.reference, evaluator.classes = reference["points"], reference["classes"]
            before = metrics(runtime, world, evaluator)
            option = force_outbound_candidate(runtime, row["candidate_id"], row["candidate_pool_sha256"])
            forced_actions = []; first_end_pose = None
            expected_actions = json.loads((folder / "actions.json").read_text())
            visible = np.zeros(len(evaluator.reference), bool)
            for p in history_packets[:row["checkpoint_action"] + 1]:
                visible |= reference_visible(evaluator.reference, p.frame, evaluator.truth,
                                            p.frame.world_from_camera, world.config.max_depth_m)
            before_visible = visible.copy()
            for recorded in expected_actions:
                action = runtime.next_local_action(0)
                if action != recorded["action"]:
                    raise ValueError("continuation policy action differs")
                active = runtime.states[0]["option"]
                forced = active is not None and active["option_id"] == option["option_id"]
                if forced != recorded["forced_option"]:
                    raise ValueError("first-outbound attribution differs")
                frame, collision, done = world.step(action)
                if forced:
                    forced_actions.append(action)
                    first_end_pose = [*world.position, world.heading]
                actual = packet(world, config, action, frame, collision, done)
                stored = load_packet(folder / f"packets/{world.step_count:04d}.npz")
                if actual.sha256() != recorded["packet_sha256"] or stored.sha256() != actual.sha256():
                    raise ValueError("continuation sensor bytes differ")
                runtime.observe(0, actual.action_id, None, None, None, sensor_packet=actual)
                visible |= reference_visible(evaluator.reference, frame, evaluator.truth,
                                               frame.world_from_camera, world.config.max_depth_m)
            if not runtime.states[0]["closed"] and runtime.next_local_action(0) is not None:
                raise ValueError("recorded continuation truncated")
            if runtime.sensor_episode_summary(0)["termination"] != row["termination"]:
                raise ValueError("termination differs")
            first_complete = forced_actions == option["outbound_actions"]
            target_reached = first_end_pose == option["states"][len(option["outbound_actions"])]
            if (first_complete != row["first_outbound_option_completed"]
                    or target_reached != row["first_outbound_target_reached"]
                    or first_end_pose != row["first_outbound_end_pose"]
                    or len(forced_actions) != row["forced_paid_actions"]
                    or len(expected_actions) != row["paid_actions"]
                    or world.step_count != row["total_paid_actions"]
                    or world.collisions != row["collisions"]):
                raise ValueError("outbound execution or task accounting differs")
            after = metrics(runtime, world, evaluator)
            metric_errors = []
            for when, values in (("before", before), ("after", after)):
                for key, value in values.items():
                    expected = row[when][key]
                    if value is None or expected is None:
                        if value != expected:
                            raise ValueError("missing-value metric mismatch")
                    else:
                        metric_errors.append(abs(value - expected))
            mesh = runtime.states[0]["mapper"].mesh()
            with np.load(folder / "final_mesh.npz", allow_pickle=False) as data:
                for key in data.files:
                    np.testing.assert_array_equal(data[key], np.asarray(getattr(mesh, key)))
            with np.load(folder / "visibility.npz", allow_pickle=False) as data:
                np.testing.assert_array_equal(before_visible, data["before"])
                np.testing.assert_array_equal(visible, data["after"])
            measured_area = float((visible & ~before_visible).sum()) * float(reference["sample_weight_m2"])
            if measured_area != row["new_visible_surface_m2"]:
                raise ValueError("visible surface area differs")
            # Reassociation of multiplication may differ at machine precision.
            if max(metric_errors, default=0.) > 1e-12:
                raise ValueError("physical metrics differ")
            checks.append(dict(branch=name, status="passed", regenerated_packets=world.step_count + 1,
                max_metric_absolute_error=max(metric_errors, default=0.), exact_mesh=True,
                exact_visibility=True, exact_actions_and_sensors=True,
                first_outbound_option_completed=first_complete, first_outbound_target_reached=target_reached))
            write(output / "checks.json", checks)
            print(name, "physical and metric replay passed", flush=True)
        for name, expected in frozen.items():
            if sha(ROOT / name) != expected:
                raise ValueError("source changed during replay")
        write(output / "summary.json", dict(status="passed", branches=len(checks),
            checks=checks, claim="same implementation, fresh physical sensors and state; no efficacy claim"))
        status = "passed"
    except Exception as error:
        status = "failed"
        write(output / "failure.json", dict(error=str(error), completed=len(checks)))
        raise
    finally:
        write(output / "manifest.json", dict(status=status, source=str(source), source_sha256=frozen))
        write(output / "artifact_hashes.json", {str(p.relative_to(output)): sha(p)
            for p in output.rglob("*") if p.is_file() and p.name != "artifact_hashes.json"})


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    main(args.source.resolve(), args.output.resolve())
