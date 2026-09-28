#!/usr/bin/env python3
"""Execute every sealed V13 history candidate with common-budget continuation."""
import os
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
from time import perf_counter
import zipfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
import open3d as o3d
from env.virtual3d_competition_v9 import create_competition_world
from nso.continuation_v13 import describe_candidates, force_full_candidate
from nso.cpu_sensor_contract_v10 import GridTransform
from nso.decision_replay_v13 import decision_state, load_packet, replay_history, save_packet
from scripts.collect_semantic_gain_v13_history import packet, sha, start, write
from utils.reconstruction_metrics import ReconstructionEvaluator
from utils.counterfactual_surface_visibility import reference_visible


def restore(config, stored, checkpoint):
    world = create_competition_world(config["parent"], config["arrangement"])
    first = packet(world, config, None)
    if first.sha256() != stored[0].sha256():
        raise ValueError("regenerated initial sensor bytes differ")
    for expected in stored[1:checkpoint["action_id"] + 1]:
        frame, collision, done = world.step(expected.action)
        actual = packet(world, config, expected.action, frame, collision, done)
        if actual.sha256() != expected.sha256():
            raise ValueError(f"physical prefix differs at {expected.action_id}")
    transform = GridTransform(tuple(world.shape), world.config.resolution_m)
    runtime = start(config, world, transform, stored[0])
    replay_history(runtime, stored[:checkpoint["action_id"] + 1])
    if decision_state(runtime)["sha256"] != checkpoint["state"]["sha256"]:
        raise ValueError("decision map/feedback/history differ")
    return world, transform, runtime


def evaluate(runtime, world, evaluator, thresholds):
    mapper = runtime.states[0]["mapper"]
    cov = float(np.count_nonzero((mapper.belief != -1) & world.reachable) / world.reachable.sum())
    result = evaluator.evaluate(mapper.mesh(), cov, thresholds=thresholds)
    area = float(world.reachable.sum()) * world.config.resolution_m ** 2
    result.update(coverage_2d=cov, covered_area_m2=cov * area,
                  area_times_f1_05cm=cov * area * result["f1_05cm"])
    for tag in ("05cm", "10cm"):
        result[f"storage_shelf_recall_{tag}"] = result.pop(f"simple_recall_{tag}")
        result[f"equipment_cabinet_recall_{tag}"] = result.pop(f"complex_recall_{tag}")
    return result


def execute(config, protocol, stored, checkpoint, candidate, evaluator, weight, folder):
    folder.mkdir(); (folder / "packets").mkdir()
    world, transform, runtime = restore(config, stored, checkpoint)
    before = evaluate(runtime, world, evaluator, protocol["thresholds_m"])
    before_visible = np.zeros(len(evaluator.reference), bool)
    for p in stored[:checkpoint["action_id"] + 1]:
        before_visible |= reference_visible(evaluator.reference, p.frame, evaluator.truth,
                                            p.frame.world_from_camera, world.config.max_depth_m)
    visible = before_visible.copy()
    option = force_full_candidate(runtime, candidate["candidate_id"], checkpoint["candidate_sha256"])
    actions, forced_observed = [], []
    started = perf_counter()
    while not runtime.states[0]["closed"]:
        action = runtime.next_local_action(0)
        if action is None:
            break
        active = runtime.states[0]["option"]
        forced = active is not None and active["option_id"] == option["option_id"]
        frame, collision, done = world.step(action)
        observed = packet(world, config, action, frame, collision, done).validate(transform, world.config)
        save_packet(folder / f"packets/{world.step_count:04d}.npz", observed)
        runtime.observe(0, world.step_count, None, None, None, sensor_packet=observed)
        visible |= reference_visible(evaluator.reference, frame, evaluator.truth,
                                      frame.world_from_camera, world.config.max_depth_m)
        actions.append(dict(action_id=world.step_count, action=action, forced_option=forced,
            pose=[*world.position, world.heading], collision=bool(collision),
            packet_sha256=observed.sha256()))
        if forced:
            forced_observed.append(action)
        write(folder / "actions.json", actions)
    elapsed = perf_counter() - started
    summary = runtime.sensor_episode_summary(0)
    after = evaluate(runtime, world, evaluator, protocol["thresholds_m"])
    mesh = runtime.states[0]["mapper"].mesh()
    np.savez_compressed(folder / "final_mesh.npz", vertices=np.asarray(mesh.vertices),
        triangles=np.asarray(mesh.triangles), vertex_colors=np.asarray(mesh.vertex_colors))
    np.savez_compressed(folder / "visibility.npz", before=before_visible, after=visible)
    result = dict(parent=config["parent"], arrangement=config["arrangement"],
        checkpoint_action=checkpoint["action_id"], candidate_id=candidate["candidate_id"],
        group=candidate["group"], state_sha256=checkpoint["state"]["sha256"],
        candidate_pool_sha256=checkpoint["candidate_sha256"], total_budget=config["total_budget"],
        available_continuation_budget=config["total_budget"] - checkpoint["action_id"],
        paid_actions=len(actions), total_paid_actions=world.step_count,
        first_full_option_completed=forced_observed == candidate["actions"],
        forced_paid_actions=len(forced_observed), collisions=world.collisions,
        termination=summary["termination"], before=before, after=after,
        joint_gain=after["joint_05cm"] - before["joint_05cm"],
        area_joint_gain=after["area_times_f1_05cm"] - before["area_times_f1_05cm"],
        new_visible_surface_m2=float(np.count_nonzero(visible & ~before_visible)) * weight,
        path_distance_m=sum(row["action"] == "forward" for row in actions) * world.config.resolution_m,
        action_time_s=len(actions) * world.config.action_duration_s,
        continuation_wall_time_s=elapsed, physical_prefix_exact=True,
        modules_called=sorted({c["module"] for c in runtime.components._cpu_backend.calls}))
    write(folder / "result.json", result)
    write(folder / "runtime_audit.json", runtime.audit)
    write(folder / "modules.json", runtime.components._cpu_backend.calls)
    return result


def main(output, protocol_path):
    protocol = json.loads(protocol_path.read_text())
    history = ROOT / protocol["history"]
    history_manifest = json.loads((history / "manifest.json").read_text())
    config = history_manifest["protocol"]
    if config["parent"] not in ("Q0", "Q1", "Q2", "Q3") or protocol["total_budget"] != config["total_budget"]:
        raise ValueError("invalid development parent or budget")
    for name, expected in history_manifest["source_sha256"].items():
        if sha(ROOT / name) != expected:
            raise ValueError(f"historical source mismatch: {name}")
    history_hashes = json.loads((history / "artifact_hashes.json").read_text())
    for name, expected in history_hashes.items():
        if sha(history / name) != expected:
            raise ValueError(f"historical artifact mismatch: {name}")
    output.mkdir(parents=True, exist_ok=False)
    sources = {ROOT / name for name in history_manifest["source_sha256"]}
    sources.update((Path(__file__).resolve(), protocol_path, ROOT / "nso/continuation_v13.py"))
    frozen = {str(p.relative_to(ROOT)): sha(p) for p in sorted(sources)}
    with zipfile.ZipFile(output / "source_snapshot.zip", "x", zipfile.ZIP_DEFLATED) as archive:
        for path in sorted(sources):
            archive.write(path, path.relative_to(ROOT))
    manifest = dict(schema_version="semantic_gain_v13_continuation_execution/1", status="running",
        created_utc=datetime.now(timezone.utc).isoformat(), protocol=protocol,
        source_sha256=frozen, history_artifact_hashes=history_hashes, independent_confirmation=False,
        training_allowed=False, worlds_before_freeze=0)
    write(output / "manifest.json", manifest)
    try:
        stored = [load_packet(p) for p in sorted((history / "packets").glob("*.npz"))]
        checkpoints = json.loads((history / "checkpoints.json").read_text())
        world = create_competition_world(config["parent"], config["arrangement"])
        evaluator = ReconstructionEvaluator(world, count=protocol["reference_sample_count"], seed=protocol["reference_seed"])
        weight = float(world.mesh.get_surface_area()) / protocol["reference_sample_count"]
        np.savez_compressed(output / "reference.npz", points=evaluator.reference,
            classes=evaluator.classes, sample_weight_m2=weight, vertices=np.asarray(world.mesh.vertices),
            triangles=np.asarray(world.mesh.triangles))
        rows = []
        for checkpoint in checkpoints:
            _, _, runtime = restore(config, stored, checkpoint)
            features = describe_candidates(runtime, checkpoint["candidates"])
            write(output / f"features_step_{checkpoint['action_id']}.json", features)
            # Persist features before any corresponding candidate outcome is executed.
            for candidate in checkpoint["candidates"]:
                name = f"step_{checkpoint['action_id']:03d}_candidate_{candidate['candidate_id']:03d}"
                row = execute(config, protocol, stored, checkpoint, candidate, evaluator, weight, output / name)
                rows.append(row); write(output / "partial.json", rows)
                print(name, row["joint_gain"], row["paid_actions"], row["termination"], flush=True)
        changed = [name for name, expected in frozen.items() if sha(ROOT / name) != expected]
        if changed:
            raise ValueError(f"source mutation: {changed}")
        summary = dict(status="complete", branches=len(rows), parents=1,
            checkpoints=len(checkpoints), failures=sum(r["termination"]["failed"] for r in rows),
            collisions=sum(r["collisions"] for r in rows),
            first_options_completed=sum(r["first_full_option_completed"] for r in rows),
            returned=sum(r["termination"]["returned_to_anchor"] for r in rows),
            budget_violations=sum(r["total_paid_actions"] > r["total_budget"] for r in rows),
            semantic_information_value_tested=False, training_allowed=False,
            per_checkpoint=[dict(action_id=c["action_id"], candidate_count=len(c["candidates"]),
                min_joint_gain=min(r["joint_gain"] for r in rows if r["checkpoint_action"] == c["action_id"]),
                max_joint_gain=max(r["joint_gain"] for r in rows if r["checkpoint_action"] == c["action_id"]))
                for c in checkpoints])
        write(output / "summary.json", summary)
        manifest["status"] = "complete"
        print(json.dumps(summary, indent=2), flush=True)
    except Exception as error:
        manifest.update(status="failed", error=str(error))
        raise
    finally:
        write(output / "manifest.json", manifest)
        write(output / "artifact_hashes.json", {str(p.relative_to(output)): sha(p)
            for p in sorted(output.rglob("*")) if p.is_file() and p.name != "artifact_hashes.json"})


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--protocol", type=Path,
        default=ROOT / "configs/virtual3d/semantic_gain_v13_continuation_smoke.json")
    args = parser.parse_args()
    main(args.output.resolve(), args.protocol.resolve())
