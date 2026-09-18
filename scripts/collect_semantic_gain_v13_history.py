#!/usr/bin/env python3
"""Collect a cold-start geometry trajectory and verify isolated decision restores."""
import os
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import platform
import sys
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import open3d as o3d

from env.virtual3d_competition_v9 import create_competition_world
from nso.components import NSO_Components
from nso.cpu_sensor_contract_v10 import GridTransform, SensorPacket, digest, json_value
from nso.decision_replay_v13 import decision_state, load_packet, replay_history, save_packet
from nso.runtime_integration import NSORuntimeIntegration


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write(path, value):
    path.write_text(json.dumps(json_value(value), ensure_ascii=False, sort_keys=True,
                               indent=2, allow_nan=False) + "\n")


def start(config, world, transform, first):
    args = SimpleNamespace(nso_backend="cpu_v10", eval=True, train_global=False,
        use_open_vocab_semantic=True, use_topo_graph=True, use_rpn_uq=True, use_igcr=True,
        cpu_score_mode=config["score_mode"], cpu_disable_feedback=False,
        cpu_max_candidates=config["candidate_cap"], cpu_coverage_slots=config["coverage_slots"],
        cpu_planner_revision=config["planner_revision"], cpu_measured_novelty_floor=.25,
        run_id="v13-history-smoke")
    components = NSO_Components(args)
    components.initialize("cpu", 1, *world.shape, *world.shape)
    runtime = NSORuntimeIntegration(components, 1, world.shape)
    runtime.start_sensor_episode(0, config=world.config, transform=transform, packets=[first],
        total_budget=config["total_budget"], return_anchor=(*first.position, first.heading))
    return runtime


def packet(world, config, action, frame=None, collision=False, done=False):
    return SensorPacket(config["parent"], "v13-history-smoke", f"frame-{world.step_count}",
        int(world.step_count), world.sense() if frame is None else frame, world.scan(),
        tuple(map(int, world.position)), int(world.heading),
        "simulator_rgbd_and_scan", "simulator_exact_discrete_odometry", action,
        bool(collision), bool(done))


def main(output, protocol):
    config = json.loads(protocol.read_text())
    if config["parent"] not in ("Q0", "Q1", "Q2", "Q3"):
        raise ValueError("only existing development parents allowed")
    output.mkdir(parents=True, exist_ok=False)
    (output / "packets").mkdir()
    sources = sorted({protocol, Path(__file__).resolve(), *[
        path for directory in ("nso", "env", "utils")
        for path in (ROOT / directory).rglob("*.py")],
        *[(ROOT / "configs/virtual3d") / name for name in
          ("competition_v9_contexts.json",) if ((ROOT / "configs/virtual3d") / name).exists()]})
    frozen = {str(path.relative_to(ROOT)): sha(path) for path in sources}
    manifest = dict(status="running", schema_version="semantic_gain_v13_history/1",
        created_utc=datetime.now(timezone.utc).isoformat(), protocol=config,
        source_sha256=frozen, environment=dict(python=platform.python_version(),
        numpy=np.__version__, open3d=o3d.__version__), worlds_before_freeze=0,
        independent_confirmation=False, efficacy_test=False)
    write(output / "manifest.json", manifest)
    try:
        world = create_competition_world(config["parent"], config["arrangement"])
        transform = GridTransform(tuple(world.shape), world.config.resolution_m)
        first = packet(world, config, None).validate(transform, world.config)
        save_packet(output / "packets/0000.npz", first)
        runtime = start(config, world, transform, first)
        checkpoints, actions = [], []
        ordinal = 0
        while not runtime.states[0]["closed"]:
            state = runtime.states[0]
            checkpoint = None
            if not state["active_actions"] and state["phase"] != "return":
                ordinal += 1
                if ordinal in config["checkpoint_decision_ordinals"]:
                    checkpoint = dict(decision_ordinal=ordinal, action_id=state["packet"].action_id,
                                      state=decision_state(runtime))
            action = runtime.next_local_action(0)
            if checkpoint is not None:
                selection = runtime.components._cpu_backend.scenes[0].get("last_selection", {})
                checkpoint.update(candidates=selection.get("candidates", []), next_action=action)
                checkpoint["candidate_sha256"] = digest(checkpoint["candidates"])
                checkpoints.append(checkpoint)
                write(output / "checkpoints.json", checkpoints)
                print(f"decision {ordinal}, paid {checkpoint['action_id']}, candidates {len(checkpoint['candidates'])}", flush=True)
            if action is None:
                break
            frame, collision, done = world.step(action)
            observed = packet(world, config, action, frame, collision, done).validate(transform, world.config)
            save_packet(output / f"packets/{world.step_count:04d}.npz", observed)
            runtime.observe(0, world.step_count, None, None, None, sensor_packet=observed)
            actions.append(dict(action=action, packet_sha256=observed.sha256()))
            write(output / "actions.json", actions)
        terminal = runtime.sensor_episode_summary(0)
        write(output / "terminal.json", terminal)
        stored = [load_packet(path) for path in sorted((output / "packets").glob("*.npz"))]
        expected_hashes = [first.sha256(), *[row["packet_sha256"] for row in actions]]
        if [p.sha256() for p in stored] != expected_hashes:
            raise RuntimeError("stored sensor bytes changed")
        checks = []
        for checkpoint in checkpoints:
            restored = start(config, world, transform, stored[0])
            replay_history(restored, stored[:checkpoint["action_id"] + 1])
            actual = decision_state(restored)
            equal = actual["sha256"] == checkpoint["state"]["sha256"]
            action = restored.next_local_action(0)
            selection = restored.components._cpu_backend.scenes[0].get("last_selection", {})
            row = dict(action_id=checkpoint["action_id"], full_state_equal=equal,
                candidates_equal=digest(selection.get("candidates", [])) == checkpoint["candidate_sha256"],
                next_action_equal=action == checkpoint["next_action"])
            checks.append(row)
            write(output / "restore_checks.json", checks)
            if not all(row[key] for key in ("full_state_equal", "candidates_equal", "next_action_equal")):
                write(output / f"restore_mismatch_{checkpoint['action_id']}.json", actual)
                raise RuntimeError("decision restore mismatch; evidence retained")
            print(f"restored action {checkpoint['action_id']}: exact state, candidates and action", flush=True)
        changed = [name for name, expected in frozen.items() if sha(ROOT / name) != expected]
        if changed:
            raise RuntimeError(f"source changed during run: {changed}")
        passed = (len(checks) >= 2 and not terminal["termination"]["failed"]
                  and world.collisions == 0)
        result = dict(status="passed" if passed else "insufficient_or_failed",
            paid_actions=len(actions), translation_actions=sum(r["action"] == "forward" for r in actions),
            decision_count=ordinal, checkpoints_checked=len(checks), collisions=world.collisions,
            termination=terminal["termination"], exact_disk_packet_roundtrips=len(stored),
            modules_called=sorted({c["module"] for c in runtime.components._cpu_backend.calls}),
            semantic_advantage_verified=False, complete_v13_algorithm=False,
            limitation="Stored-packet policy-state replay, not independent physical sensor regeneration or efficacy.")
        write(output / "result.json", result)
        manifest["status"] = result["status"]
        print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)
    except Exception as error:
        manifest.update(status="failed", error=str(error))
        raise
    finally:
        write(output / "manifest.json", manifest)
        write(output / "artifact_hashes.json", {str(p.relative_to(output)): sha(p)
            for p in sorted(output.rglob("*")) if p.is_file() and p.name != "artifact_hashes.json"})


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--protocol", type=Path,
        default=ROOT / "configs/virtual3d/semantic_gain_v13_history_smoke.json")
    args = parser.parse_args()
    main(args.output.resolve(), args.protocol.resolve())
