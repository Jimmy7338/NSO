#!/usr/bin/env python3
"""Record one real CPU four-interface development trajectory and its sources."""
import os
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
from pathlib import Path
import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import json
import shutil
import sys
from types import SimpleNamespace
import zipfile
import numpy as np
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from env.virtual3d_competition_v9 import create_competition_world
from nso.components import NSO_Components
from nso.runtime_integration import NSORuntimeIntegration
from nso.cpu_sensor_contract_v10 import GridTransform, SensorPacket, json_value
# Load transitive policy/mapper sources before freezing and constructing a world.
import nso.cpu_four_modules_v10
from nso.observed_runtime_mapper_v10 import ObservedRuntimeMapperV10


def write_json(path, value):
    Path(path).write_text(json.dumps(json_value(value), indent=2, sort_keys=True, allow_nan=False) + "\n")


def file_hash(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def map_arrays(mapper):
    keys = sorted(mapper.quality)
    return dict(belief=mapper.belief, camera_seen=mapper.camera_seen,
        quality_keys=np.asarray(keys, dtype=np.int64).reshape(-1, 3),
        quality_points=np.asarray([mapper.quality[k]["point"] for k in keys], dtype=float).reshape(-1, 3),
        quality_normals=np.asarray([mapper.quality[k]["normal"] for k in keys], dtype=float).reshape(-1, 3),
        **{"quality_" + field: np.asarray([mapper.quality[k][field] for k in keys],
             dtype=np.int64 if field in ("n", "bits", "label") else float)
           for field in ("n", "bits", "label", "information", "residual", "best_range", "normal_dispersion")})


def freeze_sources(output, protocol_path):
    paths = {Path(__file__).resolve(), protocol_path,
             ROOT / "scripts/replay_cpu_four_module_v10.py",
             ROOT / "configs/virtual3d/competition_v9_contexts.json",
             ROOT / "docs/research/FOUR_MODULE_CPU_CLOSED_LOOP_V10_CONTRACT.md"}
    for module in list(sys.modules.values()):
        name = getattr(module, "__file__", None)
        if name:
            path = Path(name).resolve()
            if path.is_relative_to(ROOT) and ".venv" not in str(path.relative_to(ROOT)) and path.suffix == ".py":
                paths.add(path)
    hashes = {str(path.relative_to(ROOT)): file_hash(path) for path in sorted(paths)}
    with zipfile.ZipFile(output / "sources.zip", "w", zipfile.ZIP_DEFLATED) as archive:
        for name in hashes:
            archive.write(ROOT / name, name)
    receipt = dict(schema_version="cpu_four_module_v10_source_freeze/1",
        frozen_utc=datetime.now(timezone.utc).isoformat(), source_sha256=hashes,
        archive_sha256=file_hash(output / "sources.zip"),
        constructed_worlds_before_this_freeze=0, scope="software development fixture")
    write_json(output / "freeze.json", receipt)
    return receipt


def run(output):
    if output.exists():
        raise FileExistsError("each source revision requires a fresh output directory")
    if shutil.disk_usage(ROOT).free < 450 * 1024**2:
        raise RuntimeError("reserve 400 MiB plus at least 50 MiB for this bounded run")
    protocol_path = ROOT / "configs/virtual3d/cpu_four_module_v10_integration_protocol.json"
    protocol = json.loads(protocol_path.read_text())
    output.mkdir(parents=True); (output / "raw").mkdir()
    freeze = freeze_sources(output, protocol_path)
    world = create_competition_world(protocol["context_id"], protocol["arrangement"])
    config = world.config; budget = protocol["total_budget"]
    args = SimpleNamespace(nso_backend="cpu_v10", eval=True, train_global=False,
        use_open_vocab_semantic=True, use_topo_graph=True, use_rpn_uq=True, use_igcr=True,
        cpu_score_mode=protocol["score_mode"], cpu_disable_feedback=not protocol["feedback_enabled"],
        cpu_max_candidates=protocol["max_candidates"], run_id=output.name)
    components = NSO_Components(args)
    components.initialize("cpu", 1, *world.shape, *world.shape)
    runtime = NSORuntimeIntegration(components, 1, world.shape)
    transform = GridTransform(tuple(world.shape), config.resolution_m)
    anchor = [*map(int, world.position), int(world.heading)]
    raw = []

    def packet(frame, action=None, collision=False, done=False):
        p = SensorPacket("Q0_shelf_west", "integration-0", f"frame-{world.step_count}",
            int(world.step_count), frame, world.scan(), tuple(map(int, world.position)), int(world.heading),
            "rendered_rgb_marker_and_planar_scan", "simulator_exact_discrete_odometry",
            action, bool(collision), bool(done)).validate(transform, config)
        frame_path = f"raw/frame_{p.action_id:04d}.npz"; scan_path = f"raw/scan_{p.action_id:04d}.npz"
        p.frame.save(output / frame_path); p.scan.save(output / scan_path)
        meta = {k: getattr(p, k) for k in p.__dataclass_fields__ if k not in ("frame", "scan")}
        raw.append(dict(**meta, frame_path=frame_path, scan_path=scan_path, packet_sha256=p.sha256(),
                        frame_sha256=file_hash(output / frame_path), scan_sha256=file_hash(output / scan_path)))
        write_json(output / "raw_manifest.json", raw)
        return p

    manifest = dict(schema_version="cpu_four_module_v10_run/1", protocol=protocol,
        context_id=protocol["context_id"], arrangement=protocol["arrangement"],
        total_budget=budget, initial_anchor=anchor, shape=world.shape, config=asdict(config),
        source_sha256=freeze["source_sha256"], archive_sha256=freeze["archive_sha256"],
        scope=protocol["scope"], raw_manifest="raw_manifest.json")
    write_json(output / "manifest.json", manifest)
    try:
        runtime.start_sensor_episode(0, config=config, transform=transform,
            packets=[packet(world.sense())], total_budget=budget, return_anchor=anchor)
        while True:
            action = runtime.next_local_action(0)
            if action is None:
                break
            frame, collision, done = world.step(action)
            observation = packet(frame, action, collision, done)
            runtime.observe(0, world.step_count, None, None, None, sensor_packet=observation)
            if world.step_count > budget:
                raise RuntimeError("paid action budget exceeded")
        summary = runtime.sensor_episode_summary(0)
        np.savez_compressed(output / "final_map.npz", **map_arrays(runtime.states[0]["mapper"]))
        mesh = runtime.states[0]["mapper"].mesh()
        np.savez_compressed(output / "final_mesh.npz", **{name: np.asarray(getattr(mesh, name))
            for name in ("vertices", "triangles", "vertex_normals", "vertex_colors")})
        calls = components._cpu_backend.calls
        choices = [c for c in calls if c["method"] == "select_topo_target"]
        authorized = [e for e in runtime.audit if e["event"] == "paid_action_authorized"]
        gates = dict(four_interfaces_called=all(any(c["module"] == m for c in calls)
                         for m in ("OV-SDF", "STGHP", "RPN-UQ", "IGCR")),
            second_global_choice_followed_by_paid_action=len(choices) >= 2 and any(
                e["selection_call_id"] == choices[1]["call_id"] for e in authorized),
            every_packet_fused_once=summary["mapper_frames"] == len(raw) == world.step_count + 1,
            within_paid_budget=world.step_count <= budget,
            returned_to_initial_pose=summary["termination"]["returned_to_anchor"],
            task_not_failed=not summary["termination"]["failed"],
            zero_collisions=world.collisions == 0)
        runtime.reset_scene(0)
        runtime.audit.append(dict(event="reset_after_close", completed_episode_count=len(runtime.completed_episodes),
            scene_state_is_none=runtime.states[0] is None,
            component_state_is_none=components._cpu_backend.scenes[0] is None,
            final_frame_id=summary["termination"]["final_frame_id"],
            final_action_id=summary["termination"]["final_action_id"]))
        summary.update(status="passed_local_checks_pending_independent_replay" if all(gates.values()) else "failed_integration_gates",
            local_gates=gates, paid_actions=world.step_count, collisions=world.collisions,
            global_choices=len(choices), raw_packets=len(raw), full_plan_performance_claim=False)
        write_json(output / "summary.json", summary)
        write_json(output / "module_calls.json", calls)
        write_json(output / "runtime_events.json", runtime.audit)
        # Source checkout must still equal the pre-world freeze at completion.
        changed = [name for name, sha in freeze["source_sha256"].items() if file_hash(ROOT / name) != sha]
        if changed:
            raise RuntimeError(f"source changed during execution: {changed}")
        write_json(output / "artifact_hashes.json", {str(p.relative_to(output)): file_hash(p)
            for p in sorted(output.rglob("*")) if p.is_file() and p.name != "artifact_hashes.json"})
        return summary
    except Exception as exc:
        write_json(output / "failure.json", dict(error_type=type(exc).__name__, reason=str(exc),
            raw_packets=len(raw), paid_actions=world.step_count, retained_for_review=True))
        write_json(output / "module_calls.json", components._cpu_backend.calls)
        write_json(output / "runtime_events.json", runtime.audit)
        raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args()
    print(json.dumps(json_value(run(arguments.output.resolve())), indent=2))
