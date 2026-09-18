#!/usr/bin/env python3
"""Collect the frozen outbound-first smoke or paired experiment, with physical replay."""
import os
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timezone
import json
import multiprocessing
from pathlib import Path
import shutil
import subprocess
import sys
import time
import zipfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
from env.virtual3d_competition_v9 import create_competition_world
from nso.continuation_v13 import describe_candidates
from nso.decision_replay_v13 import load_packet
from scripts.collect_semantic_gain_v13_history import sha, write
from scripts.collect_semantic_gain_v13_continuations import restore, evaluate
from scripts.execute_semantic_gain_v13_2_continuations import execute
from utils.reconstruction_metrics import ReconstructionEvaluator


def verify(folder):
    hashes = json.loads((folder / "artifact_hashes.json").read_text())
    for name, expected in hashes.items():
        if sha(folder / name) != expected:
            raise ValueError("Artifact changed: " + str(folder / name))
    return hashes


def seal(folder):
    write(folder / "artifact_hashes.json", {str(p.relative_to(folder)): sha(p)
        for p in sorted(folder.rglob("*")) if p.is_file() and p != folder / "artifact_hashes.json"})


def safe(rows):
    return bool(rows) and all(r["collisions"] == 0 and not r["termination"]["failed"]
        and r["termination"]["returned_to_anchor"] and r["first_outbound_option_completed"]
        and r["first_outbound_target_reached"] and r["total_paid_actions"] <= r["total_budget"] for r in rows)


def collect_context(parent, arrangement, protocol, output, frozen, mode, smoke_source):
    history = ROOT / protocol["histories"] / (parent + "_" + arrangement)
    history_hashes = verify(history)
    history_manifest = json.loads((history / "manifest.json").read_text())
    for name, expected in history_manifest["source_sha256"].items():
        if sha(ROOT / name) != expected:
            raise ValueError("Historical runtime changed: " + name)
    config = history_manifest["protocol"]
    stored = [load_packet(p) for p in sorted((history / "packets").glob("*.npz"))]
    checkpoints = {r["action_id"]: r for r in json.loads((history / "checkpoints.json").read_text())}
    folder = output / (parent + "_" + arrangement)
    folder.mkdir()
    cp = dict(history=str(history.relative_to(ROOT)), total_budget=protocol["total_budget"],
        reference_sample_count=16000, reference_seed=2026, thresholds_m=[.05, .1],
        first_option_execution="complete_outbound_then_common_geometry_continuation")
    cm = dict(status="running", protocol=cp, source_sha256=frozen,
        history_artifact_hashes=history_hashes, training_allowed=False)
    write(folder / "manifest.json", cm)
    rows, reused = [], []
    try:
        world = create_competition_world(parent, arrangement)
        evaluator = ReconstructionEvaluator(world, count=16000, seed=2026)
        weight = float(world.mesh.get_surface_area()) / 16000
        reference = dict(points=evaluator.reference, classes=evaluator.classes,
            sample_weight_m2=np.asarray(weight), vertices=np.asarray(world.mesh.vertices),
            triangles=np.asarray(world.mesh.triangles))
        np.savez_compressed(folder / "reference.npz", **reference)
        for declaration in [p for p in protocol["eligible_pairs"] if p["parent"] == parent]:
            checkpoint = checkpoints[declaration["action_id"]]
            candidates = sorted(checkpoint["candidates"], key=lambda c: c["candidate_id"])
            if len(candidates) != declaration["candidates_per_arrangement"]:
                raise ValueError("Candidate universe changed")
            _, _, runtime = restore(config, stored, checkpoint)
            before = evaluate(runtime, world, evaluator, cp["thresholds_m"])
            mesh = runtime.states[0]["mapper"].mesh()
            np.savez_compressed(folder / f"prefix_step_{checkpoint['action_id']}.npz",
                vertices=np.asarray(mesh.vertices), triangles=np.asarray(mesh.triangles),
                vertex_colors=np.asarray(mesh.vertex_colors))
            features = describe_candidates(runtime, candidates)
            feature_path = folder / f"features_step_{checkpoint['action_id']}.json"
            write(feature_path, features)
            if mode == "smoke":
                candidates = [candidates[0], candidates[-1]]
            for candidate in candidates:
                name = f"step_{checkpoint['action_id']:03d}_candidate_{candidate['candidate_id']:03d}"
                previous = None if smoke_source is None else smoke_source / folder.name / name
                if previous is not None and previous.exists():
                    previous_context = previous.parent
                    previous_manifest = json.loads((previous_context / "manifest.json").read_text())
                    if previous_manifest["protocol"] != cp or previous_manifest["source_sha256"] != frozen:
                        raise ValueError("Smoke reuse protocol or source differs")
                    if sha(previous_context / feature_path.name) != sha(feature_path):
                        raise ValueError("Smoke reuse observed descriptors differ")
                    with np.load(previous_context / "reference.npz", allow_pickle=False) as data:
                        for key in reference:
                            np.testing.assert_array_equal(data[key], reference[key])
                    row = json.loads((previous / "result.json").read_text())
                    if (row["state_sha256"] != checkpoint["state"]["sha256"]
                        or row["candidate_pool_sha256"] != checkpoint["candidate_sha256"]
                        or row["candidate_id"] != candidate["candidate_id"]
                        or row["before"] != before or not safe([row])):
                        raise ValueError("Smoke reuse state, pool or outcome contract differs")
                    shutil.copytree(previous, folder / name, copy_function=os.link)
                    reused.append(name)
                else:
                    while shutil.disk_usage(output).free < 512 * 1024**2:
                        print("WAITING_FOR_DISK_SPACE", folder.name, name, flush=True)
                        time.sleep(20)
                    row = execute(config, cp, stored, checkpoint, candidate, evaluator, weight, folder / name)
                rows.append(row)
                write(folder / "partial.json", rows)
                print(folder.name, name, "joint", round(row["joint_gain"], 8), "safe", safe([row]), flush=True)
        write(folder / "reuse.json", reused)
        cm["status"] = "complete"
    except Exception as error:
        cm.update(status="failed", error=str(error), completed_outcomes=len(rows))
        raise
    finally:
        write(folder / "manifest.json", cm)
        seal(folder)
    # A separate process regenerates the physical sensors and executes the same runtime.
    subprocess.run([sys.executable, str(ROOT / "scripts/replay_semantic_gain_v13_2_continuations.py"),
        "--source", str(folder), "--output", str(output / (folder.name + "_replay"))], check=True)
    replay = json.loads((output / (folder.name + "_replay") / "summary.json").read_text())
    if replay["status"] != "passed" or replay["branches"] != len(rows):
        raise ValueError("Incomplete physical replay")
    print("CONTEXT_REPLAY_COMPLETE", folder.name, flush=True)
    return dict(context=folder.name, rows=rows, reused=len(reused))


def main(output, mode, smoke_source, workers):
    protocol_path = ROOT / "configs/virtual3d/semantic_gain_v13_2_outbound_information_protocol.json"
    protocol = json.loads(protocol_path.read_text())
    if sha(ROOT / protocol["source_information_protocol"]) != protocol["source_information_protocol_sha256"]:
        raise ValueError("Original comparison protocol changed")
    verify(ROOT / protocol["histories"])
    pairing_path = ROOT / protocol["pairing_audit"]
    verify(pairing_path.parent)
    pairing = json.loads(pairing_path.read_text())
    for p in protocol["eligible_pairs"]:
        matches = [r for r in pairing["pairs"] if (r["parent"], r["action_id"]) == (p["parent"], p["action_id"])]
        if len(matches) != 1 or not matches[0]["matched"] or not matches[0]["semantic_history_changed"]:
            raise ValueError("Geometry-matching certificate invalid")
    files = {protocol_path, pairing_path, Path(__file__).resolve(),
        *[p for d in ("nso", "env", "utils") for p in (ROOT / d).rglob("*.py")],
        *[ROOT / "scripts" / name for name in (
            "collect_semantic_gain_v13_history.py", "collect_semantic_gain_v13_history_v13_1.py",
            "collect_semantic_gain_v13_continuations.py", "execute_semantic_gain_v13_2_continuations.py",
            "replay_semantic_gain_v13_2_continuations.py", "analyze_semantic_gain_v13_paired_information.py")],
        ROOT / "configs/virtual3d/competition_v9_contexts.json"}
    frozen = {str(p.relative_to(ROOT)): sha(p) for p in sorted(files)}
    smoke_hashes = None
    if mode == "paired":
        if smoke_source is None:
            raise ValueError("Paired collection requires the passed four-branch smoke")
        smoke_hashes = verify(smoke_source)
        sm = json.loads((smoke_source / "manifest.json").read_text())
        ss = json.loads((smoke_source / "summary.json").read_text())
        if (sm["status"] != "complete" or sm["mode"] != "smoke" or sm["source_sha256"] != frozen
            or sm["protocol"] != protocol or ss["candidate_outcomes"] != 4 or not ss["physical_screen_passed"]):
            raise ValueError("Same-source four-branch smoke has not passed")
    elif smoke_source is not None:
        raise ValueError("Smoke cannot reuse future outcomes")
    output.mkdir(parents=True, exist_ok=False)
    with zipfile.ZipFile(output / "source_snapshot.zip", "x", zipfile.ZIP_DEFLATED) as archive:
        for path in sorted(files):
            archive.write(path, path.relative_to(ROOT))
    manifest = dict(status="running", created_utc=datetime.now(timezone.utc).isoformat(), mode=mode,
        protocol=protocol, source_sha256=frozen, workers=workers, training_allowed=False, contexts=[],
        smoke_source=None if smoke_source is None else str(smoke_source), smoke_artifact_hashes=smoke_hashes)
    write(output / "manifest.json", manifest)
    cases = [("Q0", "shelf_east")] if mode == "smoke" else [
        (p, a) for p in protocol["parents"] for a in protocol["arrangements"]]
    rows, errors, reused = [], [], 0
    try:
        with ProcessPoolExecutor(max_workers=workers, mp_context=multiprocessing.get_context("spawn")) as pool:
            futures = {pool.submit(collect_context, p, a, protocol, output, frozen, mode, smoke_source):
                       (p, a) for p, a in cases}
            for future in as_completed(futures):
                try:
                    result = future.result()
                    rows.extend(result["rows"]); reused += result["reused"]
                    manifest["contexts"].append(result["context"])
                    rows.sort(key=lambda r: (r["parent"], r["arrangement"], r["checkpoint_action"], r["candidate_id"]))
                    write(output / "partial.json", rows)
                    write(output / "manifest.json", manifest)
                except Exception as error:
                    errors.append(dict(context=futures[future], error=str(error)))
                    write(output / "worker_failures.json", errors)
        if errors:
            raise ValueError("One or more context workers failed; partial evidence retained")
        expected = 4 if mode == "smoke" else protocol["expected_candidate_outcomes"]
        if len(rows) != expected:
            raise ValueError("Incomplete outcome universe")
        for name, expected_hash in frozen.items():
            if sha(ROOT / name) != expected_hash:
                raise ValueError("Frozen source changed")
        summary = dict(status="complete", candidate_outcomes=len(rows), reused=reused,
            newly_executed=len(rows) - reused, replayed=len(rows), collisions=sum(r["collisions"] for r in rows),
            failures=sum(r["termination"]["failed"] for r in rows),
            returns=sum(r["termination"]["returned_to_anchor"] for r in rows),
            first_options_complete=sum(r["first_outbound_option_completed"] for r in rows),
            first_targets_reached=sum(r["first_outbound_target_reached"] for r in rows),
            budget_violations=sum(r["total_paid_actions"] > r["total_budget"] for r in rows),
            physical_screen_passed=safe(rows), training_allowed=False)
        write(output / "summary.json", summary)
        manifest["status"] = "complete"
        print(json.dumps(summary, indent=2), flush=True)
    except Exception as error:
        manifest.update(status="failed", error=str(error), completed_outcomes=len(rows))
        raise
    finally:
        write(output / "manifest.json", manifest)
        seal(output)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--mode", choices=("smoke", "paired"), required=True)
    parser.add_argument("--smoke-source", type=Path)
    parser.add_argument("--workers", type=int, choices=(1, 2), default=1)
    args = parser.parse_args()
    main(args.output.resolve(), args.mode,
         None if args.smoke_source is None else args.smoke_source.resolve(), args.workers)
