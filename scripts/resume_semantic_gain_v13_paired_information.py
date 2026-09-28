#!/usr/bin/env python3
"""Resume the identical paired-information protocol after a storage interruption."""
import os
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import shutil
import sys
import time
import zipfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
from env.virtual3d_competition_v9 import create_competition_world
from nso.continuation_v13 import describe_candidates
from nso.cpu_sensor_contract_v10 import digest
from nso.decision_replay_v13 import load_packet
from scripts.collect_semantic_gain_v13_history import sha, write
from scripts.collect_semantic_gain_v13_continuations import execute as execute_new, restore, evaluate
from scripts.replay_semantic_gain_v13_continuations import main as replay
from utils.reconstruction_metrics import ReconstructionEvaluator


def verify_artifacts(folder):
    hashes = json.loads((folder / "artifact_hashes.json").read_text())
    for name, expected in hashes.items():
        if sha(folder / name) != expected:
            raise ValueError(f"artifact mismatch: {folder / name}")
    return hashes


def seal(folder):
    write(folder / "artifact_hashes.json", {str(p.relative_to(folder)):sha(p)
        for p in sorted(folder.rglob("*")) if p.is_file() and p != folder / "artifact_hashes.json"})


def reuse_verified(old, history, checkpoint, candidate, reference, before):
    """Fail closed unless every declared identity needed for reuse agrees."""
    old_manifest = json.loads((old / "manifest.json").read_text())
    old_history = ROOT / old_manifest["protocol"]["history"]
    previous = next(r for r in json.loads((old_history / "checkpoints.json").read_text())
                    if r["action_id"] == checkpoint["action_id"])
    if previous["state"]["sha256"] != checkpoint["state"]["sha256"]:
        raise ValueError("reuse decision-state mismatch")
    if previous["candidate_sha256"] != checkpoint["candidate_sha256"]:
        raise ValueError("reuse candidate pool mismatch")
    prior_candidate = next(c for c in previous["candidates"] if c["candidate_id"] == candidate["candidate_id"])
    if digest(candidate) != digest(prior_candidate):
        raise ValueError("reuse full candidate route mismatch")
    for index in range(checkpoint["action_id"] + 1):
        a = load_packet(history / f"packets/{index:04d}.npz")
        b = load_packet(old_history / f"packets/{index:04d}.npz")
        if a.sha256() != b.sha256():
            raise ValueError("reuse physical prefix mismatch")
    for name, expected in old_manifest["source_sha256"].items():
        if sha(ROOT / name) != expected:
            raise ValueError("reuse runtime/source mismatch")
    with np.load(old / "reference.npz", allow_pickle=False) as data:
        if set(data.files) != set(reference):
            raise ValueError("reuse reference schema mismatch")
        for key in data.files:
            np.testing.assert_array_equal(data[key], reference[key])
    name = f"step_{checkpoint['action_id']:03d}_candidate_{candidate['candidate_id']:03d}"
    row = json.loads((old / name / "result.json").read_text())
    if row["before"] != before or row["total_budget"] != 96 or not row["first_full_option_completed"]:
        raise ValueError("reuse metric, budget or execution mismatch")
    if old_manifest["protocol"]["first_option_execution"] != "complete_roundtrip_then_common_geometry_continuation":
        raise ValueError("reuse continuation definition mismatch")
    audit = ROOT / "audit_results/semantic_gain_v13_continuations_Q0_east_replay_20260914"
    verify_artifacts(audit)
    summary = json.loads((audit / "summary.json").read_text())
    if summary["status"] != "passed" or not any(r["branch"] == name and r["status"] == "passed" for r in summary["checks"]):
        raise ValueError("reuse requires successful physical replay")
    return name, row


RECOVERY_SOURCE = None
RECOVERED_NEW = []


def execute(config, protocol, stored, checkpoint, candidate, evaluator, weight, folder):
    previous = RECOVERY_SOURCE / folder.parent.name / folder.name
    if not (previous / "result.json").is_file():
        while shutil.disk_usage(ROOT).free < 512 * 1024 ** 2:
            print("WAITING_FOR_DISK_SPACE", folder.parent.name, folder.name, flush=True)
            time.sleep(20)
        return execute_new(config, protocol, stored, checkpoint, candidate, evaluator, weight, folder)
    row = json.loads((previous / "result.json").read_text())
    expected = dict(parent=config["parent"], arrangement=config["arrangement"],
        checkpoint_action=checkpoint["action_id"], candidate_id=candidate["candidate_id"],
        state_sha256=checkpoint["state"]["sha256"], candidate_pool_sha256=checkpoint["candidate_sha256"],
        total_budget=config["total_budget"])
    if any(row[key] != value for key,value in expected.items()):
        raise ValueError("completed recovery branch identity mismatch")
    if not row["physical_prefix_exact"] or not row["first_full_option_completed"]:
        raise ValueError("recovery branch lacks physical prefix or complete first option")
    actions=json.loads((previous/"actions.json").read_text())
    if len(actions)!=row["paid_actions"]:
        raise ValueError("recovery action count mismatch")
    for action in actions:
        saved=load_packet(previous/f"packets/{action['action_id']:04d}.npz")
        if saved.sha256()!=action["packet_sha256"]:
            raise ValueError("recovery packet identity mismatch")
    shutil.copytree(previous,folder,copy_function=os.link)
    RECOVERED_NEW.append(dict(source=str(previous),target=str(folder),physical_replay_required=True))
    return row


def main(output, recovery_source):
    global RECOVERY_SOURCE
    RECOVERY_SOURCE = recovery_source
    verify_artifacts(recovery_source)
    previous_manifest=json.loads((recovery_source/"manifest.json").read_text())
    if previous_manifest["status"]!="failed" or "No space left" not in previous_manifest.get("error",""):
        raise ValueError("this recovery entrypoint requires a verified storage-interrupted run")
    for name,expected in previous_manifest["source_sha256"].items():
        if sha(ROOT/name)!=expected:
            raise ValueError("recovery must use identical frozen collection and runtime sources")
    protocol_path = ROOT / "configs/virtual3d/semantic_gain_v13_paired_information_protocol.json"
    protocol = json.loads(protocol_path.read_text())
    if protocol != previous_manifest["protocol"]:
        raise ValueError("recovery protocol changed")
    histories = ROOT / protocol["histories"]
    verify_artifacts(histories)
    pairing_path = ROOT / protocol["pairing_audit"]
    verify_artifacts(pairing_path.parent)
    pairing = json.loads(pairing_path.read_text())
    for p in protocol["eligible_pairs"]:
        matches = [r for r in pairing["pairs"] if (r["parent"],r["action_id"]) == (p["parent"],p["action_id"])]
        if len(matches) != 1 or not matches[0]["matched"] or not matches[0]["semantic_history_changed"]:
            raise ValueError("predeclared geometry/semantic matching certificate missing")
    old = ROOT / protocol["existing_evidence"]["source"]
    verify_artifacts(old)
    output.mkdir(parents=True, exist_ok=False)
    files = {protocol_path, pairing_path, Path(__file__).resolve(),
        *[p for d in ("nso", "env", "utils") for p in (ROOT / d).rglob("*.py")],
        *[ROOT / "scripts" / name for name in (
            "collect_semantic_gain_v13_history.py", "collect_semantic_gain_v13_history_v13_1.py",
            "collect_semantic_gain_v13_continuations.py", "replay_semantic_gain_v13_continuations.py",
            "analyze_semantic_gain_v13_paired_information.py")],
        ROOT / "configs/virtual3d/competition_v9_contexts.json"}
    frozen = {str(p.relative_to(ROOT)): sha(p) for p in sorted(files)}
    with zipfile.ZipFile(output / "source_snapshot.zip", "x", zipfile.ZIP_DEFLATED) as archive:
        for path in sorted(files): archive.write(path, path.relative_to(ROOT))
    manifest = dict(status="running", created_utc=datetime.now(timezone.utc).isoformat(),
        protocol=protocol, source_sha256=frozen, training_allowed=False, contexts=[])
    write(output / "manifest.json", manifest)
    all_rows, reuse_checks = [], []
    try:
        for parent in protocol["parents"]:
            selected_pairs = [p for p in protocol["eligible_pairs"] if p["parent"] == parent]
            for arrangement in protocol["arrangements"]:
                history = histories / (parent + "_" + arrangement)
                history_hashes = verify_artifacts(history)
                hm = json.loads((history / "manifest.json").read_text())
                for key, expected in hm["source_sha256"].items():
                    if sha(ROOT / key) != expected: raise ValueError("history source changed")
                config = hm["protocol"]
                stored = [load_packet(p) for p in sorted((history / "packets").glob("*.npz"))]
                checkpoints = {r["action_id"]:r for r in json.loads((history / "checkpoints.json").read_text())}
                folder = output / (parent + "_" + arrangement); folder.mkdir()
                cp = dict(history=str(history.relative_to(ROOT)), total_budget=protocol["total_budget"],
                    reference_sample_count=16000, reference_seed=2026, thresholds_m=[.05,.1],
                    first_option_execution="complete_roundtrip_then_common_geometry_continuation")
                cm = dict(status="running", protocol=cp, source_sha256=frozen,
                    history_artifact_hashes=history_hashes, training_allowed=False)
                write(folder / "manifest.json", cm)
                world = create_competition_world(parent, arrangement)
                evaluator = ReconstructionEvaluator(world, count=16000, seed=2026)
                weight = float(world.mesh.get_surface_area()) / 16000
                reference = dict(points=evaluator.reference, classes=evaluator.classes,
                    sample_weight_m2=np.asarray(weight), vertices=np.asarray(world.mesh.vertices),
                    triangles=np.asarray(world.mesh.triangles))
                np.savez_compressed(folder / "reference.npz", **reference)
                rows = []
                for declaration in selected_pairs:
                    checkpoint = checkpoints[declaration["action_id"]]
                    if len(checkpoint["candidates"]) != declaration["candidates_per_arrangement"]:
                        raise ValueError("candidate cardinality changed")
                    _, _, runtime = restore(config, stored, checkpoint)
                    before = evaluate(runtime, world, evaluator, cp["thresholds_m"])
                    mesh = runtime.states[0]["mapper"].mesh()
                    np.savez_compressed(folder / f"prefix_step_{checkpoint['action_id']}.npz",
                        vertices=np.asarray(mesh.vertices), triangles=np.asarray(mesh.triangles),
                        vertex_colors=np.asarray(mesh.vertex_colors))
                    write(folder / f"features_step_{checkpoint['action_id']}.json",
                          describe_candidates(runtime, checkpoint["candidates"]))
                    for candidate in checkpoint["candidates"]:
                        name = f"step_{checkpoint['action_id']:03d}_candidate_{candidate['candidate_id']:03d}"
                        if parent == "Q0" and arrangement == "shelf_east":
                            name, row = reuse_verified(old, history, checkpoint, candidate, reference, before)
                            shutil.copytree(old / name, folder / name, copy_function=os.link)
                            reuse_checks.append(dict(context=folder.name, branch=name, source=str(old / name),
                                identity_checks="state,pool,route,physical_prefix,source,reference,budget,continuation,replay passed"))
                            write(output / "reuse_checks.json", reuse_checks)
                        else:
                            row = execute(config, cp, stored, checkpoint, candidate, evaluator, weight, folder / name)
                        rows.append(row); all_rows.append(row)
                        write(folder / "partial.json", rows); write(output / "partial.json", all_rows)
                        print(folder.name, name, "joint",round(row["joint_gain"],8), "safe",not row["termination"]["failed"],flush=True)
                cm["status"] = "complete"; write(folder / "manifest.json", cm); seal(folder)
                prior_replay=recovery_source/(folder.name+"_replay")
                if (prior_replay/"summary.json").is_file():
                    verify_artifacts(prior_replay)
                    prior_checks=json.loads((prior_replay/"summary.json").read_text())
                    old_folder=recovery_source/folder.name
                    if prior_checks["status"]!="passed" or prior_checks["branches"]!=len(rows):
                        raise ValueError("incomplete previous physical replay")
                    for row in rows:
                        branch=f"step_{row['checkpoint_action']:03d}_candidate_{row['candidate_id']:03d}"
                        if json.loads((old_folder/branch/"result.json").read_text())!=row:
                            raise ValueError("previous replay does not identify current recovered outcomes")
                    shutil.copytree(prior_replay,output/(folder.name+"_replay"),copy_function=os.link)
                else:
                    replay(folder, output / (folder.name + "_replay"))
                manifest["contexts"].append(folder.name); write(output / "manifest.json", manifest)
                print("CONTEXT_REPLAY_COMPLETE",folder.name,flush=True)
        if len(all_rows) != protocol["expected_candidate_outcomes"]:
            raise ValueError("incomplete predeclared outcome universe")
        for name, expected in frozen.items():
            if sha(ROOT / name) != expected: raise ValueError("source changed during acquisition")
        write(output / "summary.json", dict(status="complete", candidate_outcomes=len(all_rows),
            reused=len(reuse_checks), newly_executed=len(all_rows)-len(reuse_checks),
            replayed=len(all_rows), recovered_new_outcomes=len(RECOVERED_NEW), collisions=sum(r["collisions"] for r in all_rows),
            failures=sum(r["termination"]["failed"] for r in all_rows),
            returns=sum(r["termination"]["returned_to_anchor"] for r in all_rows),
            budget_violations=sum(r["total_paid_actions"]>r["total_budget"] for r in all_rows),
            first_options_complete=sum(r["first_full_option_completed"] for r in all_rows),
            training_allowed=False))
        manifest["status"]="complete"
    except Exception as error:
        manifest.update(status="failed", error=str(error), completed_outcomes=len(all_rows))
        raise
    finally:
        write(output / "recovery_checks.json", dict(source=str(recovery_source), recovered_new=RECOVERED_NEW, original_hashes=previous_manifest["source_sha256"]))
        write(output / "manifest.json", manifest); seal(output)


if __name__ == "__main__":
    parser=argparse.ArgumentParser(); parser.add_argument("--output",type=Path,required=True)
    parser.add_argument("--recovery-source",type=Path,required=True)
    args=parser.parse_args(); main(args.output.resolve(),args.recovery_source.resolve())
