#!/usr/bin/env python3
"""Audit whether paired layouts really share decision-time geometry inputs."""
from dataclasses import replace
import argparse
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
from nso.cpu_sensor_contract_v10 import digest
from nso.decision_replay_v13 import load_packet
from scripts.collect_semantic_gain_v13_history import sha, write


def prefix_geometry_hashes(folder):
    geometry, semantic = [], []
    gh, sh = hashlib.sha256(), hashlib.sha256()
    for path in sorted((folder / "packets").glob("*.npz")):
        packet = load_packet(path)
        # Geometry baseline consumes depth/scan/poses, not color-derived shape.
        # This certificate does NOT establish equivalence for RGB shape methods.
        frame = replace(packet.frame, color_rgb=np.zeros_like(packet.frame.color_rgb),
                        semantic=np.zeros_like(packet.frame.semantic))
        gh.update(replace(packet, frame=frame).sha256().encode())
        sh.update(digest(packet.frame.semantic).encode())
        geometry.append(gh.hexdigest()); semantic.append(sh.hexdigest())
    return geometry, semantic


def main(source, output):
    manifest = json.loads((source / "manifest.json").read_text())
    if manifest["status"] != "complete":
        raise ValueError("multistate collection must finish first")
    inputs = json.loads((source / "artifact_hashes.json").read_text())
    for name, expected in inputs.items():
        if sha(source / name) != expected:
            raise ValueError(f"input hash mismatch: {name}")
    output.mkdir(parents=True, exist_ok=False)
    pairs = []
    for parent in manifest["protocol"]["parents"]:
        folders = [source / (parent + "_" + a) for a in manifest["protocol"]["arrangements"]]
        for folder in folders:
            if json.loads((folder / "result.json").read_text())["status"] != "passed":
                raise ValueError("unsafe or unverified history retained; cannot pass pairing gate")
        by = [{r["action_id"]: r for r in json.loads((f / "checkpoints.json").read_text())} for f in folders]
        histories = [prefix_geometry_hashes(f) for f in folders]
        for step in sorted(set(by[0]) | set(by[1])):
            if any(step not in d for d in by):
                pairs.append(dict(parent=parent, action_id=step, matched=False, reason="no_common_decision_boundary"))
                continue
            a, b = by[0][step], by[1][step]
            geometric = histories[0][0][step] == histories[1][0][step]
            semantic_changed = histories[0][1][step] != histories[1][1][step]
            pool = a["candidate_sha256"] == b["candidate_sha256"]
            feedback = digest(a["state"]["evidence"]["modules"]) == digest(b["state"]["evidence"]["modules"])
            pairs.append(dict(parent=parent, action_id=step, depth_scan_pose_history_equal=geometric,
                semantic_history_changed=semantic_changed, candidate_pool_equal=pool,
                feedback_audit_equal=feedback, matched=geometric and pool and feedback,
                candidate_count_east=len(a["candidates"]), candidate_count_west=len(b["candidates"])))
    result = dict(status="complete", parent_count=len(manifest["protocol"]["parents"]),
        pairs=pairs, matched_pairs=sum(r["matched"] for r in pairs),
        eligible_nonempty_pairs=sum(r["matched"] and r.get("candidate_count_east", 0)>0 for r in pairs),
        training_allowed=False, semantic_information_value_computed=False,
        boundary="Exact depth/scan/pose histories, pools and feedback; excludes RGB shape information. Unmatched states remain valid grouped-development data but not exact geometry-conditioned oracle pairs.")
    write(output / "result.json", result)
    write(output / "manifest.json", dict(status="complete", source=str(source), input_hashes=inputs,
        script_sha256=sha(Path(__file__).resolve())))
    write(output / "artifact_hashes.json", {str(p.relative_to(output)):sha(p)
        for p in output.glob("*") if p.is_file() and p.name != "artifact_hashes.json"})
    print(json.dumps({k:v for k,v in result.items() if k != "pairs"}, indent=2), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    main(args.source.resolve(), args.output.resolve())
