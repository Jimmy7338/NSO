#!/usr/bin/env python3
"""Collect every natural geometry decision for the predeclared V13 parents."""
import os
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
import argparse
import json
from pathlib import Path
import sys
import zipfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.collect_semantic_gain_v13_history_v13_1 import main as collect, sha, write


def main(output):
    protocol = ROOT / "configs/virtual3d/semantic_gain_v13_multistate_development.json"
    config = json.loads(protocol.read_text())
    base = json.loads((ROOT / "configs/virtual3d/semantic_gain_v13_history_smoke.json").read_text())
    if config["parents"] != ["Q0", "Q1", "Q2", "Q3"]:
        raise ValueError("only the predeclared development parents may be used")
    output.mkdir(parents=True, exist_ok=False)
    sources = [protocol, Path(__file__).resolve(), ROOT / "scripts/collect_semantic_gain_v13_history_v13_1.py",
        ROOT / "configs/virtual3d/semantic_gain_v13_history_smoke.json",
        ROOT / "configs/virtual3d/competition_v9_contexts.json",
        *[p for folder in ("nso", "env", "utils") for p in (ROOT / folder).rglob("*.py")]]
    frozen = {str(p.relative_to(ROOT)): sha(p) for p in sources}
    with zipfile.ZipFile(output / "source_snapshot.zip", "x", zipfile.ZIP_DEFLATED) as archive:
        for path in sources:
            archive.write(path, path.relative_to(ROOT))
    manifest = dict(status="running", protocol=config, source_sha256=frozen,
                    independent_parent_count=4, training_allowed=False)
    write(output / "manifest.json", manifest)
    rows = []
    try:
        for parent in config["parents"]:
            for arrangement in config["arrangements"]:
                name = parent + "_" + arrangement
                child = {**base, "parent": parent, "arrangement": arrangement,
                    "total_budget": config["total_budget"], "candidate_cap": config["candidate_cap"],
                    "coverage_slots": config["coverage_slots"],
                    "checkpoint_decision_ordinals": list(range(1, config["total_budget"] + 2)),
                    "parent_group": parent, "sampling": config["sampling"]}
                child_protocol = output / (name + "_protocol.json")
                write(child_protocol, child)
                collect(output / name, child_protocol)
                result = json.loads((output / name / "result.json").read_text())
                checkpoints = json.loads((output / name / "checkpoints.json").read_text())
                row = dict(parent=parent, arrangement=arrangement, group=parent, result=result,
                    decision_steps=[c["action_id"] for c in checkpoints],
                    candidates=sum(len(c["candidates"]) for c in checkpoints))
                rows.append(row); write(output / "partial.json", rows)
                print("MULTISTATE", name, row["decision_steps"], flush=True)
        if any(sha(ROOT / name) != expected for name, expected in frozen.items()):
            raise ValueError("source changed during multistate collection")
        summary = dict(status="complete", episodes=len(rows), parent_groups=len(config["parents"]),
            decisions=sum(len(r["decision_steps"]) for r in rows),
            candidates=sum(r["candidates"] for r in rows),
            failed_episodes=sum(r["result"]["status"] != "passed" for r in rows),
            training_allowed=False, semantic_information_gate_passed=False, rows=rows)
        write(output / "summary.json", summary)
        manifest["status"] = "complete"
    except Exception as error:
        manifest.update(status="failed", error=str(error), completed_episodes=len(rows))
        raise
    finally:
        write(output / "manifest.json", manifest)
        write(output / "artifact_hashes.json", {str(p.relative_to(output)): sha(p)
            for p in output.rglob("*") if p.is_file() and p != output / "artifact_hashes.json"})


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    main(args.output.resolve())
