#!/usr/bin/env python3
"""Prepare compact, hash-checked evidence for the NSO draft branch."""

import hashlib
from pathlib import Path
import shutil
import sys

from runner import PREVIOUS, ROOT


sys.path.append(str(PREVIOUS))
from archive import zip_run
from regression import read, sha, write
from study import verify


REMOTE = "experiments/offline_20261122_region_global_revision"


def main():
    destination = ROOT / "publication"
    payload = destination / "payload"
    payload.mkdir(parents=True, exist_ok=False)
    sources = [ROOT / name for name in (
        "README.md", "PROTOCOL.json", "study_revision.py", "runner.py", "evaluate_revision.py",
        "report.py", "archive_revision.py", "bind_fixture.py", "formal_preactor_hash_repair.json",
        "FORMAL_FREEZE.preflight_001.json", "diagnosis", "results", "scenes/SCENES.md",
        "scenes/scene.py", "scenes/prepare_scenes.py", "scenes/geometry_audit.py",
        "scenes/actor_freeze_holdout.json", "scenes/evaluator_freeze_holdout.json",
        "scenes/formal_geometry_audit.json", "scenes/reference_protocol.json",
    )]
    bundles = payload / "run_evidence"
    bundles.mkdir()
    outcomes = []
    for phase in ("development", "formal"):
        freeze_path = ROOT / f"{phase.upper()}_FREEZE.json"
        sources.extend((freeze_path, ROOT / f"{phase.upper()}_PROTOCOL.json"))
        phase_receipt = ROOT / "execution" / phase / "phase_receipt.json"
        sources.extend(sorted(phase_receipt.parent.glob("phase_receipt*.json")))
        sealed, receipt = read(freeze_path), read(phase_receipt)
        summary = ROOT / "results" / phase / "summary.json"
        measured = read(summary)
        assert measured["status"] == "evaluated"
        assert measured["run_freeze_sha256"] == receipt["freeze_sha256"] == sha(freeze_path)
        assert len(receipt["episodes"]) == len(sealed["jobs"])
        verify(sealed["actor_files"])
        verify(sealed["evaluator_files"])
        attempts = {result: read(result) for result in phase_receipt.parent.glob("*/*/attempt_*/exit.json")}
        for exit_path, record in sorted(attempts.items()):
            run = Path(record["run_path"])
            result = Path(record["result_path"])
            if record["result_sha256"]:
                assert sha(result) == record["result_sha256"]
            if record["exit_code"] == 0:
                verify({str(run/name): digest for name, digest in read(result)["files_sha256"].items()})
            name = f"{phase}_{record['layout']}_{record['arm']}_{exit_path.parent.name}.zip"
            outcomes.append(zip_run(run.parent, bundles/name) | {
                field: record[field] for field in ("layout", "arm", "exit_code", "result_sha256")
            })
    fixtures = {name: zip_run(ROOT/"scenes"/name, bundles/f"scenes_{name}.zip", include_mesh=True)
                for name in ("prepared", "precomputed_assets", "evaluator_references")}
    for source in sources:
        assert source.exists(), source
        files = source.rglob("*") if source.is_dir() else [source]
        for path in files:
            if path.is_file() and "__pycache__" not in path.parts:
                target = payload / path.relative_to(ROOT)
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(path, target)
    write(payload/"EVIDENCE_INDEX.json", {
        "scenario": "simulator", "outcomes": outcomes, "fixture_bundles": fixtures,
        "visited_source_baseline_sha256": sha(PREVIOUS/"results/formal/summary.json"),
        "raw_mesh_files_held_in_native_directories": True,
    })
    plan = []
    for path in sorted(x for x in payload.rglob("*") if x.is_file()):
        data = path.read_bytes()
        plan.append({"stage_path": str(path), "repo_path": REMOTE+"/"+path.relative_to(payload).as_posix(),
                     "size": len(data), "sha256": hashlib.sha256(data).hexdigest(),
                     "git_sha1": hashlib.sha1(f"blob {len(data)}\0".encode()+data).hexdigest()})
    write(destination/"upload.json", plan)
    print(f"Staged {len(plan)} files")


if __name__ == "__main__":
    main()
