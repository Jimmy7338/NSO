#!/usr/bin/env python3
"""Index the frozen local study and package its compact native evidence."""

import argparse
from pathlib import Path
import tarfile

from run import FREEZE, HERE, read, sha, write


SOURCE_INDEX = tuple(
    HERE.parent.parent / experiment / "publication" / "payload" / "EVIDENCE_INDEX.json"
    for experiment in (
        "candidate_route_yield_20261121",
        "route_intervention_20261122",
        "route_top_geometry_20261122",
    )
)
EXCLUDED_NAMES = {"__pycache__", "supervisor.pid"}


def describe(path):
    return {"path": path.relative_to(HERE).as_posix(), "bytes": path.stat().st_size, "sha256": sha(path)}


def collect():
    result = HERE / "results"
    measured = read(result / "measured.json")
    execution = read(HERE / "execution" / "phase_receipt.json")
    freeze = read(FREEZE)
    assert measured["freeze_sha256"] == execution["freeze_sha256"] == sha(FREEZE)
    assert set(measured["arms"]) == {item["layout"] for item in freeze["schedule"]}
    receipts = {(item["layout"], item["arm"]): item for item in execution["execution_records"]}
    assert len(receipts) == len(freeze["schedule"]) == len(execution["execution_records"])
    assert set(receipts) == {(item["layout"], item["arm"]) for item in freeze["schedule"]}
    replacements = {}
    for retry in sorted((HERE / "execution_attempt_002").glob("*/retry_exit.json")):
        record = read(retry)
        key = (record["layout"], record["arm"])
        original = receipts[key]
        assert original["exit_code"] != 0 and original["freeze_sha256"] == record["freeze_sha256"]
        assert original["command"] == record["command"]
        replacements[key] = retry
        receipts[key] = record

    outcomes = []
    for (layout, arm), receipt in receipts.items():
        item = measured["arms"][layout][arm]
        actor_output = Path(receipt["receipt_path"])
        assert receipt["receipt_sha256"] == sha(actor_output)
        ok = receipt["exit_code"] == 0
        assert item["status"] == ("verified_native_episode" if ok else "technical_failure")
        if ok:
            assert item["manifest_sha256"] == receipt["receipt_sha256"]
        outcomes.append({
            "layout": layout,
            "arm": arm,
            "status": item["status"],
            "paid_actions": item.get("paid"),
            "route_event": item.get("route_event"),
            "runtime_receipt": describe(actor_output),
            "retry_receipt": describe(replacements[(layout, arm)]) if (layout, arm) in replacements else None,
        })

    reader = read(HERE / "MEASUREMENT_BINDING_EFFECTIVE.json")
    assert reader["effective_evaluator_sha256"] == sha(HERE / "evaluate.py")
    assert reader["previous_binding_sha256"] == sha(HERE / "MEASUREMENT_BINDING.json")
    assert reader["repair_receipt_sha256"] == sha(HERE / "evidence_checks" / "eval_attempt_001" / "REPAIR.txt")
    release_paths = []
    retained_raw = []
    for path in sorted(HERE.rglob("*")):
        if path.is_symlink() or not path.is_file():
            continue
        relative = path.relative_to(HERE)
        if EXCLUDED_NAMES.intersection(relative.parts) or path.suffix in {".pyc", ".gz"}:
            continue
        if relative.parts[0] == "evidence_checks" and relative.parts[1] not in {
            "eval_attempt_001", "raycast_diagnosis", "gc_lifetime_diagnosis",
            "aborted_checkpoint_diagnostics.json", "early_stop_48.json",
        }:
            continue
        if path.name == "EVIDENCE_INDEX.json":
            continue
        if ("runs" in relative.parts or "partial_native" in relative.parts
                or relative.parts[:2] in {
                    ("evidence_checks", "raycast_diagnosis"),
                    ("evidence_checks", "gc_lifetime_diagnosis"),
                }) and path.suffix == ".ply":
            retained_raw.append(path)
        else:
            release_paths.append(path)
    assert all(path.is_file() for path in SOURCE_INDEX)
    history = [{"path": str(path), "sha256": sha(path)} for path in SOURCE_INDEX]
    index = {
        "status": "all_prospective_actor_outcomes_accounted_for",
        "freeze_sha256": sha(FREEZE),
        "formal_schedule": outcomes,
        "compact_payload": [describe(path) for path in release_paths],
        "raw_native_meshes_on_robot": [describe(path) for path in retained_raw],
        "previous_route_experiments": history,
        "fidelity": (
            "Simulator geometry for reference scoring is stored separately from actor observations. "
            "Raw native PLY meshes are retained at their indexed paths on the robot."
        ),
    }
    target = HERE / "EVIDENCE_INDEX.json"
    write(target, index)
    return release_paths + [target]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--payload", type=Path, default=HERE / "local_native_evidence.tar.gz")
    args = parser.parse_args()
    entries = collect()
    with tarfile.open(args.payload, "x:gz", compresslevel=4) as payload:
        for path in entries:
            payload.add(path, arcname=f"local/{path.relative_to(HERE).as_posix()}", recursive=False)
    print(args.payload, sha(args.payload))


if __name__ == "__main__":
    main()
