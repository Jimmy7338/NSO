"""Seal source artifacts for the follow-up before examining its terminal observations."""

import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parent


def read(path):
    return json.loads(path.read_text())


def hash_file(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def bind(path, entries, expected=None):
    actual = hash_file(path)
    if expected is not None and actual != expected:
        raise ValueError(f"Archived hash mismatch: {path}")
    entries[str(path)] = actual


def bind_manifest(directory, filename, entries):
    manifest_path = directory / filename
    manifest = read(manifest_path)
    bind(manifest_path, entries)
    for name, expected in manifest["files_sha256"].items():
        bind(directory / name, entries, expected)


def main():
    protocol = read(ROOT / "PROTOCOL.json")
    previous = Path(protocol["source_local"])
    large_scene = previous.parent / "semantic_large_scene_20261121"
    publication = read(previous / "publication/receipt.json")
    if publication["head"] != protocol["source_commit"]:
        raise ValueError("Local publication does not correspond to specified source commit")
    prior_index = read(previous / "EVIDENCE_INDEX.json")
    runs = {row["run"]: row["files"] for row in prior_index["raw_runs"]}
    frozen = {}
    for file in (ROOT / "PROTOCOL.json", Path(__file__)):
        bind(file, frozen)
    for filename in (
        "publication/receipt.json", "completion_audit.json", "PROTOCOL.json", "PROTOCOL_FREEZE.json", "EVIDENCE_INDEX.json",
        "measurement/results/candidate_rows.json", "measurement/results/branch_b_continuation.json", "runner/execute.py",
        "runner/branch_b/continue_pair.py", "runner/branch_b/freeze.json", "ranking/results/candidate_oracle.csv",
        "measurement/measure.py", "measurement/measure_continuation.py",
    ):
        bind(previous / filename, frozen)
    for directory in ("mechanism/scenes", "mechanism/runner", "evaluation"):
        bind(large_scene / directory / "freeze.json", frozen)
    for layout in protocol["layouts_in_order"]:
        for candidate in protocol["new_routes"].get(layout, []) + protocol["archived_full_routes"].get(layout, []):
            short_rel = f"runner/runs/{layout}/candidate_{candidate}"
            for filename, digest in runs[short_rel].items():
                bind(previous / short_rel / filename, frozen, digest)
            if layout == "warehouse_right":
                extended_rel = f"runner/branch_b/runs/{layout}/candidate_{candidate}"
                for filename, digest in runs[extended_rel].items():
                    bind(previous / extended_rel / filename, frozen, digest)
        if layout != "warehouse_right":
            archived = large_scene / f"mechanism/runner/layouts/{layout}/runs/G/attempt_001"
            bind_manifest(archived, "artifact_manifest.json", frozen)
        scene = large_scene / "mechanism/scenes"
        for filename in (f"prepared/{layout}/evaluator_truth.npz", f"evaluator_references/{layout}/fixed_reference.npz",
                         f"evaluator_references/{layout}/RECEIPT.json"):
            bind(scene / filename, frozen)
    receipt = {
        "status": "input_frozen_before_new_terminal_outcomes", "source_commit": protocol["source_commit"],
        "protocol_sha256": frozen[str(ROOT / "PROTOCOL.json")], "source_evidence_tree": publication["root"],
        "new_continuations": sum(map(len, protocol["new_routes"].values())),
        "archived_comparators": sum(map(len, protocol["archived_full_routes"].values())), "files_sha256": frozen,
    }
    with (ROOT / "PROTOCOL_FREEZE.json").open("x") as stream:
        json.dump(receipt, stream, indent=2)
        stream.write("\n")


if __name__ == "__main__":
    main()
