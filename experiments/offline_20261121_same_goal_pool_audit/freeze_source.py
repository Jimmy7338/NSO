"""Check and bind the previously published inputs without changing their directory."""

import hashlib
import json
from pathlib import Path


HERE = Path(__file__).resolve().parent


def read(path):
    return json.loads(path.read_text())


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def git_blob(path):
    data = path.read_bytes()
    return hashlib.sha1(f"blob {len(data)}\0".encode() + data).hexdigest()


def record(path, entries, expected=None):
    actual = sha(path)
    if expected is not None and actual != expected:
        raise ValueError(f"Source archive changed: {path}")
    entries[str(path)] = actual


def main():
    protocol_path = HERE / "PROTOCOL.json"
    protocol = read(protocol_path)
    source = Path(protocol["source_local"])
    old = read(source / "publication/receipt.json")
    locked = read(source / "runner/development_freeze.json")
    summary = read(source / "completion_audit.json")
    if (old["head"] != protocol["source_commit"] or summary["runs"] != 8
            or summary["identical_action_observation_and_mesh_pairs"] != 4 or summary["holdout_actor_runs"] != 0
            or git_blob(source / "completion_audit.json") != "6fa9fba00f4f0cd82678d01f92e8361e336f162b"):
        raise ValueError("Published zero-difference source evidence is inconsistent")
    recorded = {}
    for path in (protocol_path, Path(__file__)):
        record(path, recorded)
    for name in ("publication/receipt.json", "completion_audit.json", "EVIDENCE_INDEX.json", "PROTOCOL.json",
                 "runner/development_freeze.json", "runner/development_supervisor_receipt.json", "measurement/development_decision.json",
                 "measurement/results/development/episodes.json", "measurement/results/development/choice_audit.json",
                 "diagnostics/development_availability_receipt.json"):
        record(source / name, recorded)
    for path, expected in locked["files_sha256"].items():
        record(Path(path), recorded, expected)
    manifests = read(source / "EVIDENCE_INDEX.json")["run_files_rooted_at_each_manifest"]
    if len(manifests) != 8:
        raise ValueError("Missing original run artifacts")
    for row in manifests:
        manifest = source / row["manifest"]
        record(manifest, recorded, row["manifest_sha256"])
        for name, digest in row["files_sha256"].items():
            record(manifest.parent / name, recorded, digest)
    data = {"status": "verified_source_archive_read_only", "source_commit": old["head"], "source_tree": old["tree"],
            "published_completion_audit_git_blob": git_blob(source / "completion_audit.json"),
            "archived_development_episodes": len(manifests), "archived_paired_zero_difference": 4,
            "files_sha256": recorded}
    with (HERE / "SOURCE_FREEZE.json").open("x") as stream:
        json.dump(data, stream, indent=2)
        stream.write("\n")


if __name__ == "__main__":
    main()
