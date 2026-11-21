"""Bind the two witnesses and their archived action/sensor evidence to GitHub."""

import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parent
GIT_BLOBS = {
    "PROTOCOL.json": "b8573de17fe0da7c5f85e7b34bc8f83f33640936",
    "earliest_witness.json": "d0dc99d0064d499d3874063e379cfb0bb401d8e3",
    "verification/results.csv": "56a8126636a629ee5d9d5dee0f29381241ddbe8e",
    "completion_audit.json": "df00eda7b31dd8f42915138b7c103e503cc11225",
}


def read(path):
    return json.loads(path.read_text())


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def git_blob(path):
    value = path.read_bytes()
    return hashlib.sha1(f"blob {len(value)}\0".encode() + value).hexdigest()


def bind(path, files, expected=None):
    actual = sha(path)
    if expected is not None and actual != expected:
        raise ValueError(f"Source archive changed: {path}")
    files[str(path)] = actual


def main():
    protocol_file = ROOT / "PROTOCOL.json"
    protocol = read(protocol_file)
    source = Path(protocol["source_local"])
    publication = read(source / "publication/receipt.json")
    prior = read(source / "completion_audit.json")
    if protocol["source_commit"] != publication["head"] or prior["counterfactual_actions_executed"] != 0:
        raise ValueError("Prior audit must match publication and have zero expanded actions")
    if any(git_blob(source / name) != expected for name, expected in GIT_BLOBS.items()):
        raise ValueError("Local source documents differ from the immutable Git blobs")
    files = {}
    for path in (protocol_file, Path(__file__)):
        bind(path, files)
    for name in (*GIT_BLOBS, "publication/receipt.json", "SOURCE_FREEZE.json", "EVIDENCE_INDEX.json", "exhaustive.py",
                 "replay/freeze.json", "replay/supervisor_receipt.json", "replay/replay.py", "verification/results.json",
                 "verification/archived_states.json"):
        bind(source / name, files)
    source_index = {row["layout"]: row for row in read(source / "EVIDENCE_INDEX.json")["raw_replays"]}
    source_events = {row["state_id"]: row for row in read(source / "verification/results.json")["nominal_category_witnesses"]}
    dev = source.parent / "same_goal_route_choice_20261121"
    dev_freeze = read(dev / "runner/development_freeze.json")
    for name in ("route_choice.py", "PROTOCOL.json", "runner/development_freeze.json", "runner/full_run.py"):
        bind(dev / name, files)
    for path, expected in dev_freeze["files_sha256"].items():
        bind(Path(path), files, expected)
    states = []
    for layout in protocol["layouts_in_order"]:
        pair = protocol["pairs"][layout]
        state_path = source / pair["source_state"]
        record = source_index[layout]
        manifest = source / record["manifest"]
        bind(manifest, files, record["manifest_sha256"])
        bind(state_path, files, record["state_files_sha256"][str(state_path.relative_to(manifest.parent))])
        witness = source_events[pair["state_id"]]
        if (witness["layout"] != layout or witness["paid_before"] != pair["paid_before"]
                or witness["global_call_id"] != pair["global_call_id"]):
            raise ValueError(f"Wrong source witness {layout}")
        receipts = {}
        for mode in ("G", "S"):
            arm = dev / f"runner/runs/development/{layout}/new{mode}"
            manifest_path = arm / "manifest.json"
            bind(manifest_path, files)
            native = read(manifest_path)
            for name, expected in native["files_sha256"].items():
                bind(arm / name, files, expected)
            receipts[mode] = sha(manifest_path)
        states.append({"layout": layout, "state_id": pair["state_id"], "state_path": str(state_path),
                       "state_sha256": sha(state_path), "archived_arm_manifest_sha256": receipts})
    with (ROOT / "SOURCE_FREEZE.json").open("x") as stream:
        json.dump({"status": "prelaunch_two_pairs_source_verified", "source_commit": protocol["source_commit"],
                   "source_root_tree": publication["tree"], "github_document_git_blobs": GIT_BLOBS,
                   "registered_pairs": states, "files_sha256": files}, stream, indent=2)
        stream.write("\n")


if __name__ == "__main__":
    main()
