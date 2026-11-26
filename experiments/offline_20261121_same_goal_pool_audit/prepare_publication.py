"""Stage the verified read-only audit and original per-state score records."""

import gzip
import hashlib
import json
import shutil

from audit import ROOT, read, sha


REPO_DIRECTORY = "experiments/offline_20261121_same_goal_pool_audit"
FILES = (
    "README.md", "RESULTS_ZH.md", "PROTOCOL.json", "SOURCE_FREEZE.json", "EVIDENCE_INDEX.json", "completion_audit.json",
    "freeze_source.py", "exhaustive.py", "audit.py", "prepare_publication.py",
    "replay/replay.py", "replay/supervise.py", "replay/freeze.json", "replay/supervisor_receipt.json",
    "replay/execution_correction.json", "replay/archived_attempts/attempt_001_terminal_json_guard/freeze.json",
    "verification/archive_states.py", "verification/verify_expansion.py",
    "verification/archived_states.csv", "verification/archived_states.json", "verification/results.csv", "verification/results.json",
)


def stage_file(path, output):
    output.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(path, output)


def main():
    work = ROOT / "publication/payload"
    work.mkdir(parents=True, exist_ok=True)
    for source in FILES:
        stage_file(ROOT / source, work / source)
    evidence = read(ROOT / "EVIDENCE_INDEX.json")
    if "earliest_witness" in evidence:
        name = evidence["earliest_witness"]
        stage_file(ROOT / name, work / name)
    for layout in evidence["raw_replays"]:
        manifest = ROOT / layout["manifest"]
        folder = work / "evidence" / layout["layout"]
        stage_file(manifest, folder / "manifest.json")
        for name, original in layout["state_files_sha256"].items():
            path = manifest.parent / name
            if sha(path) != original:
                raise ValueError(f"Raw result changed after verification: {path}")
            (folder / (path.name + ".gz")).write_bytes(gzip.compress(path.read_bytes(), mtime=0))
    index = []
    for path in sorted(work.rglob("*")):
        if path.is_file():
            data = path.read_bytes()
            index.append({"stage_path": str(path), "size": len(data), "sha256": sha(path),
                          "repo_path": f"{REPO_DIRECTORY}/{path.relative_to(work)}",
                          "git_sha1": hashlib.sha1(f"blob {len(data)}\0".encode() + data).hexdigest()})
    (work.parent / "upload.json").write_text(json.dumps(index, indent=2) + "\n")


if __name__ == "__main__":
    main()
