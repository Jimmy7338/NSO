"""Stage the compact study and keep replay-critical logs byte-for-byte."""

import gzip
import hashlib
import io
import json
from pathlib import Path
import shutil
import tarfile

from report import ROOT, read, sha


REPOSITORY_DIRECTORY = "experiments/offline_20261121_same_goal_route_choice"
DIRECT = (
    "README.md", "RESULTS_ZH.md", "PROTOCOL.json", "EVIDENCE_INDEX.json", "completion_audit.json",
    "route_choice.py", "report.py", "audit.py", "prepare_publication.py",
    "runner/full_run.py", "runner/supervise.py", "runner/development_freeze.json", "runner/development_supervisor_receipt.json",
    "measurement/evaluate.py", "measurement/development_decision.json", "measurement/results/development/episodes.csv",
    "measurement/results/development/checkpoints.csv", "diagnostics/RUNTIME_REVIEW.md", "diagnostics/DEVELOPMENT_AVAILABILITY_ZH.md",
    "diagnostics/audit_availability.py", "diagnostics/development_availability_by_arm.csv",
    "diagnostics/development_availability_receipt.json", "diagnostics/development_paired_first_actual_effect.csv",
    "results/development/paired_layouts.csv", "results/development/paired_quality.pdf", "results/development/paired_quality.svg",
    "results/development/REPORT_RECEIPT.json",
    "holdout/scene.py", "holdout/prepare.py", "holdout/seal_inputs.py", "holdout/reference_protocol.json",
    "holdout/freeze.json", "holdout/evaluator_freeze.json",
)
ZIP = ("measurement/results/development/episodes.json", "measurement/results/development/choice_audit.json",
       "measurement/results/development/choice_audit.csv", "measurement/results/development/checkpoints.json",
       "diagnostics/development_availability_decisions.csv")


def archive(path, members):
    buffer = io.BytesIO()
    with tarfile.open(fileobj=buffer, mode="w") as tar:
        for member in members:
            data = member.read_bytes()
            info = tarfile.TarInfo(member.name)
            info.size = len(data)
            tar.addfile(info, io.BytesIO(data))
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(gzip.compress(buffer.getvalue(), mtime=0))


def main():
    stage = ROOT / "publication/payload"
    stage.mkdir(parents=True, exist_ok=True)
    for name in DIRECT:
        to = stage / name
        to.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT / name, to)
    for name in ZIP:
        (stage / (name + ".gz")).write_bytes(gzip.compress((ROOT / name).read_bytes(), mtime=0))
    data = read(ROOT / "EVIDENCE_INDEX.json")
    for run in data["run_files_rooted_at_each_manifest"]:
        manifest = ROOT / run["manifest"]
        directory = manifest.parent
        filename = f"{directory.parent.name}_{directory.name}.tar.gz"
        archive(stage / "evidence" / filename, [manifest, *(directory / name for name in (
            "attempt.json", "steps.json", "local_choices.json", "runtime_summary.json",
        ))])
    listing = []
    for path in sorted(stage.rglob("*")):
        if path.is_file():
            raw = path.read_bytes()
            listing.append({"stage_path": str(path), "size": len(raw), "sha256": sha(path),
                            "repo_path": f"{REPOSITORY_DIRECTORY}/{path.relative_to(stage)}",
                            "git_sha1": hashlib.sha1(f"blob {len(raw)}\0".encode() + raw).hexdigest()})
    (stage.parent / "upload.json").write_text(json.dumps(listing, indent=2) + "\n")


if __name__ == "__main__":
    main()
