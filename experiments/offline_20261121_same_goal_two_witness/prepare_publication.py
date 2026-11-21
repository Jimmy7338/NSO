"""Prepare a compact publication with native step and intervention receipts."""

import gzip
import hashlib
import io
import json
import shutil
import tarfile

from report import ROOT, read, sha


DIRECTORY = "experiments/offline_20261121_same_goal_two_witness"
DIRECT = (
    "README.md", "RESULTS_ZH.md", "PROTOCOL.json", "SOURCE_FREEZE.json", "EVIDENCE_INDEX.json", "completion_audit.json",
    "freeze_source.py", "audit.py", "report.py", "prepare_publication.py",
    "runner/continue_witness.py", "runner/supervise.py", "runner/freeze.json", "runner/supervisor_receipt.json",
    "measurement/evaluate.py", "measurement/results/episodes.csv", "measurement/results/episodes.json",
    "measurement/results/checkpoints.csv", "measurement/results/checkpoints.json", "measurement/results/pair_causal_audit.csv",
    "measurement/results/pair_causal_audit.json", "measurement/results/engineering_gate.json",
    "diagnostics/source_mechanism.py", "diagnostics/source_mechanism.json", "diagnostics/SOURCE_MECHANISM_ZH.md",
    "diagnostics/audit_realized.py", "diagnostics/gallery_common_sensor_checks.csv",
    "results/paired_results.csv", "results/decision.json", "results/J_by_paid_action.pdf",
    "results/J_by_paid_action.svg",
)


def copy_relative(stage, name):
    destination = stage / name
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(ROOT / name, destination)


def package_arm(stage, manifest_path):
    directory = manifest_path.parent
    buffer = io.BytesIO()
    data_files = [manifest_path, *(directory / name for name in ("intervention.json", "steps.json", "audit.json",
                                                                    "runtime_summary.json", "effective_runtime.json")),
                  *sorted(directory.glob("*/checkpoint.json"))]
    with tarfile.open(fileobj=buffer, mode="w") as tar:
        for file in data_files:
            data = file.read_bytes()
            item = tarfile.TarInfo(str(file.relative_to(directory)))
            item.size = len(data)
            tar.addfile(item, io.BytesIO(data))
    out = stage / "evidence" / f"{directory.parent.name}_{directory.name}.tar.gz"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_bytes(gzip.compress(buffer.getvalue(), mtime=0))


def main():
    stage = ROOT / "publication/payload"
    stage.mkdir(parents=True, exist_ok=True)
    for filename in DIRECT:
        copy_relative(stage, filename)
    for path in sorted((ROOT / "diagnostics").rglob("realized_*.json")) + sorted((ROOT / "diagnostics").rglob("realized_*.csv")):
        copy_relative(stage, path.relative_to(ROOT))
    coverage = "measurement/results/coverage_curve.csv"
    (stage / (coverage + ".gz")).write_bytes(gzip.compress((ROOT / coverage).read_bytes(), mtime=0))
    index = read(ROOT / "EVIDENCE_INDEX.json")
    for arm in index["raw_arms"]:
        path = ROOT / arm["manifest"]
        if sha(path) != arm["manifest_sha256"]:
            raise ValueError(f"Raw receipt changed: {path}")
        package_arm(stage, path)
    for failed in index["interruptions"]:
        copy_relative(stage, failed["receipt"])
    listing = []
    for path in sorted(stage.rglob("*")):
        if path.is_file():
            raw = path.read_bytes()
            listing.append({"stage_path": str(path), "size": len(raw), "sha256": sha(path),
                            "repo_path": f"{DIRECTORY}/{path.relative_to(stage)}",
                            "git_sha1": hashlib.sha1(f"blob {len(raw)}\0".encode() + raw).hexdigest()})
    (stage.parent / "upload.json").write_text(json.dumps(listing, indent=2) + "\n")


if __name__ == "__main__":
    main()
