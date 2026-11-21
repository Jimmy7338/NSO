"""Prepare compact per-route metadata and the publication tree."""

import gzip
import hashlib
import json
from pathlib import Path
import shutil

from summarize import ROOT, read, sha


DIRECTORY = "experiments/offline_20261121_candidate_route_persistence"
FILES = (
    "README.md", "RESULTS_ZH.md", "PROTOCOL.json", "PROTOCOL_FREEZE.json", "EVIDENCE_INDEX.json", "completion_audit.json",
    "freeze_inputs.py", "audit.py", "summarize.py", "plot_pairs.py", "prepare_publication.py",
    "runner/continue_route.py", "runner/supervise.py", "runner/freeze.json", "runner/supervisor_receipt.json",
    "measurement/evaluate.py", "measurement/recover_gallery_archive.py", "measurement/results/terminal.json",
    "measurement/results/checkpoints.csv", "measurement/results/terminal.csv", "measurement/results/interval_geometry.csv",
    "interval/INTERVAL_ZH.md", "interval/analyze_gallery.py", "interval/interval_receipt.json", "interval/neutral_move_summary.csv",
    "interval/neutral_steps.csv", "interval/interval_outcomes.csv", "interval/correct_free_by_region.csv",
    "interval/fixed_surface_by_region.csv", "interval/J_product_decomposition.csv",
    "results/paired_layouts.csv", "results/paired_terminal.csv", "results/decision.json", "results/joint_quality_by_action.pdf",
    "results/joint_quality_by_action.svg",
)
COMPRESSED = ("measurement/results/checkpoints.json", "measurement/results/coverage_curve.csv", "measurement/results/interval_geometry.json")


def compress(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(gzip.compress(data, mtime=0))


def package_traces(stage):
    index = read(ROOT / "EVIDENCE_INDEX.json")
    for key in ("new_runs", "archived_gallery_sensor_recoveries"):
        for row in index[key]:
            path = ROOT / row["manifest"]
            directory = path.parent
            metadata = {"manifest": read(path), "manifest_sha256": row["manifest_sha256"]}
            for part in ("checkpoint_recovery.json", "steps.json"):
                if (directory / part).is_file():
                    metadata[part] = read(directory / part)
            choices = directory / "choices.json"
            if choices.is_file():
                metadata["selected_choices"] = [
                    {key: row["outputs"][key] for key in ("selected", "mode", "plan_number", "policy_variant")}
                    for row in read(choices)
                ]
            metadata["checkpoints"] = {str(part.relative_to(directory)): read(part)
                                       for part in sorted(directory.glob("*/checkpoint.json"))}
            output = f"{directory.parent.name}_{directory.name}.json.gz"
            compress(stage / "evidence" / output, json.dumps(metadata, sort_keys=True).encode())


def main():
    stage = ROOT / "publication/payload"
    stage.mkdir(parents=True, exist_ok=True)
    for relative in FILES:
        out = stage / relative
        out.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT / relative, out)
    for relative in COMPRESSED:
        compress(stage / (relative + ".gz"), (ROOT / relative).read_bytes())
    package_traces(stage)
    upload = []
    for path in sorted(stage.rglob("*")):
        if path.is_file():
            data = path.read_bytes()
            upload.append({"stage_path": str(path), "size": len(data), "sha256": sha(path),
                           "repo_path": f"{DIRECTORY}/{path.relative_to(stage)}",
                           "git_sha1": hashlib.sha1(f"blob {len(data)}\0".encode() + data).hexdigest()})
    (stage.parent / "upload.json").write_text(json.dumps(upload, indent=2) + "\n")


if __name__ == "__main__":
    main()
