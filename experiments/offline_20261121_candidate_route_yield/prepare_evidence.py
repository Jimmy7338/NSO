"""Index native evidence and prepare the compact GitHub evidence tree."""

import gzip
import hashlib
import json
from pathlib import Path
import shutil


ROOT = Path(__file__).resolve().parent
PUBLICATION = ROOT / "publication"
REPOSITORY_DIRECTORY = "experiments/offline_20261121_candidate_route_yield"
DIRECT_FILES = (
    "README.md", "RESULTS_ZH.md", "PROTOCOL.json", "PROTOCOL_FREEZE.json", "decision.json", "decide.py",
    "audit_evidence.py", "prepare_evidence.py", "completion_audit.json", "EVIDENCE_INDEX.json",
    "runner/execute.py", "runner/supervise.py", "runner/freeze.json", "runner/supervisor_receipt.json",
    "runner/branch_b/continue_pair.py", "runner/branch_b/freeze.json",
    "measurement/measure.py", "measurement/measure_continuation.py", "measurement/results/candidate_rows.csv",
    "measurement/results/branch_b_continuation.csv", "measurement/results/branch_b_continuation.json",
    "ranking/README.md", "ranking/archive_predictions.py", "ranking/archive_predictions.csv",
    "ranking/archive_prediction_receipt.json", "ranking/rank_routes.py", "ranking/plot_ranking.py",
    "ranking/results/per_candidate.csv", "ranking/results/pairwise.csv", "ranking/results/rank_agreement_by_parent.csv",
    "ranking/results/swap_audit.csv", "ranking/results/candidate_oracle.csv", "ranking/results/analysis_receipt.json",
    "ranking/results/candidate_rank_yield.pdf", "ranking/results/candidate_rank_yield.svg",
    "ranking/results/figure_receipt.json",
)
COMPRESSED_FILES = ("measurement/results/candidate_rows.json", "ranking/results/ranking_summary.json")


def sha256(data):
    return hashlib.sha256(data).hexdigest()


def write_json(path, data):
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False) + "\n")


def index_raw_receipts():
    roots = (ROOT / "runner/runs", ROOT / "runner/branch_b/runs")
    records = []
    metadata = {}
    for folder in roots:
        for manifest_file in sorted(folder.glob("*/*/manifest.json")):
            run = manifest_file.parent
            manifest = json.loads(manifest_file.read_text())
            selected = [manifest_file, run / "audit.json", run / "steps.json", run / "first_anchor.json"]
            if (run / "checkpoint_recovery.json").is_file():
                selected.append(run / "checkpoint_recovery.json")
            checkpoint_receipts = sorted(run.glob("*/checkpoint.json"))
            files = {}
            for path in selected + checkpoint_receipts:
                data = path.read_bytes()
                files[str(path.relative_to(run))] = sha256(data)
                if path.name in {"manifest.json", "checkpoint_recovery.json", "checkpoint.json"}:
                    metadata[str(path.relative_to(ROOT))] = json.loads(data)
            for receipt_path in checkpoint_receipts:
                receipt = json.loads(receipt_path.read_text())
                for filename, expected in receipt["files_sha256"].items():
                    path = receipt_path.parent / filename
                    actual = sha256(path.read_bytes())
                    if actual != expected:
                        raise ValueError(f"Native checkpoint hash mismatch at {path}")
                    files[str(path.relative_to(run))] = actual
            records.append({"run": str(run.relative_to(ROOT)), "status": manifest["status"], "files": files})
    if len(records) != 18:
        raise ValueError(f"Expected 16 short runs plus 2 continuation records, got {len(records)}")
    write_json(ROOT / "EVIDENCE_INDEX.json", {
        "schema": "nso_candidate_route_yield_evidence_v1", "local_root": str(ROOT),
        "source_inputs": "PROTOCOL_FREEZE.json", "raw_runs": records,
        "remote_metadata_gzip": "evidence/checkpoint_receipts.json.gz",
        "native_snapshot_location": "Local paths in raw_runs. Native meshes and beliefs are retained on the original host.",
    })
    return metadata


def main():
    metadata = index_raw_receipts()
    stage = PUBLICATION / "payload"
    stage.mkdir(parents=True, exist_ok=True)
    for relative in DIRECT_FILES:
        destination = stage / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT / relative, destination)
    for relative in COMPRESSED_FILES:
        (stage / (relative + ".gz")).write_bytes(gzip.compress((ROOT / relative).read_bytes(), mtime=0))
    receipt_payload = stage / "evidence/checkpoint_receipts.json.gz"
    receipt_payload.parent.mkdir(parents=True, exist_ok=True)
    receipt_payload.write_bytes(gzip.compress(json.dumps(metadata, sort_keys=True).encode(), mtime=0))
    continuation_steps = {
        str(path.relative_to(ROOT)): json.loads(path.read_bytes())
        for path in sorted((ROOT / "runner/branch_b/runs").glob("*/*/steps.json"))
    }
    if len(continuation_steps) != 2:
        raise ValueError("The two recorded continuation traces are required.")
    (receipt_payload.parent / "branch_b_steps.json.gz").write_bytes(
        gzip.compress(json.dumps(continuation_steps, sort_keys=True).encode(), mtime=0)
    )
    upload = []
    for path in sorted(stage.rglob("*")):
        if path.is_file():
            data = path.read_bytes()
            upload.append({
                "stage_path": str(path), "repo_path": f"{REPOSITORY_DIRECTORY}/{path.relative_to(stage)}",
                "size": len(data), "sha256": sha256(data),
                "git_sha1": hashlib.sha1(f"blob {len(data)}\0".encode() + data).hexdigest(),
            })
    write_json(PUBLICATION / "upload.json", upload)


if __name__ == "__main__":
    main()
