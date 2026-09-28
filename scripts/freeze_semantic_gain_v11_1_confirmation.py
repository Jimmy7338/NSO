#!/usr/bin/env python3
"""Seal the V11.1 E confirmation inventory before constructing any E world."""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import zipfile

ROOT = Path(__file__).resolve().parents[1]


def sha(path):
    import hashlib
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n")


def main(output, future_preparation):
    if output.exists() or future_preparation.exists():
        raise FileExistsError("freeze output or E preparation already exists")
    protocol = ROOT / "configs/virtual3d/semantic_gain_v11_1_confirmation_protocol.json"
    contexts = ROOT / "configs/virtual3d/semantic_gain_v11_1_confirmation_contexts.json"
    development = ROOT / "eval_results/semantic_gain_v11_1_closed_loop_development_20260911"
    replay = ROOT / "eval_results/semantic_gain_v11_1_closed_loop_replay_20260911/verification.json"
    if json.loads((development / "summary.json").read_text())["status"] != "passed":
        raise ValueError("development gates did not pass")
    if json.loads(replay.read_text())["status"] != "passed":
        raise ValueError("independent physical replay did not pass")
    sources = [protocol, contexts, ROOT / "env/virtual3d_confirmation_v11.py",
        ROOT / "env/virtual3d_competition_v9.py", ROOT / "nso/semantic_gain_v11.py",
        ROOT / "nso/cpu_four_modules_v10.py", ROOT / "nso/runtime_integration.py",
        ROOT / "nso/hierarchical_options_v10.py",
        ROOT / "eval_results/semantic_gain_v11_integration_bundle_20260911/integration_bundle.json"]
    output.mkdir(parents=True)
    receipt = {"schema_version": "semantic_gain_v11_1_confirmation_freeze/1",
        "status": "sealed_before_E_world_construction", "created_utc": datetime.now(
            timezone.utc).isoformat(), "future_preparation_path": str(future_preparation),
        "worlds_constructed_before_freeze": 0, "outcomes_viewed_before_freeze": 0,
        "source_sha256": {str(path.relative_to(ROOT)): sha(path) for path in sources},
        "development_artifact_hashes_sha256": sha(development / "artifact_hashes.json"),
        "development_replay_sha256": sha(replay),
        "confirmation_retraining_allowed": False}
    write(output / "receipt.json", receipt)
    with zipfile.ZipFile(output / "sources.zip", "w", zipfile.ZIP_DEFLATED) as archive:
        for path in sources:
            archive.write(path, str(path.relative_to(ROOT)))
    write(output / "artifact_hashes.json", {path.name: sha(path) for path in output.iterdir()
                                             if path.name != "artifact_hashes.json"})
    print(json.dumps(receipt, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--future-preparation", type=Path, required=True)
    args = parser.parse_args()
    main(args.output.resolve(), args.future_preparation.resolve())
