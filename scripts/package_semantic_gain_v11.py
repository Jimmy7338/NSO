#!/usr/bin/env python3
"""Package the frozen V11 development ensemble with a decision-input guard."""
import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from nso.semantic_gain_v11 import candidate_features


def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2,
        sort_keys=True, allow_nan=False) + "\n")


def main(run, data, output):
    output.mkdir(parents=True, exist_ok=False)
    hashes = json.loads((run / "artifact_hashes.json").read_text())
    if any(sha(run / name) != digest for name, digest in hashes.items()):
        raise ValueError("V11 training artifact changed")
    model = json.loads((run / "frozen_model.json").read_text())
    protocol = json.loads((run / "manifest.json").read_text())["protocol"]
    semantic, geometry = [], []
    source_hashes = {}
    for context in protocol["data"]["contexts"]:
        for arrangement in protocol["data"]["arrangements"]:
            folder = data / context / arrangement
            for name in ("candidates.json", "predictions.json", "candidate_audit.json"):
                source_hashes[str((folder / name).relative_to(data))] = sha(folder / name)
            candidates = json.loads((folder / "candidates.json").read_text())
            prediction = json.loads((folder / "predictions.json").read_text())
            audit = json.loads((folder / "candidate_audit.json").read_text())
            pred = {int(x["candidate_id"]): x for x in prediction["audit"]}
            for candidate in candidates:
                sx, gx, _ = candidate_features(pred[int(candidate["candidate_id"])],
                                                candidate, audit["measured_assets"])
                semantic.append(sx); geometry.append(gx)
    semantic, geometry = np.asarray(semantic), np.asarray(geometry)
    guard = {}
    for name, values in (("semantic", semantic), ("geometry", geometry)):
        lower, upper = values.min(axis=0), values.max(axis=0)
        margin = np.maximum(.25 * (upper - lower), 1e-6)
        guard[name] = {"training_min": lower.tolist(), "training_max": upper.tolist(),
            "lower": (lower - margin).tolist(), "upper": (upper + margin).tolist()}
    bundle = {"schema_version": "semantic_gain_v11_integration_bundle/1",
        "status": "frozen_development_model_with_input_guard",
        "model": model, "guard": guard,
        "guard_rule": "whole candidate history falls back to equal-capacity geometry MLP if any feature leaves a 25 percent training-range margin",
        "natural_confidence_calibrated": False, "independent_confirmation": False,
        "training_artifact_hashes_sha256": sha(run / "artifact_hashes.json"),
        "decision_input_files_sha256": source_hashes}
    write(output / "integration_bundle.json", bundle)
    write(output / "artifact_hashes.json", {"integration_bundle.json": sha(output / "integration_bundle.json")})
    print(json.dumps({"status": "complete", "rows": len(semantic),
                      "model": str(output / "integration_bundle.json")}, indent=2))


if __name__ == "__main__":
    p = argparse.ArgumentParser(); p.add_argument("--run", type=Path, required=True)
    p.add_argument("--data", type=Path, required=True); p.add_argument("--output", type=Path, required=True)
    a = p.parse_args(); main(a.run.resolve(), a.data.resolve(), a.output.resolve())
