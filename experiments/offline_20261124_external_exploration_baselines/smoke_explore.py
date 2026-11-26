"""Run untouched upstream search on the saved first observation of each floor."""

import hashlib
import json
from pathlib import Path

import numpy as np

from explore_search import BINARY, rank_frontiers


ROOT = Path(__file__).resolve().parent
BASE = ROOT.parent / "region_independent_floors_20261123"
SOURCE_SHA = "712bdd41027b645c9c876a4c0071478f090825ea"


def sha(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


study = json.loads((BASE / "scenes/actor_freeze_holdout.json").read_text())
smoke = []
for entry in study["layouts"]:
    cold = BASE / "execution/formal" / entry["layout"] / "F_strict/attempt_001/run/start"
    receipt = json.loads((cold / "checkpoint.json").read_text())
    assert receipt["packet_sha256"] == entry["initial_packet_sha256"]
    belief_path = cold / "belief.npz"
    assert receipt["files_sha256"]["belief.npz"] == sha(belief_path)
    with np.load(belief_path, allow_pickle=False) as archive:
        belief = archive["belief"]
    assert list(belief.shape) == entry["shape"]
    position = receipt["pose"][:2]
    result = rank_frontiers(belief, entry["actual_config"]["resolution_m"], position)
    assert all(belief[cell] == -1 for _, _, cells in result for cell in cells)
    assert all(a[0] <= b[0] for a, b in zip(result, result[1:]))
    smoke.append({"layout": entry["layout"], "initial_belief_sha256": sha(belief_path),
                  "frontier_clusters": len(result),
                  "first_ranked_cluster_cells": len(result[0][2]) if result else 0})
print(json.dumps({"scope": "online search on four archived paid-action-zero observations",
                  "upstream_commit": SOURCE_SHA, "binary_sha256": sha(BINARY),
                  "results": smoke}, indent=2))
