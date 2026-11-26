"""Paths and hashes shared by the static witness check and passive evaluation."""

from hashlib import file_digest
import json
from pathlib import Path


HERE = Path(__file__).resolve().parent
ANALYSES = HERE.parents[2]
OLD = ANALYSES / "region_global_revision_20261122/next/local_technical"
LOCAL = ANALYSES / "region_structure_floor_20261122/local"
ORIGINAL = LOCAL / "runs/west_screened/G"
RGB_ONLY = OLD / "west_screened_G_precision_native/native"
LAYOUT = "west_screened"
INITIAL = LOCAL / "scenes/prepared" / LAYOUT / "initial_state.json"
WEST_DEPTH = OLD / "west_screened_passive/receipt.json"
WEST_SCAN = OLD / "west_screened_G_precision_native/failure_observation_provenance.json"
OPPOSITE_SCAN = OLD / "west_screened_dual_endpoint_shadow/archived_detail.json"
FREEZE = HERE / "FROZEN_INPUTS.json"


def sha(path):
    with path.open("rb") as stream:
        return file_digest(stream, "sha256").hexdigest()


def read(path):
    return json.loads(path.read_text())


def inputs():
    mapper = Path(read(OPPOSITE_SCAN)["binding"]["source"])
    originals = [INITIAL, WEST_DEPTH, WEST_SCAN, OPPOSITE_SCAN, *OLD.glob("*.py"),
                 mapper, mapper.parent / "execution_guard_v8.py", mapper.parent / "route_coverage_v2.py"]
    for run in (ORIGINAL, RGB_ONLY):
        originals.extend(run / file for file in ("manifest.json", "steps.json", "observed_belief_trace.npz",
                                                  "initial_raw_rgbd_scan_sha256.json"))
    return sorted([*HERE.glob("*.py"), HERE / "README.md", *originals])


def verify():
    frozen = read(FREEZE)
    if frozen["scope"] != "simulation_fixed_sensor_replay_only":
        raise ValueError("Unexpected validation scope")
    if frozen["files"] != {str(path): sha(path) for path in inputs()}:
        raise ValueError("Frozen projection candidate or derived evidence hashes changed")
    return frozen


if __name__ == "__main__":
    with FREEZE.open("x") as stream:
        json.dump({"scope": "simulation_fixed_sensor_replay_only", "files": {str(path): sha(path) for path in inputs()}},
                  stream, indent=2)
        stream.write("\n")
