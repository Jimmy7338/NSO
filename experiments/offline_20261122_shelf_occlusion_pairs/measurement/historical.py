#!/usr/bin/env python3
"""Recompute the two historical D-minus-S local shelf comparisons from native mesh bytes."""

import argparse
import csv
import importlib.util
import json
import math
import os
from pathlib import Path
import sys

for variable in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[variable] = "1"

import numpy as np

HERE = Path(__file__).resolve().parent
ANALYSES = HERE.parents[1]
SOURCE = ANALYSES / "route_intervention_20261122/evaluation/evaluate.py"


def checks():
    spec = importlib.util.spec_from_file_location("shelf_native_comparison_contract", SOURCE)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    weighted, metric = module.load_evaluator(module.read(module.DEFAULT_PROTOCOL))
    return module, weighted, metric


def east(check, weighted, metric):
    root = ANALYSES / "region_structure_floor_20261122/local"
    weighted.SCENES = root / "scenes"
    layout = "east_visible"
    initial = check.read(weighted.SCENES / "prepared" / layout / "initial_state.json")
    reference = weighted.load_reference(layout, initial)
    verify = {}
    for arm in ("D", "S"):
        directory = root / "runs" / layout / arm
        manifest, manifest_sha = weighted.engine.verify_manifest(directory, "manifest.json")
        trigger = check.read(directory / "mechanism_trigger.json")
        assert manifest["experimental_local_arm"] == arm
        assert trigger["status"] == "triggered" and trigger["paid_before"] == 0
        trace = check.read(directory / "steps.json")
        weighted.native.verify_trace(trace, initial, 192, manifest)
        weighted.native.verify_action_guard(trace, 192)
        with np.load(directory / "observed_belief_trace.npz", allow_pickle=False) as data:
            assert np.array_equal(data["paid"], np.arange(len(trace) + 1))
            belief = data["belief"]
        run = dict(directory=directory, manifest=manifest, trace=trace, initial=initial, belief=belief)
        before, start = check.verify_measurement(weighted, metric, reference, run, "fork", 0)
        after, end = check.verify_measurement(weighted, metric, reference, run, "paid_96", 96)
        values = check.facility_metrics(start["shelf"], end["shelf"], reference["weights"]["shelf"])
        verify[arm] = dict(net=values["net_m2"], gross=values["gross_new_m2"], lost=values["lost_m2"],
                           manifest_sha=manifest_sha, fork_mesh_sha=before["mesh_sha256"],
                           clock_mesh_sha=after["mesh_sha256"])
    assert verify["D"]["fork_mesh_sha"] == verify["S"]["fork_mesh_sha"]
    with (root / "results" / "paired.csv").open(newline="") as stream:
        matched, = [row for row in csv.DictReader(stream)
                    if row["layout"] == layout and row["pair"] == "S_minus_D"]
    delta = verify["D"]["net"] - verify["S"]["net"]
    assert math.isclose(delta, -float(matched["common_shelf_net_delta_m2"]), abs_tol=1e-12)
    return dict(layout=layout, paid=96, reference="common fork at paid 0", arms=verify,
                D_minus_S_m2=delta, previous_report_sha=check.sha(root/"results/paired.csv"))


def near_field(check, weighted, metric):
    root = ANALYSES / "region_global_revision_20261122/next/local_increment"
    weighted.SCENES = root / "scenes"
    layout = "increment_screen"
    initial = check.read(weighted.SCENES / "prepared" / layout / "initial_state.json")
    reference = weighted.load_reference(layout, initial)
    plan = check.read(root / "ROUTE_FREEZE.json")["layouts"][layout]
    result = check.read(root / "results/summary.json")[layout]
    assert result["shared_paid"] == plan["common_clock"] == 32
    assert plan["paid_before"] == 0 and result["fork_paid"] == 0
    verify = {}
    for arm in ("D", "S"):
        route = result["selectors"][arm]["route_id"]
        directory = root / "runs" / layout / route / "run"
        manifest, manifest_sha = weighted.engine.verify_manifest(directory, "manifest.json")
        assert manifest["candidate_pool_sha256"] == plan["candidate_pool_sha256"]
        assert check.sha(root / "ROUTE_FREEZE.json") == manifest["route_seal_sha256"]
        measured = {}
        for name, paid in (("fork", 0), ("common", 32)):
            path = directory / name
            receipt = check.read(path / "checkpoint.json")
            assert receipt == manifest[name] and receipt["actual_paid"] == paid
            for filename, digest in receipt["files_sha256"].items():
                assert check.sha(path / filename) == digest
            _, masks = weighted.engine.evaluate_verified_snapshot(
                path / "native_mesh.ply", path / "belief.npz", receipt, reference, metric)
            measured[name] = masks["shelf"], receipt["files_sha256"]["native_mesh.ply"]
        before, after = (measured[phase] for phase in ("fork", "common"))
        values = check.facility_metrics(before[0], after[0], reference["weights"]["shelf"])
        assert math.isclose(values["net_m2"], result["selectors"][arm]["net_shelf_m2"], abs_tol=1e-12)
        verify[arm] = dict(net=values["net_m2"], gross=values["gross_new_m2"], lost=values["lost_m2"],
                           route_id=route, manifest_sha=manifest_sha, fork_mesh_sha=before[1], clock_mesh_sha=after[1])
    assert verify["D"]["fork_mesh_sha"] == verify["S"]["fork_mesh_sha"]
    return dict(layout=layout, paid=32, reference="common fork at paid 0", arms=verify,
                D_minus_S_m2=verify["D"]["net"] - verify["S"]["net"],
                previous_report_sha=check.sha(root/"results/summary.json"))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=HERE / "historical_receipt.json")
    args = parser.parse_args()
    check, weighted, metric = checks()
    report = dict(match_distance_m=0.05, references="Native fused triangle meshes and fixed independent shelf samples",
                  prior_direct_minus_structure=[east(check, weighted, metric), near_field(check, weighted, metric)])
    text = json.dumps(report, indent=2, allow_nan=False) + "\n"
    if args.out.exists():
        assert args.out.read_text() == text, "Historical comparison differs from the stored receipt."
    else:
        args.out.write_text(text)
    print(text, end="")


if __name__ == "__main__":
    main()
