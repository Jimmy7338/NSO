#!/usr/bin/env python3
"""Separate source roles and seal the complete observed nominated-route bank."""

import argparse
from pathlib import Path

from run import PREVIOUS, ROOT, SCENES, read, sha, verify


def paths(files):
    return {str(path):sha(path) for path in sorted(set(files))}


def write(path, value):
    import json
    with path.open("x") as stream:
        json.dump(value,stream,indent=2,allow_nan=False)
        stream.write("\n")


def inputs():
    spec = read(ROOT/"PROTOCOL.json")
    dependency = PREVIOUS/"local/FROZEN_INPUTS.json"
    source = read(dependency)
    assert read(PREVIOUS/"local/calibration.json")["chosen_scale"] == spec["direct_calibration_scale"]
    predictors = [ROOT/name for name in ("PROTOCOL.json","run.py","seal.py","supported_hidden.py")]
    predictors += [dependency,SCENES/"scene.py",PREVIOUS/"local/calibration.json"]
    check = [SCENES/"prepare.py",SCENES/"reference_protocol.json"]
    for layout in spec["layouts"]:
        visible = SCENES/"prepared"/layout
        truth = SCENES/"evaluator_references"/layout
        initial = read(visible/"initial_state.json")
        independent = read(truth/"RECEIPT.json")
        assert initial["policy_select_calls"] == initial["paid_actions"] == 0
        assert not independent["actor_executed"] and not independent["actor_trajectory_read"]
        verify(independent["source_inputs_sha256"])
        predictors.extend(path for path in visible.iterdir() if path.is_file())
        check.extend(path for path in truth.iterdir() if path.is_file())
    actor = {path:digest for path,digest in source["actor_files_sha256"].items()
             if "scenes" not in Path(path).parts} | paths(predictors)
    simulator = paths((SCENES/"precomputed_assets").iterdir())
    evaluator = paths(check)
    verify(actor,visible=True)
    verify(simulator)
    verify(evaluator)
    write(ROOT/"INPUT_FREEZE.json",{"status":"frozen_before_candidate_branch_endpoints",
          "actor_files_sha256":actor,"simulator_cache_sha256":simulator,"evaluator_files_sha256":evaluator})


def routes():
    spec = read(ROOT/"PROTOCOL.json")
    input_seal = sha(ROOT/"INPUT_FREEZE.json")
    layouts = {}
    for layout in spec["layouts"]:
        source = ROOT/"probes"/layout/"observed_route_bank.json"
        record = read(source)
        assert record["actor_seal_sha256"] == input_seal and record["no_candidate_witness_executed"]
        if record["status"] == "eligible_fork":
            assert record["paid_before"] <= spec["max_prefix_actions"] and len(record["branch_routes"]) in (2,3,4)
        assert not (ROOT/"runs"/layout).exists()
        layouts[layout] = record | {"observed_bank_path":str(source),"observed_bank_sha256":sha(source)}
    write(ROOT/"ROUTE_FREEZE.json",{"status":"all_nominees_frozen_before_measured_witnesses",
          "inputs_sha256":input_seal,"layouts":layouts})


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase",choices=("inputs","routes"))
    action = parser.parse_args().phase
    inputs() if action=="inputs" else routes()
