#!/usr/bin/env python3
"""Evaluate the frozen witness pool on a separate source-reference process."""

import csv
import importlib.util
from pathlib import Path
import sys

import numpy as np


ROOT = Path(__file__).resolve().parent
ANALYSES = ROOT.parents[2]
SCENES = ROOT/"scenes"
SOURCE = ANALYSES/"route_intervention_20261122/evaluation/evaluate.py"
from run import read, sha, verify
from seal import write


def evaluator():
    module_spec = importlib.util.spec_from_file_location("nso_increment_measurement_contract",SOURCE)
    module = importlib.util.module_from_spec(module_spec)
    sys.modules[module_spec.name] = module
    module_spec.loader.exec_module(module)
    weighted, metric = module.load_evaluator(module.read(module.DEFAULT_PROTOCOL))
    weighted.SCENES = SCENES
    return weighted,metric,module


def measure_snapshot(out, name, recorded, paid, reference, weighted, metric):
    path = out/name
    receipt = read(path/"checkpoint.json")
    assert receipt == recorded and receipt["actual_paid"] == paid
    verify({str(path/file):digest for file,digest in receipt["files_sha256"].items()})
    measured, masks = weighted.engine.evaluate_verified_snapshot(
        path/"native_mesh.ply",path/"belief.npz",receipt,reference,metric)
    return measured,masks


def measure_route(out, layout, planned, row, weighted, metric, checks, reference):
    filename = out/"manifest.json"
    if not filename.exists():
        return {"status":"technical_failure","route_id":row["route_id"],
                "exit_sha256":sha(out.parent/"exit.json") if (out.parent/"exit.json").exists() else None}
    manifest = read(filename)
    assert manifest["status"] == "controlled_route_completed" and manifest["layout"] == layout
    assert manifest["route_id"] == row["route_id"]
    assert manifest["candidate_pool_sha256"] == planned["candidate_pool_sha256"]
    assert sha(ROOT/"ROUTE_FREEZE.json") == manifest["route_seal_sha256"]
    assert manifest["fork_packet_sha256"] == planned["packet_sha256"]
    assert manifest["target_pose"] == planned["target_pose"]
    assert manifest["same_observed_prefix_verified"] and not manifest["evaluator_inputs_read_by_actor"]
    verify({str(out/name):digest for name,digest in manifest["files_sha256"].items()})
    trace = read(out/"steps.json")
    paid = manifest["actual_paid"]
    fork_paid = planned["paid_before"]
    assert [item["step"] for item in trace] == list(range(1,paid+1))
    assert trace[:fork_paid] == planned["prefix_trace"]
    measured_outbound = trace[fork_paid:]
    assert [item["action"] for item in measured_outbound] == row["outbound_actions"][:len(measured_outbound)]
    assert all(step["reserved_return_cost"] is not None
               and step["step"]+step["reserved_return_cost"] <= 193 for step in trace)
    if manifest["route_end"]["status"] == "arrived":
        assert trace[-1]["pose"] == planned["target_pose"]
    assert manifest["fork"]["native_mesh_fields"] == planned["fork_mesh_fields"]
    _, baseline = measure_snapshot(out,"fork",manifest["fork"],fork_paid,reference,weighted,metric)
    checkpoints, masks = {}, {"fork_shelf":baseline["shelf"]}
    clocks = {
        "route_end":(manifest["terminal"],paid),
        "common":(manifest["common"],planned["common_clock"]),
    }
    for name,(rec,clock) in clocks.items():
        if rec is None:
            assert name=="common" and paid<clock
            checkpoints[name] = None
            continue
        measured, visible = measure_snapshot(out,name,rec,clock,reference,weighted,metric)
        masks[name+"_shelf"] = visible["shelf"]
        values = {name:checks.facility_metrics(baseline[name],visible[name],reference["weights"][name])
                  for name in ("shelf","cabinet")}
        checkpoints[name] = {"paid":clock,"C_free":measured["C_free"],"F1_05cm":measured["F1_scene_05cm"]} | values
    return {"status":"verified_route","route_id":row["route_id"],"decision_scores_m2":row["scores"],
            "supported_hidden_m2":row["supported_hidden_m2"], "nominal_outbound":row["outbound_cost"],
            "manifest_sha256":sha(filename),"paid":paid,"route_status":manifest["route_end"],
            "path_m":manifest["path_m"],"collisions":manifest["collisions"],"metrics":checkpoints,"masks":masks}


def main():
    measure_seal = read(ROOT/"MEASUREMENT_FREEZE.json")
    verify(measure_seal["files_sha256"])
    actor = read(ROOT/"INPUT_FREEZE.json")
    verify(actor["evaluator_files_sha256"])
    plan = read(ROOT/"ROUTE_FREEZE.json")
    weighted, metric, checks = evaluator()
    result = ROOT/"results"
    result.mkdir(exist_ok=False)
    archive, rows = {}, []
    for layout, planned in plan["layouts"].items():
        if planned["status"] != "eligible_fork":
            archive[layout] = {"status":planned["status"]}
            continue
        initial = read(SCENES/"prepared"/layout/"initial_state.json")
        reference = weighted.load_reference(layout,initial)
        routes = {}
        masks = {"reference_shelf_xyz_m":reference["samples"]["shelf"],
                 "reference_shelf_weight_m2":np.asarray(reference["weights"]["shelf"])}
        for row in planned["branch_routes"]:
            name = row["route_id"]
            measurement = measure_route(ROOT/"runs"/layout/name/"run",layout,planned,row,weighted,metric,checks,reference)
            masks.update({f"{name}_{key}":value for key,value in measurement.pop("masks",{}).items()})
            routes[name] = measurement
        available = {r:v["metrics"]["common"]["shelf"]["net_m2"] for r,v in routes.items()
                     if v.get("metrics",{}).get("common") is not None}
        best = max(available.values()) if available else None
        selectors = {name:{"route_id":route,"net_shelf_m2":available.get(route),
                           "regret_to_best_measured_m2":best-available[route] if route in available else None}
                     for name,route in planned["selected"].items()}
        archive[layout] = {
            "status":"measured_subpool", "fork_paid":planned["paid_before"], "shared_paid":planned["common_clock"],
            "n_legal_native":len(planned["eligible_full_pool"]),"n_frozen_subpool":len(routes),"n_measured_at_clock":len(available),
            "upper_bound_on_regret_interpretation":"Regret is relative only to the measured members of this frozen chosen subpool.",
            "pool_best_complete":len(available)==len(routes), "best_net_shelf_m2":best,"selectors":selectors,
            "max_supported_hidden_m2":max(r["supported_hidden_m2"] for r in planned["eligible_full_pool"]),
            "routes":routes,
        }
        np.savez_compressed(result/f"{layout}_masks.npz",**masks)
        for name,data in routes.items():
            common = data.get("metrics",{}).get("common") or {}
            end = data.get("metrics",{}).get("route_end") or {}
            rows.append({
                "layout":layout, "route_id":name,
                "selectors":"/".join(k for k,v in planned["selected"].items() if v==name),
                "status":data["status"],"route_outcome":data.get("route_status",{}).get("status"),
                "actual_paid":data.get("paid"),"nominal_common_paid":planned["common_clock"],
                "shelf_net_common_m2":common.get("shelf",{}).get("net_m2"),
                "shelf_gross_common_m2":common.get("shelf",{}).get("gross_new_m2"),
                "shelf_lost_common_m2":common.get("shelf",{}).get("lost_m2"),
                "shelf_net_route_end_m2":end.get("shelf",{}).get("net_m2"),
                "supported_hidden_m2":data.get("supported_hidden_m2"),
                "path_m":data.get("path_m"),"collisions":data.get("collisions"),
            })
    write(result/"summary.json",archive)
    with (result/"routes.csv").open("w",newline="") as handle:
        if rows:
            writer=csv.DictWriter(handle,fieldnames=tuple(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    for name, row in archive.items():
        print(name,row.get("selectors"),row.get("pool_best_complete"),flush=True)


if __name__ == "__main__":
    main()
