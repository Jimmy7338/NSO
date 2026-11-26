#!/usr/bin/env python3
"""Measure development opportunities on exact task clocks and actual return."""

import csv
import importlib.util
from pathlib import Path
import sys

import numpy as np

from study import ANALYSES, FREEZE, PROTOCOL, RESULT, ROOT, read, sealed, selected, sha, write


METER = ANALYSES/"route_intervention_20261122/evaluation/evaluate.py"
SCENES = {
    "increment": ANALYSES/"region_global_revision_20261122/next/local_increment/scenes",
    "shelf": ANALYSES/"shelf_occlusion_pairs_20261122/scenes",
}


def require(condition, reason):
    if not condition:
        raise ValueError(reason)


def measurement_engine():
    loader = importlib.util.spec_from_file_location("nso_timed_opportunity_measuring", METER)
    module = importlib.util.module_from_spec(loader)
    sys.modules[loader.name] = module
    loader.loader.exec_module(module)
    weighted, metric = module.load_evaluator(module.read(module.DEFAULT_PROTOCOL))
    return module, weighted, metric


def csv_output(path, rows):
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("x", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def trace_and_motion(directory, manifest, case, row, initial, weighted):
    trace = read(directory/"steps.json")
    weighted.native.verify_trace(trace, initial, 192, manifest)
    weighted.native.verify_action_guard(trace, 192)
    with np.load(directory/"observed_belief_trace.npz", allow_pickle=False) as data:
        require(np.array_equal(data["paid"], np.arange(len(trace)+1)), "Belief lacks an actual action.")
        belief = data["belief"]
    fork = case["paid_before"]
    event = manifest["route_event"]
    count = event["actions"]
    require(event["goal"]==case["target_pose"] and event["actual_paid"]==fork+count,
            f"Route event differs from the predeclared fork: {directory}")
    steps = trace[fork:fork+count]
    require(len(steps)==count and 0<=count<=row["outbound_cost"], "Impossible bound route action count.")
    for i, step in enumerate(steps):
        require(step["action"]==row["outbound_actions"][i]
                and (step["collision"] or [*step["position"],step["heading"]]==row["outbound_states"][i+1]),
                f"Executed action violates the sealed route: {directory} paid{step['step']}")
    if event["status"]=="arrived":
        require(steps and [*steps[-1]["position"],steps[-1]["heading"]]==event["goal"]
                and not steps[-1]["collision"], "Arrival has no matching native pose.")
    return trace, belief, event


def execution(case, row, received, initial, reference, check, weighted, metric):
    directory=Path(received["directory"])
    manifest, manifest_sha=weighted.engine.verify_manifest(directory,"manifest.json",received["manifest_sha256"])
    require(manifest_sha==received["manifest_sha256"] and manifest["native_budget"]==192
            and manifest["layout"]==received["placement"] and manifest["global_mode"]=="G"
            and manifest["continuation_policy"]=="G_perf" and manifest["actual_closed_loop"]
            and manifest["initial_packet_sha256"]==initial["initial_packet_sha256"]
            and not manifest["evaluator_inputs_read_by_actor"],
            f"Non-matching native conditions: {directory}")
    trigger=read(directory/"mechanism_trigger.json")
    require(trigger["status"]=="triggered" and trigger["paid_before"]==case["paid_before"]
            and trigger["target_pose"]==case["target_pose"]
            and trigger["actual_route_id"]==row["route_id"]
            and trigger["actual_outbound_sha256"]==row["outbound_sha256"]
            and trigger["candidate_pool_sha256"]==case["candidate_pool_sha256"],
            f"Intervention identity differs from frozen bank: {directory}")
    trace, belief, route_event=trace_and_motion(directory,manifest,case,row,initial,weighted)
    run=dict(directory=directory,manifest=manifest,trace=trace,initial=initial,belief=belief)
    actual=len(trace)
    expected=({"start":0,"fork":case["paid_before"]} |
              {f"paid_{n}":n for n in read(PROTOCOL)["fixed_task_clocks"]} | {"terminal":actual})
    snapshots={}
    statuses={}
    for label, paid in expected.items():
        if paid>actual:
            statuses[label]="native_terminal_before_clock"
            continue
        if label not in manifest["checkpoints"]:
            if received["status"]=="reused_closed_loop" and label in {"paid_16","paid_24"}:
                statuses[label]="archived_closed_loop_checkpoint_not_captured"
                continue
            raise ValueError(f"Expected capture at {label}: {directory}")
        snapshots[label]=check.verify_measurement(weighted,metric,reference,run,label,paid)
        statuses[label]="measured"
    closure=weighted.native.close_record(snapshots["terminal"][0]["receipt"],
        read(directory/"runtime_summary.json"),trace,initial)
    require(all(closure[k]==manifest[k] for k in ("returned","collisions","path_length_m")),
            f"Unverified terminal accounting: {directory}")
    fork=snapshots["fork"][0]["receipt"]
    require(fork["packet_sha256"] == (case.get("fork_packet_sha256") or trigger["packet_sha256"]),
            f"Changed archived fork: {directory}")
    observations={
        "return":closure["returned"], "collisions":closure["collisions"],
        "path_m":closure["path_length_m"], "terminal_reason":closure["stop_reason"],
        "terminal_task_paid":actual, "route_status":route_event["status"],
        "route_reason":route_event["reason"], "route_end_task_paid":route_event["actual_paid"],
        "actual_bound_actions":route_event["actions"],
        "manifest_sha256":manifest_sha, "status":received["status"], "run_directory":str(directory)
    }
    return observations,snapshots,statuses


def area(mask, weights):
    return float(np.sum(weights[mask],dtype=np.float64))


def case_results(name, case, receipts, check, weighted, metric, destination):
    weighted.SCENES=SCENES[case["source"]]
    initial=read(weighted.SCENES/"prepared"/name/"initial_state.json")
    reference=weighted.load_reference(name,initial)
    weights=np.broadcast_to(np.asarray(reference["weights"]["shelf"],dtype=float),
                            reference["samples"]["shelf"].shape[:1])
    export={"reference_shelf_xyz_m":reference["samples"]["shelf"],"reference_shelf_weights_m2":weights}
    schedule=read(PROTOCOL)["fixed_task_clocks"]
    rows,scores,results=[],[],{}
    common=None
    for row in selected(name,case):
        rid=row["route_id"]
        receipt=receipts[name,rid]
        base={"placement":name,"route_id":rid,"roles":"/".join(row["roles"]),"source":case["source"],
              "fork_task_paid":case["paid_before"],"nominal_outbound_actions":row["outbound_cost"],
              "reserved_return_actions":row["reserved_return_cost"],"native_legal_count":case["n_legal_native"]}
        measured={}
        if receipt["status"]=="technical_failure":
            summary={"status":"technical_failure","exit_code":receipt["returncode"],
                     "run_directory":receipt["directory"]}
            states={}
            metrics={}
        else:
            summary,metrics,states=execution(case,row,receipt,initial,reference,check,weighted,metric)
            cold=metrics["start"][1]["shelf"]
            fork=metrics["fork"][1]["shelf"]
            identity=tuple((metrics[point][0]["mesh_sha256"],metrics[point][0]["receipt"]["belief_sha256"],
                            metrics[point][0]["receipt"]["packet_sha256"]) for point in ("start","fork"))
            if common is None:
                common=identity
                export.update(start_shelf=cold,fork_shelf=fork)
            require(identity==common and np.array_equal(export["start_shelf"],cold)
                    and np.array_equal(export["fork_shelf"],fork),"Cold and fork meshes differ across candidates.")
            for label, (geometry,masks) in metrics.items():
                mask=masks["shelf"]
                export[f"{rid}_{label}_shelf"]=mask
                gain=check.facility_metrics(cold,mask,weights)
                local_gain=check.facility_metrics(fork,mask,weights)
                measured[label]={"net_area_from_cold_m2":gain["net_m2"],
                    "net_area_from_fork_m2":local_gain["net_m2"],
                    "gross_new_cold_m2":gain["gross_new_m2"],"lost_cold_m2":gain["lost_m2"],
                    "C_free":geometry["C_free"],"F1_scene_05cm":geometry["F1_scene_05cm"],"J":geometry["J"],
                    "checkpoint_sha256":geometry["checkpoint_sha256"],"mesh_sha256":geometry["mesh_sha256"]}
        results[rid]={"outcome":summary,"snapshots":measured}
        scores.append(base | summary)
        for label in [*(f"paid_{n}" for n in schedule),"terminal"]:
            target_paid=(summary.get("terminal_task_paid") if label=="terminal" else int(label[5:]))
            status=states.get(label,"technical_failure")
            rows.append(base | summary | {"checkpoint":label,"checkpoint_status":status,
                "requested_task_paid":target_paid,"requested_after_fork":target_paid-case["paid_before"] if target_paid is not None else None,
                "measured_task_paid":target_paid if status=="measured" else None,
                "measured_after_fork":target_paid-case["paid_before"] if status=="measured" else None} |
                measured.get(label,{}))
    np.savez_compressed(destination/f"{name}_masks.npz",**export)
    return scores,rows,results,export


def opportunity(name, case, results, export, tolerance):
    direct,structure=(case["selected"][role] for role in ("D","S"))
    values=results[direct]["snapshots"]
    checkpoints={}
    for label in [*(f"paid_{n}" for n in read(PROTOCOL)["fixed_task_clocks"]),"terminal"]:
        available={rid:x["snapshots"][label]["net_area_from_cold_m2"] for rid,x in results.items()
                   if label in x["snapshots"]}
        if label=="terminal":
            comparable={rid:value for rid,value in available.items()
                        if results[rid]["outcome"].get("return") and results[direct]["outcome"].get("return")
                        and results[rid]["outcome"]["terminal_task_paid"]==results[direct]["outcome"]["terminal_task_paid"]}
        else:
            comparable=available
        incumbent=comparable.get(direct)
        best=max(comparable.values()) if comparable else None
        best_ids=sorted(rid for rid,value in comparable.items() if abs(value-best)<=tolerance)
        gain=best-incumbent if incumbent is not None else None
        checkpoints[label]={"n_frozen":len(results), "n_measured":len(available), "n_matched":len(comparable),
            "D_m2":incumbent,"S_m2":comparable.get(structure),"best_measured_m2":best,
            "best_route_ids":best_ids,"best_minus_D_m2":gain,"any_measured_above_D":gain is not None and gain>tolerance}
    release=checkpoints["paid_96"]
    best=next((r for r in release["best_route_ids"] if r!=direct),None) if release["any_measured_above_D"] else direct
    followed={"route_id":best,"terminal_minus_D_m2":None,"early_exclusive_m2":None,
              "still_exclusive_at_return_m2":None}
    if best and best!=direct and results[best]["outcome"].get("return") and results[direct]["outcome"].get("return"):
        w=export["reference_shelf_weights_m2"]
        early=export[f"{best}_paid_96_shelf"] & ~export[f"{direct}_paid_96_shelf"] & ~export["fork_shelf"]
        keep=early & export[f"{best}_terminal_shelf"] & ~export[f"{direct}_terminal_shelf"]
        followed.update(early_exclusive_m2=area(early,w),still_exclusive_at_return_m2=area(keep,w),
            best_actual_return_paid=results[best]["outcome"]["terminal_task_paid"],
            D_actual_return_paid=results[direct]["outcome"]["terminal_task_paid"],
            terminal_minus_D_m2=results[best]["snapshots"]["terminal"]["net_area_from_cold_m2"]-values["terminal"]["net_area_from_cold_m2"])
    statuses=[item["outcome"] for item in results.values()]
    return {"placement":name, "D_route_id":direct,"S_route_id":structure,"candidate_pool_size":case["n_legal_native"],
            "n_frozen":len(results),"technical_failures":sum(s["status"]=="technical_failure" for s in statuses),
            "native_incomplete_routes":sum(s.get("route_status") not in (None,"arrived") for s in statuses),
            "returned":sum(bool(s.get("return")) for s in statuses),
            "clocks":checkpoints,"A96_best_followed_to_return":followed}


def main():
    freeze,bank=sealed()
    ledger=read(RESULT/"receipt.json")
    require(ledger["freeze_sha256"]==sha(FREEZE),"Wrong execution receipt.")
    receipts={(r["placement"],r["route_id"]):r for r in ledger["outcomes"]}
    expected={(name,r["route_id"]) for name,case in bank["placements"].items() for r in selected(name,case)}
    require(receipts.keys()==expected and len(receipts)==len(ledger["outcomes"]), "Incomplete sealed execution schedule.")
    check,weighted,metric=measurement_engine()
    destination=ROOT/"results"
    destination.mkdir()
    tolerance=read(PROTOCOL)["tie_tolerance_m2"]
    runs,rows,summaries=[],[],{}
    for name in freeze["placement_order"]:
        case=bank["placements"][name]
        items,clocks,results,export=case_results(name,case,receipts,check,weighted,metric,destination)
        runs.extend(items)
        rows.extend(clocks)
        summaries[name]=opportunity(name,case,results,export,tolerance)
        print(name,summaries[name]["clocks"]["paid_96"],flush=True)
    csv_output(destination/"executions.csv",runs)
    csv_output(destination/"checkpoints.csv",rows)
    opportunity_rows=[{"placement":p,"checkpoint":c,"best_route_ids":"/".join(r["best_route_ids"]),
        **{k:v for k,v in r.items() if k!="best_route_ids"}}
        for p,summary in summaries.items() for c,r in summary["clocks"].items()]
    csv_output(destination/"opportunities.csv",opportunity_rows)
    wins=[n for n,x in summaries.items() if x["clocks"]["paid_96"]["any_measured_above_D"]]
    faults=sum(x["technical_failures"] for x in summaries.values())
    gate="stage_II" if len(wins)>=2 else "review_technical_failure" if faults else "stop_stage_I"
    write(destination/"summary.json",{"phase":"known_development_placements","scheduled_assignments":len(expected),
          "freeze_sha256":sha(FREEZE),"execution_receipt_sha256":sha(RESULT/"receipt.json"),
          "gate":gate,"positive_distinct_placements":wins,"tie_tolerance_m2":tolerance,
          "placements":summaries,"result_files_sha256":{p.name:sha(p) for p in sorted(destination.iterdir())}})


if __name__=="__main__":
    main()
