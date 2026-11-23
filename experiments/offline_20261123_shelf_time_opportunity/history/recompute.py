#!/usr/bin/env python3
"""Re-evaluate archived shelf trajectories and outbound witnesses from native meshes."""

import argparse
import csv
from hashlib import file_digest
import json
from pathlib import Path
from math import fsum, isclose
import os
import sys

for variable in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[variable] = "1"
sys.dont_write_bytecode = True

import numpy as np

HERE = Path(__file__).resolve().parent
ANALYSES = HERE.parents[1]
ROOT = ANALYSES / "shelf_occlusion_pairs_20261122"
WITNESS = ROOT / "followup/local_witness"
RESULT = ROOT / "results/formal"
OLD_COMMIT = "3a0f74ff09d2d38aaa6b641392256edd2ea9f4db"

sys.path.insert(0, str(ROOT / "measurement"))
from historical import checks
from evaluate import compare_common_prefix, load_context, path_xy, write_csv
from verify_execution import verified_run


def need(condition, reason):
    if not condition:
        raise ValueError(reason)


def read(path):
    return json.loads(Path(path).read_text())


def table(path):
    with path.open(newline="") as stream:
        return list(csv.DictReader(stream))


def sha(path):
    with Path(path).open("rb") as stream:
        return file_digest(stream, "sha256").hexdigest()


def close(measured, archived, context):
    if measured is None or archived in (None, ""):
        need(measured is None and archived in (None, ""), f"Missingness differs: {context}")
    else:
        need(isclose(measured, float(archived), rel_tol=0, abs_tol=1e-12), f"Value differs: {context}")


def bindings(paths):
    return {str(Path(p).resolve().relative_to(ANALYSES)): sha(p) for p in paths}


def weighted_net(weights, cold, current):
    return fsum(weights[current & ~cold]) - fsum(weights[cold & ~current])


def first_trace_difference(left, right):
    for a, b in zip(left, right):
        if (a["action"], a["position"], a["heading"]) != (b["action"], b["position"], b["heading"]):
            return a["step"]
    return None


def formal():
    contract, weighted, metric = checks()
    meta = load_context("formal", None, contract)
    completion = read(ROOT / "completion_audit.json")
    summary = read(RESULT / "summary.json")
    need(completion["formal_summary_sha256"] == sha(RESULT / "summary.json")
         and completion["formal_freeze_sha256"] == meta["freeze_sha"], "Formal release drift")
    for name, expected in summary["table_sha256"].items():
        need(sha(RESULT / name) == expected, f"Formal table drift: {name}")
    pair_table = {r["unit"]:r for r in table(RESULT / "pairs.csv")}
    episodes = {(r["placement"], r["arm"]):r for r in table(RESULT / "episodes.csv")}
    weighted.SCENES = Path(meta["spec"]["scene_root"])
    rows, trace_index = [], {}
    for parent in meta["parents"]:
        placement = parent + "_screen"
        initial_file = weighted.SCENES / "prepared" / placement / "initial_state.json"
        initial = read(initial_file)
        ref = weighted.load_reference(placement, initial)
        receipt_path = meta["receipts"][placement]
        receipt = read(receipt_path)
        mask_record = summary["masks"][placement]
        mask_path = RESULT / mask_record["artifact"]
        need(sha(mask_path) == mask_record["sha256"], f"Sample archive mismatch {placement}")
        runs, summary_rows, arm_index = {}, {}, {}
        with np.load(mask_path, allow_pickle=False) as stored:
            points = ref["samples"]["shelf"]
            weights = stored["reference_shelf_weights_m2"]
            need(np.array_equal(points, stored["reference_shelf_xyz_m"])
                 and np.array_equal(weights, np.broadcast_to(ref["weights"]["shelf"], (len(points),))),
                 f"Independent room samples drifted: {placement}")
            for arm in ("D", "S"):
                assignment = receipt["arms"][arm]
                original = episodes[placement, arm]
                directory = Path(assignment["run_directory"])
                need(original["manifest_sha256"] == assignment["manifest_sha256"]
                     and original["route_identity_sha256"] == assignment["route_identity_sha256"],
                     f"Episode table and sealed assignment differ in {placement}/{arm}")
                measured = verified_run(directory, assignment, meta, initial, ref, contract, weighted, metric, False)
                runs[arm] = measured
                stage_row = measured["bank"]["arms"][arm]
                need(stage_row["route_id"] == original["selected_route_id"]
                     and stage_row["route_identity_sha256"] == assignment["route_identity_sha256"],
                     f"Scored route identity changed in {placement}/{arm}")
                trace = measured["trace"]
                need(np.array_equal(np.arange(len(trace)+1), stored[f"{arm}_trajectory_paid"])
                     and np.array_equal(path_xy(trace, initial), stored[f"{arm}_trajectory_xy"])
                     and np.array_equal(measured["snapshots"]["start"][1]["shelf"], stored["start_shelf"]),
                     f"Stored trajectory or common cold shelf changed in {placement}/{arm}")
                start = stored["start_shelf"]
                values, captures = {}, {}
                for label in (48, 96, 144, 192, "terminal"):
                    name = "terminal" if label == "terminal" else f"paid_{label}"
                    snapshot, masks = measured["snapshots"][name]
                    need(np.array_equal(masks["shelf"], stored[f"{arm}_{name}_shelf"]),
                         f"Archived shelf mask differs from original triangle mesh: {placement}/{arm}/{name}")
                    value = weighted_net(weights, start, masks["shelf"])
                    need(int(original[f"actual_paid_{label}"]) == snapshot["receipt"]["actual_paid"],
                         f"Wrong native paid clock in {placement}/{arm}/{name}")
                    close(value, original[f"net_{label}_m2"], f"{placement}/{arm}/{name}")
                    values[label] = value
                    captures[name] = {"paid":snapshot["receipt"]["actual_paid"],
                                      "checkpoint_sha256":snapshot["checkpoint_sha256"],
                                      "mesh_sha256":snapshot["mesh_sha256"]}
                summary_rows[arm] = values
                arm_index[arm] = {"execution_arm":assignment["execution_arm"],
                                  "run_path":str(directory.relative_to(ANALYSES)),
                                  "manifest_sha256":measured["manifest_sha256"],
                                  "steps_sha256":sha(directory/"steps.json"),
                                  "fork_bank_sha256":assignment["fork_bank_sha256"],
                                  "route_id":stage_row["route_id"], "snapshots":captures}
        compare_common_prefix(runs["D"], runs["S"], False)
        trace_index[placement] = {"source_receipt_sha256":sha(receipt_path), "mask_sha256":sha(mask_path),
                                  "initial_sha256":sha(initial_file), "arms":arm_index}
        diff = {label:summary_rows["S"][label]-summary_rows["D"][label] for label in (48,96,144,192,"terminal")}
        for label, delta in diff.items():
            close(delta, pair_table[parent][f"screen_S_minus_D_{label}_m2"], f"{placement}/{label}")
        rows.append({"unit":parent,"placement":placement,"fork_task_paid":runs["D"]["bank"]["paid_before"],
                     "target_paid":96,"target_paid_since_fork":96-runs["D"]["bank"]["paid_before"],
                     "D_route_id":arm_index["D"]["route_id"],"S_route_id":arm_index["S"]["route_id"],
                     "first_motion_difference_task_paid":first_trace_difference(runs["D"]["trace"], runs["S"]["trace"]),
                     "D_outbound_status":runs["D"]["manifest"]["route_event"]["status"],
                     "S_outbound_status":runs["S"]["manifest"]["route_event"]["status"],
                     "D_outbound_task_paid":runs["D"]["manifest"]["route_event"]["actual_paid"],
                     "S_outbound_task_paid":runs["S"]["manifest"]["route_event"]["actual_paid"],
                     "D_returned":runs["D"]["closure"]["returned"],"S_returned":runs["S"]["closure"]["returned"],
                     "D_terminal_task_paid":runs["D"]["closure"]["actual_paid"],
                     "S_terminal_task_paid":runs["S"]["closure"]["actual_paid"],
                     "D_net96_m2":summary_rows["D"][96],"S_net96_m2":summary_rows["S"][96],
                     "S_minus_D_96_m2":diff[96],"two_record_hindsight_gain_vs_D_96_m2":max(0.,diff[96]),
                     "S_minus_D_48_m2":diff[48],"S_minus_D_144_m2":diff[144],"S_minus_D_at_return_m2":diff["terminal"]})
        print(placement, f"S−D A96 {diff[96]:.9f}", flush=True)
    mean = fsum(r["S_minus_D_96_m2"] for r in rows)/len(rows)
    hindsight = fsum(r["two_record_hindsight_gain_vs_D_96_m2"] for r in rows)/len(rows)
    need(len(rows)==6 and round(mean,5)==-.24130 and round(hindsight,5)==.00830, "Historical expected values drifted")
    return rows, {"n_paired_units":len(rows), "S_minus_D_mean_96_m2":mean,
                  "two_record_hindsight_gain_vs_D_mean_96_m2":hindsight,
                  "tie_tolerance_m2":meta["tolerance"], "formal_runs":trace_index}


def short_witness():
    sys.path.insert(0, str(WITNESS))
    import witness

    sealed = witness.sealed()
    witness.verify_bound(sealed["evaluator_files_sha256"])
    witness.verify_bound(read(witness.ORIGINAL_INPUTS)["evaluator_files_sha256"])
    old_meter = witness.spawn_archived_module("measure")
    weighted, metric, checks = old_meter.evaluator()
    execution = read(WITNESS / "execution_receipt.json")
    summary = read(WITNESS / "results/summary.json")
    archived_routes = WITNESS / "results/routes.csv"
    need(summary["freeze_sha256"] == execution["freeze_sha256"] == sha(witness.FREEZE)
         and summary["execution_receipt_sha256"] == sha(WITNESS / "execution_receipt.json")
         and summary["routes_csv_sha256"] == sha(archived_routes), "Archived short-branch release drift")
    registry={(r["layout"],r["route_id"]):r for r in execution["outcomes"]}
    prior = {(r["placement"],r["route_id"],r["clock"]):r for r in table(archived_routes)}
    plan = read(witness.ORIGINAL_BANK)["layouts"]
    measurements, aggregate, index = [], [], {}
    for placement, case in sealed["placements"].items():
        original = plan[placement]
        initial = read(old_meter.SCENES / "prepared" / placement / "initial_state.json")
        reference = weighted.load_reference(placement, initial)
        candidates={r["route_id"]:r for r in original["eligible_full_pool"]}
        fork=case["fork_paid"]
        scheduled=case["fixed_paid_clocks"] | {"nominal_min":case["nominal_min_clock"],
                                               "outbound_event":None,"paid_96":96,"actual_return":None}
        for selected in case["routes"]:
            route_id=selected["route_id"]
            record=registry[placement,route_id]
            directory=Path(record["directory"])
            witness.verify_bound(record["attempt_exit_sha256"] | {str(directory.parent/"worker.log"):record["worker_log_sha256"]})
            stamp_path=directory/"witness_receipt.json"
            proof=read(stamp_path)
            manifest_path=directory/"manifest.json"
            need(sha(stamp_path)==record["witness_receipt_sha256"] and proof["native_manifest_sha256"]==sha(manifest_path),
                 f"Witness receipt and native run identity differ: {placement}/{route_id}")
            run=read(manifest_path)
            old=old_meter.measure_route(directory, placement, original, candidates[route_id], weighted, metric, checks, reference)
            task_paid=run["actual_paid"]
            _, baseline=old_meter.measure_snapshot(directory,"fork",run["fork"],fork,reference,weighted,metric)
            index[f"{placement}/{route_id}"]={"run_path":str(directory.relative_to(ANALYSES)),
                                              "manifest_sha256":sha(manifest_path),
                                              "steps_sha256":sha(directory/"steps.json"),
                                              "witness_receipt_sha256":sha(stamp_path),
                                              "terminal_clock_paid":task_paid,
                                              "route_status":run["route_end"]["status"],
                                              "route_reason":run["route_end"]["reason"],
                                              "return_executed":False}
            reported={"outbound_event":old["metrics"]["route_end"],"nominal_min":old["metrics"]["common"]}
            for label, paid in case["fixed_paid_clocks"].items():
                signed=proof["paid_checkpoints"][label]
                need(signed["requested_paid"]==paid, f"Wrong witness paid clock: {placement}/{route_id}/{label}")
                if signed["status"]!="measured":
                    need(task_paid < paid, f"Paid witness frame exists but was dropped: {placement}/{route_id}/{label}")
                    reported[label]=None
                    continue
                capture=directory/label/"checkpoint.json"
                need(sha(capture)==signed["checkpoint_sha256"], f"Witness snapshot changed: {placement}/{route_id}/{label}")
                geo, match=old_meter.measure_snapshot(directory,label,read(capture),paid,reference,weighted,metric)
                reported[label]={"paid":paid,"C_free":geo["C_free"],"F1_05cm":geo["F1_scene_05cm"],
                                 "shelf":checks.facility_metrics(baseline["shelf"],match["shelf"],reference["weights"]["shelf"])}
            stopped=run["route_end"]["status"]=="native_cancelled"
            for label, requested in scheduled.items():
                value=reported.get(label)
                actual=value["paid"] if value else None
                net=value["shelf"]["net_m2"] if value else None
                if value:
                    status="measured"
                elif label in ("paid_96","actual_return"):
                    status="not_scheduled_outbound_only"
                else:
                    status="guard_cancelled_before_clock" if stopped else "other_native_stop_before_clock"
                if label in reported:
                    signed=prior[placement,route_id,label]
                    close(net,signed["shelf_net_m2"],f"{placement}/{route_id}/{label}")
                    need(status==signed["measurement_status"] and task_paid==int(signed["actual_paid"]),
                         f"Witness guard/clock does not match released table: {placement}/{route_id}/{label}")
                measurements.append({"placement":placement,"route_id":route_id,"clock":label,
                                     "fork_task_paid":fork,"requested_task_paid":requested,
                                     "requested_paid_since_fork":requested-fork if requested is not None else None,
                                     "measured_task_paid":actual,
                                     "measured_paid_since_fork":actual-fork if actual is not None else None,
                                     "route_end_task_paid":task_paid,"route_end_paid_since_fork":task_paid-fork,
                                     "measurement_status":status,"route_end_status":run["route_end"]["status"],
                                     "route_end_reason":run["route_end"]["reason"],"return_executed":False,
                                     "geometry_witness":selected["geometric_witness"],
                                     "policy_nomination":"/".join(selected["policies"]),"shelf_net_m2":net})
        for label, wanted in scheduled.items():
            cohort=[r for r in measurements if r["placement"]==placement and r["clock"]==label]
            available=[r for r in cohort if r["measurement_status"]=="measured"]
            d_route=case["policies"]["D"]
            d=next((r for r in available if r["route_id"]==d_route),None)
            best=max((r["shelf_net_m2"] for r in available),default=None) if wanted is not None else None
            measured_clocks=sorted({r["measured_task_paid"] for r in available})
            fixed_regret=best-d["shelf_net_m2"] if best is not None and d else None
            stops=sum(r["measurement_status"]=="guard_cancelled_before_clock" for r in cohort)
            if label in summary["placements"][placement]["checkpoints"]:
                archived=summary["placements"][placement]["checkpoints"][label]
                need((len(available), stops)==(archived["measured"],archived["guard_cancelled_before_clock"]),
                     f"Witness aggregate completeness differs: {placement}/{label}")
                close(fixed_regret,archived["policies"]["D"]["regret_to_measured_union_best_m2"],f"{placement}/{label}/regret")
            aggregate.append({"placement":placement,"clock":label,"fork_task_paid":fork,
                              "requested_task_paid":wanted,"requested_paid_since_fork":wanted-fork if wanted is not None else None,
                              "measured_task_clocks":measured_clocks,"measured_clocks_since_fork":[x-fork for x in measured_clocks],
                              "legal_native_routes":case["n_legal_native"],"archive_subset_routes":len(cohort),
                              "measured_routes":len(available),"missing_routes":len(cohort)-len(available),
                              "guard_cancelled_before_fixed_clock":stops,
                              "guard_cancelled_outbound_events":sum(r["route_end_status"]=="native_cancelled" for r in cohort),
                              "outbound_only_not_scheduled":sum(r["measurement_status"]=="not_scheduled_outbound_only" for r in cohort),
                              "other_stops_before_clock":sum(r["measurement_status"]=="other_native_stop_before_clock" for r in cohort),
                              "D_route_id":d_route,"D_net_m2_at_recorded_clock":d["shelf_net_m2"] if d else None,
                              "best_measured_equal_clock_m2":best,"best_measured_equal_clock_minus_D_m2":fixed_regret})
        print(placement, len(case["routes"]), "verified outbound routes",flush=True)
    chosen=[r for r in aggregate if r["clock"]=="paid_16"]
    need(len(index)==13 and sum(r["measured_routes"] for r in chosen)==13
         and all(r["best_measured_equal_clock_minus_D_m2"]==0 for r in chosen), "Historical short baseline differs")
    return measurements, aggregate, index


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out",type=Path,default=HERE)
    args=parser.parse_args()
    args.out.mkdir(exist_ok=True,parents=True)
    publication=read(ROOT/"publication/receipt.json")
    need(publication["head"]==OLD_COMMIT,"Wrong archived parent commit")
    pair_rows, result=formal()
    write_csv(args.out/"formal_screened_pairs.csv",pair_rows)
    witness_rows, witness_clocks, witness_index=short_witness()
    write_csv(args.out/"short_witness_routes.csv",witness_rows)
    write_csv(args.out/"short_witness_clocks.csv",witness_clocks)
    source_paths=[ROOT/"publication/receipt.json", ROOT/"FORMAL_FREEZE.json",ROOT/"completion_audit.json",
                  ROOT/"measurement/development_repeatability.json",RESULT/"summary.json",RESULT/"pairs.csv",RESULT/"episodes.csv",
                  WITNESS/"WITNESS_FREEZE.json",WITNESS/"execution_receipt.json", WITNESS/"results/summary.json",
                  WITNESS/"results/routes.csv", ROOT/"followup/local_scoring_diagnostic/formal/summary.json"]
    result.update(source_commit=OLD_COMMIT, match_distance_m=.05, sources_sha256=bindings(source_paths),
                  witness_runs=witness_index, witness_clock_rows=witness_clocks,
                  audit_source_sha256=sha(Path(__file__)),
                  outputs_sha256={name:sha(args.out/name) for name in
                                  ("formal_screened_pairs.csv","short_witness_routes.csv","short_witness_clocks.csv")})
    target=args.out/"historical_receipt.json"
    temporary=target.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(result,ensure_ascii=False,indent=2,allow_nan=False)+"\n")
    temporary.replace(target)
    print(f"Archived screened S−D A96 {result['S_minus_D_mean_96_m2']:+.9f} m², "
          f"two-record hindsight {result['two_record_hindsight_gain_vs_D_mean_96_m2']:+.9f} m², "
          f"{len(witness_index)} checked short routes",flush=True)


if __name__=="__main__":
    main()
