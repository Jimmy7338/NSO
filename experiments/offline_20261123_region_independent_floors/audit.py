#!/usr/bin/env python3
"""Audit the complete paired experiment against its native files and figure exports."""

import json
from math import isclose
from pathlib import Path

from paper import csv_rows, number
from study import ROOT, FREEZE, RUNS, read, sha, verify, write


def require(condition, reason):
    if not condition:
        raise ValueError(reason)


def equal(a,b):
    return a is None and b is None or a is not None and b is not None and isclose(a,b,rel_tol=0,abs_tol=1e-10)


def main():
    result=ROOT/"results/formal"
    protocol=read(ROOT/"PROTOCOL.json")
    frozen=read(FREEZE)
    known=read(ROOT/"known/audit.json")
    summary=read(result/"summary.json")
    measurement=read(result/"measurement_audit.json")
    raw_audit=read(ROOT/"evaluation/raw_qa_receipt.json")
    require(raw_audit["n_runs"]==12 and raw_audit["n_full_raw_meshes"]==72 and raw_audit["n_states"]==2316
            and raw_audit["crosscheck_output_sha256"]==sha(result/"summary.json")
            and raw_audit["paired_checkpoint_rows"]==48,"Independent raw metric reproduction is incomplete.")
    correction=read(result/"evaluation_recovery.json")
    require(correction["measurement_audit_sha256"]==sha(result/"measurement_audit.json")
            and correction["frozen_evaluator_sha256"]==sha(ROOT/"evaluation/evaluate.py")
            and correction["recovery_evaluator_sha256"]==sha(ROOT/"evaluation/evaluate_recovery.py")
            and correction["normalization"]["raw_event_data_preserved"],
            "Evaluation export correction is not independently bound to its exact inputs.")
    require(sha(FREEZE)==summary["run_freeze_sha256"]==measurement["run_freeze_sha256"],
            "Measurements came from a different scene or treatment freeze.")
    require(sha(RUNS/"phase_receipt.json")==measurement["phase_receipt_sha256"]==summary["phase_receipt_sha256"]
            and sha(result/"summary.json")==measurement["summary_sha256"],"Result/receipt ancestry differs.")
    require(measurement["results_sha256"]==summary["results_sha256"],"Result file bindings differ.")
    verify({str(result/name):digest for name,digest in summary["results_sha256"].items()})
    for key in ("actor_files","evaluator_files","simulator_only_files"):
        verify(frozen[key])
    registered=read(ROOT/"scenes/actor_freeze_holdout.json")
    reference=read(ROOT/"scenes/evaluator_freeze_holdout.json")
    pairs=[r["layout"] if isinstance(r,dict) else r for r in registered["layouts"]]
    require(pairs==protocol["layouts"] and registered["layouts"]==reference["layouts"]
            and registered["actors_run"]==reference["actors_run"]==0
            and registered["frozen_unix_s"]<=frozen["frozen_unix_s"],"Scene preregistration or chronology is absent.")
    require(not any("evaluator_ref" in p or "evaluator_truth" in p or "precomputed_assets" in p
                    for p in frozen["actor_files"]), "Offline geometry appears in actor files.")
    old=csv_rows(ROOT/"known/PAIRS.csv")
    require(known["status"]=="reproduced_from_archived_native_trajectories_and_meshes"
            and [(r["revised_A95"],r["strict_A95"]) for r in old]==[("58","112"),("81","108")]
            and all(equal(number(r,"revised_minus_strict_J96_pp"),x) for r,x in
                    zip(old,(10.341005944718518,12.409473300629392),strict=True)), "Known baseline audit incomplete.")
    ep=csv_rows(result/"episodes.csv")
    curves=csv_rows(result/"coverage_curve.csv")
    checkpoints=csv_rows(result/"checkpoints.csv")
    traces=csv_rows(result/"trajectory.csv")
    expected={(l,a) for l in protocol["layouts"] for a in protocol["arms"]}
    job_key=lambda r:(r["layout"],r["arm"])
    require(len(ep)==12 and {job_key(r) for r in ep}==expected
            and {job_key(r) for r in traces}==expected and summary["completed_native_runs"]==12,
            "Twelve declared native runs and trajectories are required.")
    checked_clocks=0
    guards=csv_rows(result/"guard_events.csv")
    repaired=[r for r in guards if r["event"]==correction["normalization"]["derived_csv_event"]]
    require(len(repaired)==5,"Previously untagged guard record count changed.")
    for event in repaired:
        original=read(event["source_file"])[int(event["source_row_index"])]
        require(json.loads(event["event_data"])==original and "event" not in original,
                "A derived guard label altered the original native event record.")
    cold={}
    for row in ep:
        case=job_key(row)
        paid=int(row["actual_paid"])
        native=Path(row["run_directory"])
        manifest=read(native/"manifest.json")
        require(sha(native/"manifest.json")==row["source_manifest_sha256"]
                and manifest["native_budget"]==192 and 0<=paid<=192
                and paid==manifest["paid_actions"],f"Native execution differs for {case}.")
        correct=sorted((r for r in curves if job_key(r)==case),key=lambda r:int(r["paid"]))
        require([int(r["paid"]) for r in correct]==list(range(paid+1)),f"Coverage clocks missing for {case}.")
        reached=[]
        for r in correct:
            hits,total=int(r["correct_free_cells"]),int(r["reachable_cells"])
            require(equal(hits/total,number(r,"C_free")),f"Invalid correct-free ratio for {case}.")
            if 100*hits>=95*total:
                reached.append(int(r["paid"]))
        require(equal(number(row,"A95_paid"),reached[0] if reached else None),f"Wrong first A95 for {case}.")
        cold.setdefault(case[0],set()).add((row["initial_packet_sha256"],row["initial_owner_sha256"],correct[0]["reachable_cells"]))
        recorded=[p for p in checkpoints if job_key(p)==case]
        require(len(recorded)==6,"Missing requested checkpoint disposition.")
        for snapshot in recorded:
            path=Path(snapshot["checkpoint_receipt_path"])
            saved=read(path)
            if snapshot["status"]!="captured":
                require(not snapshot["actual_paid"] and int(snapshot["requested_paid"])>paid,
                        f"Unsupported missing checkpoint in {case}.")
                continue
            tick=int(snapshot["actual_paid"])
            require(saved["actual_paid"]==tick and sha(path)==snapshot["source_checkpoint_sha256"],
                    f"Native checkpoint identity drift in {case}.")
            for filename,digest in saved["files_sha256"].items():
                require(sha(path.parent/filename)==digest,f"Raw measurement payload drift in {case}.")
            a,b,f,c,j=[number(snapshot,k) for k in ("p_scene_05cm","r_scene_05cm","F1_scene_05cm","C_free","J")]
            require(equal(f,2*a*b/(a+b) if a+b else 0) and equal(j,c*f)
                    and equal(c,number(correct[tick],"C_free")),f"Wrong metric relation in {case}.")
            if snapshot["checkpoint"]=="terminal":
                require(tick==paid and equal(number(row,"J"),j) and equal(number(row,"r_scene_05cm"),b),
                        f"Actual endpoint values differ in {case}.")
            checked_clocks+=1
    require(all(len(v)==1 for v in cold.values()),"Paired arms started with different sensor data or free reference.")
    require(len(csv_rows(result/"pairs.csv"))==8,"Paired results must keep both contrasts on every floor.")
    require(len(csv_rows(result/"paired_checkpoints.csv"))>=4*2*4,"Missing paired map metrics.")
    figures=[ROOT/"paper"/f for f in ("BRIEF.md","TABLES.md","CAPTION.md","coverage_joint_quality.pdf",
         "coverage_joint_quality.svg","coverage_joint_quality.png")]
    require(all(f.is_file() and f.stat().st_size>200 for f in figures),"Missing assessed paper output.")
    write(ROOT/"completion_audit.json",{"completed_verified_new_runs":len(ep),"independent_new_floors":len(pairs),
        "known_floors_remeasured_separately":len(old),"formal_summary_sha256":sha(result/"summary.json"),
        "formal_freeze_sha256":sha(FREEZE),"raw_measurement_audit_sha256":sha(result/"measurement_audit.json"),
        "executed_coverage_rows_checked":len(curves),"covered_native_checkpoints":checked_clocks,
        "guard_rows_preserved":len(guards),"guard_export_correction_sha256":sha(result/"evaluation_recovery.json"),
        "independent_raw_receipt_sha256":sha(ROOT/"evaluation/raw_qa_receipt.json"),
        "paper_files_sha256":{f.name:sha(f) for f in figures}})
    print(f"Verified {len(ep)} runs, {len(curves)} coverage clocks and {checked_clocks} native mesh receipts.",flush=True)


if __name__=="__main__":
    main()
