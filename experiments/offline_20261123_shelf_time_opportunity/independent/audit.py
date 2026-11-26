#!/usr/bin/env python3
"""Independently audit Stage I execution, meshes, matched opportunities and gate."""

import argparse
import csv
from hashlib import file_digest, sha256
import json
from math import fsum, isclose
import os
from pathlib import Path
import sys

for var in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[var] = "1"
sys.dont_write_bytecode = True

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
ANALYSES = ROOT.parent
SHELF = ANALYSES / "shelf_occlusion_pairs_20261122"
INCREMENT = ANALYSES / "region_global_revision_20261122/next/local_increment"
SCENES = {name:p/"scenes" for name,p in (("shelf",SHELF),("increment",INCREMENT))}
IDENTITY = ("outbound_states","outbound_actions","return_states","return_actions","oriented_target")

sys.path.insert(0,str(SHELF/"measurement"))
from historical import checks


def require(value, message):
    if not value:
        raise ValueError(message)


def load(file):
    return json.loads(Path(file).read_text())


def sha(file):
    with Path(file).open("rb") as stream:
        return file_digest(stream,"sha256").hexdigest()


def digest(value):
    return sha256(json.dumps(value,sort_keys=True,separators=(",",":"),allow_nan=False).encode()).hexdigest()


def verify(files):
    for file, expected in files.items():
        require(sha(file)==expected,f"Frozen bytes changed: {file}")


def table(path):
    with path.open(newline="") as stream:
        return list(csv.DictReader(stream))


def same_num(actual, recorded, reason):
    require(actual is None and recorded in (None,"") or actual is not None and recorded not in (None,"")
            and isclose(float(actual),float(recorded),rel_tol=0,abs_tol=1e-12),reason)


def relative(file):
    return str(Path(file).resolve().relative_to(ANALYSES))


def previous_attempts(stage_info):
    original=ROOT/"DEVELOPMENT_FREEZE.json"
    old_ledger=ROOT/"execution/receipt.json"
    require(stage_info["original_sha256"]==sha(original) and stage_info["original_attempts_sha256"]==sha(old_ledger),
            "Previous recovery version does not bind original attempts")
    first=load(old_ledger)
    require(first["freeze_sha256"]==sha(original) and len(first["outcomes"])==12,"Original registry drift")
    files={relative(old_ledger):sha(old_ledger),relative(original):sha(original)}
    failures=0
    for out in first["outcomes"]:
        require(out["status"] in ("technical_failure","reused_closed_loop"),"An original scientific run was replaced")
        if out["status"]!="technical_failure":
            continue
        for name,h in out["attempt_exits_sha256"].items():
            require(sha(name)==h,"Original failure receipt changed")
            summary=load(name)
            trace=Path(summary["directory"])/"steps.json"
            require(summary["status"]=="technical_failure" and (not trace.exists() or not load(trace)),
                    "An original technical attempt contains an executed action")
            files[relative(name)]=h
            failures+=1
    return {"failed_technical_attempts":failures,"files_sha256":files}


def reference_for(case, name, weighted):
    scene = SCENES[case["source"]]
    actor_seal=Path(case["source_input_freeze_path"])
    require(sha(actor_seal)==case["source_input_freeze_sha256"],f"Old reference registry drift: {name}")
    registry=load(actor_seal)
    base=scene/"evaluator_references"/name
    paths={p:h for p,h in registry["evaluator_files_sha256"].items() if Path(p).parent==base}
    expected={base/p for p in ("RECEIPT.json","evaluator_truth.npz","fixed_reference.npz","source_mesh.ply")}
    require(expected <= {Path(p) for p in paths},f"Missing frozen reference evidence: {name}")
    verify(paths)
    cold=scene/"prepared"/name/"initial_state.json"
    require(registry["actor_files_sha256"].get(str(cold))==sha(cold),f"Frozen actor cold state drift: {name}")
    initial=load(cold)
    require(initial["initial_packet_sha256"]==case["cold_packet_sha256"],f"Wrong cold packet: {name}")
    weighted.SCENES=scene
    return initial,weighted.load_reference(name,initial),{relative(p):h for p,h in paths.items()} | {relative(cold):sha(cold)}


def attest_execution(receive, route, case, stage, stage_sha):
    directory=Path(receive["directory"]).resolve()
    reuse=route["closure_reuse"]
    require(receive["freeze_sha256"]==stage_sha,f"Wrong supervisor seal: {directory}")
    if reuse:
        require(receive["status"]=="reused_closed_loop" and Path(reuse["run_directory"]).resolve()==directory
                and directory.is_relative_to(SHELF/"execution/formal")
                and receive["manifest_sha256"]==reuse["manifest_sha256"],f"Unsealed reuse: {directory}")
        receipt=Path(reuse["receipt_path"])
        verify({receipt:reuse["receipt_sha256"],directory/"fork_bank.json":reuse["fork_bank_sha256"]})
        arm=next(k for k in route["roles"] if k in ("D","S"))
        published=load(receipt)["arms"][arm]
        require(published["run_directory"]==str(directory) and published["manifest_sha256"]==reuse["manifest_sha256"],
                f"Reused route has no exact published assignment: {directory}")
        bank=load(directory/"fork_bank.json")
        signed=bank["arms"][arm]
        for key in ("outbound_states","outbound_actions","return_states","return_actions","route_identity_sha256"):
            require(signed[key]==route[key],f"Reused route changes {key}: {directory}")
        require(bank["target_pose"]==route["oriented_target"] and bank["shared_identity_sha256"]==reuse["shared_identity_sha256"]
                and bank["continuation_id"]==reuse["continuation_id"],f"Reused observed fork/continuation differs: {directory}")
        return directory,{relative(receipt):sha(receipt),relative(directory/"fork_bank.json"):sha(directory/"fork_bank.json")}

    require(directory.is_relative_to(stage/"execution"/receive["placement"]/route["route_id"]),
            f"New execution path escaped its stage placement: {directory}")
    exits=receive["attempt_exits_sha256"]
    verify(exits | receive["attempt_sha256"])
    saved=directory.parent/"exit.json"
    require(str(saved) in exits and load(saved)=={k:v for k,v in receive.items() if k!="attempt_exits_sha256"},
            f"Terminal supervisor log did not match recorded receipt: {directory}")
    launch=load(directory.parent/"launch.json")
    stage_time=load(stage/"DEVELOPMENT_FREEZE.json")["frozen_unix_s"]
    require(all(launch[k]==receive[k] for k in launch) and launch["freeze_sha256"]==stage_sha
            and launch["start_unix_s"]>=stage_time and sha(directory.parent/"worker.log")==receive["log_sha256"],
            f"Launch or process log drift: {directory}")
    proof={relative(k):v for k,v in (exits | receive["attempt_sha256"]).items()}
    proof[relative(directory.parent/"worker.log")]=receive["log_sha256"]
    if receive["status"]=="technical_failure":
        require(not (directory/"manifest.json").exists() or receive["returncode"]!=0,
                f"Successful branch classified as a technical failure: {directory}")
        return directory,proof
    require(receive["status"]=="new_execution" and receive["returncode"]==0, f"Unsuccessful sealed branch: {directory}")
    branch_path=directory/"branch_receipt.json"
    branch=load(branch_path)
    require(branch["bank_sha256"]==sha(ROOT/"bank/BANK_FREEZE.json") and branch["route"]==route["route_id"]
            and branch["route_identity_sha256"]==route["route_identity_sha256"]
            and branch["source_actor_freeze_sha256"]==case["source_input_freeze_sha256"]
            and branch["source_bank_sha256"]==case["source_bank_sha256"]
            and branch["exact_guard"] and branch["original_global_target_retained"]
            and not branch["evaluator_inputs_read_by_actor"],f"New branch provenance differs: {directory}")
    fork=load(directory/"fork_bank.json")
    require(fork["return_actions"]==route["return_actions"] and fork["return_states"]==route["return_states"]
            and fork["route_identity_sha256"]==route["route_identity_sha256"],f"New branch changed reserve: {directory}")
    if route["archived_short"]:
        bound=route["archived_short"]
        existing=Path(bound["run"] if "run" in bound else bound.get("run_directory",bound.get("directory")))
        actual=branch["archived_short_replay"]
        verify({existing/"manifest.json":bound["manifest_sha256"],existing/"steps.json":bound["archived_trace_sha256"]})
        require(actual["manifest_sha256"]==bound["manifest_sha256"] and actual["steps_sha256"]==bound["archived_trace_sha256"]
                and actual["first_to_last_archived_motion_matches"] and actual["native_route_event_matches"]
                and actual["matched_charged_actions"]==len(load(existing/"steps.json")),
                f"Archived short replay does not cover the full old execution: {directory}")
        proof |= {relative(existing/"manifest.json"):bound["manifest_sha256"],relative(existing/"steps.json"):bound["archived_trace_sha256"]}
    proof[relative(branch_path)]=sha(branch_path)
    return directory,proof


def decide(directory, route, case, receive, manifest, trace):
    fields=("outbound_states","outbound_actions","return_states","return_actions")
    trigger=load(directory/"mechanism_trigger.json")
    local=load(directory/"local_choices.json")
    globals_by_id={e["call_id"]:e for e in load(directory/"global_choices.json")}
    modes=[e["choice_audit"]["selected_mode"] for e in local]
    require(len(globals_by_id)==manifest["native_global_choices"] and len(local)==manifest["local_route_choices"]
            and all(e["outputs"]["mode"]=="G" for e in globals_by_id.values()),f"Wrong neutral global record: {directory}")
    reused=receive["status"]=="reused_closed_loop"
    flagged=[i for i,e in enumerate(local) if ("paired_first_decision_arm" in e["choice_audit"] if reused
                                             else e["choice_audit"]["selected_mode"]=="sealed_outbound")]
    require(len(flagged)==1 and all(mode=="G" for i,mode in enumerate(modes) if i!=flagged[0]),
            f"One-shot selection or G continuation failed: {directory}")
    for i,entry in enumerate(local):
        if i==flagged[0]:
            continue
        weights=entry["choice_audit"]["weighted_ranking"]
        require(weights["prior"]=="G" and weights["unknown_weight"]==.25 and weights["action_cost"]==.20
                and not weights["choice_prior_uses_category"]
                and all(weights["local_area_provider_calls"][key]==0 for key in ("S","X")),
                f"Neutral G continuation or shared weights differ at task A{entry['paid_before']}: {directory}")
    e=local[flagged[0]]
    global_selected=globals_by_id[e["selection_call_id"]]["outputs"]["selected"]
    require(e["paid_before"]==trigger["paid_before"]==case["paid_before"]
            and e["remaining_budget"]==192-case["paid_before"]
            and trigger["candidate_pool_sha256"]==e["choice_audit"]["candidate_pool_sha256"]==case["candidate_pool_sha256"]
            and trigger["packet_sha256"]==e["packet_sha256"]==case["fork_packet_sha256"]
            and global_selected["pose"]==e["same_global_target"]==trigger["target_pose"]==case["target_pose"]
            and e["option_id"]==trigger["option_id"] and e["active_local_route_id"]==trigger["actual_route_id"]==route["route_id"]
            and e["active_outbound_sha256"]==trigger["actual_outbound_sha256"]==route["outbound_sha256"],
            f"Serialized intervention omitted or changed its fork/target/route: {directory}")
    actual={name:e["selected_"+name] if name.startswith("outbound") else e["rebound_"+name] for name in fields}
    actual["oriented_target"]=e["same_global_target"]
    require(actual=={k:route[k] for k in IDENTITY} and digest(actual)==route["route_identity_sha256"]
            and global_selected["return_actions"]==e["native_return_actions"]==route["return_actions"]
            and global_selected["return_states"]==e["native_return_states"]==route["return_states"],
            f"Sourced local/native return certificate failed: {directory}")
    body=[]
    for step in trace[case["paid_before"]:]:
        if step["active_option_id"]!=trigger["option_id"]:
            break
        i=len(body)
        require(i<len(route["outbound_actions"]) and step["action"]==route["outbound_actions"][i]
                and step["guard_allowed"] and step["local_route_id"]==route["route_id"]
                and step["route_action_index_zero_based"]==i and step["active_outbound_sha256"]==route["outbound_sha256"]
                and (step["collision"] or [*step["position"],step["heading"]]==route["outbound_states"][i+1]),
                f"Native trace departed the assigned route at A{step['step']}: {directory}")
        body.append(step)
        if step["collision"] or step["matched_target_pose_after_step"]:
            break
    event=manifest["route_event"]
    require(event["actual_paid"]==case["paid_before"]+len(body) and event["actions"]==len(body)
            and event["goal"]==route["oriented_target"],f"Route event is inconsistent with charged motion: {directory}")
    last=body[-1] if body else None
    if event["status"]=="arrived":
        require(last is not None and last["matched_target_pose_after_step"] and not last["collision"],
                f"Filed arrival lacks the actual state: {directory}")
    elif event["status"]=="collision":
        require(last is not None and last["collision"],f"Filed collision lacks the action: {directory}")
    else:
        require(last is None or not(last["matched_target_pose_after_step"] or last["collision"]),
                f"Incomplete event conflicts with charged motion: {directory}")
    return event,len(local)-1


def audit_short_prefix(route, trace, event, belief, directory):
    short=route["archived_short"]
    if not short:
        return None
    before=Path(short["run_directory"])
    old=load(before/"steps.json")
    old_outcome=load(before/"manifest.json")["route_end"]
    native_status={"native_cancelled":"guard_cancelled", "native_terminal":"terminal_before_arrival"}
    require(event["actual_paid"]==old_outcome["actual_paid"]==len(old)
            and event["status"]==native_status.get(old_outcome["status"],old_outcome["status"])
            and event["reason"]==old_outcome["reason"], f"Prior short route event did not replay: {directory}")
    keys=("step","packet_sha256","depth_values_sha256","rgb_values_sha256","scan_ranges_sha256",
          "action","collision","guard_reason","reserved_return_cost","owner_sha256")
    for a,b in zip(old,trace[:len(old)],strict=True):
        current={k:b[k] for k in keys}
        current.update(pose=[*b["position"],b["heading"]],path_m=b["path_length_m"],option_id=b["active_option_id"],
                       map_sha256=sha256(np.ascontiguousarray(belief[b["step"]]).tobytes()).hexdigest())
        require(a==current,f"Recorded raw observation, action or map changed at A{b['step']}: {directory}")
    return len(old)


def evaluate_record(name, case, route, receive, stage, stage_sha, initial, reference, weights, archive, check, weighted, metric):
    directory,proof=attest_execution(receive,route,case,stage,stage_sha)
    rid=route["route_id"]
    record={"placement":name,"route_id":rid,"roles":route["roles"],"execution_status":receive["status"],
            "run_path":relative(directory),"bound_receipts_sha256":proof,"clocks":{}}
    if receive["status"]=="technical_failure":
        return record,None,None
    manifest,mhash=weighted.engine.verify_manifest(directory,"manifest.json",receive["manifest_sha256"])
    require(manifest["layout"]==name and manifest["initial_packet_sha256"]==case["cold_packet_sha256"]
            and manifest["native_budget"]==192 and manifest["global_mode"]=="G"
            and manifest["continuation_policy"]=="G_perf" and manifest["actual_closed_loop"]
            and not manifest["evaluator_inputs_read_by_actor"],f"Other actor/settings used: {directory}")
    trace=load(directory/"steps.json")
    weighted.native.verify_trace(trace, initial, 192, manifest)
    weighted.native.verify_action_guard(trace,192)
    event,neutral_calls=decide(directory,route,case,receive,manifest,trace)
    with np.load(directory/"observed_belief_trace.npz",allow_pickle=False) as data:
        require(np.array_equal(data["paid"],np.arange(len(trace)+1)),f"Unpaired belief clocks: {directory}")
        belief=data["belief"]
    verified_prior_charged_actions=audit_short_prefix(route,trace,event,belief,directory)
    run={"directory":directory,"manifest":manifest,"trace":trace,"initial":initial,"belief":belief}
    raw={}
    capture={"start":0,"fork":case["paid_before"],"terminal":len(trace)} | {f"paid_{n}":n for n in (16,24,48,96,144,192)}
    for label,paid in capture.items():
        if paid>len(trace):
            require(label in manifest["checkpoints"] and manifest["checkpoints"][label]["status"]=="unavailable"
                    and f"{rid}_{label}_shelf" not in archive,f"Unavailable clock imputed or missing receipt: {directory}/{label}")
            record["clocks"][label]={"status":"native_terminal_before_clock","requested_task_paid":paid}
            continue
        if label not in manifest["checkpoints"]:
            require(receive["status"]=="reused_closed_loop" and label in ("paid_16","paid_24")
                    and f"{rid}_{label}_shelf" not in archive,f"Missing achievable checkpoint: {directory}/{label}")
            record["clocks"][label]={"status":"archived_closed_loop_checkpoint_not_captured","requested_task_paid":paid}
            continue
        geom,masks=check.verify_measurement(weighted,metric,reference,run,label,paid)
        require(np.array_equal(masks["shelf"],archive[f"{rid}_{label}_shelf"]),f"Stored shelf mask differs: {directory}/{label}")
        raw[label]=(geom,masks["shelf"])
    keep=raw["start"][1]
    for label,(geo,presence) in raw.items():
        paid=geo["receipt"]["actual_paid"]
        record["clocks"][label]={"status":"measured","requested_task_paid":paid,
                                 "requested_paid_since_fork":paid-case["paid_before"],
                                 "net_area_m2":fsum(weights[presence & ~keep])-fsum(weights[keep & ~presence]),
                                 "mesh_sha256":geo["mesh_sha256"],"checkpoint_sha256":geo["checkpoint_sha256"]}
    terminal=raw["terminal"][0]["receipt"]
    closure=weighted.native.close_record(terminal,load(directory/"runtime_summary.json"),trace,initial)
    require(all(closure[k]==manifest[k] for k in ("returned","collisions","path_length_m")),f"Actual closure differs: {directory}")
    record.update(manifest_sha256=mhash,steps_sha256=sha(directory/"steps.json"),neutral_local_decisions=neutral_calls,
                  archived_short_charged_actions_verified=verified_prior_charged_actions,route_event=event,closure=closure)
    common=tuple((raw[k][0]["mesh_sha256"],raw[k][0]["receipt"]["packet_sha256"],raw[k][0]["receipt"]["belief_sha256"])
                 for k in ("start","fork"))
    require(common[0][1]==case["cold_packet_sha256"] and common[1][1]==case["fork_packet_sha256"]
            and common[1][2]==case["fork_belief_sha256"],f"Cold/fork source identity drift: {directory}")
    return record,common,(keep,raw["fork"][1])


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage-root",type=Path,default=ROOT/"recovery")
    parser.add_argument("--out",type=Path,default=HERE/"receipt.json")
    args=parser.parse_args()
    stage=args.stage_root.resolve()
    seal=stage/"DEVELOPMENT_FREEZE.json"
    stage_sha=sha(seal)
    stage_info=load(seal)
    verify(stage_info["input_sha256"])
    restart=previous_attempts(stage_info)
    bank_path=ROOT/"bank/BANK_FREEZE.json"
    require(stage_info["bank_sha256"]==sha(bank_path),"Changed registered branch bank")
    bank=load(bank_path)
    protocol=load(ROOT/"PROTOCOL.json")
    ledger=stage/"execution/receipt.json"
    native=load(ledger)
    results=stage/"results"
    summary_path=results/"summary.json"
    summary=load(summary_path)
    require(summary["execution_receipt_sha256"]==sha(ledger)
            and summary["freeze_sha256"]==native["freeze_sha256"]==stage_sha,"Receiver lacks stage identity")
    verify({results/name:h for name,h in summary["result_files_sha256"].items()})
    checkpoint_rows={(r["placement"],r["route_id"],r["checkpoint"]):r for r in table(results/"checkpoints.csv")}
    outcomes={(o["placement"],o["route_id"]):o for o in native["outcomes"]}
    expected={(p,r["route_id"]) for p,c in bank["placements"].items() for r in c["routes"]}
    require(len(outcomes)==len(native["outcomes"])==len(expected)==12 and set(outcomes)==expected,
            "Completed registry is missing a prespecified route or has extra routes")
    require(protocol["development_order"]==stage_info["placement_order"]==list(bank["placements"]),"Scenario order changed")
    check,weighted,metric=checks()
    all_records,comparisons,references={}, {},{}
    for name,case in bank["placements"].items():
        initial,reference,references[name]=reference_for(case,name,weighted)
        common,common_masks=None,None
        by_route={}
        mask_path=results/f"{name}_masks.npz"
        with np.load(mask_path,allow_pickle=False) as masks:
            weights=np.broadcast_to(np.asarray(reference["weights"]["shelf"],dtype=float),reference["samples"]["shelf"].shape[:1])
            require(np.array_equal(weights,masks["reference_shelf_weights_m2"])
                    and np.array_equal(reference["samples"]["shelf"],masks["reference_shelf_xyz_m"]),f"Wrong exported reference: {name}")
            for route in case["routes"]:
                rid=route["route_id"]
                require(route["route_identity_sha256"]==digest({k:route[k] for k in IDENTITY}),f"Wrong sealed route checksum: {rid}")
                rec,key,matches=evaluate_record(name,case,route,outcomes[name,rid],stage,stage_sha,initial,reference,weights,masks,check,weighted,metric)
                by_route[rid]=rec
                all_records[f"{name}/{rid}"]=rec
                if key:
                    if common is None:
                        common,common_masks=key,matches
                    require(key==common and all(np.array_equal(a,b) for a,b in zip(common_masks,matches,strict=True)),
                            f"Start/fork geometry differs across verified alternatives: {name}/{rid}")
                    require(np.array_equal(matches[0],masks["start_shelf"]) and np.array_equal(matches[1],masks["fork_shelf"]),
                            f"Released cold/fork mask mismatch: {name}/{rid}")
                for label in (16,24,48,96,144,192,"terminal"):
                    field=f"paid_{label}" if label!="terminal" else label
                    value=rec["clocks"].get(field,{"status":"technical_failure"})
                    archived=checkpoint_rows[name,rid,field]
                    require(archived["checkpoint_status"]==value["status"],f"Published missingness mismatch: {name}/{rid}/{field}")
                    same_num(value.get("net_area_m2"),archived["net_area_from_cold_m2"],f"Published net mismatch: {name}/{rid}/{field}")
        D,S=(case["selected"][p] for p in ("D","S"))
        at96={rid:r["clocks"]["paid_96"]["net_area_m2"] for rid,r in by_route.items()
              if r["clocks"].get("paid_96",{}).get("status")=="measured"}
        d=at96.get(D)
        diff={rid:area-d for rid,area in at96.items() if rid!=D} if d is not None else {}
        best=max(at96.values()) if at96 else None
        gain=best-d if d is not None else None
        tol=protocol["tie_tolerance_m2"]
        release=summary["placements"][name]["clocks"]["paid_96"]
        winners=sorted(rid for rid,area in at96.items() if abs(area-best)<=tol)
        require(release["n_measured"]==len(at96) and release["n_frozen"]==len(by_route)
                and release["best_route_ids"]==winners and release["any_measured_above_D"]==(gain is not None and gain>tol),
                f"Released matched-route classification drift: {name}")
        same_num(gain,release["best_minus_D_m2"],f"Released primary opportunity drift: {name}")
        same_num(d,release["D_m2"],f"Released direct baseline mismatch: {name}")
        same_num(at96.get(S),release["S_m2"],f"Released S value mismatch: {name}")
        positive_here=gain is not None and gain>tol
        followed=next((rid for rid in winners if rid!=D),None) if positive_here else D
        archived_follow=summary["placements"][name]["A96_best_followed_to_return"]
        require(followed==archived_follow["route_id"],f"Reported early best route unexpectedly changed: {name}")
        retained={"route_id":followed}
        if positive_here and all(by_route[rid].get("closure",{}).get("returned") for rid in (D,followed)):
            with np.load(mask_path,allow_pickle=False) as masks:
                exclusive=masks[f"{followed}_paid_96_shelf"] & ~masks[f"{D}_paid_96_shelf"] & ~masks["fork_shelf"]
                kept=exclusive & masks[f"{followed}_terminal_shelf"] & ~masks[f"{D}_terminal_shelf"]
                retained.update(early_exclusive_m2=fsum(weights[exclusive]),still_exclusive_at_return_m2=fsum(weights[kept]))
            retained.update(terminal_minus_D_m2=by_route[followed]["clocks"]["terminal"]["net_area_m2"]-
                            by_route[D]["clocks"]["terminal"]["net_area_m2"])
            for k in ("early_exclusive_m2","still_exclusive_at_return_m2","terminal_minus_D_m2"):
                same_num(retained[k],archived_follow[k],f"Released return evidence disagrees: {name}/{k}")
        comparisons[name]={"D_route_id":D,"S_route_id":S,"measured":len(at96),"frozen":len(by_route),"D_at96_m2":d,
                           "A96_best_route_ids":winners,"A96_best_minus_D_m2":gain,"compared_route_gain_m2":diff,
                           "win":sum(v>tol for v in diff.values()),"tie":sum(abs(v)<=tol for v in diff.values()),
                           "loss":sum(v< -tol for v in diff.values()),"A96_positive_placement":positive_here,
                           "A96_best_followed_to_return":retained}
        print(name,comparisons[name],flush=True)
    positive=[p for p,c in comparisons.items() if c["A96_positive_placement"]]
    faults=sum(r["execution_status"]=="technical_failure" for r in all_records.values())
    gate="stage_II" if len(positive)>=2 else "review_technical_failure" if faults else "stop_stage_I"
    require(positive==summary["positive_distinct_placements"] and gate==summary["gate"],"Independent gate disagreed with release")
    data={"status":"independently_verified_stage_I", "gate":gate,"positive_distinct_placements":positive,
          "base_commit":protocol["base_commit"],"stage_seal_sha256":stage_sha,"branch_bank_sha256":sha(bank_path),
          "execution_receipt_sha256":sha(ledger),"stage_summary_sha256":sha(summary_path),"scheduled_routes":len(outcomes),
          "technical_failures":faults,"positive_tolerance_m2":protocol["tie_tolerance_m2"],"measure_match_m":.05,
          "runs":all_records,"placements":comparisons,"references_sha256":references,"previous_attempts":restart,
          "audit_source_sha256":sha(Path(__file__)),"result_files_sha256":summary["result_files_sha256"]}
    with args.out.open("x") as stream:
        json.dump(data,stream,ensure_ascii=False,allow_nan=False,indent=2)
        stream.write("\n")
    print("Verified",len(outcomes),"sealed routes, gate",gate,flush=True)


if __name__=="__main__":
    main()
