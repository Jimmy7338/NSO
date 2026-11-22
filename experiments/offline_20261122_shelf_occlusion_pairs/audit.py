#!/usr/bin/env python3
"""Independently audit formal assignments, masks and release chronology."""

from math import isclose
from statistics import fmean, median

import numpy as np

from study import ARMS, ROOT, read, sha, verify, write
from write_paper import table, numeric


def require(condition, detail):
    if not condition:
        raise ValueError(detail)


def compare(a, b, detail):
    require((a is None and b is None) or (a is not None and b is not None and isclose(a,b,abs_tol=1e-12)), detail)


def evidence_for_parent(parent):
    path=ROOT/"scenes/pair_audits"/f"{parent}.json"
    measured=read(path)
    require(measured["same_reachable_and_swept_graph"] and measured["same_initial_raw_lidar"]
            and measured["same_initial_intrinsic_pose"] and measured["same_all_other_node_geometry_and_appearance"],
            f"Physical/sensor pairing failed in {parent}.")
    require(measured["front_visible_screen_fraction"] < measured["front_visible_open_fraction"]
            and measured["hidden_points_seen_by_side_pose"]>0, f"No optical intervention in {parent}.")
    require(measured["policy_calls"]==0 and measured["budget_actions"]==192, "Invalid scene seal.")
    return measured["shelf_reference_sha256"]


def audit_retention(record, masks, weights, rows):
    arms = record["left"], record["right"]
    early = [masks.get(f"{arm}_paid_96_shelf") for arm in arms]
    terminal = [masks.get(f"{arm}_terminal_shelf") for arm in arms]
    fork = masks.get("fork_shelf")
    returned = all(rows[record["placement"], arm]["returned"] == "True" for arm in arms)
    if fork is None:
        status = "no_observed_fork"
    elif any(mask is None for mask in early):
        status = "missing_paid_96"
    elif not returned or any(mask is None for mask in terminal):
        status = "missing_dual_return"
    else:
        status = "measured"
    require(record["status"] == status, f"Incorrect retention denominator for {record['placement']}.")
    if fork is None or any(mask is None for mask in early):
        return 0
    checks = 0
    for index, side in enumerate(("left", "right")):
        cohort = early[index] & ~early[1-index] & ~fork
        measured = {"exclusive_96": cohort}
        if status == "measured":
            own, other = terminal[index], terminal[1-index]
            measured.update({"still_exclusive_return": cohort & own & ~other,
                             "now_shared_return": cohort & own & other,
                             "only_other_return": cohort & ~own & other,
                             "neither_return": cohort & ~own & ~other})
            area = float(weights[cohort].sum())
            kept = float(weights[measured["still_exclusive_return"]].sum())
            compare(kept / area if area else None, numeric(record, f"{side}_still_exclusive_fraction"),
                    f"Wrong early-surface retention fraction for {record['placement']}.")
        for label, expected in measured.items():
            name = f"{side}_{label}"
            present = masks[f"{record['pair']}_{name}"]
            require(np.array_equal(expected, present), f"Retention mask differs for {record['placement']} {name}.")
            require(int(expected.sum()) == int(record[f"{name}_samples"]), "Retention sample count differs.")
            compare(float(weights[expected].sum()), numeric(record, f"{name}_m2"),
                    f"Retention area differs for {record['placement']} {name}.")
            checks += 1
    return checks


def main():
    result=ROOT/"results/formal"
    all_runs=ROOT/"execution/formal"
    freeze=ROOT/"FORMAL_FREEZE.json"
    seal=read(freeze)
    base=sha(freeze)
    report=read(result/"summary.json")
    require(report["phase_freeze_sha256"]==base, "Outcomes came from a different freeze.")
    verify({str(result/name):digest for name,digest in report["table_sha256"].items()})
    verify({str(result/file["artifact"]):file["sha256"] for file in report["masks"].values()})
    require(seal["native_budget"]==192 and seal["weights"]=={"unknown_weight":.25,"action_cost":.20}
            and seal["direct_scale"]==4 and seal["hidden_scale"]==1, "Frozen treatment weights changed.")
    verify(seal["actor_files_sha256"])
    verify(seal["evaluator_files_sha256"])
    verify(seal["simulator_only_cache_sha256"])
    tolerance=read(ROOT/"measurement/development_repeatability.json")
    require(tolerance["development_freeze_sha256"]==sha(ROOT/"DEVELOPMENT_FREEZE.json")
            and seal["tie_tolerance_m2"]==tolerance["tie_tolerance_m2"], "Tie tolerance must come from development.")
    require(not any("evaluator_references" in name for name in seal["actor_files_sha256"]), "Evaluator input in actor seal.")
    first=read(all_runs/"formal_first_receipt.json")
    release=read(ROOT/"results/formal_first/release.json")
    rest=read(all_runs/"formal_rest_receipt.json")
    require(all(x["phase_freeze_sha256"]==base for x in (first,rest,release)), "Stage freeze differs.")
    require(first["placements"]+rest["placements"]==seal["placements"]
            and first["placements"]==release["placements"] and len(first["placements"])==4, "Wrong staged order.")
    require(sha(ROOT/"results/formal_first/summary.json")==release["summary_sha256"], "Interim released summary drift.")
    for name in rest["placements"]:
        for launch in (all_runs/name).glob("*/*/launch.json"):
            require(read(launch)["start_unix_s"]>=release["release_unix_s"], f"{name} began before interim release.")
    episodes=table(result/"episodes.csv")
    rows={(r["placement"],r["arm"]):r for r in episodes}
    require(len(episodes)==len(rows)==48
            and rows.keys()=={(p,a) for p in seal["placements"] for a in ARMS},"An assignment is missing or replicated.")
    failures={(r["placement"],r["arm"]) for r in table(result/"failures.csv")}
    retention=table(result/"retention.csv")
    require(len(retention)==len({(r["placement"],r["pair"]) for r in retention})==36,
            "The paired retention schedule is incomplete.")
    net_by={}
    remeasured=0
    retained_remeasured=0
    for entry in first["placement_receipts"]+rest["placement_receipts"]:
        require(sha(entry["path"])==entry["sha256"], "Placement receipt drift.")
        placement=entry["placement"]
        source=read(entry["path"])
        with np.load(result/f"{placement}_masks.npz",allow_pickle=False) as masks:
            weights=masks["reference_shelf_weights_m2"]
            for arm in ARMS:
                seen=rows[placement,arm]
                registered=source["arms"][arm]
                if registered["status"]=="technical_failure":
                    require((placement,arm) in failures and seen.get("status")=="technical_failure",
                            "Technical failure disappeared from the table.")
                else:
                    require(seen["manifest_sha256"]==registered["manifest_sha256"], "Assigned source run drift.")
                for clock in (48,96,144,192,"terminal"):
                    key=f"{arm}_{'terminal' if clock=='terminal' else f'paid_{clock}'}_shelf"
                    observed=numeric(seen,f"net_{clock}_m2")
                    if key in masks and "start_shelf" in masks:
                        start,stop=masks["start_shelf"],masks[key]
                        recomputed=float(weights[stop & ~start].sum()-weights[start & ~stop].sum())
                        compare(recomputed,observed,f"Shelf net area differs for {placement} {arm} {clock}")
                        compare(numeric(seen,f"correct_free_cells_{clock}")/numeric(seen,f"reachable_cells_{clock}"),
                                numeric(seen,f"C_free_{clock}"),"Correctly observed free-area ratio differs.")
                        compare(numeric(seen,f"C_free_{clock}")*numeric(seen,f"F1_scene_05cm_{clock}"),
                                numeric(seen,f"J_{clock}"),"J is not the product of measured factors.")
                        if clock != "terminal":
                            require(int(seen[f"actual_paid_{clock}"])==clock,"Unexecuted clock imputed.")
                        remeasured+=1
                    else:
                        require(observed is None, f"Unverified shelf metric {placement} {arm} {clock}")
                    if clock==96:
                        net_by[placement,arm]=observed
            for row in (r for r in retention if r["placement"]==placement):
                retained_remeasured += audit_retention(row, masks, weights, rows)
    differences={"open":[],"screen":[],"interaction":[]}
    paired=table(result/"pairs.csv")
    require([r["unit"] for r in paired]==seal["parents"], "Parent result rows are misordered or missing.")
    for record in paired:
        parent=record["unit"]
        evidence_for_parent(parent)
        values={}
        for mode in ("open","screen"):
            s,d=(net_by[f"{parent}_{mode}",arm] for arm in ("S","D"))
            values[mode]=s-d if s is not None and d is not None else None
            compare(values[mode],numeric(record,f"{mode}_S_minus_D_96_m2"),f"{parent} {mode} contrast differs.")
            differences[mode].append(values[mode])
        difference=values["screen"]-values["open"] if all(v is not None for v in values.values()) else None
        compare(difference,numeric(record,"occlusion_interaction_96_m2"),f"{parent} interaction differs.")
        differences["interaction"].append(difference)
    for metric,values in differences.items():
        key="S_minus_D_interaction_96_m2" if metric=="interaction" else f"{metric}_S_minus_D_96_m2"
        stats=report["statistics"][key]
        known=[v for v in values if v is not None]
        require(stats["n"]==len(known) and stats["scheduled"]==6,"Wrong independent denominator.")
        if known:
            compare(fmean(known),stats["mean"],"Independent mean mismatch.")
            compare(median(known),stats["median"],"Independent median mismatch.")
            require(stats["win"]==sum(v>seal["tie_tolerance_m2"] for v in known)
                    and stats["tie"]==sum(abs(v)<=seal["tie_tolerance_m2"] for v in known)
                    and stats["loss"]==sum(v<-seal["tie_tolerance_m2"] for v in known),"Win/tie/loss mismatch.")
    items=[ROOT/"paper"/name for name in ("paired_shelf_area.pdf","paired_shelf_area.svg","mechanism_case.pdf",
             "mechanism_case.svg","figure_case.json","captions.md","manuscript_results_discussion.md")]
    require(all(p.is_file() and p.stat().st_size>200 for p in items), "An English paper or figure deliverable is missing.")
    result_record={
        "status":"verified_complete_frozen_simulation_matrix", "formal_freeze_sha256":base,
        "scheduled_policy_outcomes":len(episodes),"evaluated_unique_trajectories":report["unique_completed_executions"],
        "formal_summary_sha256":sha(result/"summary.json"),"geometry_pairs_audited":len(paired),
        "independent_mask_shelf_values_recomputed":remeasured,"failures_explicitly_listed":len(failures),
        "independent_mask_retention_values_recomputed":retained_remeasured,
        "first_two_released_before_rest":True,"primary_pair_rows":differences,
        "editable_paper_deliverables":{p.name:sha(p) for p in items},"simulated_only":True,
    }
    write(ROOT/"completion_audit.json",result_record)
    print(f"Verified {len(episodes)} appointments, {remeasured} net-area/J values, and six independently paired parents.")


if __name__ == "__main__":
    main()
