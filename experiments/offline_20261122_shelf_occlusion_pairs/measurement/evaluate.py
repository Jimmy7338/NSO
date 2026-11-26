#!/usr/bin/env python3
"""Build shelf masks, assignment and parent-pair tables from verified native executions."""

import argparse
import csv
import gc
import json
from pathlib import Path

from historical import HERE, checks
import numpy as np

from metrics import ARMS, CLOCKS, COMPARISONS, classify, compare_pair, describe, pair_name, shelf_at_clock, snap_key
from verify_execution import require, verified_run


ROOT = HERE.parent
TOLERANCE = HERE / "development_repeatability.json"
PLACEMENTS = ("open", "screen")


def write_json(path, data):
    text = json.dumps(data, indent=2, allow_nan=False) + "\n"
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(text)
    temp.replace(path)


def write_csv(path, records, known_columns=()):
    fields = list(dict.fromkeys([*known_columns, *(field for row in records for field in row)]))
    temp = path.with_suffix(path.suffix + ".tmp")
    with temp.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fields)
        writer.writeheader()
        for row in records:
            writer.writerow({key: json.dumps(value, ensure_ascii=False, allow_nan=False)
                             if isinstance(value, (list, dict)) else value for key, value in row.items()})
    temp.replace(path)


def load_context(phase, first, check):
    protocol = check.read(ROOT / "PROTOCOL.json")
    freeze_path = ROOT / f"{phase.upper()}_FREEZE.json"
    frozen = check.read(freeze_path)
    freeze_sha = check.sha(freeze_path)
    require(frozen["phase"] == phase and frozen["protocol_sha256"] == check.sha(ROOT / "PROTOCOL.json"),
            "Study protocol bytes do not match the registered phase.")
    for key in ("base_commit", "native_budget", "capture_paid", "max_trigger_paid", "weights",
                "local_route", "direct_scale", "hidden_scale", "arm_order"):
        require(frozen[key] == protocol[key], f"Phase configuration drift in {key}")
    require(frozen["arm_order"] == list(ARMS) and protocol["schedule"]["conditions"] == list(PLACEMENTS),
            "Study identifies a different arm or occlusion condition.")
    for path, wanted in frozen["evaluator_files_sha256"].items():
        require(check.sha(Path(path)) == wanted, f"Frozen evaluator file drift: {path}")
    directory = ROOT / "execution" / phase
    receipts = {}
    for file in sorted(directory.glob("*_receipt.json")):
        stage = check.read(file)
        require(stage["phase_freeze_sha256"] == freeze_sha, f"Wrong stage receipt for {phase}: {file}")
        require(set(stage["placements"]) == {r["placement"] for r in stage["placement_receipts"]},
                f"Stage report omits an assignment: {file}")
        for r in stage["placement_receipts"]:
            path = (directory / r["placement"] / "placement_receipt.json").resolve()
            require(Path(r["path"]).resolve() == path and check.sha(path) == r["sha256"],
                    f"Unsealed or changed placement receipt: {path}")
            require(r["placement"] not in receipts, f"Placement is included in two stage ledgers: {path}")
            receipts[r["placement"]] = path
    parents = frozen["parents"][:first] if first is not None else frozen["parents"]
    require(bool(parents) and (first is None or len(parents) == first), "Requested prefix is outside the frozen schedule.")
    require(frozen["placements"] == [f"{p}_{c}" for p in frozen["parents"] for c in PLACEMENTS],
            "Frozen placements differ from strict paired schedule.")
    for parent in parents:
        for condition in PLACEMENTS:
            require(f"{parent}_{condition}" in receipts, f"Scheduled placement is awaiting a sealed stage receipt: {parent}_{condition}")
    return dict(phase=phase, freeze_sha=freeze_sha, spec=frozen, parents=parents, receipts=receipts,
                tolerance=float(frozen["tie_tolerance_m2"]))


def path_xy(trace, initial):
    cells = np.array([initial["anchor"][:2], *(row["position"] for row in trace)])
    r = initial["grid_resolution_m"]
    return np.column_stack((cells[:, 1] + .5, initial["shape"][0] - cells[:, 0] - .5)) * r


def assignment_row(assigned, arm, execution, weights, g_technical_failure):
    result = {key: assigned.get(key) for key in (
        "status", "execution_arm", "shared_execution", "route_identity_sha256", "technical_exit_code",
        "technical_reason", "run_directory", "manifest_sha256", "fork_bank_sha256",
    )}
    if execution is None:
        result["status"] = "technical_failure"
        return result
    bank = execution["bank"]
    triggered = bank["status"] == "triggered"
    physical_arm = result["execution_arm"]
    unbound_fallback = (g_technical_failure and arm != "G" and physical_arm == arm
                        and not result["shared_execution"] and assigned["route_identity_sha256"] is None
                        and not execution["manifest"]["independently_reproduced_g_fork"])
    require(physical_arm == "G" or unbound_fallback or execution["manifest"]["independently_reproduced_g_fork"],
            f"Physical {physical_arm} did not independently recreate the registered G fork.")
    if triggered:
        a, s = bank["arms"][arm], bank["arms"][physical_arm]
        common_keys = ("route_identity_sha256", "outbound_actions", "outbound_states", "return_actions", "return_states",
                       "common_continuation_id", "shared_identity_sha256")
        require(all(a[key] == s[key] for key in common_keys)
                and (a["route_identity_sha256"] == assigned["route_identity_sha256"] or unbound_fallback),
                f"Alias {arm} does not share this native execution's state, oriented route and planned return.")
        result["route_identity_sha256"] = a["route_identity_sha256"]
        selected_score = next(row for row in execution["eligible_fork_rankings"][arm] if row["route_id"] == a["route_id"])
    else:
        require(assigned["route_identity_sha256"] is None and (physical_arm == "G" or unbound_fallback),
                "A case without a fork must use the exact neutral continuation.")
        a, selected_score = {}, {}
    require(result["shared_execution"] == (physical_arm != arm), "Inconsistent assignment alias flag.")
    manifest, closure = execution["manifest"], execution["closure"]
    route = manifest["route_event"]
    trace = execution["trace"]
    decision_wall = [r["decision_wall_s"] for r in trace]
    actual_paid = closure["actual_paid"]
    fork_paid = bank["paid_before"]
    result |= {
        "status": "verified_execution", "triggered": triggered, "fork_paid": fork_paid,
        "unbound_after_g_technical_failure": unbound_fallback,
        "fork_shared_sha256": bank.get("shared_identity_sha256"), "selected_route_id": a.get("route_id"),
        "oriented_target": bank.get("target_pose"), "selected_nominal_outbound": len(a["outbound_actions"]) if a else None,
        "selected_nominal_reserved_return": len(a["return_actions"]) if a else None,
        "fork_ranked_distinct_routes": len(execution["eligible_fork_rankings"][arm]) if a else None,
        "fork_selected_predicted_shelf_area_m2": selected_score.get("area_m2"),
        "fork_selected_relative_unknown_m2": selected_score.get("delta_unknown_union_m2"),
        "fork_selected_weighted_score_m2": selected_score.get("score_m2"),
        "fork_instrumented_policy_ranking_wall_s": a.get("ranking_wall_s"),
        "actual_bound_outbound_actions": execution["actual_route_actions"],
        "route_status": route["status"], "route_end_paid": route["actual_paid"], "route_reason": route.get("reason"),
        "actual_paid": actual_paid, "returned": closure["returned"], "collisions": closure["collisions"],
        "path_length_m": closure["path_length_m"], "terminal_pose": closure["terminal_pose"],
        "early_return": closure["returned"] and actual_paid < manifest["native_budget"],
        "native_terminal_kind": "returned" if closure["returned"] else "stopped_without_return",
        "native_stop_reason": closure["stop_reason"], "native_scientific_failure": closure["failed"],
        "decision_total_wall_s_shared_execution": float(sum(decision_wall)),
        "decision_fork_callback_wall_s_shared_execution": decision_wall[fork_paid]
              if fork_paid is not None and fork_paid < len(trace) else None,
        "shared_run_elapsed_s": manifest["elapsed_s"],
        "shelf_evaluable_m2": float(weights.sum()), "shelf_evaluable_samples": len(weights),
    }
    start = execution["snapshots"]["start"][1]["shelf"]
    for clock in CLOCKS:
        name = snap_key(clock)
        prefix = str(clock)
        if name not in execution["snapshots"]:
            result[f"status_{prefix}"] = "not_executed_before_actual_terminal"
            continue
        met, masks = execution["snapshots"][name]
        shelf = shelf_at_clock(start, masks["shelf"], weights)
        result |= {f"status_{prefix}": "measured", f"actual_paid_{prefix}": met["receipt"]["actual_paid"],
                   f"net_{prefix}_m2": shelf["net_m2"], f"gross_{prefix}_m2": shelf["gross_m2"],
                   f"loss_{prefix}_m2": shelf["lost_m2"], f"represented_{prefix}_m2": shelf["represented_m2"],
                   f"cold_shelf_support_{prefix}_m2": shelf["cold_support_m2"],
                   **{f"{key}_{prefix}":met[key] for key in ("C_free","F1_scene_05cm","J","correct_free_cells",
                                                           "reachable_cells","sampled_predicted_surface")}}
    return result


def compare_common_prefix(left, right, g_technical_failure):
    a, b = left["bank"], right["bank"]
    require(a["status"] == b["status"] and left["raw_initial"] == right["raw_initial"],
            "Within-placement assignments have different cold packets or trigger outcomes.")
    require(left["snapshots"]["start"][0]["mesh_sha256"] == right["snapshots"]["start"][0]["mesh_sha256"]
            and np.array_equal(left["snapshots"]["start"][1]["shelf"], right["snapshots"]["start"][1]["shelf"]),
            "Within-placement cold baseline mesh or reference-presence index differs.")
    rows = ["step", "action", "position", "heading", "packet_sha256", "packet_chain_sha256", "owner_sha256",
            "rgb_values_sha256", "depth_values_sha256", "scan_ranges_sha256", "collision"]
    if a["status"] == "triggered":
        require(a["shared_identity_sha256"] == b["shared_identity_sha256"] and a["continuation_id"] == b["continuation_id"]
                and a["candidate_pool_sha256"] == b["candidate_pool_sha256"], "Observed common fork or full legal pool differs.")
        count = a["paid_before"]
    else:
        require(g_technical_failure and a["continuation_id"] == b["continuation_id"]
                and a["prefix_observation_sha256"] == b["prefix_observation_sha256"]
                and len(left["trace"]) == len(right["trace"]),
                "Multiple independently executed no-fork paths are allowed only after G technical failure.")
        count = len(left["trace"])
    for u, v in zip(left["trace"][:count], right["trace"][:count], strict=True):
        require(all(u[key] == v[key] for key in rows), f"Executable common prefix diverged at paid {u['step']}")
    key = "fork" if a["status"] == "triggered" else "terminal"
    require(left["snapshots"][key][0]["mesh_sha256"] == right["snapshots"][key][0]["mesh_sha256"]
            and np.array_equal(left["snapshots"][key][1]["shelf"], right["snapshots"][key][1]["shelf"]),
            f"Actual common {key} reconstruction differs.")


def placement(meta, parent, condition, initial, reference, check, weighted, metric, repeat):
    name = initial["layout"]
    path = meta["receipts"][name]
    receipt = check.read(path)
    arms = receipt["arms"]
    require(receipt["placement"] == name and receipt["phase_freeze_sha256"] == meta["freeze_sha"]
            and set(arms) == set(ARMS), f"Incorrect scheduled assignment ledger: {path}")
    g_technical_failure = arms["G"]["status"] == "technical_failure"
    executions = {}
    for assigned in arms.values():
        if assigned["status"] != "verified_native_execution":
            continue
        directory = Path(assigned["run_directory"]).resolve()
        require(directory.is_relative_to(path.parent.resolve()), f"Cross-placement or foreign artifact alias: {directory}")
        if directory not in executions:
            executions[directory] = verified_run(directory, assigned, meta, initial, reference, check, weighted, metric, repeat)
        else:
            require(check.sha(directory / "fork_bank.json") == assigned["fork_bank_sha256"]
                    and executions[directory]["manifest_sha256"] == assigned["manifest_sha256"],
                    f"Assignment alias points to different execution bytes: {directory}")
    require(len(executions) == receipt["unique_completed_executions"], f"Physical trajectory count drift: {path}")
    physical = list(executions.values())
    for run in physical[1:]:
        compare_common_prefix(physical[0], run, g_technical_failure)
    samples = reference["samples"]["shelf"]
    weights = np.broadcast_to(np.asarray(reference["weights"]["shelf"], dtype=float), (len(samples),))
    export={"reference_shelf_xyz_m": samples, "reference_shelf_weights_m2":weights}
    if physical:
        export["start_shelf"] = physical[0]["snapshots"]["start"][1]["shelf"]
    fork = None
    if physical and physical[0]["bank"]["status"] == "triggered":
        run = physical[0]
        fork = run["snapshots"]["fork"][1]["shelf"]
        export |= {"fork_shelf":fork, "fork_rgb":run["fork_rgb"],
                   "fork_paid":np.asarray(run["bank"]["paid_before"], dtype=np.int16),
                   "fork_pose_rch":np.asarray(run["bank"]["pose"], dtype=np.int16)}
    per_assignment, details = [], {}
    for arm in ARMS:
        assigned = arms[arm]
        run = executions.get(Path(assigned["run_directory"]).resolve()) if assigned["run_directory"] else None
        row = dict(unit=parent, condition=condition, placement=name, arm=arm) | assignment_row(
            assigned, arm, run, weights, g_technical_failure)
        per_assignment.append(row)
        value = row.copy()
        if run is not None:
            value["trace"] = run["trace"]
            value["snapshots"], value["masks"] = {}, {}
            export[f"{arm}_trajectory_xy"] = path_xy(run["trace"], initial)
            export[f"{arm}_trajectory_paid"] = np.arange(len(run["trace"]) + 1, dtype=np.int16)
            for phase, (met, masks) in run["snapshots"].items():
                export[f"{arm}_{phase}_shelf"] = masks["shelf"]
                value["masks"][phase] = masks["shelf"]
                value["snapshots"][phase] = {"paid":met["receipt"]["actual_paid"], "C_free": met["C_free"],
                                            "F1_scene_05cm":met["F1_scene_05cm"], "J":met["J"],
                                            "net_m2": shelf_at_clock(export["start_shelf"], masks["shelf"], weights)["net_m2"]}
        details[arm] = value
    contrasts, retained = [], []
    for left,right in COMPARISONS:
        key = pair_name(left, right)
        rows, retention, extra = compare_pair(details[left],details[right],weights,fork,meta["tolerance"])
        base = dict(unit=parent, condition=condition, placement=name, pair=key, left=left, right=right)
        contrasts.extend(base | row for row in rows)
        retained.append(base | retention)
        export.update({f"{key}_{key2}":value for key2, value in extra.items()})
    repeat_rows = [row for run in physical for row in run["repeat_rows"]]
    counts = {"unique_completed_executions":len(executions),"unique_attempted_assignments":receipt["unique_attempted_assignments"],
              "verified_fork":bool(fork is not None), "placement_receipt_sha256":check.sha(path)}
    return per_assignment, contrasts, retained, export, repeat_rows, counts


def paired_table(parents, comparisons, retention, tolerance):
    key = lambda row: (row["unit"], row["condition"], row["pair"], row["clock"])
    clocks = {key(row):row for row in comparisons}
    saved = {(r["unit"],r["condition"],r["pair"]):r for r in retention}
    table = []
    for parent in parents:
        row = {"unit":parent}
        for left,right in COMPARISONS:
            name = pair_name(left,right)
            for clock in CLOCKS:
                a,b = (clocks[parent,condition,name,clock] for condition in PLACEMENTS)
                open_net, screened_net = a["net_m2_delta"], b["net_m2_delta"]
                interaction = screened_net - open_net if screened_net is not None and open_net is not None else None
                for condition, info in zip(PLACEMENTS,(a,b),strict=True):
                    row[f"{condition}_{name}_{clock}_m2"] = info["net_m2_delta"]
                    row[f"{condition}_{name}_{clock}_J_delta"] = info["J_delta"]
                    row[f"{condition}_{name}_{clock}_outcome"] = info["shelf_outcome"]
                    row[f"{condition}_{name}_{clock}_same_paid"] = info["equal_paid_clock"]
                row[f"{name}_interaction_{clock}_m2"] = interaction
                row[f"{name}_interaction_{clock}_outcome"] = classify(interaction,tolerance)
                if clock == "terminal":
                    same = a["equal_paid_clock"] and b["equal_paid_clock"] and a["left_paid"] == b["left_paid"]
                    row[f"{name}_interaction_terminal_equal_paid_m2"] = interaction if same else None
                if (left,right)==("S","D"):
                    row[f"occlusion_interaction_{clock}_m2"] = interaction
            for cond in PLACEMENTS:
                data = saved[parent,cond,name]
                row[f"{cond}_{name}_retention_status"] = data["status"]
                for item,value in data.items():
                    if item.endswith(("_m2","_fraction","_samples")):
                        row[f"{cond}_{name}_{item}"] = value
        table.append(row)
    return table


def mechanism(parent_order, episodes, contrasts):
    order = {p: index for index,p in enumerate(parent_order)}
    outcomes = {(r["placement"],r["arm"]):r for r in episodes}
    rows = sorted((r for r in contrasts if r["pair"]=="S_minus_D" and r["clock"]==96),
                  key=lambda r:(r["condition"]!="screen",order[r["unit"]]))
    def meets(r):
        return (r["measured"] and outcomes[r["placement"], "S"].get("triggered")
                and all(outcomes[r["placement"], arm].get("returned") for arm in ("S", "D")))
    for test in (lambda row: meets(row) and row["first_actual_difference_paid"] is not None, meets):
        chosen=next((r for r in rows if test(r)),None)
        if chosen is not None:
            return chosen["placement"]
    scheduled = sorted(rows, key=lambda r: (order[r["unit"]], PLACEMENTS.index(r["condition"])))
    return next((row["placement"] for row in scheduled
                 if any(outcomes[row["placement"], a].get("triggered") for a in ("S", "D"))), None)


def paired_aggregates(pairs, comparisons, retention, tolerance):
    columns = {k for row in pairs for k in row}
    description = {k:describe([row.get(k) for row in pairs],len(pairs),tolerance)
                   for k in sorted(columns) if k.endswith("_m2") and
                   (any(k.endswith(f"_{clock}_m2") for clock in CLOCKS) or k.endswith("_terminal_equal_paid_m2"))}
    for left,right in COMPARISONS:
        name=pair_name(left,right)
        for condition in PLACEMENTS:
            rows = [r for r in comparisons if r["condition"]==condition and r["pair"]==name]
            for clock in CLOCKS:
                selected = [r for r in rows if r["clock"]==clock]
                if clock=="terminal":
                    description[f"{condition}_{name}_equal_paid_terminal_m2"] = describe(
                        [r["net_m2_delta"] if r["equal_paid_clock"] else None for r in selected],len(pairs),tolerance)
                for metric in ("C_free","F1_scene_05cm","J"):
                    description[f"{condition}_{name}_{clock}_{metric}"] = describe(
                        [r[f"{metric}_delta"] if r["equal_paid_clock"] else None for r in selected],len(pairs))
                opposed = [r for r in selected if r["equal_paid_clock"] and
                           classify(r["net_m2_delta"],tolerance) in {"win","loss"}]
                assignments = [classify(r["net_m2_delta"],tolerance)==classify(r["J_delta"],1e-12) for r in opposed]
                description[f"{condition}_{name}_{clock}_shelf_and_J_same_order"] = {
                    "n_quantifiable_shelf_difference":len(assignments), "agreement":sum(assignments),
                    "disagreement_or_J_tie":len(assignments)-sum(assignments),
                }
            source = [r for r in retention if r["condition"]==condition and r["pair"]==name]
            description[f"{condition}_{name}_execution_counts"] = {
                "scheduled":len(pairs), "shared_trajectory":sum(r["same_execution"] is True for r in source),
                "same_selected_route":sum(r["same_selected_route"] is True for r in source),
                "different_selected_route":sum(r["same_selected_route"] is False for r in source),
                "actual_route_divergence":sum(r["first_actual_difference_paid"] is not None for r in source),
                "divergence_while_both_following_bound_route":sum(
                    r["difference_during_both_bound_fork_routes"] is True for r in source),
                "retention_measurable_with_dual_return":sum(r["status"]=="measured" for r in source),
            }
            for side in ("left","right"):
                for field in ("exclusive_96_m2","still_exclusive_return_m2","still_exclusive_fraction"):
                    description[f"{condition}_{name}_{side}_{field}"] = describe(
                        [r.get(f"{side}_{field}") for r in source],len(pairs))
    return description


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=("development","formal"), required=True)
    parser.add_argument("--first", type=int)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--repeatability", action="store_true")
    args = parser.parse_args()
    require(not args.repeatability or args.phase=="development", "Repeatability tolerance can use development runs only.")
    check, weighted, metric = checks()
    meta = load_context(args.phase,args.first,check)
    scene_root = Path(meta["spec"]["scene_root"])
    weighted.SCENES = scene_root
    args.out.mkdir(parents=True,exist_ok=True)
    episodes, contrasts, retentions, repeats, hashes, dispositions = [], [], [], [], {}, {}
    for parent in meta["parents"]:
        initials = [check.read(scene_root/"prepared"/f"{parent}_{c}"/"initial_state.json") for c in PLACEMENTS]
        refs = [weighted.load_reference(initial["layout"],initial) for initial in initials]
        require(all(initials[0][key]==initials[1][key] for key in ("anchor","seed","shape","actual_config","stop_budget"))
                and np.array_equal(refs[0]["samples"]["shelf"],refs[1]["samples"]["shelf"])
                and np.array_equal(refs[0]["weights"]["shelf"],refs[1]["weights"]["shelf"]),
                f"Shelf samples, weights or paired start conditions drifted between occluder variants: {parent}")
        for condition,initial,reference in zip(PLACEMENTS,initials,refs,strict=True):
            gc.collect()
            rows, comparisons, retained, archive, trial_repeat, counts = placement(
                meta,parent,condition,initial,reference,check,weighted,metric,args.repeatability)
            name=initial["layout"]
            artifact=args.out/f"{name}_masks.npz"
            np.savez_compressed(artifact,**archive)
            hashes[name]=dict(sha256=check.sha(artifact),artifact=artifact.name)
            dispositions[name]=counts
            episodes.extend(rows)
            contrasts.extend(comparisons)
            retentions.extend(retained)
            repeats.extend(trial_repeat)
            print(f"{name}: {counts['unique_completed_executions']} verified unique trajectories",flush=True)
    pairs=paired_table(meta["parents"],contrasts,retentions,meta["tolerance"])
    failures = [r for r in episodes if r["status"]!="verified_execution" or r.get("route_status") in
                 ("guard_cancelled","abandoned_or_replanned_before_arrival","terminal_before_arrival","collision")
                 or r.get("early_return") or r.get("native_terminal_kind")=="stopped_without_return"]
    write_csv(args.out/"episodes.csv",episodes)
    write_csv(args.out/"pairs.csv",pairs)
    write_csv(args.out/"comparisons.csv",contrasts)
    write_csv(args.out/"retention.csv",retentions)
    write_csv(args.out/"failures.csv",failures,("unit","condition","placement","arm","status"))
    assignments = sum(r["status"]=="verified_execution" for r in episodes)
    result = {
        "phase":args.phase,"phase_freeze_sha256":meta["freeze_sha"],"parents":meta["parents"],
        "independent_parent_units":len(meta["parents"]),"registered_policy_assignments":len(episodes),
        "measured_native_assignments":assignments,"technical_missing_assignments":len(episodes)-assignments,
        "unique_completed_executions":sum(row["unique_completed_executions"] for row in dispositions.values()),
        "unique_attempted_assignments":sum(row["unique_attempted_assignments"] for row in dispositions.values()),
        "conditions":dispositions,"tie_tolerance_m2":meta["tolerance"],"masks":hashes,
        "registered_triggered_placements":sum(row["verified_fork"] for row in dispositions.values()),
        "verified_returned_assignments":sum(bool(row.get("returned")) for row in episodes),
        "verified_arrived_assignments":sum(row.get("route_status")=="arrived" for row in episodes),
        "verified_paid_96_assignments":sum(row.get("status_96")=="measured" for row in episodes),
        "representative_mechanism_placement":mechanism(meta["parents"],episodes,contrasts),
        "statistics":paired_aggregates(pairs,contrasts,retentions,meta["tolerance"]),
        "table_sha256":{name:check.sha(args.out/name) for name in
                        ("episodes.csv","pairs.csv","comparisons.csv","retention.csv","failures.csv")},
    }
    write_json(args.out/"summary.json",result)
    if args.repeatability:
        require(len(repeats)>=2,"Development must provide at least two independently remeasured source snapshots.")
        maximum = max(row["abs_delta_m2"] for row in repeats)
        proof = dict(development_freeze_sha256=meta["freeze_sha"],repeat_snapshots=len(repeats),
                     max_abs_delta_m2=maximum,tie_tolerance_m2=max(1e-9,maximum),rows=repeats)
        if TOLERANCE.exists():
            require(check.read(TOLERANCE)==proof,"Existing frozen repeatability receipt differs.")
        else:
            write_json(TOLERANCE,proof)
        print(f"Development repeatability of {len(repeats)} sealed snapshots gives tolerance {proof['tie_tolerance_m2']:.9g} m².")


if __name__=="__main__":
    main()
