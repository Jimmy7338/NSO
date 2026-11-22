#!/usr/bin/env python3
"""Offline audit of the chain-17 native route and both archived comparison arms."""

import argparse
import csv
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import sys

for variable in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[variable] = "1"

import numpy as np

HERE = Path(__file__).resolve().parent
STUDY = HERE.parent
PREVIOUS = STUDY.parent / "route_intervention_20261122"
WEIGHTED = STUDY.parent / "weighted_geometric_control_20261122"
PROTOCOL = HERE / "MEASUREMENT_PROTOCOL.json"
PREVIOUS_PROTOCOL = PREVIOUS / "evaluation/MEASUREMENT_PROTOCOL.json"
OLD_FREEZE = PREVIOUS / "RUN_FREEZE.json"
LAYOUT = "paired_chain_17"
ARM_NAMES = ("category", "geometry_equal_cost", "geometry_top")
FORK = 105
CALL = 428
BUDGET = 192
FACILITY = "shelf"


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


legacy = load_module("prior_route_intervention_evaluation", PREVIOUS / "evaluation/evaluate.py")
read, sha, require = legacy.read, legacy.sha, legacy.require


def check_bound(record):
    path = Path(record["path"])
    require(sha(path) == record["sha256"], f"Archived input changed: {path}")
    return path


def trial_binding(directory, manifest, first_option):
    return {"directory": directory, "manifest": manifest, "receipt_sha256": sha(directory / "manifest.json"),
            "trace": read(directory / "steps.json"), "route_end": first_option}


def verify_existing(weighted, source, binding, old_arm):
    directory = Path(binding["directory"])
    require(directory == PREVIOUS / "runner/runs" / LAYOUT / old_arm, "Unexpected prior arm path")
    for record in (binding["manifest"], binding["fork_ready"], *binding["mesh_index"].values()):
        check_bound(record)
    return legacy.verify_root(weighted, directory, LAYOUT, old_arm, source, FORK, sha(OLD_FREEZE))


def audited_ranking(source, protocol):
    scorer = load_module("ranked_geometry_source", WEIGHTED / "runner/weighted_score.py")
    audit = source["choice"]["choice_audit"]
    candidates = audit["candidates"]
    by_id = {row["route_id"]: row for row in candidates}
    native = by_id["native"]
    config = protocol["continuation"]
    weights = scorer.Weights(config["unknown_weight"], config["action_cost"])
    original = audit["weighted_ranking"]
    tolerance = original["score_equality_tolerance"]
    winner, all_scores = scorer.rank_admitted(candidates, native, "G", weights,
                                             original["grid_cell_area_m2"], tolerance)
    require(len(all_scores) == 34 and winner["route_id"] == "native", "Full G_perf winner changed")
    best = max(row["score_m2"] for row in all_scores)
    tied = [row["route_id"] for row in all_scores if best - row["score_m2"] <= tolerance]
    require(len(tied) > 1 and min(by_id[item]["outbound_cost"] for item in tied) == native["outbound_cost"],
            "Native ranking did not come from the registered tie rule")
    stable_table = source["anchor"]["G_perf_all_legal_ranks"]
    scores = {row["route_id"]: row["score_m2"] for row in all_scores}
    require(len(stable_table) == len(scores) and all(scores[row["route_id"]] == row["score_m2"] for row in stable_table),
            "Archived ordinal table disagrees with independent score calculation")
    rank_by_id = {row["route_id"]: row for row in stable_table}
    retained = {arm: rank_by_id[spec["route_id"]] for arm, spec in protocol["arms"].items()}
    for arm, spec in protocol["arms"].items():
        row = by_id[spec["route_id"]]
        require(row["outbound_cost"] == spec["nominal_outbound_actions"] and retained[arm]["rank"] == spec["g_perf_rank"],
                f"Arm rank or nominal cost changed: {arm}")
    return retained, {"tolerance": tolerance, "ties_at_best": tied, "legal_candidates": len(scores),
                      "winner": winner["route_id"], "winner_nominal_actions": native["outbound_cost"]}, native, stable_table


def verify_cold_start_excluded(weighted, source, protocol):
    directory = WEIGHTED / "runner/runs/holdout" / LAYOUT / "G_perf"
    manifest, digest = weighted.engine.verify_manifest(directory, "manifest.json")
    trace = read(directory / "steps.json")
    weighted.native.verify_trace(trace, source["initial"], BUDGET, manifest)
    actual = legacy.first_difference(source["trace"], trace, 0, ("action",))
    require(actual["paid"] == protocol["cold_start_exclusion"]["g_perf_first_action_difference_from_S_at_paid"] < FORK,
            "Cold-start G_perf did not diverge at the declared earlier step")
    return {"first_action_difference_from_S": actual, "manifest_sha256": digest, "used_as_arm": False}


def source_inputs(protocol):
    require(protocol["schema"] == "nso_route_top_geometry_measurement_v1"
            and (protocol["layout"], protocol["fork_paid"], protocol["selection_call_id"]) == (LAYOUT, FORK, CALL)
            and protocol["hard_paid_budget_including_return"] == BUDGET
            and protocol["target_facility"] == "open31" and protocol["surface_match_distance_m"] == .05,
            "Unexpected measurement definition")
    for relative, expected in protocol["fixed_inputs_sha256"].items():
        require(sha(STUDY / relative) == expected, f"Frozen evaluation source changed: {relative}")
    binding = read(STUDY / "source/evaluator_bindings.json")
    require(binding["source_commit"] == protocol["source_commit"] and binding["layout"] == LAYOUT,
            "Published source binding disagrees with the measurement plan")
    for absolute, digest in binding["evaluator_files_sha256"].items():
        require(sha(Path(absolute)) == digest, f"Published evaluation source changed: {absolute}")
    actor_anchor = check_bound(binding["actor_anchor"])
    previous_protocol = read(PREVIOUS_PROTOCOL)
    weighted, metric = legacy.load_evaluator(previous_protocol)
    source = legacy.load_source(weighted, LAYOUT, previous_protocol["cases"][LAYOUT])
    reference = weighted.load_reference(LAYOUT, source["initial"])
    ranking, ties, native, score_table = audited_ranking(source, protocol)
    cold = verify_cold_start_excluded(weighted, source, protocol)
    prior_cases = read(PREVIOUS / "evaluation/results/measured.json")["cases"]
    cycle = prior_cases["paired_cycle_20"]
    require(cycle["G_perf_ranks"]["geometry"]["rank"] == protocol["cycle_20"]["geometry_route_g_perf_rank"],
            "Cycle-20 geometry rank is inconsistent with the published comparison")
    cycle_scope = cycle["primary_target"]
    cycle_context = {"source_commit": protocol["source_commit"], "geometry_rank": 1,
                     "primary_target": cycle_scope, "common_paid": cycle["shared_route_evaluation_paid"],
                     "category_terminal": cycle["arms"]["category"]["terminal"]["facilities"][cycle_scope],
                     "geometry_terminal": cycle["arms"]["geometry"]["terminal"]["facilities"][cycle_scope]}
    category = verify_existing(weighted, source, binding["prior_category"], "category")
    geometry = verify_existing(weighted, source, binding["prior_equal_cost_geometry"], "geometry")
    fork_path = check_bound(binding["prior_category"]["fork_ready"])
    require(read(fork_path) == read(check_bound(binding["prior_equal_cost_geometry"]["fork_ready"])),
            "Archived fork identities disagree")
    forks = read(fork_path)
    require(forks["identity"]["actual_paid"] == FORK and forks["identity"]["selection_call_id"] == CALL,
            "Wrong fork identity")
    source.update({"fork_ready": forks, "actor_anchor_sha256": sha(actor_anchor), "native": native,
                   "ranked_table": score_table, "ranked_arms": ranking, "ties": ties,
                   "cold_start_exclusion": cold, "cycle20_context": cycle_context})
    return weighted, metric, source, reference, {"category": category, "geometry_equal_cost": geometry}


def verify_seal(path):
    seal = read(path)
    inputs = seal["evaluator_files_sha256"]
    for required in (PROTOCOL, Path(__file__), STUDY / "source/evaluator_bindings.json"):
        require(inputs[str(required.resolve())] == sha(required), f"Evaluator input changed: {required}")
    require(seal["measurement_protocol_sha256"] == sha(PROTOCOL)
            and seal["schedule"] == [{"layout": LAYOUT, "arm": "native", "route_id": "native"}],
            "Runtime scope differs from the one-arm measurement plan")
    return sha(path)


def verify_native(weighted, directory, source, freeze_sha):
    manifest, digest = weighted.engine.verify_manifest(directory, "manifest.json")
    require(manifest["arm"] == "native" and manifest["layout"] == LAYOUT
            and manifest["status"] == "completed_execution" and manifest["native_budget"] == BUDGET
            and manifest["runner_freeze_sha256"] == freeze_sha
            and manifest["actual_closed_loop"] and not manifest["evaluator_inputs_read_by_actor"]
            and manifest["global_mode"] == "G" and manifest["local_continuation_mode"] == "G_perf"
            and (manifest["unknown_weight"], manifest["action_cost"]) == (.25, .20),
            "Native branch run binding or continuation drift")
    run = trial_binding(directory, manifest, manifest["route_end"])
    require(run["receipt_sha256"] == digest, "Native manifest changed during verification")
    trace = run["trace"]
    initial = source["initial"]
    weighted.native.verify_trace(trace, initial, BUDGET, manifest)
    weighted.native.verify_action_guard(trace, BUDGET)
    require(read(directory / "initial_raw_rgbd_scan_sha256.json") == source["raw_initial"]
            and read(directory / "fork_ready.json") == source["fork_ready"], "Native fork identity differs from archive")
    for archived, actual in zip(source["trace"][:FORK], trace[:FORK], strict=True):
        require(all(archived[key] == actual[key] for key in legacy.PREFIX_FIELDS),
                f"Native prefix motion, pose, or observation differs at {archived['step']}")
    with np.load(directory / "observed_belief_trace.npz", allow_pickle=False) as stored:
        require(np.array_equal(stored["paid"], np.arange(len(trace) + 1)), "Native belief clocks differ from trace")
        belief = stored["belief"]
    require(np.array_equal(belief[:FORK + 1], source["belief_prefix"]), "Native belief prefix differs from source")
    action = read(directory / "intervention.json")
    require(action["actual_paid"] == action["historical_prefix_actions_verified"] == FORK
            and action["selection_call_id"] == CALL and action["route_id"] == action["G_perf_first_route_id"] == "native"
            and action["outbound_sha256"] == source["native"]["outbound_sha256"] and action["selected_g_perf_ordinal"] == 1
            and action["complete_G_perf_rank_table"] == source["ranked_table"]
            and action["candidate_pool_sha256"] == source["anchor"]["candidate_pool_sha256"]
            and action["target_pose"] == source["target_pose"] and action["return_anchor"] == source["fork_ready"]["identity"]["return_anchor"]
            and action["selection_mode"] == "G_perf" and action["no_forced_route"] is True and action["applied_once"] is False,
            "First native selection did not match the archived full rank-one route")
    choices = read(directory / "local_choices.json")
    require(not any(row["choice_audit"].get("single_route_intervention") for row in choices), "Native route was forced")
    chosen = [row for row in choices if row["choice_audit"].get("native_g_perf_selection")]
    require(len(chosen) == 1 and chosen[0]["selection_call_id"] == CALL
            and chosen[0]["active_local_route_id"] == "native", "No native G_perf selection at the declared call")
    for row in choices:
        if row["selection_call_id"] < CALL:
            continue
        policy = row["choice_audit"]["weighted_ranking"]
        require((policy["prior"], policy["unknown_weight"], policy["action_cost"]) == ("G", .25, .20)
                and policy["local_area_provider_calls"]["S"] == policy["local_area_provider_calls"]["X"] == 0,
                "Native route or continuation did not use G_perf")
    return run | {"initial": initial, "belief": belief}


def executed_route(run, source, candidate_id):
    end = run["route_end"]
    require(FORK <= end["actual_paid"] <= len(run["trace"]) and end["status"] in ("arrived", "aborted")
            and end["target_pose"] == source["target_pose"], "Invalid initial route event")
    live = run["trace"][FORK:end["actual_paid"]]
    require(len(live) == end["bound_paid_actions"], "Initial route event contains continuation actions")
    candidates = source["choice"]["choice_audit"]["candidates"]
    selected = next(row for row in candidates if row["route_id"] == candidate_id)
    planned = selected["outbound_actions"]
    require(all(row["action"] == planned[index] and row["active_option_id"] == end["option_id"]
                and row["local_route_id"] == candidate_id for index, row in enumerate(live)),
            "Executed prefix disagrees with selected native option")
    fork_length = run["trace"][FORK - 1]["path_length_m"]
    arrival_length = live[-1]["path_length_m"] if live else fork_length
    require(end["status"] != "arrived" or (
        bool(live) and len(live) == len(planned) and [*live[-1]["position"], live[-1]["heading"]] == source["target_pose"]),
        "Claimed initial arrival reached a different pose or omitted actions")
    if "bound_path_length_m" in end:
        require(math.isclose(end["bound_path_length_m"], arrival_length - fork_length, abs_tol=1e-12),
                "Native bound route length differs from charged trace")
    return end | {"bound_path_length_m": arrival_length - fork_length, "collisions": sum(int(row["collision"]) for row in live)}


def first_difference(left, right, keys):
    event = legacy.first_difference(left["trace"], right["trace"], FORK, keys)
    if event is None:
        return None
    return {key.replace("geometry_", "comparator_"): value for key, value in event.items()}


def same_clock(weighted, metric, reference, source, run, paid, replay, verify_replay):
    if paid > len(run["trace"]):
        return {"status": "missing_arm_terminated_before_equal_clock", "requested_paid": paid}, None
    for name in ("route_end", f"paid_{paid}", "terminal", "fork"):
        receipt = run["manifest"]["checkpoints"].get(name, {})
        if receipt.get("status") == "captured" and receipt["actual_paid"] == paid:
            return legacy.verify_measurement(weighted, metric, reference, run, name, paid)
    require((replay / "manifest.json").is_file(), f"No exact capture for {run['directory']} at paid {paid}: {replay}")
    derived = verify_replay(replay)
    manifest = derived["manifest"]
    require(manifest["verified_against_original_manifest_sha256"] == run["receipt_sha256"]
            and manifest["original_full_trace_verified"] is True
            and manifest["files_sha256"]["terminal/native_mesh.ply"] == run["manifest"]["files_sha256"]["terminal/native_mesh.ply"]
            and np.array_equal(derived["belief"], run["belief"]), "Extra replay did not verify original execution")
    for a, b in zip(derived["trace"], run["trace"], strict=True):
        require(all(a[key] == b[key] for key in legacy.PREFIX_FIELDS), "Extra replay actual trace changed")
    require(read(derived["directory"] / "intervention.json") == read(run["directory"] / "intervention.json"),
            "Extra replay selection differs from original")
    return legacy.verify_measurement(weighted, metric, reference, derived, f"paid_{paid}", paid)


def partition(before, category, comparator, weight):
    category_new, comparator_new = category & ~before, comparator & ~before
    category_lost, comparator_lost = before & ~category, before & ~comparator
    masks = {
        "common_new": category_new & comparator_new,
        "category_only_new": category_new & ~comparator_new,
        "comparator_only_new": comparator_new & ~category_new,
        "both_lost": category_lost & comparator_lost,
        "category_only_lost": category_lost & ~comparator_lost,
        "comparator_only_lost": comparator_lost & ~category_lost,
    }
    stats = {key: legacy.nonnegative_counts(mask, weight) for key, mask in masks.items()}
    return stats | {"category_minus_comparator_new_m2":
                    (int(category_new.sum()) - int(comparator_new.sum())) * weight}, masks


def retention(selected, own, other, weight):
    comparison = {"own_only_at_terminal": selected & own & ~other, "both_at_terminal": selected & own & other,
                  "other_only_at_terminal": selected & ~own & other, "neither_at_terminal": selected & ~own & ~other}
    require(sum(int(mask.sum()) for mask in comparison.values()) == int(selected.sum()), "Retention partition changed")
    return {key: legacy.nonnegative_counts(mask, weight) for key, mask in comparison.items()}, comparison


def evaluate(protocol, weighted, metric, source, reference, runs, args, freeze_sha):
    runs["geometry_top"] = verify_native(weighted, args.run, source, freeze_sha)
    verifiers = {key: (lambda directory, arm=spec["old_arm"]: legacy.verify_root(
        weighted, directory, LAYOUT, arm, source, FORK, sha(OLD_FREEZE)))
        for key, spec in protocol["arms"].items() if "old_arm" in spec}
    verifiers["geometry_top"] = lambda directory: verify_native(weighted, directory, source, freeze_sha)
    explicit_replay = {"category": args.category_replay, "geometry_equal_cost": args.geometry_replay, "geometry_top": args.native_replay}
    cache = {}

    def measure(arm, checkpoint, paid):
        key = arm, paid
        if key not in cache:
            cache[key] = legacy.verify_measurement(weighted, metric, reference, runs[arm], checkpoint, paid)
        return cache[key]

    first, all_masks = measure("category", "fork", FORK)
    before = all_masks[FACILITY]
    weight = reference["weights"][FACILITY]
    output_masks = {"points_xyz_m": reference["samples"][FACILITY], "area_per_sample_m2": np.array(weight),
                    "fork_represented": before}
    evaluated, terminals = {}, {}

    def with_facility(measurement, masks):
        if masks is None:
            return measurement
        return measurement | {"facility": legacy.facility_metrics(before, masks[FACILITY], weight)}

    for arm, run in runs.items():
        fork_receipt, _ = measure(arm, "fork", FORK)
        require(fork_receipt["mesh_sha256"] == first["mesh_sha256"]
                and weighted.native.exact_common_observation(first["receipt"], fork_receipt["receipt"]),
                f"Recovered fork surface differs for {arm}")
        route = executed_route(run, source, protocol["arms"][arm]["route_id"])
        event, event_mask = measure(arm, "route_end", route["actual_paid"])
        end, end_mask = measure(arm, "terminal", len(run["trace"]))
        closed = weighted.native.close_record(end["receipt"], read(run["directory"] / "runtime_summary.json"),
                                               run["trace"], run["initial"])
        require((closed["actual_paid"], closed["returned"], closed["collisions"], closed["terminal_pose"]) ==
                (run["manifest"]["paid_actions"], run["manifest"]["returned"], run["manifest"]["collisions"],
                 run["manifest"]["final_pose"]), f"Terminal native manifest disagrees for {arm}")
        output_masks[f"{arm}_route_event_represented"] = event_mask[FACILITY]
        output_masks[f"{arm}_terminal_represented"] = end_mask[FACILITY]
        terminals[arm] = end_mask[FACILITY]
        evaluated[arm] = {"route": route, "route_event": with_facility(event, event_mask),
                          "terminal": with_facility(end | closed, end_mask), "manifest_sha256": run["receipt_sha256"],
                          "native_run_directory": str(run["directory"]), "fork": fork_receipt["receipt"]}

    comparisons = {}
    for identifier, (base, other) in protocol["comparisons"].items():
        pair = {"category": base, "comparator": other}
        common_paid = max(evaluated[arm]["route"]["actual_paid"] for arm in pair.values())
        exact, masks = {}, {}
        for label, arm in pair.items():
            replay = explicit_replay[arm] or args.replay_root / arm / f"paid_{common_paid}"
            key = arm, common_paid
            if key not in cache:
                cache[key] = same_clock(weighted, metric, reference, source, runs[arm], common_paid, replay, verifiers[arm])
            exact[label], masks[label] = cache[key]
            exact[label] = with_facility(exact[label], masks[label])
            if masks[label] is not None:
                output_masks[f"{identifier}_common_{arm}_represented"] = masks[label][FACILITY]
        diff = {f"first_{name}_difference": first_difference(runs[base], runs[other], keys)
                for name, keys in (("action", ("action",)), ("position", ("position",)),
                                   ("pose", ("position", "heading")), ("raw_observation", legacy.RAW_FIELDS))}
        terminal_surface, terminal_parts = partition(before, terminals[base], terminals[other], weight)
        output_masks.update({f"{identifier}_terminal_{name}": mask for name, mask in terminal_parts.items()})
        summary = {"category_arm": base, "comparator_arm": other, "common_paid": common_paid, "common_checkpoint": exact,
                   "continuation_actions_at_common": {arm: max(0, min(common_paid, len(runs[arm]["trace"])) -
                                                               evaluated[arm]["route"]["actual_paid"]) for arm in pair.values()},
                   "terminal_surface": terminal_surface, "terminal_equal_paid": (
                       evaluated[base]["terminal"]["actual_paid"] == evaluated[other]["terminal"]["actual_paid"]),
                   "terminal_shared_paid": None, **diff}
        if summary["terminal_equal_paid"]:
            summary["terminal_shared_paid"] = evaluated[base]["terminal"]["actual_paid"]
        if all(mask is not None for mask in masks.values()):
            common_stats, common_parts = partition(before, masks["category"][FACILITY], masks["comparator"][FACILITY], weight)
            output_masks.update({f"{identifier}_common_{name}": mask for name, mask in common_parts.items()})
            summary["common_surface"] = common_stats
            summary["retention"] = {}
            for arm, peer, membership in ((base, other, "category_only_new"), (other, base, "comparator_only_new")):
                stats, parts = retention(common_parts[membership], terminals[arm], terminals[peer], weight)
                summary["retention"][arm] = stats
                output_masks.update({f"{identifier}_retention_{arm}_{name}": mask for name, mask in parts.items()})
        comparisons[identifier] = summary

    return {"schema": "nso_route_top_geometry_measured_v1", "source_commit": protocol["source_commit"],
            "run_freeze_sha256": freeze_sha, "measurement_protocol_sha256": sha(PROTOCOL), "evaluator_sha256": sha(Path(__file__)),
            "layout": LAYOUT, "fork_paid": FORK, "selection_call_id": CALL, "target": protocol["target_facility"],
            "fork_identity_sha256": source["fork_ready"]["fork_identity_sha256"], "fork_mesh_sha256": first["mesh_sha256"],
            "g_perf_ranks": source["ranked_arms"], "g_perf_tie": source["ties"], "fork_facility": legacy.facility_metrics(before, before, weight),
            "arms": evaluated, "comparisons": comparisons, "cold_start_exclusion": source["cold_start_exclusion"],
            "cycle20_published_context": source["cycle20_context"]}, output_masks


def write_table(path, report):
    def row(name, values):
        return [name, *(values.get(arm) for arm in ARM_NAMES)]

    def extract(*keys):
        collected = {}
        for arm, source in report["arms"].items():
            value = source
            for key in keys:
                value = value.get(key, {}) if isinstance(value, dict) else None
            collected[arm] = value if value != {} else None
        return collected

    rows = [row("route_id", {key: rank["route_id"] for key, rank in report["g_perf_ranks"].items()}),
            row("G_perf_rank_of_34", {key: rank["rank"] for key, rank in report["g_perf_ranks"].items()}),
            row("nominal_outbound_actions", {key: rank["extra_outbound_actions"] + report["g_perf_tie"]["winner_nominal_actions"]
                                             for key, rank in report["g_perf_ranks"].items()})]
    for key in ("status", "reason", "actual_paid", "bound_paid_actions", "bound_path_length_m", "collisions"):
        rows.append(row("initial_route_" + key, extract("route", key)))
    for phase in ("route_event", "terminal"):
        for key in ("gross_new_m2", "lost_m2", "net_m2", "recall_05cm"):
            rows.append(row(f"{phase}_open31_{key}", extract(phase, "facility", key)))
        for key in ("C_free", "F1_scene_05cm", "J"):
            rows.append(row(f"{phase}_{key}", extract(phase, key)))
    for key in ("actual_paid", "path_length_m", "returned", "collisions", "stop_reason"):
        rows.append(row("terminal_" + key, extract("terminal", key)))
    for name, pair in report["comparisons"].items():
        category, comparator = pair["category_arm"], pair["comparator_arm"]
        rows.append(row(name + "_common_paid", {category: pair["common_paid"], comparator: pair["common_paid"]}))
        rows.append(row(name + "_continuation_count_at_common", pair["continuation_actions_at_common"]))
        for difference in ("action", "position", "pose", "raw_observation"):
            match = pair[f"first_{difference}_difference"]
            paid = match["paid"] if match else None
            rows.append(row(f"{name}_first_actual_{difference}_difference_paid", {category: paid, comparator: paid}))
        for key in ("gross_new_m2", "lost_m2", "net_m2", "recall_05cm"):
            rows.append(row(f"{name}_common_open31_{key}", {
                category: pair["common_checkpoint"]["category"].get("facility", {}).get(key),
                comparator: pair["common_checkpoint"]["comparator"].get("facility", {}).get(key)}))
        for phase in ("common", "terminal"):
            surface = pair.get(phase + "_surface", {})
            for label in ("common_new", "both_lost"):
                area = surface.get(label, {}).get("area_m2")
                rows.append(row(f"{name}_{phase}_{label}_m2", {category: area, comparator: area}))
            for label in ("only_new", "only_lost"):
                rows.append(row(f"{name}_{phase}_{label}_m2", {category: surface.get("category_" + label, {}).get("area_m2"),
                                                                 comparator: surface.get("comparator_" + label, {}).get("area_m2")}))
            rows.append(row(f"{name}_{phase}_signed_difference_new_m2",
                            {comparator: surface.get("category_minus_comparator_new_m2")}))
        for label in ("own_only_at_terminal", "both_at_terminal", "other_only_at_terminal", "neither_at_terminal"):
            values = {arm: matches[label]["area_m2"] for arm, matches in pair.get("retention", {}).items()}
            rows.append(row(f"{name}_common_exclusive_{label}_m2", values))

    with path.open("x", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(("metric", *ARM_NAMES))
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--freeze", type=Path, default=STUDY / "RUN_FREEZE.json")
    parser.add_argument("--run", type=Path, default=STUDY / "runner/runs" / LAYOUT / "native")
    parser.add_argument("--replay-root", type=Path, default=STUDY / "runner/replays")
    parser.add_argument("--native-replay", type=Path)
    parser.add_argument("--category-replay", type=Path)
    parser.add_argument("--geometry-replay", type=Path)
    parser.add_argument("--output", type=Path, default=HERE / "results")
    parser.add_argument("--check-inputs", action="store_true")
    args = parser.parse_args()
    protocol = read(PROTOCOL)
    weighted, metric, source, reference, runs = source_inputs(protocol)
    if args.check_inputs:
        print(f"{LAYOUT}: {len(reference['samples'][FACILITY])} evaluable shelf samples, exact prior two-arm fork, "
              f"rank-one native among {source['ties']['legal_candidates']} legal candidates")
        return
    freeze_sha = verify_seal(args.freeze)
    report, arrays = evaluate(protocol, weighted, metric, source, reference, runs, args, freeze_sha)
    args.output.mkdir(parents=True, exist_ok=False)
    mask_path = args.output / "surface_masks.npz"
    np.savez_compressed(mask_path, **arrays)
    report["surface_masks_sha256"] = sha(mask_path)
    table = args.output / "three_arms.csv"
    write_table(table, report)
    report["three_arm_table_sha256"] = sha(table)
    with (args.output / "measured.json").open("x") as stream:
        json.dump(report, stream, indent=2, allow_nan=False, ensure_ascii=False)
        stream.write("\n")
    print(f"Evaluated all three {LAYOUT} arms from actual mesh snapshots.")


if __name__ == "__main__":
    main()
