#!/usr/bin/env python3
"""Fit and audit exact local-route reproduction on independently acquired S states."""

import argparse
import csv
from dataclasses import dataclass
from fractions import Fraction
import hashlib
from itertools import product
import json
import math
from pathlib import Path


HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
SENSOR_FIELDS = ("depth_values_sha256", "rgb_values_sha256", "scan_ranges_sha256")
BEHAVIOR_COMPARISONS = (("G", "S"), ("G_route", "S"), ("G_perf", "S"), ("G_perf", "S_perf"), ("S", "X"))


def read(path):
    return json.loads(path.read_text())


def sha256(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def digest(obj):
    return hashlib.sha256(json.dumps(obj, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def require(condition, message):
    if not condition:
        raise ValueError(message)


def write_json(path, rows):
    path.write_text(json.dumps(rows, indent=2, allow_nan=False) + "\n")


def write_csv(path, rows, empty_fields=()):
    fields = tuple(dict.fromkeys(key for row in rows for key in row)) or empty_fields
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


@dataclass(frozen=True)
class Weights:
    unknown_weight: float
    action_cost: float

    @classmethod
    def read(cls, obj):
        return cls(float(obj["unknown_weight"]), float(obj["action_cost"]))

    def mapping(self):
        return {"unknown_weight": self.unknown_weight, "action_cost": self.action_cost}


ZERO = Weights(0.0, 0.0)


def weight_grid(protocol):
    score = protocol["local_score"]
    return [Weights(u, c) for u, c in product(score["unknown_weight"], score["action_cost_m2_per_charged_action"])]


def charged_route(row):
    return row["outbound_actions"], row["outbound_states"]


def score_route(row, native, prior, weights, cell_area):
    area = row["scores"][prior]
    coverage = (row["unknown_union_cells"] - native["unknown_union_cells"]) * cell_area
    extra = row["outbound_cost"] - native["outbound_cost"]
    score = area + weights.unknown_weight * coverage - weights.action_cost * extra
    require(math.isfinite(score), "Non-finite weighted candidate score")
    return score


def rank_pool(admitted, native, prior, weights, cell_area, tolerance):
    """Independent re-ranking on a frozen legal pool, retaining exact archive tie rules."""
    scored = [(row, score_route(row, native, prior, weights, cell_area)) for row in admitted]
    best = max(score for _, score in scored)
    return min((row for row, score in scored if best - score <= tolerance),
               key=lambda row: (row["outbound_cost"], row["route_id"] != native["route_id"], row["outbound_sha256"]))


def checked_file(directory, relative, manifest, sources):
    path = directory / relative
    actual = sha256(path)
    require(actual == manifest["files_sha256"][relative], f"Changed literal run artifact: {path}")
    sources[str(path)] = actual
    return read(path)


def run_directory(runs, split, layout, arm, expected, sources):
    """Read a literal run or an explicit same-building, same-configuration alias."""
    requested = runs / split / layout / arm
    directory = requested
    alias = None
    if (requested / "alias.json").is_file():
        path = requested / "alias.json"
        alias = read(path)
        sources[str(path)] = sha256(path)
        origin = alias["source_arm"]
        require(alias["schema"] == "identical_frozen_configuration_alias_v1" and
                (alias["split"], alias["layout"], alias["arm"]) == (split, layout, arm) and
                (alias["prior"], Weights.read(alias)) == expected and
                Path(origin).name == origin and origin != arm, f"Invalid arm alias: {path}")
        directory = requested.parent / origin
        require(not (directory / "alias.json").exists(), f"Nested alias: {path}")
    manifest_path = directory / "manifest.json"
    if not manifest_path.is_file():
        failure = directory / "failure.json"
        if failure.is_file():
            sources[str(failure)] = sha256(failure)
            if alias is not None:
                require(alias["source_failure_sha256"] == sources[str(failure)] and
                        alias["source_manifest_sha256"] is None, f"Alias failure fingerprint differs: {directory}")
            return directory, None
        raise FileNotFoundError(manifest_path)
    manifest = read(manifest_path)
    sources[str(manifest_path)] = sha256(manifest_path)
    if alias is not None:
        require(alias["source_manifest_sha256"] == sources[str(manifest_path)]
                and alias["run_freeze_sha256"] == manifest["runner_freeze_sha256"]
                and alias["source_failure_sha256"] is None, f"Alias source receipt fingerprint differs: {directory}")
    require(manifest["status"] == "completed_execution" and manifest["study_split"] == split
            and manifest["layout"] == layout and manifest["arm"] == directory.name
            and manifest["global_mode"] == "G" and manifest["cold_started_at_paid_zero"]
            and manifest["actual_closed_loop"] and manifest["all_distinct_local_candidates_assessed"]
            and not manifest["forced_route_or_goal"] and not manifest["evaluator_inputs_read_by_actor"]
            and manifest["native_budget"] == 192 and 0 <= manifest["paid_actions"] <= manifest["native_budget"],
            f"Policy completion and provenance differ from the registered run: {directory}")
    require((manifest["study_prior"], Weights.read(manifest)) == expected, f"Arm configuration mismatch: {requested}")
    return directory, manifest


def checked_s_decision(event, call, tolerance):
    audit = event["choice_audit"]
    pool = audit["candidates"]
    allowed = [row for row in pool if row["admitted"]]
    native = next(row for row in allowed if row["route_id"] == audit["original_route_id"])
    ranking = audit["weighted_ranking"]
    require(ranking["prior"] == "S" and Weights.read(ranking) == ZERO
            and ranking["score_equality_tolerance"] == tolerance,
            "Route imitation must start from the archived unweighted correct category policy")
    require(len({row["outbound_sha256"] for row in pool}) == len(pool)
            == audit["assessed_candidate_count"] == event["full_pool"]["original_distinct_alternates"] + 1
            and not event["full_pool"]["unassessed_after_cap"]
            and len(allowed) == audit["admitted_candidate_count"],
            "Candidate pool unexpectedly truncated or contains duplicate routes")
    require(digest([{k: v for k, v in row.items() if k != "scores"} for row in pool])
            == audit["candidate_pool_sha256"] == event["full_pool"]["full_candidate_pool_sha256"]
            and digest([row["scores"] for row in pool]) == audit["score_table_sha256"]
            == event["full_pool"]["full_score_sha256"],
            "Same-state pool or area scores do not match the sealed audit")
    require(event["global_mode"] == call["outputs"]["mode"] == "G"
            and event["selection_call_id"] == call["call_id"]
            and event["paid_before"] == call["action_id"]
            and event["same_global_target"] == call["outputs"]["selected"]["pose"],
            "Weighted local ranker modified the neutral global target ledger")
    require(all(digest({"outbound_actions": row["outbound_actions"],
                        "outbound_states": row["outbound_states"]}) == row["outbound_sha256"]
                and len(row["outbound_states"]) == row["outbound_cost"] + 1
                and len(row["outbound_actions"]) == row["outbound_cost"]
                and row["outbound_states"][-1] == event["same_global_target"]
                for row in pool), "Candidate outbound route digest or terminal heading changed")
    area = ranking["grid_cell_area_m2"]
    require(abs(area - .2 ** 2) <= 1e-12 and native["outbound_cost"] == ranking["native_outbound_cost"]
            and native["unknown_union_cells"] == ranking["native_unknown_union_cells"],
            "Weighted score uses a different candidate normalization")
    independently_scored = [
        {"route_id": row["route_id"], "area_m2": float(row["scores"]["S"]),
         "delta_unknown_union_m2": (row["unknown_union_cells"] - native["unknown_union_cells"]) * area,
         "extra_outbound_actions": row["outbound_cost"] - native["outbound_cost"],
         "score_m2": score_route(row, native, "S", ZERO, area)}
        for row in allowed
    ]
    require(digest(independently_scored) == ranking["ranking_scores_sha256"] == digest(ranking["scores"]),
            "Stored local S score rows differ from independent scoring of their archived features")
    selected = rank_pool(allowed, native, "S", ZERO, area, tolerance)
    require(selected["route_id"] == event["active_local_route_id"] == audit["selected_route_id"]
            and selected["outbound_sha256"] == event["active_outbound_sha256"]
            and charged_route(selected) == (event["selected_outbound_actions"], event["selected_outbound_states"]),
            "Archived unweighted S ranking is inconsistent with the actual outbound binding")
    return allowed, native, selected, area


def states_on_s(runs, split, layout, tolerance, sources):
    path, manifest = run_directory(runs, split, layout, "S", ("S", ZERO), sources)
    if manifest is None:
        return [], None
    records = checked_file(path, "local_choices.json", manifest, sources)
    events = {record["selection_call_id"]: record for record in records}
    calls = checked_file(path, "global_choices.json", manifest, sources)
    require(len(events) == len(records) == manifest["local_route_choices"] and
            len(calls) == manifest["native_global_choices"], "Local events do not match run-level counts")
    result = []
    for call in calls:
        native = call["outputs"]["selected"]
        event = events.pop(call["call_id"], None)
        info = {"layout": layout, "global_call_id": call["call_id"], "paid_before": call["action_id"],
                "frame_id": call["frame_id"], "global_inputs_sha256": call["input_sha256"],
                "global_outputs_sha256": call["output_sha256"]}
        if native is None:
            require(event is None, "Local scoring was recorded when the global selector selected no route")
            result.append((info | {"status": "no_local_route_opportunity", "legal_route_count": 0}, None))
            continue
        require(event is not None, "A selected global goal has no corresponding local candidate receipt")
        checked = checked_s_decision(event, call, tolerance)
        result.append((info | {"status": "measured_local_route", "packet_sha256": event["packet_sha256"],
                              "belief_sha256": event["observed_belief_sha256"],
                              "candidate_pool_sha256": event["choice_audit"]["candidate_pool_sha256"],
                              "goal_pose": json.dumps(event["same_global_target"]),
                              "legal_route_count": len(checked[0]), "S_route_id": checked[2]["route_id"],
                              "S_outbound_sha256": checked[2]["outbound_sha256"]}, checked))
    require(not events, "Orphaned local choices not tied to the neutral global selection record")
    return result, manifest


def matched_states(states, arms, tolerance):
    for info, choice in states:
        info = dict(info)
        if choice is not None:
            admitted, native, selected, area = choice
            for tag, (prior, weights) in arms.items():
                target = rank_pool(admitted, native, prior, weights, area, tolerance)
                info |= {f"{tag}_route_id": target["route_id"],
                         f"{tag}_outbound_sha256": target["outbound_sha256"],
                         f"{tag}_exact_match": charged_route(target) == charged_route(selected),
                         f"{tag}_winner_score_m2": score_route(target, native, prior, weights, area),
                         f"{tag}_S_route_score_m2": score_route(selected, native, prior, weights, area)}
        yield info


def building_counts(states, name):
    selected = [r for r in states if r["status"] == "measured_local_route"]
    informative = [r for r in selected if r["legal_route_count"] >= 2]
    f = f"{name}_exact_match"
    return {"all_local_match": sum(row[f] for row in selected), "all_local_count": len(selected),
            "multi_route_match": sum(row[f] for row in informative), "multi_route_count": len(informative),
            "no_local_route_calls": len(states) - len(selected)}


def rate(rows, prefix):
    observed = [Fraction(row[f"{prefix}_match"], row[f"{prefix}_count"]) for row in rows if row[f"{prefix}_count"]]
    return sum(observed) / len(observed) if observed else None


def fit(protocol_path, runs, output):
    protocol = read(protocol_path)
    tolerance = protocol["frozen_online_fields"]["score_equality_tolerance"]
    sources = {}
    layouts = protocol["development_layouts"]
    require(not set(layouts).intersection(protocol["heldout_layouts"]), "Development layouts overlap held-out layouts")
    experiment = {layout: states_on_s(runs, "development", layout, tolerance, sources) for layout in layouts}
    require(all(manifest is not None for _, manifest in experiment.values()),
            "Both development S acquisition runs must have a sealed terminal record before route weight fit")
    weights_rows = []
    best_key, chosen = None, None
    for weights in weight_grid(protocol):
        building = []
        for layout in layouts:
            selections = list(matched_states(experiment[layout][0], {"G_route": ("G", weights)}, tolerance))
            building.append({"layout": layout, **building_counts(selections, "G_route")})
        multi_rate, complete_rate = rate(building, "multi_route"), rate(building, "all_local")
        u, c = Fraction(str(weights.unknown_weight)), Fraction(str(weights.action_cost))
        key = (multi_rate if multi_rate is not None else -1,
               complete_rate if complete_rate is not None else -1, -(u / Fraction(".25") + c / Fraction(".20")),
               -u, -c)
        if best_key is None or key > best_key:
            best_key, chosen = key, weights
        for row in building:
            weights_rows.append(weights.mapping() | row |
                {"macro_multi_route": None if multi_rate is None else float(multi_rate),
                 "macro_all_local": None if complete_rate is None else float(complete_rate)})
    output.mkdir(parents=True, exist_ok=True)
    table = output / "development_route_weights.csv"
    for row in weights_rows:
        row["selected_for_route_fit"] = Weights.read(row) == chosen
    write_csv(table, weights_rows)
    chosen_rows = [row for row in weights_rows if row["selected_for_route_fit"]]
    dev_states = output / "development_fitted_states.csv"
    write_csv(dev_states, [row for recordings, _ in experiment.values() for row in matched_states(
        recordings, {"G": ("G", ZERO), "G_route": ("G", chosen)}, tolerance)],
        ("layout", "status", "legal_route_count"))
    fit_record = {
        "status": "fit_completed_on_development_s_states_only",
        "protocol_sha256": sha256(protocol_path), "fit_script_sha256": sha256(Path(__file__)),
        "decision": chosen.mapping(), "development_layouts": layouts,
        "per_building": chosen_rows, "macro_multi_route": chosen_rows[0]["macro_multi_route"],
        "macro_all_local": chosen_rows[0]["macro_all_local"],
        "informative": any(row["multi_route_count"] for row in chosen_rows),
        "no_route_decisions_anywhere": not any(row["all_local_count"] for row in chosen_rows),
        "source_files_sha256": sources,
        "output_files_sha256": {table.name: sha256(table), dev_states.name: sha256(dev_states)},
    }
    write_json(output / "route_fit.json", fit_record)
    print("Frozen decision score", fit_record["decision"], "from", len(chosen_rows), "development layouts")
    return fit_record


def first_diff(rows_a, rows_b, keys):
    for a, b in zip(rows_a, rows_b):
        if any(a[key] != b[key] for key in keys):
            return a, b
    return None


def first_event(event, key):
    return event[0][key] if event is not None else None


def audit_behavior(runs, layout, arm_order, configurations, sources):
    bundle = {}
    for arm in arm_order:
        directory, manifest = run_directory(runs, "holdout", layout, arm, configurations[arm], sources)
        if manifest is None:
            bundle[arm] = directory.name, None, None, None
            continue
        steps = checked_file(directory, "steps.json", manifest, sources)
        initial_raw = checked_file(directory, "initial_raw_rgbd_scan_sha256.json", manifest, sources)
        require([row["step"] for row in steps] == list(range(1, manifest["paid_actions"] + 1))
                and all(row["guard_allowed"] for row in steps), f"Closed-loop command ledger invalid: {directory}")
        bundle[arm] = directory.name, manifest, steps, initial_raw
    results = []
    for a, b in BEHAVIOR_COMPARISONS:
        origin_a, manifest_a, rows_a, initial_a = bundle[a]
        origin_b, manifest_b, rows_b, initial_b = bundle[b]
        common = {"layout": layout, "arm_left": a, "arm_right": b, "literal_arm_left": origin_a,
                  "literal_arm_right": origin_b}
        if rows_a is None or rows_b is None:
            results.append(common | {"status": "technical_failure_missing_sealed_trace"})
            continue
        require(initial_a == initial_b and all(manifest_a[k] == manifest_b[k] for k in
                ("initial_pose", "initial_packet_sha256", "initial_belief_sha256", "actual_scene_seed", "runner_freeze_sha256")),
                f"Same-building closed-loop pair starts from different states: {layout} {a} vs {b}")
        cmd = first_diff(rows_a, rows_b, ("action",))
        pose = first_diff(rows_a, rows_b, ("position", "heading"))
        sensor = first_diff(rows_a, rows_b, SENSOR_FIELDS)
        packet = first_diff(rows_a, rows_b, ("packet_sha256",))
        common_len = min(len(rows_a), len(rows_b))
        identical = cmd is None and pose is None and sensor is None and packet is None and len(rows_a) == len(rows_b)
        results.append(common | {
            "status": "measured_exact_trace_agreement" if identical else "measured_distinct_or_shorter_trace",
            "paid_left": len(rows_a), "paid_right": len(rows_b), "shared_clock_actions": common_len,
            "first_action_diff": first_event(cmd, "step"), "action_left": first_event(cmd, "action"),
            "action_right": cmd[1]["action"] if cmd is not None else None,
            "first_pose_diff": first_event(pose, "step"), "first_raw_sensor_diff": first_event(sensor, "step"),
            "first_packed_observation_diff": first_event(packet, "step"),
            "first_depth_value_diff": first_event(first_diff(rows_a, rows_b, SENSOR_FIELDS[:1]), "step"),
            "first_RGB_value_diff": first_event(first_diff(rows_a, rows_b, SENSOR_FIELDS[1:2]), "step"),
            "first_lidar_value_diff": first_event(first_diff(rows_a, rows_b, SENSOR_FIELDS[2:]), "step"),
            "shorter_trace_arm": (a if len(rows_a) < len(rows_b) else b) if len(rows_a) != len(rows_b) else None,
            "first_unpaired_paid_step": common_len + 1 if len(rows_a) != len(rows_b) else None,
            "exact_through_both_terminals": identical,
        })
    return results


def evaluate(protocol_path, runs, weights_path, output):
    protocol = read(protocol_path)
    tolerance = protocol["frozen_online_fields"]["score_equality_tolerance"]
    sources = {str(weights_path): sha256(weights_path)}
    frozen = read(weights_path)
    require(frozen["status"] == "two_developer_weight_fits_frozen_before_holdout"
            and frozen["has_not_used_holdout"] and frozen["protocol_sha256"] == sha256(protocol_path),
            "Held-out state audit requires immutable weights fit only on the development experiment")
    holdout_freeze = protocol_path.parent / "holdout_freeze.json"
    holdout_binding = read(holdout_freeze)
    require(holdout_binding["split"] == "holdout"
            and holdout_binding["weight_freeze_sha256"] == sources[str(weights_path)]
            and holdout_binding["frozen_unix_s"] >= frozen["sealed_unix_s"],
            "Held-out runtime freeze does not bind the prelaunch fit receipt")
    sources[str(holdout_freeze)] = sha256(holdout_freeze)
    for absolute, expected in frozen["fit_inputs_sha256"].items():
        require(sha256(Path(absolute)) == expected, f"Bound developer fitting input changed: {absolute}")
    decision, performance = Weights.read(frozen["decision"]), Weights.read(frozen["performance"])
    require(decision in weight_grid(protocol) and performance in weight_grid(protocol),
            "Frozen fitted weights are outside registered development grid")
    fit_path = output / "route_fit.json"
    fit_record = read(fit_path)
    sources[str(fit_path)] = sha256(fit_path)
    require(fit_record["decision"] == decision.mapping() and fit_record["protocol_sha256"] == sha256(protocol_path)
            and fit_record["development_layouts"] == protocol["development_layouts"]
            and fit_record["fit_script_sha256"] == sha256(Path(__file__)),
            "The decision weights do not match the S-only development fit")
    for relative, expected in fit_record["output_files_sha256"].items():
        require(sha256(output / relative) == expected, f"Decision-fitting development output changed: {relative}")
    same_state_arms = {"G": ("G", ZERO), "G_route": ("G", decision), "G_perf": ("G", performance),
                       "S_perf": ("S", performance)}
    configurations = same_state_arms | {"S": ("S", ZERO), "X": ("X", ZERO)}
    all_states, counts, behavior = [], [], []
    for layout in protocol["heldout_layouts"]:
        states, manifest = states_on_s(runs, "holdout", layout, tolerance, sources)
        measured = list(matched_states(states, same_state_arms, tolerance))
        all_states.extend(measured)
        for mode in same_state_arms:
            result = building_counts(measured, mode)
            counts.append({"layout": layout, "compared_to_S": mode,
                           "status": "measured" if manifest is not None else "technical_failure_missing_S_trace",
                           **result, "all_local_accuracy": (result["all_local_match"] / result["all_local_count"]
                                                             if result["all_local_count"] else None),
                           "multi_route_accuracy": (result["multi_route_match"] / result["multi_route_count"]
                                                     if result["multi_route_count"] else None)})
        behavior.extend(audit_behavior(runs, layout, protocol["heldout_modes"], configurations, sources))
    audit_rows = [row for row in counts if row["compared_to_S"] == "G_route"]
    primary = rate(audit_rows, "multi_route")
    result = {
        "status": "heldout_common_state_and_actual_behavior_audited",
        "protocol_sha256": sha256(protocol_path), "frozen_weight_file_sha256": sha256(weights_path),
        "frozen_decision_weights": decision.mapping(), "frozen_performance_weights": performance.mapping(),
        "independent_building_count": len(protocol["heldout_layouts"]),
        "macro_G_route_multi_route": None if primary is None else float(primary),
        "macro_G_route_all_local": rate(audit_rows, "all_local"),
        "all_four_informative_and_sealed": len(audit_rows) == 4 and all(
            row["status"] == "measured" and row["multi_route_count"] > 0 for row in audit_rows),
        "source_files_sha256": sources,
    }
    macro_all = result["macro_G_route_all_local"]
    result["macro_G_route_all_local"] = None if macro_all is None else float(macro_all)
    result["route_replication_engineering_gate"] = bool(
        result["all_four_informative_and_sealed"] and primary >= Fraction(".90")
        and all(Fraction(row["multi_route_match"], row["multi_route_count"]) >= Fraction(".80") for row in audit_rows))
    tables = {"heldout_states.csv": all_states, "heldout_imitation_by_building.csv": counts,
              "heldout_actual_first_differences.csv": behavior}
    for name, rows in tables.items():
        write_csv(output / name, rows, ("layout", "status"))
    result["output_files_sha256"] = {name: sha256(output / name) for name in tables}
    write_json(output / "heldout_decision_audit.json", result)
    print(result["status"], "route engineering threshold passed?", result["route_replication_engineering_gate"])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("fit", "evaluate"))
    parser.add_argument("--protocol", type=Path, default=ROOT / "PROTOCOL.json")
    parser.add_argument("--runs", type=Path, default=ROOT / "runner/runs")
    parser.add_argument("--output", type=Path, default=HERE)
    parser.add_argument("--frozen-weights", type=Path, default=ROOT / "fit_weights.json")
    args = parser.parse_args()
    if args.phase == "fit":
        fit(args.protocol.resolve(), args.runs.resolve(), args.output.resolve())
    else:
        evaluate(args.protocol.resolve(), args.runs.resolve(), args.frozen_weights.resolve(), args.output.resolve())


if __name__ == "__main__":
    main()
