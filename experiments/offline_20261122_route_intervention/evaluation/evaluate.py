#!/usr/bin/env python3
"""Measure the two predeclared route interventions from native fused meshes."""

import argparse
import csv
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path

for key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[key] = "1"

import numpy as np

HERE = Path(__file__).resolve().parent
STUDY = HERE.parent
ANALYSES = STUDY.parent
WEIGHTED = ANALYSES / "weighted_geometric_control_20261122"
DEFAULT_PROTOCOL = HERE / "MEASUREMENT_PROTOCOL.json"
RUN_FREEZE = STUDY / "RUN_FREEZE.json"
ANCHORS = STUDY / "source/frozen_anchors.json"
TARGETS = STUDY / "source/evaluator_bindings.json"
ARM_NAMES = ("category", "geometry")
FIXTURES = {"open31": "shelf", "closed118": "cabinet"}
PREFIX_FIELDS = (
    "step", "action", "position", "heading", "collision", "packet_sha256", "packet_chain_sha256",
    "frame_sha256", "nonsemantic_packet_sha256", "owner_sha256", "depth_values_sha256",
    "rgb_values_sha256", "scan_ranges_sha256", "observed_known_cells", "observed_free_cells",
)
RAW_FIELDS = ("rgb_values_sha256", "depth_values_sha256", "scan_ranges_sha256")


def read(path):
    return json.loads(path.read_text())


def sha(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def require(condition, explanation):
    if not condition:
        raise ValueError(explanation)


def load_evaluator(protocol):
    require(protocol["schema"] == "nso_route_intervention_measurement_v1"
            and protocol["surface_match_distance_m"] == .05
            and protocol["hard_paid_budget_including_return"] == 192
            and tuple(protocol["arms"]) == ARM_NAMES, "Unexpected measurement definition")
    inputs = dict(protocol["metric_and_reference_bindings_sha256"])
    for case in protocol["cases"].values():
        inputs.update(case["published_input_sha256"])
    for relative, expected in inputs.items():
        require(sha(ANALYSES / relative) == expected, f"Archived input checksum mismatch: {relative}")
    spec = importlib.util.spec_from_file_location("weighted_measurement",
                                                  WEIGHTED / "measurement/evaluate.py")
    weighted = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(weighted)
    files = weighted.files
    metric = files.load_metric(read(files.SCENES / "freeze.json"), read(files.OLD_EVAL / "freeze.json"))
    require(files.THRESHOLD_M == protocol["surface_match_distance_m"], "Matching threshold drift")
    return weighted, metric


def load_source(weighted, layout, case):
    root = WEIGHTED / "runner/runs/holdout" / layout / "S"
    source, digest = weighted.engine.verify_manifest(root, "manifest.json")
    trace = read(root / "steps.json")
    initial = read(WEIGHTED / "scenes/prepared" / layout / "initial_state.json")
    weighted.native.verify_trace(trace, initial, 192, source)
    raw_initial = read(root / "initial_raw_rgbd_scan_sha256.json")
    fork = case["fork_paid"]
    local = [entry for entry in read(root / "local_choices.json")
             if entry["selection_call_id"] == case["selection_call_id"]]
    require(len(local) == 1 and local[0]["paid_before"] == fork, f"Wrong published fork: {layout}")
    choice = local[0]
    with np.load(root / "observed_belief_trace.npz", allow_pickle=False) as stored:
        require(np.array_equal(stored["paid"], np.arange(len(trace) + 1)), f"Bad source clocks: {layout}")
        belief = stored["belief"][:fork + 1]
    expected_belief = hashlib.sha256(np.ascontiguousarray(belief[-1]).tobytes()).hexdigest()
    require(expected_belief == choice["observed_belief_sha256"], f"Source fork belief mismatch: {layout}")
    anchor = read(ANCHORS)["scenes"][layout]
    target = read(TARGETS)["scenes"][layout]
    audit = choice["choice_audit"]
    require(target["target_facility_key"] == case["primary_target"]
            and anchor["paid_before"] == fork
            and anchor["selection_call_id"] == choice["selection_call_id"]
            and anchor["candidate_pool_sha256"] == audit["candidate_pool_sha256"]
            and anchor["archived_S_score_table_sha256"] == audit["score_table_sha256"]
            and anchor["goal_pose"] == choice["same_global_target"],
            f"Measurement target or presealed anchor changed: {layout}")
    return {"manifest_sha256": digest, "trace": trace, "raw_initial": raw_initial, "belief_prefix": belief,
            "initial": initial, "choice": choice, "target_pose": choice["same_global_target"], "anchor": anchor}


def nonnegative_counts(mask, weight):
    return {"samples": int(np.count_nonzero(mask)), "area_m2": int(np.count_nonzero(mask)) * weight}


def facility_metrics(before, after, weight):
    gross = after & ~before
    lost = before & ~after
    represented = nonnegative_counts(after, weight)
    visible_area = len(after) * weight
    return {
        "evaluable_samples": len(after), "evaluable_m2": visible_area,
        "fork_represented_m2": nonnegative_counts(before, weight)["area_m2"],
        "represented_samples": represented["samples"], "represented_m2": represented["area_m2"],
        "recall_05cm": represented["area_m2"] / visible_area,
        "gross_new_samples": int(np.count_nonzero(gross)), "gross_new_m2": int(np.count_nonzero(gross)) * weight,
        "lost_samples": int(np.count_nonzero(lost)), "lost_m2": int(np.count_nonzero(lost)) * weight,
        "net_m2": (int(np.count_nonzero(gross)) - int(np.count_nonzero(lost))) * weight,
    }


def combined_facility(measures):
    keys = measures[0].keys() - {"recall_05cm"}
    result = {key: sum(measure[key] for measure in measures) for key in sorted(keys)}
    return result | {"recall_05cm": result["represented_m2"] / result["evaluable_m2"]}


def verify_root(weighted, directory, layout, arm, source, fork, freeze_sha):
    manifest_path = directory / "manifest.json"
    if not manifest_path.is_file():
        failure = directory / "failure.json"
        require(failure.is_file(), f"No actor terminal or failure receipt: {directory}")
        return {"technical_failure": read(failure), "receipt_sha256": sha(failure), "directory": directory}
    manifest, digest = weighted.engine.verify_manifest(directory, "manifest.json")
    require(manifest["status"] == "completed_execution" and manifest["layout"] == layout and manifest["arm"] == arm
            and manifest["native_budget"] == 192 and not manifest["evaluator_inputs_read_by_actor"]
            and manifest["actual_closed_loop"] and manifest["global_mode"] == "G"
            and manifest["local_continuation_mode"] == "G_perf"
            and (manifest["unknown_weight"], manifest["action_cost"]) == (.25, .20)
            and manifest["source_protocol_sha256"] == sha(STUDY / "PROTOCOL.json"),
            f"Inconsistent intervention receipt: {directory}")
    if freeze_sha:
        require(manifest["runner_freeze_sha256"] == freeze_sha, f"Actor used a different runtime seal: {directory}")
    trace = read(directory / "steps.json")
    initial = source["initial"]
    weighted.native.verify_trace(trace, initial, 192, manifest)
    weighted.native.verify_action_guard(trace, 192)
    require(len(trace) >= fork and all(all(field in step and step[field] is not None for field in RAW_FIELDS)
                                       for step in trace), f"Unverified charged trace: {directory}")
    raw_start = read(directory / "initial_raw_rgbd_scan_sha256.json")
    require(raw_start == source["raw_initial"], f"Initial RGB-D/scan differs from original: {directory}")
    for archived, enacted in zip(source["trace"][:fork], trace[:fork], strict=True):
        require(all(archived[key] == enacted[key] for key in PREFIX_FIELDS),
                f"Observation, pose or action differs from original at paid {enacted['step']}: {directory}")
    with np.load(directory / "observed_belief_trace.npz", allow_pickle=False) as archive:
        require(np.array_equal(archive["paid"], np.arange(len(trace) + 1)),
                f"Belief ledger and charged trace disagree: {directory}")
        grid = archive["belief"]
    require(np.array_equal(grid[:fork + 1], source["belief_prefix"]), f"Historical belief prefix drift: {directory}")
    anchor, action = source["anchor"], read(directory / "intervention.json")
    selected = anchor["routes"][arm]
    require(action["applied_once"] and action["actual_paid"] == fork
            and action["historical_prefix_actions_verified"] == fork
            and action["selection_call_id"] == anchor["selection_call_id"]
            and action["route_id"] == selected["route_id"]
            and action["outbound_sha256"] == selected["outbound_sha256"]
            and action["forced_outbound_actions"] == selected["outbound_actions"]
            and action["forced_outbound_states"] == selected["outbound_states"]
            and action["target_pose"] == anchor["goal_pose"] and action["return_anchor"] == anchor["return_anchor"]
            and action["candidate_pool_sha256"] == anchor["candidate_pool_sha256"]
            and action["archived_S_score_table_sha256"] == anchor["archived_S_score_table_sha256"],
            f"The recorded route intervention differs from the presealed legal route: {directory}")
    epochs = read(directory / "local_choices.json")
    injected = [row for row in epochs if row["choice_audit"].get("single_route_intervention")]
    require(len(injected) == 1 and injected[0]["selection_call_id"] == action["selection_call_id"],
            f"The local trace does not contain one intervention: {directory}")
    for epoch in epochs:
        if epoch["selection_call_id"] <= action["selection_call_id"]:
            continue
        chosen = epoch["choice_audit"]["weighted_ranking"]
        require((chosen["prior"], chosen["unknown_weight"], chosen["action_cost"]) == ("G", .25, .20)
                and chosen["local_area_provider_calls"]["S"] == chosen["local_area_provider_calls"]["X"] == 0,
                f"Continuation changed local policy: {directory}")
    route_end = manifest["route_end"]
    paid_end = route_end["actual_paid"]
    require(route_end["status"] in ("arrived", "aborted") and fork <= paid_end <= len(trace)
            and route_end["reason"], f"Missing actual first-route outcome: {directory}")
    return {"manifest": manifest, "receipt_sha256": digest, "trace": trace, "belief": grid,
            "directory": directory, "route_end": route_end, "initial": initial}


def verify_measurement(weighted, metric, reference, run, name, expected_paid):
    directory, manifest = run["directory"], run["manifest"]
    receipt, files, _, receipt_hash = weighted.engine.snapshot(directory, name, manifest, "native_observed")
    require(receipt["status"] == "captured" and receipt["actual_paid"] == expected_paid,
            f"Expected captured checkpoint at paid {expected_paid}: {directory}/{name}")
    weighted.engine.verify_checkpoint_against_trace(receipt, expected_paid, run["trace"], run["initial"])
    actual_step = run["trace"][expected_paid - 1] if expected_paid else None
    expected_chain = (actual_step["packet_chain_sha256"] if actual_step else
                      weighted.files.chain_sha256("0" * 64, run["initial"]["initial_packet_sha256"]))
    require(receipt["packet_chain_sha256"] == expected_chain, f"Checkpoint packet chain differs: {directory}/{name}")
    saved_belief = weighted.files.read_belief(files["belief.npz"])[0]
    require(np.array_equal(saved_belief, run["belief"][expected_paid]), f"Capture belief mismatch: {directory}/{name}")
    measurements, masks = weighted.engine.evaluate_verified_snapshot(
        files["native_mesh.ply"], files["belief.npz"], receipt, reference, metric)
    require(math.isclose(measurements["J"], measurements["C_free"] * measurements["F1_scene_05cm"],
                         rel_tol=0, abs_tol=1e-12), f"Global quality arithmetic differs: {directory}/{name}")
    return {"receipt": receipt, "mesh_path": str(files["native_mesh.ply"]),
            "mesh_sha256": receipt["files_sha256"]["native_mesh.ply"],
            "checkpoint_sha256": receipt_hash, "status": "measured",
            **measurements}, masks


def first_difference(a, b, start, keys):
    for left, right in zip(a[start:], b[start:]):
        if any(left[key] != right[key] for key in keys):
            return {"paid": left["step"], "category_action": left["action"], "geometry_action": right["action"],
                    "category_pose": [*left["position"], left["heading"]],
                    "geometry_pose": [*right["position"], right["heading"]],
                    "category_active_option": left["active_option_id"],
                    "geometry_active_option": right["active_option_id"],
                    "category_local_route": left["local_route_id"], "geometry_local_route": right["local_route_id"]}
    return None


def pair_groups(reference, baseline, category, geometry):
    arrays = {}
    groups = {}
    for source_key, name in FIXTURES.items():
        before, a, b = (snap[name] for snap in (baseline, category, geometry))
        a_new, b_new = a & ~before, b & ~before
        common = a_new & b_new
        a_only, b_only = a_new & ~b_new, b_new & ~a_new
        weight = reference["weights"][name]
        groups[source_key] = {key: nonnegative_counts(mask, weight)
                              for key, mask in (("common_new", common), ("category_only_new", a_only),
                                                ("geometry_only_new", b_only))}
        require(int((common | a_only).sum()) == int(a_new.sum())
                and int((common | b_only).sum()) == int(b_new.sum()), "Invalid new-surface partition")
        tag = np.zeros(len(before), dtype=np.uint8)
        tag[before] = 4
        tag[common] = 1
        tag[a_only] = 2
        tag[b_only] = 3
        arrays.update({f"{source_key}_points_xyz_m": reference["samples"][name],
                       f"{source_key}_fork_represented": before,
                       f"{source_key}_category_represented": a,
                       f"{source_key}_geometry_represented": b,
                       f"{source_key}_class_new": tag,
                       f"{source_key}_area_per_sample_m2": np.array(weight)})
    groups["all_facilities"] = {
        key: {field: sum(group[key][field] for group in groups.values()) for field in ("samples", "area_m2")}
        for key in ("common_new", "category_only_new", "geometry_only_new")}
    return groups, arrays


def retention_by_fixture(reference, common_arrays, terminal_category, terminal_geometry):
    retained = {}
    for source_key, name in FIXTURES.items():
        tag = common_arrays[f"{source_key}_class_new"]
        weight = reference["weights"][name]
        a, b = terminal_category[name], terminal_geometry[name]
        retained[source_key] = {}
        for arm, code, own, other in (("category", 2, a, b), ("geometry", 3, b, a)):
            selected = tag == code
            retained[source_key][arm] = {
                label: nonnegative_counts(selected & mask, weight)
                for label, mask in (("own_only_at_terminal", own & ~other), ("both_at_terminal", own & other),
                                    ("other_only_at_terminal", ~own & other), ("neither_at_terminal", ~own & ~other))}
    return retained


def add_capture(weighted, metric, reference, evaluations, measurements, name, run, capture_name, paid):
    result, masks = verify_measurement(weighted, metric, reference, run, capture_name, paid)
    evaluations[name], measurements[name] = result, masks


def read_equal_capture(weighted, metric, reference, run, paid, replay_directory, source, layout, arm, fork, freeze_sha):
    if len(run["trace"]) < paid:
        return {"status": "missing_arm_terminated_before_equal_clock", "requested_paid": paid}, None
    for name in ("route_end", f"paid_{paid}", "terminal", "fork"):
        checkpoint = run["directory"] / name / "checkpoint.json"
        if f"{name}/checkpoint.json" not in run["manifest"]["files_sha256"]:
            continue
        row = read(checkpoint)
        if row["status"] == "captured" and row["actual_paid"] == paid:
            return verify_measurement(weighted, metric, reference, run, name, paid)
    name = f"paid_{paid}"
    if not (replay_directory / "manifest.json").is_file():
        raise FileNotFoundError(f"Missing native capture at paid {paid} for {layout}/{arm}: {replay_directory}")
    replay = verify_root(weighted, replay_directory, layout, arm, source, fork, freeze_sha)
    replay_manifest = replay["manifest"]
    require(replay_manifest.get("verified_against_original_manifest_sha256") == run["receipt_sha256"]
            and replay_manifest.get("original_full_trace_verified") is True,
            f"Unbound exact-capture replay: {replay_directory}")
    passed = all(all(a[key] == b[key] for key in PREFIX_FIELDS)
                 for a, b in zip(run["trace"], replay["trace"], strict=True))
    require(passed and np.array_equal(run["belief"], replay["belief"])
            and run["manifest"]["files_sha256"]["terminal/native_mesh.ply"]
            == replay_manifest["files_sha256"]["terminal/native_mesh.ply"],
            f"Capture replay changed original action, belief, sensor or terminal mesh: {replay_directory}")
    return verify_measurement(weighted, metric, reference, replay, name, paid)


def evaluate_case(weighted, metric, layout, case, run_root, replay_root, freeze_sha):
    source = load_source(weighted, layout, case)
    ref = weighted.load_reference(layout, source["initial"])
    fork_paid = case["fork_paid"]
    runs = {arm: verify_root(weighted, run_root / layout / arm, layout, arm, source, fork_paid, freeze_sha)
            for arm in ARM_NAMES}
    evaluations, masks = dict.fromkeys(ARM_NAMES), dict.fromkeys(ARM_NAMES)
    equal_paid = max(run["route_end"]["actual_paid"] for run in runs.values()) if all(
        "route_end" in run for run in runs.values()) else None
    for arm, run in runs.items():
        measurements, present = {}, {}
        evaluations[arm], masks[arm] = measurements, present
        if "technical_failure" in run:
            measurements["terminal"] = {"status": "technical_failure", "failure": run["technical_failure"]}
            continue
        add_capture(weighted, metric, ref, measurements, present, "fork", run, "fork", fork_paid)
        fork = measurements["fork"]["receipt"]
        archived_choice = source["choice"]
        require(fork["belief_sha256"] == archived_choice["observed_belief_sha256"]
                and fork["pose"] == archived_choice["pose"] and fork["packet_sha256"] == archived_choice["packet_sha256"],
                f"Fork checkpoint differs from published source choice: {layout}/{arm}")
        endpoint_paid = run["route_end"]["actual_paid"]
        add_capture(weighted, metric, ref, measurements, present, "route_end", run, "route_end", endpoint_paid)
        route_pose = measurements["route_end"]["receipt"]["pose"]
        if run["route_end"]["status"] == "arrived":
            require(route_pose == source["target_pose"], f"Arrival differs from common target: {layout}/{arm}")
        measurements["route_end"]["route_event"] = run["route_end"]
        measurements["route_end"]["path_length_m"] = run["trace"][endpoint_paid - 1]["path_length_m"] if endpoint_paid else 0.0
        if equal_paid is not None:
            replay = replay_root / layout / arm / f"paid_{equal_paid}"
            measurements["equal_clock"], present["equal_clock"] = read_equal_capture(
                weighted, metric, ref, run, equal_paid, replay, source, layout, arm, fork_paid, freeze_sha)
        else:
            measurements["equal_clock"] = {"status": "missing_paired_route_outcome"}
        add_capture(weighted, metric, ref, measurements, present, "terminal", run, "terminal", len(run["trace"]))
        terminal = weighted.native.close_record(measurements["terminal"]["receipt"],
                 read(run["directory"] / "runtime_summary.json"), run["trace"], run["initial"])
        measures = measurements["terminal"]
        measures.update(terminal)
        require((terminal["actual_paid"], terminal["returned"], terminal["collisions"], terminal["terminal_pose"])
                == (run["manifest"]["paid_actions"], run["manifest"]["returned"], run["manifest"]["collisions"],
                    run["manifest"]["final_pose"]), f"Terminal outcome is inconsistent: {layout}/{arm}")
        for clock, mask in present.items():
            if mask is None:
                continue
            groups = {source_key: facility_metrics(present["fork"][name], mask[name], ref["weights"][name])
                      for source_key, name in FIXTURES.items()}
            groups["all_facilities"] = combined_facility(list(groups.values()))
            measurements[clock]["facilities"] = groups
    pair, surface_payloads = {}, {}
    if all("fork" in measure for measure in evaluations.values()):
        a, b = (evaluations[arm]["fork"] for arm in ARM_NAMES)
        require(a["mesh_sha256"] == b["mesh_sha256"]
                and weighted.native.exact_common_observation(a["receipt"], b["receipt"])
                and all(np.array_equal(masks["category"]["fork"][key], masks["geometry"]["fork"][key])
                        for key in masks["category"]["fork"]),
                f"The arms did not share the exact fused pre-action state: {layout}")
        traces = [runs[arm]["trace"] for arm in ARM_NAMES]
        pair["first_actual_action_difference"] = first_difference(*traces, fork_paid, ("action",))
        pair["first_actual_pose_difference"] = first_difference(*traces, fork_paid, ("position", "heading"))
        pair["first_actual_raw_observation_difference"] = first_difference(*traces, fork_paid, RAW_FIELDS)
        for clock in ("equal_clock", "terminal"):
            a, b = [masks[arm].get(clock) for arm in ARM_NAMES]
            if a is not None and b is not None:
                pair[clock], surface_payloads[clock] = pair_groups(ref, masks["category"]["fork"], a, b)
        if "equal_clock" in surface_payloads and "terminal" in surface_payloads:
            pair["retention"] = retention_by_fixture(ref, surface_payloads["equal_clock"],
                            masks["category"]["terminal"], masks["geometry"]["terminal"])
    bindings = {arm: {"run_manifest_or_failure_sha256": run["receipt_sha256"], "run_directory": str(run["directory"])}
                for arm, run in runs.items()}
    return {"primary_target": case["primary_target"], "target_binding": case["target_binding"],
            "fork_paid": fork_paid, "shared_route_evaluation_paid": equal_paid, "same_target_pose": source["target_pose"],
            "selection_call_id": case["selection_call_id"], "source_manifest_sha256": source["manifest_sha256"],
            "legal_source_routes": source["anchor"]["allowed_route_count"],
            "G_perf_ranks": source["anchor"]["G_perf_rank"], "route_geometry": bindings,
            "arms": evaluations, "pair": pair}, surface_payloads


def flatten_case(layout, result):
    target = result["primary_target"]
    row = {"layout": layout, "primary_target": target, "fork_paid": result["fork_paid"],
           "selection_call_id": result["selection_call_id"], "same_target_pose": result["same_target_pose"],
           "equal_clock_paid": result["shared_route_evaluation_paid"], "legal_source_routes": result["legal_source_routes"]}
    pair = result["pair"]
    for name in ("action", "pose", "raw_observation"):
        first = pair.get(f"first_actual_{name}_difference")
        row[f"first_{name}_difference_paid"] = first["paid"] if first is not None else None
    for arm in ARM_NAMES:
        row[f"{arm}_route_id"] = result["G_perf_ranks"][arm]["route_id"]
        row[f"{arm}_G_perf_rank"] = result["G_perf_ranks"][arm]["rank"]
        for phase in ("route_end", "equal_clock", "terminal"):
            evaluation = result["arms"][arm].get(phase, {})
            row[f"{phase}_{arm}_status"] = evaluation.get("status")
            row[f"{phase}_{arm}_paid"] = evaluation.get("receipt", {}).get("actual_paid")
            metrics = evaluation.get("facilities", {}).get(target, {})
            for key in ("gross_new_m2", "lost_m2", "net_m2", "recall_05cm"):
                row[f"{phase}_{arm}_{key}"] = metrics.get(key)
            if phase == "route_end":
                event = evaluation.get("route_event", {})
                row.update({f"{phase}_{arm}_{key}": event.get(key) for key in ("reason",)})
                row[f"{phase}_{arm}_route_outcome"] = event.get("status")
            if phase == "terminal":
                for key in ("C_free", "F1_scene_05cm", "J", "path_length_m", "returned", "collisions", "stop_reason"):
                    row[f"terminal_{arm}_{key}"] = evaluation.get(key)
    terminal_clocks = [row.get(f"terminal_{arm}_paid") for arm in ARM_NAMES]
    row["terminal_shared_paid_clock"] = terminal_clocks[0] if terminal_clocks[0] == terminal_clocks[1] else None
    for phase in ("equal_clock", "terminal"):
        a, b = (row.get(f"{phase}_{arm}_gross_new_m2") for arm in ARM_NAMES)
        row[f"{phase}_category_minus_geometry_m2"] = a - b if a is not None and b is not None else None
        for name in ("common_new", "category_only_new", "geometry_only_new"):
            row[f"{phase}_{name}_m2"] = pair.get(phase, {}).get(target, {}).get(name, {}).get("area_m2")
    for arm in ARM_NAMES:
        entry = pair.get("retention", {}).get(target, {}).get(arm, {})
        for key, value in entry.items():
            row[f"retention_{arm}_{key}_m2"] = value["area_m2"]
    return row


def snapshot_rows(results):
    for layout, result in results.items():
        for arm, checkpoints in result["arms"].items():
            for checkpoint, measurement in checkpoints.items():
                common = {"layout": layout, "arm": arm, "checkpoint": checkpoint,
                          "status": measurement["status"],
                          "actual_paid": measurement.get("receipt", {}).get("actual_paid"),
                          "C_free": measurement.get("C_free"), "F1_scene_05cm": measurement.get("F1_scene_05cm"),
                          "J": measurement.get("J"), "mesh_path": measurement.get("mesh_path"),
                          "mesh_sha256": measurement.get("mesh_sha256"),
                          "checkpoint_sha256": measurement.get("checkpoint_sha256")}
                if "facilities" not in measurement:
                    yield common | {"scope": None}
                for scope, entry in measurement.get("facilities", {}).items():
                    yield common | {"scope": scope, "primary_target": scope == result["primary_target"], **entry}


def write_csv(path, rows):
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("x", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, default=DEFAULT_PROTOCOL)
    parser.add_argument("--run-root", type=Path, default=STUDY / "runs")
    parser.add_argument("--replay-root", type=Path, default=STUDY / "replays")
    parser.add_argument("--run-freeze", type=Path, default=RUN_FREEZE)
    parser.add_argument("--output", type=Path, default=HERE / "results")
    parser.add_argument("--check-inputs", action="store_true")
    args = parser.parse_args()
    protocol = read(args.protocol)
    weighted, metric = load_evaluator(protocol)
    if args.check_inputs:
        for layout, case in protocol["cases"].items():
            source = load_source(weighted, layout, case)
            ref = weighted.load_reference(layout, source["initial"])
            print(layout, case["primary_target"], len(ref["samples"][FIXTURES[case["primary_target"]]]))
        return
    freeze_sha = sha(args.run_freeze)
    freeze = read(args.run_freeze)
    require(freeze["measurement_protocol_sha256"] == sha(args.protocol)
            and freeze["evaluator_files_sha256"][str(Path(__file__).resolve())] == sha(Path(__file__)),
            "Evaluator or measurement protocol differs from the pre-launch freeze")
    results, payloads = {}, {}
    for layout, case in protocol["cases"].items():
        result, surfaces = evaluate_case(weighted, metric, layout, case, args.run_root, args.replay_root, freeze_sha)
        results[layout], payloads[layout] = result, surfaces
    args.output.mkdir(parents=True, exist_ok=False)
    surface_paths = args.output / "surface_masks"
    surface_paths.mkdir()
    masks = {}
    for layout, captures in payloads.items():
        for clock, arrays in captures.items():
            path = surface_paths / f"{layout}_{clock}.npz"
            np.savez_compressed(path, **arrays)
            masks[str(path.relative_to(args.output))] = sha(path)
    write_csv(args.output / "paired_primary.csv", [flatten_case(layout, result) for layout, result in results.items()])
    write_csv(args.output / "snapshots.csv", list(snapshot_rows(results)))
    output = {"schema": "nso_route_intervention_measured_v1", "measurement_protocol_sha256": sha(args.protocol),
              "runtime_freeze_sha256": freeze_sha, "evaluator_sha256": sha(Path(__file__)), "cases": results,
              "surface_mask_codes": {"0": "absent at fork and absent in both arms", "1": "new in both arms",
                                     "2": "new only in category arm", "3": "new only in geometry arm",
                                     "4": "represented at fork"},
              "surface_mask_files_sha256": masks}
    with (args.output / "measured.json").open("x") as stream:
        json.dump(output, stream, indent=2, ensure_ascii=False, allow_nan=False)
        stream.write("\n")
    print(f"Measured {len(results)} fixed route-intervention cases from native mesh snapshots.")


if __name__ == "__main__":
    main()
