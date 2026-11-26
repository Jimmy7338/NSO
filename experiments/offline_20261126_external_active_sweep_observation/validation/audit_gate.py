"""Independently audit the sealed HM3D active-observation gate trajectory."""

import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
from scipy.sparse.csgraph import dijkstra


HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
ANALYSES = ROOT.parent
PREVIOUS = ANALYSES / "external_online_sweep_20261126"
V1 = ANALYSES / "external_support_repair_20261126"
SCENE = "b26_hm3d_00770_s1"
ARM = "region_S"
LIMIT = 192
BASELINE_FREE = 522
REFERENCE_SHA256 = "b732f50b502b9222a3dabc735ca29651a8d69658d451898d4965f8ac08a79060"
INITIAL_PACKET_SHA256 = "63d86ab1402515040c3af50173a14389a206b367d17a01e88d31150ea975ac64"
FROM = (52, 26, 2)
TO = (53, 26, 2)


def sha(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def read(path):
    return json.loads(path.read_bytes())


def require(condition, message):
    if not condition:
        raise ValueError(message)


def pose(row):
    return (*row["position"], row["heading"])


def verify_frozen(freeze_path):
    freeze = read(freeze_path)
    jobs = [(row["layout"], row["arm"]) for row in freeze["jobs"]]
    require(jobs == [(SCENE, ARM)] and freeze["budget"] == LIMIT,
            "Gate must seal one fixed 192-action HM3D start 1 region_S run")
    previous = read(PREVIOUS / "FORMAL_FREEZE.json")
    counts = {}
    for group in ("actor_files", "actor_initial_files", "simulator_only_files", "evaluator_files"):
        records = freeze[group]
        for name, digest in records.items():
            require(sha(Path(name)) == digest, f"Frozen input differs after run: {name}")
        counts[group] = len(records)
        if group != "actor_files":
            require(records == previous[group], f"Gate changed prior protected inputs in {group}")
    protocol = freeze_path.parent / freeze["protocol_relative"]
    require(sha(protocol) == freeze["protocol_sha256"], "Gate protocol hash differs")
    return freeze, counts


def verify_manifests(run, freeze_path, freeze):
    attempt = read(run / "attempt.json")
    manifest = read(run / "manifest.json")
    require(attempt["initial_packet_sha256"] == manifest["initial_packet_sha256"] == INITIAL_PACKET_SHA256,
            "Gate did not use the same cold sensor packet")
    require(attempt["runner_freeze_sha256"] == sha(freeze_path),
            "The run did not consume the gate's prospective freeze")
    require(attempt["layout"] == manifest["layout"] == SCENE and attempt["arm"] == manifest["arm"] == ARM,
            "Run identity does not match fixed registration")
    require(manifest["cold_started_at_paid_zero"] and manifest["actual_closed_loop"]
            and not manifest["evaluator_inputs_read_by_actor"], "Gate requires independent closed-loop execution from paid zero")
    examined = ("steps.json", "audit.json", "observed_belief_trace.npz", "initial_raw_rgbd_scan_sha256.json", "attempt.json")
    for filename in examined:
        require(sha(run / filename) == manifest["files_sha256"][filename],
                f"The native manifest no longer matches {filename}")
    return examined, manifest


def audit_events(run):
    actions = read(run / "steps.json")
    require([step["step"] for step in actions] == list(range(1, len(actions) + 1)),
            "The native trajectory is missing action steps")
    require(len(actions) <= LIMIT, "Gate exceeded 192 paid actions")
    audit = read(run / "audit.json")
    approvals = {event["next_action_id"]: event for event in audit if event["event"] == "paid_action_authorized"}
    dispatches = [e for e in audit if e["event"] == "active_observation_dispatched"]
    acquisitions = [e for e in audit if e["event"] == "active_observation_acquired"]
    replans = [e for e in audit if e["event"] == "active_observation_crossing_replanned"]
    crossings = [e for e in audit if e["event"] == "active_observed_edge_executed"
                 and e["from_cell"] == list(FROM[:2]) and e["to_cell"] == list(TO[:2])]
    require(dispatches and acquisitions and replans and crossings,
            "Missing a dispatch, acquired observation, replan, or executed edge")
    crossing = crossings[0]
    edge_paid = int(crossing["edge_paid"])
    scan_paid = int(crossing["scan_paid"])
    require(0 < scan_paid < edge_paid <= len(actions), "Edge must follow an actually acquired observation")
    acquired = next((event for event in acquisitions if event["paid"] == scan_paid), None)
    require(acquired is not None, "Could not bind the crossing to its acquisition")
    candidates = [row for row in dispatches if row["paid_before"] < scan_paid and row["selected_view"] == acquired["pose"]]
    require(candidates, "Could not identify dispatch for actual novel viewpoint")
    dispatched = candidates[-1]
    dispatch_paid = int(dispatched["paid_before"])
    expected_actions = dispatched["outbound_actions"]
    expected_states = dispatched["outbound_states"]
    require(dispatch_paid > 0 and dispatch_paid + len(expected_actions) == scan_paid,
            "Known outbound route was not fully tracked to acquisition")
    require(expected_states == [list(pose(row)) for row in actions[dispatch_paid-1:scan_paid]]
            and expected_actions == [row["action"] for row in actions[dispatch_paid:scan_paid]],
            "Recorded actions did not reach the dispatched viewpoint along the certified route")
    require(all(not actions[paid-1]["collision"] and actions[paid-1]["guard_allowed"]
                and approvals[paid]["assessment"]["allowed"] for paid in range(dispatch_paid+1, scan_paid+1)),
            "The observation route contains an unguarded or failed action")
    observed = actions[scan_paid-1]
    require(dispatched["pose"] == list(pose(actions[dispatch_paid-1]))
            and dispatched["packet_sha256"] == actions[dispatch_paid-1]["packet_sha256"],
            "Dispatch does not match the observation from which the route was planned")
    require(acquired["packet_sha256"] == observed["packet_sha256"] == crossing["scan_packet_sha256"]
            and acquired["depth_values_sha256"] == observed["depth_values_sha256"]
            and acquired["scan_ranges_sha256"] == observed["scan_ranges_sha256"],
            "Actual viewpoint packet does not match acquisition and crossing")
    earlier = actions[:scan_paid-1] + [read(run / "initial_raw_rgbd_scan_sha256.json")]
    old_run = PREVIOUS / "execution/formal" / SCENE / ARM / "attempt_001/run"
    comparative = read(PREVIOUS / "HM3D_START1_ONLINE_AUDIT.json")
    previous_record = next(row for row in comparative["arms"] if row["arm"] == ARM)
    require(sha(old_run / "steps.json") == previous_record["v3_steps_sha256"],
            "Reference trajectory used for sensor novelty was modified")
    old_steps = read(old_run / "steps.json")
    earlier_or_registered_control = earlier + old_steps + [read(old_run / "initial_raw_rgbd_scan_sha256.json")]
    require(any(all(row[key] != observed[key] for row in earlier_or_registered_control)
                for key in ("depth_values_sha256", "scan_ranges_sha256")),
            "Viewpoint added no sensor measurement beyond prior actual frames and control run")
    require(all(pose(step) != pose(observed) for step in old_steps + actions[:scan_paid-1]),
            "The selected viewpoint was already visited in the control or this episode")
    crossed = actions[edge_paid-1]
    require(pose(actions[edge_paid-2]) == FROM and pose(crossed) == TO and crossed["action"] == "forward"
            and not crossed["collision"] and crossed["guard_allowed"]
            and approvals[edge_paid]["assessment"]["allowed"],
            "Target edge lacks a collision-free forward action from exact source pose")
    require(crossing["complete_sweep"] is True and crossing["edge_packet_sha256"] == crossed["packet_sha256"],
            "Executed edge log does not identify the charged crossing")
    require(approvals[edge_paid]["assessment"]["next_pose"] == list(TO),
            "Native action authorization target does not match the crossed pose")
    replan = next((row for row in replans if row["paid_before"] == scan_paid), None)
    require(replan is not None and replan["packet_sha256"] == observed["packet_sha256"],
            "Post-observation routed crossing does not reference the new sensor packet")
    require(replan["outbound_states"] == [list(pose(row)) for row in actions[scan_paid-1:edge_paid]]
            and replan["outbound_actions"] == [row["action"] for row in actions[scan_paid:edge_paid]],
            "Actual motions from active observation to crossing diverged from the certified plan")
    require(all(not actions[paid-1]["collision"] and actions[paid-1]["guard_allowed"]
                and approvals[paid]["assessment"]["allowed"] for paid in range(scan_paid+1, edge_paid+1)),
            "The replanned crossing route contains an unguarded or failed action")
    active = (dispatched, acquired, replan, crossing)
    require(dispatched["prior_sweep"] == acquired["prior_sweep"]
            and dispatched["prior_sweep"]["allowed"] is False
            and dispatched["prior_sweep"]["reason"] == "missing_positive_ground_support",
            "Before active sensing the target must fail due to missing positive ground evidence")
    for event in active[1:]:
        certificate = event["target_sweep_certificate"]
        require(certificate["allowed"] and certificate["reason"] == "measured_full_capsule"
                and certificate["probes"] > 0, "Target lacked the full measured sweep certificate")
    return actions, active


def independent_route_and_certificate(run, freeze, actions, events):
    from importlib.util import module_from_spec, spec_from_file_location

    replay_file = PREVIOUS / "replay/reconstruct_history.py"
    spec = spec_from_file_location("active_gate_replay_sensor", replay_file)
    replay = module_from_spec(spec)
    spec.loader.exec_module(replay)
    cold_path = Path(freeze["scene_root"]) / "prepared" / SCENE / "initial_state.json"
    cold = read(cold_path)
    sensor = replay.RecordedPoseSensor(cold, SCENE)
    sys.path.insert(0, str(PREVIOUS))
    from observed_sweep import SweepEvidence
    from nso.cpu_sensor_contract_v10 import GridTransform, SensorPacket
    from nso.observed_runtime_mapper_v10 import ObservedRuntimeMapperV10

    cfg = sensor.config
    grid = GridTransform(sensor.shape, cfg.resolution_m)
    mapper = ObservedRuntimeMapperV10(sensor.shape, cfg, truncation_m=cfg.truncation_m)
    evidence = SweepEvidence(sensor.shape, cfg.resolution_m, cfg.robot_radius_m)
    dispatch, acquired, replan, execution = events
    before, at, crossing = dispatch["paid_before"], acquired["paid"], execution["edge_paid"]
    first = read(run / "initial_raw_rgbd_scan_sha256.json") | {
        "position": cold["anchor"][:2], "heading": cold["anchor"][2], "action": None,
        "collision": False, "done": False,
    }
    sealed = np.load(run / "observed_belief_trace.npz", allow_pickle=False)
    beliefs = sealed["belief"]
    np.testing.assert_array_equal(sealed["paid"], np.arange(len(actions) + 1))
    sealed.close()
    checked = {}
    for index, step in enumerate([first, *actions[:crossing]]):
        sensor.place(step)
        frame, scan = sensor.sense(), sensor.scan()
        for array, key in ((frame.depth_m, "depth_values_sha256"), (frame.color_rgb, "rgb_values_sha256"),
                           (scan.ranges_m, "scan_ranges_sha256")):
            require(replay.hash_array(array) == step[key], f"Raw observation differs at paid {index}, {key}")
        packet = SensorPacket(
            cold["scene_id"], "single_observation_cold_start", f"frame-{index}", index, frame, scan,
            sensor.position, sensor.heading, "current_first_hit_RGBD_scan_visible_ideal_structure_labels",
            "exact_discrete_simulator_odometry", step["action"], bool(step["collision"]), bool(step["done"]),
        ).validate(grid, cfg)
        require(packet.sha256() == step["packet_sha256"], f"Packet provenance differs at paid {index}")
        mapper.update(frame, scan)
        np.testing.assert_array_equal(mapper.belief, beliefs[index], err_msg=f"Observation map at paid {index}")
        evidence.observe(frame, scan)
        if index in (before, at-1, at, crossing-1):
            checked[index] = evidence.raw.certify(evidence.xy(FROM[:2]), evidence.xy(TO[:2]), cfg.robot_radius_m)
        if index in (before, at):
            graph = evidence.build(mapper.belief)
            fields = (np.asarray(graph.safe.shape, dtype="<i8").tobytes(), np.packbits(graph.safe.ravel()).tobytes(),
                      np.packbits(graph.edges.ravel()).tobytes())
            planned = dispatch if index == before else replan
            require(hashlib.sha256(b"".join(fields)).hexdigest() == planned["graph_sha256"],
                    "Independently reconstructed graph disagrees with actor")
            matrix, _, ids = graph.orientation_graph()
            states = planned["outbound_states"]
            for a, b in zip(states, states[1:]):
                left, right = [int(ids[tuple(pose[:2])])*4 + pose[2] for pose in (a, b)]
                require(left >= 0 and right >= 0 and matrix[left, right] == 1, "Outbound plan lacks a measured edge")
            goal, anchor = [int(ids[tuple(p[:2])])*4 + p[2] for p in (states[-1], cold["anchor"])]
            distance = dijkstra(matrix, directed=True, indices=goal)[anchor]
            require(np.isfinite(distance) and distance == planned["known_return_from_plan_goal"]
                    and index + len(states)-1 + distance <= LIMIT,
                    "Outbound observation path did not preserve a return within budget")
    require(checked[before] == dispatch["prior_sweep"] and checked[at] == acquired["target_sweep_certificate"]
            == replan["target_sweep_certificate"] and checked[crossing-1] == execution["target_sweep_certificate"],
            "Reconstructed raw measurements do not validate the rejection-to-certification sequence")
    return {"full_packet_and_observed_belief_checks": crossing+1, "certificate_checks": checked,
            "first_qualified_only_after_endpoint_packet": checked[at-1]["allowed"] is False,
            "truth_arrays_available_to_sensor_reconstruction": False}


def score_common_mask(run, actions, crossing):
    path = V1 / "scenes/evaluator_references" / SCENE / "evaluator_truth.npz"
    require(sha(path) == REFERENCE_SHA256, "Common V1 mask source was modified")
    with np.load(path, allow_pickle=False) as data:
        correct = data["reachable"] & ~data["occupancy"]
        require(np.array_equal(correct, data["reachable"]) and int(correct.sum()) == 1640,
                "The common V1 coverage definition or mask changed")
    with np.load(run / "observed_belief_trace.npz", allow_pickle=False) as data:
        paid = data["paid"]
        beliefs = data["belief"]
    np.testing.assert_array_equal(paid, np.arange(len(actions)+1))
    counts = np.count_nonzero((beliefs == 0) & correct[None, ...], axis=(1, 2))
    gains = [tick for tick in range(crossing, len(counts)) if counts[tick] > BASELINE_FREE]
    require(gains and counts[-1] > BASELINE_FREE, "Terminal coverage did not exceed 522 after crossing the target edge")
    return {"common_reference_sha256": REFERENCE_SHA256, "common_reachable_cells": 1640,
            "baseline_correct_free_cells": BASELINE_FREE, "first_post_crossing_clock_above_baseline": gains[0],
            "correct_free_at_crossing": int(counts[crossing]), "peak_correct_free_cells": int(counts.max()),
            "terminal_correct_free_cells": int(counts[-1]), "paid_actions": len(actions)}


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--run", type=Path, required=True, help="Native run directory containing manifest.json")
    cli.add_argument("--freeze", type=Path, default=ROOT / "GATE_FREEZE.json")
    cli.add_argument("--out", type=Path, default=HERE / "GATE_AUDIT.json")
    args = cli.parse_args()
    result = {"schema": "nso_active_observation_gate_audit_v1", "run": str(args.run.resolve())}
    try:
        freeze, counts = verify_frozen(args.freeze)
        examined, manifest = verify_manifests(args.run, args.freeze, freeze)
        result["input_validation"] = {"freeze_sha256": sha(args.freeze), "verified_file_counts": counts,
            "native_files_sha256": {name: sha(args.run / name) for name in examined}}
        actions, events = audit_events(args.run)
        require(manifest["paid_actions"] == len(actions), "Manifest paid count differs from trajectory")
        result["received_events"] = events
        result["independent_observation_and_sweep"] = independent_route_and_certificate(args.run, freeze, actions, events)
        result["common_mask_coverage"] = score_common_mask(args.run, actions, int(events[-1]["edge_paid"]))
        result["gate_passed"] = True
    except (AssertionError, KeyError, OSError, ValueError) as error:
        result.update(gate_passed=False, failure=str(error))
    temporary = args.out.with_suffix(args.out.suffix + ".tmp")
    temporary.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    temporary.replace(args.out)
    if not result["gate_passed"]:
        raise SystemExit(f"Gate not met: {result['failure']}")
    print("Gate accepted: archived sensor provenance, certified observation and crossing, common-mask coverage")


if __name__ == "__main__":
    main()
