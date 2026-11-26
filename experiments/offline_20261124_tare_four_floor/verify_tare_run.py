#!/usr/bin/env python3
"""Independently check the shared action ledger and TARE transport receipts."""

import argparse
from collections import Counter
import csv
import hashlib
import importlib.util
from io import BytesIO
import json
import math
from pathlib import Path
import sys
from urllib.parse import urlsplit


ROOT = Path(__file__).resolve().parent
BASE = ROOT.parent / "region_independent_floors_20261123"
ARM = "TARE_sensor_transfer"
BUDGET = 192
SCANS_PER_KEYPOSE = 5
DIRECTIONS = ((-1, 0), (0, 1), (1, 0), (0, -1))
YAW = (math.pi / 2, 0., -math.pi / 2, math.pi)
CAMERA_FIELDS = ("depth_m", "intrinsic", "world_from_camera")
SCAN_FIELDS = ("ranges_m", "angle_min_rad", "angle_increment_rad", "range_max_m", "world_from_laser")


def require(ok, message):
    if not ok:
        raise ValueError(message)


def read(path):
    return json.loads(path.read_text())


def digest(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def is_digest(value):
    return (isinstance(value, str) and len(value) == 64
            and all(c in "0123456789abcdef" for c in value))


def fmt(value):
    return json.dumps(value, ensure_ascii=False, allow_nan=False, indent=2)


def transport_sha(record):
    for field in ("bridge_packet_sha256", "bridge_packet_npz_sha256", "bridge_npz_sha256", "packet_npz_sha256"):
        if field in record:
            return record[field]
    return record["packet_sha256"] if "sensor_packet_sha256" in record else None


def make_transport_replayer():
    """Render saved executed actions in memory with no TARE or ROS process."""
    import numpy as np

    sys.path.insert(0, str(BASE.parent / "paired_unseen_gain_20261120/execution"))
    import runtime_binding
    old = runtime_binding.install()
    from nso.cpu_sensor_contract_v10 import GridTransform

    spec = importlib.util.spec_from_file_location("nso_tare_transport_checker_scenes", BASE / "scenes/scene.py")
    scenes = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = scenes
    spec.loader.exec_module(scenes)
    old.protocol = lambda: {"layouts": {name: {"scene_id": f"same-goal-holdout-{name}"}
                                        for name in scenes.LAYOUTS}}

    def replay(layout, steps, observations):
        world = scenes.make_world(layout)
        transform = GridTransform(tuple(world.shape), world.config.resolution_m)
        for clock, observed in enumerate(observations):
            if clock:
                action = steps[clock-1]["action"]
                frame, collision, done = world.step(action)
                packet = old.packet(world, transform, frame, layout, action, collision, done)
            else:
                packet = old.packet(world, transform, world.sense(), layout)
            require(packet.action_id == clock and packet.sha256() ==
                    observed.get("sensor_packet_sha256", observed["packet_sha256"]),
                    f"Independent rerender differs from native simulator packet at {layout}/{clock}")
            camera = {f"frame__{key}": getattr(packet.frame, key) for key in CAMERA_FIELDS}
            scan = {f"scan__{key}": getattr(packet.scan, key) for key in SCAN_FIELDS}
            buffer = BytesIO()
            np.savez(buffer, **camera, **scan)
            require(hashlib.sha256(buffer.getbuffer()).hexdigest() == transport_sha(observed),
                    f"Independent sensor NPZ differs from what worker read at {layout}/{clock}")
            expected_depth = np.count_nonzero(np.isfinite(packet.frame.depth_m) & (packet.frame.depth_m > 0))
            ranges = packet.scan.ranges_m
            expected_planar = np.count_nonzero(np.isfinite(ranges) & (ranges > 0)
                                               & (ranges < packet.scan.range_max_m - 1e-6))
            require(observed["rgbd_points"] == expected_depth and observed["planar_hit_points"] == expected_planar,
                    f"Bridge measured-point counts disagree at {layout}/{clock}")
        return len(observations)

    return replay


def observations_check(audit, start, steps, floor):
    ready, observations = audit["ready"], audit["observations"]
    require(ready["status"] == "ready" and urlsplit(ready["master_uri"]).hostname == "127.0.0.1"
            and ready["scans_per_keypose"] == SCANS_PER_KEYPOSE
            and ready["initial_unsensed_waypoint_ignored"] is True,
            "TARE runtime contract or private ROS endpoint mismatch")
    require(len(observations) == len(steps) + 1, "Transport count must include the initial and terminal packets")
    shape, config = floor["shape"], floor["actual_config"]
    res = config["resolution_m"]
    scans = 0
    fresh, expected, timed_out = {}, [], []
    value_bound = 0
    stamped = 0
    for clock, (observation, source) in enumerate(zip(observations, [start, *steps], strict=True)):
        context = f"action={clock}"
        require(observation["status"] == "observed" and observation["action_id"] == clock
                and observation["unique_paid_observations"] == clock + 1,
                f"Missing, replayed or out-of-order acquisition: {context}")
        packet_sha = observation.get("sensor_packet_sha256", observation["packet_sha256"])
        require(packet_sha == source["packet_sha256"], f"Observed packet hash differs from simulator: {context}")
        bridge_sha = transport_sha(observation)
        require(is_digest(bridge_sha) and bridge_sha != packet_sha, f"Missing independent transport hash: {context}")
        array_keys = ("depth_values_sha256", "scan_ranges_sha256")
        if any(key in observation for key in array_keys):
            require(all(observation.get(key) == source[key] for key in array_keys),
                    f"Transport depth/range values differ from simulator: {context}")
            value_bound += 1

        row, col, direction = source["pose"] if clock == 0 else [*source["position"], source["heading"]]
        require(observation["position"] == [row, col] and observation["heading"] == direction,
                f"Transport cell/heading differs from simulator: {context}")
        x, y, z = observation["odometry_xyz"]
        if observation["odometry_reference"] == "laser_xy_yaw_and_camera_height":
            require(math.isclose(z, config["camera_height_m"], abs_tol=1e-7),
                    f"Camera height differs from shared sensor rig: {context}")
        require(math.isfinite(z) and math.isclose(x, (col+.5)*res, abs_tol=1e-7)
                and math.isclose(y, (shape[0]-row-.5)*res, abs_tol=1e-7)
                and math.isclose(math.remainder(observation["odometry_yaw_rad"]-YAW[direction], 2*math.pi),
                                 0., abs_tol=1e-7),
                f"Odometry passed to TARE differs from simulator geometry: {context}")
        rgbd, planar = [observation[k] for k in ("rgbd_points", "planar_hit_points")]
        require(type(rgbd) is int and type(planar) is int and 0 <= planar <= config["laser_rays"]
                and 0 <= rgbd <= config["width_px"] * config["height_px"],
                f"Point counts exceed sensor capacity: {context}")
        nonempty = bool(rgbd + planar)
        scans += nonempty
        due = nonempty and scans % SCANS_PER_KEYPOSE == 0
        require(observation["published_registered_scans"] == scans
                and observation["expected_native_planning_due"] is due,
                f"Native nonempty keypose cadence mismatch: {context}")
        complete = observation["native_keypose_completed"]
        require((not complete or due) and observation["native_keypose_completion_timeout"] is (due and not complete),
                f"Completion was attributed to the wrong keypose: {context}")
        if due:
            expected.append(clock)
        if due and not complete:
            timed_out.append(clock)
        waypoint = observation["waypoint"]
        if observation["fresh_waypoint"]:
            require(complete and waypoint["source_action_id"] == clock
                    and waypoint["frame_id"].lstrip("/") == "map" and waypoint["valid"] is True
                    and len(waypoint["xyz"]) == 3 and all(math.isfinite(x) for x in waypoint["xyz"]),
                    f"Fresh waypoint is noncausal, out of frame or invalid: {context}")
            virtual_clock = observation.get("virtual_clock_s", observation.get("clock_s"))
            waypoint_stamp = waypoint.get("stamp_s")
            if virtual_clock is not None and waypoint_stamp is not None:
                require(0 < waypoint_stamp <= virtual_clock + 1e-6,
                        f"Stamped waypoint arrived from a future clock: {context}")
                stamped += 1
            fresh[clock] = waypoint
        elif waypoint is not None:
            require(waypoint["source_action_id"] < clock and fresh.get(waypoint["source_action_id"]) == waypoint,
                    f"Cached waypoint did not originate in an earlier confirmed keypose: {context}")
        for path in observation["paths"].values():
            origin = path["source_action_id"]
            require(origin <= clock and origin in expected and observations[origin]["native_keypose_completed"],
                    f"Native path was attached to an unobserved or unfinished keypose: {context}")

    require(value_bound in (0, len(observations)), "Only a subset of acquisitions has direct array hashes")
    return fresh, expected, timed_out, value_bound, stamped


def actions_check(run, manifest, steps, start, terminal, resolution, frontier_events):
    audit = read(run / "audit.json")
    authorizations = [event for event in audit if event["event"] == "paid_action_authorized"]
    require(len(authorizations) == len(steps), "Paid action lacks one-to-one common guard authorization")
    pose = manifest["initial_pose"]
    initial = pose[:]
    chain = hashlib.sha256(bytes(32)+bytes.fromhex(manifest["initial_packet_sha256"])).hexdigest()
    require(start["packet_chain_sha256"] == chain and start["actual_paid"] == 0, "Initial packet chain mismatch")
    collisions = distance_cells = return_actions = 0
    returning = False
    for clock, (step, receipt) in enumerate(zip(steps, authorizations, strict=True), 1):
        heading = pose[2]
        command = step["action"]
        require(0 <= heading < 4 and command in ("forward", "right", "left")
                and step["step"] == receipt["next_action_id"] == clock
                and receipt["phase"] == step["phase_before_action"]
                and receipt["phase"] in ("outbound", "return")
                and (not returning or receipt["phase"] == "return"),
                f"Unauthorized action or phase transition at {clock}")
        returning = receipt["phase"] == "return"
        check = receipt["assessment"]
        reserve = step["reserved_return_cost"]
        require(check["allowed"] is step["guard_allowed"] is True and check["action"] == command
                and check["reason"] == step["guard_reason"] and check["reserved_return_cost"] == reserve
                and check["remaining_budget"] == BUDGET-clock+1 and (reserve is None or 0 <= reserve <= BUDGET-clock),
                f"Common guard or return reserve mismatch at {clock}")
        if command == "forward":
            dr, dc = DIRECTIONS[heading]
            proposed = [pose[0]+dr, pose[1]+dc, heading]
        else:
            proposed = [*pose[:2], (heading+(1 if command == "right" else -1)) % 4]
        require(check["next_pose"] == proposed, f"Guard authorized a different movement at {clock}")
        expected = pose if command == "forward" and step["collision"] else proposed
        pose = [*step["position"], step["heading"]]
        require(expected == pose, f"Actual primitive transition invalid at {clock}")
        chain = hashlib.sha256(bytes.fromhex(chain)+bytes.fromhex(step["packet_sha256"])).hexdigest()
        require(step["packet_chain_sha256"] == chain, f"Broken packet chain at {clock}")
        collisions += bool(step["collision"])
        return_actions += returning
        distance_cells += command == "forward" and not step["collision"]

    stop = manifest["observed_native_termination"]
    returned = pose == initial
    require(len(steps) == manifest["paid_actions"] == stop["final_action_id"] == terminal["actual_paid"]
            and 0 <= len(steps) <= manifest["native_budget"] == BUDGET
            and stop["returned_to_anchor"] is manifest["returned"] is returned
            and terminal["pose"] == manifest["final_pose"] == pose
            and terminal["packet_chain_sha256"] == manifest["final_packet_chain_sha256"] == chain
            and terminal["packet_sha256"] == (steps[-1]["packet_sha256"] if steps else start["packet_sha256"])
            and collisions == manifest["collisions"]
            and math.isclose(distance_cells*resolution, manifest["path_length_m"], abs_tol=1e-7),
            "Endpoint pose, coverage clock, packet chain, collisions, or path length mismatch")
    guards = Counter((event.get("assessment") or {}).get("reason", event["event"])
                     for event in frontier_events
                     if event["event"] in ("guard_rejection", "return_guard_rejection"))
    return {"actual_paid": len(steps), "actual_returns": returned, "return_actions": return_actions,
            "collision_actions": collisions, "terminal_reason": stop["reason"], "terminal_failed": stop["failed"],
            "common_guard_reasons": dict(guards), "final_packet_chain_sha256": chain}


def calls_check(calls, steps, fresh, expected, observations, frontier_events):
    modes, reasons = Counter(), Counter()
    mapped = []
    previous = -1
    for call in calls:
        clock = call["paid"]
        require(previous <= clock <= len(steps), "Native waypoint selection clocks are not monotone")
        previous = clock
        waypoint = call["waypoint"]
        if waypoint is not None:
            origin = waypoint["source_action_id"]
            require(origin <= clock and fresh.get(origin) == waypoint,
                    f"Consumed waypoint not published by a causal, confirmed native keypose: {clock}")
        if call["chosen"] is None:
            reasons[call.get("reason", "no_mapped_waypoint")] += 1
        else:
            require(waypoint is not None, f"Adapter selected a target without TARE output at {clock}")
            modes[call["chosen"]["routing"]] += 1
            mapped.append(call)

    dispatched, reached = {}, set()
    for event in frontier_events:
        if event["event"] == "instrumented_frontier_dispatch":
            goal, clock = event["goal_count"], event["paid_before"]
            require(1 <= goal <= len(mapped) and 0 <= clock < len(steps),
                    "Common driver dispatched an unregistered native way point")
            selected = mapped[goal-1]
            step = steps[clock]
            require(selected["paid"] <= clock and selected["chosen"]["executable_pose"] == event["goal_pose"]
                    and event["action"] == step["action"] and step["phase_before_action"] == "outbound"
                    and event["guard"]["allowed"] is True and step["guard_allowed"] is True,
                    "Dispatched motion does not bind a causal native selection and guarded step")
            if goal not in dispatched:
                dispatched[goal] = selected["chosen"]["routing"]
        elif event["event"] == "instrumented_frontier_observation":
            goal, clock = event["goal_count"], event["paid"]
            require(goal in dispatched and 1 <= clock <= len(steps) and goal not in reached,
                    "Waypoint arrival precedes dispatch or is counted more than once")
            step = steps[clock-1]
            require(event["goal_reached"] is True and event["actual_pose"] == [*step["position"], step["heading"]]
                    == mapped[goal-1]["chosen"]["executable_pose"], "Waypoint arrival disagrees with the sensor pose")
            reached.add(goal)
    boot_end = expected[0] if expected else len(steps)
    boot = [s for s in steps if s["step"] <= boot_end and s["phase_before_action"] == "outbound"]
    require(all(s["action"] in ("left", "right") and s["position"] == observations[0]["position"] for s in boot),
            "Bootstrap moved the robot before native keypose/waypoint support")
    return {"boot_turns": len(boot), "mapped_selection_calls": dict(modes),
            "dispatched_distinct_targets": dict(Counter(dispatched.values())),
            "reached_distinct_targets": dict(Counter(dispatched[goal] for goal in reached)),
            "selection_stop_reasons": dict(reasons)}


def compare_metrics(results, layout, result):
    def rows(name):
        with (results / name).open(newline="") as stream:
            return [r for r in csv.DictReader(stream) if r["layout"] == layout]

    episode, = rows("episodes.csv")
    paid = result["actual_paid"]
    require(episode["arm"] == ARM and int(episode["actual_paid"]) == paid
            and episode["status"] == "completed_execution"
            and episode["actual_return"] == str(result["actual_returns"])
            and int(episode["collisions"]) == result["collision_actions"]
            and episode["stop_reason"] == result["terminal_reason"],
            "Common evaluator episode and native motion ledger differ")
    curve = rows("coverage.csv")
    require([int(row["paid"]) for row in curve] == list(range(paid+1)), "Common evaluator omitted paid clocks")
    earliest = next((row["paid"] for row in curve
                     if int(row["correct_free_cells"])*20 >= int(row["reachable_cells"])*19), "")
    require(earliest == episode["A95_paid"], "A95 does not agree with integer ground-truth coverage recount")
    checkpoints = rows("checkpoints.csv")
    require({p["checkpoint"] for p in checkpoints} == {"start", "paid_48", "paid_96", "paid_144", "paid_192", "terminal"},
            "Common evaluator omitted a requested checkpoint record")
    for point in checkpoints:
        label = point["checkpoint"]
        wanted = 0 if label == "start" else paid if label == "terminal" else int(label.removeprefix("paid_"))
        if wanted > paid:
            require(point["status"] != "captured" and not point["evaluation_paid"] and not point["J"],
                    f"Common evaluator fabricated metrics after the episode at {label}")
            continue
        require(point["status"] == "captured" and int(point["evaluation_paid"]) == wanted,
                f"Common evaluator captured a wrong clock at {label}")
        c, p, r, f1, j = [float(point[k]) for k in ("C_free", "p_scene_05cm", "r_scene_05cm", "F1_scene_05cm", "J")]
        require(all(0 <= value <= 1 for value in (c, p, r, f1, j))
                and math.isclose(c, float(curve[wanted]["C_free"]), abs_tol=1e-12)
                and math.isclose(f1, 2*p*r/(p+r) if p+r else 0., abs_tol=1e-12)
                and math.isclose(j, c*f1, abs_tol=1e-12),
                f"Common evaluator J, F1 or coverage mismatch at {label}")
    result["A95_paid"] = int(earliest) if earliest else None


def episode_check(runs, floor, protocol_path, protocol, results, replay):
    layout = floor["layout"]
    run, exit_file, sensor_file = (runs / layout, runs / f"{layout}.exit.json", runs / f"{layout}.sensor_audit.json")
    receipt_file = run / "tare_receipt.json"
    if not exit_file.is_file():
        return {"layout": layout, "status": "missing_exit", "partial_artifacts": run.exists()}
    exit_record = read(exit_file)
    require(exit_record["layout"] == layout
            and exit_record["end_unix"] >= exit_record["start_unix"] >= protocol["frozen_unix_s"],
            "Execution exit receipt timing or layout invalid")
    if exit_record["exit_code"] != 0:
        return {"layout": layout, "status": "technical_failure", "exit_code": exit_record["exit_code"],
                "acquired_packets_before_failure": len(read(sensor_file)["observations"]) if sensor_file.is_file() else None,
                "exit_receipt": str(exit_file)}
    require(receipt_file.is_file(), "A successful worker exit did not produce a benchmark receipt")
    receipt, manifest, sensor = read(receipt_file), read(run / "manifest.json"), read(sensor_file)
    require(receipt["arm"] == ARM and receipt["layout"] == manifest["layout"] == layout
            and receipt["native_manifest_sha256"] == digest(run / "manifest.json")
            and receipt["external_protocol_sha256"] == digest(protocol_path)
            and receipt["sensor_audit_sha256"] == digest(sensor_file)
            and receipt["selection_calls"] == sensor["selection_calls"]
            and receipt["selection_calls"], "Unbound native, sensor or experiment receipt")
    require(manifest["status"] == "completed_execution" and manifest["arm"] == "F_strict"
            and manifest["actual_closed_loop"] and not manifest["evaluator_inputs_read_by_actor"]
            and manifest["runner_freeze_sha256"] == digest(BASE / "FORMAL_FREEZE.json"),
            "Frozen scene or shared driver changed")
    upstream_binary, = [f for f in protocol["files_sha256"] if f.endswith("/tare_planner_node")]
    upstream_parameters, = [f for f in protocol["files_sha256"] if f.endswith("/indoor.yaml")]
    ready = sensor["ready"]
    require(ready["node_sha256"] == protocol["files_sha256"][upstream_binary]
            and ready["parameters_sha256"] == protocol["files_sha256"][upstream_parameters]
            and ready["parameters"]["kExtendWayPoint"] is False and not ready["boundary_inputs_published"]
            and not ready["terrain_inputs_published"], "Native run or acquired-input contract differs from freeze")
    for name in ("initial_raw_rgbd_scan_sha256.json", "steps.json", "start/checkpoint.json",
                 "terminal/checkpoint.json", "audit.json", "frontier_events.json"):
        require(manifest["files_sha256"][name] == digest(run / name), f"Native output changed after run: {name}")
    common = BASE / "execution/formal" / layout / "combined_S/attempt_001/run"
    start = read(run / "start/checkpoint.json")
    initial = read(run / "initial_raw_rgbd_scan_sha256.json")
    require(start == read(common / "start/checkpoint.json") and initial == read(common / "initial_raw_rgbd_scan_sha256.json")
            and start["pose"] == manifest["initial_pose"] == floor["anchor"]
            and start["packet_sha256"] == initial["packet_sha256"] == manifest["initial_packet_sha256"]
            == floor["initial_packet_sha256"], "Cold start pose, observations, or map does not match NSO")
    steps, terminal = read(run / "steps.json"), read(run / "terminal/checkpoint.json")
    fresh, expected, timed_out, value_bound, stamped = observations_check(
        sensor, {**start, **initial}, steps, floor)
    frontier_events = read(run / "frontier_events.json")
    actions = actions_check(run, manifest, steps, start, terminal, floor["actual_config"]["resolution_m"], frontier_events)
    result = {"layout": layout, "status": "verified_closed_loop", "packet_publishes": len(sensor["observations"]),
              "direct_sensor_array_hash_matches": value_bound, "native_keypose_due_actions": expected,
              "native_keypose_timeout_actions": timed_out, "fresh_waypoint_actions": list(fresh),
              "stamped_waypoints_checked": stamped, **actions,
              **calls_check(sensor["selection_calls"], steps, fresh, expected, sensor["observations"], frontier_events)}
    if results is not None:
        compare_metrics(results, layout, result)
    if replay is not None:
        result["independent_transport_digests"] = replay(layout, steps, sensor["observations"])
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", type=Path, default=ROOT / "execution/formal")
    parser.add_argument("--protocol", type=Path, default=ROOT / "TARE_PROTOCOL.json")
    parser.add_argument("--results", type=Path, help="Additionally verify output of shared four-floor evaluator")
    parser.add_argument("--replay-packets", action="store_true", help="Verify exact worker sensor NPZ by independent rerender")
    args = parser.parse_args()
    protocol = read(args.protocol)
    floors = read(BASE / "scenes/actor_freeze_holdout.json")["layouts"]
    require(protocol["layouts"] == [floor["layout"] for floor in floors]
            and len(floors) == 4 and all(floor["stop_budget"] == BUDGET for floor in floors),
            "Registered TARE floors or shared episode budget do not match the original four-floor series")
    replayer = make_transport_replayer() if args.replay_packets else None
    results = []
    for floor in floors:
        try:
            result = episode_check(args.runs, floor, args.protocol, protocol, args.results, replayer)
        except (KeyError, TypeError, ValueError, FileNotFoundError) as error:
            result = {"layout": floor["layout"], "status": "evidence_invalid", "error": str(error)}
        results.append(result)
    verified = sum(row["status"] == "verified_closed_loop" for row in results)
    print(fmt({"study": ARM, "protocol_sha256": digest(args.protocol), "verified_count": verified,
               "registered_runs": len(floors), "cases": results}))
    return int(verified != len(floors))


if __name__ == "__main__":
    sys.exit(main())
