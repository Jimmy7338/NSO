"""Run the published TARE planner using the frozen shared floor simulator."""

import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import selectors
import subprocess
import sys
import tempfile

import numpy as np


ROOT = Path(__file__).resolve().parent
BASE = ROOT.parent / "region_independent_floors_20261123"
NATIVE_RUNNER = ROOT.parent / "region_structure_floor_20261122"
TARE = ROOT.parent / "external_exploration_baselines_20261124/work/tare"
NODE = TARE / "catkin/devel/lib/tare_planner/tare_planner_node"
INDOOR = TARE / "catkin/src/tare_planner/config/indoor.yaml"
ARM = "TARE_sensor_transfer"
CAMERA_FIELDS = ("depth_m", "intrinsic", "world_from_camera")
SCAN_FIELDS = ("ranges_m", "angle_min_rad", "angle_increment_rad",
               "range_max_m", "world_from_laser")


def digest(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


class TareProcess:
    def __init__(self, scratch, output, log, actor_executor):
        self.output = output
        self.packet_file = scratch / "packet.npz"
        self.recordings = []
        self.response = None
        self.child = subprocess.Popen(
            ["/usr/bin/python3", "-B", str(ROOT / "tare_worker.py"), "--binary", str(NODE),
             "--parameters", str(INDOOR), "--native-log", str(output.with_suffix(".native.log")),
             "--overrides-json", str(ROOT / "tare_indoor_overrides.json")],
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=log, text=True, encoding="utf-8",
            env=actor_executor)
        self.reader = selectors.DefaultSelector()
        self.reader.register(self.child.stdout, selectors.EVENT_READ)

    def __enter__(self):
        return self

    def read(self, timeout):
        if not self.reader.select(timeout=timeout):
            raise TimeoutError("TARE bridge timed out; see worker/native logs")
        line = self.child.stdout.readline()
        if not line:
            raise RuntimeError("TARE bridge exited before replying; see worker/native logs")
        return json.loads(line)

    def send(self, request):
        self.child.stdin.write(json.dumps(request, allow_nan=False) + "\n")
        self.child.stdin.flush()
        return self.read(12)

    def observe(self, packet):
        if packet.action_id != len(self.recordings):
            raise ValueError("TARE must receive exactly one packet per consecutive action ID")
        arrays = {f"frame__{key}": getattr(packet.frame, key) for key in CAMERA_FIELDS}
        arrays.update({f"scan__{key}": getattr(packet.scan, key) for key in SCAN_FIELDS})
        np.savez(self.packet_file, **arrays)
        response = self.send({"op": "observe", "packet_path": str(self.packet_file),
                              "action_id": packet.action_id})
        if response["status"] != "observed" or response["action_id"] != packet.action_id:
            raise RuntimeError(f"TARE rejected observation {packet.action_id}: {response}")
        if response["packet_sha256"] != digest(self.packet_file):
            raise RuntimeError(f"TARE read a different wire packet at action {packet.action_id}")
        self.response = response
        self.recordings.append(dict(response, packet_sha256=packet.sha256(),
                                    bridge_packet_sha256=response["packet_sha256"],
                                    position=list(packet.position), heading=packet.heading))

    def __exit__(self, *_):
        try:
            self.child.stdin.close()
            self.child.wait(timeout=10)
        except (BrokenPipeError, subprocess.TimeoutExpired):
            self.child.terminate()
            try:
                self.child.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self.child.kill()
                self.child.wait()
        finally:
            self.reader.close()
            self.child.stdout.close()


def waypoint_cell(xy, anchor, traversable, distances, transform):
    if not np.isfinite(xy).all():
        raise ValueError("TARE returned a nonfinite waypoint")
    end = transform.world_to_cell(xy)
    bounds = traversable.shape
    if all(0 <= index < size for index, size in zip(end, bounds, strict=True)) and distances[end] >= 0:
        return end, "exact_waypoint"
    start = np.asarray(transform.cell_to_world(anchor))
    delta = np.asarray(xy) - start
    samples = max(1, int(np.ceil(np.linalg.norm(delta) / (transform.resolution_m / 4))))
    previous = anchor
    for fraction in np.linspace(0, 1, samples + 1)[1:]:
        cell = transform.world_to_cell(start + fraction * delta)
        if (not all(0 <= index < size for index, size in zip(cell, bounds, strict=True))
                or not traversable[cell] or distances[cell] < 0):
            break
        previous = cell
    return previous, "observed_prefix"


def make_policy(policy_type, config, actor, selections):
    from nso.cpu_sensor_contract_v10 import GridTransform
    from nso.frontier_policy import Decision, shortest_paths, trace_path
    from utils.grid_geometry import DIRECTIONS

    class TarePolicy(policy_type):
        waypoint = None
        missing_since = None
        last_keypose = -1

        def _select(self, obs, traversable):
            row = {"paid": obs.step, "waypoint": self.waypoint, "chosen": None}
            selections.append(row)
            if self.waypoint is None:
                row["reason"] = "no_native_waypoint"
                return False
            if self.waypoint["frame_id"].lstrip("/") != "map":
                raise ValueError(f"Unexpected native TARE waypoint frame: {self.waypoint}")
            transform = GridTransform(tuple(traversable.shape), self.config.resolution_m)
            distances, parents = shortest_paths(traversable, obs.position)
            xy = self.waypoint["xyz"][:2]
            goal, mode = waypoint_cell(xy, obs.position, traversable, distances, transform)
            if goal == obs.position:
                row["reason"] = "waypoint_yields_no_observed_translation"
                return False
            route = trace_path(parents, obs.position, goal)
            previous = route[-2] if len(route) >= 2 else obs.position
            self.goal, self.goal_heading = goal, DIRECTIONS.index((goal[0]-previous[0], goal[1]-previous[1]))
            self._gain, self._distance, self._score = 0, int(distances[goal]), 0.0
            self._path = route
            self.goal_count += 1
            row["chosen"] = {"executable_pose": [*goal, self.goal_heading], "routing": mode,
                             "outbound_translation_cells": len(route)}
            return True

        def act(self, obs):
            report = actor.response
            if report is None or report["action_id"] != obs.step:
                raise RuntimeError("TARE must act on the currently acknowledged packet")
            if report["exploration_finished"]:
                selections.append({"paid": obs.step, "waypoint": None, "chosen": None,
                                   "reason": "native_exploration_finished"})
                return Decision("stop", None, None, 0, 0, 0.0, False)
            if report["expected_native_planning_due"] and obs.step != self.last_keypose:
                self.last_keypose = obs.step
                self.waypoint = report["waypoint"] if report["fresh_waypoint"] else None
                self.goal, self._path = None, []
            decision = super().act(obs)
            if decision.goal is not None:
                self.missing_since = None
                return decision
            if self.last_keypose < 0:
                action = "right"     # One measured, budgeted camera view per turn until keypose 1.
            else:
                if self.missing_since is None:
                    self.missing_since = obs.step
                action = "right" if obs.step-self.missing_since < 5 else "stop"
            return type(decision)(action, None, None, 0, 0, 0.0, False)

    return TarePolicy(config)


def launch(layout, output):
    seal = json.loads((ROOT / "TARE_PROTOCOL.json").read_text())
    if layout not in seal["layouts"] or output.exists():
        raise ValueError("Specify a registered floor and unused output directory")
    for reference, sha in seal["files_sha256"].items():
        if digest((ROOT / reference).resolve()) != sha:
            raise ValueError(f"Frozen TARE input changed: {reference}")
    output.parent.mkdir(parents=True, exist_ok=True)
    sys.path.insert(0, str(BASE))
    sys.path.insert(0, str(NATIVE_RUNNER))
    import formal_runner
    assert Path(formal_runner.runner.__file__).resolve() == NATIVE_RUNNER / "runner.py"
    original = formal_runner.attach_frontier_receipts
    selections = []
    actor_env = dict(os.environ)
    # The ROS C++ process resolves the pinned native ARM64 OR-Tools library.
    ortools = TARE / "or-tools_aarch64_AlmaLinux-8.8_cpp_v9.8.3296/lib64"
    actor_env["LD_LIBRARY_PATH"] = f"{ortools}:/opt/ros/noetic/lib:" + actor_env.get("LD_LIBRARY_PATH", "")
    log_path = output.with_suffix(".bridge.log")
    with tempfile.TemporaryDirectory(prefix="nso-tare-packets-") as scratch, log_path.open("x") as log:
        with TareProcess(Path(scratch), output, log, actor_env) as actor:
            ready = actor.read(30)
            if ready["status"] != "ready":
                raise RuntimeError(f"TARE worker failed to start: {ready}")

            def attach(driver, runtime):
                actor.observe(runtime.states[0]["packet"])
                driver.policy = make_policy(type(driver.policy), driver.policy.config, actor, selections)
                driver = original(driver, runtime)
                accept = runtime.observe

                def bridge_observation(scene, *args, **kwargs):
                    value = accept(scene, *args, **kwargs)
                    actor.observe(runtime.states[scene]["packet"])
                    return value

                runtime.observe = bridge_observation
                return driver

            formal_runner.attach_frontier_receipts = attach
            spec = importlib.util.spec_from_file_location("frozen_tare_floor_entry", BASE / "runner.py")
            runner = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(runner)
            old_argv = sys.argv
            try:
                sys.argv = [__file__, layout, "F_strict", "--freeze", str(BASE / "FORMAL_FREEZE.json"),
                            "--out", str(output)]
                runner.main()
            finally:
                formal_runner.attach_frontier_receipts = original
                sys.argv = old_argv
                with output.with_suffix(".sensor_audit.json").open("x") as stream:
                    json.dump(dict(ready=ready, observations=actor.recordings,
                                   selection_calls=selections), stream, indent=2, allow_nan=False)
                    stream.write("\n")
    manifest = json.loads((output / "manifest.json").read_text())
    if len(actor.recordings) != manifest["paid_actions"] + 1:
        raise ValueError("Recorded frames and native paid clocks differ")
    with (output / "tare_receipt.json").open("x") as stream:
        json.dump(dict(arm=ARM, layout=layout, native_manifest_sha256=digest(output / "manifest.json"),
                       external_protocol_sha256=digest(ROOT / "TARE_PROTOCOL.json"),
                       sensor_audit_sha256=digest(output.with_suffix(".sensor_audit.json")),
                       selection_calls=selections), stream, indent=2, allow_nan=False)
        stream.write("\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("layout")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    launch(args.layout, args.out.resolve())
