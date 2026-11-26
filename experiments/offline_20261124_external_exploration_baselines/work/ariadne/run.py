"""Evaluate the published ARiADNE graph/weights with the frozen NSO simulator."""

import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import selectors
import subprocess
import sys
import time

import numpy as np


ROOT = Path(__file__).resolve().parent
BASE = ROOT.parents[2] / "region_independent_floors_20261123"
SOURCE = ROOT.parents[2] / "region_structure_floor_20261122"
ARM = "ARiADNE_policy"


def digest(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


class ActorProcess:
    def __init__(self, diagnostic_log):
        env = dict(os.environ, CUDA_VISIBLE_DEVICES="", MPLBACKEND="Agg", PYTHONUNBUFFERED="1",
                   PYTHONDONTWRITEBYTECODE="1", OPENBLAS_NUM_THREADS="1", OMP_NUM_THREADS="1")
        self.child = subprocess.Popen(["/usr/bin/python3", str(ROOT / "worker.py")], env=env,
                                      stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                      stderr=diagnostic_log, text=True, encoding="utf-8")
        self.reader = selectors.DefaultSelector()
        self.reader.register(self.child.stdout, selectors.EVENT_READ)

    def __enter__(self):
        return self

    def select(self, grid, cell):
        request = {"belief": grid.tolist(), "cell": list(map(int, cell))}
        self.child.stdin.write(json.dumps(request, allow_nan=False) + "\n")
        self.child.stdin.flush()
        if not self.reader.select(timeout=20):
            raise TimeoutError("The CPU ARiADNE actor did not return a reply within 20 seconds.")
        reply = self.child.stdout.readline()
        if not reply:
            raise RuntimeError("The CPU ARiADNE actor exited. See its diagnostic log.")
        return json.loads(reply)

    def __exit__(self, *_):
        self.child.stdin.close()
        try:
            self.child.wait(timeout=5)
        except subprocess.TimeoutExpired:
            self.child.kill()
            self.child.wait(timeout=5)
        finally:
            self.reader.close()
            self.child.stdout.close()


def build_policy(policy_type, config, actor, audit):
    from nso.frontier_policy import shortest_paths, trace_path
    from utils.grid_geometry import DIRECTIONS

    class AriadnePolicy(policy_type):
        def _select(self, obs, traversable):
            # Give the actor only current observations and the same footprint mask.
            grid = np.where((obs.belief == 0) & ~traversable, 1, obs.belief)
            grid[obs.position] = 0
            started = time.monotonic()
            record = actor.select(grid, obs.position)
            record.update(selection_latency_s=time.monotonic() - started, paid=obs.step,
                          current_pose=[*obs.position, obs.heading], chosen=None)
            audit.append(record)
            if "target_cell" not in record:
                return False
            goal = tuple(record["target_cell"])
            distances, parents = shortest_paths(traversable, obs.position)
            if any(not 0 <= pos < size for pos, size in zip(goal, grid.shape, strict=True)):
                record["reason"] = "upstream_waypoint_outside_shared_grid"
                return False
            if goal == obs.position or distances[goal] < 0:
                record["reason"] = "upstream_waypoint_not_executable"
                return False
            route = trace_path(parents, obs.position, goal)
            previous = route[-2] if len(route) >= 2 else obs.position
            heading = DIRECTIONS.index((goal[0] - previous[0], goal[1] - previous[1]))
            self.goal, self.goal_heading = goal, heading
            self._gain, self._distance, self._score = 0, int(distances[goal]), 0.
            self._path = route
            self.goal_count += 1
            record["chosen"] = {"executable_pose": [*goal, heading],
                                "outbound_translation_cells": len(route)}
            return True

        def act(self, obs):
            decision = super().act(obs)
            if decision.goal is None:
                # ARiADNE exits once it cannot generate a waypoint. Return is
                # owned by the shared execution driver.
                return type(decision)("stop", None, None, 0, 0, 0., False)
            return decision

    return AriadnePolicy(config)


def launch(layout, output):
    protocol = json.loads((ROOT / "PROTOCOL.json").read_text())
    assert layout in protocol["layouts"]
    for path, expected in protocol["files_sha256"].items():
        assert digest((ROOT / path).resolve()) == expected, path
    assert not output.exists()
    sys.path.insert(0, str(BASE))
    sys.path.insert(0, str(SOURCE))
    import formal_runner
    assert Path(formal_runner.runner.__file__).resolve() == SOURCE / "runner.py"
    original = formal_runner.attach_frontier_receipts
    choices = []

    log_path = output.with_name(output.name + ".worker.log")
    with log_path.open("x", encoding="utf-8") as worker_log, ActorProcess(worker_log) as actor:
        def attach(driver, runtime):
            driver.policy = build_policy(type(driver.policy), driver.policy.config, actor, choices)
            return original(driver, runtime)

        formal_runner.attach_frontier_receipts = attach
        spec = importlib.util.spec_from_file_location("frozen_floor_entry", BASE / "runner.py")
        runner = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(runner)
        original_argv = sys.argv
        try:
            sys.argv = [__file__, layout, "F_strict", "--freeze", str(BASE / "FORMAL_FREEZE.json"),
                        "--out", str(output)]
            runner.main()
        finally:
            formal_runner.attach_frontier_receipts = original
            sys.argv = original_argv
            with output.with_name(output.name + ".selections.json").open("x") as stream:
                json.dump(choices, stream, indent=2, allow_nan=False)
                stream.write("\n")
    with (output / "ariadne_receipt.json").open("x") as stream:
        json.dump({"arm": ARM, "layout": layout, "native_manifest_sha256": digest(output / "manifest.json"),
                   "external_protocol_sha256": digest(ROOT / "PROTOCOL.json"),
                   "selection_calls": choices}, stream, indent=2, allow_nan=False)
        stream.write("\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("layout")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    launch(args.layout, args.out.resolve())
