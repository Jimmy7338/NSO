"""Compare a frozen directional endpoint candidate on the two fixed west streams."""

import argparse
from dataclasses import asdict
import json
from pathlib import Path
import sys

import numpy as np

from inputs import FREEZE, OLD, ORIGINAL, RGB_ONLY, read, sha, verify
from projection import bind


def edge_admissions(safe, directions):
    result = np.zeros((*safe.shape, 4), dtype=bool)
    rows, cols = safe.shape
    for heading, (dr, dc) in enumerate(directions):
        r0, r1 = max(0, -dr), min(rows, rows-dr)
        c0, c1 = max(0, -dc), min(cols, cols-dc)
        result[r0:r1, c0:c1, heading] = safe[r0:r1, c0:c1] & safe[r0+dr:r1+dr, c0+dc:c1+dc]
    return result


def return_edges(position, heading, actions, anchor, edges, valid_centers, directions):
    position = tuple(position)
    swept_valid = bool(valid_centers[position])
    for action in actions:
        if action == "forward":
            swept_valid &= bool(edges[position + (heading,)])
            position = tuple(u + v for u, v in zip(position, directions[heading], strict=True))
        else:
            heading = (heading + {"left": -1, "right": 1}[action]) % 4
            swept_valid &= bool(valid_centers[position])
    if [*position, heading] != anchor:
        raise ValueError("Verified return plan does not end at the oriented anchor")
    return swept_valid


def evaluate(mode):
    sys.path.insert(0, str(OLD))
    from precision_patch import camera_cell as prior_camera_cell
    from west_replay import (LAYOUT, PROTOCOL, SCENES, SENSOR_FIELDS, native, planned_path_geometry,
                             protocol, raw_sensor_hashes)

    run = ORIGINAL if mode == "archived" else RGB_ONLY
    manifest, steps = (read(run / file) for file in ("manifest.json", "steps.json"))
    if len(steps) != manifest["paid_actions"] or len(steps) != {"archived": 61, "rgb_only": 51}[mode]:
        raise ValueError("The source stream no longer matches the frozen motion clock")
    with np.load(run / "observed_belief_trace.npz", allow_pickle=False) as raw:
        expected = raw["belief"]
        if not np.array_equal(raw["paid"], np.arange(len(steps) + 1)):
            raise ValueError("A native grid snapshot is missing")

    native.HOLDOUT = SCENES
    native.ARM_MODES = {"G": ("G", "G")}
    config = read(PROTOCOL)
    old, (world, transform, first, components, _, _, _), initial, _, _ = native.setup_runtime(
        "holdout", LAYOUT, "G", protocol(config["placements"], config))
    mapper_type = type(components._cpu_backend.scenes[0]["mapper"])
    from nso.execution_guard_v8 import ObservedExecutionGuard
    from utils.grid_geometry import DIRECTIONS

    candidate_binding = bind(prior_camera_cell)
    original, shadow = [mapper_type(world.shape, world.config, world.config.truncation_m) for _ in range(2)]
    original._endpoint_shadow_mode = "raw" if mode == "archived" else "original_rgb_only"
    original._simulator_camera_boundary_shadow = mode == "rgb_only"
    shadow._endpoint_shadow_mode = "directional"
    resolution, radius = world.config.resolution_m, world.config.robot_radius_m
    guard = ObservedExecutionGuard(resolution, radius)
    anchor = initial["anchor"]
    # These fixture references are used only in comparisons below. Neither
    # mapper nor the real guard receives arrays from the virtual world.
    legal_centers, legal_edges, occupied_truth = world._robot_valid, world._edge_valid, world._occupancy
    history = []

    def capture(packet):
        paid = packet.action_id
        for mapper in (original, shadow):
            mapper.update(packet.frame, packet.scan)
        a, b = original.belief, shadow.belief
        if not np.array_equal(a, expected[paid]):
            raise ValueError(f"Could not reproduce original measured grid at paid {paid}")
        a_safe, b_safe = [guard.safe_grid(grid) for grid in (a, b)]
        a_edges, b_edges = [edge_admissions(mask, DIRECTIONS) for mask in (a_safe, b_safe)]
        old_plan, new_plan = [guard.return_plan(grid, packet.position, packet.heading, anchor, 192-paid)
                              for grid in (a, b)]
        invalid_added_centers = ~a_safe & b_safe & ~legal_centers
        invalid_added_edges = ~a_edges & b_edges & ~legal_edges
        lost_real_support = (a == 1) & (b != 1) & occupied_truth
        row = {"paid": paid, "packet_sha256": packet.sha256(), "pose": [*packet.position, packet.heading],
               "changed_belief_cells": int(np.count_nonzero(a != b)),
               "lost_true_occupied_centers": np.argwhere(lost_real_support).tolist(),
               "new_invalid_centers": np.argwhere(invalid_added_centers).tolist(),
               "new_invalid_directed_sweeps": np.argwhere(invalid_added_edges).tolist(),
               "original_return": asdict(old_plan), "candidate_return": asdict(new_plan),
               "candidate_return_all_sweeps_clear": return_edges(packet.position, packet.heading, new_plan.actions,
                   anchor, legal_edges, legal_centers, DIRECTIONS) if new_plan.available else None}
        if paid == len(steps) and new_plan.available:
            continuous = planned_path_geometry(world, packet.position, packet.heading, new_plan.actions, anchor)
            row["candidate_terminal_continuous_clearance_m"] = min(
                (step["oracle_centerline_to_real_obstacles_m"] for step in continuous), default=None)
            row["candidate_terminal_continuous_sweep_valid"] = all(step["oracle_swept_disk_valid"] for step in continuous)
        history.append(row)

    if raw_sensor_hashes(first, native.array_sha) != read(run / "initial_raw_rgbd_scan_sha256.json"):
        raise ValueError("Cold observation differs from the actually executed source")
    capture(first)
    for recorded in steps:
        frame, hit, done = world.step(recorded["action"])
        packet = old.packet(world, transform, frame, LAYOUT, recorded["action"], hit, done)
        raw = raw_sensor_hashes(packet, native.array_sha)
        if hit or any(raw[key] != recorded[key] for key in ("packet_sha256", *SENSOR_FIELDS)):
            raise ValueError(f"Fixed path or source observations differ at paid {recorded['step']}")
        capture(packet)

    terminal = history[-1]
    pass_checks = {
       "sources_exact": True,
       "same_original_disconnect": terminal["original_return"]["reason"] == "anchor_disconnected_in_latest_map",
       "no_lost_true_occupied_centers": not any(r["lost_true_occupied_centers"] for r in history),
       "no_new_invalid_centers": not any(r["new_invalid_centers"] for r in history),
       "no_new_invalid_directed_sweeps": not any(r["new_invalid_directed_sweeps"] for r in history),
       "no_invalid_planned_return_sweep_at_any_paid_clock": all(
           r["candidate_return_all_sweeps_clear"] is not False for r in history),
       "continuous_return_from_final_disconnect": bool(terminal.get("candidate_terminal_continuous_sweep_valid")),
    }
    return {"mode": mode, "original_manifest_sha256": sha(run / "manifest.json"),
            "frozen_candidate_sha256": sha(FREEZE), "candidate_binding": candidate_binding,
            "pass_checks": pass_checks, "shadow_accepted": all(pass_checks.values()), "fixed_frames": len(history),
            "history": history, "new_trajectory_executed": False}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stream", choices=("archived", "rgb_only"))
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()
    verify()
    if args.out.exists():
        raise FileExistsError(f"Use a new receipt path: {args.out}")
    result = evaluate(args.stream)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("x") as stream:
        stream.write(json.dumps(result, indent=2) + "\n")
    print(args.stream, result["pass_checks"], flush=True)
    if not result["shadow_accepted"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
