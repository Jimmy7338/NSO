"""Compare forecast floor sampling with actual per-pixel depth at the chosen views."""

import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from evaluate import ROOT, array_digest, load_module, read, require, write_json

import numpy as np


DIAGNOSTIC_DEPTH_TOLERANCE_M = .05
PUBLISHER_FROZEN_LASER_HEIGHT_M = .25


def diagnose(phase, layout_filter):
    registered = read(ROOT / f"{phase.upper()}_FREEZE.json")
    scene_root = Path(registered["scene_root"])
    sys.path.insert(0, str(scene_root))
    scene = load_module("ground_yield_frozen_scene", scene_root / "scene.py")
    records = []
    for job in registered["jobs"]:
        layout, arm = job["layout"], job["arm"]
        if arm != "B" or layout_filter and layout != layout_filter:
            continue
        attempts = sorted((ROOT / "execution" / phase / layout / arm).glob("attempt_*/run"))
        if not attempts or not (attempts[-1] / "audit.json").exists():
            continue
        run = attempts[-1]
        events = read(run / "audit.json")
        dispatches = {row["inspection_id"]: row for row in events if row["event"] == "ground_reacquisition_dispatch"}
        arrivals = [row for row in events if row["event"] == "ground_reacquisition_view"]
        if not arrivals:
            continue
        steps = read(run / "steps.json")
        world = scene.make_world(layout)
        initial_scan = world.scan()
        ground = float(initial_scan.world_from_laser[2, 3]) - PUBLISHER_FROZEN_LASER_HEIGHT_M
        for arrival in arrivals:
            sent = dispatches[arrival["inspection_id"]]
            paid = arrival["paid"]
            raw = steps[paid-1]
            world.position, world.heading, world.step_count = tuple(raw["position"]), raw["heading"], paid
            frame, laser = world.sense(), world.scan()
            require(array_digest(frame.depth_m) == raw["depth_values_sha256"] and
                    array_digest(laser.ranges_m) == raw["scan_ranges_sha256"],
                    f"Unmatched replay observation at {layout}:{paid}")
            xy = np.asarray(sent["missing_xy_m"])
            xyz = np.column_stack((xy, np.full(len(xy), ground)))
            optical = (xyz - frame.world_from_camera[:3, 3]) @ frame.world_from_camera[:3, :3]
            depth = optical[:, 2]
            projection = (optical @ frame.intrinsic.T)[:, :2] / np.where(depth[:, None] > 0, depth[:, None], np.inf)
            u, v = np.rint(projection).astype(int).T
            h, w = frame.depth_m.shape
            inside = (depth > .15) & (depth < laser.range_max_m - .001) & (u >= 0) & (v >= 0) & (u < w) & (v < h)
            actual = frame.depth_m[v[inside], u[inside]]
            finite = np.isfinite(actual) & (actual > .15) & (actual < laser.range_max_m - .001)
            diff = actual[finite] - depth[inside][finite]
            limit = DIAGNOSTIC_DEPTH_TOLERANCE_M
            pixel = np.column_stack((u[inside][finite], v[inside][finite], np.ones(np.count_nonzero(finite))))
            camera_ray = pixel @ np.linalg.inv(frame.intrinsic).T
            world_ray = camera_ray @ frame.world_from_camera[:3, :3].T
            heights = frame.world_from_camera[2, 3] + world_ray[:, 2] * actual[finite] - ground
            closer_heights = heights[diff < -limit]
            counts = {
                "outside_valid_projection": int((~inside).sum()),
                "in_frame_no_finite_depth": int((~finite).sum()),
                "finite_hit_closer_than_nominal_floor": int((diff < -limit).sum()),
                "finite_hit_near_nominal_floor_depth": int((np.abs(diff) <= limit).sum()),
                "finite_hit_beyond_nominal_floor": int((diff > limit).sum()),
            }
            require(sum(counts.values()) == len(xy), "Projected outcome bins overlap")
            records.append(dict(layout=layout, inspection_id=sent["inspection_id"], dispatch_paid=sent["paid"],
                                arrival_paid=paid, edge=sent["edge"], view=arrival["pose"],
                                depth_sha256=raw["depth_values_sha256"], scan_sha256=raw["scan_ranges_sha256"],
                                predicted_supported_fraction=sent["candidates"][0]["predicted_supported_fraction"],
                                unsupported_before=arrival["unsupported_before"],
                                unsupported_after=arrival["unsupported_after"], projection=counts,
                                closer_hits_above_nominal_ground_m=(
                                    {"median":float(np.median(closer_heights)), "min":float(closer_heights.min()),
                                     "max":float(closer_heights.max()), "at_least_15_cm":int(np.sum(closer_heights >= .15))}
                                    if len(closer_heights) else None)))
    return {"schema":"nso_actual_view_depth_diagnostic_v1","phase":phase,
            "projection_depth_tolerance_m":DIAGNOSTIC_DEPTH_TOLERANCE_M,
            "meaning":"Nearest-pixel comparison to the observer's nominal ground plane at actual recorded views; raw values are hash matched. Deeper/nearer rays explain sampling disagreement without classifying simulator ground truth.",
            "arrived_views": records}


if __name__ == "__main__":
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--phase", choices=("development", "formal"), default="formal")
    cli.add_argument("--layout")
    cli.add_argument("--out", type=Path, required=True)
    command = cli.parse_args()
    write_json(command.out, diagnose(command.phase, command.layout))
