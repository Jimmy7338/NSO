"""Post-hoc audit of the closed-loop gate's terminal swept edge in frozen V2 geometry."""

import json
from pathlib import Path
import sys

import numpy as np

from audit_gate import ARM, HERE, ROOT, SCENE, read, require, sha


V2 = ROOT.parent / "external_support_continuity_20261126"
DIRECTIONS = ((-1, 0), (0, 1), (1, 0), (0, -1))


def v2_relief(geom, start, end, radius):
    """Mirror the frozen V2 sample positions when the raw mesh has full projection."""
    import shapely
    from shapely.geometry import LineString, Point
    from repair_geometry import BANK_REACH_M, v1

    line = Point(start) if end is None else LineString([start, end])
    capsule = line.buffer(radius / np.cos(np.pi / 128), quad_segs=32)
    left, bottom, right, top = capsule.bounds
    xs = np.linspace(left, right, int(np.ceil((right-left) / v1.PROBE_STEP_M)) + 1)
    ys = np.linspace(bottom, top, int(np.ceil((top-bottom) / v1.PROBE_STEP_M)) + 1)
    xx, yy = np.meshgrid(xs, ys)
    inside = shapely.contains_xy(capsule, xx, yy)
    grid = np.column_stack((xx[inside], yy[inside]))
    tube = capsule.buffer(BANK_REACH_M + 1e-6)
    indices = geom.raw_tree.query(tube, predicate="intersects")
    measured = shapely.union_all(geom.raw_support[indices])
    require(capsule.difference(measured).area <= 1e-12, "Probe audit requires complete raw mesh projection")
    patches = capsule.difference(geom.floor).intersection(measured)
    witnesses = [
        shapely.get_coordinates(p.representative_point())[0]
        for p in shapely.get_parts(patches) if p.area > 1e-12
    ]
    sample = np.concatenate((grid, np.asarray(witnesses).reshape(-1, 2)))
    heights = geom._rays(sample)
    require(geom._valid_height(heights).all(), "V2 exact probes disagree with the in-band diagnostic")
    image = np.full(inside.shape, np.nan)
    image[inside] = heights[:len(grid)]
    return {
        "sample_count": len(heights),
        "height_min_m": float(heights.min()), "height_max_m": float(heights.max()),
        "height_relief_m": float(np.ptp(heights)), "v2_height_relief_limit_m": v1.MAX_HEIGHT_RELIEF_M,
        "largest_adjacent_height_step_m": max(
            float(np.nanmax(np.abs(np.diff(image, axis=axis)))) for axis in (0, 1)
        ),
        "v2_adjacent_height_step_limit_m": v1.MAX_LOCAL_HEIGHT_CHANGE_M,
        "accepted_by_exact_v2_continuity_check": bool(geom._continuity(
            capsule, capsule.difference(geom.floor), capsule.difference(measured), measured,
        )),
    }


def classify():
    sys.path.insert(0, str(V2))
    from audit_rerun_collisions import geometry, profile

    phase_path = ROOT / "execution/gate/phase_receipt.json"
    phase = read(phase_path)
    row, = phase["episodes"]
    run = Path(row["run_path"])
    receipt = HERE / "GATE_AUDIT.json"
    require(read(receipt)["gate_passed"], "Ground-truth diagnosis must follow a passed gate audit")
    require((row["layout"], row["arm"], row["exit_code"]) == (SCENE, ARM, 0), "Unexpected gate episode")
    require(sha(run / "manifest.json") == row["result_sha256"], "Terminal manifest no longer matches the receipt")
    details = read(run / "manifest.json")
    actual = read(run / "steps.json")
    previous, failed = actual[-2:]
    require(sha(run / "steps.json") == details["files_sha256"]["steps.json"], "Gate trajectory changed")
    require(failed["step"] == 91 and failed["collision"] and not previous["collision"]
            and failed["action"] == "forward" and previous["position"] == failed["position"] == [58, 41]
            and previous["heading"] == failed["heading"] == 1,
            "Archived gate terminal is not the requested later transition")
    events = read(run / "audit.json")
    require(sha(run / "audit.json") == details["files_sha256"]["audit.json"], "Gate audit timeline changed")
    authorization, = [item for item in events if item["event"] == "paid_action_authorized"
                      and item["next_action_id"] == failed["step"]]
    edge_certified = any(item["event"] == "observed_swept_edge_certified"
                         and item["paid_before"] == failed["step"] - 1 for item in events)
    origin, heading = tuple(failed["position"]), failed["heading"]
    dr, dc = DIRECTIONS[heading]
    destination = (origin[0] + dr, origin[1] + dc)
    require(authorization["assessment"]["next_pose"] == [*destination, heading],
            "Actual motion approval disagrees with the simulator's failed forward command")
    freeze = read(ROOT / "GATE_FREEZE.json")
    frozen_scene = Path(freeze["scene_root"])
    cold = read(frozen_scene / "prepared" / SCENE / "initial_state.json")
    cfg = cold["actual_config"]
    metres = cfg["resolution_m"]

    def xy(cell):
        return ((cell[1] + .5) * metres, (cold["shape"][0] - cell[0] - .5) * metres)

    floor = cold["source_floor"]
    cache = frozen_scene / "precomputed_assets" / f"{floor}.npz"
    require(sha(cache) == freeze["simulator_only_files"][str(cache)], "Terminal V2 cache differs")
    with np.load(cache, allow_pickle=False) as nav:
        admitted = bool(nav["edge_valid"][origin + (heading,)])
        start_admitted, end_admitted = (bool(nav["robot_valid"][cell]) for cell in (origin, destination))
    require(not admitted, "The frozen V2 cache unexpectedly allows the failed edge")
    geom = geometry(floor)
    radius = cfg["robot_radius_m"]
    measurements = profile(geom, xy(origin), xy(destination), radius)
    if measurements["body_band_intersection"]:
        disposition = "body_band_intersection"
    elif measurements["outside_and_unmeasured_m2"] > 1e-12:
        disposition = "unmeasured_exterior_floor_support"
    elif measurements["missing_patch_no_return"] or measurements["missing_patch_below_band"] or measurements["missing_patch_above_band"]:
        disposition = "unverified_or_out_of_band_interior_floor_support"
    elif not (measurements["seam_evidence_passed"] or measurements["direct_projection_residual_measured"]):
        disposition = "same_band_ray_returns_but_insufficient_continuous_projection"
    else:
        disposition = "in_band_support_geometry_rejection"
    continuity = {
        "full_sweep": v2_relief(geom, xy(origin), xy(destination), radius),
        "endpoint": v2_relief(geom, xy(destination), None, radius),
    }
    geom_script = V2 / "audit_rerun_collisions.py"
    return {
        "scope": "Post-hoc frozen scan-mesh/capsule diagnostic. No policy run, operational map, cache or gate receipt changed.",
        "layout": SCENE, "policy": ARM, "paid": failed["step"], "source": [*origin, heading],
        "target": [*destination, heading], "known_online_guard_approved": authorization["assessment"]["allowed"],
        "online_fine_sweep_certificate_for_terminal_edge": edge_certified,
        "v2_cache_admitted": admitted, "v2_start_centre_admitted": start_admitted, "v2_target_centre_admitted": end_admitted,
        "frozen_robot_radius_m": cfg["robot_radius_m"], "disposition": disposition, "raw_geometry": measurements,
        "frozen_continuity_sample_profile": continuity,
        "hashes": {"native_trajectory_sha256": sha(run / "steps.json"), "gate_receipt_sha256": sha(receipt),
                   "v2_cache_sha256": sha(cache), "profile_sha256": sha(geom_script)},
    }


if __name__ == "__main__":
    result = classify()
    (HERE / "GATE_TERMINATION_AUDIT.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(result["disposition"], result["raw_geometry"])
