"""Audit any new terminal collision against the frozen scan and both motion caches."""

from collections import Counter, defaultdict
from functools import lru_cache
from pathlib import Path

import numpy as np
import open3d as o3d
import shapely
from shapely.geometry import LineString, box

from prepare_study import DIRECTIONS, SPECS, make_world, source_scene
from repair_geometry import BANK_REACH_M, FLOOR_BAND_M, RepairedMotionGeometry
from study import OFFLINE, ROOT, SCENES, supervisor
from validate_repair import array_sha


@lru_cache(maxsize=1)
def geometry(floor):
    pose = next(spec for spec in SPECS.values() if spec["source_floor"] == floor)
    return RepairedMotionGeometry(source_scene(floor), np.asarray(pose["world_from_source"]))


def profile(geom, start, end, radius):
    line = LineString([start, end])
    x0, y0, x1, y1 = line.bounds
    indices = geom.tree.query(box(x0-radius-1., y0-radius-1., x1+radius+1., y1+radius+1.))
    closest = float(shapely.distance(geom.obstacles[indices], line).min()) if len(indices) else None
    capsule = line.buffer(radius / np.cos(np.pi / 128), quad_segs=32)
    missing = capsule.difference(geom.floor)
    probe = capsule.buffer(BANK_REACH_M + 1e-6)
    raw = shapely.union_all(geom.raw_support[geom.raw_tree.query(probe, predicate="intersects")])
    gap = capsule.difference(raw)
    exterior_gap = gap.difference(geom.outer_ground)
    direct_residual = geom._directly_measured_residual(gap)
    seam_passed = geom._seam_receipt(gap, raw)
    left, bottom, right, top = missing.bounds if not missing.is_empty else capsule.bounds
    xs = np.linspace(left, right, max(2, int(np.ceil((right-left)/.0025))+1))
    ys = np.linspace(bottom, top, max(2, int(np.ceil((top-bottom)/.0025))+1))
    xx, yy = np.meshgrid(xs, ys)
    include = shapely.contains_xy(missing, xx, yy)
    coords = np.column_stack((xx[include], yy[include]))
    witnesses = np.array([(p.representative_point().x, p.representative_point().y)
                          for p in shapely.get_parts(missing) if p.area > 1e-12]).reshape(-1, 2)
    coords = np.concatenate([coords, witnesses], axis=0)
    z0 = .149
    rays = np.zeros((len(coords), 6), dtype=np.float32)
    heights = np.empty(0)
    if len(coords):
        rays[:, :2], rays[:, 2], rays[:, 5] = coords, z0, -1.
        heights = z0 - geom.raw_rays.cast_rays(o3d.core.Tensor(rays), nthreads=1)["t_hit"].numpy()
    finite = heights[np.isfinite(heights)]
    return {"min_body_distance_m": closest, "body_band_intersection": not geom._body_clear(line, radius),
            "repaired_sweep_supported": geom.valid(start, end, radius),
            "repaired_start_supported": geom.valid(start, radius=radius),
            "repaired_end_supported": geom.valid(end, radius=radius),
            "outside_repaired_floor_projection_m2": capsule.difference(geom.outer_ground).area,
            "unsupported_in_repaired_floor_m2": missing.area,
            "not_covered_by_source_in_band_projection_m2": gap.area,
            "outside_and_unmeasured_m2": exterior_gap.area, "seam_evidence_passed": seam_passed,
            "direct_projection_residual_measured": direct_residual,
            "missing_patch_ray_count": len(heights), "missing_patch_no_return": len(heights)-len(finite),
            "missing_patch_below_band": int(np.count_nonzero(finite < FLOOR_BAND_M[0] - 1e-6)),
            "missing_patch_above_band": int(np.count_nonzero(finite > FLOOR_BAND_M[1] + 1e-6)),
            "missing_patch_min_height_m": float(finite.min()) if len(finite) else None,
            "missing_patch_max_height_m": float(finite.max()) if len(finite) else None}


def main():
    io = supervisor()
    phase_path = ROOT / "execution/formal/phase_receipt.json"
    phase = io.read(phase_path)
    assert len(phase["episodes"]) == 12 and all(r["exit_code"] == 0 for r in phase["episodes"])
    groups = defaultdict(list)
    for row in phase["episodes"]:
        manifest_path = Path(row["result_path"])
        assert io.sha(manifest_path) == row["result_sha256"]
        manifest = io.read(manifest_path)
        if not manifest["collisions"]:
            continue
        path = manifest_path.with_name("steps.json")
        last = io.read(path)[-1]
        assert last["action"] == "forward" and last["collision"] and manifest["collisions"] == 1
        floor = SPECS[row["layout"]]["source_floor"]
        groups[floor, tuple(last["position"]), last["heading"]].append(
            {"layout": row["layout"], "arm": row["arm"], "paid": last["step"],
             "steps_sha256": io.sha(path), "last_laser_sha256": last["scan_ranges_sha256"],
             "last_depth_sha256": last["depth_values_sha256"]})
    findings = []
    for (floor, start, heading), runs in sorted(groups.items()):
        dr, dc = DIRECTIONS[heading]
        end = (start[0]+dr, start[1]+dc)
        world = make_world(runs[0]["layout"])
        world.position, world.heading = start, heading
        assert not world._edge_valid[*start, heading]
        scan, depth = array_sha(world.scan().ranges_m), array_sha(world.sense().depth_m)
        assert all((run["last_laser_sha256"], run["last_depth_sha256"]) == (scan, depth) for run in runs)
        result = profile(geometry(floor), world.xy(start), world.xy(end), world.config.robot_radius_m)
        with np.load(OFFLINE / "precomputed_assets" / f"{floor}.npz", allow_pickle=False) as old:
            orig_valid = bool(old["edge_valid"][*start, heading])
        if result["body_band_intersection"]:
            disposition = "body_band_intersection"
        elif result["outside_and_unmeasured_m2"] > 1e-12:
            disposition = "unmeasured_exterior_support"
        elif not (result["seam_evidence_passed"] or result["direct_projection_residual_measured"]):
            disposition = "source_lacks_full_in_band_support"
        else:
            disposition = "extracted_support_rejection_with_in_band_projection"
        if not result["body_band_intersection"]:
            assert not all(result[k] for k in ("repaired_sweep_supported", "repaired_start_supported",
                                                "repaired_end_supported"))
        findings.append({"source_floor": floor, "cell": start, "target": end, "heading": heading,
                         "original_cache_allowed": orig_valid, "disposition": disposition,
                         "raw_geometry": result, "runs": runs})
    counts = Counter(item["disposition"] for item in findings for _ in item["runs"])
    output = {"scope": "frozen scan model, no planner invocations and no physical ground truth",
              "formal_receipt_sha256": io.sha(phase_path), "distinct_edges": len(findings),
              "run_failure_counts": dict(counts), "failures": findings}
    io.write(ROOT / "RERUN_COLLISION_AUDIT.json", output)
    print(dict(counts), flush=True)


if __name__ == "__main__":
    main()
