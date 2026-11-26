"""Probe the archived HM3D entrance using only observed laser/RGB-D packets."""

from collections import defaultdict
import hashlib
import json
import math
from pathlib import Path
import sys

import numpy as np
from scipy.spatial import cKDTree

OUT = Path(__file__).resolve().parent
ANALYSES = OUT.parents[1]
STUDY = ANALYSES / "external_support_continuity_20261126"
LAYOUT = "b26_hm3d_00770_s1"
ARM = "combined_S"
ROOT = STUDY / f"execution/formal/{LAYOUT}/{ARM}/attempt_001/run"
SHARED = STUDY / "scenes"
sys.path.insert(0, str(SHARED))
from scene import make_world


EDGES = (
    ((49, 24), (49, 25)),
    ((49, 25), (50, 25)),
    ((50, 25), (50, 26)),
    ((50, 26), (51, 26)),
    ((51, 26), (52, 26)),
    ((52, 26), (53, 26)),
)
LEVELS_M = [.16, .25, .40, .55, .70, .85, 1.0, 1.15, 1.19]
CUTOFFS = [8, 17, 37, 192]
BODY_HEIGHT = (.15, 1.2)
RADIUS_M = .20
SAMPLE_STEP_M = .02
MIN_RANGE_RESPONSE_M = .01


def data_sha(array):
    return hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest()


def segment_dist(xy, p, q):
    vec = q - p
    t = np.clip(((xy-p) @ vec)/(vec @ vec), 0., 1.)
    return np.linalg.norm(xy - (p + t[:, None] * vec), axis=1)


def footprint_grid(p, q):
    low = np.minimum(p, q)-RADIUS_M
    high = np.maximum(p, q)+RADIUS_M
    # Center of 2 cm square probes. These are discretized diagnostics.
    axes = [np.arange(x+SAMPLE_STEP_M/2, y, SAMPLE_STEP_M) for x,y in zip(low, high, strict=True)]
    xx, yy = np.meshgrid(*axes)
    planar = np.column_stack([xx.ravel(), yy.ravel()])
    inside = planar[segment_dist(planar, p, q) <= RADIUS_M]
    new = np.linalg.norm(inside-p, axis=1) > RADIUS_M
    return inside, new


def scan_xyz(scan):
    angle = scan.angle_min_rad + np.arange(len(scan.ranges_m))*scan.angle_increment_rad
    real = (scan.ranges_m > 0) & (scan.ranges_m < scan.range_max_m - .001)
    angle, distance = angle[real], scan.ranges_m[real]
    local = np.column_stack((np.cos(angle)*distance, np.sin(angle)*distance, np.zeros(len(distance))))
    return local @ scan.world_from_laser[:3, :3].T + scan.world_from_laser[:3, 3]


def scan_visible(scan, xy):
    origin = scan.world_from_laser[:2, 3]
    vec = (xy-origin) @ scan.world_from_laser[:2, :2]
    target_dist = np.linalg.norm(vec, axis=1)
    angle = np.arctan2(vec[:, 1], vec[:, 0])
    bin_float = (angle - scan.angle_min_rad)/scan.angle_increment_rad
    n = len(scan.ranges_m)
    b = np.floor(bin_float).astype(int)
    dist_a, dist_b = (scan.ranges_m[i % n] for i in (b, b+1))
    within = (dist_a > 0) & (dist_a < scan.range_max_m-.001) & (dist_b > 0) & (dist_b < scan.range_max_m-.001)
    clear = within & (np.minimum(dist_a, dist_b) > target_dist + MIN_RANGE_RESPONSE_M)
    ray_gap = 2 * target_dist * np.sin(scan.angle_increment_rad/2)
    return np.where(clear, ray_gap, np.inf)


def depth_visible(frame, area):
    local = (area-frame.world_from_camera[:3, 3]) @ frame.world_from_camera[:3, :3]
    k = frame.intrinsic
    uv = (local @ k.T)[:, :2]/np.where(local[:, 2,None] > 0, local[:, 2,None], np.inf)
    upix, vpix = np.floor(uv).astype(int).T
    h,w = frame.depth_m.shape
    valid = (local[:,2] > .15) & (upix>=0) & (upix+1<w) & (vpix>=0) & (vpix+1<h)
    inside = np.flatnonzero(valid)
    pixel_history = [frame.depth_m[vpix[inside]+dy, upix[inside]+dx] for dx in (0,1) for dy in (0,1)]
    lo = np.min(pixel_history,axis=0)
    valid[inside] = lo > local[inside,2] + MIN_RANGE_RESPONSE_M
    return valid


def read_poses(world):
    rows = json.loads((ROOT / "steps.json").read_text())
    registers = [(0, world.start, world.heading, None)] + [(r["step"], tuple(r["position"]), r["heading"], r) for r in rows]
    processed = set()
    for t, cell, h, row in registers:
        if (cell,h) in processed:
            continue
        processed.add((cell,h))
        world.step_count, world.position, world.heading = t, cell, h
        scan, frame = world.scan(), world.sense()
        if row:
            assert data_sha(scan.ranges_m) == row["scan_ranges_sha256"], (t, cell, "scan")
            assert data_sha(frame.depth_m) == row["depth_values_sha256"], (t, cell, "depth")
        yield t, cell, h, scan, frame


def metrics(points2d, novel, lidar_gap, body_vis, ground_distance):
    groups = {}
    for key, mask in [("full_capsule", np.ones(len(novel),dtype=bool)),("beyond_start_disk", novel)]:
        n = int(mask.sum())
        ground = {str(a): int(np.count_nonzero(ground_distance[mask] <= a)) for a in (.015, .025, .05, .075)}
        beam = {str(width): int(np.count_nonzero(lidar_gap[mask] <= width)) for width in (.02, .04, .06, .08, .1)}
        groups[key] = {
            "sample_points": n,
            "positive_ground_return_within_radius_counts": ground,
            "valid_neighboring_laser_returns_clear_for_beam_gap_at_most_counts": beam,
            "body_depth_line_clear_coverage_by_height_counts": dict(zip(map(str,LEVELS_M),body_vis[mask].sum(axis=0).tolist(),strict=True)),
            "body_depth_line_clear_at_every_test_height_count": int(np.count_nonzero(np.all(body_vis[mask],axis=1))),
            "maximum_nearest_positive_ground_return_m": None if not np.isfinite(ground_distance[mask]).all() else float(ground_distance[mask].max()),
        }
    return groups


def nearest_hit(points, p, q, tick):
    if not len(points):
        return None
    ds = segment_dist(points[:, :2], p, q)
    best = int(np.argmin(ds))
    return {"body_xy_distance_to_centerline_m": float(ds[best]), "point_xyz_m": points[best].tolist(), "paid": tick}


def main():
    world = make_world(LAYOUT)
    report = {}
    for edge in EDGES:
        p,q = (np.asarray(world.xy(x)) for x in edge)
        points2d, novel = footprint_grid(p,q)
        pos3 = np.column_stack([np.repeat(points2d, len(LEVELS_M), axis=0), np.tile(LEVELS_M, len(points2d))])
        report[edge] = {"p":p,"q":q,"xy":points2d,"novel":novel,"body_vis":np.zeros((len(points2d),len(LEVELS_M)),dtype=bool),
                        "lidar_gap":np.full(len(points2d),np.inf),"body_query":pos3,"per_paid":{},"clearance":{}}
    ground_pool = []
    hit_volume = defaultdict(list)
    camera_visible_dist = 1.e6
    poses = {t:(cell,h,scan,frame) for t,cell,h,scan,frame in read_poses(world)}
    actual_used_ticks = []
    for t in range(max(CUTOFFS)+1):
        if t in poses:
            cell,h,scan,frame = poses[t]
            actual_used_ticks.append(t)
            laser_returns = scan_xyz(scan)
            full_cloud,_ = frame.points(stride=1)
            ground_obs = full_cloud[np.abs(full_cloud[:,2])<=.04]
            ground_pool.append(ground_obs)
            depth_returns = full_cloud[(full_cloud[:,2]>=BODY_HEIGHT[0])&(full_cloud[:,2]<=BODY_HEIGHT[1])]
            for row in report.values():
                p,q = row["p"],row["q"]
                for modality,points in [("lidar_25cm",laser_returns),("rgbd_body_band",depth_returns)]:
                    closest = nearest_hit(points,p,q,t)
                    prev = row["clearance"].get(modality)
                    if closest and (prev is None or closest["body_xy_distance_to_centerline_m"] < prev["body_xy_distance_to_centerline_m"]):
                        row["clearance"][modality] = closest | {"source_cell": cell, "source_heading": h}
                row["lidar_gap"] = np.minimum(row["lidar_gap"],scan_visible(scan,row["xy"]))
                row["body_vis"] |= depth_visible(frame,row["body_query"]).reshape(row["body_vis"].shape)
        if t in CUTOFFS:
            ground_points = np.vstack(ground_pool)
            ground = cKDTree(ground_points[:,:2]) if len(ground_points) else None
            for row in report.values():
                if ground is None:
                    ground_dist = np.full(len(row["xy"]),np.inf)
                else:
                    ground_dist = ground.query(row["xy"])[0]
                row["per_paid"][str(t)] = {
                    "evidence_counts":metrics(row["xy"],row["novel"],row["lidar_gap"],row["body_vis"],ground_dist),
                    "nearest_actual_body_band_hit_so_far":row["clearance"].copy(),
                }
    result = {
        "scope":"Original recorded poses and actual pixel/beam values; no policy used and no raw mesh/GT traversal labels queried in any evidence test.",
        "archive": str(ROOT), "layout":LAYOUT, "arm":ARM, "laser_height_m":world.config.laser_height_m,"laser_rays":world.config.laser_rays,
        "body_height_m":BODY_HEIGHT, "ground_fixed_diagnostic_half_band_m":.04,
        "laser_clearance_beyond_probe_m":MIN_RANGE_RESPONSE_M, "sweep_radius_m":RADIUS_M,
        "grid_sample_step_m":SAMPLE_STEP_M, "camera_height_m":world.config.camera_height_m,"camera_vertical_boundary_optical_ray":float((world.config.height_px-1-world.intrinsic[1,2])/world.intrinsic[1,1]),
        "unique_acquired_poses_checked":len(actual_used_ticks),"final_unique_acquired_pose_first_time_actions":actual_used_ticks,
        "sweeps":[{"start":a,"end":b,"evidence_available_at_paid":x["per_paid"]} for (a,b),x in report.items()],
        "limits":["Nearest endpoint distance is observed clearance only, not a lower bound on unseen obstacles.",
                  "Four-neighbor depth ray test, discrete probes and positive floor returns do not certify the unmeasured area between rays.",
                  "Floor band is a diagnostic from expected nominal sensor mounting alone. Live application needs locally observed ground level and bounded calibration and pose errors."],
    }
    (OUT / "archived_sensor_passage_probe.json").write_text(json.dumps(result,indent=2,allow_nan=False)+"\n")


if __name__ == "__main__":
    main()
