"""Exercise a sensor-only swept capsule with analytically generated observations."""

from argparse import ArgumentParser
from dataclasses import dataclass
import hashlib
import importlib
import importlib.util
import json
from pathlib import Path
import sys

import numpy as np


RADIUS_M = 0.20
TRAVEL_M = 0.20
SCAN_HEIGHT_M = 0.25
SCAN_MAX_M = 8.0
SCAN_ANGLES_RAD = np.deg2rad(np.arange(-180.0, 180.0, 2.0))
GRID_ORIGIN_M = np.array((-0.10, -0.10))
GRID_RESOLUTION_M = 0.20
ROOM = [((-2.0, -2.0), (-2.0, 2.0)), ((2.0, -2.0), (2.0, 2.0)),
        ((-2.0, -2.0), (2.0, -2.0)), ((-2.0, 2.0), (2.0, 2.0))]


@dataclass(frozen=True)
class Trial:
    name: str
    packet: dict
    expected: bool
    basis: str


def cross(a, b):
    return a[..., 0]*b[..., 1] - a[..., 1]*b[..., 0]


def cast_scan(walls, visible_limit_rad, no_return):
    directions = np.column_stack((np.cos(SCAN_ANGLES_RAD), np.sin(SCAN_ANGLES_RAD)))
    ranges = np.full(len(directions), SCAN_MAX_M)
    for p, q in walls:
        start = np.asarray(p)
        delta = np.asarray(q) - start
        determinant = cross(directions, delta)
        not_parallel = np.abs(determinant) > 1e-10
        rr = np.divide(cross(start, delta), determinant, out=np.full_like(ranges, np.inf), where=not_parallel)
        along = np.divide(cross(start, directions), determinant, out=np.full_like(ranges, np.inf), where=not_parallel)
        hit = not_parallel & (rr > 0.0) & (along >= 0.0) & (along <= 1.0)
        ranges[hit] = np.minimum(rr[hit], ranges[hit])
    ranges[np.abs(SCAN_ANGLES_RAD) > visible_limit_rad] = np.nan
    ranges[np.abs(SCAN_ANGLES_RAD) <= no_return] = SCAN_MAX_M
    return ranges


def packet(*, walls=(), direction_deg=0.0, offset=(0.0, 0.0), visible_limit_deg=180.0,
           no_return_deg=-1.0, obstacle_points=(), drop_points=(), travel_m=TRAVEL_M,
           missing_forward_deg=-1.0, floor_observed=True):
    radians = np.deg2rad(direction_deg)
    rotation = np.array(((np.cos(radians), -np.sin(radians)), (np.sin(radians), np.cos(radians))))
    start_xy = np.asarray(offset)
    end_xy = np.array((travel_m, 0.0)) @ rotation.T + start_xy

    def transform_points(points):
        values = np.asarray(points, dtype=float).reshape(-1, 3).copy()
        values[:, :2] = values[:, :2] @ rotation.T + start_xy
        return values

    ranges = cast_scan([*ROOM, *walls], np.deg2rad(visible_limit_deg), np.deg2rad(no_return_deg))
    ranges[np.abs(SCAN_ANGLES_RAD) <= np.deg2rad(missing_forward_deg)] = np.nan
    if floor_observed:
        x, y = np.meshgrid(np.arange(-0.24, travel_m + 0.261, 0.02), np.arange(-0.24, 0.261, 0.02))
        floor = np.column_stack((x.ravel(), y.ravel(), np.zeros(x.size)))
        for point in np.asarray(drop_points, dtype=float).reshape(-1, 3):
            floor = floor[np.linalg.norm(floor[:, :2]-point[:2], axis=1) > 0.07]
    else:
        floor = np.empty((0, 3))
    return {
        "start_xyz_m": np.r_[start_xy, 0.0],
        "end_xyz_m": np.r_[end_xy, 0.0],
        "radius_m": RADIUS_M,
        "laser_origin_xyz_m": np.r_[start_xy, SCAN_HEIGHT_M],
        "laser_azimuths_rad": SCAN_ANGLES_RAD + radians,
        "laser_ranges_m": ranges,
        "laser_range_max_m": SCAN_MAX_M,
        "range_uncertainty_m": 0.002,
        "floor_surface_xyz_m": transform_points(floor),
        "rgbd_obstacle_points_xyz_m": transform_points(obstacle_points),
        "rgbd_drop_points_xyz_m": transform_points(drop_points),
        "grid_origin_xy_m": GRID_ORIGIN_M.copy(),
        "map_resolution_m": GRID_RESOLUTION_M,
    }


def tests():
    def doorway(half_width):
        depth = 0.19
        return [((depth, -2.0), (depth, -half_width)), ((depth, half_width), (depth, 2.0))]

    def sidewall(y):
        return [((0.20, y), (0.65, y))]

    # Jambs fall in the same respective grid cells even though the capsules differ.
    for sign in (-1.0, 1.0):
        near = np.floor((sign*0.19-GRID_ORIGIN_M[1])/GRID_RESOLUTION_M)
        far = np.floor((sign*0.21-GRID_ORIGIN_M[1])/GRID_RESOLUTION_M)
        assert near == far

    return [
        Trial("observed_open_room_admitted", packet(), True,
              "finite returns surround the sweep and all hit points are at least 1.8 m away"),
        Trial("wall_inside_swept_radius_rejected", packet(walls=sidewall(0.16)), False,
              "a finite segment at 0.16 m intersects the footprint near the end, clear of the start disk"),
        Trial("wall_outside_swept_radius_admitted", packet(walls=sidewall(0.24)), True,
              "a finite measured wall at 0.24 m stays outside the 0.20 m capsule"),
        Trial("doorway_inside_exact_radius_rejected", packet(walls=doorway(0.19)), False,
              "0.38 m opening is too narrow although both jambs quantize like the safe doorway"),
        Trial("same_grid_cells_wider_doorway_admitted", packet(walls=doorway(0.21)), True,
              "0.42 m opening clears the exact footprint although both jamb cells match the unsafe doorway"),
        Trial("unobserved_forward_wedge_rejected", packet(missing_forward_deg=55.0, floor_observed=False), False,
              "invalid forward rays leave the newly swept area unobserved despite finite surrounding rays"),
        Trial("unobserved_lateral_area_rejected", packet(visible_limit_deg=12.0, floor_observed=False), False,
              "a narrow forward slice does not observe the newly swept lateral sides"),
        Trial("no_return_forward_wedge_rejected", packet(no_return_deg=55.0, floor_observed=False), False,
              "max-range no-return rays do not establish traversal through a large unsupported forward arc"),
        Trial("measured_lower_support_rejected", packet(drop_points=((0.33, 0.0, -0.20), (0.34, 0.05, -0.21))), False,
              "RGB-D shows a lower surface at the landing capsule while lidar sees the far enclosure"),
        Trial("rotated_open_capsule_admitted", packet(direction_deg=37.0, offset=(1.1, -0.7)), True,
              "same free sweep after translation and a non-cardinal rotation"),
        Trial("rotated_endpoint_hit_rejected", packet(direction_deg=37.0, offset=(1.1, -0.7),
                                                         walls=(((0.36, -0.06), (0.36, 0.06)),)), False,
              "return is 0.16 m beyond the target, inside only the terminal cap"),
        Trial("rotated_beyond_endpoint_admitted", packet(direction_deg=-64.0, offset=(-2.2, 0.4),
                                                           walls=(((0.44, -0.06), (0.44, 0.06)),)), True,
              "return is 0.24 m beyond the target and outside both swept caps"),
        Trial("contradictory_rgbd_obstacle_wins", packet(obstacle_points=((0.31, 0.02, 0.18),)), False,
              "a body-height RGB-D point inside the endpoint cap overrides free lidar at another height"),
        Trial("rgbd_obstacle_outside_capsule_admitted", packet(obstacle_points=((0.31, 0.29, 0.18),)), True,
              "same RGB-D obstacle height, now outside the full swept footprint"),
    ]


def load_factory(source):
    module_name, name = source.rsplit(":", 1)
    module_path = Path(module_name).resolve()
    if module_name.endswith(".py"):
        sys.path.insert(0, str(module_path.parent))
        spec = importlib.util.spec_from_file_location(module_path.stem, module_path)
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)
        receipt = hashlib.sha256(module_path.read_bytes()).hexdigest()
    else:
        module = importlib.import_module(module_name)
        receipt = None
    return getattr(module, name), receipt


def main():
    cli = ArgumentParser(description=__doc__)
    cli.add_argument("--adapter", required=True, help="A packet-to-bool function as /absolute/module.py:function")
    args = cli.parse_args()
    adapter, identity = load_factory(args.adapter)
    results = []
    for trial in tests():
        actual = bool(adapter(trial.packet))
        results.append({"fixture": trial.name, "expected_admitted": trial.expected,
                        "actual_admitted": actual, "passes": trial.expected == actual, "basis": trial.basis})
    total = sum(row["passes"] for row in results)
    print(json.dumps({"schema": "nso_online_sweep_blind_regression_v1", "adapter": args.adapter,
                      "adapter_sha256": identity, "tests": results, "passed": total,
                      "total": len(results), "all_required_pass": total == len(results)}, indent=2))
    return int(total != len(results))


if __name__ == "__main__":
    sys.exit(main())
