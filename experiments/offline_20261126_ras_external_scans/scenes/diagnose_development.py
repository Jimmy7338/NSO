"""Explain the unrelated development smoke's swept-capsule disposition."""

import json
import math

import numpy as np
from shapely.geometry import LineString
import trimesh

from prepare import HERE, read, write
from scene import make_world, source_scene
from sources import ADMISSION
import replicacad_world as geometry


def main():
    run = HERE.parent / "development_smoke/run"
    steps = read(run / "steps.json")
    last = steps[-1]
    if not last["collision"] or last["action"] != "forward":
        raise ValueError("No terminal forward collision to diagnose")
    layout = read(run / "manifest.json")["layout"]
    world = make_world(layout)
    prior = steps[-2]
    for row in steps[:-1]:
        _, clearance_collision, _ = world.step(row["action"])
        if clearance_collision or list(world.position) != row["position"] or world.heading != row["heading"]:
            raise ValueError("Development action replay did not reproduce pre-collision state")
    start_cell = tuple(prior["position"])
    dr, dc = geometry.DIRECTIONS[prior["heading"]]
    end_cell = (start_cell[0]+dr, start_cell[1]+dc)
    p, q = world.xy(start_cell), world.xy(end_cell)
    capsule_line = LineString([p, q])
    lo_xy, hi_xy = np.minimum(p, q)-.6, np.maximum(p, q)+.6
    source = source_scene(world.spec["source_floor"])
    source_count = 0
    scene = trimesh.Scene()
    closest = None
    lo, hi = ADMISSION["body_collision_band_m"]
    for node in source.graph.nodes_geometry:
        local, name = source.graph[node]
        native = source.geometry[name]
        native_coords = trimesh.transform_points(native.vertices, world.world_from_source @ local)
        faces = native_coords[native.faces]
        low, high = faces.min(1), faces.max(1)
        inside = np.all((high[:, :2] >= lo_xy) & (low[:, :2] <= hi_xy), axis=-1)
        if not inside.any():
            continue
        region = native.faces[inside]
        scene.add_geometry(trimesh.Trimesh(vertices=native_coords, faces=region, process=False))
        source_count += len(region)
        for i in np.flatnonzero(inside & (high[:, 2] >= lo) & (low[:, 2] <= hi)):
            f = faces[i]
            poly = geometry._clip_halfspace(geometry._clip_halfspace(f, lo, True), hi, False)
            if not len(poly):
                continue
            distance = capsule_line.distance(geometry._project_polygon(poly))
            if closest is None or distance < closest["distance_to_centerline_m"]:
                closest = dict(native_node=node, face_index=int(i), world_triangle_m=f.tolist(),
                               distance_to_centerline_m=float(distance))
    geometry.FLOOR_BAND_M = tuple(ADMISSION["scan_floor_support_band_m"])
    geometry.FLOOR_NORMAL_MIN = ADMISSION["scan_floor_normal_abs_cosine_minimum"]
    checker = geometry.MeshMotionGeometry(scene, np.eye(4))
    _, supported, clearance = checker.clearance(p, q)
    original_scan = world.scan()
    original_frame = world.sense()
    with np.load(run / "observed_belief_trace.npz", allow_pickle=False) as archive:
        frame_belief = archive["belief"][prior["step"]]
    direct = original_scan.ranges_m[len(original_scan.ranges_m)//2]
    report = {
        "layout": layout, "clock": prior["step"], "rejected_physical_action": last["step"],
        "actual_native_disposition": "collision_no_return", "solver_not_changed": True,
        "capsule_radius_m": world.config.robot_radius_m, "linear_displacement_m": math.dist(p, q),
        "centerline_supported_by_original_floor": supported,
        "floor_support_min_distance_m": clearance, "original_triangle_window_face_count": source_count,
        "nearest_body_band_face": closest, "cache_agrees_with_triangle_recheck":
            bool((not world._edge_valid[start_cell+(prior["heading"],)])
                 and not checker.valid(p, q, world.config.robot_radius_m)),
        "start_cell": start_cell, "proposed_end_cell": end_cell,
        "start_observed_belief": int(frame_belief[start_cell]),
        "target_observed_belief": int(frame_belief[end_cell]),
        "observed_guard": last["guard_reason"],
        "measured_scan_height_m": world.config.laser_height_m,
        "measured_forward_scan_depth_m": float(direct),
        "measured_forward_rgbd_valid_count": int((original_frame.depth_m > 0).sum()),
        "geometry_source": world.spec["source_floor"],
    }
    if not report["cache_agrees_with_triangle_recheck"]:
        raise ValueError("Fresh original-triangle check disagrees with cache")
    write(HERE.parent / "development_smoke/COLLISION_DIAGNOSIS.json", report)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
