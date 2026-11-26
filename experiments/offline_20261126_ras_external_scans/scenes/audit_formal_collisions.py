"""Recheck formal failed swept edges without modifying or running planners."""

from collections import Counter, defaultdict
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import shapely
from shapely.geometry import LineString, Polygon
import trimesh

from scene import HERE, make_world, sha, source_scene
from sources import ADMISSION
import replicacad_world as geometry


def read(path):
    return json.loads(path.read_text())


def array_sha(values):
    return hashlib.sha256(np.ascontiguousarray(values).tobytes()).hexdigest()


def observed_hit_margin(points, p, q, lower, upper, radius):
    points = points[(points[:, 2] >= lower) & (points[:, 2] <= upper), :2]
    line = q - p
    down = np.clip((points - p) @ line / (line @ line), 0, 1)
    d = np.linalg.norm(points - (p + down[:, None] * line), axis=1)
    return dict(body_band_hits=len(d), min_centerline_distance_m=float(d.min()) if len(d) else None,
                points_inside_swept_footprint=int(np.count_nonzero(d <= radius + 1e-9)))


def local_original_triangles(source, world_tf, p, q):
    line = LineString([p, q])
    lo_xy, hi_xy = np.minimum(p, q)-.6, np.maximum(p, q)+.6
    source_count = 0
    scene = trimesh.Scene()
    closest = None
    lo, hi = ADMISSION["body_collision_band_m"]
    for node in source.graph.nodes_geometry:
        local, name = source.graph[node]
        native = source.geometry[name]
        points = trimesh.transform_points(native.vertices, world_tf @ local)
        faces = points[native.faces]
        low, high = faces.min(1), faces.max(1)
        nearby = np.all((high[:, :2] >= lo_xy) & (low[:, :2] <= hi_xy), axis=-1)
        if not nearby.any():
            continue
        region = native.faces[nearby]
        scene.add_geometry(trimesh.Trimesh(vertices=points, faces=region, process=False))
        source_count += len(region)
        for index in np.flatnonzero(nearby & (high[:, 2] >= lo) & (low[:, 2] <= hi)):
            face = faces[index]
            clipped = geometry._clip_halfspace(geometry._clip_halfspace(face, lo, True), hi, False)
            if not len(clipped):
                continue
            distance = line.distance(geometry._project_polygon(clipped))
            if closest is None or distance < closest["distance_to_centerline_m"]:
                closest = dict(native_node=node, face_index=int(index), world_triangle_m=face.tolist(),
                               distance_to_centerline_m=float(distance))
    return scene, source_count, closest


def floor_support(scene, p, q):
    floor_parts = []
    low, high = ADMISSION["scan_floor_support_band_m"]
    for mesh in scene.geometry.values():
        faces = np.asarray(mesh.triangles)
        possible = faces[(faces[:,:,2].min(1) >= low) & (faces[:,:,2].max(1) <= high)]
        normal = np.cross(possible[:,1]-possible[:,0],possible[:,2]-possible[:,0])
        length = np.linalg.norm(normal,axis=1)
        keep = (np.abs(normal[:,2]) >= ADMISSION["scan_floor_normal_abs_cosine_minimum"]
                * np.maximum(length,1e-15)) & (length > 1e-12)
        for part in possible[keep]:
            poly=Polygon(part[:,:2])
            if poly.area>1e-12:
                floor_parts.append(poly)
    if not floor_parts:
        return False,0.0,0
    floor=shapely.union_all(floor_parts)
    line=LineString([p,q])
    covered=bool(floor.covers(line))
    return covered,float(line.distance(floor.boundary)) if covered else 0.0,len(floor_parts)


def sense_recheck(world, p, q, original):
    scan = world.scan()
    frame = world.sense()
    expected = {key: original[key] for key in ("scan_ranges_sha256", "depth_values_sha256", "rgb_values_sha256")}
    replay = dict(scan_ranges_sha256=array_sha(scan.ranges_m), depth_values_sha256=array_sha(frame.depth_m),
                  rgb_values_sha256=array_sha(frame.color_rgb))
    if expected != replay:
        raise ValueError("Before-action raw sensor arrays differ from formal observed hashes")
    lidar_hits = scan.ranges_m < scan.range_max_m
    angles = scan.angle_min_rad + np.flatnonzero(lidar_hits)*scan.angle_increment_rad
    lidar_points = scan.ranges_m[lidar_hits, None] * np.column_stack(
        [np.cos(angles), np.sin(angles), np.zeros(len(angles))])
    lidar_points = lidar_points @ scan.world_from_laser[:3,:3].T + scan.world_from_laser[:3,3]
    v, u = np.nonzero(frame.depth_m > 0)
    z = frame.depth_m[v,u]
    k = frame.intrinsic
    rgbd_points = z[:,None]*np.column_stack([(u-k[0,2])/k[0,0],(v-k[1,2])/k[1,1],np.ones(len(z))])
    rgbd_points = rgbd_points @ frame.world_from_camera[:3,:3].T + frame.world_from_camera[:3,3]
    low, high = ADMISSION["body_collision_band_m"]
    radius = world.config.robot_radius_m
    return dict(raw_value_hashes_verified=replay, laser_forward_depth_m=float(scan.ranges_m[len(scan.ranges_m)//2]),
                laser_height_m=world.config.laser_height_m, rgbd_valid_count=len(z),
                lidar=observed_hit_margin(lidar_points,p,q,low,high,radius),
                rgbd=observed_hit_margin(rgbd_points,p,q,low,high,radius))


def recheck_case(source_floor, cell, heading, runs):
    layout, paid, before, _, _ = runs[0]
    world = make_world(layout)
    world.position, world.heading, world.step_count = cell, heading, paid-1
    dr, dc = geometry.DIRECTIONS[heading]
    target = cell[0]+dr, cell[1]+dc
    p, q = np.asarray(world.xy(cell)), np.asarray(world.xy(target))
    local, faces, closest = local_original_triangles(source_scene(source_floor),world.world_from_source,p,q)
    supported,floor_distance,floor_faces=floor_support(local,p,q)
    r = world.config.robot_radius_m
    floor_block = not supported or floor_distance < r - 1e-9
    body_block = closest is not None and closest["distance_to_centerline_m"] <= r + 1e-9
    if world._edge_valid[cell+(heading,)] or not (floor_block or body_block):
        raise ValueError("Original source triangles fail to reproduce the recorded swept-edge disposition")
    sensor = sense_recheck(world,p,q,before)
    dossier = []
    for layout, paid, before, failed, directory in runs:
        with np.load(directory/"observed_belief_trace.npz",allow_pickle=False) as stored:
            assert int(stored["paid"][paid-1]) == paid-1
            belief = stored["belief"][paid-1]
            observed_start, observed_end = int(belief[cell]), int(belief[target])
            surrounding_occupied = int(np.count_nonzero(belief[target[0]-1:target[0]+2,target[1]-1:target[1]+2] == 1))
        repeated_observations = {key: before[key] for key in sensor["raw_value_hashes_verified"]}
        if repeated_observations != sensor["raw_value_hashes_verified"]:
            raise ValueError("Duplicate geometry edges under different actors have different raw sensor hashes")
        dossier.append(dict(layout=layout, policy=directory.parts[-3], paid_collision=paid,
                            steps_path=str(directory/"steps.json"), steps_sha256=sha(directory/"steps.json"),
                            observed_start_belief=observed_start, observed_target_belief=observed_end,
                            adjacent_3x3_observed_occupied=surrounding_occupied, guard_reason=failed["guard_reason"]))
    return dict(source_floor=source_floor, cell=list(cell), heading=heading,target=list(target),
                capsule_radius_m=r, centerline_supported=supported,floor_clearance_m=floor_distance,
                body_clearance_m=None if closest is None else closest["distance_to_centerline_m"],
                geometry_window_triangles=faces,geometry_window_floor_support_triangles=floor_faces,nearest_body_face=closest,
                floor_support_rejection=floor_block,body_band_intersection=body_block,
                disposition=("body_and_floor" if floor_block else "body_band") if body_block else "floor_support_only",
                original_triangle_recheck_matches_frozen_edge=True, before_action_sensor=sensor, runs=dossier)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output",type=Path,default=HERE.parent/"FORMAL_COLLISION_AUDIT.json")
    args=parser.parse_args()
    phase=HERE.parent/"execution/formal"
    sources=read(HERE/"STARTS.json")["layouts"]
    groups=defaultdict(list)
    manifests=list(phase.glob("*/*/attempt_*/run/manifest.json"))
    if len(manifests) != 12:
        raise ValueError("Formal twelve-run matrix not terminal yet")
    run_outcomes=[]
    for path in sorted(manifests):
        receipt=read(path)
        layout=receipt["layout"]
        outcome=receipt["observed_native_termination"]["reason"]
        run_outcomes.append(dict(layout=layout,arm=receipt["arm"],termination=outcome,
                                 actual_paid=receipt["paid_actions"],returned=receipt["returned"]))
        if not receipt["collisions"]:
            continue
        rows=read(path.with_name("steps.json"))
        before,last=rows[-2:]
        assert last["action"] == "forward" and last["collision"] and receipt["collisions"] == 1
        key=sources[layout]["source_floor"],tuple(before["position"]),before["heading"]
        groups[key].append((layout,last["step"],before,last,path.parent))
    details=[recheck_case(*key,values) for key,values in sorted(groups.items())]
    report={"scope":"original scanned triangle and raw sensor verification; no rerun of any formal policy",
            "source_geometry_postprocess_changed":False,"runtime_changed":False,
            "failure_classification_limit":"Floor support failure concerns the scanned support mesh under the frozen tolerance; an actual drop-off versus scan fragmentation cannot be inferred from this test.",
            "total_formal_outcomes":len(run_outcomes),"total_collision_outcomes":sum(map(len,groups.values())),
            "distinct_source_swept_edges":len(details),
            "unique_pose_failure_counts":dict(Counter(row["disposition"] for row in details)),
            "run_failure_counts":dict(Counter(row["disposition"] for row in details for _ in row["runs"])),
            "run_outcomes":run_outcomes,"failures":details}
    with args.output.open("x") as file:
        json.dump(report,file,indent=2,allow_nan=False)
        file.write("\n")
    print(json.dumps({key:report[key] for key in ("total_formal_outcomes","total_collision_outcomes",
        "distinct_source_swept_edges","unique_pose_failure_counts","run_failure_counts")},indent=2))


if __name__=="__main__":
    main()
