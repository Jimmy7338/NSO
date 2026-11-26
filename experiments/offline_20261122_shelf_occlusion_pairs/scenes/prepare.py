#!/usr/bin/env python3
"""Prepare actual zero-action packets, paired references and geometric audits."""

import argparse
from collections import deque
import hashlib
import json
from pathlib import Path

import numpy as np
import open3d as o3d
import shapely

from scene import (ANALYSES, ASSETS, BOARD_NODE, BUDGET, DEV_PARENTS, HERE, PARENTS,
                   SHELF_LABEL, SOURCE, SPECS, build_scene, load_module, make_world, panel, source)


PREPARER = ANALYSES / "same_goal_route_choice_20261121/holdout/prepare.py"
METRIC = ANALYSES / "dual_template_exploration_20261119/audit/surface_metrics.py"
NATIVE = load_module("nso_shelf_native_reference_preparation", PREPARER)
NATIVE.native_reference.build_scene = build_scene
RULES = NATIVE.read(HERE / "reference_protocol.json")


def arrays(path):
    with np.load(path, allow_pickle=False) as entries:
        return {name: entries[name] for name in entries.files}


def array_sha(array):
    return hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest()


def source_audit(open_name, screen_name):
    scene_a, records_a, metadata_a = build_scene(open_name)
    scene_b, records_b, metadata_b = build_scene(screen_name)
    assert records_a == records_b
    assert set(scene_b.graph.nodes_geometry) - set(scene_a.graph.nodes_geometry) == {BOARD_NODE}
    assert set(metadata_b) - set(metadata_a) == {BOARD_NODE}
    for node in scene_a.graph.nodes_geometry:
        mat_a, mesh_a = scene_a.graph[node]
        mat_b, mesh_b = scene_b.graph[node]
        assert metadata_a[node] == metadata_b[node]
        assert np.array_equal(mat_a, mat_b)
        piece_a, piece_b = scene_a.geometry[mesh_a], scene_b.geometry[mesh_b]
        assert np.array_equal(piece_a.vertices, piece_b.vertices)
        assert np.array_equal(piece_a.faces, piece_b.faces)
        assert np.array_equal(piece_a.visual.face_colors, piece_b.visual.face_colors)
    center, extents = panel(screen_name)
    z_min = center[2] - extents[2] / 2
    assert z_min > source.old.COLLISION_BAND_M[1]
    return dict(pair_delta_nodes=[BOARD_NODE], same_all_other_node_geometry_and_appearance=True,
                panel_center_xyz_m=center, panel_extents_xyz_m=extents, panel_bottom_m=z_min,
                native_collision_band_m=source.old.COLLISION_BAND_M)


def shortest(edges, start, target):
    previous = {start: (None, None)}
    queue = deque([start])
    while queue:
        current = queue.popleft()
        if current == target:
            states, actions = [current], []
            while previous[current][0] is not None:
                current, action = previous[current]
                states.append(current)
                actions.append(action)
            return states[::-1], actions[::-1]
        row, col, heading = current
        dr, dc = source.old.DIRECTIONS[heading]
        options = [((row, col, (heading - 1) % 4), "left"),
                   ((row, col, (heading + 1) % 4), "right")]
        if edges[current]:
            options.insert(0, ((row + dr, col + dc, heading), "forward"))
        for following, action in options:
            if following not in previous:
                previous[following] = current, action
                queue.append(following)
    raise ValueError(f"No native oriented footprint path from {start} to {target}")


def clearance(world, states, actions):
    least = float("inf")
    geometry = world._geometry
    for index, action in enumerate(actions):
        start, stop = states[index][:2], states[index + 1][:2]
        start_xyz, stop_xyz = world.xy(start), world.xy(stop)
        endpoint = stop_xyz if action == "forward" else None
        assert geometry.valid(start_xyz, endpoint, world.config.robot_radius_m)
        line, supported, floor_distance = geometry.clearance(start_xyz, endpoint)
        assert supported
        obstacle_distance = float(np.min(shapely.distance(geometry.obstacles, line)))
        least = min(least, floor_distance, obstacle_distance)
    return least


def cold(layout):
    folder = HERE / "prepared" / layout
    path = folder / "initial_state.json"
    if not path.exists():
        NATIVE.SEED = SPECS[layout]["seed"]
        NATIVE.initial(layout)
    result = NATIVE.read(path)
    assert result["policy_select_calls"] == result["paid_actions"] == 0 and result["stop_budget"] == BUDGET
    assert result["seed"] == SPECS[layout]["seed"] and result["anchor"] == [*SPECS[layout]["start"], SPECS[layout]["heading"]]
    return result


def metadata(mesh, all_points, visible, seed, rays):
    area = float(mesh.get_surface_area())
    return dict(sampled_source_area_m2=area, drawn_samples=len(all_points), visible_samples=int(visible.sum()),
                estimated_visible_area_m2=float(area * visible.mean()), area_per_sample_m2=area / len(all_points),
                seed=seed, first_hit_rays=sum(rays))


def prepare(parent):
    labels = [f"{parent}_open", f"{parent}_screen"]
    initials = [cold(layout) for layout in labels]
    assert all(initials[0][key] == initials[1][key] for key in ("anchor", "seed", "shape", "actual_config"))
    dirs = [HERE / "evaluator_references" / layout for layout in labels]
    truths = [arrays(directory / "evaluator_truth.npz") for directory in dirs]
    assert truths[0].keys() == truths[1].keys()
    assert all(np.array_equal(truths[0][key], truths[1][key]) for key in truths[0])
    packet_dirs = [HERE / "prepared" / layout for layout in labels]
    scans = [arrays(directory / "initial_scan.npz") for directory in packet_dirs]
    assert scans[0].keys() == scans[1].keys()
    assert all(np.array_equal(scans[0][key], scans[1][key]) for key in scans[0])
    frames = [arrays(directory / "initial_frame.npz") for directory in packet_dirs]
    assert np.array_equal(frames[0]["world_from_camera"], frames[1]["world_from_camera"])
    assert np.array_equal(frames[0]["intrinsic"], frames[1]["intrinsic"])
    assert not np.array_equal(frames[0]["depth_m"], frames[1]["depth_m"])

    physical = source_audit(*labels)
    q = SPECS[labels[0]]
    config = source.ControlledConfig(**initials[0]["actual_config"])
    reachable = truths[0]["reachable"]
    anchor = tuple(initials[0]["anchor"])
    x, y = q["shelf"]
    dx, dy = q["flank"]
    side_rc = source.xy_cell(x + dx, y + dy, q["height_m"])
    side = (*side_rc, 3 if dx > 0 else 1)
    side_pose = source.old.camera_pose(side[:2], side[2], config, reachable.shape[0])
    front_pose = source.old.camera_pose(anchor[:2], anchor[2], config, reachable.shape[0])
    assert reachable[side_rc], f"{parent} side witness is disconnected"
    paths = [shortest(truths[0]["edge_valid"], first, last) for first, last in ((anchor, side), (side, anchor))]
    assert sum(len(actions) for _, actions in paths) <= BUDGET
    world = make_world(labels[0])
    margin = min(clearance(world, states, actions) for states, actions in paths)
    del world

    # Add the predeclared side witness to the common fixed camera lattice so
    # jointly visible points there count against the same shelf denominator.
    regular_poses, lattice_cells = NATIVE.fixed_camera_poses(reachable, config, anchor[:2])
    poses = np.concatenate((regular_poses, side_pose[None]), axis=0)
    source_geometry = [NATIVE.native_reference.source_mesh(layout) for layout in labels]
    sampled = [NATIVE.unique_sampling_faces(mesh, tag) for mesh, tag, _ in source_geometry]
    rays = [NATIVE.ray_scene(mesh) for mesh, _, _ in source_geometry]
    count = RULES["per_target_area_samples"]
    base = RULES["base_seed"] + 1000 * tuple(PARENTS).index(parent)
    eps = RULES["first_hit_tolerance_m"]

    def visible(points, ray, selected_poses=poses):
        return NATIVE.visible_reference(points, ray, selected_poses, config, tolerance_m=eps)

    shared, meta = {}, {}
    for name, tag in (("open31", SHELF_LABEL), ("closed118", 118)):
        meshes = [NATIVE.subset_mesh(mesh, tags == tag) for mesh, tags, _ in sampled]
        points, indices = NATIVE.surface_samples(meshes[0], count, base + tag)
        both_points, both_indices = NATIVE.surface_samples(meshes[1], count, base + tag)
        assert np.array_equal(points, both_points) and np.array_equal(indices, both_indices)
        masks = [visible(points, ray) for ray in rays]
        common = masks[0][0] & masks[1][0]
        assert common.any()
        shared[name] = (points, masks, common)
        meta[name] = metadata(meshes[0], points, common, base + tag, [entry[2] for entry in masks])

    points, _, common = shared["open31"]
    front_a = visible(points, rays[0], front_pose[None])[0]
    front_b = visible(points, rays[1], front_pose[None])[0]
    side_b = visible(points, rays[1], side_pose[None])[0]
    lost = front_a & ~front_b & common
    recovered = lost & side_b
    assert not np.any(front_b & ~front_a), "Adding an opaque board cannot reveal shelf points"
    shelf_pixels = [int(np.count_nonzero(frame["semantic"] == 2)) for frame in frames]
    assert shelf_pixels[1] > 30 and shelf_pixels[1] < shelf_pixels[0]
    if min(int(lost.sum()), int(recovered.sum())) < 30:
        raise ValueError(f"Inadequate geometry-only occlusion manipulation for {parent}: "
                         f"{lost.sum()} newly occluded with side witness {recovered.sum()}")

    for i, layout in enumerate(labels):
        directory = dirs[i]
        values, info = {"camera_poses": poses}, {}
        full_mesh, _, records = source_geometry[i]
        scene_sampling_mesh, _, duplicates = sampled[i]
        points_i, _ = NATIVE.surface_samples(scene_sampling_mesh, RULES["global_area_samples"], base)
        global_visible, first, ray_count = visible(points_i, rays[i])
        values["all"] = points_i[global_visible].astype(np.float32)
        values["all_first_pose"] = first[global_visible]
        info["all"] = metadata(scene_sampling_mesh, points_i, global_visible, base, [ray_count])
        for name, (common_points, masks, common_visible) in shared.items():
            values[name] = common_points[common_visible].astype(np.float32)
            values[f"{name}_first_pose"] = masks[i][1][common_visible]
            info[name] = meta[name] | {
                "shared_pair_eligible_intersection": True,
                "condition_visible_before_intersection": int(masks[i][0].sum()),
                "shared_points_sha256": array_sha(values[name]),
            }
        reference_file = directory / "fixed_reference.npz"
        if reference_file.exists():
            previous = arrays(reference_file)
            assert previous.keys() == values.keys() and all(np.array_equal(previous[key], values[key])
                                                            for key in previous)
        else:
            np.savez_compressed(reference_file, **values)
        source_file = directory / "source_mesh.ply"
        if not source_file.exists() and not o3d.io.write_triangle_mesh(str(source_file), full_mesh):
            raise RuntimeError(f"Could not write truth mesh for {layout}")
        deps = (Path(__file__), HERE / "scene.py", SOURCE, HERE / "reference_protocol.json", PREPARER,
                packet_dirs[i] / "initial_state.json", directory / "evaluator_truth.npz",
                Path(NATIVE.native_reference.__file__), METRIC)
        receipt = dict(status="fixed_source_only_surface_reference_prepared", layout=layout,
                       reference_npz_sha256=NATIVE.sha(reference_file), source_mesh_sha256=NATIVE.sha(source_file),
                       source_inputs_sha256={str(path): NATIVE.sha(path) for path in deps}, metadata=info,
                       reachable_cells=int(reachable.sum()), camera_lattice_cells=lattice_cells,
                       additional_predeclared_side_pose=list(side), camera_pose_count=len(poses),
                       duplicate_sampling_handling=duplicates, source_object_records=records,
                       actor_executed=False, actor_trajectory_read=False,
                       reference_source_is_native_geometry_not_template=True)
        path = directory / "RECEIPT.json"
        if path.exists():
            assert NATIVE.read(path)["reference_npz_sha256"] == receipt["reference_npz_sha256"]
        else:
            NATIVE.write(path, receipt)

    shared_weight = meta["open31"]["area_per_sample_m2"]
    audit = dict(
        parent=parent, layouts=labels, role="development" if parent in DEV_PARENTS else "heldout",
        shelf_asset=q["asset"], shelf_structure=ASSETS[q["asset"]], common_seed=q["seed"],
        fixed_robot_start=anchor, budget_actions=BUDGET, policy_calls=0,
        paired_path_clearance_min_m=float(margin), robot_radius_m=q["robot_radius_m"],
        side_pose=side, shortest_outbound_actions=paths[0][1], shortest_return_actions=paths[1][1],
        shortest_return_including_heading_actions=len(paths[0][1]) + len(paths[1][1]),
        reachable_cells=int(reachable.sum()), open_sensed_shelf_pixels=shelf_pixels[0],
        screen_sensed_shelf_pixels=shelf_pixels[1], shelf_reference_points=len(points[common]),
        shelf_reference_sha256=array_sha(points[common].astype(np.float32)), shelf_sample_weight_m2=shared_weight,
        front_visible_open_points=int((front_a & common).sum()), front_visible_screen_points=int((front_b & common).sum()),
        front_visible_open_fraction=float(front_a[common].mean()),
        front_visible_screen_fraction=float(front_b[common].mean()),
        front_hidden_by_screen_points=int(lost.sum()), hidden_points_seen_by_side_pose=int(recovered.sum()),
        hidden_area_seen_by_side_pose_m2=float(shared_weight * recovered.sum()), same_initial_raw_lidar=True,
        same_initial_intrinsic_pose=True, same_reachable_and_swept_graph=True,
        occupancy_sha256=array_sha(truths[0]["occupancy"]), swept_edges_sha256=array_sha(truths[0]["edge_valid"]),
        reference_evaluator_only=True) | physical
    path = HERE / "pair_audits" / f"{parent}.json"
    path.parent.mkdir(exist_ok=True)
    if path.exists():
        old = NATIVE.read(path)
        assert old["shelf_reference_sha256"] == audit["shelf_reference_sha256"]
    else:
        NATIVE.write(path, audit)
    print(json.dumps({key: audit[key] for key in (
        "parent", "front_visible_open_points", "front_visible_screen_points", "hidden_points_seen_by_side_pose",
        "shortest_return_including_heading_actions", "paired_path_clearance_min_m", "policy_calls")}), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("parent", choices=tuple(PARENTS))
    prepare(parser.parse_args().parent)
