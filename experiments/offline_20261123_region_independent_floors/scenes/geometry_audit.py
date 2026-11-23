#!/usr/bin/env python3
"""Apply the registered geometry and evidence checks before any actor runs."""

from collections import deque
import json
from pathlib import Path

import numpy as np
import open3d as o3d

from scene import (
    ANALYSES, ATTRIBUTION, BUDGET, HERE, LAYOUTS, PRIOR_SCENES, RADIUS_M,
    REGISTERED, RESOLUTION_M, SPECS, load_module, make_world,
)


prior = load_module("independent_floor_geometry_audit_contract", PRIOR_SCENES / "geometry_audit.py")
known = load_module("independent_floor_tour_contract", ATTRIBUTION / "geometry_audit.py")
read, sha = prior.read, prior.sha


def require(condition, reason):
    if not condition:
        raise ValueError(reason)


def verify_registration():
    receipt = read(HERE / "REGISTRATION_RECEIPT.json")
    require(receipt["observed_policy_results"] == REGISTERED["policy_calls_at_registration"] == 0,
            "Registration must precede any policy outcomes.")
    require(receipt["registered_layouts"] == list(LAYOUTS), "Registered layout order changed.")
    for member, expected in receipt["file_sha256"].items():
        require(sha(HERE / member) == expected, f"Registered source changed: {member}")
    return receipt


def old_seed_check():
    active = {spec["seed"] for spec in SPECS.values()}
    require(len(active) == len(SPECS), "Repeated seed within the registered cohort.")
    count = 0
    for path in sorted(ANALYSES.rglob("initial_state.json")):
        if HERE.parent in path.parents:
            continue
        previous = read(path).get("seed")
        if previous is not None:
            count += 1
            require(previous not in active, f"Registered seed repeats prior cold fixture: {path}")
    return count


def room_graph(spec):
    nodes, edges = spec["room_graph"]["rooms"], spec["room_graph"]["edges"]
    neighbors = {node: [] for node in nodes}
    for a, b in edges:
        neighbors[a].append(b)
        neighbors[b].append(a)
    queued, reached = deque([nodes[0]]), set()
    while queued:
        node = queued.popleft()
        if node not in reached:
            reached.add(node)
            queued.extend(neighbors[node])
    require(reached == set(nodes), "Room graph is disconnected.")
    return {"rooms": nodes, "edges": edges, "degree_by_room": {k: len(v) for k, v in neighbors.items()},
            "independent_cycle_count": len(edges) - len(nodes) + 1}


def sweep_margin(world, source, target):
    start, end = world.xy(source), world.xy(target)
    geometry = world._geometry
    line, supported, floor_distance = geometry.clearance(start, None if source == target else end)
    require(supported and geometry.valid(start, None if source == target else end, RADIUS_M),
            f"Native circle or capsule rejects a reported witness at {source} → {target}.")
    obstacle_distance = float(line.distance(geometry.obstacles[geometry.tree.nearest(line)]))
    margin = min(floor_distance, obstacle_distance) - RADIUS_M
    require(margin >= -1e-9, "Swept footprint exceeds available clearance.")
    return margin


def prove_clearance(layout, row, truth, spec):
    world = make_world(layout)
    require(world.step_count == 0 and world.position == tuple(row["start_rc_heading"][:2]), "Scene has moved.")
    _, interiors, _, _ = prior.map_rooms(spec, truth["reachable"].shape)
    tour = known.interior_visit_witness(
        truth["edge_valid"], truth["reachable"], interiors, tuple(row["start_rc_heading"]))
    require(tour["fits_cap"], "No source-only tour of all room interiors and exact return within 192 actions.")
    poses = tour["poses_rc_heading"]
    tour["min_clearance_beyond_robot_radius_m"] = min(
        sweep_margin(world, a[:2], b[:2]) for a, b in zip(poses, poses[1:]))
    row["nearest_interior_all_room_tour"] = tour
    row["initial_pose_clearance_beyond_robot_radius_m"] = sweep_margin(world, poses[0][:2], poses[0][:2])
    for door in row["doors"]:
        require(door["clear_width_m"] + 1e-9 >= REGISTERED["door_min_clear_width_m"], "Door width below registry.")
        candidates = []
        for r, c, heading in door["exact_crossing_states_rc_heading"]:
            dr, dc = prior.DIRECTIONS[heading]
            target = (r + dr, c + dc)
            require(truth["edge_valid"][(*target, (heading + 2) % 4)], "Door lacks reverse crossing.")
            candidates.append((sweep_margin(world, (r, c), target), (r, c, heading)))
        require(candidates, "A declared door has no admissible native crossing.")
        best_margin, best_pose = max(candidates)
        door["best_bidirectional_crossing_clearance_m"] = best_margin
        door["best_bidirectional_crossing_rc_heading"] = list(best_pose)
    require(world.step_count == 0, "Auditing geometry must not charge any actions.")


def references(layout, truth, initial):
    directory = HERE / "evaluator_references" / layout
    receipt = read(directory / "RECEIPT.json")
    rules = read(HERE / "reference_protocol.json")
    old_rules = read(ATTRIBUTION / "reference_protocol.json")
    require({k: v for k, v in rules.items() if k != "base_seed"} ==
            {k: v for k, v in old_rules.items() if k != "base_seed"},
            "A reference metric or sampling setting changed.")
    require(receipt["layout"] == layout and not receipt["actor_executed"] and not receipt["actor_trajectory_read"]
            and receipt["reference_source_is_native_geometry_not_template"], "Reference provenance is invalid.")
    for name, field in (("source_mesh.ply", "source_mesh_sha256"), ("fixed_reference.npz", "reference_npz_sha256")):
        require(sha(directory / name) == receipt[field], f"Reference SHA mismatch: {name}")
    for path, expected in receipt["source_inputs_sha256"].items():
        require(sha(Path(path)) == expected, f"Reference dependency changed: {path}")
    shape = tuple(initial["shape"])
    require(all(truth[k].shape == shape and truth[k].dtype == np.bool_
                for k in ("occupancy", "robot_valid", "reachable"))
            and truth["edge_valid"].shape == (*shape, 4) and truth["edge_valid"].dtype == np.bool_,
            "Reference occupancy, footprint and directed sweep shapes disagree.")
    require(not np.any(truth["reachable"] & (~truth["robot_valid"] | truth["occupancy"])),
            "Reference labels a reachable centre as physically unsafe.")
    require(receipt["reachable_cells"] == int(truth["reachable"].sum()) > 0,
            "Reachable evaluation denominator is invalid.")
    mesh = o3d.io.read_triangle_mesh(str(directory / "source_mesh.ply"))
    vertices, triangles = np.asarray(mesh.vertices), np.asarray(mesh.triangles)
    require(vertices.ndim == 2 and vertices.shape[1] == 3 and np.isfinite(vertices).all()
            and len(vertices) and triangles.ndim == 2 and triangles.shape[1] == 3
            and len(triangles) and triangles.min() >= 0 and triangles.max() < len(vertices),
            "Reference source mesh is empty or structurally invalid.")
    with np.load(directory / "fixed_reference.npz", allow_pickle=False) as fixed:
        cameras = fixed["camera_poses"]
        require(cameras.shape == (receipt["camera_pose_count"], 4, 4) and len(cameras)
                and np.isfinite(cameras).all(), "Reference camera lattice is structurally invalid.")
        counts = {}
        for key, count in (("all", rules["global_area_samples"]),
                           ("open31", rules["per_target_area_samples"]),
                           ("closed118", rules["per_target_area_samples"])):
            points, first = fixed[key], fixed[key + "_first_pose"]
            metadata = receipt["metadata"][key]
            require(metadata["drawn_samples"] == count and len(points) == metadata["visible_samples"]
                    and 0 < len(points) <= count and points.shape == (len(first), 3) and np.isfinite(points).all()
                    and first.min() >= 0 and first.max() < len(cameras) and metadata["area_per_sample_m2"] > 0,
                    f"Reference samples or ray-provenance indices are invalid: {key}.")
            counts[key] = {"drawn": count, "eligible_visible": len(points)}
    return {"vertices": len(vertices), "triangles": len(triangles),
            "camera_lattice_poses": receipt["camera_pose_count"], "area_sampling": counts,
            "mesh_sha256": receipt["source_mesh_sha256"], "reference_sha256": receipt["reference_npz_sha256"]}


def cold_view(layout, row, initial):
    directory = HERE / "prepared" / layout
    config = initial["actual_config"]
    expected = [read(ATTRIBUTION / "prepared" / name / "initial_state.json")["actual_config"]
                for name in ("attr26_corridor_crosspass", "attr26_loop_bypass")]
    require(all(config == previous for previous in expected), "Cold sensor or mapper configuration changed.")
    with np.load(directory / "initial_frame.npz", allow_pickle=False) as frame:
        expected_shape = (config["height_px"], config["width_px"])
        depth = frame["depth_m"]
        valid = np.isfinite(depth) & (depth > 0) & (depth <= config["max_depth_m"])
        require(depth.shape == expected_shape and frame["color_rgb"].shape == (*expected_shape, 3)
                and frame["semantic"].shape == expected_shape and float(frame["timestamp_s"]) == 0
                and valid.any(), "Cold camera input is structurally invalid.")
        view = {"initial_camera_fov_deg": config["fov_deg"],
                "depth_valid_pixels": int(valid.sum()), "depth_total_pixels": int(depth.size),
                "depth_10_50_90_percentile_m": np.quantile(depth[valid], [.1, .5, .9]).tolist(),
                "world_from_camera": frame["world_from_camera"].tolist(),
                "intrinsic": frame["intrinsic"].tolist()}
    with np.load(directory / "initial_scan.npz", allow_pickle=False) as scan:
        ranges = scan["ranges_m"]
        maximum = float(scan["range_max_m"])
        require(ranges.shape == (config["laser_rays"],) and np.isfinite(ranges).all()
                and float(scan["timestamp_s"]) == 0 and np.all((ranges >= 0) & (ranges <= maximum + 1e-6)),
                "Cold laser input is structurally invalid.")
        view.update({"laser_maximum_range_m": maximum,
                     "laser_returns_within_range": int((ranges < maximum - 1e-3).sum()),
                     "laser_10_50_90_percentile_m": np.quantile(ranges, [.1, .5, .9]).tolist(),
                     "laser_angle_min_rad": float(scan["angle_min_rad"]),
                     "laser_angle_increment_rad": float(scan["angle_increment_rad"])})
    view["true_reachable_cells_unknown_at_start"] = row["first_frame_known_unknown_reachable_cells"]
    return view


def main():
    verify_registration()
    seeds = old_seed_check()
    archived, counts = prior.known_archive_signatures()
    seen = {}
    accepted, rejected = [], []
    for layout, spec in SPECS.items():
        try:
            row = prior.check_geometry(layout, archived, counts, seen)
            initial = read(HERE / "prepared" / layout / "initial_state.json")
            with np.load(HERE / "evaluator_references" / layout / "evaluator_truth.npz", allow_pickle=False) as truth:
                row["references"] = references(layout, truth, initial)
                prove_clearance(layout, row, truth, spec)
            row["room_graph"] = room_graph(spec)
            row["cold_start_view"] = cold_view(layout, row, initial)
            row["gross_floor_area_m2"] = spec["width_m"] * spec["height_m"]
            row["geometric_admission"] = "accepted"
            accepted.append(row)
            print(layout, row["nearest_interior_all_room_tour"]["actions"], row["reachable_area_m2"], flush=True)
        except (ValueError, OSError, KeyError, RuntimeError) as error:
            rejected.append({"layout": layout, "geometric_admission": "excluded", "reason": str(error)})
            print(layout, "excluded", error, flush=True)
    record = {"status": "all_geometry_admitted" if not rejected else "registered_geometry_exclusions",
              "policy_calls": 0, "actor_executions": 0,
              "registry_sha256": sha(HERE / "LAYOUT_REGISTRATION.json"),
              "registration_receipt_sha256": sha(HERE / "REGISTRATION_RECEIPT.json"),
              "source_only_no_truth_leak": True, "archive_seed_records_checked": seeds,
              "archive_grids_checked": counts, "layout_read_order": list(LAYOUTS),
              "layouts": accepted, "excluded": rejected,
              "selection_conditions": REGISTERED["admission"], "exclusion_rule": REGISTERED["exclusion"]}
    with (HERE / "formal_geometry_audit.json").open("x") as stream:
        json.dump(record, stream, indent=2, allow_nan=False)
        stream.write("\n")
    if rejected:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
