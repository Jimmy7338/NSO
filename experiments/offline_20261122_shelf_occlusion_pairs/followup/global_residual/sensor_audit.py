#!/usr/bin/env python3
"""Separate held-out boundary visibility, acquired depth and archived TSDF matches."""

import argparse
import csv
import hashlib
import json
from pathlib import Path
import sys

from audit import ARMS, HERE, SOURCE, STUDY, Tensor, csv_data, load_module, np, read, sha, surface_bins


CAPTURES = ("start", "paid_96", "paid_144", "terminal")
FRAME_FIELDS = ("depth_m", "color_rgb", "semantic", "intrinsic", "world_from_camera")
RAW_FIELDS = ("depth_values_sha256", "rgb_values_sha256", "scan_ranges_sha256")
RANGE_MIN_M = 0.15
SOURCE_HIT_TOLERANCE_M = 0.001
DEPTH_THRESHOLDS_M = (0.025, 0.05)
MESH_THRESHOLD_M = 0.05


def array_sha(value):
    return hashlib.sha256(np.ascontiguousarray(value).tobytes()).hexdigest()


def frame_sha(frame):
    digest = hashlib.sha256()
    for field in FRAME_FIELDS:
        value = np.ascontiguousarray(getattr(frame, field))
        digest.update(str((value.dtype.str, value.shape)).encode())
        digest.update(value.tobytes())
    digest.update(repr(float(frame.timestamp_s)).encode())
    return digest.hexdigest()


def frame_support(frame, points, truth, range_max):
    """Return frustum, continuous first-hit and two acquired pixel support masks."""
    height, width = frame.depth_m.shape
    fx, fy = frame.intrinsic[0, 0], frame.intrinsic[1, 1]
    cx, cy = frame.intrinsic[0, 2], frame.intrinsic[1, 2]
    eye, camera_rotation = frame.world_from_camera[:3, 3], frame.world_from_camera[:3, :3]
    local = (points - eye) @ camera_rotation
    axial = local[:, 2]
    frustum = (axial >= RANGE_MIN_M) & (axial <= range_max)
    ids = np.flatnonzero(frustum)
    u = fx * local[ids, 0] / axial[ids] + cx
    v = fy * local[ids, 1] / axial[ids] + cy
    clipped = (u >= -0.5) & (u < width - 0.5) & (v >= -0.5) & (v < height - 0.5)
    frustum[ids[~clipped]] = False
    ids, u, v = ids[clipped], u[clipped], v[clipped]
    visible = np.zeros(len(points), dtype=bool)
    small = visible.copy()
    large = visible.copy()
    if not len(ids):
        return frustum, visible, small, large
    delta = points[ids] - eye
    rays = np.column_stack((np.broadcast_to(eye, delta.shape), delta)).astype(np.float32)
    hit = truth.cast_rays(Tensor(rays), nthreads=1)["t_hit"].numpy()
    exposed = np.isfinite(hit) & (np.abs(hit - 1.) * np.linalg.norm(delta, axis=1) <= SOURCE_HIT_TOLERANCE_M)
    ids, u, v = ids[exposed], u[exposed], v[exposed]
    visible[ids] = True
    pixel_x = np.floor(u + 0.5).astype(np.intp)
    pixel_y = np.floor(v + 0.5).astype(np.intp)
    nearest_distance_sq = np.full(len(ids), np.inf)
    for x_offset, y_offset in ((0, 0), (-1, 0), (1, 0), (0, -1), (0, 1),
                               (-1, -1), (-1, 1), (1, -1), (1, 1)):
        x, y = pixel_x + x_offset, pixel_y + y_offset
        valid = (x >= 0) & (x < width) & (y >= 0) & (y < height)
        chosen = np.flatnonzero(valid)
        depth = frame.depth_m[y[chosen], x[chosen]]
        acquired = np.isfinite(depth) & (depth > 0.) & (depth <= range_max)
        chosen, depth = chosen[acquired], depth[acquired]
        xyz = np.column_stack(((x[chosen] - cx) * depth / fx, (y[chosen] - cy) * depth / fy, depth))
        differences = xyz - local[ids[chosen]]
        nearest_distance_sq[chosen] = np.minimum(nearest_distance_sq[chosen], np.sum(differences**2, axis=1))
    small[ids] = nearest_distance_sq <= DEPTH_THRESHOLDS_M[0] ** 2
    large[ids] = nearest_distance_sq <= DEPTH_THRESHOLDS_M[1] ** 2
    return frustum, visible, small, large


def verify_cold(frame, scan, folder):
    for packet, name in ((frame, "initial_frame.npz"), (scan, "initial_scan.npz")):
        with np.load(folder/name, allow_pickle=False) as saved:
            if not all(np.array_equal(saved[key], np.asarray(getattr(packet, key))) for key in saved.files):
                raise ValueError(f"Cold packet fields differ from sealed samples: {folder / name}")


def native_match(run, manifest, label, paid, weighted, metric, points):
    receipt, files, _, receipt_sha = weighted.engine.snapshot(run, label, manifest, "native_observed")
    if receipt["status"] != "captured" or receipt["actual_paid"] != paid:
        raise ValueError(f"Incorrect measured mesh clock: {run}/{label}")
    mesh = weighted.files.read_mesh(files["native_mesh.ply"])
    ray = metric.ray_scene(mesh)
    distance = ray.compute_distance(Tensor(points), nthreads=1).numpy()
    return distance <= MESH_THRESHOLD_M, dict(actual_paid=paid, checkpoint_sha256=receipt_sha,
                                             native_mesh_sha256=receipt["files_sha256"]["native_mesh.ply"])


def passive_trace(layout, arm, entry, definition, cold, points, source_ray, cache, weighted, metric):
    run = Path(entry["run_directory"])
    manifest, manifest_sha = weighted.engine.verify_manifest(run, "manifest.json", entry["source_manifest_sha256"])
    trace = read(run / "steps.json")
    initial = read(cold / "initial_state.json")
    weighted.native.verify_trace(trace, initial, 192, manifest)
    paid_end = len(trace)
    if paid_end != entry["actual_paid"] or not manifest["returned"]:
        raise ValueError(f"Run is not a bound completed actual return: {run}")
    cold_record = read(run / "initial_raw_rgbd_scan_sha256.json")
    world = definition.make_world(layout)
    if world.config.depth_sigma_m or world.config.dropout or world.seed != initial["seed"]:
        raise ValueError(f"Pose cache requires the measured no-noise scene contract: {layout}")
    frame, scan = world.sense(), world.scan()
    verify_cold(frame, scan, cold)
    clocks = {"start": 0, "paid_96": 96, "paid_144": 144, "terminal": paid_end}
    support = np.zeros((4, len(points)), dtype=bool)
    last_depth_support = np.full(len(points), -1, dtype=np.int16)
    result = {}
    packet_digest = hashlib.sha256()
    snapshots = {}
    for paid, row in enumerate((cold_record, *trace)):
        if paid:
            frame, collision, _ = world.step(row["action"])
            if (world.step_count, list(world.position), world.heading, collision) != (
                row["step"], row["position"], row["heading"], row["collision"]
            ):
                raise ValueError(f"Passive motion or clock drift: {run}, action {paid}")
            scan = world.scan()
        values = [array_sha(value) for value in (frame.depth_m, frame.color_rgb, scan.ranges_m)]
        full_frame = frame_sha(frame)
        if values != [row[key] for key in RAW_FIELDS] or (paid and full_frame != row["frame_sha256"]):
            raise ValueError(f"Archived sensor checksum mismatch: {run}, action {paid}")
        packet_digest.update(bytes.fromhex(full_frame + "".join(values)))
        pose = (*world.position, world.heading)
        if pose not in cache:
            cache[pose] = np.stack(frame_support(frame, points, source_ray, world.config.max_depth_m))
        current = cache[pose]
        support |= current
        last_depth_support[current[-1]] = paid
        for label, at in clocks.items():
            if paid == at:
                matched, proof = native_match(run, manifest, label, paid, weighted, metric, points)
                snapshots[label] = proof
                result[label] = dict(paid=paid, matched=matched, frustum=support[0].copy(),
                                     visible=support[1].copy(), depth_025=support[2].copy(),
                                     depth_050=support[3].copy(), last_depth=last_depth_support.copy())
    return result, dict(manifest_sha256=manifest_sha, steps_sha256=manifest["files_sha256"]["steps.json"],
                        cold_sensor_receipt_sha256=sha(run/"initial_raw_rgbd_scan_sha256.json"),
                        ordered_verified_sensor_digest_sha256=packet_digest.hexdigest(),
                        verified_raw_sensor_clocks=paid_end + 1, verified_full_paid_frame_hashes=paid_end,
                        snapshots=snapshots)


def counts(mask):
    return int(np.count_nonzero(mask))


def deficit_partition(group, state):
    frustum, visible, depth = (state[key] for key in ("frustum", "visible", "depth_050"))
    parts = {
        "never_in_frustum": counts(group & ~frustum),
        "frustum_without_continuous_visibility": counts(group & frustum & ~visible),
        "visible_without_acquired_depth_05cm": counts(group & visible & ~depth),
        "acquired_depth_but_no_mesh_match": counts(group & depth),
    }
    if sum(parts.values()) != counts(group) or np.any(depth & ~visible) or np.any(visible & ~frustum):
        raise ValueError("Visibility/support hierarchy does not conserve unmatched reference samples.")
    return parts


def output_summary(layout, arms, weight, expected):
    stages, pairs, changes = [], [], []
    for arm, observed in arms.items():
        before = None
        prior_mesh = np.zeros_like(observed["start"]["matched"])
        for clock, state in observed.items():
            matched, depth = state["matched"], state["depth_050"]
            missing = ~matched
            stages.append(dict(layout=layout, arm=arm, checkpoint=clock, actual_paid=state["paid"],
                               boundary_vertical_reference_samples=len(matched), frustum_samples=counts(state["frustum"]),
                               continuously_visible_samples=counts(state["visible"]),
                               depth_supported_025m_samples=counts(state["depth_025"]),
                               depth_supported_050m_samples=counts(depth), archived_mesh_matches=counts(matched),
                               mesh_matches_without_acquired_depth_050m=counts(matched & ~depth),
                               depth_supported_and_mesh_matched=counts(depth & matched),
                               all_unmatched=counts(missing), **deficit_partition(missing, state),
                               depth_supported_missing_after_prior_archived_match=counts(depth & missing & prior_mesh)))
            if clock != "start":
                row = expected[layout, clock]
                prefix = "combined" if arm == ARMS[0] else "strict"
                if counts(matched) != int(row["both_samples"]) + int(row[f"{prefix}_only_samples"]):
                    raise ValueError("Captured boundary mesh support differs from the independent full-scene audit.")
            if before is not None:
                prior_clock, earlier = before
                lost = earlier["matched"] & missing
                gained = matched & ~earlier["matched"]
                changes.append(dict(layout=layout, arm=arm, before=prior_clock, after=clock,
                                    paid_before=earlier["paid"], paid_after=state["paid"],
                                    gained_mesh_matches=counts(gained), lost_mesh_matches=counts(lost),
                                    lost_mesh_matches_with_acquired_depth_before_start=counts(lost & earlier["depth_050"]),
                                    lost_mesh_matches_with_acquired_depth_after_start=counts(lost & (state["last_depth"] > earlier["paid"])),
                                    missing_at_end_after_any_previous_archived_match=counts(prior_mesh & missing),
                                    missing_at_end_after_previous_archived_match_with_acquired_depth=counts(prior_mesh & missing & depth),
                                    area_per_reference_sample_m2=weight))
            prior_mesh |= matched
            before = clock, state
    for clock in CAPTURES:
        for reference, other in (ARMS, ARMS[::-1]):
            a, b = arms[reference][clock], arms[other][clock]
            exclusive = a["matched"] & ~b["matched"]
            prior_any = np.zeros_like(exclusive)
            for prev in CAPTURES[:CAPTURES.index(clock)]:
                prior_any |= arms[other][prev]["matched"]
            pairs.append(dict(layout=layout, checkpoint=clock, represented_by=reference, missing_in=other,
                              represented_paid=a["paid"], missing_paid=b["paid"], equal_clock=a["paid"] == b["paid"],
                              exclusive_sample_count=counts(exclusive), area_per_reference_sample_m2=weight,
                              **deficit_partition(exclusive, b),
                              acquired_depth_at_025m_but_no_mesh_match=counts(exclusive & b["depth_025"]),
                              formerly_mesh_matched_in_missing_arm=counts(exclusive & prior_any),
                              formerly_mesh_matched_and_acquired_depth_in_missing_arm=counts(exclusive & prior_any & b["depth_050"])))
    return stages, pairs, changes


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="Verify recorded outputs without writing.")
    args = parser.parse_args()
    sys.path.insert(0, str(STUDY / "measurement"))
    from historical import checks

    check, weighted, metric = checks()
    scenes = SOURCE / "scenes"
    weighted.SCENES = scenes
    source_scene = scenes / "scene.py"
    geometry = load_module("nso_frozen_boundary_sensor_scene", source_scene)
    previous_receipt = read(HERE/"receipt.json")
    if sha(source_scene) != previous_receipt["source_geometry_sha256"]:
        raise ValueError("Room definitions differ from the earlier residual audit.")
    previous_boundary = HERE/"boundary_surface.csv"
    if sha(previous_boundary) != previous_receipt["output_sha256"][previous_boundary.name]:
        raise ValueError("Earlier boundary recall table has changed.")
    with previous_boundary.open() as file:
        expected = {(row["layout"], row["checkpoint"]): row for row in csv.DictReader(file)
                    if row["source_face_orientation"] == "vertical"}
    formal = SOURCE/"results/formal/summary.json"
    if sha(formal) != previous_receipt["source_summary_sha256"]:
        raise ValueError("Formal measurement summary has changed.")
    entries = {(row["layout"], row["arm"]): row for row in read(formal)["episodes"]}
    receipt = dict(source_residual_receipt_sha256=sha(HERE/"receipt.json"), source_scene_sha256=sha(source_scene),
                   audit_source_sha256=sha(Path(__file__)), sealed_source_geometry_name="source_mesh.ply",
                   horizontal_raster_convention="columns or rows from -0.5 inclusive to size-0.5 exclusive",
                   source_hit_tolerance_m=SOURCE_HIT_TOLERANCE_M, depth_neighbor_radius_pixels=1,
                   acquired_depth_thresholds_m=DEPTH_THRESHOLDS_M, native_mesh_threshold_m=MESH_THRESHOLD_M,
                   actor_runs=0, hardware_io=False, layouts={})
    all_stages, all_pairs, all_changes = [], [], []
    for layout, layout_config in geometry.HOLDOUT.items():
        files = previous_receipt["layouts"][layout]["reference_files_sha256"]
        if any(sha(file) != copied for file, copied in files.items()):
            raise ValueError(f"Fixed source reference has changed: {layout}")
        cold = scenes/"prepared"/layout
        reference = weighted.load_reference(layout, read(cold/"initial_state.json"))
        points = reference["samples"]["scene"]
        source_ray = metric.ray_scene(reference["source_mesh"])
        normal_z = np.abs(source_ray.compute_closest_points(Tensor(points), nthreads=1)["primitive_normals"].numpy()[:, 2])
        fixed = (surface_bins(points[:, :2], layout_config["rooms"], previous_receipt["room_boundary_margin_m"]) == -1) & (normal_z <= 1e-3)
        points = np.ascontiguousarray(points[fixed])
        if len(points) != int(expected[layout, "paid_144"]["fixed_reference_samples"]):
            raise ValueError("Boundary source partition differs from the earlier room audit.")
        weight = reference["weights"]["scene"]
        cache, arms, runs = {}, {}, {}
        for arm in ARMS:
            arms[arm], runs[arm] = passive_trace(layout, arm, entries[layout, arm], geometry, cold, points,
                                                source_ray, cache, weighted, metric)
        receipt["layouts"][layout] = dict(fixed_boundary_vertical_xyz_sha256=array_sha(points),
                                          source_files_sha256=files, fixed_samples=len(points),
                                          fixed_area_m2=len(points) * weight,
                                          unique_oriented_camera_states=len(cache), arms=runs)
        stage_rows, pair_rows, changes = output_summary(layout, arms, weight, expected)
        all_stages.extend(stage_rows)
        all_pairs.extend(pair_rows)
        all_changes.extend(changes)
    products = {"sensor_boundary_clocks.csv": csv_data(all_stages),
                "sensor_boundary_exclusive.csv": csv_data(all_pairs),
                "sensor_boundary_mesh_transitions.csv": csv_data(all_changes)}
    receipt["outputs_sha256"] = {name: hashlib.sha256(body.encode()).hexdigest() for name, body in products.items()}
    products["sensor_receipt.json"] = json.dumps(receipt, indent=2, allow_nan=False) + "\n"
    for name, body in products.items():
        path = HERE / name
        if args.check:
            if path.read_bytes() != body.encode():
                raise ValueError(f"Recorded sensor audit has changed: {path}")
        else:
            path.write_bytes(body.encode())
    packets = sum(arm["verified_raw_sensor_clocks"] for layout in receipt["layouts"].values() for arm in layout["arms"].values())
    print(f"Verified {packets} sensor packets, {len(all_stages)} mesh captures and {len(all_pairs)} exclusive comparisons.")


if __name__ == "__main__":
    main()
