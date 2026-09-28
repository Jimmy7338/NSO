"""Continuous-pose CPU sensor adapter, with a mandatory persistent-space gate.

Analytical helpers accept explicit small arrays and create no study World.
Every actual DEV asset constructor checks resources before opening any asset.
Only PaidRGBDObservationV40 plus a label-free PlanarScan leave the renderer.
"""
from copy import deepcopy
from dataclasses import dataclass
import hashlib
import math
from pathlib import Path
import shutil

import numpy as np

from nso.instance_belief_v40 import PaidRGBDObservationV40
from nso.scene_contract_v40 import load_json_strict, validate_public_spec
from utils.rgbd_contract import PlanarScan


GIB = 1024 ** 3
_COUNTERS = {"worlds_created": 0, "rgbd_frames": 0, "scans": 0,
             "paid_actions": 0, "blocked_before_world_creation": 0}


def runtime_counts_v41():
    return dict(_COUNTERS)


def _positive(value, name, zero=False):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, float, np.integer, np.floating)):
        raise ValueError(name + " requires a finite number")
    if not math.isfinite(value) or value < 0 or (not zero and value == 0):
        raise ValueError(name + " requires a valid positive/nonnegative number")
    return float(value)


def _integer(value, name, zero=False):
    if type(value) is not int or value < (0 if zero else 1):
        raise ValueError(name + " requires an integer")
    return value


def _readonly(value, dtype=None):
    result = np.array(value, dtype=dtype, copy=True)
    result.flags.writeable = False
    return result


def wrap_yaw(yaw):
    if not math.isfinite(yaw):
        raise ValueError("finite yaw required")
    return float((yaw + math.pi) % (2 * math.pi) - math.pi)


def _pose(pose):
    result = np.asarray(pose, dtype=float)
    if result.shape != (3,) or not np.isfinite(result).all():
        raise ValueError("finite x,y,yaw_rad pose required")
    return np.asarray([result[0], result[1], wrap_yaw(result[2])])


def camera_transform_xyyaw(pose_xyyaw_rad, height_m=.9):
    """yaw=0 faces world +X; optical x right, y down, z forward."""
    x, y, yaw = _pose(pose_xyyaw_rad)
    height = _positive(height_m, "camera height")
    forward = np.asarray([math.cos(yaw), math.sin(yaw), 0.])
    transform = np.eye(4)
    transform[:3, :3] = np.column_stack((np.cross(forward, [0., 0., 1.]), [0., 0., -1.], forward))
    transform[:3, 3] = [x, y, height]
    return transform


def propose_motion(pose_xyyaw_rad, action, translation_step_m=.25, rotation_step_deg=30.):
    pose = _pose(pose_xyyaw_rad)
    step = _positive(translation_step_m, "translation step")
    turn = math.radians(_positive(rotation_step_deg, "rotation step"))
    if turn > math.pi:
        raise ValueError("turn must be <= 180 degrees")
    if action == "forward":
        pose[:2] += step * np.asarray([math.cos(pose[2]), math.sin(pose[2])])
    elif action == "turn_left":
        pose[2] = wrap_yaw(pose[2] + turn)
    elif action == "turn_right":
        pose[2] = wrap_yaw(pose[2] - turn)
    elif action != "observe":
        raise ValueError("paid action must be forward/turn_left/turn_right/observe")
    return pose


def return_pose_matches(pose, start_pose, *, position_tolerance_m=1e-8, yaw_tolerance_rad=1e-8):
    pose, start = _pose(pose), _pose(start_pose)
    position_tolerance_m = _positive(position_tolerance_m, "position tolerance", zero=True)
    yaw_tolerance_rad = _positive(yaw_tolerance_rad, "yaw tolerance", zero=True)
    return bool(np.linalg.norm(pose[:2] - start[:2]) <= position_tolerance_m
                and abs(wrap_yaw(pose[2] - start[2])) <= yaw_tolerance_rad)


def _point_segment_distance(point, start, end):
    delta = end - start
    length = float(delta @ delta)
    if length == 0:
        return float(np.linalg.norm(point - start))
    fraction = float(np.clip((point - start) @ delta / length, 0., 1.))
    return float(np.linalg.norm(point - (start + fraction * delta)))


def _segment_distance(a, b, c, d):
    cross = lambda u, v: float(u[0]*v[1] - u[1]*v[0])
    ab, cd = b-a, d-c
    determinant = cross(ab, cd)
    if abs(determinant) > 1e-12:
        t, u = cross(c-a, cd)/determinant, cross(c-a, ab)/determinant
        if -1e-12 <= t <= 1+1e-12 and -1e-12 <= u <= 1+1e-12:
            return 0.
    return min(_point_segment_distance(a, c, d), _point_segment_distance(b, c, d),
               _point_segment_distance(c, a, b), _point_segment_distance(d, a, b))


def _convex_polygon(polygon):
    polygon = np.asarray(polygon, dtype=float)
    if polygon.ndim != 2 or polygon.shape[1] != 2 or len(polygon) < 3 or not np.isfinite(polygon).all():
        raise ValueError("finite convex polygon with at least three XY vertices required")
    if np.array_equal(polygon[0], polygon[-1]):
        polygon = polygon[:-1]
    edges = np.roll(polygon, -1, axis=0)-polygon
    if len(polygon) < 3 or np.any(np.linalg.norm(edges, axis=1) < 1e-12):
        raise ValueError("polygon edges must be nondegenerate")
    cross = edges[:, 0]*np.roll(edges[:, 1], -1)-edges[:, 1]*np.roll(edges[:, 0], -1)
    if not (np.all(cross >= -1e-10) or np.all(cross <= 1e-10)) or np.max(abs(cross)) < 1e-10:
        raise ValueError("convex nonzero-area footprint required")
    return polygon


def _inside_convex(point, polygon):
    edges = np.roll(polygon, -1, axis=0)-polygon
    delta = point-polygon
    cross = edges[:, 0]*delta[:, 1]-edges[:, 1]*delta[:, 0]
    return bool(np.all(cross >= -1e-12) or np.all(cross <= 1e-12))


def swept_circle_collision(start_xy, end_xy, obstacle_polygons, radius_m=.2):
    """Continuous capsule-vs-convex-footprint test for any segment direction.

    Tangency counts as collision. P1 facility AABBs are conservative physical
    footprints; no axis-aligned movement restriction or endpoint-only test.
    """
    a, b = np.asarray(start_xy, dtype=float), np.asarray(end_xy, dtype=float)
    if a.shape != (2,) or b.shape != (2,) or not np.isfinite([a, b]).all():
        raise ValueError("finite XY segment required")
    radius = _positive(radius_m, "robot radius", zero=True)
    for raw in obstacle_polygons:
        polygon = _convex_polygon(raw)
        if _inside_convex(a, polygon) or _inside_convex(b, polygon):
            return True
        for i in range(len(polygon)):
            if _segment_distance(a, b, polygon[i], polygon[(i+1) % len(polygon)]) <= radius + 1e-12:
                return True
    return False


def paid_motion_transition(pose, action, *, paid_step, max_actions, start_pose,
                            obstacle_polygons, radius_m=.2,
                            translation_step_m=.25, rotation_step_deg=30.):
    """Pure accounting helper; no World, observation, asset or implicit return."""
    _integer(paid_step, "paid step", zero=True); _integer(max_actions, "maximum actions")
    if paid_step >= max_actions:
        raise ValueError("paid action budget exhausted")
    before = _pose(pose)
    proposed = propose_motion(before, action, translation_step_m, rotation_step_deg)
    collision = action == "forward" and swept_circle_collision(before[:2], proposed[:2], obstacle_polygons, radius_m)
    after = before if collision else proposed
    return {"action": action, "paid_step": paid_step+1, "action_cost": 1,
            "collision": bool(collision), "pose_before_xyyaw_rad": tuple(map(float, before)),
            "pose_xyyaw_rad": tuple(map(float, after)),
            "returned_xy_and_yaw": return_pose_matches(after, start_pose)}


def first_hit_triangles(origins, directions, vertices, triangles, *, max_distance_m,
                        min_distance_m=1e-8, ray_chunk=128, triangle_chunk=256):
    """Private two-sided Moller-Trumbore first hit, radial metres.

    Directions must be unit length. Misses return (inf,-1). Equal-distance
    ties retain the first indexed triangle. Bounded tiles avoid an NxM tensor.
    """
    vertices = np.asarray(vertices, dtype=float)
    triangles = np.asarray(triangles)
    directions = np.asarray(directions, dtype=float)
    origins = np.asarray(origins, dtype=float)
    if vertices.ndim != 2 or vertices.shape[1] != 3 or not np.isfinite(vertices).all():
        raise ValueError("finite Nx3 vertices required")
    if triangles.ndim != 2 or triangles.shape[1] != 3 or triangles.dtype.kind not in "iu" or not len(triangles):
        raise ValueError("nonempty integer Mx3 triangles required")
    if triangles.min() < 0 or triangles.max() >= len(vertices):
        raise ValueError("triangle index out of bounds")
    if directions.ndim != 2 or directions.shape[1] != 3 or not np.isfinite(directions).all() or not np.allclose(np.linalg.norm(directions, axis=1), 1., rtol=0, atol=1e-8):
        raise ValueError("unit Nx3 ray directions required")
    if origins.shape == (3,):
        origins = np.broadcast_to(origins, directions.shape)
    if origins.shape != directions.shape or not np.isfinite(origins).all():
        raise ValueError("one origin or one finite origin per ray required")
    maximum = _positive(max_distance_m, "maximum range")
    minimum = _positive(min_distance_m, "minimum ray distance", zero=True)
    if minimum > maximum:
        raise ValueError("ray interval must be increasing")
    _integer(ray_chunk, "ray chunk"); _integer(triangle_chunk, "triangle chunk")
    p = vertices[triangles]
    e1, e2 = p[:, 1]-p[:, 0], p[:, 2]-p[:, 0]
    distances, identities = np.full(len(directions), np.inf), np.full(len(directions), -1, np.int32)
    for start in range(0, len(directions), ray_chunk):
        stop = min(start+ray_chunk, len(directions))
        o, d = origins[start:stop], directions[start:stop]
        best, best_ids = distances[start:stop], identities[start:stop]
        for first in range(0, len(triangles), triangle_chunk):
            last = min(first+triangle_chunk, len(triangles))
            pvec = np.cross(d[:, None, :], e2[None, first:last, :])
            determinant = np.einsum("rti,ti->rt", pvec, e1[first:last])
            valid = abs(determinant) > 1e-12
            inverse = np.divide(1., determinant, out=np.zeros_like(determinant), where=valid)
            delta = o[:, None, :]-p[None, first:last, 0, :]
            u = np.einsum("rti,rti->rt", delta, pvec)*inverse
            qvec = np.cross(delta, e1[None, first:last, :])
            v = np.einsum("ri,rti->rt", d, qvec)*inverse
            distance = np.einsum("ti,rti->rt", e2[first:last], qvec)*inverse
            valid &= (u >= -1e-10) & (v >= -1e-10) & (u+v <= 1+1e-10) & (distance >= minimum) & (distance <= maximum)
            candidates = np.where(valid, distance, np.inf)
            local = np.argmin(candidates, axis=1)
            candidate = candidates[np.arange(len(d)), local]
            improve = candidate < best
            best[improve], best_ids[improve] = candidate[improve], first+local[improve]
    return distances, identities


def _camera_calibration(intrinsic, world_from_camera, width, height):
    _integer(width, "image width"); _integer(height, "image height")
    k, t = np.asarray(intrinsic, dtype=float), np.asarray(world_from_camera, dtype=float)
    if k.shape != (3, 3) or not np.isfinite(k).all() or k[0, 0] <= 0 or k[1, 1] <= 0 or not np.array_equal(k[2], [0., 0., 1.]) or k[0, 1] != 0 or k[1, 0] != 0:
        raise ValueError("finite zero-skew pinhole intrinsic required")
    if t.shape != (4, 4) or not np.isfinite(t).all() or not np.allclose(t[3], [0., 0., 0., 1.], rtol=0, atol=1e-10):
        raise ValueError("finite homogeneous camera pose required")
    r = t[:3, :3]
    if not np.allclose(r.T@r, np.eye(3), rtol=0, atol=1e-8) or not math.isclose(float(np.linalg.det(r)), 1., abs_tol=1e-8):
        raise ValueError("proper rigid camera transform required")
    return k, t


def apply_marker_decals(points, hit_triangles, triangle_instance_id, base_rgb,
                         marker_patches, camera_origin, plane_tolerance_m=1e-6):
    """Private first-hit appearance; actual instance IDs never leave renderer."""
    result = np.array(base_rgb, dtype=np.uint8, copy=True)
    owners = np.asarray(triangle_instance_id)
    valid = hit_triangles >= 0
    hit_owners = np.full(len(hit_triangles), -2, dtype=int)
    hit_owners[valid] = owners[hit_triangles[valid]]
    for patch in marker_patches:
        center = np.asarray(patch["center_world_m"], dtype=float)
        normal, u, v = [np.asarray(patch[key], dtype=float) for key in ("normal_world", "u_world", "v_world")]
        frame = np.column_stack((u, v, normal))
        if center.shape != (3,) or frame.shape != (3, 3) or not np.isfinite(frame).all() or not np.isfinite(center).all() or not np.allclose(frame.T@frame, np.eye(3), rtol=0, atol=1e-8):
            raise ValueError("finite orthonormal marker plane required")
        width, height = _positive(patch["width_m"], "marker width"), _positive(patch["height_m"], "marker height")
        color = np.asarray(patch["rgb"])
        if color.shape != (3,) or color.dtype.kind not in "iu" or np.any(color < 0) or np.any(color > 255):
            raise ValueError("integer RGB marker color required")
        delta = points-center
        select = valid & (hit_owners == patch["instance_id"]) & (abs(delta@normal) <= plane_tolerance_m)
        select &= (abs(delta@u) <= width/2) & (abs(delta@v) <= height/2)
        select &= ((np.asarray(camera_origin)-points)@normal) > 1e-10
        result[select] = color
    return result


def add_relative_depth_noise(clean_depth, *, relative_sigma, noise_seed, step,
                             minimum_depth_m=.1, maximum_depth_m=4.):
    clean = np.asarray(clean_depth, dtype=float)
    if clean.ndim != 2 or not np.isfinite(clean).all() or np.any(clean < 0):
        raise ValueError("finite nonnegative clean axial depth image required")
    sigma = _positive(relative_sigma, "relative depth noise", zero=True)
    _integer(noise_seed, "independent depth seed", zero=True); _integer(step, "paid step", zero=True)
    low, high = _positive(minimum_depth_m, "depth minimum"), _positive(maximum_depth_m, "depth maximum")
    if low >= high:
        raise ValueError("depth interval must be increasing")
    valid = (clean >= low) & (clean <= high)
    generator = np.random.default_rng(np.random.SeedSequence([noise_seed, step, 0x563431]))
    depth = clean*(1+sigma*generator.standard_normal(clean.shape))
    valid &= (depth >= low) & (depth <= high)
    return np.where(valid, depth, 0.).astype(np.float32)


def render_rgbd_arrays(vertices, triangles, *, intrinsic, world_from_camera, width, height,
                       triangle_rgb=None, triangle_instance_id=None, marker_patches=(),
                       minimum_depth_m=.1, maximum_depth_m=4., relative_sigma=0., noise_seed=0, step=0):
    """Return only (uint8 RGB, float32 axial depth), with optical-z clipping."""
    k, transform = _camera_calibration(intrinsic, world_from_camera, width, height)
    low, high = _positive(minimum_depth_m, "depth minimum"), _positive(maximum_depth_m, "depth maximum")
    if low >= high:
        raise ValueError("depth interval must be increasing")
    yy, xx = np.mgrid[:height, :width]
    local = np.column_stack(((xx.ravel()-k[0, 2])/k[0, 0], (yy.ravel()-k[1, 2])/k[1, 1], np.ones(width*height)))
    ray_norm = np.linalg.norm(local, axis=1)
    directions = (local/ray_norm[:, None])@transform[:3, :3].T
    distance, indices = first_hit_triangles(transform[:3, 3], directions, vertices, triangles,
                                           max_distance_m=high*float(ray_norm.max()))
    axial = distance/ray_norm
    valid = np.isfinite(axial) & (axial >= low) & (axial <= high)
    clean = np.where(valid, axial, 0.)
    palette = np.full((len(triangles), 3), 165, np.uint8) if triangle_rgb is None else np.asarray(triangle_rgb)
    if palette.dtype != np.uint8 or palette.shape != (len(triangles), 3):
        raise ValueError("aligned uint8 triangle RGB required")
    rgb = np.zeros((width*height, 3), np.uint8)
    rgb[valid] = palette[indices[valid]]
    if marker_patches:
        owners = np.asarray(triangle_instance_id)
        if owners.shape != (len(triangles),) or owners.dtype.kind not in "iu":
            raise ValueError("private triangle instance identity required for physical decals")
        points = transform[:3, 3] + np.where(np.isfinite(distance), distance, 0.)[:, None]*directions
        rgb = apply_marker_decals(points, np.where(valid, indices, -1), owners, rgb,
                                  marker_patches, transform[:3, 3])
    depth = add_relative_depth_noise(clean.reshape(height, width), relative_sigma=relative_sigma,
        noise_seed=noise_seed, step=step, minimum_depth_m=low, maximum_depth_m=high)
    return rgb.reshape(height, width, 3), depth


def render_planar_scan(vertices, triangles, pose_xyyaw_rad, *, maximum_range_m=8.,
                        laser_height_m=.3, number_of_rays=360, timestamp_s=0.):
    pose = _pose(pose_xyyaw_rad)
    maximum = _positive(maximum_range_m, "laser maximum")
    laser_height = _positive(laser_height_m, "laser height")
    _integer(number_of_rays, "scan rays")
    _positive(timestamp_s, "timestamp", zero=True)
    angle_min, increment = -math.pi, 2*math.pi/number_of_rays
    angles = angle_min + np.arange(number_of_rays)*increment + pose[2]
    directions = np.column_stack((np.cos(angles), np.sin(angles), np.zeros(number_of_rays)))
    origin = np.asarray([pose[0], pose[1], laser_height])
    distance, _ = first_hit_triangles(origin, directions, vertices, triangles, max_distance_m=maximum)
    ranges = _readonly(np.where(np.isfinite(distance), distance, maximum), np.float32)
    c, s = math.cos(pose[2]), math.sin(pose[2])
    transform = np.asarray([[c, -s, 0., origin[0]], [s, c, 0., origin[1]],
                            [0., 0., 1., origin[2]], [0., 0., 0., 1.]])
    return PlanarScan(float(timestamp_s), ranges, angle_min, increment, maximum, _readonly(transform))


def _filesystem_type(path):
    target = Path(path).resolve()
    best, kind = -1, None
    for line in Path("/proc/self/mountinfo").read_text().splitlines():
        before, after = line.split(" - ", 1)
        mount = before.split()[4]
        for code, value in (("\\040", " "), ("\\011", "\t"), ("\\012", "\n"), ("\\134", "\\")):
            mount = mount.replace(code, value)
        location = Path(mount)
        if target.is_relative_to(location) and len(str(location)) > best:
            best, kind = len(str(location)), after.split()[0]
    return kind


def storage_report_v41(persistent_output_root, expected_batch_peak_bytes):
    _integer(expected_batch_peak_bytes, "batch peak bytes", zero=True)
    requested = Path(persistent_output_root).resolve()
    existing = requested
    while not existing.exists():
        existing = existing.parent
    kind = _filesystem_type(existing)
    free = shutil.disk_usage(existing).free
    required = max(10*GIB, 2*expected_batch_peak_bytes + 2*GIB)
    persistent = kind is not None and kind not in ("tmpfs", "ramfs", "devtmpfs")
    passed = persistent and free >= required
    return {"status": "resources_ready" if passed else "blocked_before_world_creation",
            "persistent_output_root": str(requested), "checked_filesystem_path": str(existing),
            "filesystem_type": kind, "non_ram_filesystem": persistent,
            "free_bytes": free, "required_free_bytes": required,
            "expected_batch_peak_bytes": expected_batch_peak_bytes, "passed": passed,
            "reason": None if passed else ("nonpersistent_or_unknown_filesystem" if not persistent else "insufficient_persistent_space")}


class ResourceGateBlocked(RuntimeError):
    status = "blocked_before_world_creation"

    def __init__(self, report):
        self.report = deepcopy(report)
        super().__init__(f"{self.status}: {report['reason']}; free={report['free_bytes']} required={report['required_free_bytes']}")


@dataclass(frozen=True)
class SensorStepV41:
    rgbd: PaidRGBDObservationV40
    scan: PlanarScan
    receipt: dict


class DevelopmentSensorV41:
    """Actual DEV adapter; construction is never a resource-free test fixture."""
    def __init__(self, asset_dir, public_spec, *, episode_id, noise_seed,
                 persistent_output_root, expected_batch_peak_bytes,
                 robot_radius_m=.2, laser_height_m=.3, scan_rays=360):
        report = storage_report_v41(persistent_output_root, expected_batch_peak_bytes)
        if not report["passed"]:
            _COUNTERS["blocked_before_world_creation"] += 1
            raise ResourceGateBlocked(report)
        validate_public_spec(public_spec)
        sensor, motion = public_spec["sensor"], public_spec["motion"]
        if (motion["translation_step_m"] != .25 or motion["rotation_step_deg"] != 30.
                or sensor["width"] != 96 or sensor["height"] != 72
                or sensor["intrinsic"] != [[48., 0., 47.5], [0., 48., 35.5], [0., 0., 1.]]
                or sensor["depth_min_m"] != .1 or sensor["depth_max_m"] != 4.
                or sensor["lidar_max_range_m"] != 8.):
            raise ValueError("V41 requires the declared .25m/30deg and 96x72 axial-.1..4m/8m sensor contract")
        if public_spec["pose_model"]["kind"] != "exact":
            raise ValueError("V41 adapter currently supports exact simulated odometry only")
        if not isinstance(episode_id, str) or not episode_id or len(episode_id) > 100:
            raise ValueError("nonempty bounded opaque episode ID required")
        _integer(noise_seed, "depth noise seed", zero=True)
        self._radius = _positive(robot_radius_m, "robot radius")
        self._laser_height = _positive(laser_height_m, "laser height")
        self._scan_rays = _integer(scan_rays, "scan ray count")
        asset_dir = Path(asset_dir).resolve()
        if asset_dir.name not in {f"DEV_{family}_00" for family in "ABCDEF"}:
            raise ValueError("only registered P1 development assets are supported")
        manifest = load_json_strict(asset_dir.parent/"manifest.json")
        names = ("renderer_private/geometry.npz", "renderer_private/markers.json", "evaluation_private/instances.json")
        for name in names:
            relative = asset_dir.name + "/" + name
            if hashlib.sha256((asset_dir/name).read_bytes()).hexdigest() != manifest["artifact_sha256"].get(relative):
                raise ValueError("P1 asset hash mismatch")
        from nso.development_geometry_v40 import load_mesh_arrays
        arrays = load_mesh_arrays(asset_dir/names[0])
        metadata = load_json_strict(asset_dir/names[2])
        self._markers = load_json_strict(asset_dir/names[1])["marker_patches"]
        self._vertices, self._triangles = arrays["vertices"], arrays["triangles"]
        self._owners, self._rgb = arrays["triangle_instance_id"], arrays["triangle_rgb"]
        self._footprints = [item["conservative_footprint_xy_m"] for item in metadata["private_instances"]]
        for box in metadata["background_boxes"]:
            if box[5] > 0:
                self._footprints.append([[box[0], box[2]], [box[1], box[2]], [box[1], box[3]], [box[0], box[3]]])
        workspace = metadata["public_workspace"]
        start = workspace["start_position_world_m"]
        self._pose = _pose([start[0], start[1], math.radians(workspace["start_yaw_deg"])])
        self._start = self._pose.copy()
        if swept_circle_collision(self._pose[:2], self._pose[:2], self._footprints, self._radius):
            raise ValueError("declared start collides with a conservative footprint")
        self._spec = deepcopy(public_spec)
        self._episode, self._noise_seed = episode_id, noise_seed
        self._steps, self._initial_frames, self._collisions = 0, 0, 0
        self._closed = False
        self.resource_report = report
        _COUNTERS["worlds_created"] += 1

    @property
    def pose(self):
        return tuple(float(value) for value in self._pose)

    def public_status(self):
        return {"pose_xyyaw_rad": self.pose, "paid_actions": self._steps,
                "remaining_actions": self._spec["task"]["max_actions"]-self._steps,
                "initial_frames": self._initial_frames, "collisions": self._collisions,
                "returned_xy_and_yaw": return_pose_matches(self._pose, self._start), "closed": self._closed}

    def _capture(self, receipt):
        sensor, motion = self._spec["sensor"], self._spec["motion"]
        transform = camera_transform_xyyaw(self._pose, motion["camera_height_m"])
        k = np.asarray(sensor["intrinsic"], dtype=float)
        rgb, depth = render_rgbd_arrays(self._vertices, self._triangles, intrinsic=k, world_from_camera=transform,
            width=sensor["width"], height=sensor["height"], triangle_rgb=self._rgb,
            triangle_instance_id=self._owners, marker_patches=self._markers,
            minimum_depth_m=sensor["depth_min_m"], maximum_depth_m=sensor["depth_max_m"],
            relative_sigma=sensor["depth_noise_relative_std"], noise_seed=self._noise_seed, step=self._steps)
        observation = PaidRGBDObservationV40(f"{self._episode}:frame:{self._steps}", self._steps, rgb, depth, k, transform)
        scan = render_planar_scan(self._vertices, self._triangles, self._pose,
            maximum_range_m=sensor["lidar_max_range_m"], laser_height_m=self._laser_height,
            number_of_rays=self._scan_rays, timestamp_s=float(self._steps))
        _COUNTERS["rgbd_frames"] += 1; _COUNTERS["scans"] += 1
        return SensorStepV41(observation, scan, deepcopy(receipt))

    def initial_observation(self):
        if self._closed or self._initial_frames or self._steps:
            raise ValueError("initial grant is allowed exactly once before any paid action")
        self._initial_frames = 1
        return self._capture({"action": "initial_observation", "paid_step": 0, "action_cost": 0,
                              "initial_frames": 1, "collision": False, "pose_xyyaw_rad": self.pose})

    def step(self, action):
        if self._closed or not self._initial_frames:
            raise ValueError("one explicit initial observation is required before paid actions")
        motion = self._spec["motion"]
        receipt = paid_motion_transition(self._pose, action, paid_step=self._steps,
            max_actions=self._spec["task"]["max_actions"], start_pose=self._start,
            obstacle_polygons=self._footprints, radius_m=self._radius,
            translation_step_m=motion["translation_step_m"], rotation_step_deg=motion["rotation_step_deg"])
        self._pose = np.asarray(receipt["pose_xyyaw_rad"])
        self._steps = receipt["paid_step"]; self._collisions += int(receipt["collision"])
        _COUNTERS["paid_actions"] += 1
        return self._capture(receipt)

    def close(self):
        self._closed = True
        return self.public_status()


def create_development_sensor(asset_dir, public_spec, **kwargs):
    return DevelopmentSensorV41(asset_dir, public_spec, **kwargs)
