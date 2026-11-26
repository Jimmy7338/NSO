"""Measured-only CPU TSDF and occupancy adapter for the V40 paid RGB-D packet.

No structure prototype, owner label, private scene, or completed surface enters
fusion. Occupancy reports sensor-touched cell area, not ground-truth coverage.
"""
from copy import deepcopy
import hashlib
import math

import numpy as np

from nso.instance_belief_v40 import PaidRGBDObservationV40
from utils.rgbd_contract import PlanarScan


def _array_sha256(array):
    value = np.ascontiguousarray(array)
    header = f"{value.dtype.str}:{value.shape}:".encode()
    return hashlib.sha256(header + value.tobytes()).hexdigest()


def _positive(value, name):
    if isinstance(value, (bool, np.bool_)) or not np.isscalar(value):
        raise ValueError(name + " must be a positive finite scalar")
    result = float(value)
    if not math.isfinite(result) or result <= 0:
        raise ValueError(name + " must be a positive finite scalar")
    return result


def _scan_copy(scan, previous_timestamp):
    if type(scan) is not PlanarScan:
        raise TypeError("strict PlanarScan required")
    ranges = np.asarray(scan.ranges_m)
    if (ranges.ndim != 1 or not 1 <= len(ranges) <= 8192 or ranges.dtype.kind not in "fiu"
            or not np.isfinite(ranges).all() or np.any(ranges < 0)):
        raise ValueError("finite nonnegative 1D scan ranges required, at most 8192 rays")
    maximum = _positive(scan.range_max_m, "scan range maximum")
    if maximum > 100:
        raise ValueError("scan range maximum exceeds bounded 100 m adapter contract")
    timestamp = float(scan.timestamp_s)
    angle_min = float(scan.angle_min_rad)
    increment = float(scan.angle_increment_rad)
    if not np.isfinite([timestamp, angle_min, increment]).all() or timestamp < 0:
        raise ValueError("finite scan timestamp and angles required")
    if increment == 0 or abs(increment) * max(1, len(ranges)-1) > 2*math.pi + 1e-8:
        raise ValueError("nonzero scan increment within one revolution required")
    if previous_timestamp is not None and timestamp <= previous_timestamp:
        raise ValueError("scan timestamps must strictly increase; paid_step is not a timestamp")
    transform = np.asarray(scan.world_from_laser, dtype=float)
    if (transform.shape != (4, 4) or not np.isfinite(transform).all()
            or not np.allclose(transform[3], [0, 0, 0, 1], atol=1e-10, rtol=0)
            or not np.allclose(transform[:3, :3].T @ transform[:3, :3], np.eye(3), atol=1e-6, rtol=0)
            or not math.isclose(np.linalg.det(transform[:3, :3]), 1., abs_tol=1e-6)):
        raise ValueError("proper homogeneous laser pose required")
    # The public contract is planar in world XY, not a tilted 3D scan.
    if not np.allclose(transform[:3, 2], [0., 0., 1.], atol=1e-6, rtol=0):
        raise ValueError("upright planar laser required")
    tolerance = 4 * np.finfo(ranges.dtype).eps * maximum if ranges.dtype.kind == "f" else 0.
    if np.any(ranges > maximum + tolerance):
        raise ValueError("scan range exceeds range_max_m")
    digest = hashlib.sha256()
    for name in PlanarScan.__dataclass_fields__:
        value = np.asarray(getattr(scan, name))
        digest.update(name.encode() + _array_sha256(value).encode())
    copied = dict(timestamp_s=timestamp, ranges_m=np.minimum(np.array(ranges, dtype=float, copy=True), maximum),
                  angle_min_rad=angle_min, angle_increment_rad=increment, range_max_m=maximum,
                  world_from_laser=transform.copy(), no_hit_tolerance_m=tolerance)
    return copied, digest.hexdigest()


class ObservedMapperV42:
    """Once-per-paid-frame integration; all input checks precede map mutation.

    Grid row zero is the minimum-y edge (unlike older image-oriented grids).
    Invalid scan rays are encoded as zero; maximum-range rays are clear rays.
    Observed occupied cells persist under later free rays (static conservative
    occupancy). A fresh identical paid packet can improve TSDF averaging but
    cannot gain additional known area merely through repeated integration.
    """
    def __init__(self, *, shape, resolution_m=.1, origin_xy_m=(0., 0.),
                 voxel_m=.04, sdf_trunc_m=.12, near_m=.1, far_m=4.,
                 obstacle_height_m=(.15, 1.2), maximum_frames=10000):
        if (len(shape) != 2 or any(type(v) is not int or v < 1 for v in shape)
                or math.prod(shape) > 4_000_000):
            raise ValueError("bounded positive integer (rows, columns) shape required")
        self.shape = tuple(shape)
        self.resolution_m = _positive(resolution_m, "resolution")
        self.voxel_m = _positive(voxel_m, "voxel")
        self.sdf_trunc_m = _positive(sdf_trunc_m, "sdf truncation")
        self.near_m = _positive(near_m, "near depth")
        self.far_m = _positive(far_m, "far depth")
        if self.near_m >= self.far_m or self.sdf_trunc_m < self.voxel_m:
            raise ValueError("near < far and truncation >= voxel required")
        # Float32 is the actual backend depth representation. Keep its next
        # representable value as a STRICT upper threshold, preserving far.
        self._backend_far_m = float(np.nextafter(np.float32(self.far_m), np.float32(np.inf)))
        if not np.isfinite(self._backend_far_m):
            raise ValueError("far depth must fit float32")
        origin = np.asarray(origin_xy_m, dtype=float)
        heights = np.asarray(obstacle_height_m, dtype=float)
        if origin.shape != (2,) or not np.isfinite(origin).all():
            raise ValueError("finite origin_xy_m required")
        if heights.shape != (2,) or not np.isfinite(heights).all() or heights[0] >= heights[1]:
            raise ValueError("ordered obstacle height interval required")
        if type(maximum_frames) is not int or maximum_frames < 1:
            raise ValueError("positive integer maximum_frames required")
        self.origin_xy_m = origin.copy()
        self.obstacle_height_m = heights.copy()
        self.maximum_frames = maximum_frames
        self._belief = np.full(shape, -1, np.int8)
        self._observed = np.zeros(shape, bool)
        self._receipts = []
        self._frame_ids = set()
        self._last_scan_timestamp = None
        self._integration_count = 0
        self._poisoned = False
        import open3d as o3d
        self._o3d = o3d
        self._volume = o3d.pipelines.integration.ScalableTSDFVolume(
            voxel_length=self.voxel_m, sdf_trunc=self.sdf_trunc_m,
            color_type=o3d.pipelines.integration.TSDFVolumeColorType.RGB8)

    def grid_cell(self, point):
        xy = np.asarray(point, dtype=float)[:2]
        if xy.shape != (2,) or not np.isfinite(xy).all():
            raise ValueError("finite world point required")
        col, row = np.floor((xy-self.origin_xy_m)/self.resolution_m).astype(np.int64)
        return int(row), int(col)

    def _inside(self, cell):
        return 0 <= cell[0] < self.shape[0] and 0 <= cell[1] < self.shape[1]

    def _ray_cells(self, start, end):
        """Cells with a nonzero segment interval, using continuous ray geometry.

        Parametric clipping bounds work even when sensor/end are outside the
        map. Grid-corner-only touches do not assert free area in side cells.
        """
        start = (np.asarray(start[:2])-self.origin_xy_m)/self.resolution_m
        end = (np.asarray(end[:2])-self.origin_xy_m)/self.resolution_m
        delta = end-start
        lo, hi = 0., 1.
        for axis, limit in enumerate((self.shape[1], self.shape[0])):
            if abs(delta[axis]) < 1e-14:
                if not 0 <= start[axis] < limit:
                    return []
            else:
                first, last = sorted((-start[axis]/delta[axis], (limit-start[axis])/delta[axis]))
                lo, hi = max(lo, first), min(hi, last)
        if hi <= lo:
            return []
        crossings = [lo, hi]
        for axis, limit in enumerate((self.shape[1], self.shape[0])):
            if abs(delta[axis]) >= 1e-14:
                positions = (np.arange(1, limit, dtype=float)-start[axis])/delta[axis]
                crossings.extend(positions[(positions > lo) & (positions < hi)].tolist())
        intervals = np.unique(crossings)
        centers = start + ((intervals[:-1]+intervals[1:])/2)[:, None]*delta
        cells = np.floor(centers).astype(np.int64)
        return [(int(y), int(x)) for x, y in cells if 0 <= y < self.shape[0] and 0 <= x < self.shape[1]]

    def update(self, observation, scan=None):
        if self._poisoned:
            raise RuntimeError("TSDF backend failed previously; discard this mapper")
        if type(observation) is not PaidRGBDObservationV40:
            raise TypeError("strict PaidRGBDObservationV40 required")
        # Revalidate and copy, including ostensibly read-only user arrays.
        obs = PaidRGBDObservationV40.from_mapping(
            {name: getattr(observation, name) for name in PaidRGBDObservationV40.__dataclass_fields__})
        if obs.frame_id in self._frame_ids or obs.paid_step != len(self._receipts):
            raise ValueError("unique frame_id and consecutive paid_step starting at zero required")
        if len(self._receipts) >= self.maximum_frames:
            raise ValueError("mapper frame capacity reached")
        canonical = np.asarray([[obs.intrinsic[0, 0], 0., obs.intrinsic[0, 2]],
                                [0., obs.intrinsic[1, 1], obs.intrinsic[1, 2]], [0., 0., 1.]])
        if not np.allclose(obs.intrinsic, canonical, atol=1e-12, rtol=0):
            raise ValueError("Open3D adapter requires canonical zero-skew pinhole intrinsics")
        scan_values, scan_sha = (None, None) if scan is None else _scan_copy(scan, self._last_scan_timestamp)
        valid = (obs.depth_m >= self.near_m) & (obs.depth_m <= self.far_m)
        depth = np.where(valid, obs.depth_m, 0.).astype(np.float32)
        next_belief = self._belief.copy()
        observed = np.zeros(self.shape, bool)
        scan_hits = scan_clear = scan_invalid = 0
        if scan_values is not None:
            pose = scan_values["world_from_laser"]
            for i, distance in enumerate(scan_values["ranges_m"]):
                if distance == 0:
                    scan_invalid += 1
                    continue
                angle = scan_values["angle_min_rad"] + i*scan_values["angle_increment_rad"]
                endpoint = pose[:3, :3] @ [math.cos(angle)*distance, math.sin(angle)*distance, 0.] + pose[:3, 3]
                target = self.grid_cell(endpoint)
                hit = distance < scan_values["range_max_m"] - scan_values["no_hit_tolerance_m"]
                scan_hits += int(hit)
                scan_clear += int(not hit)
                for cell in self._ray_cells(pose[:3, 3], endpoint):
                    if hit and cell == target:
                        continue
                    observed[cell] = True
                    if next_belief[cell] != 1:
                        next_belief[cell] = 0
                if hit and self._inside(target):
                    next_belief[target] = 1
                    observed[target] = True
        rows, cols = np.nonzero(valid)
        local = (np.column_stack((cols, rows, np.ones(len(rows)))) @ np.linalg.inv(obs.intrinsic).T
                 * depth[rows, cols, None])
        points = local @ obs.world_from_camera[:3, :3].T + obs.world_from_camera[:3, 3]
        obstacles = points[(points[:, 2] >= self.obstacle_height_m[0]) & (points[:, 2] <= self.obstacle_height_m[1])]
        if len(obstacles):
            cells = np.unique(np.floor((obstacles[:, :2]-self.origin_xy_m)/self.resolution_m).astype(np.int64), axis=0)
            for col, row in cells:
                if self._inside((row, col)):
                    next_belief[row, col] = 1
                    observed[row, col] = True
        origin = self.grid_cell(obs.world_from_camera[:3, 3])
        conflict = self._inside(origin) and next_belief[origin] == 1
        receipt = dict(schema_version="v42.observed_mapping_receipt.v1", frame_id=obs.frame_id,
            paid_step=obs.paid_step, observation_sha256=obs.sha256(), scan_sha256=scan_sha,
            scan_timestamp_s=None if scan_values is None else scan_values["timestamp_s"],
            scan_pairing="caller-supplied same acquisition; timestamp monotonicity checked; no paid_step-to-time assumption",
            clipped_depth_float32_sha256=_array_sha256(depth),
            axial_clip_m=[self.near_m, self.far_m], valid_depth_pixels=int(valid.sum()),
            invalid_zero_depth_pixels=int(np.count_nonzero(obs.depth_m == 0)),
            clipped_near_pixels=int(np.count_nonzero((obs.depth_m > 0) & (obs.depth_m < self.near_m))),
            clipped_far_pixels=int(np.count_nonzero(obs.depth_m > self.far_m)),
            tsdf_integrated=bool(valid.any()), tsdf_integration_count=self._integration_count + int(valid.any()),
            scan_hit_rays=scan_hits, scan_max_range_clear_rays=scan_clear, scan_invalid_zero_rays=scan_invalid,
            depth_obstacle_points=len(obstacles), observed_cells_this_frame=int(observed.sum()),
            newly_known_cells=int(np.count_nonzero((self._belief < 0) & (next_belief >= 0))),
            current_camera_cell=list(origin), current_cell_obstacle_conflict=bool(conflict),
            geometry_source="measured RGB-D only", semantic_or_prototype_fusion=False,
            world_from_camera=obs.world_from_camera.tolist())
        # Everything above is validation/preparation. Invalid inputs have not
        # mutated TSDF, occupancy, frame counters, IDs, or scan timestamps.
        if valid.any():
            rgbd = self._o3d.geometry.RGBDImage.create_from_color_and_depth(
                self._o3d.geometry.Image(np.ascontiguousarray(obs.rgb)),
                self._o3d.geometry.Image(np.ascontiguousarray(depth)),
                depth_scale=1., depth_trunc=self._backend_far_m, convert_rgb_to_intensity=False)
            intrinsic = self._o3d.camera.PinholeCameraIntrinsic(depth.shape[1], depth.shape[0],
                obs.intrinsic[0, 0], obs.intrinsic[1, 1], obs.intrinsic[0, 2], obs.intrinsic[1, 2])
            try:
                self._volume.integrate(rgbd, intrinsic, np.linalg.inv(obs.world_from_camera))
            except Exception:
                # Open3D provides no transactional rollback after an internal
                # backend error. Fail closed instead of claiming a clean retry.
                self._poisoned = True
                raise
            self._integration_count += 1
        self._belief, self._observed = next_belief, observed
        self._frame_ids.add(obs.frame_id)
        self._receipts.append(receipt)
        if scan_values is not None:
            self._last_scan_timestamp = scan_values["timestamp_s"]
        return deepcopy(receipt)

    def occupancy_arrays(self):
        return self._belief.copy(), self._observed.copy()

    def mesh_arrays(self):
        if self._poisoned:
            raise RuntimeError("TSDF backend failed previously; discard this mapper")
        mesh = self._volume.extract_triangle_mesh()
        return dict(vertices=np.asarray(mesh.vertices).copy(), triangles=np.asarray(mesh.triangles).copy(),
                    vertex_colors=np.asarray(mesh.vertex_colors).copy())

    def snapshot(self):
        area = self.resolution_m**2
        free, occupied = int(np.count_nonzero(self._belief == 0)), int(np.count_nonzero(self._belief == 1))
        return deepcopy(dict(schema_version="v42.observed_mapping.v1", frames=len(self._receipts),
            tsdf_integration_count=self._integration_count, backend_poisoned=self._poisoned,
            shape=list(self.shape), resolution_m=self.resolution_m, origin_xy_m=self.origin_xy_m.tolist(),
            grid_convention="row increases with world y; column increases with world x; origin is lower-left boundary",
            occupancy_sha256=_array_sha256(self._belief), known_cells=free+occupied,
            observed_free_cells=free, observed_occupied_cells=occupied,
            measured_known_area_m2=(free+occupied)*area, measured_free_area_m2=free*area,
            measured_occupied_area_m2=occupied*area,
            coverage_fraction=None, coverage_denominator=None,
            area_interpretation="union of sensor-touched grid cells; no ground-truth free-space denominator",
            metric_groundtruth_or_scene_input=False, semantic_or_prototype_fusion=False,
            near_m=self.near_m, far_m=self.far_m, voxel_m=self.voxel_m, sdf_trunc_m=self.sdf_trunc_m,
            backend="Open3D " + self._o3d.__version__, receipts=self._receipts))
