"""Measured packet and coordinate contract for the opt-in CPU ANS backend."""
from dataclasses import dataclass, fields
import hashlib
import json
import numpy as np
from utils.rgbd_contract import RGBDFrame, PlanarScan


def json_value(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(k): json_value(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [json_value(v) for v in value]
    return value


def digest(value):
    return hashlib.sha256(json.dumps(json_value(value), sort_keys=True,
                          separators=(",", ":"), allow_nan=False).encode()).hexdigest()


@dataclass(frozen=True)
class GridTransform:
    shape: tuple
    resolution_m: float
    origin_xy_m: tuple = (0., 0.)

    def __post_init__(self):
        if (len(self.shape) != 2 or any(type(x) is not int or x < 1 for x in self.shape)
                or len(self.origin_xy_m) != 2
                or not np.isfinite([self.resolution_m, *self.origin_xy_m]).all()
                or self.resolution_m <= 0):
            raise ValueError("invalid metric grid transform")

    def world_to_cell(self, xy):
        xy = np.asarray(xy, float)
        if xy.shape != (2,) or not np.isfinite(xy).all():
            raise ValueError("expected finite world x/y metres")
        col, north = np.floor((xy - self.origin_xy_m) / self.resolution_m).astype(int)
        return self.shape[0] - 1 - int(north), int(col)

    def cell_to_world(self, cell):
        if (len(cell) != 2 or any(type(x) is not int for x in cell)
                or any(not 0 <= p < limit for p, limit in zip(cell, self.shape))):
            raise ValueError("expected an in-bounds integer grid cell")
        r, c = cell
        return (self.origin_xy_m[0] + (c + .5) * self.resolution_m,
                self.origin_xy_m[1] + (self.shape[0] - r - .5) * self.resolution_m)

    def validate_cpu_mapper(self, config):
        # Existing measured-patch and TSDF helpers use this coordinate frame.
        if tuple(self.origin_xy_m) != (0., 0.) or not np.isclose(self.resolution_m, .2, rtol=0, atol=1e-12):
            raise ValueError("current CPU mapper requires origin (0,0) and a 0.2 m grid")
        if not np.isclose(config.resolution_m, self.resolution_m, rtol=0, atol=1e-12):
            raise ValueError("mapper and transform resolutions disagree")


@dataclass(frozen=True)
class SensorPacket:
    scene_id: str
    episode_id: str
    frame_id: str
    action_id: int
    frame: RGBDFrame
    scan: PlanarScan
    position: tuple
    heading: int
    sensor_source: str
    pose_source: str
    action: str | None = None
    collision: bool = False
    done: bool = False

    def validate(self, transform, config):
        transform.validate_cpu_mapper(config)
        if not isinstance(self.frame, RGBDFrame) or not isinstance(self.scan, PlanarScan):
            raise ValueError("CPU packet requires RGBDFrame and PlanarScan records")
        if type(self.collision) is not bool or type(self.done) is not bool:
            raise ValueError("collision and done must be booleans")
        frame = self.frame
        for name in ("depth_m", "color_rgb", "intrinsic", "world_from_camera", "semantic"):
            if not isinstance(getattr(frame, name), np.ndarray):
                raise ValueError(f"{name} must be an ndarray")
        if (frame.depth_m.dtype not in (np.dtype("float32"), np.dtype("float64"))
                or frame.color_rgb.dtype != np.dtype("uint8")
                or not np.issubdtype(frame.semantic.dtype, np.integer)
                or (frame.semantic < 0).any()
                or not np.issubdtype(frame.intrinsic.dtype, np.floating)
                or not np.issubdtype(frame.world_from_camera.dtype, np.floating)):
            raise ValueError("expected floating metre depth/calibration, uint8 RGB and nonnegative integer labels")
        self.frame.validate()
        if (type(config.width_px) is not int or type(config.height_px) is not int
                or config.width_px < 8 or config.height_px < 8
                or not np.isfinite([config.fov_deg, config.max_depth_m, config.camera_height_m,
                                    config.laser_height_m]).all()
                or not 0 < config.fov_deg < 180 or config.max_depth_m <= 0
                or config.camera_height_m <= 0 or config.laser_height_m <= 0):
            raise ValueError("invalid CPU sensor configuration")
        focal = config.width_px / (2 * np.tan(np.deg2rad(config.fov_deg) / 2))
        expected_intrinsic = np.array([[focal, 0., (config.width_px - 1) / 2],
                                       [0., focal, (config.height_px - 1) / 2], [0., 0., 1.]])
        if (frame.depth_m.shape != (config.height_px, config.width_px)
                or not np.allclose(frame.intrinsic, expected_intrinsic, rtol=0, atol=1e-6)
                or (frame.depth_m > config.max_depth_m + 1e-6).any()
                or (frame.semantic[frame.depth_m == 0] != 0).any()):
            raise ValueError("RGB-D shape/calibration/range must match the current rectified CPU camera")
        if any(not isinstance(x, str) or not x for x in
               (self.scene_id, self.episode_id, self.frame_id, self.sensor_source, self.pose_source)):
            raise ValueError("packet identities and sensor/pose provenance are required")
        if type(self.action_id) is not int or self.action_id < 0:
            raise ValueError("action_id must be a nonnegative integer")
        if self.action not in (None, "forward", "left", "right"):
            raise ValueError("unsupported paid action")
        if (len(self.position) != 2 or any(type(x) is not int for x in self.position)
                or type(self.heading) is not int or self.heading not in range(4)):
            raise ValueError("expected integer cell and N/E/S/W heading")
        if any(not 0 <= p < limit for p, limit in zip(self.position, transform.shape)):
            raise ValueError("odometry lies outside map")
        if transform.world_to_cell(self.frame.world_from_camera[:2, 3]) != self.position:
            raise ValueError("camera odometry and robot cell disagree")
        # This backend has a rigid, centred, horizontal optical camera. Reject
        # other mounts until the ROS base/camera extrinsic adapter exists.
        from env.virtual3d import camera_pose
        expected = camera_pose(self.position, self.heading, config, transform.shape[0])
        if not np.allclose(self.frame.world_from_camera, expected, rtol=0, atol=1e-6):
            raise ValueError("current CPU backend requires centred discrete base/camera pose")
        scan = self.scan
        if (not isinstance(scan.ranges_m, np.ndarray)
                or not np.issubdtype(scan.ranges_m.dtype, np.floating)
                or not isinstance(scan.world_from_laser, np.ndarray)
                or not np.issubdtype(scan.world_from_laser.dtype, np.floating)):
            raise ValueError("scan ranges and pose must be floating ndarrays")
        ranges = np.asarray(scan.ranges_m)
        pose = np.asarray(scan.world_from_laser)
        if (ranges.ndim != 1 or not len(ranges) or not np.isfinite(ranges).all()
                or (ranges <= 0).any() or (ranges > scan.range_max_m + 1e-6).any()
                or not np.isfinite([scan.timestamp_s, scan.angle_min_rad,
                                    scan.angle_increment_rad, scan.range_max_m]).all()
                or scan.angle_increment_rad <= 0 or scan.range_max_m <= 0):
            raise ValueError("invalid planar scan; normalize invalid returns before packet creation")
        if (pose.shape != (4, 4) or not np.isfinite(pose).all()
                or not np.allclose(pose[3], [0, 0, 0, 1], rtol=0, atol=1e-6)
                or not np.allclose(pose[:3, :3].T @ pose[:3, :3], np.eye(3), rtol=0, atol=1e-5)
                or not np.isclose(np.linalg.det(pose[:3, :3]), 1., rtol=0, atol=1e-5)):
            raise ValueError("invalid planar scan transform")
        if abs(scan.timestamp_s - self.frame.timestamp_s) > 1e-6:
            raise ValueError("CPU packets require synchronized camera and scan timestamps")
        if (type(config.laser_rays) is not int or config.laser_rays < 1
                or len(ranges) != config.laser_rays
                or not np.isclose(scan.range_max_m, config.max_depth_m, rtol=0, atol=1e-6)
                or not np.isclose(scan.angle_min_rad, -np.pi, rtol=0, atol=1e-6)
                or not np.isclose(scan.angle_increment_rad, 2 * np.pi / config.laser_rays, rtol=0, atol=1e-6)):
            raise ValueError("current CPU scan requires the configured complete 360-degree calibration")
        expected_laser = np.eye(4)
        expected_laser[:3, 0] = expected[:3, 2]
        expected_laser[:3, 1] = -expected[:3, 0]
        expected_laser[:3, 3] = expected[:3, 3]
        expected_laser[2, 3] = config.laser_height_m
        if not np.allclose(pose, expected_laser, rtol=0, atol=1e-6):
            raise ValueError("current CPU scan must match base heading and centred horizontal laser mount")
        return self

    def sha256(self):
        h = hashlib.sha256()
        def append(data):
            # Length framing avoids ambiguity between arbitrary array bytes and
            # the next field's header. Dtype and shape are part of the identity.
            h.update(len(data).to_bytes(8, "big"))
            h.update(data)
        for entry in fields(self):
            value = getattr(self, entry.name)
            append(entry.name.encode())
            if entry.name in ("frame", "scan"):
                for inner in fields(value):
                    arr = np.ascontiguousarray(np.asarray(getattr(value, inner.name)))
                    if arr.dtype.hasobject:
                        raise ValueError("object arrays cannot identify sensor bytes")
                    append(inner.name.encode()); append(arr.dtype.str.encode())
                    append(str(arr.shape).encode()); append(arr.tobytes())
            else:
                append(digest(value).encode())
        return h.hexdigest()
