"""V34 packet contract: camera4m Euclidean, independent planar lidar8m.

Explicit versioned copy of V10's validation checks, with separate laser range.
No modified configuration or fabricated/clipped scan is passed to V10.validate.
The record inherits the same fields/hash and is accepted by old packet IO;
call validate_sensor_packet_v34 on records loaded by the old IO function.
No rendering, scan query, mapper or fusion is performed by validation.
"""
from dataclasses import dataclass
import numpy as np
from nso.cpu_sensor_contract_v10 import SensorPacket
from utils.rgbd_contract import RGBDFrame, PlanarScan


@dataclass(frozen=True)
class SensorPacketV34(SensorPacket):
    def validate(self, transform, config):
        return validate_sensor_packet_v34(self, transform, config)


def validate_sensor_packet_v34(packet, transform, config):
    if not isinstance(packet, SensorPacket):
        raise ValueError("SensorPacket or subclass required")
    fixed = dict(width_px=96, height_px=72, resolution_m=.2, fov_deg=90.,
                 max_depth_m=4., laser_range_m=8., camera_height_m=.9,
                 laser_height_m=.25, laser_rays=180, robot_radius_m=.2)
    for key, value in fixed.items():
        actual = getattr(config, key, None)
        if actual is None or isinstance(actual, (bool, np.bool_)) or not np.isfinite(actual) or actual != value:
            raise ValueError("V34 fixed sensor calibration mismatch: "+key)
    self = packet
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
            or not np.isclose(scan.range_max_m, config.laser_range_m, rtol=0, atol=1e-6)
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
    v, u = np.mgrid[:config.height_px, :config.width_px]
    ray_norm = np.sqrt(1 + ((u-frame.intrinsic[0, 2])/frame.intrinsic[0, 0])**2
                       + ((v-frame.intrinsic[1, 2])/frame.intrinsic[1, 1])**2)
    if (frame.depth_m*ray_norm > config.max_depth_m+1e-6).any():
        raise ValueError("V34 camera valid returns must also obey Euclidean4m range")
    if ((frame.depth_m > 0) & (frame.depth_m <= .15)).any():
        raise ValueError("V34 has the canonical0.15m axial near clip")
    return self

