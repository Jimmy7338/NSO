"""Sensor/coordinate boundaries used by the opt-in CPU runtime, not efficacy."""
from dataclasses import replace
import unittest

import numpy as np

from env.virtual3d import VirtualConfig, VirtualWorld
from nso.cpu_sensor_contract_v10 import GridTransform, SensorPacket


class SensorContractTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.config = VirtualConfig(width_px=32, height_px=24, depth_sigma_m=0., dropout=0.)
        cls.world = VirtualWorld(cls.config, seed=21, layout="rooms")
        cls.transform = GridTransform(cls.world.shape, cls.config.resolution_m)

    def packet(self, heading=0):
        self.world.heading = heading
        return SensorPacket("scene", "episode", "frame-0", 0,
                            self.world.sense(), self.world.scan(),
                            tuple(map(int, self.world.position)), heading,
                            "rendered_rgb_marker", "simulator_exact")

    def assert_invalid(self, packet):
        with self.assertRaises(ValueError):
            packet.validate(self.transform, self.config)

    def test_nonzero_origin_rectangular_grid_round_trip_and_boundaries(self):
        transform = GridTransform((13, 21), .2, (-2.4, 5.1))
        for row in range(13):
            for column in range(21):
                xy = transform.cell_to_world((row, column))
                self.assertEqual(transform.world_to_cell(xy), (row, column))
        southwest = np.array(transform.origin_xy_m)
        self.assertEqual(transform.world_to_cell(southwest + [1e-8, 1e-8]), (12, 0))
        self.assertEqual(transform.world_to_cell(southwest - [1e-8, 1e-8]), (13, -1))
        for cell in ((-1, 0), (13, 0), (0, 21), (1.5, 2), (True, 2)):
            with self.assertRaises(ValueError):
                transform.cell_to_world(cell)
        for xy in ((0., np.nan), (np.inf, 0.), (0., 1., 2.)):
            with self.assertRaises(ValueError):
                transform.world_to_cell(xy)

    def test_cpu_support_limit_is_checked_at_packet_boundary(self):
        packet = self.packet()
        for transform in (GridTransform(self.world.shape, .2, (.01, 0.)),
                          GridTransform(self.world.shape, .1),
                          GridTransform(self.world.shape, .200001)):
            with self.assertRaises(ValueError):
                packet.validate(transform, self.config)
        with self.assertRaises(ValueError):
            self.transform.validate_cpu_mapper(replace(self.config, resolution_m=.1))

    def test_real_renderer_four_headings_and_independent_backprojection(self):
        forwards = ((0., 1., 0.), (1., 0., 0.), (0., -1., 0.), (-1., 0., 0.))
        for heading, forward in enumerate(forwards):
            packet = self.packet(heading).validate(self.transform, self.config)
            np.testing.assert_allclose(packet.scan.world_from_laser[:3, 0], forward, atol=1e-12)
            self.assertAlmostEqual(packet.scan.world_from_laser[2, 3], self.config.laser_height_m)
            v, u = np.argwhere(packet.frame.depth_m > 0)[0]
            d = packet.frame.depth_m[v, u]
            focal = self.config.width_px / (2 * np.tan(np.deg2rad(self.config.fov_deg) / 2))
            camera = np.array([(u - (self.config.width_px - 1) / 2) * d / focal,
                               (v - (self.config.height_px - 1) / 2) * d / focal, d])
            right = np.cross(forward, (0., 0., 1.))
            expected_point = (np.array([*self.transform.cell_to_world(packet.position), self.config.camera_height_m])
                              + camera[0] * right + camera[1] * np.array([0., 0., -1.])
                              + camera[2] * np.array(forward))
            actual_points, _ = packet.frame.points()
            np.testing.assert_allclose(actual_points[0], expected_point, atol=1e-6)

    def test_scan_wrong_yaw_roll_height_and_xy_are_rejected(self):
        packet = self.packet(1)
        for mode in ("yaw", "roll", "height", "xy", "homogeneous"):
            pose = packet.scan.world_from_laser.copy()
            if mode == "yaw":
                pose[:3, :3] = pose[:3, :3] @ np.array([[0., -1., 0.], [1., 0., 0.], [0., 0., 1.]])
            elif mode == "roll":
                pose[:3, :3] = pose[:3, :3] @ np.array([[1., 0., 0.], [0., 0., -1.], [0., 1., 0.]])
            elif mode == "height":
                pose[2, 3] += .1
            elif mode == "xy":
                pose[0, 3] += .1
            else:
                pose[3, 0] = .001
            self.assert_invalid(replace(packet, scan=replace(packet.scan, world_from_laser=pose)))
        camera = packet.frame.world_from_camera.copy()
        camera[:3, :3] = self.packet(2).frame.world_from_camera[:3, :3]
        self.assert_invalid(replace(packet, frame=replace(packet.frame, world_from_camera=camera)))

    def test_rgbd_types_calibration_depth_range_and_invalid_label_mask(self):
        packet = self.packet()
        for change in (dict(depth_m=packet.frame.depth_m.astype(np.uint16)),
                       dict(color_rgb=packet.frame.color_rgb.astype(float)),
                       dict(semantic=packet.frame.semantic.astype(float)),
                       dict(semantic=np.full(packet.frame.semantic.shape, -1, dtype=np.int32)),
                       dict(depth_m=packet.frame.depth_m.tolist()),
                       dict(depth_m=np.full(packet.frame.depth_m.shape, self.config.max_depth_m + 1.))):
            self.assert_invalid(replace(packet, frame=replace(packet.frame, **change)))
        for index, value in (((0, 1), .2), ((0, 2), 0.), ((2, 2), 2.), ((1, 1), 10.)):
            intrinsic = packet.frame.intrinsic.copy()
            intrinsic[index] = value
            self.assert_invalid(replace(packet, frame=replace(packet.frame, intrinsic=intrinsic)))
        depth, semantic = packet.frame.depth_m.copy(), packet.frame.semantic.copy()
        depth[0, 0], semantic[0, 0] = 0., 2
        self.assert_invalid(replace(packet, frame=replace(packet.frame, depth_m=depth, semantic=semantic)))
        for change in (dict(collision="false"), dict(done=1), dict(heading=True),
                       dict(action_id=True), dict(action="stop"), dict(scan=None)):
            self.assert_invalid(replace(packet, **change))

    def test_scan_calibration_invalid_returns_and_timestamp_mismatch(self):
        packet = self.packet()
        for change in (dict(timestamp_s=packet.scan.timestamp_s + .01),
                       dict(angle_min_rad=0.), dict(angle_increment_rad=.02),
                       dict(ranges_m=packet.scan.ranges_m[:-1]),
                       dict(ranges_m=packet.scan.ranges_m.astype(np.int32)),
                       dict(range_max_m=100.)):
            self.assert_invalid(replace(packet, scan=replace(packet.scan, **change)))
        for value in (0., -1., np.inf, np.nan, self.config.max_depth_m + .01):
            ranges = packet.scan.ranges_m.copy()
            ranges[0] = value
            self.assert_invalid(replace(packet, scan=replace(packet.scan, ranges_m=ranges)))

    def test_packet_digest_covers_sensor_bytes_dtypes_and_terminal_metadata(self):
        packet = self.packet().validate(self.transform, self.config)
        original = packet.sha256()
        copied = replace(packet, frame=replace(packet.frame, depth_m=packet.frame.depth_m.copy(),
                                               color_rgb=np.asfortranarray(packet.frame.color_rgb)),
                         scan=replace(packet.scan, ranges_m=packet.scan.ranges_m.copy()))
        self.assertEqual(copied.sha256(), original)
        changes = [replace(packet, done=True), replace(packet, collision=True),
                   replace(packet, episode_id="next"), replace(packet, frame_id="new"),
                   replace(packet, action_id=1), replace(packet, action="right"),
                   replace(packet, pose_source="odometry"),
                   replace(packet, frame=replace(packet.frame, depth_m=packet.frame.depth_m.astype(np.float64)))]
        ranges = packet.scan.ranges_m.copy()
        ranges[0] = np.nextafter(ranges[0], np.float32(0.))
        changes.append(replace(packet, scan=replace(packet.scan, ranges_m=ranges)))
        rgb = packet.frame.color_rgb.copy()
        rgb[0, 0, 0] ^= 1
        changes.append(replace(packet, frame=replace(packet.frame, color_rgb=rgb)))
        for changed in changes:
            self.assertNotEqual(changed.sha256(), original)
        with self.assertRaises(ValueError):
            replace(packet, frame=replace(packet.frame, depth_m=packet.frame.depth_m.astype(object))).sha256()


if __name__ == "__main__":
    unittest.main()
