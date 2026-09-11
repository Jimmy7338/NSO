"""Synthetic sensor regressions only; no world or efficacy data is constructed."""
import unittest

import numpy as np

from env.virtual3d import VirtualConfig, camera_pose
from nso.cpu_sensor_contract_v10 import GridTransform, SensorPacket
from nso.execution_guard_v8 import ObservedExecutionGuard
from nso.observed_runtime_mapper_v10 import ObservedRuntimeMapperV10
from nso.semantic_completion_v3 import SemanticHistoryMapperV3
from utils.rgbd_contract import RGBDFrame, PlanarScan


class ObservedRuntimeMapperTests(unittest.TestCase):
    def setUp(self):
        self.config = VirtualConfig(width_px=16, height_px=16, laser_rays=180)
        self.shape = (31, 31)
        self.position = (15, 15)
        self.guard = ObservedExecutionGuard(.2, .2)

    def packet(self, *, depth=0., laser_hit=None, timestamp=0., heading=0):
        config = self.config
        pose = camera_pose(self.position, heading, config, self.shape[0])
        focal = config.width_px / (2 * np.tan(np.deg2rad(config.fov_deg / 2)))
        intrinsic = np.array([[focal, 0., 7.5], [0., focal, 7.5], [0., 0., 1.]])
        frame = RGBDFrame(timestamp, np.full((16, 16), depth, np.float32),
                          np.zeros((16, 16, 3), np.uint8), intrinsic, pose,
                          np.zeros((16, 16), np.int32))
        laser_pose = np.eye(4)
        laser_pose[:3, 0] = pose[:3, 2]
        laser_pose[:3, 1] = -pose[:3, 0]
        laser_pose[:3, 3] = pose[:3, 3]
        laser_pose[2, 3] = config.laser_height_m
        ranges = np.full(config.laser_rays, config.max_depth_m, np.float32)
        if laser_hit is not None:
            ranges[config.laser_rays // 2] = laser_hit
        scan = PlanarScan(timestamp, ranges, -np.pi, 2*np.pi/config.laser_rays,
                          config.max_depth_m, laser_pose)
        packet = SensorPacket("synthetic", "mapper-review", f"frame-{timestamp}",
                              int(timestamp), frame, scan, self.position, heading,
                              "synthetic_sensor_regression", "declared_discrete_pose")
        return packet.validate(GridTransform(self.shape, .2), config)

    def assert_guard_rejects(self, mapper):
        self.assertEqual(mapper.belief[self.position], 1)
        self.assertTrue(mapper.visible[self.position])
        self.assertTrue(mapper.current_footprint_conflict)
        for action in ("right", "left", "forward"):
            check = self.guard.assess(mapper.belief, self.position, 0, action, 20)
            self.assertFalse(check.allowed)
            self.assertEqual(check.reason, "current_footprint_not_known_safe")
        returning = self.guard.return_plan(mapper.belief, self.position, 0,
                                           (*self.position, 0), 20)
        self.assertFalse(returning.available)
        self.assertEqual(returning.reason, "current_footprint_not_known_safe")

    def test_valid_short_scan_exposes_legacy_clear_and_new_guard_rejects(self):
        packet = self.packet(laser_hit=.05)
        old = SemanticHistoryMapperV3(self.shape, self.config)
        new = ObservedRuntimeMapperV10(self.shape, self.config)
        old.update(packet.frame, packet.scan)
        new.update(packet.frame, packet.scan)
        self.assertEqual(old.belief[self.position], 0)
        self.assertTrue(self.guard.assess(old.belief, self.position, 0, "right", 20).allowed)
        self.assert_guard_rejects(new)
        self.assertIn("current_scan_hit_in_current_cell",
                      new.current_footprint_conflict_details["reasons"])
        self.assertEqual(new.current_footprint_conflict_count, 1)

    def test_current_cell_depth_hit_is_preserved_without_scan(self):
        packet = self.packet(depth=.05)
        mapper = ObservedRuntimeMapperV10(self.shape, self.config)
        mapper.update(packet.frame)
        self.assert_guard_rejects(mapper)
        self.assertEqual(mapper.current_footprint_conflict_details["reasons"],
                         ["current_depth_obstacle_in_current_cell"])

    def test_previous_current_occupancy_survives_subsequent_no_hit(self):
        mapper = ObservedRuntimeMapperV10(self.shape, self.config)
        for packet in (self.packet(laser_hit=.05), self.packet(timestamp=1.)):
            mapper.update(packet.frame, packet.scan)
        self.assert_guard_rejects(mapper)
        self.assertEqual(mapper.current_footprint_conflict_count, 2)
        self.assertEqual(mapper.current_footprint_conflict_details["reasons"],
                         ["previously_observed_occupied_current_cell"])
        self.assertEqual(mapper.current_footprint_conflict_details["map_version"], 2)

    def test_ordinary_raw_sequence_preserves_mapping_tsdf_and_quality_exactly(self):
        old = SemanticHistoryMapperV3(self.shape, self.config)
        new = ObservedRuntimeMapperV10(self.shape, self.config)
        for index, heading in enumerate((0, 1, 2, 3)):
            packet = self.packet(depth=1., timestamp=float(index), heading=heading)
            old.update(packet.frame, packet.scan)
            new.update(packet.frame, packet.scan)
            self.assertFalse(new.current_footprint_conflict)
            for name in ("belief", "visible", "camera_seen"):
                np.testing.assert_array_equal(getattr(old, name), getattr(new, name))
            self.assertEqual(old.frames, new.frames)
        self.assertEqual(set(old.quality), set(new.quality))
        for key in old.quality:
            for name in old.quality[key]:
                np.testing.assert_array_equal(old.quality[key][name], new.quality[key][name])
        for a, b in zip(old.evidence(), new.evidence()):
            np.testing.assert_array_equal(a, b)
        old_mesh, new_mesh = old.mesh(), new.mesh()
        for name in ("vertices", "triangles", "vertex_normals", "vertex_colors"):
            np.testing.assert_array_equal(np.asarray(getattr(old_mesh, name)),
                                          np.asarray(getattr(new_mesh, name)))
        self.assertEqual(len(old.keyframes), len(new.keyframes))
        self.assertEqual(new.current_footprint_conflict_count, 0)


if __name__ == "__main__":
    unittest.main()
