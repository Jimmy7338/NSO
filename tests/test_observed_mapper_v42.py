"""Small analytic measured planes and scan rays; never a DEV World rollout."""
from dataclasses import replace
import math
import unittest

import numpy as np

from nso.instance_belief_v40 import PaidRGBDObservationV40
from nso.observed_mapper_v42 import ObservedMapperV42
from utils.rgbd_contract import PlanarScan


def packet(step=0, depth=2., pose=None, shape=(32, 48), frame_id=None):
    h, w = shape
    array = np.full(shape, depth, dtype=np.float64) if np.isscalar(depth) else np.asarray(depth)
    return PaidRGBDObservationV40(frame_id or f"analytic_{step}", step,
        np.full(array.shape+(3,), 130, np.uint8), array,
        np.asarray([[40., 0., (w-1)/2], [0., 40., (h-1)/2], [0., 0., 1.]]),
        np.eye(4) if pose is None else pose)


def scan(*, ranges=(1.,), timestamp=0., pose=None, maximum=8., angle=0., increment=math.pi/180):
    transform = np.eye(4) if pose is None else np.asarray(pose).copy()
    if pose is None:
        transform[:3, 3] = [.15, .15, .3]
    return PlanarScan(timestamp, np.asarray(ranges, dtype=float), angle, increment, maximum, transform)


def mapper(**kwargs):
    defaults = dict(shape=(50, 50), origin_xy_m=(-1., -1.), resolution_m=.1)
    defaults.update(kwargs)
    return ObservedMapperV42(**defaults)


class ObservedMapperV42Tests(unittest.TestCase):
    def test_actual_cpu_tsdf_reconstructs_analytic_measured_plane(self):
        m = mapper()
        receipt = m.update(packet())
        mesh = m.mesh_arrays()
        self.assertGreater(len(mesh["triangles"]), 100)
        self.assertLess(np.max(np.abs(mesh["vertices"][:, 2]-2.)), .001)
        self.assertEqual(receipt["valid_depth_pixels"], 32*48)
        self.assertEqual(m.snapshot()["tsdf_integration_count"], 1)
        self.assertFalse(m.snapshot()["semantic_or_prototype_fusion"])

    def test_far_boundary_four_metres_survives_backend_strict_truncation(self):
        m = mapper(voxel_m=.06, sdf_trunc_m=.18)
        receipt = m.update(packet(depth=4.))
        vertices = m.mesh_arrays()["vertices"]
        self.assertGreater(len(vertices), 100)
        self.assertLess(np.max(np.abs(vertices[:, 2]-4.)), .002)
        self.assertEqual(receipt["clipped_far_pixels"], 0)

    def test_near_boundary_and_optical_z_clip_not_radial_clip(self):
        m = mapper(voxel_m=.005, sdf_trunc_m=.02)
        receipt = m.update(packet(depth=.1))
        vertices = m.mesh_arrays()["vertices"]
        self.assertGreater(len(vertices), 10)
        self.assertLess(np.max(np.abs(vertices[:, 2]-.1)), .002)
        self.assertEqual(receipt["valid_depth_pixels"], 32*48)
        # At far=4, corner rays have radial length >4 but remain valid above.

    def test_zero_near_and_far_invalid_depth_never_fuse(self):
        m = mapper()
        receipt = m.update(packet(depth=np.asarray([[0., .09, 4.01]]), shape=(1, 3)))
        self.assertFalse(receipt["tsdf_integrated"])
        self.assertEqual((receipt["invalid_zero_depth_pixels"], receipt["clipped_near_pixels"],
                          receipt["clipped_far_pixels"]), (1, 1, 1))
        self.assertEqual(m.snapshot()["frames"], 1)
        self.assertEqual(m.snapshot()["known_cells"], 0)
        self.assertEqual(len(m.mesh_arrays()["vertices"]), 0)

    def test_continuous_pose_is_used_without_grid_snapping(self):
        m = mapper()
        pose = np.eye(4)
        angle = .173
        pose[:3, :3] = [[math.cos(angle), 0., math.sin(angle)], [0., 1., 0.],
                        [-math.sin(angle), 0., math.cos(angle)]]
        pose[:3, 3] = [.123, -.071, .137]
        m.update(packet(pose=pose))
        vertices = m.mesh_arrays()["vertices"]
        distances = (vertices-pose[:3, 3]) @ pose[:3, 2]
        self.assertGreater(len(vertices), 100)
        self.assertLess(np.max(np.abs(distances-2.)), .004)

    def test_same_id_or_skipped_step_rejects_without_mutation(self):
        m = mapper()
        first = packet()
        m.update(first)
        before = m.snapshot()
        vertices = m.mesh_arrays()["vertices"]
        for value in (first, packet(step=2), packet(step=1, frame_id=first.frame_id)):
            with self.assertRaisesRegex(ValueError, "unique frame_id"):
                m.update(value)
            self.assertEqual(before, m.snapshot())
            np.testing.assert_array_equal(vertices, m.mesh_arrays()["vertices"])

    def test_identical_fresh_paid_frame_accepted_once_without_extra_area(self):
        m = mapper()
        m.update(packet(), scan())
        before = m.snapshot()
        receipt = m.update(packet(step=1), scan(timestamp=1.))
        self.assertEqual(receipt["newly_known_cells"], 0)
        self.assertEqual(m.snapshot()["known_cells"], before["known_cells"])
        self.assertEqual(m.snapshot()["tsdf_integration_count"], 2)
        self.assertNotEqual(receipt["observation_sha256"], before["receipts"][0]["observation_sha256"])

    def test_scan_nan_negative_excess_range_and_bad_pose_fail_before_fusion(self):
        m = mapper()
        tilted = np.eye(4)
        tilted[:3, :3] = [[1., 0., 0.], [0., 0., -1.], [0., 1., 0.]]
        invalid = [scan(ranges=(float("nan"),)), scan(ranges=(-.1,)), scan(ranges=(8.1,)),
                   scan(pose=tilted), scan(timestamp=float("nan")), scan(increment=0.)]
        before = m.snapshot()
        for value in invalid:
            with self.assertRaises(ValueError):
                m.update(packet(), value)
            self.assertEqual(m.snapshot(), before)
            self.assertEqual(len(m.mesh_arrays()["vertices"]), 0)
        m.update(packet(), scan())
        self.assertEqual(m.snapshot()["frames"], 1)

    def test_scan_timestamp_is_monotonic_not_equal_to_paid_step(self):
        m = mapper()
        receipt = m.update(packet(depth=0.), scan(timestamp=1730000000.25))
        self.assertEqual(receipt["paid_step"], 0)
        self.assertEqual(receipt["scan_timestamp_s"], 1730000000.25)
        before = m.snapshot()
        with self.assertRaisesRegex(ValueError, "timestamps"):
            m.update(packet(step=1), scan(timestamp=1730000000.25))
        self.assertEqual(before, m.snapshot())
        m.update(packet(step=1), scan(timestamp=1730000000.75))

    def test_occupied_origin_is_not_erased_by_current_or_future_free_ray(self):
        m = mapper(shape=(20, 20), origin_xy_m=(0., 0.))
        pose = np.eye(4)
        pose[:3, 3] = [.15, .15, .9]
        receipt = m.update(packet(depth=0., pose=pose), scan(ranges=(.02,)))
        belief, _ = m.occupancy_arrays()
        self.assertEqual(belief[1, 1], 1)
        self.assertTrue(receipt["current_cell_obstacle_conflict"])
        m.update(packet(step=1, depth=0., pose=pose), scan(ranges=(8.,), timestamp=1.))
        self.assertEqual(m.occupancy_arrays()[0][1, 1], 1)

    def test_depth_obstacle_at_camera_xy_is_not_erased(self):
        m = mapper(shape=(20, 20), origin_xy_m=(0., 0.))
        pose = np.eye(4)
        pose[:3, 3] = [.15, .15, .3]
        receipt = m.update(packet(depth=.5, pose=pose, shape=(1, 1)))
        self.assertTrue(receipt["current_cell_obstacle_conflict"])
        self.assertEqual(m.occupancy_arrays()[0][1, 1], 1)

    def test_nonzero_origin_and_continuous_diagonal_scan_crossing(self):
        m = mapper(shape=(3, 3), origin_xy_m=(-.1, -.1), resolution_m=.1)
        pose = np.eye(4)
        pose[:3, 3] = [-.099, -.049, .3]
        m.update(packet(depth=0.), scan(ranges=(.24,), pose=pose, angle=math.pi/4))
        belief, _ = m.occupancy_arrays()
        self.assertEqual(belief[0, 0], 0)
        self.assertEqual(belief[1, 0], 0)
        self.assertEqual(belief[1, 1], 0)
        self.assertEqual(belief[2, 1], 1)
        self.assertEqual(belief[0, 1], -1)

    def test_scan_outside_map_clips_ray_without_border_obstacle(self):
        m = mapper(shape=(3, 3), origin_xy_m=(0., 0.), resolution_m=.1)
        pose = np.eye(4)
        pose[:3, 3] = [-1., .15, .3]
        receipt = m.update(packet(depth=0.), scan(ranges=(2.,), pose=pose))
        self.assertEqual(receipt["scan_hit_rays"], 1)
        belief, _ = m.occupancy_arrays()
        np.testing.assert_array_equal(belief[1], [0, 0, 0])
        self.assertEqual(np.count_nonzero(belief == 1), 0)

    def test_scan_maximum_range_is_free_and_zero_ray_is_unknown(self):
        m = mapper(shape=(3, 3), origin_xy_m=(0., 0.))
        receipt = m.update(packet(depth=0.), scan(ranges=(0., 8.), increment=.01))
        self.assertEqual(receipt["scan_max_range_clear_rays"], 1)
        self.assertEqual(receipt["scan_invalid_zero_rays"], 1)
        self.assertEqual(m.snapshot()["observed_occupied_cells"], 0)
        self.assertGreater(m.snapshot()["observed_free_cells"], 0)

    def test_source_hash_binds_scan_to_receipt_and_copies_are_isolated(self):
        m = mapper()
        measurement = scan()
        receipt = m.update(packet(depth=0.), measurement)
        saved = m.snapshot()
        measurement.ranges_m[0] = 3.
        receipt["frame_id"] = "modified"
        belief, visible = m.occupancy_arrays()
        belief[:] = 1
        visible[:] = False
        self.assertEqual(saved, m.snapshot())
        self.assertEqual(len(saved["receipts"][0]["scan_sha256"]), 64)
        self.assertIsNone(saved["coverage_fraction"])
        self.assertAlmostEqual(saved["measured_known_area_m2"], saved["known_cells"]*.01)

    def test_rejects_skew_intrinsics_and_rich_old_packet(self):
        m = mapper()
        value = packet()
        intrinsic = value.intrinsic.copy()
        intrinsic[0, 1] = .02
        with self.assertRaisesRegex(ValueError, "zero-skew"):
            m.update(replace(value, intrinsic=intrinsic))
        with self.assertRaises(TypeError):
            m.update(value.__dict__)
        self.assertEqual(m.snapshot()["frames"], 0)

    def test_backend_failure_prevents_unjustified_retry_or_mesh_claim(self):
        m = mapper()
        class FailedBackend:
            def integrate(self, *args):
                raise RuntimeError("injected backend error")
        m._volume = FailedBackend()
        with self.assertRaisesRegex(RuntimeError, "injected"):
            m.update(packet())
        self.assertTrue(m.snapshot()["backend_poisoned"])
        self.assertEqual(m.snapshot()["frames"], 0)
        self.assertEqual(m.snapshot()["known_cells"], 0)
        with self.assertRaisesRegex(RuntimeError, "discard"):
            m.update(packet())
        with self.assertRaisesRegex(RuntimeError, "discard"):
            m.mesh_arrays()


if __name__ == "__main__":
    unittest.main()
