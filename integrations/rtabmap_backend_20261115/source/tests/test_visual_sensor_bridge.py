"""Pure data tests: no ROS master, camera, or CAN interaction."""
import importlib.util
from pathlib import Path
import sys
import unittest

import numpy as np


SOURCE = Path(__file__).resolve().parents[1]/'runtime/visual_sensor_bridge.py'
spec = importlib.util.spec_from_file_location('visual_sensor_bridge_under_test', str(SOURCE))
bridge = importlib.util.module_from_spec(spec)
spec.loader.exec_module(bridge)


class VisualDepthTests(unittest.TestCase):
    def test_import_has_no_ros_side_effects(self):
        self.assertNotIn('rospy', sys.modules)
        self.assertNotIn('tf2_ros', sys.modules)

    def setUp(self):
        self.k = np.array([[80., 0., 39.5], [0., 80., 29.5], [0., 0., 1.]])
        self.depth = np.full((60, 80), 2., dtype=np.float32)
        self.projector = bridge.DepthProjector(stride=1)

    def test_nan_depth_is_unknown_and_cannot_be_healthy(self):
        result = self.projector.project(np.full_like(self.depth, np.nan), self.k)
        self.assertFalse(result['valid'])
        self.assertTrue(np.isnan(result['ranges']).all())
        self.assertFalse(np.isinf(result['ranges']).any())

    def test_front_real_obstacle_has_measured_endpoint(self):
        result = self.projector.project(self.depth, self.k)
        self.assertTrue(result['valid'])
        measured = result['ranges'][np.isfinite(result['ranges'])]
        self.assertTrue(len(measured) > 0)
        self.assertGreater(float(measured.min()), 2.)
        self.assertLess(float(measured.min()), 2.1)
        self.assertTrue(np.isnan(result['ranges']).any())
        self.assertFalse(np.isinf(result['ranges']).any())

    def test_sparse_supported_real_depth_and_obstacle_are_healthy(self):
        # Only 128 / 4800 pixels (2.67%) contain observations. Unobserved
        # ground/image regions do not make these real obstacle samples invalid.
        sparse = np.full_like(self.depth, np.nan)
        rows = np.linspace(12, 47, 8).astype(int)
        cols = np.linspace(20, 59, 16).astype(int)
        sparse[np.ix_(rows, cols)] = 2.
        result = self.projector.project(sparse, self.k)
        self.assertTrue(result['valid'])
        self.assertLess(result['statistics']['valid_fraction'], .1)
        self.assertEqual(result['statistics']['valid_pixels'], 128)
        self.assertGreater(result['statistics']['obstacle_points'], 0)
        self.assertTrue(np.isfinite(result['ranges']).any())
        self.assertTrue(np.isnan(result['ranges']).any())
        self.assertFalse(np.isinf(result['ranges']).any())
        self.assertFalse(result['statistics']['whole_image_completeness_required'])

    def test_one_bad_sensor_line_cannot_claim_spatial_depth_health(self):
        line = np.full((60, 160), np.nan, dtype=np.float32)
        line[30, :] = 2.
        result = self.projector.project(line, self.k)
        self.assertFalse(result['valid'])
        self.assertEqual(result['reason'], 'insufficient_depth_spatial_support')
        self.assertTrue(np.isnan(result['ranges']).all())

    def test_too_few_valid_points_cannot_claim_depth_health(self):
        sparse = np.full_like(self.depth, np.nan)
        sparse[20:30, 30:40] = 2.
        result = self.projector.project(sparse, self.k)
        self.assertFalse(result['valid'])
        self.assertEqual(result['reason'], 'insufficient_finite_depth')

    def test_approved_mask_rejects_only_exact_pixels_as_unknown(self):
        mask = np.zeros_like(self.depth, dtype=bool)
        mask[20:40, 30:50] = True
        raw = self.depth.copy()
        projector = bridge.DepthProjector(self_mask=mask, stride=1)
        result = projector.project(self.depth, self.k)
        self.assertTrue(np.array_equal(raw, self.depth))
        self.assertTrue(np.isnan(result['filtered_depth'][mask]).all())
        self.assertTrue(np.isfinite(result['filtered_depth'][~mask]).all())
        self.assertFalse(result['statistics']['unknown_is_free'])

    def test_invalid_depth_never_turns_into_infinity(self):
        self.depth[:20, :] = np.inf
        self.depth[20:30, :] = 0.
        result = self.projector.project(self.depth, self.k)
        self.assertTrue(result['valid'])
        self.assertFalse(np.isinf(result['ranges']).any())
        self.assertTrue(np.isnan(result['filtered_depth'][:30]).all())

    def test_calibration_change_is_rejected(self):
        self.projector.project(self.depth, self.k)
        changed = self.k.copy(); changed[0, 0] += 1.
        with self.assertRaisesRegex(ValueError, 'calibration_changed'):
            self.projector.project(self.depth, changed)

    def test_depth_decoding_row_padding_and_big_endian(self):
        pixels = np.array([[1., np.nan], [2., 3.]], dtype='>f4')
        raw = pixels[0].tobytes()+b'1234'+pixels[1].tobytes()+b'5678'
        decoded = bridge.depth_array(raw, 2, 2, 12, '32FC1', True)
        self.assertTrue(np.allclose(decoded, pixels, equal_nan=True))
        decoded_mm = bridge.depth_array(np.array([0, 1000, 2000], dtype='<u2').tobytes(),
                                       3, 1, 6, '16UC1')
        self.assertTrue(np.isnan(decoded_mm[0, 0]))
        self.assertEqual(decoded_mm[0, 1], 1.)

    def test_deployed_approved_mask_hash_and_geometry(self):
        mask, receipt = bridge.load_approved_mask()
        self.assertEqual(mask.shape, (720, 1280))
        self.assertEqual(int(mask.sum()), 46499)
        self.assertFalse(receipt['excluded_is_free'])
        # Actual RGB-reviewed IMU side and curved upper shoulder; never mask
        # the separate gray-floor failures merely because their depth is bad.
        for u, v in [(608, 712), (664, 656), (672, 656), (680, 656), (600, 696)]:
            self.assertTrue(mask[v, u])
        for u, v in [(336, 704), (512, 672), (320, 680)]:
            self.assertFalse(mask[v, u])


if __name__ == '__main__':
    unittest.main()
