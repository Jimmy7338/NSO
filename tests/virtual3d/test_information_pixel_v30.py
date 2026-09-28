"""Geometry/calibration contracts only; no project world or TSDF construction."""
import unittest
import numpy as np
from env.information_pixel_v30 import scene_boxes, swept_clear, SHIFT
from env.canonical_rgbd_v15 import render_axial_depth
from nso.box_union_geometry_v30 import BoxV30


class PixelContractV30(unittest.TestCase):
    def test_sweep_detects_between_endpoints(self):
        box = BoxV30((.45, .55, -.1, .1, 0., 1.))
        self.assertTrue(swept_clear((0., 0.), (0., 0.), [box], .1))
        self.assertTrue(swept_clear((1., 0.), (1., 0.), [box], .1))
        self.assertFalse(swept_clear((0., 0.), (1., 0.), [box], .1))
        self.assertTrue(swept_clear((0., .3), (1., .3), [box], .1))

    def test_floor_not_navigation_obstacle(self):
        floor = BoxV30((-2., 2., -2., 2., -.1, 0.))
        self.assertTrue(swept_clear((0., 0.), (1., 0.), [floor]))

    def test_shift_preserves_body_dimensions(self):
        _, _, facilities = scene_boxes(0)
        bounds = np.asarray(facilities[0][0].bounds).reshape(3, 2)
        np.testing.assert_allclose(bounds[:, 1]-bounds[:, 0], [2., 3., 1.6])
        np.testing.assert_allclose(bounds[:, 0]-SHIFT, [-4., 2., 0.])

    def test_axis_depth_is_not_euclidean_range(self):
        # Synthetic infinite-looking front slab, no project scene acquisition.
        k = np.array([[2., 0., 1.], [0., 2., 1.], [0., 0., 1.]])
        depth, points = render_axial_depth(np.array([[-10., -10., 3., 20., 20., .1]]),
            k, np.eye(4), 3, 3, 4.)
        np.testing.assert_allclose(depth, 3., atol=3e-7)
        self.assertAlmostEqual(float(np.linalg.norm(points[0, 0])), 3*np.sqrt(1.5), places=6)

    def test_collision_touch_is_rejected(self):
        box = BoxV30((.2, 1., -1., 1., 0., 2.))
        self.assertFalse(swept_clear((0., 0.), (0., 0.), [box], .2))


if __name__ == '__main__':
    unittest.main()
