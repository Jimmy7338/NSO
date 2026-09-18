import copy
import unittest
import numpy as np
from nso.observed_asset_axes_v16 import reorient_observed_asset


def asset():
    return dict(observed_low=[4., 3.6, .2], observed_high=[5.2, 3.61, 1.4],
        normal_out_xy=[0., 1.], normal_sign_depth_support=131,
        orientation_available=True, normal_fallback=False,
        front_axis=[-1., 0.], class_vote=-1., marked_points=15)


class ObservedAxesTest(unittest.TestCase):
    def test_side_facing_facility_uses_measured_face(self):
        a = asset(); original = copy.deepcopy(a); out = reorient_observed_asset(a)
        np.testing.assert_array_equal(out['front_axis'], [0., 1.])
        self.assertAlmostEqual(out['measured_width_m'], 1.2)
        self.assertAlmostEqual(out['measured_depth_m'], .01)
        self.assertAlmostEqual(out['rear_boundary_xy'][1], 3.6)
        self.assertEqual(a, original)

    def test_category_and_obsolete_front_do_not_control_axes(self):
        a = asset(); b = asset(); b.update(class_vote=1., marked_points=0, front_axis=[1., 0.])
        x, y = reorient_observed_asset(a), reorient_observed_asset(b)
        for k in ('front_axis', 'back_axis', 'side_axis', 'rear_boundary_xy'):
            np.testing.assert_array_equal(x[k], y[k])
        self.assertEqual(x['class_vote'], -1.); self.assertEqual(y['class_vote'], 1.)

    def test_cardinal_rotation_preserves_width_and_depth(self):
        a = asset(); b = asset()
        b.update(observed_low=[-3.61, 4., .2], observed_high=[-3.6, 5.2, 1.4], normal_out_xy=[-1., 0.])
        x, y = reorient_observed_asset(a), reorient_observed_asset(b)
        for k in ('measured_width_m', 'measured_depth_m', 'measured_height_m'):
            self.assertAlmostEqual(x[k], y[k])
        np.testing.assert_allclose(y['rear_boundary_xy'], [-x['rear_boundary_xy'][1], x['rear_boundary_xy'][0]])

    def test_uncertain_or_invalid_direction_is_explicit(self):
        for change in (dict(orientation_available=False), dict(normal_out_xy=[0., 0.]),
                       dict(normal_sign_depth_support=0), dict(observed_high=[1., 2., 3.])):
            a = asset(); a.update(change)
            with self.assertRaises(ValueError): reorient_observed_asset(a)
        a = asset(); a.update(normal_out_xy=[1., 1.], normal_fallback=True)
        out = reorient_observed_asset(a)
        self.assertTrue(out['axis_evidence']['discrete_axis_tie'])
        self.assertEqual(out['axis_evidence']['source'], 'observed_camera_direction_fallback')


if __name__ == '__main__': unittest.main()
