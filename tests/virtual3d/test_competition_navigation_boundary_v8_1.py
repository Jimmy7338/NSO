import unittest
from types import SimpleNamespace
import numpy as np

from nso.competition_candidates_v8_1 import navigation_rear_boundary


class NavigationRearBoundaryTests(unittest.TestCase):
    def fixture(self, back=(0., 1.)):
        mapper = SimpleNamespace(config=SimpleNamespace(resolution_m=.2),
                                 shape=(20, 20), belief=np.zeros((20, 20), dtype=int))
        asset = {'observed_low': np.array([1.2, 1.2, .35]),
                 'observed_high': np.array([1.6, 1.6, 1.6]),
                 'rear_boundary_xy': np.array([1.4, 1.6]) if back == (0., 1.) else np.array([1.2, 1.4]),
                 'back_axis': np.array(back), 'class_vote': -1., 'marked_points': 20}
        return mapper, asset

    def cell(self, mapper, x, y, value=1):
        r = mapper.shape[0] - 1 - int(np.floor(y / .2)); c = int(np.floor(x / .2))
        mapper.belief[r, c] = value

    def test_full_cell_edge_is_half_resolution_beyond_occupied_center(self):
        mapper, asset = self.fixture(); self.cell(mapper, 1.5, 1.7)
        original = asset['rear_boundary_xy'].copy()
        np.testing.assert_allclose(navigation_rear_boundary(mapper, asset), [1.4, 1.8], atol=1e-12)
        np.testing.assert_array_equal(asset['rear_boundary_xy'], original)

    def test_far_wall_and_other_asset_do_not_extend_local_group(self):
        mapper, asset = self.fixture(); self.cell(mapper, 1.5, 1.7)
        expected = navigation_rear_boundary(mapper, asset)
        self.cell(mapper, 3.9, 3.9); self.cell(mapper, 2.7, 1.7)
        np.testing.assert_array_equal(navigation_rear_boundary(mapper, asset), expected)

    def test_unknown_and_free_cells_do_not_extend_measured_boundary(self):
        mapper, asset = self.fixture(); self.cell(mapper, 1.5, 1.7, -1)
        np.testing.assert_array_equal(navigation_rear_boundary(mapper, asset), asset['rear_boundary_xy'])
        self.cell(mapper, 1.5, 1.7, 0)
        np.testing.assert_array_equal(navigation_rear_boundary(mapper, asset), asset['rear_boundary_xy'])

    def test_negative_cardinal_axis_and_labels_do_not_change_geometry_rule(self):
        mapper, asset = self.fixture(back=(-1., 0.)); self.cell(mapper, 1.1, 1.5)
        expected = navigation_rear_boundary(mapper, asset)
        np.testing.assert_allclose(expected, [1., 1.4], atol=1e-12)
        asset.update(class_vote=1., marked_points=0, physical_class='unavailable_to_rule')
        np.testing.assert_array_equal(navigation_rear_boundary(mapper, asset), expected)


if __name__ == '__main__':
    unittest.main()
