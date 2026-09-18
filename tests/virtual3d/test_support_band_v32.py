"""Independent geometry identities for the evaluation-only tolerance band."""
import unittest
import numpy as np
import shapely
from shapely.geometry import GeometryCollection, LineString, Point, box
from nso.support_band_v32 import support_band_v32, compare_support_band_v32


class SupportBandV32Tests(unittest.TestCase):
    def test_segment_band_has_capsule_area(self):
        band = support_band_v32(LineString([(0, 0), (2, 0)]), .05)
        # Polygonized round caps approximate pi with64 edges.
        expected = .2 + np.pi*.05**2
        self.assertLess(abs(band.area-expected), 2e-5)

    def test_nearby_bands_can_merge_without_connecting_inputs(self):
        lines = shapely.union_all([LineString([(0, 0), (1, 0)]),
                                   LineString([(0, .08), (1, .08)])])
        before = lines.wkb
        self.assertEqual(len(lines.geoms), 2)
        self.assertEqual(support_band_v32(lines, .05).geom_type, 'Polygon')
        self.assertEqual(lines.wkb, before)

    def test_empty_prediction_has_zero_not_nan(self):
        empty = GeometryCollection()
        p = dict(area=empty, support=empty, receipt={})
        r = compare_support_band_v32(p, box(0, 0, 2, 3))
        for tag in ('02cm', '05cm', '10cm'):
            self.assertEqual(r[tag]['quality'], 0.)
            self.assertEqual(r[tag]['missed_reference_support_m'], 10.)

    def test_band_retains_interior_hole(self):
        band = support_band_v32(box(0, 0, 2, 3).boundary, .05)
        self.assertFalse(band.covers(Point(1, 1.5)))
        self.assertTrue(band.covers(Point(.02, 1.5)))
        self.assertEqual(len(band.interiors), 1)

    def test_radius_requires_positive_finite_number(self):
        for value in (0., -1., float('nan'), float('inf'), True):
            with self.assertRaises(ValueError):
                support_band_v32(LineString([(0, 0), (1, 0)]), value)


if __name__ == '__main__':
    unittest.main()
