import unittest
import numpy as np
import open3d as o3d
from env.facility_documentation_v18 import FacilityWorldV18
from utils.facility_metrics_v18 import FacilityEvaluatorV18


class FacilityV18Tests(unittest.TestCase):
    def test_initial_geometry_and_neutral_rgb_pair_with_opposite_visible_cues(self):
        for parent in ('D18-P00','D18-P01'):
            a=FacilityWorldV18(parent,'A_open_B_closed');b=FacilityWorldV18(parent,'A_closed_B_open')
            fa,fb=a.sense(),b.sense()
            np.testing.assert_array_equal(fa.depth_m,fb.depth_m)
            np.testing.assert_array_equal(a.scan().ranges_m,b.scan().ranges_m)
            np.testing.assert_array_equal(fa.semantic>0,fb.semantic>0)
            marker=fa.semantic>0;self.assertTrue(marker.any())
            np.testing.assert_array_equal(fa.color_rgb[~marker],fb.color_rgb[~marker])
            np.testing.assert_array_equal(fa.semantic[marker]+fb.semantic[marker],np.full(marker.sum(),5))
            self.assertEqual({x['category'] for x in a.objects},{2,3})
            self.assertFalse(np.array_equal(np.asarray(a.mesh.triangles),np.asarray(b.mesh.triangles)))

    def test_missing_instances_remain_in_denominator_and_truth_is_nearly_perfect(self):
        world=FacilityWorldV18();e=FacilityEvaluatorV18(world,count_per_asset=300)
        empty=e.evaluate(o3d.geometry.TriangleMesh(),.9,returned=True)
        self.assertEqual(empty['05cm']['joint_asset'],0)
        one=e.evaluate(world.instance_mesh(0),.9,returned=True)
        self.assertEqual(one['mission_asset_count'],2)
        self.assertEqual(one['instances'][1]['05cm']['f1'],0)
        self.assertAlmostEqual(one['05cm']['asset_macro_f1'],.5,places=6)
        perfect=e.evaluate(world.mesh,.9,returned=True)
        self.assertGreater(perfect['05cm']['asset_macro_f1'],.99)
        self.assertEqual(perfect['05cm']['completion_fraction'],1.)
        failed=e.evaluate(world.mesh,.7,returned=True)
        self.assertFalse(failed['eligible']);self.assertGreater(failed['05cm']['joint_asset'],0)


if __name__=='__main__':unittest.main()
