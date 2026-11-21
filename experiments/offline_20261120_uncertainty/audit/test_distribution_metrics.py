import unittest
import numpy as np
from distribution_metrics import depth_pmf,joint_posterior,metrics,projected_truth,equal_object_summary

class MetricTests(unittest.TestCase):
    def test_point_crps_equals_absolute_error(self):
        r=metrics([.65],[1.],1.)
        self.assertAlmostEqual(r['crps_m'],.35)
        self.assertEqual(r['intervals']['0.95']['width_m'],0)
        self.assertFalse(r['intervals']['0.95']['covered'])
    def test_two_point_analytic_crps(self):
        self.assertAlmostEqual(metrics([0,1],[.5,.5],.5)['crps_m'],.25)
    def test_duplicate_particle_halving_preserves_metrics(self):
        a=metrics([.2,.6,1.],[.2,.3,.5],.7)
        b=metrics([.2,.2,.6,.6,1,1],[.1,.1,.15,.15,.25,.25],.7)
        self.assertEqual(a,b)
    def test_constant_geometry_likelihood_retains_depth_prior(self):
        q=np.array([.1,.4,.5]);E=np.tile([.002,.004],(3,1))
        for sigma in (.01,.06,.22):
            w=joint_posterior(E,q,sigma)
            np.testing.assert_allclose(w.sum(axis=1),q,atol=1e-14)
    def test_shape_prior_changes_only_declared_joint_factor(self):
        q=[.3,.7];E=np.array([[.01,.002],[.002,.01]])
        neutral=joint_posterior(E,q,.06);semantic=joint_posterior(E,q,.06,[.9,.1])
        self.assertAlmostEqual(semantic.sum(),1)
        ratios=(semantic/neutral);self.assertAlmostEqual(ratios[0,0]/ratios[0,1],9)
        self.assertAlmostEqual(ratios[1,0]/ratios[1,1],9)
    def test_projection_reports_swapped_axis_without_reassignment(self):
        vertices=np.array([[x,y,z] for x in (0,2) for y in (0,.5) for z in (0,1)])
        r=projected_truth({'object_vertices_world':vertices,'world_rotation':np.eye(3)},[1,0,0])
        self.assertEqual(r['projected_true_extent_m'],2)
        self.assertTrue(r['swapped_width_depth_axis']);self.assertEqual(r['error_to_true_depth_axis_deg'],90)
    def test_equal_object_not_equal_frame_weighting(self):
        rows=[dict(object_id='a',metrics=metrics([0],[1],1))]+[dict(object_id='b',metrics=metrics([0],[1],0)) for _ in range(3)]
        self.assertEqual(equal_object_summary(rows)['crps_m'],.5)
    def test_invalid_mass_not_silently_repaired(self):
        with self.assertRaises(ValueError):depth_pmf([.2,.5],[1,-1])
        with self.assertRaises(ValueError):joint_posterior([[float('nan'),0]],[1],.06)

if __name__=='__main__':unittest.main(verbosity=2)
