"""Full-mesh parity and positive micro-facet accounting; analytic inputs only."""
import hashlib
from pathlib import Path
import unittest

import numpy as np

from nso.complete_surface_evaluation import (LEGACY_EVALUATOR_SHA256,
    _positive_area_mesh, evaluate_complete_surface)
from nso.surface_evaluation_v40 import (CandidateViewV40, evaluate_surface_v40,
    freeze_reference_v40)


class CompleteSurfaceEvaluationTests(unittest.TestCase):
    def setUp(self):
        self.v = np.array([[0.,0.,2.],[0.,1.,2.],[1.,1.,2.],[1.,0.,2.]])
        self.t = np.array([[0,1,2],[0,2,3]])
        view = CandidateViewV40(intrinsic=[[10.,0.,30.],[0.,10.,30.],[0.,0.,1.]],
            world_from_camera=np.eye(4),width=100,height=100)
        self.ref = freeze_reference_v40(self.v,self.t,np.array([0,0]),[view],
            sample_spacing_m=.3,seed=4001,max_samples=50000)
        self.kw = dict(C_map=.8,threshold_m=.05,sample_spacing_m=.3,seed=4002,max_samples=1000000)

    def test_legacy_source_binding(self):
        path = Path(__file__).resolve().parents[1]/'nso/surface_evaluation_v40.py'
        self.assertEqual(hashlib.sha256(path.read_bytes()).hexdigest(),LEGACY_EVALUATOR_SHA256)

    def test_all_legacy_outputs_exact_on_normal_meshes(self):
        ghost = np.array([[3.,0.,2.],[3.,1.,2.],[4.,1.,2.]])
        cases = [(self.v,self.t),(self.v,self.t[:1]),
                 (self.v,np.vstack([self.t,self.t])),
                 (np.vstack([self.v,ghost]),np.vstack([self.t,[[4,5,6]]])),
                 (np.empty((0,3)),np.empty((0,3),int))]
        for v,t in cases:
            old = evaluate_surface_v40(self.ref,v,t,**self.kw)
            new = evaluate_complete_surface(self.ref,v,t,**self.kw)
            self.assertEqual(new.pop('positive_subthreshold_faces_preserved'),0)
            new.pop('prediction_mesh_validation')
            self.assertEqual(new,old)

    def test_positive_micro_facet_is_sampled_and_counted_as_false_area(self):
        tiny = np.array([[3.,0.,2.],[3.,0.0000005,2.],[3.0000005,0.,2.]])
        v,t = np.vstack([self.v,tiny]),np.vstack([self.t,[[4,5,6]]])
        with self.assertRaisesRegex(ValueError,'degenerate'):
            evaluate_surface_v40(self.ref,v,t,**self.kw)
        result = evaluate_complete_surface(self.ref,v,t,**self.kw)
        base = evaluate_complete_surface(self.ref,self.v,self.t,**self.kw)
        self.assertEqual(result['positive_subthreshold_faces_preserved'],1)
        self.assertEqual(result['submitted_prediction_triangles'],3)
        self.assertEqual(result['canonical_prediction_triangles'],3)
        self.assertEqual(result['prediction_samples'],base['prediction_samples']+1)
        self.assertGreater(result['extra_false_positive_area_m2'],0.)
        self.assertAlmostEqual(result['extra_false_positive_area_m2'],1.25e-13,delta=1e-21)

    def test_original_small_tsdf_facet_is_preserved_without_moving_vertices(self):
        v = np.array([[1.2599993022741385,3.9,.02],
                      [1.26,3.9,.020000359263769595],[1.26,3.900000804731252,.02]])
        vv,tt,_,area,_ = _positive_area_mesh(v,np.array([[0,1,2]]))
        np.testing.assert_array_equal(vv,v)
        np.testing.assert_array_equal(tt,[[0,1,2]])
        self.assertGreater(float(area[0]),0.)
        self.assertLess(float(area[0]),5e-13)

    def test_zero_area_is_rejected_not_silently_removed(self):
        with self.assertRaisesRegex(ValueError,'degenerate'):
            evaluate_complete_surface(self.ref,np.array([[0.,0.,0.],[1.,0.,0.],[2.,0.,0.]]),
                                      np.array([[0,1,2]]),**self.kw)


if __name__ == '__main__':
    unittest.main()
