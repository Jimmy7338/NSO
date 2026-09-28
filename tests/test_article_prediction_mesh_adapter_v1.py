import unittest

import numpy as np

from nso.article_prediction_mesh_adapter_v1 import prepare_prediction_mesh_v1, array_sha256
from nso.surface_evaluation_v40 import _mesh


class PredictionNumericAdapterTests(unittest.TestCase):
    def test_frozen_threshold_and_order_without_surface_evaluation(self):
        vertices=np.array([[0.,0,0],[1.,0,0],[0,2e-13,0],[0,1e-12,0],[0,2e-12,0],[0,1.,0]])
        faces=np.array([[0,1,2],[0,1,4],[0,1,3],[0,1,5]],dtype=np.int32)
        original_vertices=vertices.copy(); original_faces=faces.copy()
        with self.assertRaisesRegex(ValueError,'degenerate'):_mesh(vertices,faces,allow_empty=True)
        v,t,report=prepare_prediction_mesh_v1(vertices,faces)
        self.assertIs(v,vertices)
        np.testing.assert_array_equal(t,faces[[1,3]])
        np.testing.assert_array_equal(vertices,original_vertices);np.testing.assert_array_equal(faces,original_faces)
        self.assertEqual(t.dtype,faces.dtype)
        self.assertEqual(report['removed_original_face_indices'],[0,2])
        self.assertEqual(report['new_surface_evaluations'],0)
        _mesh(v,t,allow_empty=True)  # Validation only; never calls _sample/evaluate_surface_v40.

    def test_zero_removals_return_exact_same_arrays(self):
        v=np.array([[0.,0,0],[1.,0,0],[0.,1.,0]],dtype=np.float32)
        t=np.array([[2,1,0]],dtype=np.uint32)
        actual_v,actual_t,report=prepare_prediction_mesh_v1(v,t)
        self.assertIs(actual_v,v);self.assertIs(actual_t,t)
        self.assertTrue(report['prediction_arrays_exactly_unchanged'])
        self.assertEqual(report['adapted_triangles_array_sha256'],array_sha256(t))
        self.assertEqual(report['removed_maximum_edge_m'],0.)

    def test_invalid_geometry_is_rejected_instead_of_silently_repaired(self):
        v=np.array([[0.,0,0],[1.,0,0],[0.,1.,0]])
        for invalid in (np.array([[0.,1.,2.]]),np.array([[0,1,3]]),np.array([[0,1,-1]]),np.array([0,1,2])):
            with self.subTest(invalid=invalid),self.assertRaises(ValueError):prepare_prediction_mesh_v1(v,invalid)
        for value in (np.nan,np.inf):
            bad=v.copy();bad[0,0]=value
            with self.assertRaises(ValueError):prepare_prediction_mesh_v1(bad,np.array([[0,1,2]]))
        huge=np.array([[0.,0,0],[1e308,0,0],[0,1e308,0]])
        with self.assertRaisesRegex(ValueError,'nonfinite triangle area'):prepare_prediction_mesh_v1(huge,np.array([[0,1,2]]))

    def test_cell_numerical_face_fixture(self):
        v=np.array([[4.780001982228847,2.42,1.02],[4.779999999999999,2.4200001382927856,1.02],
            [4.779999999999999,2.42,1.0199995966074467],[0.,0,0],[1.,0,0],[0,1.,0]])
        t=np.array([[0,1,2],[3,4,5]],dtype=np.int32)
        _,filtered,report=prepare_prediction_mesh_v1(v,t)
        self.assertEqual(report['removed_faces'],1)
        self.assertEqual(report['removed_total_area_m2'],4.235695216559833e-13)
        self.assertEqual(report['removed_maximum_edge_m'],2.0228585607699087e-6)
        np.testing.assert_array_equal(filtered,t[1:])

    def test_empty_and_degenerate_only_predictions_have_explicit_zero_area(self):
        for vertices,faces in [(np.empty((0,3)),np.empty((0,3),dtype=np.int64)),
            (np.zeros((3,3)),np.array([[0,1,2]]))]:
            v,t,report=prepare_prediction_mesh_v1(vertices,faces)
            self.assertEqual(len(t),0);self.assertEqual(report['retained_total_area_m2'],0.)
            _mesh(v,t,allow_empty=True)


if __name__=='__main__':unittest.main()
