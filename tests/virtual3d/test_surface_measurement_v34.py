"""Analytic measurement counterexamples; no world/sensor/mapper/TSDF."""
import copy
import unittest
import numpy as np
import open3d as o3d

from nso.box_union_geometry_v30 import BoxV30, union_exterior_faces_v30
from nso.surface_measurement_v34 import SurfaceMeasurementV34, extract_observed_asset_mesh


def mesh(triangles):
    faces = np.asarray(triangles, float).reshape(-1, 3, 3)
    out = o3d.geometry.TriangleMesh()
    out.vertices = o3d.utility.Vector3dVector(faces.reshape(-1, 3))
    out.triangles = o3d.utility.Vector3iVector(np.arange(len(faces)*3).reshape(-1, 3))
    return out


def surface(faces, alternate=False):
    triangles = []
    for face in faces:
        a,b,c,d = face.vertices()
        triangles.extend(((a,b,d),(b,c,d)) if alternate else ((a,b,c),(a,c,d)))
    return mesh(triangles)


def panel(x):
    return mesh([[(x,0,0),(x,3,0),(x,3,1.6)],[(x,0,0),(x,3,1.6),(x,0,1.6)]])


def area(value):
    p = np.asarray(value.vertices)[np.asarray(value.triangles)]
    return float(np.linalg.norm(np.cross(p[:,1]-p[:,0], p[:,2]-p[:,0]),axis=1).sum()/2)


class SurfaceMeasurementV34Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.faces = union_exterior_faces_v30([BoxV30((0,2,0,3,0,1.6),0)])
        cls.sides = tuple(f for f in cls.faces if f.axis != 2)
        cls.full, cls.vertical = surface(cls.faces), surface(cls.sides)
        cls.metric = SurfaceMeasurementV34(cls.full, cls.vertical)
        cls.public_bounds = np.array([[0.,0.,0.],[2.,3.,1.6]])

    def score(self, prediction, **kwargs):
        return self.metric.evaluate(prediction,.9,returned=True,paid_actions=42,budget=42,**kwargs)

    def test_correct_vertical_only_has_full_credit_without_invented_roof(self):
        result = self.score(self.vertical)
        for tag in ('02cm','05cm','10cm'):
            self.assertEqual(result[tag],dict(precision=1.,recall=1.,f1=1.,joint=.9))
        self.assertLess(result['precision_error']['p95_m'],2e-6)
        self.assertLess(result['recall_error']['p95_m'],2e-6)
        self.assertTrue(result['eligible'])

    def test_missing_whole_side_reduces_fixed_reference_recall(self):
        prediction = surface(f for f in self.sides if not(f.axis == 0 and f.sign == 1))
        result = self.score(prediction)
        self.assertEqual(result['05cm']['precision'],1.)
        # Removed face is 4.8 / 16 m2; a small border remains within tolerance.
        self.assertGreater(result['05cm']['recall'],.69)
        self.assertLess(result['05cm']['recall'],.75)
        self.assertLess(result['05cm']['f1'],.86)
        self.assertGreater(result['recall_error']['p95_m'],.4)

    def test_extra_geometry_inside_public_roi_reduces_precision(self):
        prediction = self.vertical+panel(2.15)
        kept,audit = extract_observed_asset_mesh(prediction,self.public_bounds)
        result = self.score(kept)
        self.assertAlmostEqual(audit['outside_area_m2'],0.,places=10)
        self.assertAlmostEqual(audit['kept_area_m2'],20.8,places=10)
        self.assertAlmostEqual(result['05cm']['precision'],16/20.8,delta=.015)
        self.assertEqual(result['05cm']['recall'],1.)
        self.assertLess(result['05cm']['f1'],.9)

    def test_outside_contamination_recorded_but_not_claimed_penalized(self):
        prediction = self.vertical+panel(3.)
        kept,audit = extract_observed_asset_mesh(prediction,self.public_bounds)
        self.assertAlmostEqual(audit['outside_area_m2'],4.8,places=10)
        self.assertEqual(audit['fully_removed_by_roi_triangles'],2)
        self.assertFalse(audit['outside_roi_geometry_penalized_by_precision'])
        self.assertEqual(self.score(kept)['05cm']['f1'],1.)
        self.assertLess(self.score(prediction)['05cm']['precision'],.8)

    def test_translation_error_is_not_aligned_away(self):
        prediction = copy.deepcopy(self.vertical).translate((.12,0,0))
        result = self.score(prediction)
        self.assertLess(result['05cm']['precision'],.8)
        self.assertLess(result['05cm']['recall'],.8)
        self.assertGreater(result['precision_error']['mean_m'],.04)
        self.assertFalse(result['alignment_performed'])

    def test_retriangulation_is_numerically_consistent_for_positive_and_negative(self):
        full,vertical = surface(self.faces,True),surface(self.sides,True)
        alternate = SurfaceMeasurementV34(full,vertical)
        self.assertEqual(alternate.evaluate(vertical,.9)['05cm']['f1'],1.)
        missing = tuple(f for f in self.sides if not(f.axis == 0 and f.sign == 1))
        first = self.metric.evaluate(surface(missing),.9)
        second = alternate.evaluate(surface(missing,True),.9)
        for key in ('precision','recall','f1'):
            self.assertAlmostEqual(first['05cm'][key],second['05cm'][key],delta=.015)

    def test_empty_is_zero_with_undefined_errors_and_reference_cannot_be_empty(self):
        empty = mesh([])
        result = self.score(empty)
        self.assertTrue(result['missing'])
        self.assertEqual(result['05cm'],dict(precision=0.,recall=0.,f1=0.,joint=0.))
        self.assertIsNone(result['precision_error']['mean_m'])
        self.assertIsNone(result['recall_error']['p95_m'])
        with self.assertRaises(ValueError): SurfaceMeasurementV34(empty,self.vertical)

    def test_common_rigid_rotation_preserves_score_but_prediction_rotation_does_not(self):
        theta=.37
        rotation=np.array([[np.cos(theta),-np.sin(theta),0],[np.sin(theta),np.cos(theta),0],[0,0,1]])
        rotate=lambda x:copy.deepcopy(x).rotate(rotation,center=(1.,1.5,.8))
        rotated_metric=SurfaceMeasurementV34(rotate(self.full),rotate(self.vertical))
        self.assertEqual(rotated_metric.evaluate(rotate(self.vertical),.9)['05cm']['f1'],1.)
        self.assertLess(self.metric.evaluate(rotate(self.vertical),.9)['05cm']['f1'],.5)

    def test_ground_rule_preserves_near_ground_vertical_attachment_and_raw(self):
        ground=mesh([[(-.1,-.1,0),(2.1,-.1,0),(2.1,3.1,0)],
                     [(-.1,-.1,0),(2.1,3.1,0),(-.1,3.1,0)]])
        low=mesh([[(.5,.5,0),(.5,.7,0),(.5,.7,.015)],
                  [(.5,.5,0),(.5,.7,.015),(.5,.5,.015)]])
        raw=self.vertical+ground+low
        vertices=np.asarray(raw.vertices).copy(); triangles=np.asarray(raw.triangles).copy()
        kept,audit=extract_observed_asset_mesh(raw,self.public_bounds)
        self.assertAlmostEqual(audit['ground_removed_area_m2'],2.2*3.2,places=10)
        self.assertEqual(audit['ground_removed_triangles'],2)
        self.assertAlmostEqual(area(kept),16+.2*.015,places=10)
        np.testing.assert_array_equal(np.asarray(raw.vertices),vertices)
        np.testing.assert_array_equal(np.asarray(raw.triangles),triangles)

    def test_crossing_triangle_is_clipped_not_dropped_or_capped(self):
        raw=mesh([[(-1,0,.5),(3,0,.5),(1,0,1.5)]])
        kept,audit=extract_observed_asset_mesh(raw,self.public_bounds)
        # Two removed edge triangles have base .8, height .4: total .32 m2.
        self.assertAlmostEqual(area(kept),2-.32,places=10)
        self.assertAlmostEqual(audit['outside_area_m2'],.32,places=10)
        self.assertEqual(audit['source_triangles_with_outside_portion'],1)
        self.assertEqual(audit['fully_removed_by_roi_triangles'],0)
        self.assertTrue(np.all(np.asarray(kept.vertices)[:,1] == 0))
        self.assertFalse(audit['caps_added'])

    def test_unknown_budget_or_failure_does_not_erase_score_or_create_eligibility(self):
        unknown=self.metric.evaluate(self.vertical,1.,returned=True)
        self.assertFalse(unknown['eligible']); self.assertEqual(unknown['05cm']['f1'],1.)
        failed=self.score(self.vertical,failed=True)
        self.assertFalse(failed['eligible']); self.assertEqual(failed['05cm']['f1'],1.)
        with self.assertRaises(ValueError): self.metric.evaluate(self.vertical,float('nan'))


if __name__ == '__main__': unittest.main()
