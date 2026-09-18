import copy
from dataclasses import replace
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import open3d as o3d

from env.virtual3d_v2 import VirtualConfigV2
from utils.facility_metrics_v19 import FacilityEvaluatorV19, clip_mesh_to_bounds


class SixSolidFixture:
    """Small independent six-asset world for evaluator behavior, not a pilot."""
    def __init__(self):
        self.config = VirtualConfigV2(width_m=9., height_m=8., max_depth_m=5.)
        self.shape = (40, 45)
        self.reachable = np.ones(self.shape, bool)
        self.reachable[:2] = False; self.reachable[-2:] = False
        self.reachable[:, :2] = False; self.reachable[:, -2:] = False
        self.mesh = o3d.geometry.TriangleMesh.create_box(9., 8., .1).translate((0, 0, -.1))
        self.triangle_classes = [1] * len(self.mesh.triangles)
        self.objects = []; self.meshes = []
        for identifier, (x, y) in enumerate([(1., 2.), (4., 2.), (7., 2.), (1., 5.), (4., 5.), (7., 5.)]):
            mesh = o3d.geometry.TriangleMesh.create_box(.8, .6, .9).translate((x, y, 0))
            self.mesh += mesh; self.triangle_classes.extend([2] * len(mesh.triangles)); self.meshes.append(mesh)
            self.objects.append(dict(id=identifier, owner=100 + identifier, category=2,
                evaluation_bounds=[[x - .2, y - .2, .01], [x + 1., y + .8, 1.3]]))
            for r, col in np.argwhere(self.reachable):
                xx = (col + .5) * .2; yy = (self.shape[0] - r - .5) * .2
                if x - .3 < xx < x + 1.1 and y - .3 < yy < y + .9:
                    self.reachable[r, col] = False
        self.triangle_classes = np.asarray(self.triangle_classes)
        c = self.config; focal = c.width_px / (2 * np.tan(np.deg2rad(c.fov_deg / 2)))
        self.intrinsic = np.asarray([[focal, 0, (c.width_px - 1) / 2], [0, focal, (c.height_px - 1) / 2], [0, 0, 1.]])

    def instance_mesh(self, identifier):
        return copy.deepcopy(self.meshes[identifier])


class FacilityMetricsV19Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.world = SixSolidFixture()
        cls.evaluator = FacilityEvaluatorV19(cls.world, count_per_asset=240, predicted_per_asset=1200, global_count=800)

    def test_all_six_count_even_when_missing_and_failure_prevents_eligibility(self):
        evaluator = self.evaluator; world = self.world
        missing = evaluator.evaluate(o3d.geometry.TriangleMesh(), .9, returned=True)
        self.assertEqual(missing['mission_asset_count'], 6)
        self.assertEqual(missing['missing_asset_count'], 6)
        self.assertEqual(missing['05cm']['joint_external'], 0.)
        self.assertIsNone(missing['continuous_external']['accuracy']['complete_mission_macro_mean_m'])
        one = evaluator.evaluate(world.instance_mesh(0), .9, returned=True)
        self.assertAlmostEqual(one['05cm']['external_macro_f1'], 1 / 6, places=6)
        self.assertEqual(one['instances'][1]['05cm']['recall'], 0.)
        truth = evaluator.evaluate(world.mesh, .9, returned=True)
        self.assertGreater(truth['05cm']['external_macro_f1'], .999)
        self.assertEqual(truth['05cm']['completion_fraction'], 1.)
        self.assertTrue(truth['eligible'])
        self.assertFalse(evaluator.evaluate(world.mesh, .9, returned=True, failed=True)['eligible'])
        self.assertFalse(evaluator.evaluate(world.mesh, .79, returned=True)['eligible'])
        self.assertFalse(evaluator.evaluate(world.mesh, .9, returned=True, collisions=1)['eligible'])
        for row in truth['instances']:
            self.assertLess(row['surface_error_rmse_m'], 1e-5)
            self.assertLess(row['dimensions']['mean_absolute_size_error_m'], .025)
            self.assertTrue(row['silhouette']['views'])
            self.assertGreater(row['silhouette']['mean_iou'], .99)
        json.dumps(truth, allow_nan=False)

    def test_false_geometry_lowers_precision_and_crossing_triangles_are_not_lost(self):
        false = copy.deepcopy(self.world.mesh)
        false += o3d.geometry.TriangleMesh.create_box(.6, .5, .1).translate((1.05, 2.05, 1.15))
        result = self.evaluator.evaluate(false, .9, returned=True)
        self.assertLess(result['instances'][0]['05cm']['precision'], .95)
        self.assertGreater(result['instances'][0]['surface_error_rmse_m'], .05)
        self.assertGreater(result['instances'][0]['dimensions']['absolute_size_error_xyz_m'][2], .3)
        crossing = o3d.geometry.TriangleMesh()
        crossing.vertices = o3d.utility.Vector3dVector([[-2, -2, .5], [3, -2, .5], [-2, 3, .5]])
        crossing.triangles = o3d.utility.Vector3iVector([[0, 1, 2]])
        clipped = clip_mesh_to_bounds(crossing, [[0, 0, 0], [1, 1, 1]])
        self.assertAlmostEqual(clipped.get_surface_area(), .5, places=8)
        self.assertTrue((np.asarray(clipped.vertices) >= 0).all())
        self.assertTrue((np.asarray(clipped.vertices) <= 1).all())

    def test_reference_cache_reuses_fixed_views_and_rejects_changed_contract(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / 'references.npz'
            original = FacilityEvaluatorV19(self.world, count_per_asset=100, global_count=300, reference_cache=path)
            first = original.evaluate(self.world.mesh, .9, returned=True)
            with patch('utils.facility_metrics_v19.observable_reference', side_effect=AssertionError('recomputed reference')):
                cached = FacilityEvaluatorV19(self.world, count_per_asset=100, global_count=300, reference_cache=path)
            self.assertEqual(first, cached.evaluate(self.world.mesh, .9, returned=True))
            modified = SixSolidFixture()
            modified.config = replace(modified.config, max_depth_m=4.)
            with self.assertRaisesRegex(ValueError, 'cache does not match'):
                FacilityEvaluatorV19(modified, count_per_asset=100, global_count=300, reference_cache=path)


if __name__ == '__main__':
    unittest.main()
