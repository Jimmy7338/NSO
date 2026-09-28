"""V5 changes sensed support and shared objectness, never truth or candidate pools."""
from dataclasses import asdict, replace
from types import SimpleNamespace
import unittest
import numpy as np

from env.virtual3d import VirtualConfig
from nso.semantic_completion_v3 import SemanticHistoryMapperV3
from nso.semantic_completion_v5 import CompletionConfigV5, ObjectCompletionModel, support_geometry
from utils.inspection_benchmark import InspectionWorld


class QualityFixture:
    def __init__(self, mode, marker_domain=True):
        self.shape = (30, 40)
        values = asdict(VirtualConfig(voxel_m=.03, depth_sigma_m=0., dropout=0.))
        if marker_domain:
            values.update(semantic_source='rgb_marker', appearance='marked')
        self.config = SimpleNamespace(**values)
        self.keyframes = []
        self.quality = {}
        for cluster_x, labeled in ((4., True), (1.5, False)):
            for y in np.linspace(2.5, 3.15, 6):
                for z in np.linspace(.3, 1.2, 6):
                    point = np.array([cluster_x, y, z])
                    category = 2 if labeled else 0
                    if mode == 'shuffled' and category: category = 3
                    if mode == 'absent': category = 0
                    self.quality[tuple(np.floor(point/.15).astype(int))] = dict(
                        point=point, normal=np.array([1., 0., 0.]), n=1, bits=1,
                        label=category, information=.1, best_range=2., residual=.001,
                        normal_dispersion=0.)

    def quality_evidence(self, max_points=1600):
        rows = list(self.quality.values())
        if len(rows) > max_points:
            rows = [rows[i] for i in np.linspace(0, len(rows)-1, max_points, dtype=int)]
        return {key: np.asarray([row[key] for row in rows]) for key in rows[0]}


class SemanticCompletionV5Tests(unittest.TestCase):
    def test_actual_missing_pathway_recovers_geometry(self):
        g = ObjectCompletionModel(QualityFixture('aligned'), False)
        m = ObjectCompletionModel(QualityFixture('absent'), True)
        np.testing.assert_array_equal(g.points, m.points)
        np.testing.assert_array_equal(g.weights, m.weights)
        self.assertFalse(m.marker_objectness_active)

    def test_all_scorers_have_identical_geometry_and_candidates(self):
        mapper = QualityFixture('aligned')
        models = [ObjectCompletionModel(mapper, False),
                  ObjectCompletionModel(mapper, True, fine_categories=False),
                  ObjectCompletionModel(mapper, True),
                  ObjectCompletionModel(QualityFixture('shuffled'), True),
                  ObjectCompletionModel(QualityFixture('absent'), True)]
        safe, distances = np.ones(mapper.shape, bool), np.ones(mapper.shape)
        for other in models[1:]:
            np.testing.assert_array_equal(models[0].points, other.points)
            np.testing.assert_array_equal(models[0].hypothesis_ids, other.hypothesis_ids)
            self.assertEqual(models[0].candidates(safe, distances), other.candidates(safe, distances))
            for a, b in zip(models[0].objects, other.objects):
                np.testing.assert_array_equal(a['center'], b['center'])
                np.testing.assert_array_equal(a['dims'], b['dims'])
                self.assertEqual(a['fit_errors_m2'], b['fit_errors_m2'])

    def test_objectness_rule_is_shared_by_binary_and_fine_semantics(self):
        mapper = QualityFixture('aligned')
        g = ObjectCompletionModel(mapper, False)
        o = ObjectCompletionModel(mapper, True, fine_categories=False)
        s = ObjectCompletionModel(mapper, True)
        self.assertTrue(o.marker_objectness_active)
        self.assertEqual(len(o.objects), 2)
        for i, (go, oo, so) in enumerate(zip(g.objects, o.objects, s.objects)):
            self.assertEqual(oo['object_probability'], so['object_probability'])
            expected = 1. if oo['marker_supported'] else .25
            self.assertEqual(oo['object_probability'], expected)
            self.assertEqual(go['object_probability'], 1.)
            mask = (g.hypothesis_ids//2) == i
            np.testing.assert_allclose(o.weights[mask], g.weights[mask]*expected, atol=1e-12, rtol=0)
        self.assertTrue(any(a['shelf_prior_probability'] != b['shelf_prior_probability']
                            for a, b in zip(o.objects, s.objects)))

    def test_nonmarker_domain_does_not_suppress_unmarked_geometry(self):
        mapper = QualityFixture('aligned', marker_domain=False)
        g = ObjectCompletionModel(mapper, False)
        o = ObjectCompletionModel(mapper, True, fine_categories=False)
        self.assertFalse(o.marker_objectness_active)
        np.testing.assert_array_equal(g.weights, o.weights)

    def test_measured_front_plane_is_not_shifted_by_minimum_extent_prior(self):
        points = np.array([[4., y, z] for y in np.linspace(2., 3.2, 8)
                           for z in np.linspace(.25, 1.2, 7)])
        normals = np.tile([1., 0., 0.], (len(points), 1))
        center, dims, audit = support_geometry(points, normals, np.floor(points/.15).astype(int),
                                                np.array([-1., 0.]), CompletionConfigV5())
        self.assertAlmostEqual((center-dims/2)[0], 4.)
        self.assertGreaterEqual(dims[0], .9)
        self.assertFalse(audit['axes'][0]['opposing_faces_observed'])
        self.assertTrue(audit['axes'][0]['minimum_extent_prior_used'])
        # The tangent extent includes actual measured support, with padding
        # derived from the quality voxel rather than 5/95 quantile shrinkage.
        self.assertLessEqual((center-dims/2)[1], points[:, 1].min())
        self.assertGreaterEqual((center+dims/2)[1], points[:, 1].max())

    def test_observed_opposing_planes_override_incomplete_shape_size_prior(self):
        points, normals = [], []
        for x, sign in ((2., 1.), (2.7, -1.)):
            for y in np.linspace(1.5, 2.1, 5):
                for z in np.linspace(.3, 1.2, 5):
                    points.append([x, y, z]); normals.append([sign, 0., 0.])
        points, normals = np.asarray(points), np.asarray(normals)
        center, dims, audit = support_geometry(points, normals, np.floor(points/.15).astype(int),
                                                np.array([-1., 0.]), CompletionConfigV5())
        self.assertAlmostEqual(dims[0], .7)
        self.assertAlmostEqual(center[0], 2.35)
        self.assertTrue(audit['axes'][0]['opposing_faces_observed'])
        self.assertFalse(audit['axes'][0]['minimum_extent_prior_used'])

    def test_true_voxel_key_alignment_is_required(self):
        mapper = QualityFixture('aligned')
        original = mapper.quality_evidence
        mapper.quality_evidence = lambda max_points: {key: value[::-1] for key, value in original(max_points).items()}
        with self.assertRaisesRegex(ValueError, 'voxel-key'):
            ObjectCompletionModel(mapper, True)

    def test_real_sensor_mapper_is_not_modified(self):
        config = VirtualConfig(voxel_m=.03, depth_sigma_m=0., dropout=0.)
        world = InspectionWorld(config, 223, 'box')
        pose, _ = world.pose(1.4, -90)
        frame = world.capture(pose, 0, 'none')
        mapper = SemanticHistoryMapperV3((30, 40), config)
        mapper.update(frame)
        vertices = np.asarray(mapper.mesh().vertices).copy()
        triangles = np.asarray(mapper.mesh().triangles).copy()
        quality = {key: value.copy() for key, value in mapper.quality_evidence().items()}
        ObjectCompletionModel(mapper, True)
        np.testing.assert_array_equal(vertices, np.asarray(mapper.mesh().vertices))
        np.testing.assert_array_equal(triangles, np.asarray(mapper.mesh().triangles))
        for key, value in quality.items():
            np.testing.assert_array_equal(value, mapper.quality_evidence()[key])


if __name__ == '__main__':
    unittest.main()
