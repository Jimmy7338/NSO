"""Analytic geometry/provenance checks; no actual scene, replay or World."""
from copy import deepcopy
from pathlib import Path
import unittest
from unittest.mock import patch

import numpy as np

from nso.surface_evaluation_v40 import CandidateViewV40, freeze_reference_v40
from scripts import supplement_semantic_continuous_distance as module
from tests import test_reuse_semantic_scene_endpoint_evaluation as fixtures


def analytic_reference():
    vertices, triangles, owners = [], [], []
    for x, width, owner in ((0., 1., 0), (3., 2., 1), (7., 3., -1)):
        start = len(vertices)
        vertices.extend([[x, 0., 2.], [x, 1., 2.], [x+width, 1., 2.], [x+width, 0., 2.]])
        triangles.extend([[start, start+1, start+2], [start, start+2, start+3]])
        owners.extend([owner, owner])
    vertices, triangles = np.asarray(vertices), np.asarray(triangles)
    view = CandidateViewV40(intrinsic=[[10.,0.,100.],[0.,10.,100.],[0.,0.,1.]],
        world_from_camera=np.eye(4), width=300, height=300)
    reference = freeze_reference_v40(vertices, triangles, np.asarray(owners), [view],
        sample_spacing_m=.3, seed=4001, max_samples=50000)
    return reference, vertices, triangles


class ContinuousDistanceGeometryTests(unittest.TestCase):
    def setUp(self):
        self.reference, self.vertices, self.triangles = analytic_reference()
        self.kw = dict(sample_spacing_m=.3, seed=4002, max_samples=1000000)

    def test_parallel_surfaces_have_known_distance_and_equal_facility_mass(self):
        shifted = self.vertices.copy(); shifted[:,2] += .1
        summary, arrays = module.continuous_distances(self.reference, shifted, self.triangles, **self.kw)
        for row in summary['reference_to_prediction']['per_instance']:
            for key in ('mean_m','p50_m','p95_m'):
                self.assertAlmostEqual(row[key], .1, places=12)
            mask = arrays['reference_instance_id'] == row['instance_id']
            self.assertAlmostEqual(arrays['reference_equal_facility_macro_weight'][mask].sum(), .5)
        self.assertAlmostEqual(summary['prediction_to_full_scene']['mean_m'], .1, places=12)
        self.assertAlmostEqual(summary['prediction_to_full_scene']['nearest_gt_background_area_m2'], 3.)
        self.assertAlmostEqual(summary['prediction_to_full_scene']['prediction_area_m2'], 6.)
        self.assertTrue(summary['prediction_to_full_scene']['includes_background'])
        self.assertIn('not symmetric Chamfer', summary['prediction_domain'])
        np.testing.assert_array_equal(arrays['reference_points_m'], self.reference.points)
        self.assertEqual(arrays['reference_macro_ecdf_probability'][-1], 1.)

    def test_unreconstructed_facility_stays_in_macro_denominator(self):
        summary, arrays = module.continuous_distances(self.reference, self.vertices[:4], self.triangles[:2], **self.kw)
        rows = summary['reference_to_prediction']['per_instance']
        self.assertEqual([r['instance_id'] for r in rows], [0,1])
        self.assertAlmostEqual(rows[0]['mean_m'], 0., places=12)
        self.assertGreater(rows[1]['mean_m'], 2.)
        self.assertAlmostEqual(summary['reference_to_prediction']['equal_facility_macro']['mean_m'],
            .5*(rows[0]['mean_m']+rows[1]['mean_m']))
        self.assertEqual(len(arrays['reference_distance_m']), len(self.reference.points))

    def test_empty_prediction_retains_infinite_distance_and_unmatched_area(self):
        summary, arrays = module.continuous_distances(self.reference,
            np.empty((0,3)), np.empty((0,3), dtype=int), **self.kw)
        self.assertTrue(np.isinf(arrays['reference_distance_m']).all())
        self.assertTrue(np.isnan(arrays['reference_nearest_prediction_point_m']).all())
        self.assertTrue((arrays['reference_nearest_prediction_canonical_triangle'] == -1).all())
        for row in summary['reference_to_prediction']['per_instance']:
            self.assertEqual(row['unmatched_area_m2'], row['reference_area_m2'])
            self.assertIsNone(row['mean_m']); self.assertIsNone(row['p95_m'])
            self.assertTrue(row['mean_is_infinite'])
        macro = summary['reference_to_prediction']['equal_facility_macro']
        self.assertAlmostEqual(macro['unmatched_weight'], 1.)
        self.assertTrue(np.isinf(arrays['reference_macro_ecdf_distance_m'][0]))
        self.assertEqual(arrays['reference_macro_ecdf_probability'][0], 1.)
        self.assertTrue(summary['prediction_to_full_scene']['empty_domain'])
        module.canonical_bytes(summary)  # JSON rejects no NaN/Infinity fields.

    def test_weighted_quantile_never_renormalizes_away_infinite_mass(self):
        result = module.weighted_summary([0., .2, np.inf], [1., 2., 1.])
        self.assertEqual(result['p50_m'], .2)
        self.assertIsNone(result['p95_m']); self.assertIsNone(result['mean_m'])
        self.assertEqual(result['unmatched_fraction'], .25)
        support, cdf = module.weighted_ecdf([0., .2, .2, np.inf], [1., 1., 1., 1.])
        np.testing.assert_array_equal(support, [0., .2, np.inf])
        np.testing.assert_array_equal(cdf, [.25, .75, 1.])

    def test_duplicate_rule_and_all_positive_micro_faces_match_main(self):
        tiny = np.array([[11.,0.,2.],[11.,.0000005,2.],[11.0000005,0.,2.]])
        vertices = np.vstack([self.vertices, tiny])
        triangles = np.vstack([self.triangles, self.triangles[:1], [[12,13,14]]])
        summary, arrays = module.continuous_distances(self.reference, vertices, triangles, **self.kw)
        self.assertEqual(summary['sampling']['duplicate_prediction_triangles_removed'], 1)
        self.assertEqual(summary['sampling']['distinct_positive_micro_faces_preserved'], 1)
        self.assertEqual(summary['sampling']['canonical_prediction_triangles'], 7)
        self.assertIn(7, arrays['prediction_sample_original_triangle'])
        self.assertNotIn(6, arrays['prediction_sample_original_triangle'])
        with self.assertRaisesRegex(ValueError, 'sample limit'):
            module.continuous_distances(self.reference, vertices, triangles,
                sample_spacing_m=.3, seed=4002, max_samples=1)


class ContinuousDistanceReceiptTests(unittest.TestCase):
    def setUp(self):
        # Reuse the existing purely analytic pinned-episode double. Its guards
        # prohibit main numerical evaluation and policy/TSDF replay throughout.
        fixture = fixtures.SemanticSceneEndpointReuseTests()
        fixture.setUpClass(); fixture.setUp(); self.addCleanup(fixture.doCleanups)
        self.fixture = fixture
        self.reference, _, _ = analytic_reference()
        self.patcher = patch.object(module.proof, 'load_semantic_scene_reference',
            return_value=(self.reference, np.ones((2,2), bool), dict(coverage=fixture.coverage)))
        self.patcher.start(); self.addCleanup(self.patcher.stop)
        fixture.episodes['source']['protocol']['navigation_root'] = 'explicit_analytic_navigation'
        metrics = fixture.reviews['source']['evaluation']['metrics']
        summary, _ = module.continuous_distances(self.reference, fixture.vertices, fixture.triangles,
            **{key:module.EVALUATION[key] for key in ('sample_spacing_m','seed','max_samples')})
        metrics.update(reference_fingerprint=self.reference.fingerprint,
            reference_samples=len(self.reference.points), prediction_samples=summary['sampling']['prediction_samples'],
            canonical_prediction_triangles=1, per_instance=[dict(instance_id=0),dict(instance_id=1)])
        fixture.refresh('source')
        self.review = fixture.root/'source_review.json'
        self.args = dict(episode=fixture.episodes['source']['root'],
            manifest_sha256=fixture.episodes['source']['manifest_sha256'], review=self.review,
            review_sha256=module.file_sha256(self.review), output_dir=fixture.root/'continuous')

    def test_receipt_archives_every_distance_and_preserves_primary_score(self):
        before = deepcopy(self.fixture.reviews['source'])
        result = module.supplement(**self.args)
        self.assertEqual(result['status'], 'continuous_distance_completed')
        self.assertFalse(result['primary_numerical_evaluation_recomputed'])
        self.assertEqual(result['unchanged_primary_metrics_from_review']['J_nav'], .4)
        self.assertEqual(self.fixture.reviews['source'], before)
        self.assertTrue((self.args['output_dir']/'receipt.json').exists())
        with np.load(self.args['output_dir']/'distances.npz', allow_pickle=False) as data:
            self.assertEqual(len(data['reference_distance_m']), len(self.reference.points))
            self.assertEqual(set(data.files), set(result['distance_archive']['arrays']))
        self.fixture.forbid_score.assert_not_called(); self.fixture.forbid_replay.assert_not_called()
        with self.assertRaisesRegex(ValueError, 'never overwritten'):
            module.supplement(**self.args)

    def test_incomplete_or_wrongly_bound_review_rejected_before_distances(self):
        self.fixture.change_review('source', lambda row:row['replay'].update(frames_verified=1))
        self.args['review_sha256'] = module.file_sha256(self.review)
        with patch.object(module, 'continuous_distances', side_effect=AssertionError('must not calculate')) as compute:
            with self.assertRaisesRegex(ValueError, 'independent complete'):
                module.supplement(**self.args)
            compute.assert_not_called()
        self.assertFalse(self.args['output_dir'].exists())

    def test_final_pin_failure_retains_failed_output_without_retry(self):
        original = module._checked_inputs
        count = [0]
        def changed(*args):
            count[0] += 1
            if count[0] == 2:
                raise ValueError('analytic final pin drift')
            return original(*args)
        with patch.object(module, '_checked_inputs', side_effect=changed):
            with self.assertRaisesRegex(ValueError, 'final pin drift'):
                module.supplement(**self.args)
        self.assertTrue((self.args['output_dir']/'error.json').exists())
        self.assertTrue((self.args['output_dir']/'distances.npz').exists())
        self.assertFalse((self.args['output_dir']/'receipt.json').exists())
        self.assertEqual(count[0], 2)


if __name__ == '__main__':
    unittest.main()
