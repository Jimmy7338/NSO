"""Analytic forecast contracts; no World, rollout or measured performance gain."""
from copy import deepcopy
import unittest

import numpy as np

from nso.analytic_fixture_v42 import analytic_forward_sequence_v42
from nso.instance_belief_v40 import PaidRGBDObservationV40
from nso.observed_instances_v41 import ObservedInstancesV41
from nso.surface_evaluation_v40 import CandidateViewV40
from nso.view_quality_v42 import ViewQualityPredictorV42, _sample_union


NAMES = ('planar', 'recessed', 'louvered', 'open_frame')
PALETTE = {'cabinet': (40, 100, 220)}


def ledger(prior=(.1, .7, .1, .1), mode='S'):
    return ObservedInstancesV41(palette=PALETTE, structure_names=NAMES,
                               class_structure_prior={'cabinet': prior}, mode=mode)


def public_view(name, x=.75, y=.75, yaw=0.):
    forward = np.array([np.cos(yaw), np.sin(yaw), 0.])
    transform = np.eye(4)
    transform[:3, :3] = np.column_stack((np.cross(forward, [0., 0., 1.]), [0., 0., -1.], forward))
    transform[:3, 3] = [x, y, .9]
    intrinsic = np.array([[48., 0., 47.5], [0., 48., 35.5], [0., 0., 1.]])
    # This constructs no RGBD packet for unchosen candidates.
    return CandidateViewV40(intrinsic, transform, 96, 72, view_id=name)


class ViewQualityV42Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.packets = analytic_forward_sequence_v42()
        cls.predictor = ViewQualityPredictorV42()
        cls.ledgers = {'G': ledger(mode='G'), 'S': ledger(),
                       'flat': ledger((.25,)*4), 'swapped': ledger((.1, .1, .7, .1))}
        cls.receipts = []
        for packet in cls.packets:
            observed = {name: obj.observe(packet) for name, obj in cls.ledgers.items()}
            cls.receipts.append(observed['G'])
            cls.predictor.observe(packet, observed['G']['accepted'])
        cls.views = [public_view('front'), public_view('left', y=1.25),
                     public_view('right', y=.25), public_view('behind_camera', yaw=np.pi),
                     public_view('too_far', x=-4.5)]
        cls.snapshots = {name: obj.snapshot()['instances'][0] for name, obj in cls.ledgers.items()}
        cls.forecasts = {name: cls.predictor.forecast(snap, cls.views) for name, snap in cls.snapshots.items()}

    def test_forecast_is_exact_posterior_expectation_not_class_weight(self):
        for name, result in self.forecasts.items():
            for row in result['candidates']:
                self.assertAlmostEqual(row['expected_new_surface_area_m2'],
                    np.asarray(row['structure_new_surface_area_m2'])@np.asarray(result['structure_probabilities']), places=12)
            self.assertFalse(result['class_importance_weights_used'])
            self.assertFalse(result['tsdf_quality_measured'])
        self.assertTrue(self.forecasts['S']['semantic_conditioning_used'])
        self.assertFalse(self.forecasts['G']['semantic_conditioning_used'])

    def test_flat_prior_matches_geometry_only_and_swapped_prior_changes_expectation(self):
        g, s, flat, swapped = (self.forecasts[name] for name in ('G', 'S', 'flat', 'swapped'))
        self.assertEqual(g['candidates'], flat['candidates'])
        self.assertNotEqual([r['expected_new_surface_area_m2'] for r in s['candidates']],
                            [r['expected_new_surface_area_m2'] for r in swapped['candidates']])
        for i in range(len(self.views)):
            self.assertEqual(g['candidates'][i]['structure_new_surface_area_m2'],
                             s['candidates'][i]['structure_new_surface_area_m2'])
            self.assertEqual(g['candidates'][i]['components'], swapped['candidates'][i]['components'])
        self.assertEqual(g['geometry_evidence_sha256'], s['geometry_evidence_sha256'])

    def test_seen_support_discount_is_nonnegative_and_applied(self):
        components = self.forecasts['G']['candidates'][0]['components']
        self.assertTrue(any(c['already_observed_area_m2'] > .05 for c in components))
        for c in components:
            self.assertGreaterEqual(c['new_surface_area_m2'], 0.)
            self.assertAlmostEqual(c['visible_area_m2'], c['new_surface_area_m2']+c['already_observed_area_m2'])
            self.assertLessEqual(c['visible_area_m2'], c['total_prototype_area_m2'])

    def test_self_occlusion_and_front_faces_hide_box_back_surface(self):
        # Camera is in front of every nominal prototype; at most front, side,
        # and top/bottom of a closed box can be visible, never its back face.
        rows = self.forecasts['G']['candidates'][0]['components']
        planar = [r for r in rows if r['structure'] == 'planar']
        for row in planar:
            self.assertGreater(row['visible_area_m2'], .5)
            self.assertLess(row['visible_area_m2'], .5*row['total_prototype_area_m2'])
        # Recesses/open frames must expose their actual nominal union boundary;
        # each estimate is independently ray-tested, never all surface area.
        self.assertTrue(all(r['visible_area_m2'] < r['total_prototype_area_m2'] for r in rows))

    def test_outside_fov_and_depth_candidates_have_zero_area(self):
        for row in self.forecasts['G']['candidates'][-2:]:
            self.assertEqual(row['expected_new_surface_area_m2'], 0.)
            self.assertEqual(row['structure_visible_area_m2'], [0.]*4)

    def test_missing_or_ambiguous_plane_has_explicit_zero_fallback(self):
        packet = self.packets[0]
        p = ViewQualityPredictorV42()
        receipts = deepcopy(self.receipts[0]['accepted'])
        for item in receipts:
            item['marker_pixel_indices'] = []
        p.observe(packet, receipts)
        l = ledger(); l.observe(packet)
        result = p.forecast(l.snapshot()['instances'][0], [self.views[0]])['candidates'][0]
        self.assertEqual(result['fallback_reason'], 'no_reliable_observed_label_plane')
        self.assertEqual(result['expected_new_surface_area_m2'], 0.)
        changed = deepcopy(self.snapshots['G']); changed['association_uncertain'] = True
        result = self.predictor.forecast(changed, [self.views[0]])['candidates'][0]
        self.assertEqual(result['fallback_reason'], 'association_uncertain')

    def test_private_fields_are_rejected_at_all_mapping_boundaries(self):
        for field in ('true_structure', 'world_aabb_m', 'future_depth', 'scene_mesh', 'class_importance'):
            changed = deepcopy(self.snapshots['G']); changed[field] = 'forbidden'
            with self.subTest(field=field), self.assertRaises(ValueError):
                self.predictor.forecast(changed, [self.views[0]])
            associations = deepcopy(self.receipts[0]['accepted']); associations[0][field] = 'forbidden'
            with self.assertRaises(ValueError):
                ViewQualityPredictorV42().observe(self.packets[0], associations)
        with self.assertRaises(TypeError):
            self.predictor.forecast(self.snapshots['G'], [dict(world_aabb_m=[])])

    def test_support_points_hash_and_posterior_are_verified(self):
        for field in ('support_points_world_m', 'support_sha256', 'structure_probabilities'):
            changed = deepcopy(self.snapshots['G'])
            if field == 'support_points_world_m': changed[field][0][0] += .01
            if field == 'support_sha256': changed[field] = '0'*64
            if field == 'structure_probabilities': changed[field] = [.7, .1, .1, .1]
            with self.subTest(field=field), self.assertRaises(ValueError):
                self.predictor.forecast(changed, [self.views[0]])

    def test_current_pixel_provenance_and_mask_are_verified(self):
        for field in ('observation_sha256', 'points_world_m', 'support_sha256', 'marker_pixel_indices'):
            associations = deepcopy(self.receipts[0]['accepted'])
            if field.endswith('sha256'): associations[0][field] = '0'*64
            if field == 'points_world_m': associations[0][field][0][0] += .01
            if field == 'marker_pixel_indices': associations[0][field] = [0]
            with self.subTest(field=field), self.assertRaises(ValueError):
                ViewQualityPredictorV42().observe(self.packets[0], associations)

    def test_cached_plane_is_auditable_without_private_residual_cache(self):
        result = self.forecasts['G']
        geometry = result['observed_geometry']
        self.assertEqual(geometry['plane_fit']['observation_sha256'], self.packets[0].sha256())
        self.assertEqual([s['observation_sha256'] for s in geometry['source_frames']],
                         [p.sha256() for p in self.packets])
        self.assertEqual(result['prototype_bank_sha256'], self.predictor.bank.sha256)
        self.assertFalse(result['ground_truth_used']); self.assertFalse(result['future_observation_used'])
        self.assertFalse(result['external_occlusion_modeled'])
        copied = self.predictor.observed_plane_receipts()
        copied[next(iter(copied))]['plane_fit']['anchor_world_m'][0] += 99.
        self.assertNotEqual(copied, self.predictor.observed_plane_receipts())

    def test_retransmission_rejected_but_new_paid_static_packet_adds_no_support(self):
        p, l = ViewQualityPredictorV42(), ledger()
        obs = self.packets[0]; first = l.observe(obs)
        p.observe(obs, first['accepted'])
        original = l.snapshot()['instances'][0]['support_sha256']
        with self.assertRaises(ValueError): p.observe(obs, first['accepted'])
        new = PaidRGBDObservationV40('static-repaid', 1, obs.rgb, obs.depth_m, obs.intrinsic, obs.world_from_camera)
        second = l.observe(new); receipt = p.observe(new, second['accepted'])
        self.assertTrue(receipt['results'][0]['duplicate_measurement'])
        self.assertEqual(l.snapshot()['instances'][0]['support_sha256'], original)
        p.forecast(l.snapshot()['instances'][0], [self.views[0]])

    def test_identical_raw_bytes_with_different_image_shape_are_distinct_rays(self):
        p = ViewQualityPredictorV42()
        obs = self.packets[0]
        p.observe(obs, [])
        shaped = PaidRGBDObservationV40('reshape-not-retransmission', 1,
            obs.rgb.reshape(48, 144, 3), obs.depth_m.reshape(48, 144),
            obs.intrinsic, obs.world_from_camera)
        self.assertEqual(obs.rgb.tobytes(), shaped.rgb.tobytes())
        self.assertEqual(obs.depth_m.tobytes(), shaped.depth_m.tobytes())
        self.assertFalse(p.observe(shaped, [])['duplicate_measurement'])

    def test_candidates_are_bounded_unique_and_use_fixed_axial_sensor_contract(self):
        v = self.views[0]
        with self.assertRaises(ValueError): self.predictor.forecast(self.snapshots['G'], [v, v])
        changed = CandidateViewV40(v.intrinsic, v.world_from_camera, 96, 72, far_m=8.)
        with self.assertRaises(ValueError): self.predictor.forecast(self.snapshots['G'], [changed])
        with self.assertRaises(ValueError): ViewQualityPredictorV42(surface_spacing_m=.001)

    def test_union_area_quadrature_excludes_overlapping_internal_faces(self):
        boxes = np.array([[0., 1., 0., 1., 0., 1.], [.5, 1.5, 0., 1., 0., 1.]])
        _, weights, normals = _sample_union(boxes, .1)
        self.assertAlmostEqual(weights.sum(), 8.)
        np.testing.assert_allclose(np.linalg.norm(normals, axis=1), 1.)

    def test_observed_area_is_monotonic_when_only_support_increases(self):
        p, l = ViewQualityPredictorV42(), ledger(mode='G')
        first = l.observe(self.packets[0]); p.observe(self.packets[0], first['accepted'])
        before = p.forecast(l.snapshot()['instances'][0], [self.views[0]])['candidates'][0]
        second = l.observe(self.packets[1]); p.observe(self.packets[1], second['accepted'])
        after = p.forecast(l.snapshot()['instances'][0], [self.views[0]])['candidates'][0]
        np.testing.assert_array_less(np.asarray(after['structure_new_surface_area_m2'])-1e-12,
                                     np.asarray(before['structure_new_surface_area_m2']))


if __name__ == '__main__':
    unittest.main()
