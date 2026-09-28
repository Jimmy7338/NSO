"""Analytic sensor-array checks; no World, TSDF, planner or mesh evaluation."""
import json
import unittest
import numpy as np
from nso.cpu_four_modules_v35 import ObservationV35
from nso.observed_geometry_v39 import ObservedGeometryV39


def calibration(size=17):
    return np.array([[24., 0., (size-1)/2], [0., 24., (size-1)/2], [0., 0., 1.]])


def observation(depth, step=0, pose=(0, 0, 0), color=0):
    depth = np.asarray(depth, np.float32)
    return ObservationV35(frame_id=f'paid-{step}', step=step, pose=pose,
        action=None if step == 0 else 'forward', depth=depth,
        rgb=np.full(depth.shape+(3,), color, np.uint8), ranges=np.ones(8))


class ObservedGeometryTests(unittest.TestCase):
    def setUp(self):
        self.k, self.t = calibration(), np.eye(4)
        self.depth = np.full((17, 17), 2.)

    def ledger(self, **kwargs):
        return ObservedGeometryV39(stride=1, voxel_m=.01, **kwargs)

    def test_backprojection_and_normals_are_measured(self):
        ledger = self.ledger()
        ledger.observe(observation(self.depth), self.k, self.t)
        s = ledger.snapshot()
        np.testing.assert_allclose(s['points'][:, 2], 2.)
        self.assertTrue(np.any(np.all(np.isclose(s['points'], [0, 0, 2]), axis=1)))
        self.assertEqual(int(s['normal_valid'].sum()), 15*15)
        np.testing.assert_allclose(s['normals'][s['normal_valid']], np.tile([0, 0, -1.], (225, 1)))
        np.testing.assert_allclose(np.linalg.norm(s['past_view_directions'][:, 0], axis=1), 1.)

    def test_depth_discontinuity_invalidates_normal_not_surface(self):
        depth = np.full((9, 9), 2.)
        depth[:, 4:] = 4.
        ledger = self.ledger(max_range_m=6.)
        receipt = ledger.observe(observation(depth), calibration(9), self.t)
        self.assertEqual(receipt['valid_sampled_normals'], 7*5)
        self.assertEqual(len(ledger.snapshot()['points']), 81)
        self.assertGreater(receipt['sampled_boundary_points'], 32)
        self.assertFalse(receipt['boundary_is_true_hole'])

    def test_empty_paid_frame_is_valid_and_scores_zero(self):
        ledger = self.ledger()
        receipt = ledger.observe(observation(np.zeros_like(self.depth)), self.k, self.t)
        self.assertTrue(receipt['empty_observation'])
        score = ledger.score_view(self.k, self.t, mode='vista')
        self.assertEqual(score['observed_surfels'], 0)
        self.assertEqual(score['geometry_gain'], 0.)
        self.assertEqual(score['semantic_gain'], 0.)
        json.dumps(score, allow_nan=False)

    def test_repeated_frame_rejected_without_state_change(self):
        ledger = self.ledger()
        obs = observation(self.depth)
        ledger.observe(obs, self.k, self.t)
        before = ledger.snapshot()
        with self.assertRaises(ValueError):
            ledger.observe(obs, self.k, self.t)
        self.assertIs(ledger.snapshot(), before)

    def test_repeated_center_never_adds_direction_or_angular_gain(self):
        ledger = self.ledger()
        ledger.observe(observation(self.depth), self.k, self.t)
        ledger.observe(observation(self.depth, 1, (0, 0, 1)), self.k, self.t)
        s = ledger.snapshot()
        self.assertTrue(np.all(s['history_direction_count'] == 1))
        self.assertTrue(np.all(s['distinct_pose_count'] == 2))
        score = ledger.score_view(self.k, self.t, mode='vista', semantic_relevance=0.)
        self.assertEqual(score['geometry_gain'], 0.)
        self.assertEqual(score['new_inspection_surfels'], 0)
        t = self.t.copy(); t[0, 3] = .5
        self.assertGreater(ledger.score_view(self.k, t, mode='vista')['geometry_gain'], 0.)

    def test_readonly_snapshot_and_roi(self):
        ledger = self.ledger(public_bounds=[[-.1, -.1, 1.9], [.1, .1, 2.1]])
        ledger.observe(observation(self.depth), self.k, self.t)
        s = ledger.snapshot()
        self.assertEqual(len(s['points']), 9)
        self.assertTrue(np.all(np.abs(s['points'][:, :2]) <= .1))
        with self.assertRaises(ValueError):
            s['points'][0, 0] = 9
        with self.assertRaises(TypeError):
            s['points'] = np.zeros((1, 3))

    def test_observed_front_surface_occludes_saved_back_surface(self):
        ledger = self.ledger(max_range_m=6.)
        ledger.observe(observation(self.depth), self.k, self.t)
        ledger.observe(observation(self.depth*2, 1), self.k, self.t)
        score = ledger.score_view(self.k, self.t, mode='vista', max_range_m=5.)
        visible = ledger.snapshot()['points'][score['visible_indices']]
        self.assertGreater(len(visible), 0)
        np.testing.assert_allclose(visible[:, 2], 2.)
        self.assertGreater(score['observed_surfels'], score['visible_surfels'])

    def test_inspection_revisit_and_bounded_history_do_not_recreate_debt(self):
        ledger = self.ledger(max_range_m=6., max_directions=1)
        ledger.observe(observation(self.depth*2), self.k, self.t)
        close = self.t.copy(); close[2, 3] = 2.
        pre = ledger.score_view(self.k, close, mode='inspection')
        self.assertGreater(pre['new_inspection_surfels'], 0)
        self.assertEqual(pre['new_inspection_surfels'], len(pre['new_inspection_indices']))
        ledger.observe(observation(self.depth, 1, (1, 0, 0)), self.k, close)
        self.assertLessEqual(int(ledger.snapshot()['history_direction_count'].max()), 1)
        post = ledger.score_view(self.k, close, mode='inspection')
        self.assertEqual(post['new_inspection_surfels'], 0)
        self.assertGreater(post['previously_qualified_surfels'], 0)

    def test_relevance_zero_disables_inspection_and_rgb_does_not_change_geometry(self):
        ledgers = [self.ledger(max_range_m=6.) for _ in range(2)]
        for ledger, color in zip(ledgers, (0, 220)):
            ledger.observe(observation(self.depth*2, color=color), self.k, self.t)
        for key in ledgers[0].snapshot():
            np.testing.assert_array_equal(ledgers[0].snapshot()[key], ledgers[1].snapshot()[key])
        close = self.t.copy(); close[2, 3] = 2.
        score = ledgers[0].score_view(self.k, close, semantic_relevance=0.)
        self.assertEqual(score['inspection_gain'], 0.)
        self.assertEqual(score['semantic_gain'], 0.)
        relevant = ledgers[0].score_view(self.k, close, semantic_relevance=.75)
        self.assertAlmostEqual(relevant['semantic_gain'], .75)

    def test_resolution_scales_with_camera_not_claimed_original_sensor(self):
        ledger = self.ledger()
        ledger.observe(observation(self.depth), self.k, self.t)
        k48 = self.k.copy(); k48[0, 0] = k48[1, 1] = 48.
        k480 = k48.copy(); k480[0, 0] = k480[1, 1] = 480.
        self.assertAlmostEqual(ledger.score_view(k48, self.t)['minimum_resolution_px_per_cm2'], .0256)
        self.assertAlmostEqual(ledger.score_view(k480, self.t)['minimum_resolution_px_per_cm2'], 2.56)

    def test_invalid_calibration_and_relevance_fail_closed(self):
        ledger = self.ledger()
        bad = self.t.copy(); bad[0, 0] = 2.
        with self.assertRaises(ValueError):
            ledger.observe(observation(self.depth), self.k, bad)
        self.assertEqual(len(ledger.snapshot()['points']), 0)
        with self.assertRaises(ValueError):
            ledger.score_view(self.k, self.t, semantic_relevance=float('nan'))
        with self.assertRaises(ValueError):
            ledger.score_view(self.k, self.t, mode='true_holes')

    def test_score_is_json_and_never_mutates_paid_ledger(self):
        ledger = self.ledger()
        ledger.observe(observation(self.depth), self.k, self.t)
        before = ledger.snapshot()
        t = self.t.copy(); t[0, 3] = .2
        score = ledger.score_view(self.k, t, mode='vista')
        json.dumps(score, allow_nan=False)
        self.assertIs(ledger.snapshot(), before)
        self.assertEqual(len(score['visible_indices']), len(score['visible_novelty']))
        self.assertTrue(0 <= score['geometry_gain'] <= 1)
        self.assertTrue(0 <= score['semantic_gain'] <= 1)


if __name__ == '__main__':
    unittest.main()
