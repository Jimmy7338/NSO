"""Analytic measurement regressions; no World or experiment score is used."""
import unittest

import numpy as np

from nso.instance_belief_v40 import PaidRGBDObservationV40
from nso.observed_instances_article_v1 import ObservedInstancesArticleV1
from nso.observed_instances_local import ObservedInstancesLocal


def model(cls=ObservedInstancesArticleV1, **kwargs):
    return cls(palette={'a': (40, 100, 220)}, structure_names=('front', 'side', 'both'),
               class_structure_prior={'a': (.7, .2, .1)}, **kwargs)


def packet(step, *, origin_x=0., color_shift=0):
    rgb = np.full((32, 64, 3), 127 + color_shift, np.uint8)
    depth = np.zeros((32, 64))
    depth[10:22, 8:32] = 2.
    rgb[14:18, 10:14] = (40, 100, 220)
    intrinsic = np.array([[40., 0., 31.5], [0., 40., 15.5], [0., 0., 1.]])
    pose = np.eye(4)
    pose[0, 3] = origin_x
    return PaidRGBDObservationV40(f'article-{step}', step, rgb, depth, intrinsic, pose)


def feedback(ledger, receipt, values=(0., -2., -1.)):
    return ledger.apply_geometry_feedback(receipt['accepted'][0]['instance_id'],
        frame_id=receipt['frame_id'], observation_sha256=receipt['observation_sha256'],
        log_likelihoods=values)


class ArticleEvidenceTests(unittest.TestCase):
    def test_cache_full_does_not_discard_first_reliable_measurement(self):
        old, new = model(ObservedInstancesLocal, maximum_support_points=24), model(maximum_support_points=24)
        # A plane estimator may need several paid frames before producing its
        # first reliable residual. No feedback is supplied at the first frame.
        for ledger in (old, new):
            ledger.observe(packet(0))
        a, b = old.observe(packet(1, color_shift=1)), new.observe(packet(1, color_shift=1))
        self.assertFalse(a['accepted'][0]['geometry_feedback_eligible'])
        self.assertTrue(b['accepted'][0]['geometry_feedback_eligible'])
        self.assertFalse(feedback(old, a)['applied'])
        self.assertTrue(feedback(new, b)['applied'])
        self.assertEqual(old.snapshot()['instances'][0]['support_sha256'],
                         new.snapshot()['instances'][0]['support_sha256'])

    def test_repeated_or_tiny_changed_pose_does_not_reinforce_cache_rescue(self):
        ledger = model(maximum_support_points=24)
        first = ledger.observe(packet(0)); feedback(ledger, first)
        before = ledger.snapshot()['instances'][0]['geometry_log_scores']
        for step, offset in [(1, 0.), (2, .001)]:
            receipt = ledger.observe(packet(step, origin_x=offset, color_shift=step))
            self.assertFalse(receipt['accepted'][0]['geometry_feedback_eligible'])
            self.assertFalse(feedback(ledger, receipt)['applied'])
        self.assertEqual(before, ledger.snapshot()['instances'][0]['geometry_log_scores'])

    def test_new_paid_view_can_contribute_after_cache_full(self):
        ledger = model(maximum_support_points=24)
        first = ledger.observe(packet(0)); feedback(ledger, first)
        second = ledger.observe(packet(1, origin_x=.25))
        self.assertEqual(len(second['accepted']), 1)
        self.assertTrue(second['accepted'][0]['article_feedback_eligibility']['cache_rescue_eligible'])
        self.assertTrue(feedback(ledger, second)['applied'])
        self.assertFalse(feedback(ledger, second)['applied'])

    def test_duplicate_payload_remains_rejected_even_before_first_feedback(self):
        ledger = model(maximum_support_points=24)
        ledger.observe(packet(0))
        same = ledger.observe(packet(1))
        self.assertTrue(same['duplicate_measurement'])
        self.assertFalse(feedback(ledger, same)['applied'])

    def test_uninformative_rescue_does_not_consume_feedback_budget(self):
        ledger = model(maximum_support_points=24)
        ledger.observe(packet(0))
        flat = ledger.observe(packet(1, color_shift=1))
        self.assertFalse(feedback(ledger, flat, (0., 0., 0.))['applied'])
        self.assertEqual(ledger.snapshot()['instances'][0]['geometric_feedback_frames'], 0)
        informative = ledger.observe(packet(2, color_shift=2))
        self.assertTrue(feedback(ledger, informative)['applied'])

    def test_successful_feedback_cap_is_preserved(self):
        ledger = model(maximum_support_points=24, maximum_evidence_frames=2)
        for step, offset in [(0, 0.), (1, .25)]:
            receipt = ledger.observe(packet(step, origin_x=offset))
            self.assertTrue(feedback(ledger, receipt)['applied'])
        third = ledger.observe(packet(2, origin_x=-.25))
        self.assertFalse(feedback(ledger, third)['applied'])
        self.assertEqual(ledger.snapshot()['instances'][0]['geometric_feedback_frames'], 2)

    def test_g_and_s_have_identical_measurement_eligibility(self):
        g, s = model(mode='G', maximum_support_points=24), model(mode='S', maximum_support_points=24)
        for step, offset in [(0, 0.), (1, .25)]:
            a, b = g.observe(packet(step, origin_x=offset)), s.observe(packet(step, origin_x=offset))
            self.assertEqual(a['accepted'], b['accepted'])
            feedback(g, a); feedback(s, b)
            self.assertEqual(g.geometry_snapshot(), s.geometry_snapshot())

    def test_wrong_hash_and_old_frame_are_rejected(self):
        ledger = model(maximum_support_points=24)
        a = ledger.observe(packet(0)); key = a['accepted'][0]['instance_id']
        with self.assertRaises(ValueError):
            ledger.apply_geometry_feedback(key, frame_id=a['frame_id'],
                observation_sha256='bad', log_likelihoods=(0., -1., -1.))
        ledger.observe(packet(1, color_shift=1))
        with self.assertRaises(ValueError):
            feedback(ledger, a)

    def test_unsaturated_behavior_preserves_geometry_and_category_state(self):
        old, new = model(ObservedInstancesLocal), model()
        for step, offset in [(0, 0.), (1, .25), (2, .5)]:
            a, b = old.observe(packet(step, origin_x=offset)), new.observe(packet(step, origin_x=offset))
            self.assertEqual(a['instances'], b['instances'])
            self.assertEqual([r['geometry_feedback_eligible'] for r in a['accepted']],
                             [r['geometry_feedback_eligible'] for r in b['accepted']])
            feedback(old, a); feedback(new, b)
            before, after = old.geometry_snapshot(), new.geometry_snapshot()
            for frame in after['paid_frame_associations'].values():
                for association in frame['instances'].values():
                    association.pop('article_feedback_eligibility')
            self.assertEqual(before, after)


if __name__ == '__main__':
    unittest.main()
