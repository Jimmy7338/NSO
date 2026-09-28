import copy
from types import SimpleNamespace
import unittest

import numpy as np

from nso.surface_feedback_v16 import SurfaceFeedbackV16, capture_support


def row(information=1., bits=1, residual=0.):
    return dict(point=np.array([.5, .5, .5]), information=information,
                bits=bits, residual=residual, n=1)


def mapper():
    return SimpleNamespace(frames=1, belief=np.zeros((4, 4), dtype=np.int8),
        camera_seen=np.ones((4, 4), dtype=bool), config=SimpleNamespace(resolution_m=.2),
        grid_cell=lambda p: (1, 1), quality={(1, 1, 1): row()})


class SurfaceFeedbackTest(unittest.TestCase):
    def test_new_3d_support_despite_saturated_2d_camera(self):
        m = mapper(); f = SurfaceFeedbackV16(m, action_id=0, coordinate_epoch='fixed')
        m.frames += 1; m.quality[(1, 1, 2)] = row(bits=4)
        out = f.observe(m, action_id=1, coordinate_epoch='fixed')
        self.assertEqual(out['first_observed_support_count'], 1)
        self.assertEqual(out['signed_prior_quality_mean_change'], 0.)
        self.assertEqual(out['initial_directions_on_new_support'], 1)
        self.assertTrue(m.camera_seen.all())

    def test_later_support_in_next_action_fixed_comparison_set(self):
        m = mapper(); f = SurfaceFeedbackV16(m, action_id=0, coordinate_epoch='fixed')
        m.frames += 1; m.quality[(1, 1, 2)] = row(information=.25)
        f.observe(m, action_id=1, coordinate_epoch='fixed')
        m.frames += 1; m.quality[(1, 1, 2)]['information'] = 1.
        out = f.observe(m, action_id=2, coordinate_epoch='fixed')
        self.assertEqual(out['prior_support_count'], 2)
        self.assertAlmostEqual(out['signed_prior_quality_mean_change'], (.8 - .5) / 2)

    def test_loss_reappearance_and_view_novelty_not_double_counted(self):
        m = mapper(); f = SurfaceFeedbackV16(m, action_id=0, coordinate_epoch='fixed')
        m.frames += 1; m.quality.clear()
        lost = f.observe(m, action_id=1, coordinate_epoch='fixed')
        self.assertEqual(lost['signed_prior_quality_mean_change'], -.8)
        m.frames += 1; m.quality[(1, 1, 1)] = row(bits=3)
        seen = f.observe(m, action_id=2, coordinate_epoch='fixed')
        self.assertEqual(seen['first_observed_support_count'], 0)
        self.assertEqual(seen['reacquired_support_count'], 1)
        self.assertEqual(seen['new_directions_on_previously_seen_support'], 1)
        m.frames += 1
        again = f.observe(m, action_id=3, coordinate_epoch='fixed')
        self.assertEqual(again['new_directions_on_previously_seen_support'], 0)
        self.assertEqual(again['signed_prior_quality_mean_change'], 0.)

    def test_signed_quality_degradation_retained(self):
        m = mapper(); f = SurfaceFeedbackV16(m, action_id=0, coordinate_epoch='fixed')
        m.frames += 1; m.quality[(1, 1, 1)]['residual'] = .01
        out = f.observe(m, action_id=1, coordinate_epoch='fixed')
        self.assertLess(out['signed_prior_quality_mean_change'], 0)
        self.assertEqual(out['worsened_prior_support_count'], 1)

    def test_invalid_transition_is_atomic(self):
        for aid, epoch, increment in [(0, 'fixed', 1), (2, 'fixed', 1),
                                      (1, 'changed', 1), (1, 'fixed', 2)]:
            m = mapper(); f = SurfaceFeedbackV16(m, action_id=0, coordinate_epoch='fixed')
            before = copy.deepcopy(vars(f)); m.frames += increment
            with self.assertRaises(ValueError): f.observe(m, action_id=aid, coordinate_epoch=epoch)
            self.assertEqual(vars(f), before)

    def test_observer_does_not_alias_mapper_or_read_labels(self):
        m = mapper(); m.quality[(1, 1, 1)]['label'] = object()
        f = SurfaceFeedbackV16(m, action_id=0, coordinate_epoch='fixed')
        m.quality[(1, 1, 1)]['information'] = 0.; m.frames += 1
        out = f.observe(m, action_id=1, coordinate_epoch='fixed')
        self.assertEqual(out['signed_prior_quality_mean_change'], -.8)
        self.assertEqual(m.quality[(1, 1, 1)]['information'], 0.)

    def test_invalid_support_rejected(self):
        for field, value in [('bits', 256), ('bits', True), ('n', 0),
                             ('information', float('nan')), ('residual', -1.)]:
            m = mapper(); m.quality[(1, 1, 1)][field] = value
            with self.assertRaises(ValueError): capture_support(m)


if __name__ == '__main__':
    unittest.main()
