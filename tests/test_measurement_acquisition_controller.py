"""Paid analytic sequences for the common initialization scheduler; no World."""
from copy import deepcopy
import unittest
from unittest.mock import patch

import numpy as np

from nso.controller_semantic_mechanism import METHODS
from nso.instance_belief_v40 import PaidRGBDObservationV40
from nso.primitive_navigation_v41 import PrimitiveStateV41
from test_controller_semantic_mechanism import controller
from test_controller_v43 import packet, mapper, after_action


def partial_label():
    """Retain measured depth but only part of the label's observed RGB support."""
    obs = packet(measured_plane=True)
    rgb = obs.rgb.copy()
    yy, xx = np.indices(obs.depth_m.shape)
    rgb[(xx >= obs.depth_m.shape[1]//2) & np.all(rgb == [40, 100, 220], axis=2)] = 165
    return PaidRGBDObservationV40('partial-analytic-initial-label', 0, rgb,
        obs.depth_m, obs.intrinsic, obs.world_from_camera)


def rig(method='G', **kwargs):
    c = controller(method, measurement_acquisition=True, **kwargs)
    m = mapper()
    obs = partial_label()
    m.update(obs)
    receipt = c.accept(obs, m)
    assert len(c._ledger.snapshot()['instances']) == 1
    assert receipt['first_actual_reliable_planes'] == []
    return c, m


class MeasurementAcquisitionControllerTests(unittest.TestCase):
    def test_initialization_target_and_cost_are_common_to_all_semantic_arms(self):
        rows = []
        for method in METHODS:
            c, _ = rig(method)
            decision = c.choose()
            selected = decision['global_selection']['selected']
            self.assertEqual(selected['kind'], 'measurement_initialization')
            self.assertIsNone(selected['score'])
            self.assertIsNone(selected['expected_gain'])
            self.assertIsNone(selected['evi'])
            self.assertFalse(selected['semantic_information_used'])
            self.assertEqual(c._initialization_spent, 0)
            rows.append((decision['action'], selected['target'], selected['total_cost']))
        self.assertTrue(all(row == rows[0] for row in rows))

    def test_real_paid_movement_can_finish_initialization_before_extra_observe(self):
        c, m = rig()
        decision = c.choose()
        self.assertEqual(decision['action'], 'forward')
        self.assertEqual(sum(c._initialization_attempts.values()), 1)
        state, obs = after_action(c._graph, PrimitiveStateV41('home', 0),
                                  decision['action'], 1, measured_plane=True)
        m.update(obs)
        receipt = c.accept(obs, m)
        self.assertEqual(c._initialization_spent, 1)
        self.assertEqual(len(receipt['first_actual_reliable_planes']), 1)
        next_decision = c.choose()
        self.assertEqual(next_decision['initialization_cancelled']['reason'], 'plane_acquired')
        self.assertEqual(c._initialization_spent, 1)
        self.assertEqual(sum(c._initialization_attempts.values()), 1)
        self.assertFalse(next_decision['initialization_primitive_pending'])

    def test_shared_budget_exhaustion_and_attempt_limit_preserve_normal_planning(self):
        for reason in ('budget', 'attempts'):
            c, _ = rig()
            key = c._ledger.snapshot()['instances'][0]['instance_id']
            if reason == 'budget':
                c._initialization_spent = c._configuration['initialization_action_cap']
            else:
                c._initialization_attempts[key] = 2
            decision = c.choose()
            selected = decision['global_selection']['selected']
            self.assertTrue(selected is None or selected['kind'] != 'measurement_initialization')
            self.assertFalse(decision['initialization_primitive_pending'])
            self.assertNotEqual(decision['action'], 'blocked')

    def test_uncertain_association_cancels_without_refunding_attempt_or_paid_cost(self):
        c, _ = rig()
        c._select_global()
        self.assertIsNotNone(c._initialization_active)
        c._committed_target = c._initialization_active['target']
        c._initialization_spent = 1
        changed = deepcopy(c._ledger.snapshot())
        changed['instances'][0]['association_uncertain'] = True
        with patch.object(c._ledger, 'snapshot', return_value=changed):
            reason = c._check_initialization_commitment()
        self.assertEqual(reason['reason'], 'association_uncertain')
        self.assertEqual(c._initialization_spent, 1)
        self.assertEqual(sum(c._initialization_attempts.values()), 1)
        self.assertIsNone(c._committed_target)

    def test_plane_progress_is_once_per_instance_and_does_not_release_return_latch(self):
        c, m = rig()
        first = c.choose()
        state, obs = after_action(c._graph, PrimitiveStateV41('home', 0),
                                 first['action'], 1, measured_plane=True)
        c._return_latched = True
        m.update(obs)
        accepted = c.accept(obs, m)
        self.assertEqual(len(accepted['first_actual_reliable_planes']), 1)
        self.assertTrue(c._return_latched)
        decision = c.choose()
        self.assertEqual(decision['initialization_cancelled']['reason'], 'return_already_latched')
        state, obs = after_action(c._graph, state, decision['action'], 2, measured_plane=True)
        m.update(obs)
        repeated = c.accept(obs, m)
        self.assertEqual(repeated['first_actual_reliable_planes'], [])
        self.assertEqual(c._initialization_spent, 1)
        self.assertTrue(c._return_latched)

    def test_changed_paid_allowance_return_budget_and_plane_conflict_cancel_macro(self):
        for expected in ('initialization_allowance', 'full_return_budget', 'observed_plane_conflict'):
            c, _ = rig()
            c._select_global()
            active = c._initialization_active
            self.assertIsNotNone(active)
            c._committed_target = active['target']
            out = c._graph.route(c._router.state, active['target']).cost
            back = c._graph.route(active['target'], c._router.home).cost
            if expected == 'initialization_allowance':
                c._initialization_spent = c._configuration['initialization_action_cap']-out
            elif expected == 'full_return_budget':
                c._router.budget = c._router.step+out+back
            else:
                c._predictor._instances[active['instance_id']]['plane_conflict'] = True
            before = c._initialization_spent
            cancelled = c._check_initialization_commitment()
            self.assertEqual(cancelled['reason'], expected)
            self.assertEqual(c._initialization_spent, before)
            self.assertEqual(sum(c._initialization_attempts.values()), 1)
            self.assertIsNone(c._committed_target)


if __name__ == '__main__':
    unittest.main()
