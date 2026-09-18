from copy import deepcopy
import json
import unittest

import numpy as np

from nso.coverage_budget_v21 import ObservedCoverageLedgerV21, union_unknown_mask


class CoverageBudgetV21Tests(unittest.TestCase):
    def test_navigation_inflation_and_obstacle_confirmation_do_not_change_mapping_progress(self):
        belief = np.full((15,15), -1, np.int8)
        belief[5:10,3:8] = 0; belief[:,10] = 1
        hits = np.zeros_like(belief, bool); hits[:,10] = True
        ledgers = [ObservedCoverageLedgerV21(belief.shape, (7,5), radius) for radius in (0.,1.,3.)]
        initial = [ledger.start(belief, radar_hits=hits) for ledger in ledgers]
        self.assertGreater(initial[0]['safe_cells'], initial[2]['safe_cells'])
        self.assertNotEqual(initial[0]['navigation_diagnostics']['status'], initial[2]['navigation_diagnostics']['status'])
        for snapshot in initial:
            self.assertEqual(snapshot['planning_coverage'], 40/225)
            self.assertEqual(snapshot['deficit_cells'], 140)
            self.assertEqual(snapshot['status'], 'ok')
        for ledger in ledgers:
            old = ledger.snapshot(); ledger.observe(belief, 1, True, radar_hits=hits)
            now = ledger.snapshot()
            self.assertLess(now['possible_cells'], old['possible_cells'])
            self.assertEqual(now['planning_coverage'], old['planning_coverage'])
            self.assertEqual(now['deficit_cells'], old['deficit_cells'])
            self.assertEqual(now['conservative_rate'], 0.)
            self.assertFalse(now['navigation_used_for_budget'])
            # Occupancy reclassification changes navigation, not whether a
            # cell has already been mapped, including the current anchor.
            changed = belief.copy(); changed[:,10] = 0; changed[7,5] = 1
            event = ledger.observe(changed, 2, True)
            self.assertEqual(event['actual_new_known_cells'], 0)
            self.assertEqual(event['known_lost_cells'], 0)
            self.assertEqual(ledger.snapshot()['planning_coverage'], 40/225)
            self.assertEqual(ledger.snapshot()['deficit_cells'], 140)
            self.assertEqual(ledger.snapshot()['conservative_rate'], 0.)

    def test_union_predictions_exclude_known_cells_and_outside_roi_and_lost_knowledge_is_debited(self):
        belief = np.full((4,5), -1, np.int8); belief[1,1] = 0; belief[1,2] = 1
        roi = np.zeros_like(belief, bool); roi[1:3,1:4] = True
        ledger = ObservedCoverageLedgerV21(belief.shape, (1,1), 0., task_mask=roi)
        ledger.start(belief)
        a = np.zeros_like(roi); a[2,1] = a[1,1] = a[0,0] = True
        b = a.copy(); b[2,2] = b[1,2] = True
        proposal = union_unknown_mask(belief, a, b, a, task_mask=roi)
        self.assertEqual(ledger.predict_gain(proposal)['predicted_union_cells'], 2)
        self.assertEqual(ledger.predict_gain(proposal)['discounted_known_gain_cells'], .5)
        seen = belief.copy(); seen[2,1] = 1; seen[0,0] = 0
        event = ledger.observe(seen, 1, True, predicted_mask=proposal)
        self.assertEqual(event['actual_new_known_cells'], 1)
        self.assertEqual(event['realized_predicted_cells'], 1)
        self.assertEqual(ledger.snapshot()['known_cells'], 3)
        self.assertEqual(ledger.snapshot()['union_yield'], .5)
        event = ledger.observe(seen, 2, True, predicted_mask=proposal)
        self.assertEqual(event['predicted_union_cells'], 1)
        self.assertEqual(event['actual_new_known_cells'], 0)
        self.assertEqual(ledger.snapshot()['union_yield'], .4)
        seen[2,1] = -1
        event = ledger.observe(seen, 3, True)
        self.assertEqual(event['known_lost_cells'], 1)
        self.assertEqual(event['net_known_gain_cells'], -1)
        self.assertEqual(ledger.snapshot()['known_cells'], 2)
        self.assertEqual(ledger.snapshot()['conservative_rate'], 0.)
        self.assertEqual(ledger.snapshot()['rate_history_net_known_gain'], 0)
        # Re-observation is new evidence after a real loss, not double credit.
        seen[2,1] = 0; ledger.observe(seen, 4, True)
        self.assertEqual(ledger.snapshot()['conservative_rate'], .5/4)
        self.assertEqual(ledger.snapshot()['known_cells'], 3)

    def test_paid_turns_zero_gain_steps_and_short_prefix_stay_in_rate(self):
        initial = np.full((10,10), -1, np.int8); initial[5,5] = 0
        ledger = ObservedCoverageLedgerV21(initial.shape, (5,5), 0.)
        self.assertIsNone(ledger.start(initial)['conservative_rate'])
        seen = initial.copy(); seen[5,6] = 0
        ledger.observe(seen, 1, True)
        # Turning or travelling without new map cells still consumes one action.
        for action in range(2,6): ledger.observe(seen, action, True)
        self.assertEqual(ledger.snapshot()['rate_history_paid_actions'], 5)
        self.assertEqual(ledger.snapshot()['conservative_rate'], .1)
        seen[5,7] = 1; ledger.observe(seen, 6, True)
        ledger.flush_prefix()
        self.assertIsNone(ledger.snapshot()['pending_prefix'])
        self.assertEqual(ledger.snapshot()['conservative_rate'], 1/6)
        ledger.observe(initial, 7, True)
        self.assertEqual(ledger.snapshot()['rate_history_net_known_gain'], 0)
        self.assertEqual(ledger.snapshot()['conservative_rate'], 0.)
        self.assertEqual(ledger.events[-1]['known_lost_cells'], 2)

    def test_all_budget_terms_use_known_cells_and_classes_do_not_change_admission(self):
        belief = np.full((10,10), -1, np.int8); belief[4,:] = 0
        ledger = ObservedCoverageLedgerV21(belief.shape, (4,4), 1.)
        ledger.start(belief)
        seen = belief.copy(); seen[5,:] = 1
        ledger.observe(seen, 1, True)
        snapshot = ledger.snapshot()
        self.assertEqual(snapshot['known_cells'], 20)
        self.assertEqual(snapshot['deficit_cells'], 60)
        self.assertEqual(snapshot['conservative_rate'], 5.)
        # New occupied observations count just like free observations. The
        # navigation anchor can be unsafe; budget passage never permits motion.
        mask = np.zeros_like(belief, bool); mask[6,:] = True
        route = dict(outbound_cost=3, return_cost=4, category_vote=1.)
        original_belief = ledger.belief.copy(); original_snapshot = deepcopy(snapshot)
        refused = ledger.assess_route(route, mask, 23)
        allowed = ledger.assess_route(route, mask, 24)
        self.assertEqual(allowed['predicted_gain_cells'], 2.5)
        self.assertEqual(allowed['coverage_reserve_actions'], 12)
        self.assertEqual(allowed['total_required_actions'], 24)
        self.assertTrue(allowed['allowed']); self.assertFalse(refused['allowed'])
        route['category_vote'] = -1.
        self.assertEqual(allowed, ledger.assess_route(route, mask, 24))
        self.assertFalse(allowed['movement_authorized'])
        self.assertFalse(allowed['evaluation_truth_used'])
        self.assertEqual(original_snapshot, ledger.snapshot())
        np.testing.assert_array_equal(original_belief, ledger.belief)

    def test_quality_intent_changes_only_rate_history_not_map_credit_or_prediction_feedback(self):
        belief = np.full((10,10), -1, np.int8); belief[4,:] = 0
        ledger = ObservedCoverageLedgerV21(belief.shape, (4,4), 0.)
        ledger.start(belief)
        mask = np.zeros_like(belief, bool); mask[5,:] = True
        seen = belief.copy(); seen[5,:] = 0
        ledger.observe(seen, 1, False, predicted_mask=mask)
        self.assertEqual(ledger.snapshot()['known_cells'], 20)
        self.assertEqual(ledger.snapshot()['union_yield'], 11/12)
        self.assertIsNone(ledger.snapshot()['conservative_rate'])
        ledger.observe(seen, 2, True)
        self.assertEqual(ledger.snapshot()['conservative_rate'], 0.)
        seen[6,:4] = 1; ledger.observe(seen, 3, True)
        self.assertEqual(ledger.snapshot()['conservative_rate'], 1.)
        seen[7,:5] = 0; ledger.observe(seen, 4, False)
        snapshot = ledger.snapshot()
        self.assertEqual(snapshot['known_cells'], 29)
        self.assertEqual(snapshot['deficit_cells'], 51)
        self.assertEqual(snapshot['conservative_rate'], 1.)
        self.assertEqual(snapshot['rate_history_paid_actions'], 2)

    def test_zero_rate_is_explicit_but_met_roi_target_needs_no_future_coverage_reserve(self):
        belief = np.full((5,5), -1, np.int8); belief[2,2] = 0
        ledger = ObservedCoverageLedgerV21(belief.shape, (2,2), 1.)
        ledger.start(belief); mask = np.zeros_like(belief, bool)
        route = dict(outbound_cost=1, return_cost=1)
        self.assertEqual(ledger.assess_route(route, mask, 100)['reason'], 'coverage_rate_unavailable')
        ledger.observe(belief, 1, True)
        unavailable = ledger.assess_route(route, mask, 100)
        self.assertEqual(unavailable['reason'], 'coverage_rate_unavailable')
        self.assertIsNone(unavailable['coverage_reserve_actions'])
        json.dumps(unavailable, allow_nan=False)
        full = np.ones_like(belief); full[2,2] = 0
        achieved = ObservedCoverageLedgerV21(full.shape, (2,2), 2.)
        achieved.start(full)
        answer = achieved.assess_route(route, mask, 7)
        self.assertTrue(answer['allowed']); self.assertEqual(answer['coverage_reserve_actions'], 0)
        self.assertFalse(answer['movement_authorized'])
        self.assertFalse(answer['true_coverage_guaranteed'])
        self.assertFalse(achieved.snapshot()['proxy_is_true_reachable_coverage'])
        self.assertTrue(achieved.snapshot()['closed_interiors_may_be_unobservable'])

    def test_fixed_roi_and_state_do_not_share_mutable_caller_memory_and_duplicate_paid_id_is_rejected(self):
        belief = np.full((8,8), -1, np.int8); belief[3:5,3:5] = 0
        roi = np.ones_like(belief, bool)
        ledger = ObservedCoverageLedgerV21(belief.shape, (3,3), 0., task_mask=roi)
        ledger.start(belief); before = deepcopy(ledger.snapshot())
        belief[:] = 1; roi[:] = False
        self.assertEqual(ledger.snapshot(), before)
        snapshot = ledger.snapshot(); snapshot['navigation_diagnostics']['C_plan'] = 99.
        self.assertEqual(ledger.snapshot(), before)
        ledger.observe(ledger.belief.copy(), 1, True)
        before = deepcopy(ledger.snapshot())
        with self.assertRaisesRegex(ValueError, 'exactly once'):
            ledger.observe(ledger.belief, 1, True, radar_hits=np.ones_like(roi))
        self.assertEqual(ledger.snapshot(), before)
        with self.assertRaisesRegex(ValueError, 'at least one'):
            ObservedCoverageLedgerV21(belief.shape, (3,3), 0., task_mask=roi)


if __name__ == '__main__': unittest.main()
