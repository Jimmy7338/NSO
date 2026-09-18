"""Prevent class leakage, fake coverage options and first-choice misattribution."""
from copy import deepcopy
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import numpy as np

from nso.facility_runtime_v22 import initial_option_map, FacilityRuntimeV22
from nso.recovery_runtime_v14 import RecoveryRuntimeV14


class InitialCoverageChoiceTests(unittest.TestCase):
    def fixture(self):
        routes = [dict(candidate_id=i, group='coverage_v20', outbound_cost=2,
            outbound_actions=['forward', 'forward'], return_cost=4,
            return_actions=['left', 'left', 'forward', 'forward'], cost=6)
            for i in range(3)]
        assets = [dict(marked_points=2, aabb_center=[1., 3., 1.], class_vote=-1.),
                  dict(marked_points=3, aabb_center=[5., 3., 1.], class_vote=1.)]
        responses = [dict(per_asset_expected_precision_change=value)
                     for value in ([.1, .8], [.5, .1], [.9, .2])]
        return routes, [3., 1., 2.], assets, responses

    def test_votes_cannot_change_routes_and_N_B_alias_is_preserved(self):
        routes, scores, assets, responses = self.fixture()
        expected = ({'N': 0, 'A': 2, 'B': 0}, [0, 1])
        self.assertEqual(initial_option_map(routes, scores, assets, responses), expected)
        for asset in assets: asset['class_vote'] *= -1
        self.assertEqual(initial_option_map(routes, scores, assets, responses), expected)

    def test_huge_quality_or_zero_coverage_does_not_create_an_admissible_route(self):
        routes, scores, assets, responses = self.fixture()
        routes[2]['group'] = 'asset_0_deep'
        responses[2]['per_asset_expected_precision_change'] = [1000., 1000.]
        self.assertEqual(initial_option_map(routes, scores, assets, responses)[0]['A'], 1)
        routes[2]['group'] = 'coverage_v20'; scores[2] = 0.
        self.assertEqual(initial_option_map(routes, scores, assets, responses)[0]['A'], 1)

    def test_missing_opportunity_fails_instead_of_claiming_distinct_AB_options(self):
        routes, scores, assets, responses = self.fixture()
        for row in responses: row['per_asset_expected_precision_change'][0] = 0.
        with self.assertRaisesRegex(ValueError, 'No positive observed precision'):
            initial_option_map(routes, scores, assets, responses)

    def test_refreshed_route_retains_identity_and_uses_original_guard_once(self):
        option = dict(option_id='episode/plan-1', selection_call_id=8, pose=[3, 5, 1],
            group='coverage_v20', candidate_id=2, outbound_actions=['left', 'forward'],
            v21_coverage_intent=True, route_revision=0, v22_first_option='A',
            v22_observed_target_index=0, asset_index=None)
        fresh = {**deepcopy(option), 'outbound_actions':['forward'], 'outbound_cost':1, 'return_cost':4}
        backend_state = dict(selected=deepcopy(option), ledger=SimpleNamespace(remaining_budget=20),
            coverage_v21=SimpleNamespace(assess_route=lambda *args: {'allowed':False}), region_continuation=None)
        space = SimpleNamespace(refresh=lambda old: deepcopy(fresh), route_mask=lambda route:np.ones((3, 3), bool))
        backend = SimpleNamespace(scenes=[backend_state], route_space=lambda index:space, _record=lambda *args:None)
        runtime = object.__new__(FacilityRuntimeV22)
        runtime.args=SimpleNamespace(cpu_v20_replan_interval=5)
        runtime.components=SimpleNamespace(_cpu_backend=backend); runtime.full_shape=(8,8);runtime.audit=[]
        runtime.states=[dict(closed=False, pending=None, phase='outbound',
            packet=SimpleNamespace(action_id=5,position=(3,4),heading=1),mapper=SimpleNamespace(frames=6),
            option=option,active_actions=['left','forward'],v21_last_route_update=0)]
        with patch.object(RecoveryRuntimeV14, 'next_local_action', return_value='guarded') as guard:
            self.assertEqual(runtime.next_local_action(0),'guarded')
        guard.assert_called_once_with(runtime,0)
        selected = runtime.states[0]['option']
        for key in ('pose','option_id','selection_call_id','v22_first_option','v22_observed_target_index','asset_index'):
            self.assertEqual(selected[key],option[key])
        self.assertEqual(selected['route_revision'],1)
        # An original coverage route stays under the N rule, not the G quality gate.
        self.assertTrue(runtime.audit[-1]['target_retained'])
        space.route_mask=lambda route:np.zeros((3,3),bool)
        runtime.states[0]['packet'].action_id=10
        with patch.object(RecoveryRuntimeV14,'next_local_action',return_value=None):
            runtime.next_local_action(0)
        self.assertIsNone(runtime.states[0]['option'])
        self.assertEqual(runtime.audit[-1]['reason'],'no_remaining_unique_coverage_gain')


if __name__ == '__main__': unittest.main()
