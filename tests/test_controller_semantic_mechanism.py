"""Paid-route and observed-only integration checks; no simulated World."""
import json
import unittest
from functools import lru_cache
from unittest.mock import patch

import numpy as np

from nso.controller_semantic_mechanism import (
    SemanticMechanismController, paid_followup_options, paid_diagnostic_value, select_unique_target,
)
from nso.primitive_navigation_v41 import PrimitiveStateV41
from nso.semantic_reliability import SemanticReliabilityBelief
from test_controller_v43 import graph, mapper, packet, after_action


def controller(method='G', **kwargs):
    values = dict(home=PrimitiveStateV41('home', 0), budget=48,
        palette={'cabinet': [40, 100, 220]},
        structure_names=['planar', 'recessed', 'louvered', 'open_frame'],
        class_structure_prior={'cabinet': [.1, .7, .1, .1]},
        maximum_candidates=4, maximum_views_per_instance=4)
    values.update(kwargs)
    return SemanticMechanismController(graph(), method=method, **values)


class SemanticMechanismControllerTests(unittest.TestCase):
    def test_selection_route_cache_preserves_paid_macro_options(self):
        public=graph()
        args=dict(current=PrimitiveStateV41('home',0),diagnostic=PrimitiveStateV41('advance',0),
                  home=PrimitiveStateV41('home',0),remaining=48,
                  candidates=[PrimitiveStateV41('north',3),PrimitiveStateV41('south',9)])
        expected=paid_followup_options(public,**args)
        with patch.object(public,'route',wraps=public.route) as route:
            query=lru_cache(maxsize=None)(route)
            first=paid_followup_options(public,route_query=query,**args)
            calls=route.call_count
            second=paid_followup_options(public,route_query=query,**args)
            self.assertEqual(route.call_count,calls)
        self.assertEqual(first,expected)
        self.assertEqual(second,expected)

    def test_random_ties_give_one_vote_per_executable_target(self):
        a, b = PrimitiveStateV41('a', 0), PrimitiveStateV41('b', 0)
        direct_a = (1., a, {'kind': 'direct'})
        direct_b = (1., b, {'kind': 'direct'})
        repeated = [direct_a, (1., a, {'kind': 'diagnose', 'instance_id': 'i'}),
                    (1., a, {'kind': 'diagnose', 'instance_id': 'j'}), direct_b]
        for seed in range(10):
            baseline = select_unique_target([direct_a, direct_b], tie_rule='seeded_random',
                                            rng=np.random.default_rng(seed))
            duplicate = select_unique_target(repeated, tie_rule='seeded_random',
                                             rng=np.random.default_rng(seed))
            self.assertEqual(baseline[1], duplicate[1])
        # A higher-valued diagnostic must still set its target's value.
        better = select_unique_target(repeated + [(2., a, {'kind': 'diagnose'})],
                                      tie_rule='reverse', rng=np.random.default_rng(0))
        self.assertEqual(better[0:2], (2., a))

    def test_route_budget_includes_two_paid_observations_and_return_heading(self):
        public = graph()
        home, diagnostic, last = (PrimitiveStateV41('home', 0),
            PrimitiveStateV41('advance', 0), PrimitiveStateV41('north', 3))
        routes = [public.route(home, diagnostic), public.route(diagnostic, last), public.route(last, home)]
        actions = routes[0].actions + ('observe',) + routes[1].actions + ('observe',) + routes[2].actions
        state = home
        for action in actions:
            state = public.successor(state, action)
        self.assertEqual(state, home)
        for remaining, count in ((len(actions)-1, 0), (len(actions), 1)):
            rows = paid_followup_options(public, current=home, diagnostic=diagnostic,
                home=home, remaining=remaining, candidates=[diagnostic, last])
            self.assertEqual(len(rows), count)
            if rows:
                self.assertEqual(rows[0]['total_cost'], len(actions))

    def test_paid_information_score_accounts_for_cost_and_preserves_geometry_evi(self):
        belief = SemanticReliabilityBelief(geometry_prior=[.5, .5],
            class_structure_priors={'c': [.9, .1]}, share_across_instances=False)
        belief.register('observed', None)  # No semantic input is needed for geometry EVI.
        gains = {'observed': {'a': [10., 0.], 'b': [0., 10.]}}
        rows = [{'target': 'a', 'total_cost': 20}, {'target': 'b', 'total_cost': 20}]
        value = paid_diagnostic_value(belief, instance_id='observed', channel=np.eye(2),
            followups=rows, structure_gains=gains, discovery_gains={'a': 0., 'b': 0.})
        self.assertAlmostEqual(value['before_observation_score'], .25)
        self.assertAlmostEqual(value['score'], .5)
        slower = paid_diagnostic_value(belief, instance_id='observed', channel=np.eye(2),
            followups=[dict(row, total_cost=40) for row in rows], structure_gains=gains,
            discovery_gains={'a': 0., 'b': 0.})
        self.assertAlmostEqual(slower['score'], value['score']/2)
        no_future = paid_diagnostic_value(belief, instance_id='observed', channel=np.eye(2),
            followups=rows, structure_gains=gains, discovery_gains={'a': 0., 'b': 0.}, allow_future=False)
        self.assertAlmostEqual(no_future['score'], .25)

    def test_macro_is_committed_until_actual_extra_observe_is_accepted(self):
        c, mapping = controller(no_progress_patience=1), mapper()
        state = PrimitiveStateV41('home', 0)
        observation = packet(); mapping.update(observation); c.accept(observation, mapping)
        target = PrimitiveStateV41('advance', 0)
        with patch.object(c, '_select_global', return_value=target) as select:
            first = c.choose()
            self.assertEqual(first['action'], 'forward')
            state, observation = after_action(c._graph, state, first['action'], 1)
            mapping.update(observation); c.accept(observation, mapping)
            second = c.choose()
            self.assertEqual(second['action'], 'observe')
            self.assertFalse(second['global_replanned'])
            self.assertEqual(select.call_count, 1)
            state, observation = after_action(c._graph, state, second['action'], 2)
            mapping.update(observation); receipt = c.accept(observation, mapping)
            self.assertEqual(receipt['completed_macro_id'], 1)
            self.assertIsNone(c.snapshot()['macro_target'])

    def test_all_methods_share_action_zero_candidates_without_semantics(self):
        selections = []
        for method in ('G', 'bayes_semantic', 'shared_semantic'):
            c, mapping, observation = controller(method), mapper(), packet()
            mapping.update(observation); c.accept(observation, mapping)
            choice = c.choose()
            selections.append(choice)
            self.assertTrue(choice['global_replanned'])
            self.assertFalse(choice['ground_truth_scene_input'])
            json.dumps(choice, allow_nan=False)
        for choice in selections[1:]:
            self.assertEqual(choice['action'], selections[0]['action'])
            self.assertEqual(choice['global_selection'], selections[0]['global_selection'])

    def test_observed_geometry_and_qualified_single_instance_have_matched_initial_priors(self):
        controllers = [controller(method) for method in ('bayes_semantic', 'shared_semantic')]
        maps = [mapper(), mapper()]
        target = PrimitiveStateV41('advance', 0)
        for c, mapping in zip(controllers, maps):
            observation = packet(measured_plane=True)
            mapping.update(observation); c.accept(observation, mapping)
            with patch.object(c, '_select_global', return_value=target):
                decision = c.choose()
            state, observation = after_action(c._graph, PrimitiveStateV41('home', 0),
                decision['action'], 1, measured_plane=True)
            mapping.update(observation); c.accept(observation, mapping)
            instances = c._planning_instances()
            self.assertEqual(len(instances), 1)
            self.assertTrue(instances[0]['semantic_conditioning_used'])
            # Exercise the original observed-support and posterior consistency validation.
            c._predictor.forecast(instances[0], [c._candidate_view(target)])
        self.assertEqual(controllers[0]._planning_instances(), controllers[1]._planning_instances())

    def test_feedback_and_future_ablations_are_distinct_configuration(self):
        actual = controller('shared_no_feedback')
        future = controller('shared_no_future')
        self.assertFalse(actual._actual_feedback)
        self.assertTrue(actual._future_information)
        self.assertTrue(future._actual_feedback)
        self.assertFalse(future._future_information)
        with self.assertRaises(ValueError):
            controller(uncertainty_penalty=.5)

    def test_shared_belief_failure_after_paid_accept_poisoning_prevents_retry(self):
        c, mapping, observation = controller('shared_semantic'), mapper(), packet()
        mapping.update(observation); c.accept(observation, mapping)
        target = PrimitiveStateV41('advance', 0)
        with patch.object(c, '_select_global', return_value=target):
            decision = c.choose()
        _, observation = after_action(c._graph, PrimitiveStateV41('home', 0),
            decision['action'], 1, measured_plane=True)
        mapping.update(observation)
        original_update = c._belief.replace_log_evidence
        failure = RuntimeError('injected failure after shared-belief mutation')

        def mutate_then_fail(key, scores):
            original_update(key, scores)
            raise failure

        with patch.object(c._belief, 'replace_log_evidence', side_effect=mutate_then_fail):
            with self.assertRaises(RuntimeError) as raised:
                c.accept(observation, mapping)
        self.assertIs(raised.exception, failure)
        snapshot = c.snapshot()
        self.assertEqual(snapshot['paid_step'], 1)
        self.assertEqual(snapshot['last_observation_sha256'], observation.sha256())
        self.assertEqual(len(c._belief.instance_ids), 1)
        self.assertIsNone(snapshot['pending_action'])
        self.assertTrue(snapshot['poisoned'])
        with self.assertRaisesRegex(RuntimeError, 'controller transaction failed'):
            c.choose()
        with self.assertRaisesRegex(RuntimeError, 'controller transaction failed'):
            c.accept(observation, mapping)

    def test_current_footprint_veto_records_terminal_receipt_without_paid_action(self):
        c, mapping, observation = controller('shared_semantic'), mapper(), packet()
        mapping.update(observation); c.accept(observation, mapping)
        target = PrimitiveStateV41('advance', 0)
        with patch.object(c, '_select_global', return_value=target):
            first = c.choose()
        _, observation = after_action(c._graph, PrimitiveStateV41('home', 0),
            first['action'], 1)
        mapping.update(observation); c.accept(observation, mapping)
        before = c.snapshot()
        self.assertIsNotNone(before['macro_target'])
        guard = dict(allowed=False, action='blocked', reason='observed_footprint_conflict')
        with patch.object(c._safety, 'guard_action', return_value=guard), \
                patch.object(c, '_select_global') as select, \
                patch.object(c._router, 'choose') as route:
            decision = c.choose()
        select.assert_not_called()
        route.assert_not_called()
        self.assertEqual(set(decision), set(first))
        self.assertEqual(decision['schema'], 'mechanism.controller_decision.v1')
        self.assertEqual(decision['action'], 'blocked')
        self.assertEqual(decision['reason'], 'observed_current_footprint_conflict')
        self.assertEqual(decision['configuration_sha256'], before['configuration_sha256'])
        self.assertEqual(decision['method'], 'shared_semantic')
        self.assertEqual(decision['source_observation_sha256'], observation.sha256())
        self.assertEqual(decision['paid_step'], before['paid_step'])
        self.assertFalse(decision['global_replanned'])
        self.assertFalse(decision['macro_observe_submitted'])
        self.assertIsNone(decision['routing'])
        self.assertIsNone(decision['macro_target'])
        self.assertEqual(c._last_choose, decision)
        json.dumps(decision, allow_nan=False)
        # Returned receipts must not provide a mutable alias to the stored one.
        decision['safety_guard']['allowed'] = True
        self.assertFalse(c._last_choose['safety_guard']['allowed'])
        after = c.snapshot()
        self.assertEqual(after['paid_step'], before['paid_step'])
        self.assertEqual(after['current_state'], before['current_state'])
        self.assertIsNone(after['pending_action'])
        self.assertIsNone(after['macro_target'])
        self.assertFalse(after['macro_observe_pending'])
        self.assertTrue(after['terminal'])
        with self.assertRaisesRegex(RuntimeError, 'controller is terminal'):
            c.choose()


if __name__ == '__main__':
    unittest.main()
