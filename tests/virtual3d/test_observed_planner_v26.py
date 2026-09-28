"""Counterfactual information, paid-return, and feedback causality checks.

Handwritten observed maps only: no world construction or physical episodes.
"""
from dataclasses import replace
import unittest
import numpy as np
from nso.observed_state_v26 import GeometryStateV26, GeometryPatchV26, SemanticCueV26
from nso.observed_planner_v26 import ObservedPlannerV26
from nso.execution_guard_v8 import ObservedExecutionGuard
from utils.grid_geometry import DIRECTIONS


def state_for(*, action=0, position=(33, 21), heading=0, budget=100, patches=True, belief=None):
    if belief is None:
        belief = np.zeros((43, 43), np.int8)
        belief[[0, -1], :] = 1; belief[:, [0, -1]] = 1
        belief[4:10, 5:13] = -1
        belief[20:23, 21] = 1
    support = tuple(GeometryPatchV26((i, 1, 1), (4.3+.04*i, 4.3, .8+.04*i),
        (0., -1., 0.), 1, 1, 3., 0.) for i in range(6)) if patches else ()
    return GeometryStateV26(belief, .2, .2, 5., 90., 96, 72, .8,
        position, heading, (33, 21, 0), budget, action, support, 'handwritten-observation-fixture')


def cue_for(class_id=3, action=0):
    return SemanticCueV26('observed-cue-0', (4.3, 4.3, .8), (0., -1., 0.),
                          class_id, 1., action, 'observed_artificial_RGB_marker')


class ObservedPlannerV26Tests(unittest.TestCase):
    def test_geometry_can_generate_revisits_without_marker_or_class(self):
        state = state_for()
        result = ObservedPlannerV26('G').plan(state)
        self.assertTrue(result['candidates'])
        self.assertTrue(any(r['group'] == 'observed_geometry_revisit' for r in result['candidates']))
        self.assertEqual(result['audit']['semantic_proposal_queries'], 0)
        self.assertEqual(result['audit']['semantic_cue_count'], 0)
        for row in result['candidates']:
            self.assertFalse(row['semantic_evidence'])
            self.assertEqual(row['semantic_gain'], 0)
            self.assertTrue(all('class_id' not in src and 'cue_id' not in src for src in row['sources']))
        with self.assertRaises(ValueError):
            ObservedPlannerV26('G').plan(state, (cue_for(),))

    def test_ranking_ablation_uses_identical_final_geometry_pool(self):
        state = state_for()
        g = ObservedPlannerV26('G').plan(state)
        gr = ObservedPlannerV26('GR').plan(state, (cue_for(),))
        self.assertEqual({tuple(r['pose']) for r in g['candidates']},
                         {tuple(r['pose']) for r in gr['candidates']})
        paths = {tuple(r['pose']): r['actions'] for r in g['candidates']}
        for route in gr['candidates']:
            self.assertEqual(paths[tuple(route['pose'])], route['actions'])

    def test_semantic_proposals_require_observed_cue_and_respect_capacity(self):
        state = state_for()
        no_cue = ObservedPlannerV26('S').plan(state)
        with_cue = ObservedPlannerV26('S').plan(state, (cue_for(),))
        self.assertEqual(no_cue['audit']['semantic_proposal_queries'], 0)
        self.assertGreater(with_cue['audit']['semantic_proposal_queries'], 0)
        for mode in ('G', 'GP', 'GR', 'S', 'S_no_feedback'):
            result = ObservedPlannerV26(mode).plan(state, () if mode == 'G' else (cue_for(),))
            self.assertLessEqual(len(result['candidates']), 12)
        with self.assertRaises(ValueError):
            ObservedPlannerV26('S').plan(state, (cue_for(action=1),))

    def test_all_routes_pay_turns_return_and_never_traverse_unknown(self):
        state = state_for(budget=36)
        result = ObservedPlannerV26('S').plan(state, (cue_for(),))
        safe = ObservedExecutionGuard(.2, .2).safe_grid(state.belief)
        for route in result['candidates']:
            self.assertLessEqual(route['cost'], 36)
            self.assertEqual(route['cost'], len(route['actions']))
            self.assertEqual(route['states'][-1], list(state.anchor))
            pose = (*state.position, state.heading)
            for action, following in zip(route['actions'], route['states'][1:]):
                r, c, h = pose
                if action == 'forward':
                    dr, dc = DIRECTIONS[h]; pose = (r+dr, c+dc, h)
                else:
                    pose = (r, c, (h+(1 if action == 'right' else -1)) % 4)
                self.assertEqual(list(pose), following)
                self.assertTrue(safe[pose[:2]])

    def test_unknown_or_unaffordable_targets_fall_back_to_paid_return(self):
        state = state_for(position=(32, 21), budget=5)
        planner = ObservedPlannerV26('G')
        action = planner.local_action(state, dict(pose=[6, 6, 0], group='invalid_unknown'))
        self.assertIn(action, ('left', 'right', 'forward'))
        self.assertEqual(planner.calls[-1]['module'], 'RPN-UQ')
        blocked = replace(state, belief=np.full(state.belief.shape, -1, np.int8))
        self.assertIsNone(planner.local_action(blocked))

    def test_known_wall_blocks_semantic_template_reward(self):
        state = state_for()
        belief = state.belief.copy(); belief[27, :] = 1
        blocked = replace(state, belief=belief)
        gain, evidence = ObservedPlannerV26('S')._semantic_gain(blocked, (33, 21, 0), (cue_for(),), {})
        self.assertEqual(gain, 0.)
        self.assertEqual(evidence, [])

    def test_unrelated_candidate_or_sample_churn_cannot_create_feedback(self):
        before = state_for(position=(31, 21))
        after = replace(before, action_id=1, position=(30, 21), remaining_budget=99)
        planner = ObservedPlannerV26('S')
        selected = dict(pose=[30, 21, 0], semantic_evidence=[])
        event = planner.observe_transition(before, after, selected, (cue_for(),))
        self.assertEqual(event['directional_updates'], [])
        self.assertEqual(planner.feedback, {})
        with self.assertRaises(ValueError):
            planner.observe_transition(before, after, selected, (cue_for(),))
        with self.assertRaises(ValueError):
            ObservedPlannerV26('S').observe_transition(before, replace(after, action_id=2))

    def test_feedback_needs_arrival_prediction_and_comparable_measured_support(self):
        before = state_for(position=(31, 21))
        after = replace(before, action_id=1, position=(30, 21), remaining_budget=99,
            patches=tuple(replace(p, bits=5, best_range=2.) for p in before.patches))
        planner = ObservedPlannerV26('S')
        _, evidence = planner._semantic_gain(before, (30, 21, 0), (cue_for(),), {})
        self.assertTrue(evidence)
        selected = dict(pose=[30, 21, 0], semantic_evidence=evidence)
        event = planner.observe_transition(before, after, selected, (cue_for(),))
        self.assertTrue(event['directional_updates'])
        self.assertEqual(event['directional_updates'][0]['status'], 'updated')
        self.assertGreater(event['directional_updates'][0]['observed_yield_proxy'], 0)
        sparse = replace(after, patches=tuple(replace(p, key=(99+i, 1, 1))
            for i, p in enumerate(after.patches)))
        second = ObservedPlannerV26('S')
        event = second.observe_transition(before, sparse, selected, (cue_for(),))
        self.assertEqual(event['improved_common_patches'], 0)
        self.assertFalse(second.feedback)
        self.assertEqual(event['directional_updates'][0]['status'],
                         'unavailable_insufficient_common_measured_support')


if __name__ == '__main__':
    unittest.main()
