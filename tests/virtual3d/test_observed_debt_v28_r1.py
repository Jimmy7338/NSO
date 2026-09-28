"""Mechanism and information-boundary tests, without a simulated world."""
from dataclasses import replace
import unittest
import numpy as np
from nso.observed_state_v26 import GeometryPatchV26
from nso.observed_planner_v26 import _xy, _sector

from nso.observed_debt_v28_r1 import (
    DebtContextV28, build_context, describe_neighborhood, directional_weights, rank_context)
from nso.execution_guard_v8 import ObservedExecutionGuard
from test_observed_planner_v26 import state_for, cue_for


class ObservedDebtV28R1Tests(unittest.TestCase):
    def test_descriptors_ignore_class_and_reject_future_support(self):
        state = state_for()
        first, _ = describe_neighborhood(state, cue_for(2))
        second, _ = describe_neighborhood(state, cue_for(3))
        self.assertEqual(first, second)
        self.assertIsNone(first['effective_baseline_m'])
        self.assertEqual(first['median_observation_frames'], 1.)
        self.assertEqual(first['median_point_plane_residual_m2'], 0.)
        with self.assertRaises(ValueError):
            describe_neighborhood(state, cue_for(action=1))
        with self.assertRaises(ValueError):
            build_context(state, replace(state, action_id=1), ())
        with self.assertRaises(ValueError):
            build_context(state, replace(state, patches=()), ())

    def test_prior_has_equal_mass_and_objectness_ignores_class(self):
        a, b = directional_weights(cue_for(2)), directional_weights(cue_for(3))
        self.assertAlmostEqual(float(a.sum()), 1.)
        self.assertAlmostEqual(float(b.sum()), 1.)
        self.assertFalse(np.allclose(a, b))
        np.testing.assert_array_equal(directional_weights(cue_for(2), False),
                                      directional_weights(cue_for(3), False))

    def test_real_pool_is_safe_paid_and_class_independent(self):
        state = state_for(budget=36)
        context = build_context(state, state, (cue_for(3),))
        changed = build_context(state, state, (cue_for(2),), context.geometry_plan)
        self.assertEqual(context.pool, changed.pool)
        self.assertEqual(context.descriptors, changed.descriptors)
        self.assertEqual(rank_context(context, 'O'), rank_context(changed, 'O'))
        safe = ObservedExecutionGuard(.2, .2).safe_grid(state.belief)
        for mode in ('O', 'S', 'X'):
            answer = rank_context(context, mode)
            self.assertLessEqual(len(answer['candidates']), 12)
            for route in answer['candidates']:
                self.assertLessEqual(route['cost'], state.remaining_budget)
                self.assertEqual(route['cost'], len(route['actions']))
                self.assertEqual(route['states'][-1], list(state.anchor))
                self.assertTrue(all(safe[tuple(p[:2])] for p in route['states']))

    def test_actual_low_confidence_cues_fall_back_exactly(self):
        state = state_for()
        low = replace(cue_for(), confidence=.59)
        context = build_context(state, state, (low,))
        self.assertEqual(context.semantic_route_queries, 0)
        for mode in ('G', 'O', 'S', 'X'):
            result = rank_context(context, mode)
            result.pop('v28_audit')
            self.assertEqual(result, context.geometry_plan)

    def test_reservation_is_symmetric_but_does_not_force_selection(self):
        state = state_for()
        cues = (cue_for(2), replace(cue_for(3), cue_id='observed-cue-1'))
        descriptors = {c.cue_id: describe_neighborhood(state, c)[0] for c in cues}
        rows, geometry = [], {}
        for i in range(14):
            pose = (i, 10, 0)
            rows.append(dict(pose=list(pose), group='coverage_test' if i < 4 else 'test',
                geometry_score=100.-i if i < 12 else 0., outbound_cost=1, cost=2))
            geometry[pose] = {} if i < 12 else {cues[i-12].cue_id:
                dict(sector=1, visible_patches=6, local_patches=6, visible_fraction=1., novel_direction_patches=6, candidate_direction_deficit=1.)}
        context = DebtContextV28(state, cues, descriptors, {}, tuple(rows), geometry, 0)
        reserved = rank_context(context, 'S')
        unreserved = rank_context(context, 'S', reserve_instances=False)
        self.assertEqual(reserved['v28_audit']['retained_cue_ids'], [c.cue_id for c in cues])
        self.assertEqual(unreserved['v28_audit']['retained_cue_ids'], [])
        self.assertEqual(reserved['v28_audit']['coverage_routes_retained'], 4)
        self.assertEqual(reserved['selected']['pose'], [0, 10, 0])
        self.assertEqual(reserved['v28_audit']['selected_cue_ids'], [])
        self.assertEqual(len(reserved['candidates']), 12)
        scores = lambda mode: {tuple(r['pose']): r['score'] for r in rank_context(context, mode)['candidates']}
        self.assertNotEqual(scores('S'), scores('X'))


    def _same_camera_context(self, invisible_patch=False):
        state = state_for()
        pose = (28, 24, 0)
        camera = _xy(state, pose)
        patches = [GeometryPatchV26((i, 1, 1), (x, 4.3, .8), (0., -1., 0.), 3,
            1 << _sector(camera-np.array([x, 4.3])), 2., 0.)
            for i, x in enumerate((4., 4.2, 4.4, 4.6, 5., 5.2))]
        if invisible_patch:
            patches.append(GeometryPatchV26((9, 1, 1), (3., 3.7, .8), (0., -1., 0.), 1, 1, 2., 0.))
        state = replace(state, patches=tuple(patches))
        context = build_context(state, state, (cue_for(3),))
        route = next(r for r in context.pool if tuple(r['pose']) == pose)
        return replace(context, pool=(route,)), pose

    def test_identical_camera_cannot_earn_new_direction_from_cue_origin(self):
        context, pose = self._same_camera_context()
        geometry = context.cue_geometry[pose][cue_for().cue_id]
        self.assertEqual(geometry['visible_patches'], 6)
        self.assertEqual(geometry['novel_direction_patches'], 0)
        self.assertEqual(geometry['candidate_direction_deficit'], 0.)
        for mode in ('O', 'S', 'X'):
            result = rank_context(context, mode)
            self.assertEqual(result['candidates'][0]['semantic_gain'], 0.)
            self.assertEqual(result['v28_audit']['eligible_cue_ids'], [])

    def test_invisible_patch_debt_cannot_reward_seen_visible_patches(self):
        context, pose = self._same_camera_context(invisible_patch=True)
        geometry = context.cue_geometry[pose][cue_for().cue_id]
        self.assertEqual(geometry['local_patches'], 7)
        self.assertEqual(geometry['visible_patches'], 6)
        self.assertEqual(geometry['novel_direction_patches'], 0)
        self.assertEqual(rank_context(context, 'S')['candidates'][0]['semantic_gain'], 0.)


if __name__ == '__main__':
    unittest.main()
