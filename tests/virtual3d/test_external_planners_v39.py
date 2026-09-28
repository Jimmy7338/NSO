"""Analytic common-graph checks; no World, TSDF, renderer or saved episodes."""
from types import SimpleNamespace
import unittest

import numpy as np

from nso.cpu_four_modules_v35 import ObservationV35
from nso.external_planners_v39 import ExternalControllerV39, ExternalPlannerV39
from nso.observation_belief_v35 import PublicTemplatesV35


K = np.array([[48., 0., 47.5], [0., 48., 35.5], [0., 0., 1.]])
POSES = tuple((x, 0, h) for x in range(3) for h in range(4))
NODE = {p: i for i, p in enumerate(POSES)}


def models():
    edges = []
    for x, y, h in POSES:
        row = [('left', NODE[(x, y, (h-1) % 4)]), ('right', NODE[(x, y, (h+1) % 4)])]
        nxt = x+(1 if h == 1 else -1 if h == 3 else 0)
        if h in (1, 3) and 0 <= nxt < 3:
            row.append(('forward', NODE[(nxt, y, h)]))
        edges.append(tuple(row))
    def terminal(mask):
        coverage = (mask >> 2).bit_count()/3
        return dict(coverage=coverage, surface=(mask & 3).bit_count()/2,
            joint=coverage*(mask & 3).bit_count()/2, feasible=coverage >= .8)
    return tuple(SimpleNamespace(poses=POSES, edges=tuple(edges), anchor=1,
        target_bits=2, floor_cells=((0, 0), (1, 0), (2, 0)),
        observed_masks=tuple((1 << (2+x)) | (1 << (x % 2)) for x, _, _ in POSES),
        terminal=terminal) for _ in range(2))


def observation(step, pose=(0, 0, 1), code=None, action=None):
    rgb = np.zeros((4, 4, 3), np.uint8)
    if code is not None:
        rgb[:] = [40, 100, 220] if code == 2 else [220, 60, 40]
    return ObservationV35(f'analytic-{step}', step, pose, action,
        np.ones((4, 4), np.float32), rgb, np.ones(4, np.float32))


class AnalyticLedger:
    """Geometry depends on the candidate's actual public camera position."""
    def snapshot(self):
        return dict(points=np.zeros((10, 3)), normal_valid=np.ones(10, bool))

    def score_view(self, intrinsic, world_from_camera, *, semantic_relevance, **kwargs):
        x = world_from_camera[0, 3]
        visible = x >= 1.
        return dict(geometry_gain=.05 if x >= 2. else 0.,
            semantic_gain=float(semantic_relevance) if visible else 0.,
            new_inspection_indices=[0, 1] if visible and semantic_relevance else [])


class ExternalPlannerV39Tests(unittest.TestCase):
    def planner(self, method='VISTA', registered=True):
        p = ExternalPlannerV39(models(), AnalyticLedger(), method=method,
            intrinsic=K, translation=(0., 0.))
        p.observe(observation(0, code=2 if registered else None))
        if registered:
            p.observe(observation(1, (1, 0, 1), code=2))
        return p

    def masks(self, p):
        return tuple(m.observed_masks[m.anchor] for m in p.models)

    def test_camera_calibration_has_correct_forward_and_rigid_rotation(self):
        p = self.planner()
        for h, forward in enumerate(((0, 1, 0), (1, 0, 0), (0, -1, 0), (-1, 0, 0))):
            t = p.camera_transform(NODE[(0, 0, h)])
            np.testing.assert_allclose(t[:3, 2], forward)
            np.testing.assert_allclose(t[:3, :3].T @ t[:3, :3], np.eye(3))
            self.assertAlmostEqual(np.linalg.det(t[:3, :3]), 1.)
            np.testing.assert_allclose(t[:3, 3], [0., 0., .9])

    def test_category_quorum_uses_distinct_xy_and_conflict_clears_membership(self):
        p = self.planner(registered=False)
        p.observe(observation(1, code=2))
        p.observe(observation(2, code=2))
        self.assertIsNone(p.registered_class)
        p.observe(observation(3, (1, 0, 1), code=2))
        self.assertEqual(p.registered_class, 2)
        p.observe(observation(4, (2, 0, 1), code=3))
        self.assertIsNone(p.registered_class)
        self.assertTrue(p.last_observation_receipt['class_conflict'])

    def test_structural_posterior_cannot_change_external_decision(self):
        for method in ('SWAP', 'VISTA'):
            a, b = self.planner(method), self.planner(method)
            ra = a.select(1, 12, self.masks(a), .001)
            rb = b.select(1, 12, self.masks(b), .999)
            self.assertEqual(ra['action'], rb['action'])
            self.assertEqual(ra['candidates'], rb['candidates'])
            self.assertFalse(ra['posterior_used_to_choose_action'])
            self.assertFalse(ra['class_conditioned_structure_forecast'])

    def test_full_paths_can_cross_zero_local_gain_and_reserve_return(self):
        p = self.planner()
        r = p.select(1, 12, self.masks(p), .5, excluded_information_nodes=range(12))
        remote = [c for c in r['candidates'] if c['destination'] == NODE[(2, 0, 1)]][0]
        self.assertEqual(remote['path_actions'], ['forward', 'forward'])
        self.assertGreater(remote['discounted_semantics'], 0.)
        for c in r['candidates']:
            self.assertLessEqual(c['outbound_cost']+c['reserved_return_cost'], 12)
            self.assertTrue(c['both_template_coverage_continuation'])
        self.assertEqual(r['excluded_information_nodes_not_forbidden'], 12)

    def test_exact_remaining_return_budget_forces_first_return_edge(self):
        p = self.planner()
        current = NODE[(2, 0, 1)]
        remaining = p.return_distance[current]
        all_masks = tuple((1 << 5)-1 for _ in p.models)
        r = p.select(current, remaining, all_masks, .9)
        following = dict(p.edges[current])[r['action']]
        self.assertEqual(p.return_distance[following], remaining-1)

    def test_swap_macro_union_does_not_count_same_surfel_twice(self):
        p = self.planner('SWAP')
        r = p.select(1, 12, self.masks(p), .5)
        target = next(c for c in r['candidates'] if c['destination'] == NODE[(2, 0, 1)])
        self.assertEqual(target['inspection_new_surfel_count'], 2)
        self.assertAlmostEqual(target['inspection_gain'], .2)
        self.assertAlmostEqual(target['inspection_score'], .1)
        self.assertEqual(r['mechanism_phase'], 'inspection')

    def test_without_category_swap_still_explores_and_vista_semantics_not_fabricated(self):
        for method in ('SWAP', 'VISTA'):
            p = self.planner(method, registered=False)
            r = p.select(1, 12, self.masks(p), .5)
            self.assertIsNotNone(r['action'])
            self.assertFalse(r['semantic_term_nonzero'])
            self.assertEqual(r['positive_semantic_views'], 0)
            self.assertTrue(any(c['public_support_gain_sum'] > 0 for c in r['candidates']))
            if method == 'SWAP':
                self.assertEqual(r['mechanism_phase'], 'geometry_exploration')

    def test_vista_reverse_path_discount_and_actual_positive_semantic_activation(self):
        p = self.planner()
        r = p.select(1, 12, self.masks(p), .5)
        target = next(c for c in r['candidates'] if c['destination'] == NODE[(2, 0, 1)])
        self.assertAlmostEqual(target['discounted_semantics'], .8+1.)
        self.assertTrue(r['semantic_term_nonzero'])
        self.assertGreater(r['positive_semantic_views'], 0)
        self.assertGreater(r['semantic_candidate_range'][1], r['semantic_candidate_range'][0])
        self.assertGreater(r['measured_direction_candidate_range'][1], r['measured_direction_candidate_range'][0])

    def test_repeated_select_does_not_advance_decay_but_new_paid_steps_do(self):
        p = self.planner()
        masks = self.masks(p)
        first = p.select(1, 12, masks, .5)
        repeated = p.select(1, 12, masks, .8)
        self.assertEqual(first['decision_index'], repeated['decision_index'])
        self.assertEqual(repeated['received_probability0_ignored'], .8)
        for i in range(1, 7):
            p.observe(observation(i+1, code=2))
            r = p.select(1, 12, masks, .5)
            self.assertEqual(r['decision_index'], i)
            self.assertAlmostEqual(r['geometric_weight'], .9 ** max(0, i-5))

    def test_infeasible_coverage_is_reported_before_score_execution(self):
        p = self.planner()
        with self.assertRaisesRegex(ValueError, 'coverage-qualified'):
            p.select(1, 1, self.masks(p), .5)

    def test_complete_receipt_replays_exactly_and_timing_is_separate(self):
        a, b = self.planner(), self.planner()
        ra = a.select(1, 12, self.masks(a), .5)
        rb = b.select(1, 12, self.masks(b), .5)
        self.assertEqual(ra, rb)
        self.assertNotIn('planning_seconds', ra)
        self.assertEqual(len(a.planning_timings), 1)
        self.assertGreaterEqual(a.planning_timings[0]['seconds'], 0.)

    def test_controller_receipts_do_not_restore_inherited_false_posterior_claim(self):
        p = self.planner(registered=False)
        public = PublicTemplatesV35(POSES, np.ones((2, 12, 4, 4)), np.ones((2, 12, 4)))
        runtime = ExternalControllerV39(p.models, public, (), mode='S', total_budget=12, planner=p)
        runtime.accept(observation(0))
        selection = runtime.select_target()
        self.assertFalse(selection['planning']['posterior_used_to_choose_action'])
        self.assertEqual(selection['planning']['phase'], 'external_cpu_mechanism_planning')
        self.assertEqual(runtime.summary()['online_global_plans'], 1)
        self.assertFalse(runtime.summary()['structural_posterior_used'])
        self.assertFalse(runtime.calls[-1]['planning']['posterior_used_to_choose_action'])


if __name__ == '__main__':
    unittest.main()
