"""Safety/accounting contracts for the CPU development guard, not efficacy tests."""
from collections import deque
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

from env.grid_exploration import GridConfig
from env.virtual3d_v2 import VirtualConfigV2
from nso.feedback_joint_planner_v4 import FeedbackJointPlannerV4, evidence_snapshot, evidence_increment
from nso.semantic_completion_v3 import ObjectJointPlannerV3


class FeedbackTests(unittest.TestCase):
    def planner(self, **options):
        p = FeedbackJointPlannerV4(GridConfig(max_steps=100), VirtualConfigV2(max_steps=100), **options)
        p.set_mapping(SimpleNamespace(quality={}, evidence=lambda: (np.empty((0, 3)), np.empty(0, int), np.empty(0, int))))
        return p

    def obs(self, step=0):
        belief = np.zeros((30, 30), dtype=np.int8)
        belief[:, 25:] = -1
        return SimpleNamespace(position=(15, 15), heading=0, belief=belief, step=step, collision=False)

    @staticmethod
    def proposal(planner, obs, safe=None, actions=('right', 'right'), heading=2):
        planner.goal, planner.goal_heading = obs.position, heading
        planner._actions = deque(actions)
        planner._gain = planner._distance = 0
        planner._score = 1.
        planner.goal_count += 1
        planner.selection_audit.append(dict(step=obs.step, selected=True))
        planner._active = dict(start_step=obs.step, goal=list(obs.position), planned_actions=len(actions), predicted_gain=0)
        return True

    def test_single_frame_and_repeated_identical_evidence_do_not_mint_returns(self):
        row = dict(point=np.array([1., 1., .8]), n=1, bits=1, information=1., label=3)
        mapper = SimpleNamespace(quality={(1, 1, 1): row})
        self.assertEqual(evidence_snapshot(mapper), {})
        row['n'] = 2
        first = evidence_snapshot(mapper)
        self.assertGreater(evidence_increment({}, first), 0)
        row['n'] = 200
        row['label'] = 2
        self.assertEqual(evidence_increment(first, evidence_snapshot(mapper)), 0)

    def test_fixed_keys_separate_new_evidence_from_old_information_improvement(self):
        row = dict(point=np.array([1., 1., .8]), n=2, bits=1, information=1.)
        mapper = SimpleNamespace(quality={(1, 1, 1): row})
        before = evidence_snapshot(mapper)
        row['bits'] = 3
        after = evidence_snapshot(mapper)
        self.assertAlmostEqual(evidence_increment(before, after), .15**2*.3)
        self.assertEqual(evidence_increment(after, before), 0.)

    def test_reached_view_is_rejected_until_local_occupancy_changes(self):
        p = self.planner(coverage_guard=False)
        obs = self.obs()
        with patch.object(ObjectJointPlannerV3, '_select', self.proposal), patch.object(p.anchor, '_select', return_value=False):
            self.assertTrue(p._select(obs, p._safe(obs)))
            obs.step = 2
            p._actions.clear()
            p.observe_outcome(obs)
            self.assertFalse(p._select(obs, p._safe(obs)))
            self.assertEqual(p.feedback_rejections, 1)
            obs.belief[15, 25] = 0
            self.assertTrue(p._select(obs, p._safe(obs)))

    def test_inspection_sequence_is_rejected_before_budget_overrun(self):
        p = self.planner(inspection_fraction=.01, feedback_guard=False)
        obs = self.obs()
        with patch.object(ObjectJointPlannerV3, '_select', self.proposal), patch.object(p.anchor, '_select', return_value=False), patch.object(p, '_route_gain', return_value=0):
            self.assertFalse(p._select(obs, p._safe(obs)))
            self.assertEqual(len(p._actions), 0)
            self.assertIsNone(p._active)

    def test_coverage_anchor_is_kept_when_inspection_budget_is_zero(self):
        p = self.planner(inspection_fraction=0, feedback_guard=False)
        obs = self.obs()
        with patch.object(ObjectJointPlannerV3, '_select', self.proposal), patch.object(p.anchor, '_select', side_effect=lambda o, s: self.proposal(p.anchor, o, actions=('forward',), heading=0)), patch.object(p, '_route_gain', side_effect=[10, 0]):
            self.assertTrue(p._select(obs, p._safe(obs)))
        self.assertEqual(list(p._actions), ['forward'])
        self.assertEqual(p._active['controller'], 'coverage_anchor')
        self.assertFalse(p._active['inspection'])
        self.assertEqual(sum(bool(r.get('selected')) for r in p.selection_audit), 1)

    def test_no_feedback_ablation_allows_same_view_when_budget_allows(self):
        p = self.planner(coverage_guard=False, feedback_guard=False)
        obs = self.obs()
        with patch.object(ObjectJointPlannerV3, '_select', self.proposal), patch.object(p.anchor, '_select', return_value=False):
            self.assertTrue(p._select(obs, p._safe(obs)))
            obs.step = 2
            p._actions.clear()
            p.observe_outcome(obs)
            self.assertTrue(p._select(obs, p._safe(obs)))

    def test_reset_removes_previous_scene_budget_and_view_memory(self):
        p = self.planner()
        p.inspection_actions = 10
        p._view_history[(1, 2, 3)] = 'old'
        p.low_return_streak = 3
        p.reset()
        self.assertEqual((p.inspection_actions, p.low_return_streak), (0, 0))
        self.assertEqual(p._view_history, {})

    def test_inspection_may_not_cross_into_final_coverage_reserve(self):
        p = self.planner(inspection_fraction=.5, reserve_fraction=.2, feedback_guard=False)
        obs = self.obs(step=79)
        def ten_actions(planner, o, safe):
            return self.proposal(planner, o, actions=('right',)*10)
        with patch.object(ObjectJointPlannerV3, '_select', ten_actions), patch.object(p.anchor, '_select', return_value=False), patch.object(p, '_route_gain', return_value=0):
            self.assertFalse(p._select(obs, p._safe(obs)))

    def test_noisy_centroid_crossing_height_cutoff_does_not_reset_identity(self):
        row = dict(point=np.array([1., 1., .13]), n=2, bits=1, information=1.)
        mapper = SimpleNamespace(quality={(1, 1, 1): row})
        first = evidence_snapshot(mapper)
        row['point'][2] = .11
        self.assertEqual(evidence_snapshot(mapper), first)

    def test_pruned_evidence_cannot_be_paid_again_on_reappearance(self):
        p = self.planner(coverage_guard=False, feedback_guard=False)
        obs = self.obs()
        row = dict(point=np.array([1., 1., .8]), n=2, bits=1, information=1.)
        with patch.object(ObjectJointPlannerV3, '_select', self.proposal), patch.object(p.anchor, '_select', return_value=False):
            p._select(obs, p._safe(obs))
            p.mapper.quality[(1, 1, 1)] = row
            obs.step = 2; p._actions.clear(); p.observe_outcome(obs)
            self.assertGreater(p.attempts[-1]['measured_evidence_gain'], 0)
            p.mapper.quality.clear()
            p._select(obs, p._safe(obs))
            p.mapper.quality[(1, 1, 1)] = row
            obs.step = 4; p._actions.clear(); p.observe_outcome(obs)
            self.assertEqual(p.attempts[-1]['measured_evidence_gain'], 0)

    def test_automatic_scans_cannot_bypass_zero_inspection_budget(self):
        p = self.planner(inspection_fraction=0)
        with patch.object(p, '_select', return_value=False):
            self.assertEqual(p.act(self.obs()).action, 'stop')
        self.assertEqual(p.inspection_actions, 0)

    def test_remove_hypotheses_does_not_construct_hidden_geometry(self):
        p = self.planner(use_object_hypotheses=False)
        safe = np.ones((30, 30), bool)
        with patch('nso.semantic_completion_v3.ObjectCompletionModel', side_effect=AssertionError('hypothesis constructed')):
            candidates = p._inspection_cells(safe, np.ones((30, 30)))
        self.assertIsNone(p.object_model)
        self.assertGreater(len(candidates), 0)
        self.assertLessEqual(len(candidates), 20)


if __name__ == '__main__':
    unittest.main()
