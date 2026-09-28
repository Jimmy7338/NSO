"""V35 boundary tests with analytic arrays and a non-fusing planner stub.

No World, renderer, TSDF, experiment packet replay or acquisition is used.
The real observation posterior, action lifecycle, public-graph return guard
and all four module interfaces execute on a declared eight-node toy graph.
"""
from copy import deepcopy
from types import SimpleNamespace
import unittest

import numpy as np

from nso.cpu_four_modules_v35 import CPUFourModuleControllerV35, ObservationV35
from nso.observation_belief_v35 import PublicTemplatesV35


POSES = tuple((x, 0, heading) for x in (0, 1) for heading in range(4))
NODE = {pose: index for index, pose in enumerate(POSES)}


def fixtures(*, informative=False):
    edges = []
    for x, y, heading in POSES:
        row = [('left', NODE[(x, y, (heading - 1) % 4)]),
               ('right', NODE[(x, y, (heading + 1) % 4)])]
        if (x, heading) == (0, 1):
            row.append(('forward', NODE[(1, 0, 1)]))
        elif (x, heading) == (1, 3):
            row.append(('forward', NODE[(0, 0, 3)]))
        edges.append(tuple(row))
    def terminal(mask):
        ratio = mask.bit_count() / len(POSES)
        return {'coverage': 1., 'surface': ratio, 'joint': ratio,
                'feasible': True, 'scope': 'analytic unit potential, not measured quality'}
    models = tuple(SimpleNamespace(poses=POSES, edges=tuple(edges), anchor=0,
        observed_masks=tuple(1 << n for n in range(len(POSES))), terminal=terminal)
        for _ in (0, 1))
    depth = np.full((2, len(POSES), 4, 4), 2., np.float32)
    ranges = np.full((2, len(POSES), 4), 5., np.float32)
    if informative:
        depth[1, NODE[(1, 0, 1)]] = 3.
        ranges[1, NODE[(1, 0, 1)]] = 4.
    return models, PublicTemplatesV35(POSES, depth, ranges)


class PlannerStub:
    def __init__(self, action='right'):
        self.action = action
        self.calls = []

    def select(self, node, remaining, masks, probability0, **kwargs):
        self.calls.append(deepcopy({'node': node, 'remaining': remaining,
            'masks': list(masks), 'probability0': probability0, **kwargs}))
        return {'action': self.action, 'unit_stub': True}


def observation(step=0, pose=POSES[0], action=None, *, frame_id=None,
                color=None, collision=False, depth_value=2., map_digest=None):
    rgb = np.zeros((4, 4, 3), np.uint8)
    if color is not None:
        rgb[:] = color
    return ObservationV35(frame_id or f'unit-{step}', step, pose, action,
        np.full((4, 4), depth_value, np.float32), rgb,
        np.full(4, 5., np.float32), collision=collision,
        measured_map_sha256=map_digest)


class CPUFourModulesV35Tests(unittest.TestCase):
    def runtime(self, mode='S', prefix=(), budget=8, action='right', informative=False):
        models, templates = fixtures(informative=informative)
        planner = PlannerStub(action)
        runtime = CPUFourModuleControllerV35(models, templates, prefix,
            mode=mode, total_budget=budget, planner=planner)
        return runtime, planner

    def advance(self, runtime, *, color=None):
        action = runtime.next_action()
        self.assertIsNotNone(action)
        node = runtime.pending['node']
        return runtime.accept(observation(runtime.state['step'] + 1,
            runtime.poses[node], action, color=color))

    def test_four_interfaces_common_prefix_then_online_and_exact_return(self):
        runtime, planner = self.runtime(prefix=('right',) * 4, budget=8)
        runtime.accept(observation(map_digest='a' * 64))
        for _ in range(4):
            self.advance(runtime)
        self.assertFalse(planner.calls)
        for _ in range(4):
            self.advance(runtime)
        self.assertEqual(len(planner.calls), 4)
        self.assertEqual(runtime.summary()['modules_called'], ['IGCR', 'OV-SDF', 'RPN-UQ', 'STGHP'])
        self.assertEqual(runtime.summary()['online_global_plans'], 4)
        self.assertEqual(runtime.summary()['actual_paid_actions'], 8)
        self.assertEqual(runtime.state['pose'], list(POSES[0]))
        self.assertEqual(runtime.terminal_reason, 'sensor_or_budget_end_returned')
        self.assertIsNone(runtime.next_action())
        self.assertFalse(runtime.summary()['full_method_efficacy_proven'])
        self.assertEqual(runtime.calls[0]['measured_map_sha256'], 'a' * 64)
        self.assertFalse(runtime.calls[0]['measured_quality_available'])

    def test_initial_observation_and_metadata_boundary_fail_before_mutation(self):
        for bad in (observation(1, action='right'),
                    observation(pose=POSES[1]), observation(action='right')):
            runtime, _ = self.runtime()
            with self.assertRaisesRegex(ValueError, 'action-zero'):
                runtime.accept(bad)
            self.assertEqual(runtime.belief.updates, 0)
            self.assertFalse(runtime.calls)
            self.assertEqual(runtime.masks, (0, 0))
        runtime, _ = self.runtime()
        with self.assertRaisesRegex(TypeError, 'sanitized'):
            runtime.accept(SimpleNamespace(step=0, hypothesis=1))
        with self.assertRaises(TypeError):
            ObservationV35('x', 0, POSES[0], None,
                np.ones((4, 4)), np.zeros((4, 4, 3), np.uint8), np.ones(4),
                actual_hypothesis=1)

    def test_duplicate_id_step_unsolicited_action_and_wrong_odometry_rejected(self):
        runtime, _ = self.runtime()
        runtime.accept(observation())
        with self.assertRaisesRegex(ValueError, 'issued paid action'):
            runtime.accept(observation(1, POSES[1], 'right'))
        self.assertEqual(runtime.next_action(), 'right')
        old_calls = deepcopy(runtime.calls)
        for bad in (observation(1, POSES[1], 'right', frame_id='unit-0'),
                    observation(0, POSES[1], 'right', frame_id='new'),
                    observation(1, POSES[3], 'left'),
                    observation(1, POSES[0], 'right')):
            with self.subTest(pose=bad.pose, step=bad.step, action=bad.action):
                with self.assertRaises(ValueError):
                    runtime.accept(bad)
                self.assertEqual(runtime.calls, old_calls)
                self.assertEqual(runtime.belief.updates, 1)
                self.assertEqual(runtime.state['step'], 0)
                self.assertEqual(runtime.pending['action'], 'right')
        runtime.accept(observation(1, POSES[1], 'right'))
        self.assertEqual(runtime.belief.updates, 2)

    def test_repeated_pose_has_no_additional_potential_or_geometry_evidence(self):
        runtime, _ = self.runtime(budget=8)
        runtime.accept(observation())
        for _ in range(4):
            state = self.advance(runtime)
        self.assertTrue(state['repeated_pose'])
        self.assertEqual(state['potential']['new_pose_potential_gain'], [0., 0.])
        self.assertFalse(runtime.posterior_receipts[-1]['new_geometry_pose'])
        self.assertEqual(runtime.posterior_receipts[-1]['geometry_applied_log_odds'], 0.)
        self.assertEqual(len(runtime.visited_nodes), 4)

    def test_pending_target_cannot_be_reissued_and_stale_selection_rejected(self):
        runtime, planner = self.runtime()
        runtime.accept(observation())
        first = runtime.select_target()
        self.assertEqual(runtime.select_target(), first)
        self.assertEqual(len(planner.calls), 1)
        self.assertEqual(runtime.next_action(), 'right')
        with self.assertRaisesRegex(ValueError, 'previous paid observation'):
            runtime.next_action()
        with self.assertRaisesRegex(ValueError, 'previous paid observation'):
            runtime.select_target()
        runtime.accept(observation(1, POSES[1], 'right'))
        with self.assertRaisesRegex(ValueError, 'current global selection'):
            runtime.assess_action(first)
        with self.assertRaisesRegex(ValueError, 'validated paid observation'):
            runtime.update_semantic(observation(2, POSES[2], 'right'))

    def test_local_guard_rejects_legal_edge_without_heading_restoring_return(self):
        runtime, planner = self.runtime(budget=1)
        runtime.accept(observation())
        self.assertIsNone(runtime.next_action())
        self.assertEqual(len(planner.calls), 1)
        self.assertFalse(runtime.calls[-1]['allowed'])
        self.assertEqual(runtime.calls[-1]['return_actions'], 1)
        self.assertEqual(runtime.summary()['actual_paid_actions'], 0)
        self.assertIsNone(runtime.pending)
        self.assertEqual(runtime.terminal_reason, 'returned_no_affordable_action')

    def test_collision_is_paid_terminal_and_does_not_credit_quality(self):
        runtime, _ = self.runtime()
        runtime.accept(observation())
        self.assertEqual(runtime.next_action(), 'right')
        runtime.accept(observation(1, action='right', collision=True))
        self.assertEqual(runtime.state['step'], 1)
        self.assertEqual(runtime.state['potential']['new_pose_potential_gain'], [0., 0.])
        self.assertEqual(runtime.terminal_reason, 'collision_observed')
        self.assertIsNone(runtime.next_action())
        feedback = [row for row in runtime.calls if row['module'] == 'IGCR'][-1]
        self.assertFalse(feedback['actual_reconstruction_quality_credit'])
        self.assertEqual(feedback['geometry_applied_log_odds'], 0.)
        with self.assertRaisesRegex(ValueError, 'already stopped'):
            runtime.accept(observation(2, POSES[1], 'right'))

    def test_geometry_mode_never_uses_marker_class_for_online_action_input(self):
        traces = []
        for color in ((40, 100, 220), (220, 60, 40)):
            runtime, planner = self.runtime('G', action='forward')
            runtime.accept(observation(color=color))
            planner.action = 'right'; self.advance(runtime, color=color)
            planner.action = 'forward'; self.advance(runtime, color=color)
            planner.action = 'left'; runtime.next_action()
            traces.append(deepcopy(planner.calls))
            self.assertEqual(runtime.belief.probabilities, (.5, .5))
        self.assertEqual(traces[0], traces[1])

    def test_semantic_prior_reaches_online_planner_after_two_distinct_xy(self):
        runtime, planner = self.runtime('S')
        color = (40, 100, 220)
        runtime.accept(observation(color=color))
        self.advance(runtime, color=color)
        self.assertEqual(runtime.belief.probabilities, (.5, .5))
        planner.action = 'forward'; self.advance(runtime, color=color)
        self.assertAlmostEqual(runtime.belief.probabilities[0], .9)
        planner.action = 'left'; runtime.next_action()
        self.assertAlmostEqual(planner.calls[-1]['probability0'], .9)
        self.assertIn(NODE[(1, 0, 1)], planner.calls[-1]['excluded_information_nodes'])
        self.assertTrue(planner.calls[-1]['geometry_feedback'])

    def test_feedback_ablation_is_passed_to_global_planner_and_igcr(self):
        runtime, planner = self.runtime('swapped_no_feedback', informative=True)
        color = (40, 100, 220)
        runtime.accept(observation(color=color))
        self.advance(runtime, color=color)
        planner.action = 'forward'; self.advance(runtime, color=color)
        self.assertAlmostEqual(runtime.belief.probabilities[0], .1)
        self.assertEqual(runtime.posterior_receipts[-1]['geometry_log_odds'], 0.)
        planner.action = 'left'; runtime.next_action()
        self.assertFalse(planner.calls[-1]['geometry_feedback'])
        self.assertFalse([row for row in runtime.calls if row['module'] == 'IGCR'][-1]['feedback_enabled'])

    def test_observation_arrays_are_immutable_owned_copies_and_finite(self):
        depth = np.ones((4, 4), np.float32)
        rgb = np.zeros((4, 4, 3), np.uint8)
        ranges = np.ones(4, np.float32)
        packet = ObservationV35('f0', 0, POSES[0], None, depth, rgb, ranges)
        depth[:] = 7.; rgb[:] = 7; ranges[:] = 7.
        self.assertTrue(np.all(packet.depth == 1.))
        self.assertTrue(np.all(packet.rgb == 0))
        self.assertTrue(np.all(packet.ranges == 1.))
        with self.assertRaises(ValueError):
            packet.depth[0, 0] = 3.
        for bad in (float('nan'), float('inf'), -1.):
            with self.assertRaisesRegex(ValueError, 'finite nonnegative'):
                observation(depth_value=bad)


if __name__ == '__main__':
    unittest.main()
