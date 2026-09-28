"""Analytical primitive routing; constructed observations are not sensor runs."""
import math
import unittest
import numpy as np

from nso.instance_belief_v40 import PaidRGBDObservationV40
from nso.primitive_navigation_v41 import (PrimitiveStateV41, PublicPrimitiveGraphV41,
                                          ReturnAwarePrimitiveRouterV41)


def graph_spec(nodes=None, edges=None):
    return {'schema_version': 'v41.public_navigation.v1', 'source_kind': 'provided_navigation_prior',
            'nodes': nodes or {'home': [0., 0.], 'east': [1., 0.]},
            'edges': edges if edges is not None else [['home', 'east']]}


def observation(graph, state, step):
    yaw = state.heading * math.pi/6
    transform = np.eye(4)
    transform[:3, :3] = [[math.sin(yaw), 0, math.cos(yaw)],
                         [-math.cos(yaw), 0, math.sin(yaw)], [0, -1, 0]]
    transform[:3, 3] = [*graph.positions[state.node], .9]
    return PaidRGBDObservationV40(f'analytic_{step}', step, np.zeros((2, 2, 3), np.uint8),
                                  np.ones((2, 2)), np.array([[2., 0, .5], [0, 2., .5], [0, 0, 1.]]), transform)


class PrimitiveNavigationTests(unittest.TestCase):
    def setUp(self):
        self.graph = PublicPrimitiveGraphV41(graph_spec())
        self.home = PrimitiveStateV41('home', 0)
        self.east = PrimitiveStateV41('east', 0)

    def test_one_meter_coarse_edge_expands_to_four_paid_moves(self):
        route = self.graph.route(self.home, self.east)
        self.assertEqual(route.actions, ('forward',)*4)
        for left, right in zip(route.states, route.states[1:]):
            self.assertAlmostEqual(np.linalg.norm(np.subtract(self.graph.positions[right.node], self.graph.positions[left.node])), .25)

    def test_return_includes_all_turns_and_terminal_heading(self):
        outbound = self.graph.route(self.home, self.east)
        inbound = self.graph.route(self.east, self.home)
        self.assertEqual(outbound.cost, 4)
        self.assertEqual(inbound.cost, 16)
        self.assertEqual(inbound.actions.count('forward'), 4)
        self.assertEqual(inbound.states[-1], self.home)

    def test_insufficient_budget_rejects_goal_before_motion(self):
        router = ReturnAwarePrimitiveRouterV41(self.graph, home=self.home, budget=20)
        router.accept(observation(self.graph, self.home, 0))
        decision = router.choose({self.east: 1.})
        self.assertEqual(decision['action'], 'stop')
        self.assertEqual(decision['candidates'][0]['total_with_observation_and_return'], 21)

    def test_action_zero_selection_has_no_forced_prefix(self):
        router = ReturnAwarePrimitiveRouterV41(self.graph, home=self.home, budget=21)
        with self.assertRaises(ValueError):
            router.choose({self.east: 1.})
        router.accept(observation(self.graph, self.home, 0))
        decision = router.choose({self.east: 1.})
        self.assertEqual(decision['action'], 'forward')
        self.assertEqual(decision['fixed_prefix_actions'], 0)

    def test_complete_constructed_route_consumes_exact_budget(self):
        router = ReturnAwarePrimitiveRouterV41(self.graph, home=self.home, budget=21)
        router.accept(observation(self.graph, self.home, 0))
        observed = False
        actions = []
        for step in range(1, 23):
            decision = router.choose({} if observed else {self.east: 1.})
            if decision['action'] == 'stop':
                break
            self.assertNotEqual(decision['action'], 'blocked')
            action = decision['action']
            actions.append(action)
            observed |= action == 'observe'
            expected = router.pending['expected_state']
            router.accept(observation(self.graph, expected, step))
        self.assertTrue(observed)
        self.assertEqual(len(actions), 21)
        self.assertEqual(router.state, self.home)
        self.assertEqual(router.step, 21)

    def test_collision_attempt_consumes_budget_and_blocks_edge(self):
        router = ReturnAwarePrimitiveRouterV41(self.graph, home=self.home, budget=21)
        router.accept(observation(self.graph, self.home, 0))
        router.choose({self.east: 1.})
        router.accept(observation(self.graph, self.home, 1), execution_outcome='collision')
        self.assertEqual(router.step, 1)
        self.assertIsNone(self.graph.route(self.home, self.east))
        self.assertEqual(router.choose({self.east: 1.})['action'], 'stop')

    def test_pose_cannot_teleport_to_coarse_endpoint(self):
        router = ReturnAwarePrimitiveRouterV41(self.graph, home=self.home, budget=21)
        router.accept(observation(self.graph, self.home, 0))
        router.choose({self.east: 1.})
        with self.assertRaisesRegex(ValueError, 'disagrees'):
            router.accept(observation(self.graph, self.east, 1))
        self.assertEqual(router.step, 0)

    def test_double_decision_or_unpaid_observation_rejected(self):
        router = ReturnAwarePrimitiveRouterV41(self.graph, home=self.home, budget=21)
        initial = observation(self.graph, self.home, 0)
        router.accept(initial)
        with self.assertRaises(ValueError):
            router.accept(observation(self.graph, self.home, 1))
        router.choose({self.east: 1.})
        with self.assertRaises(ValueError):
            router.choose({self.east: 1.})
        with self.assertRaises(ValueError):
            router.accept(initial)

    def test_diagonal_thirty_degree_primitive_is_supported(self):
        end = [.25*math.cos(math.pi/6), .25*math.sin(math.pi/6)]
        graph = PublicPrimitiveGraphV41(graph_spec({'home': [0., 0.], 'end': end}, [['home', 'end']]))
        route = graph.route(PrimitiveStateV41('home', 0), PrimitiveStateV41('end', 1))
        self.assertEqual(route.actions, ('left', 'forward'))

    def test_illegal_geometry_and_private_fields_rejected(self):
        for spec in (dict(graph_spec(), owner_ids=[0]),
                     graph_spec({'home': [0., 0.], 'east': [.3, 0.]}, [['home', 'east']]),
                     graph_spec({'home': [0., 0.], 'east': [.25/math.sqrt(2)]*2}, [['home', 'east']])):
            with self.subTest(spec=spec), self.assertRaises(ValueError):
                PublicPrimitiveGraphV41(spec)

    def test_no_free_observation_when_budget_is_exhausted(self):
        router = ReturnAwarePrimitiveRouterV41(self.graph, home=self.home, budget=1)
        router.accept(observation(self.graph, self.home, 0))
        self.assertEqual(router.choose({self.home: 1.})['action'], 'observe')
        router.accept(observation(self.graph, self.home, 1))
        self.assertEqual(router.choose({self.home: 1.})['action'], 'stop')

    def test_camera_height_cannot_teleport_between_paid_steps(self):
        router = ReturnAwarePrimitiveRouterV41(self.graph, home=self.home, budget=21)
        router.accept(observation(self.graph, self.home, 0))
        router.choose({self.east: 1.})
        paid = observation(self.graph, router.pending['expected_state'], 1)
        transform = paid.world_from_camera.copy()
        transform[2, 3] = 100.
        changed = PaidRGBDObservationV40(paid.frame_id, 1, paid.rgb, paid.depth_m, paid.intrinsic, transform)
        with self.assertRaisesRegex(ValueError, 'camera height'):
            router.accept(changed)
        self.assertEqual(router.step, 0)

    def test_undeclared_optical_roll_is_rejected(self):
        router = ReturnAwarePrimitiveRouterV41(self.graph, home=self.home, budget=21)
        paid = observation(self.graph, self.home, 0)
        transform = paid.world_from_camera.copy()
        roll = np.array([[0., -1, 0], [1, 0, 0], [0, 0, 1.]])
        transform[:3, :3] = transform[:3, :3] @ roll
        changed = PaidRGBDObservationV40(paid.frame_id, 0, paid.rgb, paid.depth_m, paid.intrinsic, transform)
        with self.assertRaisesRegex(ValueError, 'roll or pitch'):
            router.accept(changed)
        self.assertEqual(router.step, -1)


if __name__ == '__main__':
    unittest.main()
