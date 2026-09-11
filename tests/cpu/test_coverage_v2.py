"""Behavioral checks for coverage redesign: safety, accounting and information."""
import unittest
import numpy as np
from scipy.sparse.csgraph import dijkstra

from env.grid_exploration import GridConfig, GridExplorationEnv
from env.targeted_scenes import generate_scene
from nso.navigable_frontier_v2 import ProjectedCoveragePolicy
from nso.route_coverage_v2 import orientation_graph, recover_actions, RouteCoveragePolicy


class CoverageV2Tests(unittest.TestCase):
    def test_horizon_is_not_a_reachability_deadline(self):
        from dataclasses import replace
        ec = GridConfig(sensor_range_m=.6, max_steps=100)
        world = np.ones((40,40), np.uint8); world[3:-3,3:-3] = 0
        obs = GridExplorationEnv(world, ec).reset(start=(10,10), heading=1)
        belief = np.full(world.shape, -1, np.int8)
        belief[3:22,3:22] = 0
        obs = replace(obs, belief=belief)
        policy = RouteCoveragePolicy(ec, horizon=3)
        decision = policy.act(obs)
        self.assertTrue(decision.new_goal)
        self.assertGreater(policy._active['planned_actions'],3)
        self.assertEqual(policy.transfer_fallbacks,1)

    def test_orientation_cost_includes_turns_and_actions_reach_goal(self):
        safe = np.zeros((9, 9), bool)
        safe[4, 2:7] = True
        safe[2:5, 6] = True
        graph, cells, ids = orientation_graph(safe)
        start = int(ids[4, 2])*4 + 3  # west; must first turn around
        end = int(ids[2, 6])*4 + 0
        cost, previous = dijkstra(graph, directed=True, indices=start, return_predecessors=True)
        actions = recover_actions(previous, start, end, cells)
        self.assertEqual(cost[end], 9)  # six moves + two initial turns + one bend
        self.assertEqual(len(actions), 9)
        from utils.grid_geometry import DIRECTIONS
        pos, heading = (4, 2), 3
        for a in actions:
            if a == 'forward':
                dr, dc = DIRECTIONS[heading]
                pos = (pos[0]+dr, pos[1]+dc)
                self.assertTrue(safe[pos])
            else:
                heading = (heading+(1 if a=='right' else -1))%4
        self.assertEqual((pos, heading), ((2, 6), 0))

    def test_projected_frontiers_do_not_stop_at_corridor_wall(self):
        ec = GridConfig(resolution_m=.25, robot_radius_m=.2, sensor_range_m=3., max_steps=60)
        scene = generate_scene('office_spine', 192, 51, .25)
        env = GridExplorationEnv(scene.occupancy, ec)
        obs = env.reset(start=scene.entrances[0], heading=1)
        policy = ProjectedCoveragePolicy(ec, bundle=False, path_gain=False)
        done = False
        while not done:
            action = policy.act(obs).action
            obs, done = env.step(action)
            policy.observe_outcome(obs, done)
            self.assertFalse(obs.collision)
        # Regression: the original band-based prototype stopped at action 24.
        self.assertEqual(env.termination_reason, 'budget_exhausted')
        self.assertGreater(env.moves, 30)

    def test_hidden_world_and_semantics_do_not_change_initial_route(self):
        ec = GridConfig(sensor_range_m=.6, max_steps=50)
        world = np.ones((40,40), np.uint8); world[3:-3,3:-3] = 0
        other = world.copy(); other[25:30,25:30] = 1
        field1, field2 = np.zeros_like(world,float), np.ones_like(world,float)
        e1 = GridExplorationEnv(world, ec, field1)
        e2 = GridExplorationEnv(other, ec, field2)
        o1, o2 = e1.reset(start=(10,10),heading=1), e2.reset(start=(10,10),heading=1)
        np.testing.assert_array_equal(o1.belief,o2.belief)
        p1,p2 = RouteCoveragePolicy(ec),RouteCoveragePolicy(ec)
        self.assertEqual(p1.act(o1),p2.act(o2))
        self.assertEqual(p1.selection_audit,p2.selection_audit)

    def test_reset_discards_planned_actions_and_goal(self):
        ec = GridConfig(sensor_range_m=.6, max_steps=50)
        world = np.ones((40,40), np.uint8); world[3:-3,3:-3] = 0
        obs = GridExplorationEnv(world,ec).reset(start=(10,10),heading=1)
        policy = RouteCoveragePolicy(ec)
        first = policy.act(obs)
        policy.reset()
        self.assertEqual(policy.act(obs),first)
        self.assertEqual(policy.goal_count,1)

    def test_extra_observations_are_paid_actions(self):
        ec = GridConfig(sensor_range_m=.6, max_steps=12)
        world = np.ones((40,40), np.uint8); world[3:-3,3:-3] = 0
        env = GridExplorationEnv(world,ec)
        obs = env.reset(start=(10,10),heading=1)
        policy = ProjectedCoveragePolicy(ec, bundle=True,path_gain=False)
        before = int((obs.belief!=-1).sum())
        decision = policy.act(obs)
        self.assertEqual(env.steps,0)
        self.assertEqual(int((obs.belief!=-1).sum()),before)
        obs,_ = env.step(decision.action)
        self.assertEqual(env.steps,1)


if __name__ == '__main__':
    unittest.main()
