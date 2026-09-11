import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import numpy as np

from env.grid_exploration import GridConfig, GridExplorationEnv, GridObservation
from env.grid_layouts import generate_layout, LAYOUTS
from nso.frontier_policy import FrontierPolicy, shortest_paths, trace_path
from utils.grid_geometry import visible_mask, inflated_obstacles

ROOT = Path(__file__).resolve().parents[2]


class SensorTests(unittest.TestCase):
    def test_wall_occludes_and_range_fov_are_limited(self):
        world = np.zeros((25, 25), dtype=bool); world[:, 12] = True
        seen = visible_mask(world, (12, 8), 1, 8, 90)
        self.assertTrue(seen[12, 12])
        self.assertFalse(seen[:, 13:].any())
        self.assertFalse(seen[12, 7])
        self.assertFalse(seen[0, 8])
        open_seen = visible_mask(np.zeros_like(world), (12, 8), 1, 4, 90)
        self.assertTrue(open_seen[12, 12]); self.assertFalse(open_seen[12, 13])

    def test_diagonal_crack_is_opaque(self):
        world = np.zeros((9, 9), dtype=bool)
        world[3, 4] = world[4, 5] = True
        seen = visible_mask(world, (4, 4), 0, 4, 90)
        self.assertFalse(seen[3, 5])

    def test_boundary_does_not_wrap(self):
        world = np.zeros((10, 10), dtype=bool)
        seen = visible_mask(world, (0, 0), 0, 8, 90)
        self.assertFalse(seen[-1].any())
        self.assertFalse(seen[:, -1].any())


class GridEnvironmentTests(unittest.TestCase):
    def test_observation_contains_no_ground_truth_or_evaluation(self):
        world = generate_layout('multi_room')
        env = GridExplorationEnv(world)
        obs = env.reset(10)
        self.assertEqual(set(vars(obs)), {'belief', 'visible', 'position', 'heading', 'step', 'collision'})
        self.assertTrue((obs.belief == -1).any())
        with self.assertRaises(ValueError):
            obs.belief[0, 0] = 0
        obs.belief.setflags(write=True); obs.belief[:] = 0
        self.assertTrue((env.step('right')[0].belief == -1).any())

    def test_start_component_excludes_sealed_room(self):
        world = np.ones((20, 30), dtype=np.uint8)
        world[2:18, 2:12] = 0; world[2:18, 16:28] = 0
        env = GridExplorationEnv(world, GridConfig(robot_radius_m=0, sensor_range_m=5))
        env.reset(start=(10, 6), heading=0)
        assets = env.evaluation_assets()
        self.assertFalse(assets['reachable'][:, 16:].any())
        count = env.evaluation_metrics()['explorable_free_cells']
        self.assertEqual(count, 16 * 10)
        for _ in range(4):
            env.step('right')
        metrics = env.evaluation_metrics()
        self.assertAlmostEqual(metrics['coverage_ratio'], 1)
        self.assertAlmostEqual(metrics['explored_area_m2'], count * .01)

    def test_robot_radius_blocks_narrow_door(self):
        world = np.ones((20, 20), dtype=np.uint8); world[2:18, 2:18] = 0
        world[10, :] = 1; world[10, 10] = 0
        self.assertTrue(inflated_obstacles(world, 1)[10, 10])
        world[10, 8:13] = 0
        self.assertFalse(inflated_obstacles(world, 1)[10, 10])

    def test_collision_rotation_revisits_and_step_budget(self):
        world = np.ones((12, 12), dtype=np.uint8); world[2:10, 2:10] = 0
        env = GridExplorationEnv(world, GridConfig(robot_radius_m=0, max_steps=8))
        obs = env.reset(start=(2, 5), heading=0)
        obs, done = env.step('forward')
        self.assertTrue(obs.collision); self.assertEqual(obs.position, (2, 5))
        env.step('right'); env.step('forward')
        env.step('right'); env.step('right'); env.step('forward')
        env.step('left'); _, done = env.step('left')
        metrics = env.evaluation_metrics()
        self.assertTrue(done); self.assertEqual(metrics['steps'], 8)
        self.assertAlmostEqual(metrics['path_length_m'], .2)
        self.assertEqual(metrics['revisited_moves'], 1)
        self.assertEqual(metrics['collisions'], 1)
        with self.assertRaises(RuntimeError):
            env.step('left')

    def test_coverage_monotone_and_reset_has_no_leakage(self):
        env = GridExplorationEnv(generate_layout('loop'))
        first = env.reset(10)
        coverage = [env.evaluation_metrics()['coverage_ratio']]
        for action in ['right', 'forward', 'left', 'forward'] * 10:
            env.step(action); coverage.append(env.evaluation_metrics()['coverage_ratio'])
        self.assertTrue(np.all(np.diff(coverage) >= 0))
        reset = env.reset(10)
        np.testing.assert_array_equal(first.belief, reset.belief)
        self.assertEqual(env.evaluation_metrics()['steps'], 0)
        self.assertEqual(env.evaluation_metrics()['path_length_m'], 0)

    def test_all_layouts_valid_and_seeded(self):
        for layout in LAYOUTS:
            a = generate_layout(layout, 64, 11)
            np.testing.assert_array_equal(a, generate_layout(layout, 64, 11))
            env = GridExplorationEnv(a)
            self.assertEqual(env.reset(8).position, env.reset(8).position)
            self.assertGreater(env.evaluation_metrics()['explorable_free_cells'], 0)


class PolicyTests(unittest.TestCase):
    def test_shortest_observed_path_cannot_cross_wall(self):
        free = np.ones((10, 10), dtype=bool); free[:, 5] = False
        distance, parents = shortest_paths(free, (4, 2))
        self.assertEqual(distance[4, 8], -1)
        path = trace_path(parents, (4, 2), (8, 4))
        self.assertEqual(len(path), 6)
        self.assertTrue(all(free[cell] for cell in path))

    def test_closed_room_stops_and_turns_do_not_count_as_revisits(self):
        world = np.ones((12, 12), dtype=np.uint8); world[2:10, 2:10] = 0
        config = GridConfig(robot_radius_m=0, sensor_range_m=3, max_steps=60)
        for method in ('nearest_frontier', 'gain_per_cost'):
            env = GridExplorationEnv(world, config); obs = env.reset(start=(6, 6), heading=0)
            policy = FrontierPolicy(config, method)
            for _ in range(60):
                obs, done = env.step(policy.act(obs).action)
                if done:
                    break
            metrics = env.evaluation_metrics()
            self.assertEqual(metrics['termination_reason'], 'policy_stop')
            self.assertEqual(metrics['coverage_ratio'], 1)
            self.assertEqual(metrics['revisited_moves'], 0)
            self.assertLess(metrics['steps'], 60)

    def test_policies_reproduce_identical_actions_from_same_seed(self):
        world = generate_layout('dead_end')
        config = GridConfig(max_steps=70)
        for method in ('nearest_frontier', 'gain_per_cost'):
            sequences = []
            for repeat in range(2):
                env = GridExplorationEnv(world, config); obs = env.reset(32)
                policy = FrontierPolicy(config, method)
                actions = []
                for _ in range(config.max_steps):
                    decision = policy.act(obs)
                    if decision.goal:
                        self.assertEqual(obs.belief[decision.goal], 0)
                    actions.append(decision)
                    obs, done = env.step(decision.action)
                    if done:
                        break
                sequences.append((actions, env.evaluation_metrics()))
            self.assertEqual(sequences[0], sequences[1])

    def test_hidden_world_changes_do_not_change_policy_decision(self):
        world = np.ones((40, 40), dtype=np.uint8); world[2:38, 2:38] = 0
        altered = world.copy(); altered[25:35, 25:35] = 1
        config = GridConfig(sensor_range_m=.8)
        observations = [GridExplorationEnv(w, config).reset(start=(8, 8), heading=0) for w in (world, altered)]
        np.testing.assert_array_equal(observations[0].belief, observations[1].belief)
        for method in ('nearest_frontier', 'gain_per_cost'):
            decisions = [FrontierPolicy(config, method).act(obs) for obs in observations]
            self.assertEqual(*decisions)

    def test_core_imports_do_not_load_torch_or_habitat(self):
        code = ('import env.grid_exploration, nso.frontier_policy, sys; '
                'assert not any(n in sys.modules for n in ["torch", "habitat", "cv2", "clip"])')
        subprocess.run([sys.executable, '-c', code], cwd=ROOT, check=True)


if __name__ == '__main__':
    unittest.main()
