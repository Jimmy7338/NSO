"""Frozen development check that V10.1 repairs the zero-translation failure."""
import unittest

from tests.virtual3d.test_cpu_closed_loop_v10 import fixture, perform


class CPUClosedLoopV101Tests(unittest.TestCase):
    def test_real_sensor_episode_translates_replans_and_returns(self):
        world, components, runtime, _ = fixture(
            budget=24, max_steps=30, cpu_planner_revision="v10_1",
            cpu_coverage_slots=4)
        cells = {tuple(world.position)}
        actions = []
        planning_feedback_versions = []
        while not runtime.states[0]["closed"]:
            action = runtime.next_local_action(0)
            if action is None:
                break
            actions.append(action)
            before = runtime.sensor_episode_summary(0)["modules"]
            pending = runtime.states[0]["pending"]
            self.assertIsNotNone(pending["observed_gain_prediction"])
            self.assertLessEqual(1 + pending["reserved_return_cost"],
                                 before["feedback"]["remaining_budget"])
            _, result = perform(world, runtime, action)
            self.assertTrue(result["accepted"])
            cells.add(tuple(world.position))
            after = runtime.sensor_episode_summary(0)["modules"]
            planning_feedback_versions.append(after["feedback"]["planning_feedback_version"])
            self.assertEqual(after["observed_gain_calibration"]["observed_actions"], len(actions))
        summary = runtime.sensor_episode_summary(0)
        self.assertTrue(summary["closed"])
        self.assertFalse(summary["termination"]["failed"])
        self.assertTrue(summary["termination"]["returned_to_anchor"])
        self.assertEqual(world.collisions, 0)
        self.assertIn("forward", actions)
        self.assertGreaterEqual(len(cells), 2)
        self.assertGreaterEqual(summary["arrived_count"], 1)
        self.assertGreaterEqual(summary["modules"]["plans"], 2)
        self.assertEqual(planning_feedback_versions, sorted(planning_feedback_versions))

    def test_no_feedback_ablation_freezes_attempt_posterior_but_not_cost(self):
        world, components, runtime, _ = fixture(
            budget=8, max_steps=12, cpu_planner_revision="v10_1",
            cpu_coverage_slots=4, cpu_disable_feedback=True)
        for _ in range(3):
            action = runtime.next_local_action(0)
            self.assertIsNotNone(action)
            perform(world, runtime, action)
        modules = runtime.sensor_episode_summary(0)["modules"]
        self.assertEqual(modules["feedback"]["paid_actions"], 3)
        self.assertEqual(modules["feedback"]["actual_camera_pose_count"], 4)
        self.assertEqual(modules["feedback"]["planning_camera_pose_count"], 1)
        gain = modules["observed_gain_calibration"]
        self.assertEqual(gain["alpha"], {"radar": 1, "camera": 1})
        self.assertEqual(gain["beta"], {"radar": 1, "camera": 1})
        self.assertEqual(gain["observed_actions"], 3)


if __name__ == "__main__":
    unittest.main()
