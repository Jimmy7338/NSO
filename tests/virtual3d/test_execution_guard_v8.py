import unittest
import numpy as np
from nso.execution_guard_v8 import ObservedExecutionGuard
from utils.grid_geometry import DIRECTIONS


class ExecutionGuardTests(unittest.TestCase):
    def setUp(self):
        self.guard = ObservedExecutionGuard(.2, .2)
        self.belief = np.zeros((15, 15), np.int8)

    def test_new_observation_revokes_previously_safe_forward(self):
        self.assertTrue(self.guard.assess(self.belief, (7, 7), 1, 'forward', 10).allowed)
        updated = self.belief.copy(); updated[7, 9] = 1
        result = self.guard.assess(updated, (7, 7), 1, 'forward', 10)
        self.assertFalse(result.allowed)
        self.assertEqual(result.reason, 'next_footprint_not_known_safe')
        self.assertEqual(result.target, (7, 8))

    def test_unknown_footprint_and_unsafe_current_cannot_be_cleared(self):
        self.belief[7, 8] = -1
        snapshot = self.belief.copy()
        self.assertFalse(self.guard.assess(self.belief, (7, 7), 3, 'forward', 20).allowed)
        self.assertFalse(self.guard.return_plan(self.belief, (7, 7), 3, (7, 3, 1), 20).available)
        np.testing.assert_array_equal(snapshot, self.belief)

    def test_return_restores_position_and_heading_and_pays_rotations(self):
        anchor = (7, 4, 1)
        plan = self.guard.return_plan(self.belief, (7, 9), 1, anchor, 20)
        self.assertTrue(plan.available)
        self.assertEqual(plan.paid_cost, 9)  # five translations plus four turns
        r, c, heading = 7, 9, 1
        for index, action in enumerate(plan.actions):
            self.assertTrue(self.guard.assess(self.belief, (r, c), heading, action, 20-index).allowed)
            if action == 'forward':
                dr, dc = DIRECTIONS[heading]; r += int(dr); c += int(dc)
            else:
                heading = (heading + (1 if action == 'right' else -1)) % 4
        self.assertEqual((r, c, heading), anchor)
        self.assertFalse(self.guard.return_plan(self.belief, (7, 9), 1, anchor, 8).available)

    def test_new_obstacle_can_make_return_unavailable(self):
        self.belief[:, 7] = 1
        self.assertFalse(self.guard.return_plan(self.belief, (7, 10), 1, (7, 4, 1), 48).available)
        self.assertFalse(self.guard.assess(self.belief, (7, 10), 1, 'right', 0).allowed)


if __name__ == '__main__':
    unittest.main()
