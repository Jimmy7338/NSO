"""Mock ROS only: no device, sensor or action server is used."""
import math
from pathlib import Path
import sys
from types import SimpleNamespace as NS
import unittest
from unittest.mock import Mock, patch
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'runtime'))
from standard_navigation_goal import run_standard_navigation_goal, validate_goal_odom


def goal_message():
    return NS(target_pose=NS(header=NS(), pose=NS(position=NS(), orientation=NS())))


class StandardNavigationGoalTests(unittest.TestCase):
    def test_validation_has_no_distance_gate(self):
        self.assertEqual(validate_goal_odom([1234, -4321, 19]), [1234., -4321., 19.])
        for value in ([1, 2], [1, 2, 3, 4], [0, 0, math.nan], [0, math.inf, 0], None):
            with self.assertRaises(ValueError):
                validate_goal_odom(value)

    def run_goal(self, state=3, final_x=1.234):
        target = [1.234, -.567, .321]
        client = Mock()
        client.gh.comm_state_machine.action_goal.goal_id.id = 'actual-action-id'
        client.wait_for_result.side_effect = [False, True]
        client.get_state.return_value = state
        poses = [dict(pose=[0, 0, 0], target_frame='odom', source_age_s=.01,
                      source_stamp_s=1, receipt_monotonic=1),
                 dict(pose=[final_x, target[1], target[2]], target_frame='odom',
                      source_age_s=.01, source_stamp_s=2, receipt_monotonic=2)]
        can = dict(distance_m=0, absolute_yaw_rad=0, accounting_complete=True, feedback_fresh=True)
        ctx = dict(goal_odom=target, client=client, capture_odom_pose=Mock(side_effect=poses),
                   can_snapshot=Mock(side_effect=[can, dict(can, distance_m=1)]),
                   heartbeat=Mock(), result={}, journal=Mock())
        modules = {'rospy': NS(Time=NS(now=lambda:123), Duration=lambda x:x),
                   'move_base_msgs': NS(), 'move_base_msgs.msg': NS(MoveBaseGoal=goal_message)}
        with patch.dict(sys.modules, modules):
            evidence = run_standard_navigation_goal(ctx)
        return ctx, evidence

    def test_exact_goal_and_real_odom_evidence(self):
        ctx, evidence = self.run_goal()
        goal = ctx['client'].send_goal.call_args.args[0]
        self.assertEqual(goal.target_pose.header.frame_id, 'odom')
        self.assertEqual(goal.target_pose.pose.position.x, 1.234)
        self.assertEqual(goal.target_pose.pose.position.y, -.567)
        self.assertAlmostEqual(goal.target_pose.pose.orientation.z, math.sin(.321/2))
        self.assertTrue(evidence['verified'])
        self.assertEqual(evidence['before']['target_frame'], 'odom')
        self.assertFalse(evidence['credited_to_NSO'])
        self.assertEqual(ctx['result']['status'], 'standard_navigation_setup_goal_reached')
        self.assertEqual(ctx['heartbeat'].call_count, 3)

    def test_action_failure_cannot_report_success(self):
        ctx, evidence = self.run_goal(state=4)
        self.assertFalse(evidence['verified'])
        self.assertEqual(ctx['result']['status'], 'standard_navigation_setup_goal_failed')

    def test_pose_failure_cannot_report_success(self):
        ctx, evidence = self.run_goal(final_x=3)
        self.assertFalse(evidence['verified'])
        self.assertEqual(ctx['result']['status'], 'standard_navigation_setup_goal_failed')


if __name__ == '__main__':
    unittest.main()
