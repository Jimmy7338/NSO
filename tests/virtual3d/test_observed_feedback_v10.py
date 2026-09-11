"""IGCR state transitions and paid observation accounting, not efficacy tests."""
from copy import deepcopy
from types import SimpleNamespace
import unittest

import numpy as np

from nso.observed_feedback_v10 import FeedbackLedger
from utils.rgbd_contract import RGBDFrame


def frame(t, x=1.):
    pose = np.eye(4)
    pose[:3, :3] = [[0., 0., 1.], [-1., 0., 0.], [0., -1., 0.]]
    pose[:3, 3] = [x, 1., .8]
    return RGBDFrame(float(t), np.ones((18, 24), np.float32),
                     np.full((18, 24, 3), 128, np.uint8),
                     np.array([[30., 0., 11.5], [0., 30., 8.5], [0., 0., 1.]]),
                     pose, np.zeros((18, 24), np.int32)).validate()


def row(x=.1, information=1., residual=0.):
    return dict(point=np.array([x, .1, .8]), information=information,
                residual=residual, n=1, bits=1, label=2)


def mapper():
    result = SimpleNamespace(config=SimpleNamespace(resolution_m=.2),
                             belief=np.full((2, 3), -1, np.int8),
                             camera_seen=np.zeros((2, 3), bool), quality={})
    result.belief[0, 0] = 0
    result.camera_seen[0, 0] = True
    result.quality[(0, 0, 5)] = row()
    result.grid_cell = lambda point: (int(point[1] // .2), int(point[0] // .2))
    return result


class FeedbackLedgerTests(unittest.TestCase):
    def bootstrap(self, mapping=None, budget=3, enabled=True):
        mapping = mapper() if mapping is None else mapping
        ledger = FeedbackLedger("scene", "episode", budget, feedback_enabled=enabled)
        ledger.bootstrap(mapping, frames=[frame(0), frame(1)], frame_ids=[149, 150],
                         map_version=151, last_action_id=150)
        return mapping, ledger

    def test_actual_paid_terminal_frame_and_duplicate_delivery(self):
        mapping, ledger = self.bootstrap()
        mapping.belief[0, 1] = 0
        mapping.quality[(1, 0, 5)] = row(.3)
        event = ledger.observe(mapping, frame=frame(2, 1.2), frame_id=151,
                               action_id=151, map_version=152, done=True,
                               failed=True, reason="budget_without_return")
        self.assertAlmostEqual(event["parts"]["new_observed_2d_area_m2"], .04)
        self.assertAlmostEqual(event["parts"]["measured_support_proxy_m2"], .0225)
        self.assertTrue(event["failed"])
        before = ledger.snapshot()
        duplicate = ledger.observe(mapping, frame=frame(2, 1.2), frame_id=151,
                                   action_id=151, map_version=152, done=True,
                                   failed=True, reason="budget_without_return")
        self.assertFalse(duplicate["accepted"])
        self.assertTrue(duplicate["duplicate"])
        self.assertEqual(before, ledger.snapshot())
        self.assertEqual(before["paid_actions"], 1)
        self.assertEqual(before["consumed_action_ids"], [151])
        self.assertEqual(before["failure_events"], 1)
        with self.assertRaises(RuntimeError):
            ledger.observe(mapping, frame=frame(3), frame_id=152, action_id=152, map_version=153)
        with self.assertRaises(ValueError):
            ledger.observe(mapping, frame=frame(2, 1.3), frame_id=151,
                           action_id=151, map_version=152, done=True, failed=True)
        self.assertEqual(before, ledger.snapshot())

    def test_unknown_or_reappearing_cells_and_support_cannot_mint_fresh_reward(self):
        mapping, ledger = self.bootstrap()
        mapping.camera_seen[0, 1] = True
        mapping.quality[(1, 0, 5)] = row(.3)
        unknown = ledger.observe(mapping, frame=frame(2), frame_id=151,
                                 action_id=151, map_version=152)
        self.assertEqual(unknown["reward"], 0.)
        self.assertEqual(unknown["unknown_support_excluded"], 1)
        self.assertTrue(unknown["no_progress"])
        mapping.belief[0, 1] = 0
        observed = ledger.observe(mapping, frame=frame(3), frame_id=152,
                                  action_id=152, map_version=153)
        self.assertAlmostEqual(observed["parts"]["new_observed_2d_area_m2"], .04)
        self.assertAlmostEqual(observed["parts"]["measured_support_proxy_m2"], .0225)
        # A quality counter or class change alone is not fresh geometry.
        mapping.quality[(1, 0, 5)]["n"] = 1000
        mapping.quality[(1, 0, 5)]["label"] = 3
        same = ledger.observe(mapping, frame=frame(4), frame_id=153,
                              action_id=153, map_version=154)
        self.assertEqual(same["reward"], 0.)
        self.assertTrue(same["no_progress"])
        self.assertEqual(ledger.remaining_budget, 0)
        before = ledger.snapshot()
        with self.assertRaises(ValueError):
            ledger.observe(mapping, frame=frame(5), frame_id=154, action_id=154, map_version=155)
        self.assertEqual(before, ledger.snapshot())

    def test_fixed_old_quality_keeps_negative_change_and_a_fixed_denominator(self):
        mapping, ledger = self.bootstrap()
        mapping.quality[(0, 0, 5)]["residual"] = .01
        mapping.quality[(1, 0, 5)] = row(.3, information=1000.)
        mapping.belief[0, 1] = 0
        loss = ledger.observe(mapping, frame=frame(2), frame_id=151,
                              action_id=151, map_version=152)
        expected = 1. / 2.25 - 1. / 1.25
        self.assertAlmostEqual(loss["parts"]["fixed_old_quality_proxy_change"], expected)
        self.assertEqual(loss["fixed_old_support_count"], 1)
        del mapping.quality[(0, 0, 5)]
        missing = ledger.observe(mapping, frame=frame(3), frame_id=152,
                                 action_id=152, map_version=153)
        self.assertAlmostEqual(missing["parts"]["fixed_old_quality_proxy_change"], -1. / 2.25)
        self.assertEqual(missing["old_quality_proxy"], 0.)
        mapping.quality[(0, 0, 5)] = row()
        recovered = ledger.observe(mapping, frame=frame(4), frame_id=153,
                                   action_id=153, map_version=154)
        self.assertEqual(recovered["parts"]["measured_support_proxy_m2"], 0.)
        self.assertAlmostEqual(ledger.snapshot()["old_quality_proxy_total_change"], 0.)

    def test_no_feedback_freezes_only_planning_poses_not_safety_accounting(self):
        mapping, normal = self.bootstrap()
        _, ablated = self.bootstrap(mapping, enabled=False)
        mapping.belief[0, 1] = 0
        for ledger in (normal, ablated):
            ledger.observe(mapping, frame=frame(2, 1.2), frame_id=151,
                           action_id=151, map_version=152, collision=True)
        a, b = normal.snapshot(), ablated.snapshot()
        for name in ("paid_actions", "remaining_budget", "actual_camera_pose_count",
                     "actual_camera_poses_sha256", "old_quality_proxy", "failure_events"):
            self.assertEqual(a[name], b[name])
        self.assertEqual(a["actual_camera_pose_count"], 3)
        self.assertEqual(a["planning_camera_pose_count"], 3)
        self.assertEqual(b["planning_camera_pose_count"], 2)
        self.assertEqual(a["feedback_version"], b["feedback_version"])
        self.assertNotEqual(a["planning_feedback_version"], b["planning_feedback_version"])
        # The caller cannot turn a planned pose into ledger history by mutation.
        ablated.planning_camera_poses()[0][:] = 0
        np.testing.assert_array_equal(ablated.planning_camera_poses()[0], frame(0).world_from_camera)

    def test_misaligned_packets_are_rejected_atomically_and_episodes_isolated(self):
        mapping, ledger = self.bootstrap()
        before = ledger.snapshot()
        for change in (dict(episode_id="other"), dict(scene_id="other"),
                       dict(action_id=152), dict(map_version=151),
                       dict(action_cost=0), dict(frame=frame(1))):
            arguments = dict(frame=frame(2), frame_id=151, action_id=151, map_version=152)
            arguments.update(change)
            with self.assertRaises(ValueError):
                ledger.observe(mapping, **arguments)
            self.assertEqual(before, ledger.snapshot())
        other = FeedbackLedger("scene", "next", 3)
        other.bootstrap(mapping, frames=[frame(0)], frame_ids=[0], map_version=1)
        other.observe(mapping, frame=frame(1), frame_id=1, action_id=1, map_version=2)
        self.assertEqual(before, ledger.snapshot())
        self.assertEqual(other.snapshot()["consumed_action_ids"], [1])

    def test_mapper_geometry_validation_and_repeated_initial_packet(self):
        mapping, ledger = self.bootstrap()
        before = ledger.snapshot()
        duplicate = ledger.observe(mapping, frame=frame(1), frame_id=150,
                                   action_id=150, map_version=151)
        self.assertTrue(duplicate["duplicate"])
        self.assertEqual(before, ledger.snapshot())
        with self.assertRaises(ValueError):
            ledger.observe(mapping, frame=frame(1), frame_id=150,
                           action_id=150, map_version=151, done=True)
        mapping.config.resolution_m = .1
        with self.assertRaises(ValueError):
            ledger.observe(mapping, frame=frame(2), frame_id=151,
                           action_id=151, map_version=152)
        self.assertEqual(before, ledger.snapshot())

    def test_real_rgbd_mapper_history_is_read_only_and_not_decimated(self):
        from env.virtual3d_v2 import VirtualConfigV2
        from nso.semantic_completion_v3 import SemanticHistoryMapperV3
        mapping = SemanticHistoryMapperV3((20, 30), VirtualConfigV2())
        prefix = [frame(0), frame(1, 1.01), frame(2, 1.02)]
        for observed in prefix:
            mapping.update(observed)
        self.assertEqual(len(mapping.keyframes), 1)
        ledger = FeedbackLedger("actual_mapper", "first", 2)
        ledger.bootstrap(mapping, frames=prefix, frame_ids=[0, 1, 2],
                         map_version=mapping.frames, last_action_id=2)
        observed = frame(3, 1.03)
        mapping.update(observed)
        quality_before = deepcopy(mapping.quality)
        belief_before, camera_before, count_before = mapping.belief.copy(), mapping.camera_seen.copy(), mapping.frames
        ledger.observe(mapping, frame=observed, frame_id=3, action_id=3, map_version=mapping.frames)
        self.assertEqual(len(ledger.planning_camera_poses()), 4)
        self.assertEqual(len(mapping.keyframes), 1)
        self.assertEqual(mapping.frames, count_before)
        np.testing.assert_array_equal(mapping.belief, belief_before)
        np.testing.assert_array_equal(mapping.camera_seen, camera_before)
        self.assertEqual(mapping.quality.keys(), quality_before.keys())
        for key, state in quality_before.items():
            for name, value in state.items():
                np.testing.assert_array_equal(mapping.quality[key][name], value)

    def test_logical_finish_uses_last_frame_without_free_observation_or_reward(self):
        mapping, ledger = self.bootstrap()
        last = ledger.observe(mapping, frame=frame(2, 1.2), frame_id=151,
                              action_id=151, map_version=152)
        before = ledger.snapshot()
        ended = ledger.finish("no_positive_option_at_anchor")
        after = ledger.snapshot()
        self.assertTrue(after["done"])
        self.assertEqual(ended["event_type"], "logical_termination")
        self.assertEqual(ended["reference_feedback_version"], last["feedback_version"])
        self.assertEqual((ended["frame_id"], ended["action_id"], ended["map_version"]), (151, 151, 152))
        self.assertEqual((ended["action_cost"], ended["reward"]), (0, 0.))
        for name in ("paid_actions", "remaining_budget", "consumed_frame_ids", "consumed_action_ids",
                     "actual_camera_poses_sha256", "planning_camera_poses_sha256", "map_version",
                     "planning_feedback_version", "old_quality_proxy"):
            self.assertEqual(before[name], after[name])
        duplicate = ledger.finish("no_positive_option_at_anchor")
        self.assertFalse(duplicate["accepted"])
        self.assertTrue(duplicate["duplicate"])
        self.assertEqual(after, ledger.snapshot())
        with self.assertRaises(ValueError):
            ledger.finish("current_footprint_unsafe", failed=True)
        with self.assertRaises(RuntimeError):
            ledger.observe(mapping, frame=frame(3), frame_id=152, action_id=152, map_version=153)
        self.assertEqual(after, ledger.snapshot())
        # An exact late delivery of the original last nonterminal packet does
        # not rewrite its event to done=True and cannot reopen the episode.
        replay = ledger.observe(mapping, frame=frame(2, 1.2), frame_id=151,
                                action_id=151, map_version=152)
        self.assertFalse(replay["done"])
        self.assertTrue(replay["duplicate"])
        self.assertEqual(after, ledger.snapshot())

    def test_actual_terminal_observation_precedes_finish_and_keeps_failure(self):
        mapping, ledger = self.bootstrap()
        actual = ledger.observe(mapping, frame=frame(2), frame_id=151,
                                action_id=151, map_version=152, done=True,
                                collision=True, reason="physical_collision")
        before = ledger.snapshot()
        existing = ledger.finish("guard_stopped", failed=True)
        self.assertEqual(existing["reason"], "physical_collision")
        self.assertEqual(existing["feedback_version"], actual["feedback_version"])
        self.assertFalse(existing["accepted"])
        self.assertEqual(before, ledger.snapshot())
        self.assertEqual(before["consumed_action_ids"], [151])


if __name__ == "__main__":
    unittest.main()
