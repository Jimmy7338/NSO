"""Real CPU four-interface lifecycle tests; no reconstruction efficacy claim.

The small renderer is a development fixture. No reserved test world, simulator
truth map, mocked candidate score or synthetic completed trajectory enters the
planner. The only wrapped calls observe ordering while executing real methods.
"""
from contextlib import ExitStack
from dataclasses import replace
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

from env.virtual3d import VirtualWorld
from env.virtual3d_v2 import VirtualConfigV2
from nso.components import NSO_Components
from nso.cpu_sensor_contract_v10 import GridTransform, SensorPacket, digest
from nso.runtime_integration import NSORuntimeIntegration
from nso.observed_runtime_mapper_v10 import ObservedRuntimeMapperV10


def arguments(**changes):
    values = dict(nso_backend="cpu_v10", eval=True, train_global=False,
                  use_open_vocab_semantic=True, use_topo_graph=True,
                  use_rpn_uq=True, use_igcr=True, cpu_score_mode="S",
                  cpu_max_candidates=12, run_id="cpu_v10_integration_test")
    values.update(changes)
    return SimpleNamespace(**values)


def fixture(*, max_steps=24, budget=24, episode="episode-0", start=True,
            **arg_changes):
    config = VirtualConfigV2(width_px=32, height_px=24, depth_sigma_m=0.,
                             dropout=0., max_steps=max_steps)
    world = VirtualWorld(config, seed=21, layout="rooms")
    components = NSO_Components(arguments(**arg_changes))
    components.initialize("cpu", 1, *world.shape, *world.shape)
    runtime = NSORuntimeIntegration(components, 1, world.shape)
    initial = packet(world, world.sense(), episode=episode)
    if start:
        start_episode(runtime, world, [initial], budget)
    return world, components, runtime, initial


def packet(world, frame, *, episode="episode-0", action=None,
           collision=False, done=False):
    return SensorPacket("development-rooms", episode, f"frame-{world.step_count}",
                        world.step_count, frame, world.scan(),
                        tuple(map(int, world.position)), int(world.heading),
                        "rendered_rgb_marker", "simulator_exact", action,
                        bool(collision), bool(done))


def start_episode(runtime, world, packets, budget, paid_prefix=0):
    first = packets[0]
    return runtime.start_sensor_episode(0, config=world.config,
        transform=GridTransform(world.shape, world.config.resolution_m),
        packets=packets, total_budget=budget,
        return_anchor=(*first.position, first.heading), paid_prefix_actions=paid_prefix)


def perform(world, runtime, action):
    frame, collision, done = world.step(action)
    episode = runtime.states[0]["packet"].episode_id
    observed = packet(world, frame, episode=episode, action=action,
                      collision=collision, done=done)
    result = runtime.observe(0, world.step_count, None, None, None,
                             sensor_packet=observed)
    return observed, result


def measured_signature(components, runtime):
    """Include actual TSDF voxel values, measured quality and the cost ledger."""
    state = runtime.states[0]
    mapper = state["mapper"]
    voxels = mapper.volume.extract_voxel_point_cloud()
    return digest(dict(frames=mapper.frames, belief=mapper.belief,
        camera_seen=mapper.camera_seen, visible=mapper.visible,
        quality={str(k): v for k, v in mapper.quality.items()},
        tsdf_points=np.asarray(voxels.points), tsdf_values=np.asarray(voxels.colors),
        packet_hashes=state["packet_hashes"], action_frames=state["action_frames"],
        ledger=components._cpu_backend.scenes[0]["ledger"].snapshot(),
        module_calls=components._cpu_backend.calls, runtime_audit=runtime.audit))


class CPUClosedLoopTests(unittest.TestCase):
    def test_real_paid_episode_replans_through_all_four_interfaces_and_returns(self):
        world, components, runtime, initial = fixture(start=False)
        names = ("update_semantic", "update_topo", "select_topo_target",
                 "assess_local_action", "compute_reward")
        with ExitStack() as stack:
            invoked = {name: stack.enter_context(patch.object(
                components, name, wraps=getattr(components, name))) for name in names}
            start_episode(runtime, world, [initial], 24)
            while not runtime.states[0]["closed"]:
                before = runtime.sensor_episode_summary(0)["modules"]["feedback"]
                action = runtime.next_local_action(0)
                if action is None:
                    break
                state = runtime.states[0]
                self.assertIsNotNone(state["pending"])
                self.assertLessEqual(1 + state["pending"]["reserved_return_cost"],
                                     before["remaining_budget"])
                # Proposal poses and authorized actions are not observations.
                pending = runtime.sensor_episode_summary(0)["modules"]["feedback"]
                self.assertEqual(pending["actual_camera_pose_count"],
                                 before["actual_camera_pose_count"])
                self.assertEqual(pending["paid_actions"], before["paid_actions"])
                observed, result = perform(world, runtime, action)
                self.assertTrue(result["accepted"])
                self.assertFalse(observed.collision)
                after = runtime.sensor_episode_summary(0)["modules"]["feedback"]
                self.assertEqual(after["paid_actions"], before["paid_actions"] + 1)
                self.assertEqual(after["actual_camera_pose_count"],
                                 before["actual_camera_pose_count"] + 1)
                self.assertLessEqual(world.step_count, 24)
            self.assertTrue(all(method.call_count for method in invoked.values()))
            self.assertGreaterEqual(invoked["select_topo_target"].call_count, 2)
            self.assertEqual(invoked["compute_reward"].call_count, world.step_count)

        summary = runtime.sensor_episode_summary(0)
        self.assertTrue(summary["closed"])
        self.assertTrue(summary["termination"]["returned_to_anchor"])
        self.assertFalse(summary["termination"]["failed"])
        self.assertGreater(world.step_count, 1)
        self.assertEqual(summary["mapper_frames"], world.step_count + 1)
        self.assertEqual(summary["unique_packet_count"], world.step_count + 1)
        feedback = summary["modules"]["feedback"]
        self.assertEqual(feedback["paid_actions"], world.step_count)
        self.assertEqual(feedback["consumed_action_ids"], list(range(1, world.step_count + 1)))
        self.assertEqual(feedback["remaining_budget"], 24 - world.step_count)
        self.assertEqual(feedback["actual_camera_pose_count"], world.step_count + 1)
        self.assertEqual(feedback["planning_camera_pose_count"], world.step_count + 1)

        calls = components._cpu_backend.calls
        self.assertEqual(set(call["module"] for call in calls),
                         {"OV-SDF", "STGHP", "RPN-UQ", "IGCR"})
        choices = [call for call in calls if call["method"] == "select_topo_target"]
        self.assertGreaterEqual(len(choices), 2)
        first, second = choices[:2]
        self.assertGreater(second["map_version"], first["map_version"])
        self.assertGreater(second["inputs"]["planning_feedback_version"],
                           first["inputs"]["planning_feedback_version"])
        self.assertLess(second["inputs"]["remaining_budget"], first["inputs"]["remaining_budget"])
        self.assertNotEqual(second["inputs"]["planning_camera_poses_sha256"],
                            first["inputs"]["planning_camera_poses_sha256"])
        for call in choices:
            self.assertEqual(call["inputs"]["graph_version"], call["map_version"])
            self.assertEqual(call["inputs"]["semantic_version"], call["map_version"])
            selected = call["outputs"]["selected"]
            if selected is not None:
                self.assertEqual(len(selected["pose"]), 3)
                self.assertEqual(selected["selection_call_id"], call["call_id"])
                self.assertEqual(selected["return_anchor"], [*initial.position, initial.heading])
                self.assertLessEqual(selected["cost"], call["inputs"]["remaining_budget"])

        selected_by_id = {call["outputs"]["selected"]["option_id"]: call
                          for call in choices if call["outputs"]["selected"] is not None}
        authorizations = [event for event in runtime.audit
                          if event["event"] == "paid_action_authorized"]
        self.assertEqual(len(authorizations), world.step_count)
        self.assertTrue(any(event["option_id"] for event in authorizations))
        for event in authorizations:
            paid = [call for call in calls if call["method"] == "compute_reward"
                    and call["action_id"] == event["next_action_id"]]
            self.assertEqual(len(paid), 1)
            self.assertEqual(paid[0]["executed_option_id"], event["option_id"])
            self.assertEqual(paid[0]["selection_call_id"], event["selection_call_id"])
            if event["option_id"] is not None:
                self.assertEqual(selected_by_id[event["option_id"]]["call_id"],
                                 event["selection_call_id"])
        for call in calls:
            self.assertFalse(call["truth_used"])
            self.assertFalse(call["trained"])
            self.assertFalse(call["calibrated"])
        self.assertEqual(components.predict_reachability_uq(None), (None, None))

    def test_duplicate_packets_do_not_fuse_tsdf_or_charge_or_emit_feedback(self):
        world, components, runtime, initial = fixture()
        for observed in (initial, None):
            if observed is None:
                action = runtime.next_local_action(0)
                self.assertIsNotNone(action)
                observed, _ = perform(world, runtime, action)
            before = measured_signature(components, runtime)
            mapper = runtime.states[0]["mapper"]
            with patch.object(mapper, "update", wraps=mapper.update) as integrate:
                for _ in range(2):
                    result = runtime.observe(0, observed.action_id, None, None, None,
                                             sensor_packet=observed)
                    self.assertFalse(result["accepted"])
                    self.assertTrue(result["duplicate"])
                integrate.assert_not_called()
            self.assertEqual(measured_signature(components, runtime), before)

    def test_wrong_identity_conflicting_duplicate_and_unpaid_packet_are_atomic(self):
        world, components, runtime, initial = fixture()
        frame, collision, done = world.step("left")
        unpaid = packet(world, frame, action="left", collision=collision, done=done)
        rgb = initial.frame.color_rgb.copy()
        rgb[0, 0, 0] ^= 1
        variants = (replace(initial, scene_id="another-scene"),
                    replace(initial, episode_id="another-episode"),
                    replace(initial, frame=replace(initial.frame, color_rgb=rgb)), unpaid)
        before = measured_signature(components, runtime)
        mapper = runtime.states[0]["mapper"]
        with patch.object(mapper, "update", wraps=mapper.update) as integrate:
            for invalid in variants:
                with self.subTest(packet=invalid.frame_id, episode=invalid.episode_id):
                    with self.assertRaises(ValueError):
                        runtime.observe(0, invalid.action_id, None, None, None,
                                        sensor_packet=invalid)
                    self.assertEqual(measured_signature(components, runtime), before)
            integrate.assert_not_called()

    def test_pending_action_blocks_replan_close_and_reset(self):
        world, components, runtime, _ = fixture()
        with self.assertRaises(RuntimeError):
            runtime.reset_scene(0)
        action = runtime.next_local_action(0)
        self.assertIsNotNone(action)
        before = measured_signature(components, runtime)
        operations = (lambda: runtime.next_local_action(0),
                      lambda: runtime.choose_goal(0, list(world.position), (0, world.shape[0], 0, world.shape[1])),
                      lambda: runtime.close_sensor_episode(0, reason="external_stop"),
                      lambda: runtime.reset_scene(0))
        for operation in operations:
            with self.assertRaises(RuntimeError):
                operation()
            self.assertEqual(measured_signature(components, runtime), before)
        observed, _ = perform(world, runtime, action)
        self.assertEqual(runtime.states[0]["packet"].frame_id, observed.frame_id)

    def test_terminal_frame_is_fused_before_feedback_close_and_episode_reset(self):
        world, components, runtime, initial = fixture(max_steps=1)
        action = runtime.next_local_action(0)
        self.assertIsNotNone(action)
        events = []
        mapper = runtime.states[0]["mapper"]

        def observe_real(name, method):
            def invoke(*args, **kwargs):
                events.append((name, mapper.frames))
                return method(*args, **kwargs)
            return invoke

        with patch.object(mapper, "update", side_effect=observe_real("fusion", mapper.update)), \
             patch.object(components, "compute_reward", side_effect=observe_real("feedback", components.compute_reward)), \
             patch.object(runtime, "close_sensor_episode", side_effect=observe_real("close", runtime.close_sensor_episode)):
            terminal, result = perform(world, runtime, action)
        self.assertTrue(terminal.done)
        self.assertTrue(result["accepted"])
        self.assertEqual(events, [("fusion", 1), ("feedback", 2), ("close", 2)])
        # The fixture's first paid rotation has not restored the home heading.
        self.assertEqual(terminal.position, initial.position)
        self.assertNotEqual(terminal.heading, initial.heading)
        summary = runtime.sensor_episode_summary(0)
        self.assertTrue(summary["closed"])
        self.assertTrue(summary["termination"]["failed"])
        self.assertFalse(summary["termination"]["returned_to_anchor"])
        feedback = summary["modules"]["feedback"]
        self.assertTrue(feedback["done"])
        self.assertTrue(feedback["last_event"]["failed"])
        self.assertEqual(feedback["failure_events"], 1)
        self.assertEqual(feedback["map_version"], 2)
        self.assertEqual(feedback["last_event"]["frame_id"], terminal.frame_id)
        self.assertEqual(feedback["paid_actions"], 1)
        self.assertEqual(feedback["remaining_budget"], 23)
        self.assertIsNone(runtime.next_local_action(0))
        before = measured_signature(components, runtime)
        duplicate = runtime.observe(0, 1, None, None, None, sensor_packet=terminal)
        self.assertTrue(duplicate["duplicate"])
        self.assertEqual(measured_signature(components, runtime), before)
        runtime.reset_scene(0)
        self.assertIsNone(runtime.states[0])
        self.assertIsNone(components._cpu_backend.scenes[0])
        self.assertEqual(runtime.completed_episodes[-1], summary)
        fresh_world = VirtualWorld(world.config, seed=21, layout="rooms")
        same_identity = packet(fresh_world, fresh_world.sense())
        with self.assertRaises(ValueError):
            start_episode(runtime, fresh_world, [same_identity], 24)
        next_episode = replace(same_identity, episode_id="episode-1")
        start_episode(runtime, fresh_world, [next_episode], 24)
        next_feedback = runtime.sensor_episode_summary(0)["modules"]["feedback"]
        self.assertEqual(next_feedback["paid_actions"], 0)
        self.assertEqual(next_feedback["map_version"], 1)
        self.assertEqual(next_feedback["actual_camera_pose_count"], 1)
        with self.assertRaises(ValueError):
            runtime.observe(0, 1, None, None, None, sensor_packet=terminal)

    def test_invalid_budget_and_inconsistent_prefix_reject_before_mapper_creation(self):
        world, components, runtime, initial = fixture(start=False)
        frame, collision, done = world.step("right")
        follow = packet(world, frame, action="right", collision=collision, done=done)
        invalid_prefix = [initial, replace(follow, action="left")]
        with patch("nso.observed_runtime_mapper_v10.ObservedRuntimeMapperV10",
                   wraps=ObservedRuntimeMapperV10) as mapper_constructor:
            with self.assertRaises(ValueError):
                start_episode(runtime, world, [initial], 0)
            with self.assertRaises(ValueError):
                start_episode(runtime, world, invalid_prefix, 8, paid_prefix=1)
            with self.assertRaises(ValueError):
                start_episode(runtime, world, [initial, follow], 8, paid_prefix=0)
            mapper_constructor.assert_not_called()
        self.assertIsNone(runtime.states[0])
        self.assertIsNone(components._cpu_backend.scenes[0])
        self.assertEqual(components._cpu_backend.calls, [])
        summary = start_episode(runtime, world, [initial, follow], 8, paid_prefix=1)
        self.assertEqual(summary["externally_scripted_prefix_actions"], 1)
        self.assertEqual(summary["mapper_frames"], 2)
        self.assertEqual(summary["modules"]["feedback"]["paid_actions"], 1)
        self.assertEqual(summary["modules"]["feedback"]["remaining_budget"], 7)

    def test_sensor_obstacle_at_anchor_stops_without_paid_action_and_records_failure(self):
        world, components, runtime, initial = fixture(start=False)
        # Explicit sensor-conflict negative control, not a claim about the
        # renderer's normal output: a 5 cm lidar hit lies in the current cell.
        ranges = initial.scan.ranges_m.copy()
        ranges[0] = .05
        conflict = replace(initial, scan=replace(initial.scan, ranges_m=ranges))
        start_episode(runtime, world, [conflict], 24)
        state = runtime.states[0]
        self.assertEqual(int(state["mapper"].belief[conflict.position]), 1)
        self.assertTrue(state["mapper"].current_footprint_conflict)
        self.assertIsNone(runtime.next_local_action(0))
        summary = runtime.sensor_episode_summary(0)
        self.assertTrue(summary["closed"])
        self.assertTrue(summary["termination"]["returned_to_anchor"])
        self.assertTrue(summary["termination"]["failed"])
        feedback = summary["modules"]["feedback"]
        self.assertTrue(feedback["last_event"]["failed"])
        self.assertEqual(feedback["failure_events"], 1)
        self.assertEqual(feedback["paid_actions"], 0)
        self.assertEqual(feedback["remaining_budget"], 24)
        self.assertEqual(feedback["actual_camera_pose_count"], 1)
        self.assertEqual(summary["mapper_frames"], 1)
        self.assertFalse(any(event["event"] == "paid_action_authorized" for event in runtime.audit))

    def test_feedback_ablation_keeps_actual_cost_but_freezes_planning_history(self):
        cases = [fixture(cpu_disable_feedback=disabled) for disabled in (False, True)]
        first_actions = []
        for world, components, runtime, _ in cases:
            action = runtime.next_local_action(0)
            self.assertIsNotNone(action)
            first_actions.append(action)
            perform(world, runtime, action)
            runtime.choose_goal(0, list(world.position), (0, world.shape[0], 0, world.shape[1]))
        self.assertEqual(first_actions[0], first_actions[1])
        actual, frozen = [runtime.sensor_episode_summary(0)["modules"]["feedback"]
                          for _, _, runtime, _ in cases]
        for feedback in (actual, frozen):
            self.assertEqual(feedback["paid_actions"], 1)
            self.assertEqual(feedback["remaining_budget"], 23)
            self.assertEqual(feedback["actual_camera_pose_count"], 2)
            self.assertEqual(feedback["map_version"], 2)
        self.assertEqual(actual["planning_camera_pose_count"], 2)
        self.assertEqual(frozen["planning_camera_pose_count"], 1)
        self.assertEqual(actual["actual_camera_poses_sha256"], frozen["actual_camera_poses_sha256"])
        self.assertNotEqual(actual["planning_camera_poses_sha256"], frozen["planning_camera_poses_sha256"])
        for (_, components, _, _), feedback in zip(cases, (actual, frozen)):
            choice = [call for call in components._cpu_backend.calls
                      if call["method"] == "select_topo_target"][-1]
            self.assertEqual(choice["inputs"]["planning_feedback_version"],
                             feedback["planning_feedback_version"])
            self.assertEqual(choice["inputs"]["remaining_budget"], 23)

    def test_cpu_backend_cannot_replace_ppo_training_or_silently_disable_interfaces(self):
        for changes in (dict(train_global=True, eval=False), dict(use_igcr=False)):
            components = NSO_Components(arguments(**changes))
            with self.assertRaises(ValueError):
                components.initialize("cpu", 1, 30, 40, 30, 40)
            self.assertIsNone(components._cpu_backend)


if __name__ == "__main__":
    unittest.main()
