"""Runtime contract tests with handwritten packets and a non-fusing mapper.

No World, step, sense, saved experiment, or TSDF integration is used. Most
tests stub only global target selection; the real local return guard and
packet/state validation still run. One tiny static map uses the real planner
to check the four-interface call chain, not closed-loop experimental efficacy.
"""
from dataclasses import replace
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

from nso.cpu_sensor_contract_v10 import SensorPacket
from nso.observed_planner_v26 import MODES
from nso.observed_runtime_v26 import ObservedANSRuntimeV26
from utils.rgbd_contract import RGBDFrame, PlanarScan


SHAPE = (31, 31)
START = (15, 15)


def config():
    return SimpleNamespace(resolution_m=.2, robot_radius_m=.2, max_depth_m=5.,
        fov_deg=90., width_px=16, height_px=16, camera_height_m=.8,
        laser_height_m=.25, laser_rays=16)


def packet(action_id=0, *, position=START, heading=0, action=None,
           frame_id=None, collision=False, done=False, marker=False):
    """A declared analytic input, not a sensor sample from any virtual world."""
    forward = np.asarray(((0., 1., 0.), (1., 0., 0.),
                          (0., -1., 0.), (-1., 0., 0.))[heading])
    right = np.cross(forward, (0., 0., 1.))
    camera = np.eye(4)
    camera[:3, :3] = np.column_stack((right, (0., 0., -1.), forward))
    camera[:3, 3] = ((position[1]+.5)*.2, (SHAPE[0]-position[0]-.5)*.2, .8)
    laser = np.eye(4)
    laser[:3, 0], laser[:3, 1] = forward, -right
    laser[:3, 3] = camera[:3, 3]; laser[2, 3] = .25
    intrinsic = np.array([[8., 0., 7.5], [0., 8., 7.5], [0., 0., 1.]])
    rgb = np.zeros((16, 16, 3), np.uint8)
    if marker:
        rgb[5:9, 5:9] = (220, 60, 40)
    frame = RGBDFrame(float(action_id), np.full((16, 16), 2., np.float32),
        rgb, intrinsic, camera, np.zeros((16, 16), np.int32))
    scan = PlanarScan(float(action_id), np.full(16, 5., np.float32),
        -np.pi, 2*np.pi/16, 5., laser)
    return SensorPacket('analytic-runtime-contract', 'one-unit-episode',
        frame_id or f'unit-frame-{action_id}', action_id, frame, scan,
        tuple(position), heading, 'handwritten_unit_fixture_not_sensor_collection',
        'declared_discrete_test_pose', action, collision, done)


class NonFusingMapper:
    """Consume counters/stamps only; the map is a hand-authored unit fixture."""
    def __init__(self, shape, sensor_config):
        self.shape, self.config = tuple(shape), sensor_config
        self.belief = np.zeros(shape, np.int8)
        self.belief[0, :] = -1
        self.quality = {}
        self.frames = 0
        self.current_footprint_conflict_details = None
        self.update_times = []

    def update(self, frame, scan):
        self.frames += 1
        self.update_times.append(frame.timestamp_s)
        xyz = frame.world_from_camera[:3, 3]
        cell = (self.shape[0]-1-int(np.floor(xyz[1]/.2)), int(np.floor(xyz[0]/.2)))
        self.current_footprint_conflict_details = dict(map_version=self.frames,
            timestamp_s=frame.timestamp_s, current_cell=cell)


class ObservedRuntimeV26Tests(unittest.TestCase):
    def runtime(self, mode='G', budget=20):
        # The import path patches runtime construction only. No real mapper
        # or Open3D integration object is constructed by these tests.
        with patch('nso.observed_runtime_v26.ObservedRuntimeMapperV10', NonFusingMapper):
            return ObservedANSRuntimeV26(SHAPE, config(), budget, mode=mode)

    def forward(self, runtime):
        r, c = runtime.state.position
        self.assertEqual(runtime.state.heading, 0)
        selected = dict(pose=[r-1, c, 0], group='unit_target', semantic_evidence=[])
        with patch.object(runtime.planner, 'plan', return_value=dict(
                selected=selected, candidates=[selected], audit={'unit_stub': True})):
            return runtime.next_action()

    def test_real_action_zero_required_and_nonzero_prefix_rejected_before_mapping(self):
        for first in (packet(7, action='forward'), packet(0, action='right')):
            with self.subTest(action_id=first.action_id, action=first.action):
                runtime = self.runtime()
                with self.assertRaisesRegex(ValueError, 'action-zero'):
                    runtime.accept(first)
                self.assertEqual(runtime.mapper.frames, 0)
                self.assertIsNone(runtime.packet)
                self.assertIsNone(runtime.anchor)
                self.assertFalse(runtime.seen_frame_ids)
                state = runtime.accept(packet())
                self.assertEqual(state.action_id, 0)
                self.assertEqual(state.remaining_budget, 20)

    def test_actual_interface_chain_with_real_planner_and_analytic_mapper(self):
        runtime = self.runtime()
        initial = runtime.accept(packet())
        action = runtime.next_action()
        self.assertEqual(action, 'forward')
        self.assertEqual(runtime.mapper.frames, 1)
        current = runtime.accept(packet(1, position=(14, 15), action=action))
        self.assertEqual(current.action_id, 1)
        self.assertEqual(current.remaining_budget, 19)
        self.assertEqual(initial.position, START)
        self.assertEqual(runtime.mapper.update_times, [0., 1.])
        self.assertIsNone(runtime.pending)
        self.assertIsNone(runtime.semantic)
        self.assertEqual(runtime.cues, ())
        calls = runtime.planner.calls
        self.assertEqual({r['module'] for r in calls}, {'OV-SDF', 'STGHP', 'RPN-UQ', 'IGCR'})
        self.assertEqual(calls[-1]['operation'], 'measured_transition_feedback')
        self.assertEqual(calls[-1]['action_id'], 1)
        self.assertEqual(runtime.summary()['actual_paid_actions'], 1)
        self.assertFalse(runtime.summary()['full_method_efficacy_proven'])

    def test_pending_action_cannot_be_reissued_and_unsolicited_observation_rejected(self):
        runtime = self.runtime(); runtime.accept(packet())
        with self.assertRaisesRegex(ValueError, 'issued paid action'):
            runtime.accept(packet(1, position=(14, 15), action='forward'))
        self.assertEqual(runtime.mapper.frames, 1)
        self.assertEqual(self.forward(runtime), 'forward')
        calls = len(runtime.planner.calls)
        with self.assertRaisesRegex(ValueError, 'previous paid observation'):
            runtime.next_action()
        self.assertEqual(len(runtime.planner.calls), calls)
        self.assertEqual(runtime.pending, 'forward')
        runtime.accept(packet(1, position=(14, 15), action='forward'))
        self.assertEqual(runtime.mapper.frames, 2)

    def test_wrong_action_or_self_consistent_wrong_pose_rejected_without_consumption(self):
        bad_packets = [
            packet(1, heading=1, action='right'),
            packet(1, position=START, action='forward'),
            packet(1, position=(14, 15), heading=1, action='forward'),
        ]
        for bad in bad_packets:
            with self.subTest(action=bad.action, position=bad.position, heading=bad.heading):
                runtime = self.runtime(); runtime.accept(packet())
                self.assertEqual(self.forward(runtime), 'forward')
                bad.validate(runtime.transform, runtime.config)
                old_state = runtime.state
                with self.assertRaises(ValueError):
                    runtime.accept(bad)
                self.assertIs(runtime.state, old_state)
                self.assertEqual(runtime.mapper.frames, 1)
                self.assertEqual(runtime.pending, 'forward')
                self.assertEqual(runtime.seen_frame_ids, {'unit-frame-0'})
                runtime.accept(packet(1, position=(14, 15), action='forward'))
                self.assertEqual(runtime.mapper.frames, 2)

    def test_every_mode_rejects_reused_nonadjacent_frame_identity(self):
        for mode in MODES:
            with self.subTest(mode=mode):
                runtime = self.runtime(mode); runtime.accept(packet())
                self.assertEqual(self.forward(runtime), 'forward')
                runtime.accept(packet(1, position=(14, 15), action='forward'))
                self.assertEqual(self.forward(runtime), 'forward')
                replay_id = packet(2, position=(13, 15), action='forward', frame_id='unit-frame-0')
                with self.assertRaisesRegex(ValueError, 'identity already consumed'):
                    runtime.accept(replay_id)
                self.assertEqual(runtime.mapper.frames, 2)
                self.assertEqual(runtime.state.action_id, 1)
                self.assertEqual(runtime.pending, 'forward')
                self.assertEqual(runtime.seen_frame_ids, {'unit-frame-0', 'unit-frame-1'})
                runtime.accept(packet(2, position=(13, 15), action='forward'))
                self.assertEqual(runtime.mapper.frames, 3)
                self.assertEqual(runtime.state.action_id, 2)

    def test_collision_is_paid_and_terminal_without_gain_calibration(self):
        runtime = self.runtime('S'); runtime.accept(packet(marker=True))
        self.assertTrue(runtime.cues)
        self.assertEqual(self.forward(runtime), 'forward')
        with patch.object(runtime.planner, 'observe_transition', wraps=runtime.planner.observe_transition) as feedback:
            runtime.accept(packet(1, position=START, action='forward', collision=True, marker=True))
            feedback.assert_not_called()
        self.assertEqual(runtime.mapper.frames, 2)
        self.assertEqual(runtime.state.remaining_budget, 19)
        self.assertEqual(runtime.summary()['actual_paid_actions'], 1)
        self.assertEqual(runtime.terminal_reason, 'collision_observed')
        self.assertEqual(runtime.planner.feedback, {})
        self.assertEqual(runtime.planner.calls[-1]['operation'], 'collision_transition_no_gain_calibration')
        self.assertFalse(runtime.planner.calls[-1]['quality_credit'])
        self.assertIsNone(runtime.next_action())
        with self.assertRaisesRegex(ValueError, 'already stopped'):
            runtime.accept(packet(2, position=(14, 15), action='forward'))
        self.assertEqual(runtime.mapper.frames, 2)

    def test_foreign_episode_noncontinuous_action_and_stale_time_rejected_before_fusion(self):
        good = packet(1, position=(14, 15), action='forward')
        invalid = [replace(good, episode_id='other-episode'), replace(good, scene_id='other-scene'),
            replace(good, action_id=2), replace(good, frame=replace(good.frame, timestamp_s=0.),
                scan=replace(good.scan, timestamp_s=0.))]
        for bad in invalid:
            with self.subTest(episode=bad.episode_id, scene=bad.scene_id, action_id=bad.action_id,
                              time=bad.frame.timestamp_s):
                runtime = self.runtime(); runtime.accept(packet())
                self.assertEqual(self.forward(runtime), 'forward')
                with self.assertRaisesRegex(ValueError, 'issued paid action'):
                    runtime.accept(bad)
                self.assertEqual(runtime.mapper.frames, 1)
                self.assertIsNone(runtime.planner.feedback_action)
                runtime.accept(good)
                self.assertEqual(runtime.mapper.frames, 2)

    def test_target_without_affordable_heading_restoring_return_does_not_issue_action(self):
        # One northward step then returning to the original north-facing pose
        # costs six paid actions (F, two turns, F, two turns), not two.
        runtime = self.runtime(budget=5); runtime.accept(packet())
        self.assertIsNone(self.forward(runtime))
        self.assertIsNone(runtime.pending)
        self.assertEqual(runtime.mapper.frames, 1)
        self.assertEqual(runtime.summary()['actual_paid_actions'], 0)
        self.assertEqual(runtime.state.position, START)


if __name__ == '__main__':
    unittest.main()
