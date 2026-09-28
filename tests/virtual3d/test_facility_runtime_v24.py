"""Small analytic checks for full-prefix accounting; no simulator or saved data.

The sequence mapper exposes loss/re-observation that a monotonic map cannot
produce in a short fixture. Separate 8x8 calibrated packet tests exercise the
real authoritative mapper and unmodified subsequent execution guards.
"""
from copy import deepcopy
from dataclasses import replace
from types import SimpleNamespace
import unittest

import numpy as np

from env.virtual3d import VirtualConfig
from nso.cpu_sensor_contract_v10 import GridTransform, SensorPacket
from nso.facility_runtime_v21 import FacilityRuntimeV21, facility_components_v21
from nso.facility_runtime_v24 import (FacilityRuntimeV24, PREFIX_INTENT_SOURCE,
    _map_prefix_with_coverage_v24, facility_components_v24)
from utils.rgbd_contract import RGBDFrame, PlanarScan


SHAPE = (12, 12)
CONFIG = VirtualConfig(width_px=8, height_px=8, robot_radius_m=0.,
    max_depth_m=.8, depth_sigma_m=0., dropout=0., laser_rays=4, voxel_m=.04)
TRANSFORM = GridTransform(SHAPE, CONFIG.resolution_m)
ANCHOR = (6, 6, 0)


def args():
    return SimpleNamespace(nso_backend='cpu_v10', eval=True, train_global=False,
        use_open_vocab_semantic=True, use_topo_graph=True, use_rpn_uq=True, use_igcr=True,
        cpu_score_mode='N', cpu_disable_feedback=False, cpu_max_candidates=5,
        cpu_coverage_slots=4, cpu_planner_revision='v10_3_1',
        cpu_semantic_source_schema='inspection_v4', cpu_measured_novelty_floor=.25,
        cpu_task_asset_count=2, cpu_v20_replan_interval=5, cpu_v20_coverage_slots=8)


def packet(action_id, *, pose=None, action=None, category=None, episode='analytic'):
    """Analytic calibrated camera/scan, without constructing a truth world."""
    pose = (*ANCHOR[:2], action_id % 4) if pose is None else tuple(pose)
    action = (None if action_id == 0 else 'right') if action is None else action
    forward = np.asarray(((0., 1., 0.), (1., 0., 0.),
                          (0., -1., 0.), (-1., 0., 0.))[pose[2]])
    optical = np.eye(4)
    optical[:3, :3] = np.column_stack((np.cross(forward, (0., 0., 1.)),
                                       (0., 0., -1.), forward))
    optical[:3, 3] = (*TRANSFORM.cell_to_world(pose[:2]), CONFIG.camera_height_m)
    laser = np.eye(4)
    laser[:3, 0], laser[:3, 1] = forward, -optical[:3, 0]
    laser[:3, 3] = optical[:3, 3]; laser[2, 3] = CONFIG.laser_height_m
    depth = np.zeros((8, 8), np.float32)
    labels = np.zeros((8, 8), np.uint8)
    color = np.zeros((8, 8, 3), np.uint8)
    if category is not None:
        depth[:] = .6; labels[:] = category
        color[:] = (76, 153, 204) if category == 2 else (230, 128, 51)
    frame = RGBDFrame(float(action_id), depth, color,
        np.array([[4., 0., 3.5], [0., 4., 3.5], [0., 0., 1.]]), optical, labels)
    scan = PlanarScan(float(action_id), np.full(4, .4, np.float32),
        -np.pi, np.pi/2, CONFIG.max_depth_m, laser)
    return SensorPacket('fixture', episode, 'frame-' + str(action_id), action_id,
        frame, scan, pose[:2], pose[2], 'analytic_arrays', 'analytic_calibration', action)


class SequenceMapper:
    """Inject post-fusion beliefs, without replacing any production factory."""
    shape = SHAPE
    config = CONFIG

    def __init__(self, beliefs):
        self.beliefs = [b.copy() for b in beliefs]
        self.belief = np.full(SHAPE, -1, np.int8)
        self.frames = 0
        self.inputs = []

    def update(self, frame, scan):
        self.inputs.append((frame, scan))
        self.belief[:] = self.beliefs[self.frames]
        self.frames += 1


def initial_belief():
    value = np.full(SHAPE, -1, np.int8)
    value[ANCHOR[:2]] = 0
    return value


def new_runtime(version=24):
    build, runtime_class = ((facility_components_v24, FacilityRuntimeV24) if version == 24
                            else (facility_components_v21, FacilityRuntimeV21))
    return runtime_class(build(args(), SHAPE), 1, SHAPE)


def start(runtime, packets, *, budget=50, intent=True):
    extra = ({'external_prefix_coverage_intent': intent}
             if isinstance(runtime, FacilityRuntimeV24) and len(packets) > 1 else {})
    return runtime.start_sensor_episode(0, config=CONFIG, transform=TRANSFORM,
        packets=packets, total_budget=budget, paid_prefix_actions=len(packets)-1,
        return_anchor=ANCHOR, **extra)


class FullPrefixCoverageV24Tests(unittest.TestCase):
    def test_one_pass_records_actual_gain_loss_and_every_paid_turn_including_short_tail(self):
        first = initial_belief(); gained = first.copy(); gained[4, 3:7] = (0, 1, 0, 1)
        lost = gained.copy(); lost[4, 3:5] = -1
        final = lost.copy(); final[4, 3] = 0
        beliefs = [first, gained, gained, lost, lost, lost, final]
        packets = [packet(i) for i in range(len(beliefs))]
        mapper = SequenceMapper(beliefs)
        ledger, receipt = _map_prefix_with_coverage_v24(mapper, packets, ANCHOR, 5)
        now = ledger.snapshot()
        self.assertEqual(mapper.frames, 7)
        self.assertTrue(all(actual[0] is p.frame and actual[1] is p.scan
                            for actual, p in zip(mapper.inputs, packets)))
        self.assertEqual([e['net_known_gain_cells'] for e in ledger.events], [4, 0, -2, 0, 0, 1])
        self.assertEqual(now['known_cells'], 4)
        self.assertEqual(now['observed_actions'], 6)
        self.assertEqual(now['rate_history_paid_actions'], 6)
        self.assertEqual(now['rate_history_net_known_gain'], 3)
        self.assertEqual(now['conservative_rate'], .25)
        self.assertEqual([p['paid_actions'] for p in now['coverage_prefixes']], [5, 1])
        self.assertIsNone(now['pending_prefix'])
        self.assertEqual(receipt['final_short_prefix']['paid_actions'], 1)
        self.assertEqual(receipt['intent_source'], PREFIX_INTENT_SOURCE)
        self.assertTrue(all(e['coverage_intent'] for e in ledger.events))
        # A scripted prefix did not make route-mask predictions. Real repeated
        # radar observations count as scans, never as invented yield successes.
        self.assertEqual(now['scan_observations'], 7)
        self.assertEqual(now['union_yield'], .5)
        self.assertTrue(all(e['predicted_union_cells'] is None for e in ledger.events))
        self.assertFalse(receipt['positive_rate_or_budget_admission_guaranteed'])
        before = deepcopy(now)
        mapper.belief[:] = 1
        receipt['final_coverage']['known_cells'] = 999
        self.assertEqual(ledger.snapshot(), before)

    def test_late_zero_gain_return_history_does_not_inherit_optimistic_lifetime_rate(self):
        first = initial_belief(); seen = first.copy(); seen[4, :] = 0
        beliefs = [first] + [seen] * 35
        ledger, receipt = _map_prefix_with_coverage_v24(SequenceMapper(beliefs),
            [packet(i) for i in range(36)], ANCHOR, 5)
        now = ledger.snapshot()
        self.assertEqual(now['observed_actions'], 35)
        self.assertEqual(now['rate_history_paid_actions'], 30)
        self.assertEqual(now['rate_history_net_known_gain'], 0)
        self.assertEqual(now['conservative_rate'], 0.)
        self.assertEqual(now['known_cells'], 13)
        self.assertGreater(sum(e['actual_new_known_cells'] for e in ledger.events), 0)
        answer = ledger.assess_route(dict(outbound_cost=2, return_cost=2),
            np.zeros(SHAPE, bool), 100)
        self.assertFalse(answer['allowed'])
        self.assertEqual(answer['reason'], 'coverage_rate_unavailable')
        self.assertIsNone(receipt['final_short_prefix'])

    def test_real_mapper_full_prefix_has_one_ledger_and_no_double_fusion_or_paid_credit(self):
        runtime = new_runtime()
        history = [packet(i) for i in range(7)]
        start(runtime, history)
        state = runtime.states[0]; backend = runtime.components._cpu_backend
        scene = backend.scenes[0]; ledger = scene['coverage_v21']
        self.assertIs(scene['mapper'], state['mapper'])
        self.assertEqual(state['mapper'].frames, 7)
        self.assertEqual(scene['ledger'].snapshot()['paid_actions'], 6)
        self.assertEqual(scene['ledger'].remaining_budget, 44)
        self.assertEqual(len(scene['axis_frames']), 7)
        self.assertEqual(len(state['packet_hashes']), 7)
        self.assertEqual(len(ledger.events), 6)
        self.assertEqual(ledger.action_id, 6)
        self.assertEqual(ledger.snapshot()['scan_observations'], 7)
        np.testing.assert_array_equal(ledger.belief, state['mapper'].belief)
        receipt = scene['v24_prefix_coverage_receipt']
        self.assertEqual([r['mapper_frames'] for r in receipt['paid_observations']], list(range(2, 8)))
        self.assertEqual([r['frame_id'] for r in receipt['paid_observations']],
                         [p.frame_id for p in history[1:]])
        methods = [c['method'] for c in backend.calls]
        self.assertEqual(methods.count('coverage_bootstrap_v24'), 1)
        self.assertNotIn('coverage_bootstrap_v21', methods)
        self.assertNotIn('compute_reward', methods)
        # Receiving a saved prefix does not authorize any new physical action.
        self.assertIsNone(state['pending'])
        self.assertEqual(runtime.audit, [])

    def test_swapping_actual_visible_categories_does_not_change_prefix_coverage_history(self):
        ledgers = []
        for category in (2, 3):
            runtime = new_runtime()
            start(runtime, [packet(i, category=category) for i in range(3)])
            ledgers.append(runtime.components._cpu_backend.scenes[0]['coverage_v21'])
        np.testing.assert_array_equal(ledgers[0].belief, ledgers[1].belief)
        self.assertEqual(ledgers[0].events, ledgers[1].events)
        self.assertEqual(ledgers[0].snapshot(), ledgers[1].snapshot())

    def test_action_zero_start_preserves_old_map_feedback_and_module_bootstrap(self):
        old, new = new_runtime(21), new_runtime()
        history = [packet(0)]
        start(old, history); start(new, history)
        old_scene = old.components._cpu_backend.scenes[0]
        new_scene = new.components._cpu_backend.scenes[0]
        self.assertEqual(old_scene['coverage_v21'].snapshot(), new_scene['coverage_v21'].snapshot())
        self.assertEqual(old_scene['ledger'].snapshot(), new_scene['ledger'].snapshot())
        np.testing.assert_array_equal(old.states[0]['mapper'].belief, new.states[0]['mapper'].belief)
        self.assertNotIn('v24_prefix_coverage_receipt', new_scene)
        def calls(runtime):
            return [{k: v for k, v in c.items() if k != 'elapsed_s'}
                    for c in runtime.components._cpu_backend.calls]
        self.assertEqual(calls(old), calls(new))
        self.assertIsNone(new_scene['coverage_v21'].snapshot()['conservative_rate'])

    def test_normal_return_observation_uses_same_ledger_and_its_original_noncoverage_intent(self):
        runtime = new_runtime(); start(runtime, [packet(i) for i in range(3)])
        state = runtime.states[0]; backend = runtime.components._cpu_backend
        scene = backend.scenes[0]; ledger = scene['coverage_v21']
        history_before = deepcopy(ledger.snapshot()['coverage_prefixes'])
        # Enter the ordinary return phase. The inherited planner still computes
        # the actions and guard authorization; no action or prediction is faked.
        state['phase'] = 'return'
        action = runtime.next_local_action(0)
        self.assertIn(action, ('left', 'right'))
        p = packet(3, pose=state['pending']['next_pose'], action=action)
        event = runtime._observe_sensor_packet(0, p)
        self.assertTrue(event['accepted'])
        self.assertIs(scene['coverage_v21'], ledger)
        self.assertEqual(scene['ledger'].snapshot()['paid_actions'], 3)
        self.assertEqual(state['mapper'].frames, 4)
        self.assertEqual(ledger.action_id, 3)
        self.assertEqual(ledger.snapshot()['observed_actions'], 3)
        self.assertFalse(ledger.events[-1]['coverage_intent'])
        self.assertEqual(ledger.snapshot()['coverage_prefixes'], history_before)
        before = deepcopy(ledger.snapshot())
        duplicate = runtime._observe_sensor_packet(0, p)
        self.assertTrue(duplicate['duplicate'])
        self.assertEqual(ledger.snapshot(), before)

    def test_paid_prefix_requires_explicit_common_intent_and_correct_new_backend(self):
        history = [packet(0), packet(1)]
        for intent in (None, False, 1, 'coverage'):
            with self.subTest(intent=intent):
                runtime = new_runtime()
                with self.assertRaisesRegex(ValueError, 'explicit'):
                    start(runtime, history, intent=intent)
                self.assertIsNone(runtime.states[0])
                self.assertIsNone(runtime.components._cpu_backend.scenes[0])
        runtime = FacilityRuntimeV24(facility_components_v21(args(), SHAPE), 1, SHAPE)
        with self.assertRaisesRegex(RuntimeError, 'facility_components_v24'):
            start(runtime, history)
        runtime = new_runtime()
        with self.assertRaisesRegex(ValueError, 'action-zero'):
            runtime.start_sensor_episode(0, config=CONFIG, transform=TRANSFORM,
                packets=[packet(0)], total_budget=10, return_anchor=ANCHOR,
                external_prefix_coverage_intent=True)

    def test_incomplete_or_inconsistent_paid_history_is_rejected_before_scene_installation(self):
        initial, following = packet(0), packet(1)
        earlier = replace(following, frame=replace(following.frame, timestamp_s=0.),
                          scan=replace(following.scan, timestamp_s=0.))
        bad_histories = {
            'missing_initial': [following],
            'skipped_action': [initial, replace(following, action_id=2)],
            'duplicate_frame': [initial, replace(following, frame_id=initial.frame_id)],
            'different_episode': [initial, replace(following, episode_id='foreign')],
            'nonincreasing_time': [initial, earlier],
            'wrong_recorded_turn': [initial, replace(following, action='left')],
            'terminal_prefix': [initial, replace(following, done=True)],
            'collision_prefix': [initial, replace(following, collision=True)],
        }
        for name, history in bad_histories.items():
            with self.subTest(name=name):
                runtime = new_runtime()
                with self.assertRaises(ValueError):
                    runtime.start_sensor_episode(0, config=CONFIG, transform=TRANSFORM,
                        packets=history, total_budget=10, return_anchor=ANCHOR,
                        paid_prefix_actions=1, external_prefix_coverage_intent=True)
                self.assertIsNone(runtime.states[0])
                self.assertIsNone(runtime.components._cpu_backend.scenes[0])
        for paid, budget in ((2, 10), (1, 0), (1, True), (True, 10)):
            with self.subTest(paid=paid, budget=budget):
                runtime = new_runtime()
                with self.assertRaises(ValueError):
                    runtime.start_sensor_episode(0, config=CONFIG, transform=TRANSFORM,
                        packets=[initial, following], total_budget=budget,
                        return_anchor=ANCHOR, paid_prefix_actions=paid,
                        external_prefix_coverage_intent=True)
                self.assertIsNone(runtime.states[0])

    def test_reset_requires_fresh_episode_identity_without_retaining_old_coverage_state(self):
        runtime = new_runtime(); history = [packet(i) for i in range(5)]
        start(runtime, history)
        previous = runtime.components._cpu_backend.scenes[0]['coverage_v21']
        runtime.close_sensor_episode(0, reason='analytic_prefix_at_anchor')
        runtime.reset_scene(0)
        with self.assertRaisesRegex(ValueError, 'fresh sensor episode'):
            start(runtime, history)
        start(runtime, [packet(0, episode='fresh'), packet(1, episode='fresh')])
        current = runtime.components._cpu_backend.scenes[0]['coverage_v21']
        self.assertIsNot(current, previous)
        previous.belief[:] = 1
        self.assertFalse(np.array_equal(previous.belief, current.belief))


if __name__ == '__main__':
    unittest.main()
