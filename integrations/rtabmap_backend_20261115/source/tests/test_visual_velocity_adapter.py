"""Policy tests use synthetic clocks/messages only; no ROS or CAN connection."""
import importlib.util
import copy
import json
import math
from pathlib import Path
import socket
import struct
import unittest
from unittest.mock import patch

MODULE = Path(__file__).resolve().parents[1] / 'runtime' / 'visual_velocity_adapter.py'
spec = importlib.util.spec_from_file_location('visual_velocity_adapter', MODULE)
adapter = importlib.util.module_from_spec(spec)
with patch.object(socket, 'socket', side_effect=AssertionError('import opened socket')):
    spec.loader.exec_module(adapter)


def hardware(gate, now, v=0, w=0):
    gate.hardware(0x211, bytes([0, 1, 0, 0, 0, 0, 0, 0]), now, now)
    gate.hardware(0x221, struct.pack('>hh', v, w) + bytes(4), now, now)
    gate.driver_status(0, now)


def observe(gate, kind, now, source=None, **kwargs):
    frames = dict(odom='odom', scan='base_link', depth_health='zed2i_left_camera_optical_frame', odom_health='odom')
    values = dict(caller='/nso_visual_sensor_bridge', frame=frames[kind])
    if kind == 'odom':
        values.update(x=0., y=0., yaw=0., v=0., w=0., child_frame='base_link')
    values.update(kwargs)
    gate.observe(kind, 1000. + (now if source is None else source), now, 1000. + now, **values)


def vision(gate, now, **odom):
    observe(gate, 'odom', now, **odom)
    for kind in ['odom_health', 'scan', 'depth_health']:
        observe(gate, kind, now)


def ready_gate(drift=.0):
    gate = adapter.Gate()
    gate.driver_connected = True
    gate.zero_published = 12
    for tick in range(5, 21):
        now = tick / 10.
        hardware(gate, now)
        vision(gate, now, x=drift * (now-.5))
    return gate


class VisualGateTests(unittest.TestCase):
    def arm(self, gate):
        self.assertEqual(gate.arm(2., 1002.), (True, 'armed_once'))

    def test_import_and_validation_have_no_device_side_effect(self):
        with patch.object(socket, 'socket', side_effect=AssertionError('validation opened socket')):
            cfg = adapter.configuration({})
            self.assertEqual(cfg['master_uri'], 'http://127.0.0.1:11382')
            adapter.Gate()
        with self.assertRaises(ValueError):
            adapter.configuration({'master_uri': 'http://127.0.0.1:11381'})

    def test_real_chassis_zero_allows_bounded_visual_drift(self):
        gate = ready_gate(drift=.015)
        self.arm(gate)
        self.assertGreater(gate.arm_stationarity_metrics['maximum_position_displacement_m'], .01)
        self.assertIn('raw chassis', gate.arm_stationarity_metrics['stationary_basis'])

    def test_moving_chassis_cannot_arm_despite_static_visual_odometry(self):
        gate = ready_gate()
        hardware(gate, 2.01, v=10)
        self.assertEqual(gate.arm(2.01, 1002.01)[1], 'need_one_second_actual_zero_feedback')

    def test_visual_jump_still_latches_before_arm(self):
        gate = ready_gate()
        observe(gate, 'odom', 2.1, x=.2)
        self.assertEqual(gate.fault, 'odometry_jump')
        self.assertFalse(gate.arm(2.1, 1002.1)[0])

    def test_source_health_pairs_commit_atomically(self):
        gate = ready_gate()
        previous = gate.odom
        observe(gate, 'odom', 2.1, v=.07, w=.13)
        self.assertIs(gate.odom, previous)
        self.assertIsNone(gate.sources_reason(2.1, 1002.1))
        observe(gate, 'odom_health', 2.1)
        self.assertEqual((gate.odom['v'], gate.odom['w']), (.07, .13))
        self.assertEqual(gate.odom['stamp'], gate.odom_health['stamp'])

    def test_duplicate_and_fresh_receipt_do_not_renew_original_time(self):
        gate = ready_gate()
        original = gate.odom['receipt']
        observe(gate, 'odom', 2.3, source=2.)
        observe(gate, 'odom_health', 2.3, source=2.)
        self.assertEqual(gate.odom['receipt'], original)
        observe(gate, 'odom', 3., source=2.1)
        self.assertEqual(gate.odom['receipt'], original)

    def test_future_regressed_wrong_frame_and_wrong_publisher_latch(self):
        cases = [dict(kind='odom', now=2.1, source=2.2),
                 dict(kind='odom', now=2.1, source=1.9),
                 dict(kind='depth_health', now=2.1, frame='base_link'),
                 dict(kind='scan', now=2.1, caller='/other_bridge')]
        for case in cases:
            with self.subTest(case=case):
                gate = ready_gate()
                observe(gate, **case)
                self.assertIsNotNone(gate.fault)
                self.assertFalse(gate.arm(2.1, 1002.1)[0])

    def test_only_fresh_post_arm_proposals_can_move(self):
        gate = ready_gate()
        gate.propose([.1, 0, 0, 0, 0, .2], 1.99, '/nso_visual/move_base')
        self.arm(gate)
        self.assertEqual(gate.decision(2., 1002.)[:2], (0., 0.))
        gate.propose([.1, 0, 0, 0, 0, .2], 2.01, '/nso_visual/move_base')
        self.assertEqual(gate.decision(2.02, 1002.02), (.1, .2, 'current_proposal'))

    def test_trusted_planar_proposals_saturate_at_unchanged_physical_limits(self):
        cases=[(.10008002072572708,.09052,.1,.09052),
               (.10000000149011612,-.30000001192092896,.1,-.3),
               (.1,.30001,.1,.3),(.05,-.30001,.05,-.3),
               (1000.,-1000.,.1,-.3),(0.,.2,0.,.2),
               (-.0015950807337154864,0.,0.,0.),(-.0016,0.,0.,0.),
               (-1.,.2,0.,.2),(-1000.,-.8,0.,-.3)]
        for v,w,expected_v,expected_w in cases:
            with self.subTest(v=v,w=w):
                gate=ready_gate();self.arm(gate)
                values=[v,0,0,0,0,w]
                gate.propose(values,2.01,'/nso_visual/move_base')
                self.assertEqual(gate.decision(2.02,1002.02),
                                 (expected_v,expected_w,'current_proposal'))
                record=adapter.velocity_proposal_record(gate,values,2.01,'/nso_visual/move_base')
                self.assertEqual(record['raw_values'],values)
                self.assertEqual(record['bounded_planar_command'],dict(v=expected_v,w=expected_w))
                self.assertEqual(record['saturated'],(v,w)!=(expected_v,expected_w))
                self.assertGreaterEqual(gate.proposal['v'],0.)
                self.assertIsNone(gate.fault)

    def test_nonplanar_and_nonfinite_proposals_still_latch(self):
        cases=[]
        for index in (1,2,3,4):
            values=[.1,0,0,0,0,0];values[index]=.001;cases.append(values)
        for index in range(6):
            for value in (math.nan,math.inf,-math.inf):
                values=[.1,0,0,0,0,0];values[index]=value;cases.append(values)
        for values in cases:
            gate = ready_gate()
            self.arm(gate)
            gate.propose(values, 2.01, '/nso_visual/move_base')
            self.assertIsNotNone(gate.fault)
            self.assertEqual(gate.decision(2.02,1002.02)[:2],(0.,0.))

    def test_saturated_command_cannot_bypass_actual_CAN_fault_or_stop(self):
        for case in ('linear_overspeed','angular_overspeed','chassis_error','explicit_stop'):
            with self.subTest(case=case):
                gate=ready_gate();self.arm(gate)
                gate.propose([.10008,0,0,0,0,.30001],2.01,'/nso_visual/move_base')
                if case=='linear_overspeed':hardware(gate,2.015,v=151)
                elif case=='angular_overspeed':hardware(gate,2.015,w=451)
                elif case=='chassis_error':
                    gate.hardware(0x211,bytes([0,1,0,0,0,1,0,0]),2.015,2.015)
                else:gate.fail('explicit_stop')
                self.assertEqual(gate.decision(2.02,1002.02)[:2],(0.,0.))
                expected=('chassis_speed_feedback_limit' if case.endswith('overspeed') else
                          'chassis_state_mode_or_fault' if case=='chassis_error' else 'explicit_stop')
                self.assertEqual(gate.fault,expected)

    def test_unexpected_proposal_publisher_stops_and_cannot_rearm(self):
        gate = ready_gate()
        self.arm(gate)
        gate.propose([.1, 0, 0, 0, 0, 0], 2.01, '/nso_standard/move_base')
        self.assertTrue(gate.fault.startswith('unexpected_proposal_publisher'))
        self.assertEqual(gate.decision(2.02, 1002.02)[:2], (0., 0.))
        self.assertFalse(gate.arm(2.02, 1002.02)[0])

    def test_proposal_expiry_uses_original_receipt(self):
        gate = ready_gate()
        self.arm(gate)
        hardware(gate, 2.1)
        hardware(gate, 2.2)
        gate.propose([.1, 0, 0, 0, 0, 0], 2.01, '/nso_visual/move_base')
        self.assertEqual(gate.decision(2.22, 1002.22), (0., 0., 'proposal_expired'))

    def test_source_loss_zeroes_then_needs_new_proposal_after_recovery(self):
        gate = ready_gate()
        self.arm(gate)
        for tick in range(21, 27):
            hardware(gate, tick/10.)
        self.assertEqual(gate.decision(2.6, 1002.6)[:2], (0., 0.))
        self.assertIsNone(gate.fault)
        hardware(gate, 2.64)
        vision(gate, 2.64)
        self.assertEqual(gate.decision(2.64, 1002.64)[2], 'await_new_proposal')
        gate.propose([.05, 0, 0, 0, 0, -.1], 2.65, '/nso_visual/move_base')
        self.assertEqual(gate.decision(2.66, 1002.66)[:2], (.05, -.1))

    def test_depth_health_loss_does_not_get_hidden_by_fresh_other_sources(self):
        gate = ready_gate()
        self.arm(gate)
        for tick in range(21, 27):
            now = tick/10.
            hardware(gate, now)
            observe(gate, 'odom', now)
            observe(gate, 'odom_health', now)
            observe(gate, 'scan', now)
        self.assertEqual(gate.decision(2.6, 1002.6)[:2], (0., 0.))
        self.assertEqual(gate.scan['stamp'], 1002.)

    def test_persistent_visual_loss_latches_and_does_not_rearm(self):
        gate = ready_gate()
        self.arm(gate)
        for tick in range(21, 27):
            hardware(gate, tick/10.)
        gate.decision(2.6, 1002.6)
        for tick in range(1, 53):
            now = 2.6 + tick*.02
            hardware(gate, now)
            gate.decision(now, 1000.+now)
        self.assertTrue(gate.fault.startswith('source_loss:'))
        vision(gate, 3.65)
        self.assertFalse(gate.arm(3.65, 1003.65)[0])

    def test_hardware_fault_latches_zero(self):
        gate = ready_gate()
        self.arm(gate)
        gate.hardware(0x211, bytes([0, 2, 0, 0, 0, 0, 0, 0]), 2.01, 2.01)
        self.assertEqual(gate.decision(2.02, 1002.02)[:2], (0., 0.))
        self.assertEqual(gate.fault, 'chassis_state_mode_or_fault')

    def test_duration_is_full_600_seconds_without_distance_or_angle_cap(self):
        gate = ready_gate()
        self.arm(gate)
        gate.trajectory_distance = 100.
        gate.trajectory_absolute_yaw = 100.
        self.assertEqual(gate.duration, 600.)
        gate.propose([.1, 0, 0, 0, 0, .3], 2.01, '/nso_visual/move_base')
        self.assertEqual(gate.decision(2.02, 1002.02)[:2], (.1, .3))
        deadline_gate = ready_gate()
        self.arm(deadline_gate)
        self.assertEqual(deadline_gate.decision(602., 1602.), (0., 0., 'duration_complete'))

    def test_depth_scan_unknown_is_allowed_but_infinite_clearing_is_rejected(self):
        valid = adapter.visual_scan_geometry_valid
        self.assertTrue(valid(-.1, .1, .1, .1, 3., [math.nan, math.nan, math.nan]))
        self.assertTrue(valid(-.1, .1, .1, .1, 3., [math.nan, 1., 2.]))
        self.assertFalse(valid(-.1, .1, .1, .1, 3., [math.inf, 1., 2.]))
        self.assertFalse(valid(-.1, .1, .1, .1, 3., [.01, 1., 2.]))

    def test_actuator_snapshot_keeps_original_CAN_clocks_and_no_mutation(self):
        gate=ready_gate()
        gate.session_id='visual_test_session'
        self.arm(gate)
        original=copy.deepcopy(gate.__dict__)
        state=adapter.snapshot(gate,2.05,1002.05)
        self.assertEqual(gate.__dict__,original)
        self.assertEqual(state['session_id'],'visual_test_session')
        self.assertEqual(state['sampled_monotonic'],2.05)
        self.assertEqual(state['sampled_unix_s'],1002.05)
        self.assertEqual(state['can_status']['source'],2.)
        self.assertEqual(state['can_velocity']['receipt'],2.)
        self.assertEqual(state['can_zero_since_monotonic'],.5)
        self.assertTrue(state['ever_armed'])
        self.assertTrue(state['zero_feedback_confirmed'])
        self.assertIsNone(state['hardware_reason'])
        self.assertEqual(state['arm_stationarity_metrics'],gate.arm_stationarity_metrics)
        json.dumps(state,allow_nan=False)

    def test_status_heartbeat_does_not_make_stale_chassis_fresh(self):
        gate=ready_gate()
        fresh=adapter.snapshot(gate,2.05,1002.05)
        stale=adapter.snapshot(gate,2.2,1002.2)
        self.assertEqual(fresh['can_velocity'],stale['can_velocity'])
        self.assertEqual(fresh['can_status'],stale['can_status'])
        self.assertEqual(stale['hardware_reason'],'chassis_velocity_stale')
        self.assertFalse(stale['zero_feedback_confirmed'])
        self.assertEqual(gate.can_velocity['source'],2.)

    def test_snapshot_nested_rows_are_detached_and_fault_is_preserved(self):
        gate=ready_gate()
        self.arm(gate)
        gate.fail('explicit_stop')
        state=adapter.snapshot(gate,2.05)
        state['can_status']['mode']=99
        state['arm_stationarity_metrics']['count']=999
        self.assertEqual(gate.can_status['mode'],1)
        self.assertNotEqual(gate.arm_stationarity_metrics['count'],999)
        self.assertEqual(state['fault'],'explicit_stop')
        self.assertFalse(state['armed'])
        self.assertTrue(state['ever_armed'])
        self.assertIsNone(state['sampled_unix_s'])

    def test_empty_snapshot_reports_missing_CAN_and_invalid_sample_clock_rejected(self):
        gate=adapter.Gate()
        state=adapter.snapshot(gate,1.,1001.)
        self.assertIsNone(state['can_status']);self.assertIsNone(state['can_velocity'])
        self.assertEqual(state['hardware_reason'],'chassis_status_stale')
        self.assertFalse(state['zero_feedback_confirmed'])
        self.assertIsNone(state['can_zero_since_monotonic'])
        with self.assertRaises(ValueError):adapter.snapshot(gate,math.nan)


if __name__ == '__main__':
    unittest.main()
