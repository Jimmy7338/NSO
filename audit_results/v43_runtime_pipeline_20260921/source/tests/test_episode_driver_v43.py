"""Finite packet doubles validate driver order; no live study environment."""
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from env.development_sensor_v41 import SensorStepV41
from nso.analytic_fixture_v42 import analytic_forward_sequence_v42
from nso.episode_driver_v43 import (BoundedRunWriterV43, DevelopmentStartLedgerV43,
    OutputLimitExceededV43, execute_episode_v43, validate_step_v43)
from utils.rgbd_contract import PlanarScan


def packet_steps():
    result = []
    for index, obs in enumerate(analytic_forward_sequence_v42()):
        laser = np.eye(4); laser[:3, 3] = [obs.world_from_camera[0, 3], .75, .3]
        scan = PlanarScan(float(index), np.zeros(4), -np.pi, np.pi/2, 8., laser)
        receipt = dict(action='initial_observation' if index == 0 else 'forward',
            paid_step=index, action_cost=0 if index == 0 else 1, collision=False,
            pose_xyyaw_rad=[float(obs.world_from_camera[0, 3]), .75, 0.])
        if index == 0:
            receipt['initial_frames'] = 1
        else:
            receipt['pose_before_xyyaw_rad'] = [.75,.75,0.]
        result.append(SensorStepV41(obs, scan, receipt))
    return result


class FixedPackets:
    def __init__(self, events, packets=None, returned=False):
        self.events = events; self.packets = packet_steps() if packets is None else packets
        self.calls = []; self.returned = returned

    def initial_observation(self):
        self.events.append('sensor:0'); self.calls.append('initial')
        return self.packets[0]

    def step(self, action):
        self.events.append('sensor:1'); self.calls.append(action)
        return self.packets[1]

    def close(self):
        return dict(returned_xy_and_yaw=self.returned, paid_actions=len(self.calls)-1, closed=True)


class RecordingMapper:
    def __init__(self, events):
        self.events = events; self.frames = []

    def update(self, observation, scan):
        self.events.append('map:'+str(observation.paid_step)); self.frames.append(observation)
        return dict(frame_id=observation.frame_id, observation_sha256=observation.sha256())

    def mesh_arrays(self):
        self.events.append('mesh')
        return dict(vertices=np.zeros((0, 3)), triangles=np.zeros((0, 3), np.int32))

    def occupancy_arrays(self):
        return np.full((2, 2), -1, np.int8), np.zeros((2, 2), bool)

    def snapshot(self):
        return dict(frames=len(self.frames), tsdf_integration_count=len(self.frames))


class RecordingController:
    def __init__(self, events, decisions=('forward', 'blocked')):
        self.events=events; self.decisions=list(decisions); self.accepted=[]

    def accept(self, observation, mapper, execution_outcome):
        assert len(mapper.frames) == observation.paid_step+1
        self.events.append('accept:'+str(observation.paid_step))
        self.accepted.append((observation.paid_step, execution_outcome))
        return dict(observation_sha256=observation.sha256())

    def choose(self):
        self.events.append('choose:'+str(len(self.accepted)-1))
        return dict(action=self.decisions.pop(0), reason='finite test double, not a planning policy')


class EpisodeDriverV43Tests(unittest.TestCase):
    def test_driver_saves_then_fuses_once_updates_current_evidence_then_selects(self):
        events=[]; sensor=FixedPackets(events); mapper=RecordingMapper(events); controller=RecordingController(events)
        with tempfile.TemporaryDirectory() as directory:
            writer=BoundedRunWriterV43(directory)
            result=execute_episode_v43(sensor, controller, mapper, writer, budget=1)
            self.assertEqual(events[:8], ['sensor:0','map:0','accept:0','choose:0',
                                         'sensor:1','map:1','accept:1','choose:1'])
            self.assertEqual(result['mapper_frames'], 2)
            self.assertEqual(result['executed_paid_actions'], 1)
            self.assertEqual(result['status'], 'controller_blocked')
            self.assertEqual(sensor.calls, ['initial','forward'])
            self.assertEqual(len(list((Path(directory)/'packets').glob('*_rgbd.npz'))), 2)
            saved=json.loads((Path(directory)/'packets/001_receipt.json').read_text())
            self.assertEqual(saved['observation_sha256'], mapper.frames[-1].sha256())
            self.assertFalse(result['actual_performance_evaluated'])

    def test_invalid_execution_receipt_preserved_and_rejected_before_fusion(self):
        packets=packet_steps(); receipt=dict(packets[0].receipt, action_cost=1)
        packets[0]=SensorStepV41(packets[0].rgbd, packets[0].scan, receipt)
        events=[]; mapper=RecordingMapper(events)
        with tempfile.TemporaryDirectory() as directory:
            result=execute_episode_v43(FixedPackets(events,packets),RecordingController(events),mapper,
                                      BoundedRunWriterV43(directory),budget=1)
            self.assertEqual(result['status'],'episode_error'); self.assertEqual(mapper.frames,[])
            self.assertEqual(result['received_sensor_packets'],1)
            self.assertTrue((Path(directory)/'packets/000_rgbd.npz').exists())
            self.assertEqual(result['error']['stage'],'packet_accounting')

    def test_scan_time_and_extrinsics_pairing_checked_before_mapper(self):
        step=packet_steps()[0]
        for change in ('timestamp','translation','yaw'):
            fields={key:deepcopy(getattr(step.scan,key)) for key in PlanarScan.__dataclass_fields__}
            if change=='timestamp': fields['timestamp_s']=1.
            if change=='translation': fields['world_from_laser'][0,3] += .25
            if change=='yaw': fields['world_from_laser'][:2,:2]=[[0,-1],[1,0]]
            with self.subTest(change=change), self.assertRaises(ValueError):
                validate_step_v43(SensorStepV41(step.rgbd,PlanarScan(**fields),step.receipt),
                                  expected_step=0,expected_action='initial_observation')

    def test_stop_requires_confirmed_xy_and_yaw_return(self):
        for returned, expected in ((False,'stopped_without_confirmed_return'),(True,'controller_stop')):
            events=[]
            with tempfile.TemporaryDirectory() as directory:
                result=execute_episode_v43(FixedPackets(events,returned=returned),
                    RecordingController(events,('stop',)),RecordingMapper(events),BoundedRunWriterV43(directory),budget=1)
                self.assertEqual(result['status'],expected)
                self.assertEqual(result['executed_paid_actions'],0)

    def test_primitive_and_camera_height_checked_before_fusion(self):
        first, second = packet_steps()
        bad = dict(second.receipt, pose_before_xyyaw_rad=[.5,.75,0.])
        with self.assertRaises(ValueError):
            validate_step_v43(SensorStepV41(second.rgbd,second.scan,bad),expected_step=1,
                              expected_action='forward',expected_previous_pose=[.75,.75,0.])
        wrong_height=first.rgbd.world_from_camera.copy(); wrong_height[2,3]=1.1
        from nso.instance_belief_v40 import PaidRGBDObservationV40
        packet=PaidRGBDObservationV40(first.rgbd.frame_id,0,first.rgbd.rgb,first.rgbd.depth_m,
                                      first.rgbd.intrinsic,wrong_height)
        with self.assertRaises(ValueError):
            validate_step_v43(SensorStepV41(packet,first.scan,first.receipt),expected_step=0,
                              expected_action='initial_observation')

    def test_failed_submitted_action_is_counted_and_not_retried(self):
        events=[]
        class BrokenSensor(FixedPackets):
            def step(self, action):
                self.calls.append(action)
                raise RuntimeError('capture failed after action submission')
        sensor=BrokenSensor(events)
        with tempfile.TemporaryDirectory() as directory:
            result=execute_episode_v43(sensor,RecordingController(events),RecordingMapper(events),
                                      BoundedRunWriterV43(directory),budget=1)
            self.assertEqual(result['submitted_paid_actions'],1)
            self.assertEqual(result['executed_paid_actions'],0)
            self.assertFalse(result['automatic_retry'])
            self.assertEqual(sensor.calls,['initial','forward'])

    def test_collision_attempt_is_paid_and_pose_stays_at_verified_origin(self):
        from nso.instance_belief_v40 import PaidRGBDObservationV40
        packets=packet_steps(); first=packets[0]
        obs=PaidRGBDObservationV40('paid-collision-frame',1,first.rgbd.rgb,first.rgbd.depth_m,
                                  first.rgbd.intrinsic,first.rgbd.world_from_camera)
        scan=PlanarScan(1.,first.scan.ranges_m,first.scan.angle_min_rad,
                        first.scan.angle_increment_rad,first.scan.range_max_m,first.scan.world_from_laser)
        receipt=dict(action='forward',paid_step=1,action_cost=1,collision=True,
                     pose_before_xyyaw_rad=[.75,.75,0.],pose_xyyaw_rad=[.75,.75,0.])
        packets[1]=SensorStepV41(obs,scan,receipt)
        events=[]; controller=RecordingController(events)
        with tempfile.TemporaryDirectory() as directory:
            result=execute_episode_v43(FixedPackets(events,packets),controller,RecordingMapper(events),
                                      BoundedRunWriterV43(directory),budget=1)
            self.assertEqual(result['executed_paid_actions'],1)
            self.assertEqual(result['collisions'],1)
            self.assertEqual(controller.accepted[-1],(1,'collision'))

    def test_terminal_record_survives_ordinary_output_cap(self):
        with tempfile.TemporaryDirectory() as directory:
            writer=BoundedRunWriterV43(directory,maximum_bytes=1024,maximum_file_bytes=1024,terminal_reserve_bytes=256)
            writer._write('filled.bin',b'x'*768)
            with self.assertRaises(OutputLimitExceededV43): writer.json('overflow.json',{})
            writer.json('result.json',dict(status='artifact_limit'),terminal=True)
            self.assertLessEqual(writer.bytes_written,1024)

    def test_artifacts_cannot_overwrite_or_escape_output(self):
        with tempfile.TemporaryDirectory() as directory:
            writer=BoundedRunWriterV43(directory); writer.json('same.json',{})
            with self.assertRaises(FileExistsError): writer.json('same.json',{})
            with self.assertRaises(ValueError): writer.json('../escape.json',{})

    def test_start_ledger_records_failure_and_refuses_retry_or_sixth_start(self):
        with tempfile.TemporaryDirectory() as directory:
            ledger=DevelopmentStartLedgerV43(Path(directory)/'ledger.json')
            digest=hashlib.sha256(b'retained result').hexdigest()
            for index in range(5):
                ledger.reserve('run'+str(index),dict(scope='test-double reservations only'))
                ledger.finish('run'+str(index),status='failed_fixture',world_created=False,result_sha256=digest)
            with self.assertRaises(ValueError): ledger.reserve('sixth',{})
            with self.assertRaises(ValueError): ledger.reserve('run0',{})
            data=json.loads((Path(directory)/'ledger.json').read_text())
            self.assertEqual(len(data['entries']),5)
            self.assertTrue(all(row['status']=='failed_fixture' for row in data['entries']))

    def test_diagnostic_sequence_requires_complete_position_and_heading_return(self):
        from nso.diagnostic_policy_v43 import DiagnosticPolicyV43
        from nso.primitive_navigation_v41 import PublicPrimitiveGraphV41, PrimitiveStateV41
        graph=PublicPrimitiveGraphV41(dict(schema_version='v41.public_navigation.v1',
            source_kind='provided_navigation_prior',nodes={'home':[.75,.75],'next':[1.,.75]},edges=[['home','next']]))
        home=PrimitiveStateV41('home',0)
        actions=['forward']+['left']*6+['forward']+['left']*6
        DiagnosticPolicyV43(graph,home=home,budget=14,actions=actions)
        with self.assertRaises(ValueError): DiagnosticPolicyV43(graph,home=home,budget=14,actions=actions[:-1])
        with self.assertRaises(ValueError): DiagnosticPolicyV43(graph,home=home,budget=13,actions=actions)

    def test_diagnostic_accepts_only_the_current_measured_mapper(self):
        from nso.diagnostic_policy_v43 import DiagnosticPolicyV43
        from nso.observed_mapper_v42 import ObservedMapperV42
        from nso.primitive_navigation_v41 import PublicPrimitiveGraphV41, PrimitiveStateV41
        graph=PublicPrimitiveGraphV41(dict(schema_version='v41.public_navigation.v1',
            source_kind='provided_navigation_prior',nodes={'home':[.75,.75],'next':[1.,.75]},edges=[['home','next']]))
        policy=DiagnosticPolicyV43(graph,home=PrimitiveStateV41('home',0),budget=14,
            actions=['forward']+['left']*6+['forward']+['left']*6)
        mapper=ObservedMapperV42(shape=(30,30),origin_xy_m=(-.5,-.5))
        step=packet_steps()[0]
        with self.assertRaises(ValueError): policy.accept(step.rgbd,mapper)
        mapper.update(step.rgbd,step.scan); policy.accept(step.rgbd,mapper)
        self.assertEqual(policy.choose()['action'],'forward')
        with self.assertRaises(ValueError): policy.choose()

    def test_real_entrypoint_resource_rejection_precedes_assets_or_reservation(self):
        import importlib.util
        from env.development_sensor_v41 import storage_report_v41,runtime_counts_v41
        root=Path(__file__).resolve().parents[1]
        spec=importlib.util.spec_from_file_location('development_entrypoint_v43',root/'scripts/run_development_v43.py')
        module=importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
        with tempfile.TemporaryDirectory() as directory:
            if storage_report_v41(directory,384*1024**2)['passed']:
                self.skipTest('resource-negative probe only; never launch a real World from this test')
            before=runtime_counts_v41()
            result=module.run_development_v43('R3_A_G',output_root=Path(directory)/'unused',preflight_only=False)
            self.assertEqual(result['status'],'blocked_before_world_creation')
            self.assertFalse(result['start_slot_reserved']); self.assertFalse(result['asset_read'])
            self.assertFalse((Path(directory)/'unused').exists())
            self.assertEqual(runtime_counts_v41(),before)


if __name__=='__main__': unittest.main()
