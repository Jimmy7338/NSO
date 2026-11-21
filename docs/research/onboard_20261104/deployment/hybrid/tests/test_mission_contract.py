"""Pure supervisor/evidence tests: importing mission starts no ROS or devices."""
import copy
import hashlib
import importlib.util
import json
import math
import sys
from pathlib import Path
from types import SimpleNamespace as NS
import unittest
import tempfile
from contextlib import redirect_stdout,redirect_stderr
from io import StringIO
from unittest.mock import patch

RUNTIME=Path(__file__).resolve().parents[1]/'runtime'
sys.path.insert(0,str(RUNTIME))
spec=importlib.util.spec_from_file_location('mission_contract',RUNTIME/'mission.py')
mission=importlib.util.module_from_spec(spec);spec.loader.exec_module(mission)


def transform(stamp=100.,x=0.,yaw=0.):
    return NS(header=NS(frame_id='map',stamp=NS(to_sec=lambda:stamp)),child_frame_id='base_link',
              transform=NS(translation=NS(x=x,y=0.,z=0.),
                           rotation=NS(x=0.,y=0.,z=math.sin(yaw/2),w=math.cos(yaw/2))))


class MissionContractTests(unittest.TestCase):
    def test_manual_footprint_diagnostic_reports_unknown_and_lethal(self):
        import numpy as np
        grid=NS(header=NS(frame_id='map',stamp=NS(to_sec=lambda:100.)),
            info=NS(height=80,width=80,resolution=.05,origin=NS(position=NS(x=-2.,y=-2.))),
            data=np.zeros((80,80),np.int16).ravel().tolist())
        self.assertTrue(mission.navigation_start_footprint(grid,transform())['clear'])
        for value,field in [(-1,'unknown_cells'),(100,'lethal_cells')]:
            altered=copy.deepcopy(grid);altered.data[46*80+32]=value
            result=mission.navigation_start_footprint(altered,transform())
            self.assertFalse(result['clear']);self.assertEqual(result[field],1)

    def test_manual_footprint_diagnostic_reports_bounds_and_frame(self):
        import numpy as np
        grid=NS(header=NS(frame_id='map',stamp=NS(to_sec=lambda:100.)),
            info=NS(height=80,width=80,resolution=.05,origin=NS(position=NS(x=-2.,y=-2.))),
            data=np.zeros((80,80),np.int16).ravel().tolist())
        self.assertFalse(mission.navigation_start_footprint(grid,transform(x=1.9))['clear'])
        bad=transform();bad.header.frame_id='odom'
        with self.assertRaises(ValueError):mission.navigation_start_footprint(grid,bad)

    def evidence(self):
        goal=dict(id='g1',x=.5,y=0.,yaw=0.,frame='map')
        terminal=dict(id='g1',status=3)
        before=mission.pose_evidence(transform(100.,0.),100.01,1.)
        after=mission.pose_evidence(transform(105.,.5),105.01,6.)
        start=dict(distance_m=0.,absolute_yaw_rad=0.,accounting_complete=True,feedback_fresh=True)
        end=dict(start,distance_m=.5,absolute_yaw_rad=.1)
        return [goal,terminal,before,after,start,end]

    def test_verified_arrival_needs_target_pose_and_fresh_CAN(self):
        row=mission.arrival_evidence(*self.evidence())
        self.assertTrue(row['verified']);self.assertEqual(row['CAN_distance_delta_m'],.5)
        for kind in ['wrong_target','same_TF','stale_TF','incomplete_CAN','wrong_ID','preempted']:
            a=self.evidence()
            if kind=='wrong_target':a[0]['x']=1.
            if kind=='same_TF':a[3]['source_stamp_s']=a[2]['source_stamp_s']
            if kind=='stale_TF':a[3]['source_age_s']=.6
            if kind=='incomplete_CAN':a[5]['accounting_complete']=False
            if kind=='wrong_ID':a[1]['id']='g2'
            if kind=='preempted':a[1]['status']=2
            with self.subTest(kind=kind):self.assertFalse(mission.arrival_evidence(*a)['verified'])

    def test_execution_frames_must_match_goal_and_both_poses(self):
        rows=self.evidence()
        rows[0]['frame']='odom'
        for row in rows[2:4]:row['target_frame']='odom'
        self.assertTrue(mission.arrival_evidence(*rows,expected_frame='odom')['verified'])
        self.assertFalse(mission.arrival_evidence(*rows)['verified'])
        for index in (2,3):
            bad=copy.deepcopy(rows);bad[index]['target_frame']='map'
            self.assertEqual(mission.arrival_evidence(*bad,expected_frame='odom')['reason'],'pose_frame_different')
            del bad[index]['target_frame']
            self.assertFalse(mission.arrival_evidence(*bad,expected_frame='odom')['verified'])

    def test_native_arrival_in_odom_is_not_checked_against_legacy_map(self):
        rows=self.evidence()
        rows[0]['frame']='odom'
        for row in rows[2:4]:row['target_frame']='odom'
        self.assertEqual(mission.arrival_evidence(*rows)['reason'],'target_frame_not_map')
        frame=mission.arrival_recording_frame('visual','map',rows[0]['frame'])
        self.assertTrue(mission.arrival_evidence(*rows,expected_frame=frame)['verified'])
        rows[3]['target_frame']='map'
        self.assertEqual(mission.arrival_evidence(*rows,expected_frame=frame)['reason'],'pose_frame_different')

    def test_native_frame_selection_preserves_declared_NSO_contract(self):
        for mode in ('nso','joint','reference','navigation'):
            rows=self.evidence()
            rows[0]['frame']='odom'
            for row in rows[2:4]:row['target_frame']='odom'
            frame=mission.arrival_recording_frame(mode,'map',rows[0]['frame'])
            self.assertEqual(mission.arrival_evidence(*rows,expected_frame=frame)['reason'],'target_frame_not_map')
        for frame in ('map','odom'):
            self.assertEqual(mission.arrival_recording_frame('joint',frame,'other'),frame)
        rows=self.evidence();rows[0]['frame']='unsupported'
        frame=mission.arrival_recording_frame('visual','map',rows[0]['frame'])
        self.assertEqual(mission.arrival_evidence(*rows,expected_frame=frame)['reason'],'unsupported_execution_frame')

    def test_in_place_rotation_is_a_valid_goal_without_forced_translation(self):
        rows=self.evidence();rows[0].update(x=0.,yaw=.6)
        rows[3]=mission.pose_evidence(transform(105.,0.,.6),105.01,6.)
        rows[5].update(distance_m=0.,absolute_yaw_rad=.6)
        result=mission.arrival_evidence(*rows)
        self.assertTrue(result['verified'])
        self.assertEqual(result['CAN_distance_delta_m'],0.)
        self.assertEqual(result['displacement_m'],0.)
        self.assertAlmostEqual(result['CAN_absolute_yaw_delta_rad'],.6)
        rows[0]['yaw']=1.5
        self.assertEqual(mission.arrival_evidence(*rows)['reason'],'target_pose_error')

    def test_yaw_wrap_is_compared_at_goal(self):
        a=self.evidence();a[0]['yaw']=math.pi-.02;a[3]['pose'][2]=-math.pi+.02
        self.assertTrue(mission.arrival_evidence(*a)['verified'])
        a[3]['pose'][2]=0.
        self.assertEqual(mission.arrival_evidence(*a)['reason'],'target_pose_error')

    def test_TF_must_be_finite_fresh_and_in_expected_frames(self):
        for stamp in [0.,99.,101.]:
            with self.assertRaises(ValueError):mission.pose_evidence(transform(stamp),100.,1.)
        with self.assertRaises(ValueError):mission.pose_evidence(transform(x=math.nan),100.,1.)
        bad=transform();bad.child_frame_id='camera'
        with self.assertRaises(ValueError):mission.pose_evidence(bad,100.,1.)

    def test_transient_health_recovers_and_reason_changes_do_not_reset_grace(self):
        h=mission.HealthWindow()
        self.assertEqual(h.update('fit_warming',1.),'waiting')
        self.assertEqual(h.update(None,1.4),'ready')
        self.assertEqual(h.update('depth_stale',2.),'waiting')
        self.assertEqual(h.update('fit_warming',2.6),'waiting')
        self.assertEqual(h.update('odom_stale',3.01),'expired')

    def test_prearm_needs_continuous_one_second_health(self):
        h=mission.HealthWindow()
        self.assertEqual(h.update(None,0.,True),'waiting')
        self.assertEqual(h.update('not_ready',.7,True),'waiting')
        self.assertEqual(h.update(None,.8,True),'waiting')
        self.assertEqual(h.update(None,1.79,True),'waiting')
        self.assertEqual(h.update(None,1.81,True),'ready')

    def test_health_reports_specific_reason_and_checks_original_stamps(self):
        health=(1.,dict(odom=dict(valid=True,source_stamp_ns=100000000000),
                        depth=dict(valid=True,source_stamp_ns=100000000000)))
        odom=(1.,NS(header=NS(stamp=NS(to_sec=lambda:100.))))
        grid=(1.,object())
        self.assertIsNone(mission.visual_health_reason(health,odom,grid,1.,100.))
        bad=copy.deepcopy(health);bad[1]['odom'].update(valid=False,reason='velocity_fit_warming')
        self.assertEqual(mission.visual_health_reason(bad,odom,grid,1.,100.),'odom:velocity_fit_warming')
        bad=copy.deepcopy(health);bad[1]['odom']['fault']='visual_pose_jump'
        with self.assertRaises(RuntimeError):mission.visual_health_reason(bad,odom,grid,1.,100.)
        self.assertEqual(mission.visual_health_reason(health,odom,grid,1.1,101.),'odom_source_stamp_stale_or_future')

    def test_hybrid_requires_actual_fresh_laser_and_declared_source(self):
        h=(1.,dict(navigation_source='lidar_rgbd',
            odom=dict(valid=True,source_stamp_ns=100000000000),
            depth=dict(valid=True,source_stamp_ns=100000000000),
            laser_source=dict(valid=True,source_stamp_ns=100000000000)))
        odom=(1.,NS(header=NS(stamp=NS(to_sec=lambda:100.))))
        self.assertIsNone(mission.hybrid_health_reason(h,odom,(1.,object()),1.,100.))
        for stamp in (99000000000,101000000000,True,float('nan')):
            bad=copy.deepcopy(h);bad[1]['laser_source']['source_stamp_ns']=stamp
            self.assertEqual(mission.hybrid_health_reason(bad,odom,(1.,object()),1.,100.),
                             'laser_source_stamp_stale_or_future')
        bad=copy.deepcopy(h);bad[1]['navigation_source']='visual'
        self.assertEqual(mission.hybrid_health_reason(bad,odom,(1.,object()),1.,100.),
                         'navigation_source_not_lidar_rgbd')
        bad=copy.deepcopy(h);bad[1]['laser_source']['fault']='laser_lost'
        with self.assertRaises(RuntimeError):mission.hybrid_health_reason(bad,odom,(1.,object()),1.,100.)

    def test_experiment_is_validated_and_exact_bytes_are_frozen(self):
        config=dict(schema='nso_physical_experiment/1',experiment_id='synthetic-contract-test',
            navigation_source='lidar_rgbd')
        with tempfile.TemporaryDirectory() as folder:
            p=Path(folder)/'experiment.json';raw=(json.dumps(config,indent=3)+'\n').encode();p.write_bytes(raw)
            decoded,frozen,sha=mission.load_experiment_config(p)
            self.assertEqual(decoded,config);self.assertEqual(frozen,raw)
            self.assertEqual(sha,hashlib.sha256(raw).hexdigest())
            config['experiment_id']='';p.write_text(json.dumps(config))
            with self.assertRaises(ValueError):mission.load_experiment_config(p)
            p.write_text('{"x": NaN}')
            with self.assertRaises(ValueError):mission.load_experiment_config(p)

    def test_removed_setup_and_fixed_arrival_options_cannot_launch(self):
        for argv in (['--mode','prepare'],['--prepare-goal-odom','0','0','0'],
                     ['--pilot-arrivals','3']):
            with self.subTest(argv=argv),redirect_stderr(StringIO()),self.assertRaises(SystemExit):
                mission.argument_parser().parse_args(argv)

    def test_preview_never_starts_children_or_writes_trial_data(self):
        out=StringIO()
        with patch.object(mission.subprocess,'Popen',side_effect=AssertionError('unexpected child')):
            with patch.object(mission,'save',side_effect=AssertionError('unexpected write')),redirect_stdout(out):
                mission.main(['--mode','visual'])
        self.assertEqual(json.loads(out.getvalue())['status'],'validated_no_motion')

    def test_nso_execution_fails_before_config_IO_data_stack_or_chassis_start(self):
        with patch.object(mission.subprocess,'run',side_effect=AssertionError('data stack started')):
            with patch.object(mission.subprocess,'Popen',side_effect=AssertionError('child started')):
                with patch.object(Path,'read_bytes',side_effect=AssertionError('config read before readiness')):
                    with self.assertRaises(SystemExit) as caught:
                        mission.main(['--mode','reference','--execute','--experiment-config','/not-read.json'])
        self.assertIn('missing reference task inputs',str(caught.exception))

    def test_manifest_records_hashes_without_copying_model_or_source_tree(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);work=root/'work';output=root/'output'
            (work/'runtime').mkdir(parents=True);output.mkdir()
            (work/'runtime/mission.py').write_text('pass\n')
            (work/'nso_bridge/models').mkdir(parents=True)
            (work/'nso_bridge/models/model.pt').write_bytes(b'not copied')
            with patch.object(mission,'WORK',work):mission.write_source_manifest(output)
            data=json.loads((output/'source_manifest.json').read_text())
            self.assertEqual(len(data['files']),1)
            self.assertEqual(data['files'][0]['sha256'],hashlib.sha256(b'pass\n').hexdigest())
            self.assertEqual([p.name for p in output.iterdir()],['source_manifest.json'])

    def test_execution_fault_cannot_be_reported_as_a_method_stop(self):
        result=dict(status='nso_budget_or_gain_stop')
        mission.execution_interruption(result,'chassis_stop_not_confirmed')
        self.assertEqual(result['status'],'execution_interrupted')
        self.assertEqual(result['decision_status_before_interruption'],'nso_budget_or_gain_stop')
        self.assertTrue(result['execution_interrupted'])

    def test_explorer_exit_needs_explicit_stop(self):
        self.assertTrue(mission.unexpected_child_exit('explore',0,False))
        self.assertFalse(mission.unexpected_child_exit('explore',0,True))
        self.assertTrue(mission.unexpected_child_exit('native',0,True))
        self.assertFalse(mission.unexpected_child_exit('explore',None,False))

    def test_actuator_heartbeat_cannot_refresh_original_CAN_time(self):
        mirror=mission.ActuatorTelemetry('s')
        row=dict(session_id='s',sampled_monotonic=1.,fault=None,ever_armed=True,
                 can_distance_m=.3,can_absolute_yaw_rad=.2,can_zero_since_monotonic=.1,
                 feedback_accounting_complete=True,hardware_reason=None,
                 can_status=dict(source=1.,receipt=1.,state=0,mode=1,error=0),
                 can_velocity=dict(source=1.,receipt=1.,v=0.,w=0.))
        mirror.update(row,'/nso_radius_guard',1.)
        self.assertIsNone(mirror.hardware_reason(1.1))
        row['sampled_monotonic']=1.4
        mirror.update(row,'/nso_radius_guard',1.4)
        self.assertEqual(mirror.hardware_reason(1.51),'original_CAN_status_stale')

    def test_actuator_telemetry_rejects_other_session_or_publisher(self):
        mirror=mission.ActuatorTelemetry('s')
        for row,publisher in [(dict(session_id='other'),'/nso_radius_guard'),
                              (dict(session_id='s'),'/another_node')]:
            with self.assertRaises(ValueError):mirror.update(row,publisher,1.)


if __name__=='__main__':unittest.main()
