"""Pure opt-in correction permission/config lifecycle; no ROS or hardware."""
import json
import pathlib
import sys
import unittest
from types import SimpleNamespace
sys.path.insert(0,str(pathlib.Path(__file__).resolve().parents[1]/'runtime'))
from visual_velocity_adapter import Gate,configuration
from correction_reverse_scope import correction_reverse_scope,PROFILE
import mission


class CorrectionReverseGateTests(unittest.TestCase):
    def gate(self,**kw):
        g=Gate(nso_correction_reverse=True,mission_mode='nso',**kw)
        g.session_id='test_session';return g

    def propose(self,g,v):
        g.propose([v,0.,0.,0.,0.,.2],10.,'/nso_visual/move_base')

    def phase(self,g,enabled):
        return g.set_correction_reverse(enabled,'/nso_onboard_supervisor','test_session')

    def test_normal_negative_clipped_only_authorized_phase_allows_minus_point05(self):
        g=self.gate();self.propose(g,-.1);self.assertEqual(g.proposal['v'],0.)
        g.armed=True;g.ever_armed=True
        self.assertTrue(self.phase(g,True)[0]);self.propose(g,-.1)
        self.assertEqual(g.proposal['v'],-.05)
        old=dict(g.proposal)
        self.assertTrue(self.phase(g,False)[0]);self.assertEqual(g.proposal['v'],0.)
        self.assertEqual(g.proposal['receipt'],old['receipt'])
        self.assertEqual(g.proposal['number'],old['number'])
        self.assertEqual(g.proposal['w'],old['w'])
        self.assertEqual(g.proposal_ttl,.2);self.assertEqual(g.observation_ttl,.5)
        self.propose(g,-.04);self.assertEqual(g.proposal['v'],0.)

    def test_unauthorized_unarmed_fault_wrong_caller_and_session_rejected(self):
        g=self.gate()
        self.assertFalse(self.phase(g,True)[0]);g.armed=True
        self.assertFalse(g.set_correction_reverse(True,'/other','test_session')[0])
        self.assertFalse(g.set_correction_reverse(True,'/nso_onboard_supervisor','other')[0])
        g.fail('source_fault');self.assertFalse(self.phase(g,True)[0])
        g=Gate();g.session_id='test_session';g.armed=True
        self.assertFalse(self.phase(g,True)[0]);self.assertEqual(g.minimum_linear_m_s,0.)

    def test_fault_or_stop_disables_and_reclamps_without_renewing_receipt(self):
        for reason in ['explicit_stop','native_driver_fault:8','duration_complete','chassis_velocity_stale']:
            g=self.gate();g.armed=True;self.phase(g,True);self.propose(g,-.04);before=dict(g.proposal)
            g.fail(reason)
            self.assertFalse(g.correction_reverse_active);self.assertFalse(g.armed)
            self.assertEqual(g.minimum_linear_m_s,0.);self.assertEqual(g.proposal['v'],0.)
            self.assertEqual(g.proposal['receipt'],before['receipt'])
            self.assertEqual(g.decision(10.,10.)[:2],(0.,0.))
            self.assertTrue(self.phase(g,False)[0])

    def test_explicit_config_and_cli_mode_scope_default_unchanged(self):
        self.assertFalse(configuration({})['nso_correction_reverse'])
        self.assertFalse(mission.argument_parser().parse_args([]).nso_correction_reverse)
        for mode in ['navigation','reference','visual']:
            with self.assertRaises(ValueError):configuration(dict(nso_correction_reverse=True,mission_mode=mode))
            with self.assertRaises(ValueError):mission.main(['--mode',mode,'--nso-correction-reverse'])
        for mode in ['nso','joint']:
            self.assertTrue(configuration(dict(nso_correction_reverse=True,mission_mode=mode))['nso_correction_reverse'])
        g=Gate(navigation_allow_reverse=True,mission_mode='navigation');self.propose(g,-.1)
        self.assertEqual(g.proposal['v'],-.05)


class CorrectionReverseLifecycleTests(unittest.TestCase):
    def exercise(self,fail=None,body_error=False):
        events=[];values=dict(min_vel_x=0.,forward_point_distance=.325);records=[]
        class Config:
            def get_configuration(self,**kw):events.append('get');return dict(values)
            def update_configuration(self,change):
                entering=change['min_vel_x']<0;events.append('apply' if entering else 'restore')
                values.update(change) # Partial update may occur even if service raises.
                if fail==('apply' if entering else 'restore'):raise RuntimeError(fail)
                return dict(values)
        def phase(enabled):
            events.append('enable'if enabled else'disable')
            if fail==('enable'if enabled else'disable'):raise RuntimeError(fail)
        fixture=pathlib.Path('/home/ouc/nso_robot/reports/nso-hybrid-20261102/circle_warm_S_B24_02/result.json')
        row=json.loads(fixture.read_text())['joint_actions'][-1]
        endpoint=dict(action_id=row['action_id'],action=row['action'],pose=row['grid_pose'],target_odom_pose=row['target_odom_pose'])
        error=None
        try:
            with correction_reverse_scope(Config(),phase,lambda event,**kw:records.append((event,kw)),endpoint):
                events.append('goal_sent')
                self.assertEqual(values,dict(min_vel_x=-.05,forward_point_distance=0.))
                if body_error:raise RuntimeError('navigation_failed')
        except RuntimeError as exc:error=str(exc)
        return events,values,records,error

    def test_normal_scope_restore_and_real_same_pending_fixture(self):
        events,values,records,error=self.exercise()
        self.assertIsNone(error)
        self.assertEqual(events,['get','enable','apply','goal_sent','disable','restore'])
        self.assertEqual(values,dict(min_vel_x=0.,forward_point_distance=.325))
        receipt=records[-1][1]['receipt'];self.assertEqual(receipt['action_id'],13)
        self.assertEqual(receipt['profile'],PROFILE);self.assertTrue(receipt['disabled']);self.assertTrue(receipt['restored'])

    def test_real_gate_cached_reverse_is_clamped_on_scope_exception(self):
        gate=Gate(nso_correction_reverse=True,mission_mode='joint');gate.session_id='s';gate.armed=True
        values=dict(min_vel_x=0.,forward_point_distance=.325)
        class Config:
            def get_configuration(self,**kw):return dict(values)
            def update_configuration(self,change):values.update(change);return dict(values)
        def phase(enabled):
            ok,reason=gate.set_correction_reverse(enabled,'/nso_onboard_supervisor','s')
            if not ok:raise RuntimeError(reason)
        endpoint=dict(action_id=13,action='right',pose=[210,200,3],target_odom_pose=[0.,0.,0.])
        with self.assertRaisesRegex(RuntimeError,'execution failed'):
            with correction_reverse_scope(Config(),phase,lambda *a,**kw:None,endpoint):
                gate.propose([-.04,0,0,0,0,.1],42.,'/nso_visual/move_base')
                self.assertEqual(gate.proposal['v'],-.04)
                raise RuntimeError('execution failed')
        self.assertFalse(gate.correction_reverse_active);self.assertEqual(gate.proposal['v'],0.)
        self.assertEqual(gate.proposal['receipt'],42.)
        self.assertEqual(values,dict(min_vel_x=0.,forward_point_distance=.325))

    def test_navigation_exception_restores_before_propagating(self):
        events,values,records,error=self.exercise(body_error=True)
        self.assertEqual(error,'navigation_failed');self.assertEqual(events[-2:],['disable','restore'])
        self.assertEqual(values,dict(min_vel_x=0.,forward_point_distance=.325))

    def test_enable_or_apply_failure_cannot_send_goal_and_attempts_cleanup(self):
        for fail in ['enable','apply']:
            events,values,records,error=self.exercise(fail)
            self.assertIsNotNone(error);self.assertNotIn('goal_sent',events)
            self.assertEqual(events[-2:],['disable','restore'])
            self.assertEqual(values,dict(min_vel_x=0.,forward_point_distance=.325))

    def test_disable_failure_still_restores_and_never_continues_success(self):
        events,values,records,error=self.exercise('disable')
        self.assertIn('cleanup_failed',error);self.assertEqual(events[-1],'restore')
        self.assertEqual(values,dict(min_vel_x=0.,forward_point_distance=.325))

    def test_restore_failure_still_clears_adapter_phase_and_propagates(self):
        events,values,records,error=self.exercise('restore')
        self.assertIn('cleanup_failed',error);self.assertLess(events.index('disable'),events.index('restore'))
        self.assertTrue(records[-1][1]['receipt']['disabled'])
        self.assertFalse(records[-1][1]['receipt']['restored'])


if __name__=='__main__':unittest.main()
