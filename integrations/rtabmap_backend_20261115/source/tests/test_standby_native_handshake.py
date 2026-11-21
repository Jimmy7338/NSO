"""Mock-only handshake ordering and original adapter prearm zero protection."""
import copy
import pathlib
import sys
import unittest
from unittest.mock import Mock
sys.path.insert(0,str(pathlib.Path(__file__).resolve().parents[1]/'runtime'))
from standby_native_handshake import standby_native_eligibility,NativeStartOnce
from visual_velocity_adapter import Gate


def snapshot(mode=0):
    return dict(receipt=10.,fault=None,ever_armed=False,can_zero_since=8.9,
                can_status=dict(source=9.95,state=0,mode=mode,error=0),
                can_velocity=dict(source=9.98,v=0.,w=0.))


class StandbyHandshakeTests(unittest.TestCase):
    def test_standby_zero_launches_before_original_check_and_arm(self):
        calls=[];state=snapshot()
        def native():calls.append('native');state['can_status']['mode']=1
        start=NativeStartOnce(native)
        self.assertTrue(start.consider_standby(state,10.,execute=True))
        calls.append('oldcheck');self.assertEqual(state['can_status']['mode'],1)
        start.start('normal');calls.append('old_active_check');calls.append('arm')
        self.assertEqual(calls,['native','oldcheck','old_active_check','arm'])

    def test_already_CAN1_retains_original_check_before_one_native(self):
        calls=[];start=NativeStartOnce(lambda:calls.append('native'))
        self.assertFalse(start.consider_standby(snapshot(1),10.,execute=True))
        calls.append('oldcheck');start.start('normal');start.start('normal_again')
        calls.extend(['old_active_check','arm'])
        self.assertEqual(calls,['oldcheck','native','old_active_check','arm'])

    def test_RC_fault_nonzero_stale_and_used_session_never_launch_early(self):
        variants=[]
        for mode in [1,2,3]:variants.append(snapshot(mode))
        for path,value in [('fault','native_fault'),('ever_armed',True),('can_zero_since',None),('can_zero_since',9.5),('receipt',9.4)]:
            s=snapshot();s[path]=value;variants.append(s)
        for group,key,value in [('can_status','source',9.4),('can_status','source',10.1),('can_status','error',1),('can_status','state',1),('can_velocity','source',9.84),('can_velocity','source',10.1),('can_velocity','v',.006),('can_velocity','w',.01)]:
            s=snapshot();s[group][key]=value;variants.append(s)
        for s in variants:
            with self.subTest(snapshot=s):
                launch=Mock();start=NativeStartOnce(launch)
                self.assertFalse(start.consider_standby(s,10.,execute=True));launch.assert_not_called()
        launch=Mock();self.assertFalse(NativeStartOnce(launch).consider_standby(snapshot(),10.,execute=False));launch.assert_not_called()

    def test_failed_transition_keeps_original_check_failure_and_runs_cleanup(self):
        calls=[];arm=Mock();state=snapshot();start=NativeStartOnce(lambda:calls.append('native'))
        try:
            start.consider_standby(state,10.,execute=True)
            calls.append('oldcheck')
            if state['can_status']['mode']!=1:raise RuntimeError('original_CAN1_required')
            arm()
        except RuntimeError as exc:self.assertEqual(str(exc),'original_CAN1_required')
        finally:calls.extend(['stop','native_close','adapter_close'])
        arm.assert_not_called();self.assertEqual(calls,['native','oldcheck','stop','native_close','adapter_close'])

    def test_launch_exception_never_marks_started_or_arms_and_cleanup_runs(self):
        arm=Mock();cleanup=Mock();start=NativeStartOnce(Mock(side_effect=RuntimeError('spawn failed')))
        try:
            start.consider_standby(snapshot(),10.,execute=True);arm()
        except RuntimeError:pass
        finally:cleanup()
        self.assertFalse(start.started);arm.assert_not_called();cleanup.assert_called_once()

    def test_unarmed_existing_gate_emits_only_zero_even_with_nonzero_proposal(self):
        gate=Gate(60.);gate.proposal=dict(v=.1,w=.3,receipt=10.,number=1)
        v,w,reason=gate.decision(10.,10.)
        self.assertEqual((v,w),(0.,0.));self.assertEqual(reason,'not_armed')
        self.assertIsNotNone(gate.hardware_reason(10.))

    def test_original_gate_still_rejects_standby_hardware_mode(self):
        gate=Gate(60.);s=snapshot();gate.can_status=s['can_status'];gate.can_velocity=s['can_velocity']
        self.assertEqual(gate.hardware_reason(10.),'chassis_state_mode_or_fault')
        self.assertTrue(standby_native_eligibility(s,10.,execute=True)[0])
        gate.can_status['mode']=1;self.assertIsNone(gate.hardware_reason(10.))


if __name__=='__main__':unittest.main()
