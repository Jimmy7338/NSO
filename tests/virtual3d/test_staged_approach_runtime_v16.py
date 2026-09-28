from types import SimpleNamespace
import unittest
from nso.staged_approach_runtime_v16 import StagedApproachRuntimeV16


def fixture(allowed=True):
    assessments=[]
    comp=SimpleNamespace(args=SimpleNamespace(eval=True),use_ov_sem=True,use_topo=True,
        use_rpn_uq=True,use_igcr=True,bind_execution_option=lambda *args:None)
    def assess(scene,action):
        assessments.append(action)
        return dict(allowed=allowed,action=action,reason='fixture')
    comp.assess_local_action=assess
    runtime=StagedApproachRuntimeV16(comp,1,(10,10))
    runtime.states[0]=dict(closed=False,pending=None,phase='outbound',active_actions=['forward'],
        option=dict(option_id='one'),packet=SimpleNamespace(action_id=26))
    return runtime,assessments


class StagedRuntimeTests(unittest.TestCase):
    def test_denial_does_not_authorize_or_consume_an_action(self):
        runtime,calls=fixture(False)
        self.assertIsNone(runtime.authorize_probe_action('forward',phase='outbound'))
        self.assertIsNone(runtime.states[0]['pending'])
        self.assertEqual(runtime.states[0]['active_actions'],['forward'])
        self.assertEqual(calls,['forward'])
        runtime.begin_return()
        self.assertEqual(runtime.states[0]['phase'],'return')
        self.assertEqual(runtime.states[0]['active_actions'],[])

    def test_pending_observation_blocks_a_second_authorization_and_return(self):
        runtime,calls=fixture()
        self.assertEqual(runtime.authorize_probe_action('forward',phase='outbound'),'forward')
        with self.assertRaises(RuntimeError):runtime.authorize_probe_action('forward',phase='outbound')
        with self.assertRaises(RuntimeError):runtime.begin_return()
        self.assertEqual(calls,['forward'])

    def test_undeclared_outbound_primitive_is_rejected_before_guard(self):
        runtime,calls=fixture()
        with self.assertRaises(ValueError):runtime.authorize_probe_action('left',phase='outbound')
        self.assertEqual(calls,[])
        self.assertIsNone(runtime.states[0]['pending'])


if __name__=='__main__':unittest.main()
