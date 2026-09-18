"""Route refresh must preserve intention without paying or authorizing twice."""
from copy import deepcopy
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import numpy as np
from nso.facility_runtime_v20 import FacilityRuntimeV20
from nso.recovery_runtime_v14 import RecoveryRuntimeV14


class RuntimeRefreshTests(unittest.TestCase):
    def fixture(self, *, action_id=5, quality=False, admitted=True):
        original=dict(option_id='episode/plan-1', selection_call_id=7, pose=[3,5,1],
            group='asset_0_deep' if quality else 'coverage_v20_0', candidate_id=2,
            outbound_actions=['left','forward','forward'], v20_coverage_intent=not quality,
            route_revision=0)
        fresh={**deepcopy(original), 'outbound_actions':['forward'], 'outbound_cost':1, 'return_cost':4}
        events=[]
        bstate=dict(selected=deepcopy(original), ledger=SimpleNamespace(remaining_budget=20),
            coverage_v20=SimpleNamespace(assess_route=lambda *args: {'allowed':admitted}),region_continuation=None)
        space=SimpleNamespace(refresh=lambda old: deepcopy(fresh),route_mask=lambda route:np.ones((3,3),bool))
        backend=SimpleNamespace(scenes=[bstate],route_space=lambda index:space,
            _record=lambda *args:events.append(args))
        runtime=object.__new__(FacilityRuntimeV20)
        runtime.args=SimpleNamespace(cpu_v20_replan_interval=5)
        runtime.components=SimpleNamespace(_cpu_backend=backend)
        runtime.full_shape=(8,8);runtime.audit=[]
        runtime.states=[dict(closed=False,pending=None,phase='outbound',
            packet=SimpleNamespace(action_id=action_id,position=(3,4),heading=1),
            mapper=SimpleNamespace(frames=action_id+1),option=original,
            active_actions=['left','forward','forward'],v20_last_route_update=0)]
        return runtime,events

    def test_refresh_preserves_goal_and_attribution_and_uses_original_guard_once(self):
        runtime,events=self.fixture()
        with patch.object(RecoveryRuntimeV14,'next_local_action',return_value='guarded_action') as guard:
            self.assertEqual(runtime.next_local_action(0),'guarded_action')
        guard.assert_called_once_with(runtime,0)
        state=runtime.states[0]
        self.assertEqual(state['option']['pose'],[3,5,1])
        self.assertEqual(state['option']['option_id'],'episode/plan-1')
        self.assertEqual(state['option']['selection_call_id'],7)
        self.assertEqual(state['option']['route_revision'],1)
        self.assertEqual(state['active_actions'],['forward'])
        self.assertEqual(len(events),1)
        self.assertEqual(runtime.components._cpu_backend.scenes[0]['selected'],state['option'])

    def test_no_free_refresh_before_five_observed_actions_or_while_pending(self):
        runtime,events=self.fixture(action_id=4)
        with patch.object(RecoveryRuntimeV14,'next_local_action',return_value='guarded_action'):
            runtime.next_local_action(0)
        self.assertEqual(events,[])
        runtime.states[0]['pending']={'action':'forward'}
        with patch.object(RecoveryRuntimeV14,'next_local_action') as guard:
            with self.assertRaises(RuntimeError):runtime.next_local_action(0)
            guard.assert_not_called()

    def test_quality_commitment_cancelled_when_coverage_reserve_no_longer_fits(self):
        runtime,events=self.fixture(quality=True,admitted=False)
        with patch.object(RecoveryRuntimeV14,'next_local_action',return_value=None):
            runtime.next_local_action(0)
        self.assertIsNone(runtime.states[0]['option'])
        self.assertEqual(runtime.states[0]['phase'],'planning')
        self.assertEqual(runtime.audit[-1]['reason'],'reserve_remaining_coverage_budget')
        self.assertFalse(runtime.audit[-1]['target_retained'])

if __name__=='__main__':unittest.main()
