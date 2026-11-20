import dataclasses
from types import SimpleNamespace as NS
import unittest
from controller import FrontierDriver

@dataclasses.dataclass
class Decision:
    action: str

class Fixtures:
    def __init__(self, actions=('forward',), allowed=(True,), remaining=256, returning=('left',)):
        self.actions=list(actions);self.allowed=list(allowed);self.calls=[]
        self.s=dict(closed=False,pending=None,phase='planning',active_actions=[],option=None,
                    packet=NS(frame_id='frame-0',action_id=0))
        self.policy=NS(goal=(1,2),_path=[(1,2)],act=self.act)
        self.comp=NS(_cpu_backend=NS(scenes=[dict(ledger=NS(remaining_budget=remaining))]),
            bind_execution_option=lambda *x:None,assess_local_action=self.assess,
            plan_observed_return=lambda _:NS(available=True,actions=returning,reason='return_available'))
        self.runtime=NS(states=[self.s],audit=[],close_sensor_episode=self.close)
        self.driver=FrontierDriver(self.comp,self.runtime,self.policy,lambda:'observed_only')
    def act(self, obs):
        assert obs=='observed_only'
        return Decision(self.actions.pop(0))
    def assess(self, index, action):
        self.calls.append(action)
        return dict(allowed=self.allowed.pop(0),action=action,reason='fixture',next_pose=[1,2,0])
    def close(self,index,reason):
        self.s['closed']=True;self.s['reason']=reason

class TestController(unittest.TestCase):
    def test_actual_budget_zero_stops_without_policy_or_motion(self):
        f=Fixtures(actions=(),remaining=0)
        self.assertIsNone(f.driver.next_action());self.assertEqual(f.calls,[])
        self.assertEqual(f.s['reason'],'budget_exhausted')
    def test_accepted_action_requires_fresh_observation(self):
        f=Fixtures();self.assertEqual(f.driver.next_action(),'forward')
        self.assertEqual(f.s['pending']['action'],'forward')
        with self.assertRaises(RuntimeError):f.driver.next_action()
    def test_guard_replans_once_without_paying_or_removing_candidates(self):
        f=Fixtures(actions=['forward','left'],allowed=[False,True])
        self.assertEqual(f.driver.next_action(),'left');self.assertEqual(f.calls,['forward','left'])
        self.assertEqual(f.comp._cpu_backend.scenes[0]['ledger'].remaining_budget,256)
        self.assertEqual(f.s['packet'].action_id,0)
        self.assertEqual(f.s['phase'],'outbound')
    def test_two_rejections_trigger_checked_return(self):
        f=Fixtures(actions=['forward','forward'],allowed=[False,False,True],returning=['right','forward'])
        self.assertEqual(f.driver.next_action(),'right');self.assertEqual(f.s['phase'],'return')
        self.assertEqual(f.s['active_actions'],['right','forward'])
        self.assertEqual(f.calls,['forward','forward','right'])
    def test_return_rejection_never_executes_unapproved_action(self):
        f=Fixtures(actions=['stop'],allowed=[False,False])
        self.assertIsNone(f.driver.next_action());self.assertTrue(f.s['closed'])
        self.assertIsNone(f.s['pending'])
    def test_policy_early_stop_at_anchor_preserves_zero_actual_actions(self):
        f=Fixtures(actions=['stop'],returning=[])
        self.assertIsNone(f.driver.next_action());self.assertEqual(f.calls,[])
        self.assertEqual(f.s['reason'],'frontier_policy_stopped_at_anchor')
    def test_return_queue_has_no_new_frontier_selection(self):
        f=Fixtures(actions=(),allowed=[True]);f.s['phase']='return';f.s['active_actions']=['left','forward']
        self.assertEqual(f.driver.next_action(),'left');self.assertEqual(f.actions,[])

if __name__=='__main__':unittest.main()
