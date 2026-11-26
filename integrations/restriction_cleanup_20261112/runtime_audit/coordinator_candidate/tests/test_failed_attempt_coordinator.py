import sys,time,unittest
from types import SimpleNamespace as N
from unittest.mock import patch
from joint_ros_coordinator import run_joint
class FailedAttemptTests(unittest.TestCase):
    def run_case(self,v2=True,status=4,mismatch=False,hardware_fault=False):
        class Clock:
            @staticmethod
            def now():return Clock()
        def goal():return N(target_pose=N(header=N(),pose=N(position=N(),orientation=N())))
        calls=[];captures=[];events=[];sent=[];result={}
        class Worker:
            count=0
            def call(self,method,params,heartbeat):
                calls.append((method,params))
                if method=='next_action':
                    self.count+=1
                    return (dict(action='forward',pending=dict(action_id=1,pose=[17,2,1],target_odom_pose=[.5,.5,0.]),selected={}) if self.count==1 else dict(action=None,pending=None,summary={}))
                response=dict(state=dict(action_id=params['action_id'],pose=[17,1,1]),summary={})
                if method=='observe_failed_action':response['failed_execution_attempt']=dict(arrival_credited=False,paid_attempt_charged=True)
                return response
        client=N(send_goal=lambda g:sent.append(g),wait_for_result=lambda t:True,get_state=lambda:status,
                 gh=N(comm_state_machine=N(action_goal=N(goal_id=N(id='goal1')))))
        def check(active):
            if hardware_fault and sent:raise RuntimeError('chassis:actual_error')
        def capture(calibration,action_id,target,action):
            captures.append((action_id,target,action));return 'fresh-actual.npz'
        pose=dict(pose=[.3,.5,0.],source_stamp_s=1.)
        ctx=dict(task_receipt=dict(schema='nso_joint_runtime_v10/1',execution_allowed=True,calibration={},summary={},
            configuration={'policy_variant':'ros_checked_exploration_v2' if v2 else 'exploration_joint_v1'},
            grid=dict(shape=[20,20],origin_xy_m=[0.,0.],resolution_m=.2,planning_from_odom=[[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]])),
            worker=Worker(),client=client,result=result,journal=N(add=events.append),check=check,progress=lambda:None,
            run_start=time.monotonic(),duration=100,execution_frame='odom',capture_odom_pose=lambda:dict(pose),
            can_snapshot=lambda:{},capture_observation=capture,xy_tolerance_m=.05,yaw_tolerance_rad=.35)
        modules={'rospy':N(Time=Clock,Duration=lambda v:v),'move_base_msgs':N(),'move_base_msgs.msg':N(MoveBaseGoal=goal),
                 'mission':N(arrival_evidence=lambda *a,**k:dict(verified=status==3))}
        with patch.dict(sys.modules,modules):run_joint(ctx)
        return calls,captures,events,result,sent
    def test_v2_abort_observed_once_then_next_decision(self):
        calls,captures,events,result,sent=self.run_case()
        self.assertEqual([x[0] for x in calls],['observe','next_action','observe_failed_action','next_action'])
        self.assertEqual(captures,[(0,None,None),(1,None,'forward')])
        self.assertEqual(len(sent),1);self.assertEqual(result['status'],'nso_joint_core_stopped')
        failure=next(x for x in events if x['event']=='NSO_joint_failed_attempt_observed')
        self.assertFalse(failure['arrival_credited'])
        self.assertEqual([x['action_id'] for x in events if x['event']=='NSO_joint_observation_accepted'],[0])
    def test_v2_success_wrong_grid_no_fakearrival_or_retry(self):
        calls,captures,events,result,sent=self.run_case(status=3)
        self.assertEqual(len(sent),1)
        self.assertFalse(result['joint_actions'][0]['endpoint_error']['verified'])
        self.assertFalse(result['joint_actions'][0]['arrival_credited'])
        self.assertEqual(sum(m=='observe_failed_action' for m,p in calls),1)
        self.assertEqual(sum(m=='observe' for m,p in calls),1)
    def test_legacy_abort_stays_terminal(self):
        calls,captures,events,result,sent=self.run_case(v2=False)
        self.assertEqual(result['status'],'navigation_execution_failed')
        self.assertFalse(any(m=='observe_failed_action' for m,p in calls));self.assertEqual(len(captures),1)
    def test_hardware_error_fatal_not_swallowed(self):
        with self.assertRaisesRegex(RuntimeError,'actual_error'):self.run_case(hardware_fault=True)
if __name__=='__main__':unittest.main()
