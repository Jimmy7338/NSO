"""Synthetic ROS statuses and actual-pose capture; no ROS/CAN connections."""
import math,sys,time,unittest
from types import SimpleNamespace as N
from unittest.mock import patch
from joint_ros_coordinator import run_joint

class BackoffTests(unittest.TestCase):
    def run_case(self,authorized=True,v2=True,original_status=4,backoff_status=3,stop_on_backoff=False):
        class Clock:
            @staticmethod
            def now():return Clock()
        def goal():return N(target_pose=N(header=N(),pose=N(position=N(),orientation=N())))
        calls=[];captures=[];events=[];sent=[];result={};actual=[.3,.5,math.pi/2]
        class Worker:
            count=0
            def call(self,method,params,heartbeat):
                calls.append((method,dict(params)))
                if method=='next_action':
                    self.count+=1
                    return (dict(action='forward',pending=dict(action_id=1,pose=[17,2,1],target_odom_pose=[.5,.5,0.]),selected={}) if self.count==1 else dict(action=None,pending=None,summary={}))
                r=dict(state=dict(action_id=params['action_id']),summary={})
                if method=='observe_failed_action':r['failed_execution_attempt']=dict(arrival_credited=False,paid_attempt_charged=True)
                return r
        client=N(gh=N(comm_state_machine=N(action_goal=N(goal_id=N(id='unset')))))
        def send(g):
            sent.append(g);client.gh.comm_state_machine.action_goal.goal_id.id='goal%d'%len(sent)
            if len(sent)==2:actual[:]=[.3,.3,math.pi/2]  # Real recovery ends short, never assigned target.
        client.send_goal=send;client.wait_for_result=lambda duration:True
        client.get_state=lambda:original_status if len(sent)==1 else backoff_status
        def check(active):
            if stop_on_backoff and len(sent)>1:raise RuntimeError('hardware_stop')
        def capture(calibration,aid,target,action):
            captures.append(dict(action_id=aid,target=target,action=action,actual_pose=list(actual)));return 'actual-%d.npz'%aid
        ctx=dict(task_receipt=dict(schema='nso_joint_runtime_v10/1',execution_allowed=True,calibration={},summary={},
            configuration={'policy_variant':'ros_checked_exploration_v2' if v2 else 'exploration_joint_v1'},
            grid=dict(shape=[20,20],origin_xy_m=[0.,0.],resolution_m=.2,planning_from_odom=[[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]])),
            ordinary_reverse_authorized=authorized,worker=Worker(),client=client,result=result,journal=N(add=events.append),
            check=check,progress=lambda:None,run_start=time.monotonic(),duration=100,execution_frame='odom',
            capture_odom_pose=lambda:dict(pose=list(actual),source_stamp_s=1.),can_snapshot=lambda:{},
            capture_observation=capture,xy_tolerance_m=.05,yaw_tolerance_rad=.35)
        modules={'rospy':N(Time=Clock,Duration=lambda v:v),'move_base_msgs':N(),'move_base_msgs.msg':N(MoveBaseGoal=goal),
                 'mission':N(arrival_evidence=lambda *args,**kw:dict(verified=False))}
        with patch.dict(sys.modules,modules):run_joint(ctx)
        return calls,captures,events,result,sent
    def test_abort_single_backoff_preserves_original_and_captures_actual_partial_pose(self):
        calls,captures,events,result,sent=self.run_case()
        self.assertEqual(len(sent),2)
        target=sent[1].target_pose.pose
        self.assertAlmostEqual(target.position.x,.3);self.assertAlmostEqual(target.position.y,.1)
        self.assertAlmostEqual(target.orientation.z,math.sin(math.pi/4))
        self.assertEqual([m for m,p in calls],['observe','next_action','observe_failed_action','next_action'])
        self.assertEqual(captures[-1]['actual_pose'],[.3,.3,math.pi/2]);self.assertIsNone(captures[-1]['target'])
        action=result['joint_actions'][0];recovery=result['recovery_interventions'][0]
        self.assertEqual(action['original_ros_terminal'],{'id':'goal1','status':4})
        self.assertEqual(recovery['terminal'],{'id':'goal2','status':3})
        self.assertEqual(recovery['additional_NSO_primitive_charge'],0)
        self.assertFalse(recovery['NSO_arrival_credit']);self.assertFalse(action['arrival_credited'])
        self.assertEqual(len(result['joint_actions']),1)
        self.assertAlmostEqual(recovery['actual_odom_translation_m'],.2)
    def test_rejected_backoff_still_observes_and_replans_once(self):
        calls,captures,events,result,sent=self.run_case(backoff_status=4)
        self.assertEqual(len(sent),2);self.assertEqual(result['recovery_interventions'][0]['terminal']['status'],4)
        self.assertEqual(sum(m=='observe_failed_action' for m,p in calls),1)
        self.assertEqual(result['status'],'nso_joint_core_stopped')
    def test_legacy_unauthorized_and_success_mismatch_do_not_backoff(self):
        for kwargs in [dict(v2=False),dict(authorized=False),dict(original_status=3)]:
            with self.subTest(kwargs=kwargs):
                *_,result,sent=self.run_case(**kwargs)
                self.assertEqual(len(sent),1);self.assertNotIn('recovery_interventions',result)
    def test_hardware_stop_is_not_swallowed_or_fused_as_recovered(self):
        with self.assertRaisesRegex(RuntimeError,'hardware_stop'):self.run_case(stop_on_backoff=True)
if __name__=='__main__':unittest.main()
