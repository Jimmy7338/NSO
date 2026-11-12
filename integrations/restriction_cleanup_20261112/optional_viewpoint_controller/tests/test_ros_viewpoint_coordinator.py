"""Fake ROS/worker contracts: one full candidate goal, one actual outcome."""
import sys,time,types,unittest,ast
import numpy as np
from unittest.mock import patch
from ros_viewpoint_coordinator import run_viewpoints
N=types.SimpleNamespace
class ViewpointTests(unittest.TestCase):
    def exercise(self,status=3,timeout=False,hardware=False,profile='ros_viewpoint_exploration_v3'):
        calls=[];sent=[];captures=[];events=[];result={};cancel=[]
        class Clock:
            @staticmethod
            def now():return Clock()
        def goal():return N(target_pose=N(header=N(),pose=N(position=N(),orientation=N())))
        class Worker:
            decisions=0
            def call(self,method,params,heartbeat):
                calls.append((method,params))
                if method=='observe_viewpoint_result':return dict(state=dict(attempt_id=params['attempt_id'],pose=[5,7,2]),summary={},outcome=dict(waypoint_attempt_charged=params['attempt_id']>0,primitive_arrival_credit=False))
                self.decisions+=1
                return dict(target=dict(attempt_id=1,pose=[1,8,0],target_odom_pose=[2.5,1.7,1.2],selected={'candidate_id':4},predicted_nominal_grid_cost=23,predicted_nominal_outbound_cost=11,goal_role='coverage'),summary={}) if self.decisions==1 else dict(target=None,summary={})
        class Client:
            wait_calls=0
            gh=N(comm_state_machine=N(action_goal=N(goal_id=N(id='whole-viewpoint'))))
            def send_goal(self,g):sent.append(g)
            def cancel_goal(self):cancel.append(True)
            def wait_for_result(self,t):
                self.wait_calls+=1
                return not timeout or self.wait_calls>1
            def get_state(self):return status
        def check(active):
            if hardware and sent:raise RuntimeError('chassis_error')
        def capture(calibration,attempt,target,action):captures.append((attempt,target,action));return '/fresh-actual.npz'
        ctx=dict(task_receipt=dict(schema='nso_joint_runtime_v10/1',execution_allowed=True,
                    configuration=dict(policy_variant=profile),calibration={},summary={},grid=dict(shape=[20,20],origin_xy_m=[0,0],resolution_m=.2,planning_from_odom=[[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]])),
            worker=Worker(),client=Client(),result=result,journal=N(add=events.append),check=check,progress=lambda:None,
            capture_odom_pose=lambda:dict(pose=[.3,.4,.2],source_stamp_s=1.),capture_observation=capture,
            can_snapshot=lambda:dict(distance_m=.5,accounting_complete=True),run_start=0.,duration=600.,
            execution_frame='odom',xy_tolerance_m=.08,yaw_tolerance_rad=.08)
        clock=iter([i*.01 for i in range(1000)])
        if timeout:ctx['viewpoint_goal_timeout_s']=.001
        modules={'rospy':N(Time=Clock,Duration=lambda v:v),'move_base_msgs':N(),'move_base_msgs.msg':N(MoveBaseGoal=goal)}
        with patch.dict(sys.modules,modules),patch('ros_viewpoint_coordinator.time.monotonic',side_effect=lambda:next(clock)):run_viewpoints(ctx)
        return calls,sent,captures,events,result,cancel
    def test_one_absolute_macro_goal_no_primitive_or_cardinal_gate(self):
        calls,sent,captures,events,result,cancel=self.exercise()
        self.assertEqual(len(sent),1);g=sent[0]
        self.assertEqual((g.target_pose.pose.position.x,g.target_pose.pose.position.y),(2.5,1.7))
        self.assertEqual(captures,[(0,None,None),(1,None,None)])
        self.assertEqual([m for m,p in calls],['observe_viewpoint_result','next_viewpoint','observe_viewpoint_result','next_viewpoint'])
        self.assertEqual(result['viewpoint_attempts'][0]['predicted_nominal_grid_cost'],23)
        self.assertFalse(result['viewpoint_attempts'][0]['metric_endpoint_error']['verified'])
        self.assertFalse(result['viewpoint_attempts'][0]['scientific_cardinal_cell_gate'])
        self.assertEqual(calls[-2][1]['ros_status'],3)
    def test_abort_fuses_actual_then_replans(self):
        calls,sent,captures,events,result,cancel=self.exercise(status=4)
        self.assertEqual(calls[-2][1]['ros_status'],4)
        self.assertEqual(len(captures),2);self.assertEqual(len(result['viewpoint_attempts']),1)
    def test_timeout_cancel_ack_then_status2_actual_failure_not_whole_session_abort(self):
        calls,sent,captures,events,result,cancel=self.exercise(status=2,timeout=True)
        self.assertEqual(len(cancel),1)
        self.assertEqual(calls[-2][1]['ros_status'],2)
        self.assertEqual(calls[-2][1]['reason'],'ROS_goal_timeout_cancel_acknowledged')
        self.assertEqual(result['status'],'nso_viewpoint_core_stopped')
    def test_hardware_exception_fatal(self):
        with self.assertRaisesRegex(RuntimeError,'chassis_error'):self.exercise(hardware=True)
    def test_mission_dispatch_is_profile_opt_in(self):
        source=(__import__('pathlib').Path(__file__).resolve().parents[1]/'runtime/mission.py').read_text()
        tree=ast.parse(source)
        branch=next(n for n in ast.walk(tree) if isinstance(n,ast.If) and any(isinstance(x,ast.Constant) and x.value=='ros_viewpoint_exploration_v3' for x in ast.walk(n.test)))
        self.assertTrue(any(isinstance(x,ast.ImportFrom) and x.module=='ros_viewpoint_coordinator' for x in branch.body))
        self.assertIn('joint_runner=run_joint',source)
        self.assertIn('joint_runner(dict(worker=reference_worker',source)
    def test_legacy_identity_cannot_enable_new_coordinator(self):
        with self.assertRaises(ValueError):self.exercise(profile='exploration_joint_v1')
if __name__=='__main__':unittest.main()

class ControllerSelectionTests(unittest.TestCase):
    def test_default_passthrough_and_both_explicit_backends(self):
        import importlib.util
        from pathlib import Path
        path=Path(__file__).resolve().parents[1]/'runtime/mission.py'
        spec=importlib.util.spec_from_file_location('candidate_mission_selector',path);mission=importlib.util.module_from_spec(spec);spec.loader.exec_module(mission)
        self.assertEqual(mission.argument_parser().parse_args([]).execution_controller,'config')
        original={'configuration':{'policy_variant':'exploration_joint_v1'},'calibration':{'sensor':'actual'}}
        for choice,profile in [('config','exploration_joint_v1'),('nso_primitive','ros_checked_exploration_v2'),('ros_viewpoint','ros_viewpoint_exploration_v3')]:
            selected,receipt=mission.select_execution_controller(original,choice)
            self.assertEqual(selected['configuration']['policy_variant'],profile)
            self.assertEqual(selected['calibration'],original['calibration'])
            self.assertEqual(receipt['original_policy_variant'],'exploration_joint_v1')
            self.assertEqual(original['configuration']['policy_variant'],'exploration_joint_v1')
        self.assertIn('joint_runner=run_joint',path.read_text())
        self.assertIn("if args.execution_controller!='config' and args.mode not in ('nso','joint'):",path.read_text())
