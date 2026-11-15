"""No-hardware transport tests: fake ROS terminal outcomes and exact RPC ordering."""
import sys,unittest,types
from pathlib import Path
from types import SimpleNamespace as NS
W=Path('/home/ouc/nso_robot/workspaces/nso_hybrid_20261029');O=Path(__file__).parent
sys.path.insert(0,str(W/'runtime'));sys.path.insert(0,str(O/'candidate'))
import primitive_feedback_coordinator as coordinator
class Goal:
 def __init__(self):self.target_pose=NS(header=NS(),pose=NS(position=NS(),orientation=NS()))
class Tests(unittest.TestCase):
 def run_case(self,status,count):
  rpc=[];sent=[];events=[]
  class Worker:
   def call(self,method,params,heartbeat):
    heartbeat();rpc.append((method,params))
    if method=='observe_primitive_result':return dict(state=dict(action_id=params['action_id']),summary=dict(macro_arrivals_verified=0),primitive_outcome=dict(primitive_arrival_verified=status==3))
    if method=='next_action':
     index=sum(x[0]=='next_action'for x in rpc)
     return dict(action='forward',pending=dict(action='forward',action_id=index,target_world_frame='odom',target_odom_pose=[.2*index,0.,0.]))
    raise AssertionError('unexpected method '+method)
  class Client:
   gh=NS(comm_state_machine=NS(action_goal=NS(goal_id=NS(id='software-goal'))))
   def send_goal(self,goal):sent.append(goal)
   def wait_for_result(self,duration):return True
   def get_state(self):return status
   def cancel_goal(self):raise AssertionError('no cancellation expected')
  ros=types.ModuleType('rospy');ros.Duration=lambda x:x;ros.Time=NS(now=lambda:0)
  move=types.ModuleType('move_base_msgs.msg');move.MoveBaseGoal=Goal
  oldros=sys.modules.get('rospy');oldmove=sys.modules.get('move_base_msgs.msg');sys.modules['rospy']=ros;sys.modules['move_base_msgs.msg']=move
  validate=coordinator.validate_joint_identity;coordinator.validate_joint_identity=lambda x:x
  result={};context=dict(task_receipt=dict(configuration=dict(policy_variant=coordinator.PROFILE),calibration={}),duration=None,primitive_feedback_pilot_attempts=count,worker=Worker(),client=Client(),result=result,journal=NS(add=events.append),check=lambda x:None,progress=lambda:None,capture_observation=lambda *args:'/software_fixture_only.npz',capture_odom_pose=lambda:dict(pose=[.2*len(sent),0.,0.]),can_snapshot=lambda:dict(distance_m=.2*len(sent)),execution_frame='odom',xy_tolerance_m=.15,yaw_tolerance_rad=.35)
  try:coordinator.run_primitive_feedback(context)
  finally:
   coordinator.validate_joint_identity=validate
   if oldros is None:sys.modules.pop('rospy',None)
   else:sys.modules['rospy']=oldros
   if oldmove is None:sys.modules.pop('move_base_msgs.msg',None)
   else:sys.modules['move_base_msgs.msg']=oldmove
  return rpc,sent,result
 def test_two_real_outcome_rpcs_harness_no_method_stop(self):
  rpc,sent,result=self.run_case(3,2)
  self.assertEqual([x[0]for x in rpc],['observe_primitive_result','next_action','observe_primitive_result','next_action','observe_primitive_result'])
  self.assertEqual(len(sent),2);self.assertTrue(all(x.target_pose.header.frame_id=='odom'for x in sent));self.assertFalse(result['method_stop']);self.assertFalse(result['full_exploration_complete']);self.assertFalse(result['pilot_limit_is_method_budget']);self.assertEqual(result['primitive_verified_arrivals'],2)
 def test_failed_ros_goal_observed_without_arrival_credit(self):
  rpc,sent,result=self.run_case(4,1)
  self.assertEqual(rpc[-1][1]['ros_status'],4);self.assertEqual(result['primitive_verified_arrivals'],0);self.assertEqual(len(result['primitive_feedback_attempts']),1);self.assertFalse(result['full_macro_success_claimed'])
if __name__=='__main__':unittest.main()
