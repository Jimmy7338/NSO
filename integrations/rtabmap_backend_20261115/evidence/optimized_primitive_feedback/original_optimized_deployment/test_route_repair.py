"""Software-only route repair contracts; no ROS, actuation or invented samples."""
import sys,unittest,copy
from pathlib import Path
from types import SimpleNamespace as NS
import numpy as np
W=Path('/home/ouc/nso_robot/workspaces/nso_hybrid_20261029');O=Path(__file__).parent;sys.path.insert(0,str(W/'nso_bridge'))
import nso_bridge
from nso_bridge import joint_engine
nso_bridge.__path__=[str(O/'candidate')]+list(nso_bridge.__path__)
from nso_bridge.optimized_primitive_feedback_v1 import repair_same_fixed_map_target
from nso_bridge.primitive_geometry_rebase_v1 import literal_prefix_endpoint
class Tests(unittest.TestCase):
 def fixture(self,*,target=(3,4,1),actions=('forward',),current=(3,2),heading=1):
  packet=NS(position=current,heading=heading);option=dict(pose=list(target),option_id='original-fixed-map-target',outbound_states=[[3,3,1],list(target)],scores='unchanged-original')
  state=dict(packet=packet,option=option,active_actions=list(actions),pending=None,arrived_count=0,phase='outbound')
  scene=dict(gain=NS(_pending={}),selected=copy.deepcopy(option),plans=1)
  inner=NS(runtime=NS(states=[state],audit=[]),backend=NS(scenes=[scene]),mapper=NS(belief=np.zeros((9,9),np.int8)),configuration=dict(planning_from_odom=np.eye(4).tolist()))
  return inner,state,scene
 def test_same_fixed_target_repair_does_not_change_choice(self):
  inner,state,scene=self.fixture();old=copy.deepcopy(state['option']);receipt=repair_same_fixed_map_target(inner)
  self.assertTrue(receipt['repaired']);self.assertEqual(state['active_actions'],['forward','forward']);self.assertEqual(state['option'],old);self.assertEqual(scene['plans'],1);self.assertEqual(state['arrived_count'],0);self.assertFalse(receipt['global_chooser_called'])
 def test_original_consistent_prefix_preserved(self):
  inner,state,scene=self.fixture(actions=('forward','forward'));receipt=repair_same_fixed_map_target(inner)
  self.assertFalse(receipt['repaired']);self.assertEqual(state['active_actions'],['forward','forward']);self.assertEqual(inner.runtime.audit,[])
 def test_virtual_current_and_unknown_are_admitted(self):
  inner,state,scene=self.fixture();inner.mapper.belief[:]=-1;inner.mapper.belief[3,2]=1
  receipt=repair_same_fixed_map_target(inner);self.assertTrue(receipt['route_available']);self.assertEqual(state['active_actions'],['forward','forward'])
 def test_unavailable_route_uses_existing_rechoose_without_credit(self):
  inner,state,scene=self.fixture();inner.mapper.belief[3,4]=1;oldplans=scene['plans'];r=repair_same_fixed_map_target(inner)
  self.assertFalse(r['route_available']);self.assertEqual(r['reason'],'navigation_route_invalidated_by_rebase');self.assertEqual(state['active_actions'],[]);self.assertIsNone(state['option']);self.assertEqual(scene['plans'],oldplans);self.assertEqual(state['arrived_count'],0);self.assertFalse(r['paid_action_or_observation_created']);self.assertFalse(r['new_stop_or_limit'])
 def test_disconnected_does_not_force_a_route(self):
  inner,state,scene=self.fixture();inner.mapper.belief[:]=1;inner.mapper.belief[3,4]=0
  r=repair_same_fixed_map_target(inner);self.assertFalse(r['route_available']);self.assertEqual(state['active_actions'],[])
 def test_completed_commitment_never_resurrected(self):
  inner,state,scene=self.fixture(actions=());state['arrived_count']=1
  r=repair_same_fixed_map_target(inner);self.assertFalse(r['repaired']);self.assertEqual(state['active_actions'],[]);self.assertEqual(state['arrived_count'],1)
 def test_pose_optimization_reaches_goal_without_fake_arrival(self):
  inner,state,scene=self.fixture(target=(3,2,1));r=repair_same_fixed_map_target(inner)
  self.assertTrue(r['repaired']);self.assertEqual(state['active_actions'],[]);self.assertTrue(r['zero_action_route_without_arrival_credit']);self.assertEqual(state['arrived_count'],0)
 def test_no_execution_repair_during_pending_prediction(self):
  inner,state,scene=self.fixture();state['pending']={'actual_primitive':True}
  with self.assertRaises(ValueError):repair_same_fixed_map_target(inner)
if __name__=='__main__':unittest.main()
