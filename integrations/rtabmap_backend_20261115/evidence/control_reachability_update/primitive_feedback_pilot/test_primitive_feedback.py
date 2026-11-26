"""Software-only synthetic contract tests; these are not experimental robot evidence."""
import sys,unittest,copy,math,tempfile
from pathlib import Path
from types import SimpleNamespace
import numpy as np
W=Path('/home/ouc/nso_robot/workspaces/nso_hybrid_20261029');O=Path(__file__).parent
sys.path.insert(0,str(W/'nso_bridge'));sys.path.insert(0,str(O/'candidate'))
from nso_bridge import joint_engine
from primitive_feedback_unbounded_v1 import PrimitiveCPU,PrimitiveRuntime,FrozenEpochPrimitivePilot
from env.virtual3d_v2 import VirtualConfigV2
from env.virtual3d import camera_pose
from utils.rgbd_contract import RGBDFrame
class Mapper:
 def __init__(self):
  self.shape=(13,13);self.belief=np.full(self.shape,-1,np.int8);self.belief[6,6]=0;self.camera_seen=np.zeros(self.shape,bool);self.quality={};self.frames=1
  self.config=VirtualConfigV2(resolution_m=.2,robot_radius_m=.4,max_depth_m=1.,width_px=8,height_px=8,fov_deg=90)
class Packet:
 def __init__(self,mapper,aid,position,heading):
  self.position=position;self.heading=heading;self.action_id=aid;self.frame_id='software-fixture-'+str(aid);self.scene_id='software';self.episode_id='not_robot';self.collision=False;self.done=False
  self.frame=RGBDFrame(100.+aid,np.full((8,8),.7,np.float32),np.zeros((8,8,3),np.uint8),np.array([[4.,0,3.5],[0,4.,3.5],[0,0,1.]]),camera_pose(position,heading,mapper.config,13),np.zeros((8,8),np.int16))
 def sha256(self):return 'software-fixture-hash-'+str(self.action_id)
class Tests(unittest.TestCase):
 def make(self):
  mapper=Mapper();packet=Packet(mapper,0,(6,6),1);args=SimpleNamespace(cpu_planner_revision='v10_3_1',run_id='software',cpu_score_mode='S')
  backend=PrimitiveCPU(args,1,mapper.shape);backend.start_scene(0,mapper=mapper,packet=packet,prefix_packets=[packet],total_budget=None,paid_prefix_actions=0,return_anchor=(6,6,1));return mapper,backend
 def test_original_prepare_and_observe_changes_beta_once(self):
  mapper,b=self.make();assessment=b.assess_action(0,'forward');self.assertEqual(assessment['next_pose'],[6,7,1]);self.assertIsNone(assessment['remaining_budget'])
  gain=b.scenes[0]['gain'];pending=copy.deepcopy(gain._pending[1]);known_before=mapper.belief!=-1
  for cell in np.argwhere(pending['radar'])[:7]:mapper.belief[tuple(cell)]=0
  for cell in np.argwhere(pending['camera'])[:3]:mapper.camera_seen[tuple(cell)]=True
  mapper.frames=2;p=Packet(mapper,1,(6,7),1);b.bind_packet(0,p,arrived=False);b.compute_reward(0)
  event=gain.events[-1];actual_known=(mapper.belief!=-1)&~known_before
  self.assertEqual(event['radar']['realized_predicted_cells'],int(np.count_nonzero(pending['radar']&actual_known)))
  self.assertEqual(gain.alpha['radar'],1+event['radar']['realized_predicted_cells'])
  self.assertEqual(gain.beta['radar'],1+event['radar']['unrealized_predicted_cells'])
  self.assertEqual(gain.snapshot()['observed_actions'],1);self.assertFalse(gain._pending);self.assertEqual(b.scenes[0]['ledger'].snapshot()['paid_actions'],1)
  with self.assertRaises(ValueError):gain.observe(mapper,action_id=1)
 def test_turn_prediction_has_no_translation_and_quarter_turn(self):
  mapper,b=self.make();a=b.assess_action(0,'left');self.assertEqual(a['next_pose'],[6,6,0]);self.assertEqual(a['observed_gain_prediction']['next_pose'],[6,6,0]);self.assertEqual(b.scenes[0]['gain'].snapshot()['pending_action_ids'],[1])
 def test_unchanged_committed_prefix_does_not_rechoose_per_step(self):
  mapper,b=self.make();calls=[]
  class Components:
   _cpu_backend=b
   args=SimpleNamespace(eval=True,train_global=False)
   use_ov_sem=use_topo=use_rpn_uq=use_igcr=True
   def bind_execution_option(self,index,option):pass
   def assess_local_action(self,index,action):return dict(allowed=True,action=action,next_pose=[6,7,1],observed_gain_prediction={})
  runtime=PrimitiveRuntime(Components(),1,mapper.shape)
  state=dict(packet=b.scenes[0]['packet'],mapper=mapper,pending=None,closed=False,active_actions=[],option=None)
  runtime.states[0]=state
  def choose(index,position,bounds):calls.append(1);state['active_actions']=['forward','left','forward'];state['option']=dict(option_id='same-original-selection')
  runtime.choose_goal=choose
  for expected in ['forward','left','forward']:
   self.assertEqual(runtime.next_local_action(0),expected);state['pending']=None;state['active_actions'].pop(0)
  self.assertEqual(len(calls),1)
 def test_frozen_epoch_goal_converts_back_to_real_raw_odom(self):
  pilot=FrozenEpochPrimitivePilot.__new__(FrozenEpochPrimitivePilot)
  class Inner:
   backend=SimpleNamespace(scenes=[dict(gain=SimpleNamespace(_pending={1:{k:np.zeros((2,2),bool)for k in ['radar','camera','known_before','camera_before']}},snapshot=lambda:dict(posterior_mean=dict(radar=.5,camera=.5))))])
   def next_action(self):return dict(pending=dict(action_id=1,target_odom_pose=[3.,5.,math.pi/2]))
  pilot.inner=Inner();pilot.directory=Path(tempfile.mkdtemp(prefix='nso-software-test-'));pilot.frozen_prediction_inputs={};pose=np.eye(4);pose[:2,3]=[1.,2.]
  pilot.snapshot=dict(graph_revision_sha256='software',graph_source_stamp_ns=100,nodes=[dict(id=1,stamp_ns=100,odom_from_base=np.eye(4).tolist(),map_from_base=pose.tolist())]);pilot.records=[dict(original_sha256='software',geometry=dict(rgbd=dict(sensor_stamp_ns=101)))]
  r=pilot.next_action()['pending'];self.assertEqual(r['target_world_frame'],'odom');self.assertTrue(np.allclose(r['target_odom_pose'],[2.,3.,math.pi/2]));self.assertEqual(r['target_prediction_epoch_pose'],[3.,5.,math.pi/2])
if __name__=='__main__':unittest.main()
