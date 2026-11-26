import sys,unittest,tempfile,json,hashlib
from pathlib import Path
from copy import deepcopy
import numpy as np
WORK=Path('/home/ouc/nso_robot/workspaces/nso_hybrid_20261029')
sys.path.insert(0,str(WORK/'nso_bridge'))
from nso_bridge.rtabmap_viewpoint_session import RtabmapViewpointSession,assert_conserved,projected_anchor
from nso_bridge.rtabmap_geometry import corrected_geometry

def trans(x):
 t=np.eye(4);t[0,3]=x;return t

def graph(rev,offset):return dict(graph_revision_sha256=rev,graph_source_stamp_ns=30,nodes=[dict(id=1,stamp_ns=10,odom_from_base=trans(1).tolist(),map_from_base=trans(1+offset).tolist()),dict(id=2,stamp_ns=20,odom_from_base=trans(2).tolist(),map_from_base=trans(2+offset+1).tolist())])

class FakeSession:
 instances=[]
 def __init__(self,configuration,**kwargs):
  self.configuration=configuration;self.viewpoint_pending=None;self.waypoint_outcomes=[];self.semantic_frontend=None;self._instance_owner_map=None;self._attempt_semantic_receipt=None;self.calls=0;self.inferences=0;self.observations=[];self.instances.append(self)
 def _semantic_labels(self,z,meta,sha):
  self.inferences+=1;self._instance_owner_map=np.array([[5]],np.int32);self._attempt_semantic_receipt={'actual':True};return np.array([[2]],np.int32)
 def observe_viewpoint_result(self,path,attempt_id,ros_status=None,reason=None):
  with np.load(path,allow_pickle=False)as z:
   m=json.loads(str(z['joint_metadata_json'].item()));labels=self._semantic_labels(z,m,'derivative_sha');self.observations.append(dict(id=attempt_id,grid=z['joint_belief_grid'].copy(),base=z['base_odom_pose'].copy(),labels=labels,owner=self._instance_owner_map.copy()))
  if attempt_id:
   self.waypoint_outcomes.append(dict(ros_status=ros_status,reason=reason,target=deepcopy(self.viewpoint_pending)));self.viewpoint_pending=None
  return dict(accepted=True,outcome={'ros_status':ros_status,'reason':reason})
 def next_viewpoint(self,**kwargs):
  self.calls+=1;self.viewpoint_pending=dict(attempt_id=len(self.waypoint_outcomes)+1,pose=[0,0,0],target_odom_pose=[1,2,0]);return dict(target=deepcopy(self.viewpoint_pending))
 def summary(self):return dict(waypoint_attempts=len(self.waypoint_outcomes))
 def identity(self):return {'receipt':{}}

class Tests(unittest.TestCase):
 def setUp(self):
  self.tmp=tempfile.TemporaryDirectory();self.root=Path(self.tmp.name);self.c=dict(policy_variant='ros_viewpoint_exploration_v4_unbounded',shape=[400,400],planning_from_odom=np.eye(4).tolist(),return_anchor=[399,0,1])
 def tearDown(self):self.tmp.cleanup()
 def packet(self,n,snapshot):
  p=self.root/f'raw{n}.npz';receipt=dict(headers={'depth':{'stamp_ns':15},'scan':{'stamp_ns':25}},historical_tf={'world_from_camera':{'matrix':trans(.2).tolist()},'odom_from_base':{'matrix':trans(.1).tolist()},'odom_from_laser':{'matrix':trans(.3).tolist()}});meta=dict(frame_id=str(n),geometry_world_frame='nso_pair_map',rtabmap_graph_snapshot=snapshot,scope='real_observed_joint_input',planning_from_odom=np.eye(4).tolist());np.savez(p,receipt_json=np.asarray(json.dumps(receipt)),joint_metadata_json=np.asarray(json.dumps(meta)),world_from_camera=trans(999),world_from_laser=trans(999),base_odom_pose=np.array([999,0,0]),joint_belief_grid=np.zeros((400,400),np.int8),depth=np.ones((1,1)));return p
 def test_geometry_uses_raw_not_already_corrected_and_sensor_stamps(self):
  p=self.packet(0,graph('a',2))
  with np.load(p)as z:u,r=corrected_geometry(z,graph('a',2))
  self.assertAlmostEqual(u['world_from_camera'][0,3],2.2);self.assertAlmostEqual(u['world_from_laser'][0,3],3.3);self.assertAlmostEqual(u['base_odom_pose'][0],2.1)
  self.assertEqual(r['rgbd']['node_id'],1);self.assertEqual(r['scan']['node_id'],2)
 def test_replay_preserves_outcomes_semantics_and_never_rechooses_history(self):
  s=RtabmapViewpointSession(self.c,replay_directory=self.root/'derived',session_factory=FakeSession)
  p0=self.packet(0,graph('a',0));h0=hashlib.sha256(p0.read_bytes()).hexdigest();s.observe_viewpoint_result(p0,0);t=s.next_viewpoint()['target'];self.assertEqual(t['target_world_frame'],'nso_pair_map');p1=self.packet(1,graph('a',0));s.observe_viewpoint_result(p1,1,4,'real_abort')
  self.assertEqual(len(s.inner.waypoint_outcomes),1);self.assertEqual(s.inner.calls,0);self.assertEqual(s.inner.inferences,1)
  self.assertTrue(np.all(s.inner.observations[0]['grid']==-1));self.assertTrue(np.all(s.inner.observations[1]['grid']==0))
  s.next_viewpoint(graph_snapshot=graph('b',2),current_navigation_grid=np.ones((400,400),np.int8))
  self.assertEqual(s.inner.calls,1);self.assertEqual(s.inner.inferences,0);self.assertEqual(s.inner.waypoint_outcomes[0]['ros_status'],4);self.assertEqual(s.inner.waypoint_outcomes[0]['reason'],'real_abort');self.assertEqual(len(s.records),2);self.assertEqual(s.inner.observations[0]['labels'][0,0],2);self.assertEqual(s.inner.observations[0]['owner'][0,0],5);self.assertEqual(hashlib.sha256(p0.read_bytes()).hexdigest(),h0)
 def test_missing_new_grid_rejected_not_old_map_silently_reused(self):
  s=RtabmapViewpointSession(self.c,replay_directory=self.root/'derived',session_factory=FakeSession);s.observe_viewpoint_result(self.packet(0,graph('a',0)),0)
  with self.assertRaises(ValueError):s.next_viewpoint(graph_snapshot=graph('b',2))
 def test_count_guard_and_legacy_prefix(self):
  with self.assertRaises(ValueError):assert_conserved({'attempts':1},{'attempts':0})
  with self.assertRaises(ValueError):RtabmapViewpointSession(self.c,session_factory=FakeSession,shared_observed_prefix=[1])
 def test_semantic_metadata_and_frontend_reused_without_reload(self):
  import types
  class SemanticFake(FakeSession):
   loads=0
   def __init__(self,c,**kwargs):
    super().__init__(c,**kwargs);self.semantic_configuration=deepcopy(kwargs.get('semantic_configuration'))
    self.backend=types.SimpleNamespace(capabilities={'semantic':'native labels'},scenes=[{'plans':0}]);self.components=types.SimpleNamespace(capabilities={})
    if kwargs.get('semantic_configuration') is not None:
     SemanticFake.loads+=1;self.semantic_frontend=object()
  s=RtabmapViewpointSession(self.c,replay_directory=self.root/'semantic-derived',session_factory=SemanticFake,semantic_configuration={'prior_profile':'real_scene_structural_v1'})
  original_frontend=s.frontend;s.observe_viewpoint_result(self.packet(0,graph('a',0)),0)
  s.next_viewpoint(graph_snapshot=graph('b',2),current_navigation_grid=np.zeros((400,400),np.int8))
  self.assertEqual(SemanticFake.loads,1);self.assertIs(s.inner.semantic_frontend,original_frontend)
  self.assertEqual(s.inner.semantic_configuration['prior_profile'],'real_scene_structural_v1')
  self.assertIn('real prompt',s.inner.backend.capabilities['semantic']);self.assertEqual(s.inner.backend.capabilities['semantic'],s.inner.components.capabilities['semantic'])
 def test_anchor_original_grid_convention(self):self.assertEqual(projected_anchor(self.c,[.1,.1,0]),[399,0,1])

if __name__=='__main__':unittest.main()
