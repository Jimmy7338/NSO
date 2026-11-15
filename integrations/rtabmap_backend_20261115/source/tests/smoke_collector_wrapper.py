"""Fake ROS transport, actual retained sensor packet, real collector and v4 mapper; no ROS master/actors."""
import os
for k in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']:os.environ[k]='1'
import sys,types,pathlib,tempfile,json,io,hashlib
import numpy as np
W=pathlib.Path('/home/ouc/nso_robot/workspaces/nso_hybrid_20261029');sys.path[:0]=[str(W/'runtime'),str(W/'nso_bridge')]
from nso_bridge.rtabmap_viewpoint_session import RtabmapViewpointSession
from nso_bridge.rtabmap_geometry import corrected_geometry
P=pathlib.Path('/home/ouc/nso_robot/reports/nso-hybrid-20261115/NSO_surroundings_20261115_07');config=json.loads((P/'joint_configuration.json').read_text());raw=next((P/'observed').glob('*.npz'))
with np.load(raw,allow_pickle=False)as z:arrays={k:z[k].copy()for k in z.files}
receipt=json.loads(str(arrays['receipt_json'].item()));stamp=receipt['headers']['depth']['stamp_ns'];raw_base=np.array(receipt['historical_tf']['odom_from_base']['matrix']);C=np.eye(4);C[0,3]=2;T=C@raw_base
class Stamp:
 def __init__(self,sec=0,nsec=0):self.ns=int(sec)*10**9+int(nsec)
 def to_nsec(self):return self.ns
 def to_sec(self):return self.ns/1e9
 @classmethod
 def now(cls):return cls(0,stamp+100000000)
class Sub:
 def __init__(self,*a,**k):pass
 def unregister(self):pass
fake=types.ModuleType('rospy');fake.Subscriber=Sub;fake.Time=Stamp;fake.Duration=lambda x:x;sys.modules['rospy']=fake
nav=types.ModuleType('nav_msgs');msg=types.ModuleType('nav_msgs.msg');msg.OccupancyGrid=object;nav.msg=msg;sys.modules['nav_msgs']=nav;sys.modules['nav_msgs.msg']=msg
from joint_observation import JointObservationCollector
ns=types.SimpleNamespace
def tf(matrix,parent,child):
 yaw=np.arctan2(matrix[1,0],matrix[0,0]);return ns(header=ns(stamp=Stamp(0,stamp),frame_id=parent,seq=0),child_frame_id=child,transform=ns(translation=ns(x=float(matrix[0,3]),y=float(matrix[1,3]),z=float(matrix[2,3])),rotation=ns(x=0.,y=0.,z=float(np.sin(yaw/2)),w=float(np.cos(yaw/2)))))
class Buffer:
 def lookup_transform(self,parent,child,*a):return tf(raw_base,'odom','base_link')if parent=='odom'else tf(C,'nso_pair_map','odom')
class Map:
 def __init__(self):
  self.header=ns(stamp=Stamp(0,stamp),frame_id='nso_pair_map',seq=1);self._connection_header={'callerid':'/fake_transport_actual_grid'};self.info=ns(width=24,height=24,resolution=.2,map_load_time=Stamp(),origin=ns(position=ns(x=float(T[0,3]-2.4),y=float(T[1,3]-2.4),z=0.),orientation=ns(x=0.,y=0.,z=0.,w=1.)));self.data=[0]*(24*24)
 def serialize(self,b):b.write(b'fake ROS transport only; no new experimental map')
with tempfile.TemporaryDirectory()as td:
 td=pathlib.Path(td);snapshot=dict(schema='fake_transport_test_only',graph_revision_sha256='collector-test-revision',map_stamp_ns=stamp,graph_source_stamp_ns=stamp,map_frame='nso_loop_map',alias_frame='nso_pair_map',map_from_odom=C.tolist(),nodes=[dict(id=1,stamp_ns=stamp,odom_from_base=raw_base.tolist(),map_from_base=T.tolist())]);sp=td/'graph.json';sp.write_text(json.dumps(snapshot));source_hash=hashlib.sha256(raw.read_bytes()).hexdigest()
 collector=JointObservationCollector(Buffer(),dict(shape=config['configuration']['shape'],resolution_m=.2,planning_from_odom=config['configuration']['planning_from_odom']),topic='/test/map',publisher='/fake_transport_actual_grid',map_frame='nso_pair_map',pixel_stride=config['calibration']['pixel_stride'],apply_approved_self_mask=False,geometry_pose_backend='rtabmap_optimized_replay_v1',graph_snapshot_path=str(sp));collector._receive(Map());path=collector.capture(td/'capture',raw)
 with np.load(path,allow_pickle=False)as z:
  m=json.loads(str(z['joint_metadata_json'].item()));assert m['geometry_world_frame']=='nso_pair_map';assert np.isclose(z['base_odom_pose'][0],T[0,3]);updates,_=corrected_geometry(z,snapshot);assert np.allclose(updates['world_from_camera'],z['world_from_camera']);assert np.allclose(z['rgb_raw'],arrays['rgb_raw'])
 session=RtabmapViewpointSession(config['configuration'],calibration=config['calibration'],semantic_configuration=None,execution_allowed=False,replay_directory=td/'replay');r=session.observe_viewpoint_result(path,0);assert session.summary()['waypoint_attempts']==0;assert session.summary()['waypoint_ROS_successes']==0
 assert hashlib.sha256(raw.read_bytes()).hexdigest()==source_hash
 print(json.dumps(dict(passed=True,collector_real_code=True,wrapper_real_code=True,mapper_original_v4=True,ROS_transport='fake only',retained_actual_sensor_packet=True,pixels_unchanged=True,RTAB_base_translation_applied_once_m=2,bootstrap_accepted=r['accepted'],paid_attempts=0,chooser_calls=0,motion=False,experimental_result=False)))
