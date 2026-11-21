import importlib.util
from pathlib import Path
from types import SimpleNamespace as N
import unittest
import tempfile,json,copy
p=Path(__file__).parents[1]/'runtime/rtabmap_mapping_backend.py'
s=importlib.util.spec_from_file_location('backend',p);m=importlib.util.module_from_spec(s);s.loader.exec_module(m)
class Stamp:
 def __init__(self,n):self.n=n
 def to_nsec(self):return self.n

def header(n=100,frame='nso_loop_map'):return N(stamp=Stamp(n),frame_id=frame,seq=1)
def transform(x=2):return N(translation=N(x=x,y=3.,z=0.),rotation=N(x=0.,y=0.,z=0.,w=1.))
def graph(n=100,x=2):return N(header=header(n),mapToOdom=transform(x),posesId=[1],poses=[N(position=N(x=0.,y=0.,z=0.),orientation=transform().rotation)],links=[],_connection_header={'callerid':'/nso_loop/rtabmap'})
def grid(n=100):return N(header=header(n),info=N(width=2,height=2,resolution=.05,origin=N(position=N(x=0.,y=0.,z=0.),orientation=transform().rotation)),data=[-1,0,50,100],_connection_header={'callerid':'/nso_loop/rtabmap'})
def odom(n):return N(header=header(n,'odom'),child_frame_id='base_link',_connection_header={'callerid':'/nso_csm_odometry_bridge'})
class Tests(unittest.TestCase):
 def test_exact_pair_either_order(self):
  for first in ['grid','graph']:
   a=m.EpochAdapter();self.assertIsNone(a.receive(first,grid() if first=='grid' else graph()))
   r=a.receive('graph' if first=='grid' else 'grid',graph() if first=='grid' else grid())
   self.assertEqual(r[0].data,[-1,0,50,100]);self.assertEqual(r[0].header.frame_id,'nso_pair_map');self.assertEqual(r[0].header.stamp.to_nsec(),100)
   self.assertEqual(r[1].header.frame_id,'nso_loop_map');self.assertEqual(r[2]['graph_source_stamp_ns'],100)
 def test_no_nearest_stamp_pair(self):
  a=m.EpochAdapter();a.receive('grid',grid(100));self.assertIsNone(a.receive('graph',graph(101)));self.assertIsNone(a.odom(odom(200)))
 def test_source_reject(self):
  for key,val in [('frame_id','wrong')]:
   a=m.EpochAdapter();g=grid();setattr(g.header,key,val)
   with self.assertRaises(ValueError):a.receive('grid',g)
  g=graph();g._connection_header['callerid']='/other'
  with self.assertRaises(ValueError):m.EpochAdapter().receive('graph',g)
 def test_held_correction_real_stamp_no_expiry(self):
  a=m.EpochAdapter();a.receive('grid',grid());a.receive('graph',graph());r=a.odom(odom(100000000000))
  self.assertEqual(r[1].stamp.to_nsec(),100000000000);self.assertEqual(r[2],100);self.assertEqual(r[0].translation.x,-2)
  self.assertIsNone(a.odom(odom(99)))
 def test_loop_revision_updates_not_compounds(self):
  a=m.EpochAdapter();a.receive('grid',grid());a.receive('graph',graph());a.receive('grid',grid(200));a.receive('graph',graph(200,10))
  self.assertEqual(a.odom(odom(201))[0].translation.x,-10)
  self.assertIsNone(a.receive('graph',graph(100)))
 def test_inverse_rotation(self):
  t=transform();t.rotation=N(x=0.,y=0.,z=2**-.5,w=2**-.5)
  r=m.inverse(t);self.assertAlmostEqual(r.translation.x,-3);self.assertAlmostEqual(r.translation.y,2)
 def test_original_immutable(self):
  a=m.EpochAdapter();g=grid();h=graph();a.receive('grid',g);r=a.receive('graph',h);r[0].data[0]=0;r[1].posesId.append(9)
  self.assertEqual(g.data[0],-1);self.assertEqual(h.posesId,[1]);self.assertEqual(a.epoch[1].data[0],-1)
 def test_bad_geometry_and_graph(self):
  g=grid();g.data=[0]
  with self.assertRaises(ValueError):m.EpochAdapter().receive('grid',g)
  h=graph();h.mapToOdom.translation.x=float('nan')
  with self.assertRaises(ValueError):m.EpochAdapter().receive('graph',h)
 def test_metadata_preserves_real_time_raw_and_optimized(self):
  a=m.EpochAdapter();h=graph();node=N(id=1,map_id=0,stamp=.00000009,pose=N(position=N(x=7.,y=0.,z=0.),orientation=transform().rotation))
  data=N(header=header(),graph=h,nodes=[node],_connection_header={'callerid':'/nso_loop/rtabmap'})
  derived=a.node_metadata(data);a.receive('grid',grid());out=a.receive('graph',derived)
  snap=a.graph_snapshot(out[1],out[2]);r=snap['optimized_nodes'][0]
  self.assertEqual(r['node_metadata']['raw_odom_pose']['translation'][0],7.)
  self.assertEqual(r['optimized_pose']['translation'][0],0.)
  self.assertEqual(r['node_metadata']['stamp_s'],.00000009)
  self.assertEqual(snap['metadata_coverage'],{'known':1,'total':1})
 def test_missing_old_metadata_explicit(self):
  a=m.EpochAdapter();a.receive('grid',grid());out=a.receive('graph',graph())
  self.assertIsNone(a.graph_snapshot(out[1],out[2])['optimized_nodes'][0]['node_metadata'])
 def test_revision_stable_across_time_changes_with_poses(self):
  hashes=[]
  for ns,x in [(100,2),(200,2),(300,3)]:
   a=m.EpochAdapter();a.receive('grid',grid(ns));out=a.receive('graph',graph(ns,x))
   snap=a.graph_snapshot(out[1],out[2]);hashes.append(snap['graph_revision_sha256'])
   self.assertEqual(snap['map_stamp_ns'],out[0].header.stamp.to_nsec())
   self.assertEqual(snap['schema'],'rtabmap_mapping_backend/1')
   self.assertEqual(len(snap['map_from_odom']),4)
  # Revision is optimized poses+links; mapToOdom changes alone is separately recorded.
  self.assertEqual(hashes[0],hashes[1]);self.assertEqual(hashes[1],hashes[2])
 def test_mapdata_uses_outer_generation_header(self):
  a=m.EpochAdapter();h=graph(0);data=N(header=header(200),graph=h,nodes=[],_connection_header={'callerid':'/nso_loop/rtabmap'})
  h2=a.node_metadata(data);self.assertEqual(h2.header.stamp.to_nsec(),200);self.assertEqual(h.header.stamp.to_nsec(),0)
 def metadata_fixture(self):
  return dict(schema='rtabmap_node_metadata/1',source_database_path='/actual.db',source_epoch='sessionA',
   source_service='/nso_loop/get_map_data2',source_caller='/nso_loop/rtabmap',
   request={'global':True,'optimized':False,'with_images':False,'with_scans':False,'with_user_data':False,'with_grids':False,'with_words':False,'with_global_descriptors':False},
   nodes=[dict(id=1,map_id=0,stamp_s=123.,raw_odom_pose=dict(translation=[7.,8.,0.],rotation_xyzw=[0.,0.,0.,1.]))])
 def test_initial_metadata_real_raw_pose_and_hash(self):
  with tempfile.TemporaryDirectory() as d:
   p=Path(d)/'metadata.json';p.write_text(json.dumps(self.metadata_fixture()))
   a=m.EpochAdapter();r=a.load_initial_metadata(p,'/actual.db','sessionA')
   self.assertEqual(r['node_count'],1);self.assertEqual(len(r['sha256']),64)
   a.receive('grid',grid());out=a.receive('graph',graph());snap=a.graph_snapshot(out[1],out[2])
   self.assertEqual(snap['nodes'][0]['stamp_ns'],123000000000)
   self.assertEqual(snap['nodes'][0]['odom_from_base'][0][3],7.)
   self.assertIn('GetMap2',a.nodes[1]['source'])
 def test_initial_metadata_reject_wrong_provenance_atomically(self):
  for key,value in [('source_epoch','other'),('source_database_path','/wrong.db'),('source_service','/nso_loop/get_map_data'),('source_caller','/other')]:
   with tempfile.TemporaryDirectory() as d:
    data=self.metadata_fixture();data[key]=value;p=Path(d)/'metadata.json';p.write_text(json.dumps(data));a=m.EpochAdapter()
    with self.assertRaises(ValueError):a.load_initial_metadata(p,'/actual.db','sessionA')
    self.assertEqual(a.nodes,{})
 def test_initial_metadata_reject_optimized_or_images(self):
  for key in ['optimized','with_images']:
   with tempfile.TemporaryDirectory() as d:
    data=self.metadata_fixture();data['request'][key]=True;p=Path(d)/'metadata.json';p.write_text(json.dumps(data))
    with self.assertRaises(ValueError):m.EpochAdapter().load_initial_metadata(p,'/actual.db','sessionA')
 def test_initial_metadata_reject_invalid_node_no_partial_import(self):
  with tempfile.TemporaryDirectory() as d:
   data=self.metadata_fixture();bad=copy.deepcopy(data['nodes'][0]);bad['id']=2;bad['raw_odom_pose']['translation'][0]=float('nan');data['nodes'].append(bad)
   p=Path(d)/'metadata.json';p.write_text(json.dumps(data));a=m.EpochAdapter()
   with self.assertRaises(ValueError):a.load_initial_metadata(p,'/actual.db','sessionA')
   self.assertEqual(a.nodes,{})
 def test_recorded_exact_graph_bootstrap(self):
  class SerializedGraph:
   def deserialize(self,raw):self.__dict__.update(graph().__dict__);del self._connection_header
  with tempfile.TemporaryDirectory() as d:
   p=Path(d)/'graph.ros';raw=b'recorded-original-bytes';p.write_bytes(raw)
   data=dict(schema='rtabmap_recorded_epoch_graph/1',source_database_path='/actual.db',source_epoch='sessionA',graph_sha256=m.hashlib.sha256(raw).hexdigest(),grid_data_sha256=m.hashlib.sha256(m.np.asarray(grid().data,dtype=m.np.int8).tobytes()).hexdigest(),graph_stamp_ns=100,actual_recorded_caller='/nso_rtabmap_mapping_backend',original_source_caller='/nso_loop/rtabmap')
   Path(str(p)+'.json').write_text(json.dumps(data))
   a=m.EpochAdapter();a.load_initial_epoch_graph(p,'/actual.db','sessionA',SerializedGraph)
   self.assertIsNone(a.odom(odom(200)));self.assertIsNone(a.receive('grid',grid(101)))
   out=a.receive('grid',grid());self.assertEqual(out[2]['graph_source_topic'],'recorded exact generation')
   self.assertFalse(hasattr(a.graphs[100],'_connection_header'));self.assertEqual(a.odom(odom(200))[1].stamp.to_nsec(),200)
   bad=m.EpochAdapter();bad.load_initial_epoch_graph(p,'/actual.db','sessionA',SerializedGraph);g=grid();g.data[0]=0
   with self.assertRaises(ValueError):bad.receive('grid',g)
   self.assertIsNone(bad.tf_epoch)
   data['graph_sha256']='0'*64;Path(str(p)+'.json').write_text(json.dumps(data))
   with self.assertRaises(ValueError):m.EpochAdapter().load_initial_epoch_graph(p,'/actual.db','sessionA',SerializedGraph)
if __name__=='__main__':unittest.main()
