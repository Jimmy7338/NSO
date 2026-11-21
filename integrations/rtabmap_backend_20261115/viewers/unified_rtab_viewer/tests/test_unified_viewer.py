import sys,pathlib,json,math,unittest,tempfile,types,numpy as np
P=pathlib.Path(__file__).parents[1];sys.path.insert(0,str(P))
from unified_viewer import Bundle,FRAME,project_semantics,cloud_message,point,age
from full_history_relay import build_request
S=types.SimpleNamespace
def stamp(t):return S(to_sec=lambda:t,to_nsec=lambda:int(t*1e9))
def transform(x=0.,y=0.,yaw=0.):return S(translation=S(x=x,y=y,z=0.),rotation=S(x=0.,y=0.,z=math.sin(yaw/2),w=math.cos(yaw/2)))
class TestUnified(unittest.TestCase):
 def test_exact_global_history_request_no_robot(self):
  r=build_request();self.assertTrue(r.global_);self.assertTrue(r.optimized);self.assertFalse(r.graphOnly)
 def test_semantic_polygon_same_map_frame(self):
  source=dict(frame='odom',stamp=20.,objects=[dict(x=1.,y=2.,polygon=[[1.,2.],[2.,2.]])]);r=project_semantics(source,[10,20,math.pi/2],10)
  self.assertEqual(r['frame'],FRAME);self.assertAlmostEqual(r['objects'][0]['x'],8.);self.assertEqual(r['objects'][0]['polygon'],[[8.,21.],[8.,22.]])
  self.assertEqual(source['objects'][0]['x'],1.)
  with self.assertRaises(ValueError):project_semantics(dict(source,frame='map'),[0,0,0],10)
  with self.assertRaises(ValueError):project_semantics(source,[0,0,0],30)
 def test_graph_optimized_trajectory_not_raw_upstream_and_robot_same_correction(self):
  with tempfile.TemporaryDirectory() as path:
   b=Bundle('epoch',10,path+'/missing',path+'/none');b.graph_msg(S(header=S(frame_id=FRAME,stamp=stamp(20)),posesId=[2,1],poses=[S(position=S(x=22.,y=32.)),S(position=S(x=21.,y=31.))],mapToOdom=transform(10,20)))
   b.odom=dict(x=1.,y=2.,yaw=0.,stamp=20.)
   r=b.snapshot(dict(map={'frame':'BAD'},trajectory=[[999,999]],pose={'frame':'BAD'}))
   self.assertEqual(r['trajectory'],[[21.,31.],[22.,32.]])
   self.assertEqual(r['pose']['frame'],FRAME);self.assertEqual(r['pose']['x'],11.);self.assertEqual(r['pose']['y'],22.)
   self.assertIsNone(r['map']);self.assertNotIn([999,999],r['trajectory'])
 def test_epoch_and_frame_reject_without_relabel(self):
  with tempfile.TemporaryDirectory() as p:
   b=Bundle('epoch',10,p+'/missing',p+'/none')
   for frame,t in [('nso_pair_map',20),(FRAME,9)]:
    with self.assertRaises(ValueError):b.check(S(header=S(frame_id=frame,stamp=stamp(t))),FRAME)
 def test_cloud_uniformly_samples_full_array_not_first_prefix(self):
  data=np.zeros(100,dtype=[('x','<f4'),('y','<f4'),('z','<f4'),('rgb','<u4')]);data['x']=np.arange(100);data['z']=1;data['rgb']=0xFF0000
  fields=[S(name=k,offset=i*4,datatype=7 if k!='rgb' else 6) for i,k in enumerate(('x','y','z','rgb'))]
  m=S(header=S(frame_id=FRAME,stamp=stamp(20)),fields=fields,is_bigendian=False,height=1,width=100,point_step=16,row_step=1600,data=data.tobytes());r=cloud_message(m,maximum=10)
  self.assertEqual(r['point_count'],10);self.assertEqual(r['positions'][-3],90.);self.assertEqual(r['colors'][:3],[1.,0.,0.]);self.assertEqual(r['source_point_count'],100)
 def test_map_payload_frame_origin_and_dimensions(self):
  with tempfile.TemporaryDirectory() as p:
   b=Bundle('epoch',10,p+'/missing',p+'/none');m=S(header=S(frame_id=FRAME,stamp=stamp(20)),info=S(height=2,width=2,resolution=.05,origin=S(position=S(x=-1.,y=-2.),orientation=S(x=0.,y=0.,z=0.,w=1.))),data=[-1,0,100,0]);b.grid(m)
   self.assertEqual(b.map['source_bundle'],'unified_current_session_RTAB');self.assertEqual(b.map['frame'],FRAME);self.assertTrue(b.png.startswith(b'\x89PNG'))
 def test_future_stamp_not_fresh(self):self.assertFalse(age(21,20,10)['fresh'])
if __name__=='__main__':unittest.main()
