import importlib.util,sys,unittest
from pathlib import Path
from types import SimpleNamespace as N
import numpy as np
W=Path('/home/ouc/nso_robot/workspaces/nso_hybrid_20261029')
sys.path.insert(0,str(W/'nso_bridge'))
from nso_bridge.joint_engine import VENDOR
from env.virtual3d import VirtualConfig
from nso_bridge import exploration_joint_policy as original
p=Path(__file__).parents[1]/'nso_bridge/exact_candidate_batch_v1.py'
s=importlib.util.spec_from_file_location('batch',p);batch=importlib.util.module_from_spec(s);s.loader.exec_module(batch)

def equal(a,b):
 if isinstance(a,np.ndarray) or isinstance(b,np.ndarray):return np.array_equal(a,b)
 if isinstance(a,dict):return a.keys()==b.keys() and all(equal(a[k],b[k]) for k in a)
 if isinstance(a,(list,tuple)):return type(a)==type(b) and len(a)==len(b) and all(equal(x,y) for x,y in zip(a,b))
 return a==b

class Tests(unittest.TestCase):
 def test_random_complete_output(self):
  candidate=batch.build_pool(original.generate_options,original.__dict__)
  for seed in range(16):
   rng=np.random.default_rng(seed);shape=(int(rng.integers(12,24)),int(rng.integers(12,24)))
   belief=rng.choice([-1,0,1],shape,p=[.55,.4,.05]).astype(np.int8);current=(shape[0]//2,shape[1]//2);belief[current]=0
   m=N(belief=belief,shape=shape,camera_seen=rng.random(shape)<.2,keyframes=[],config=VirtualConfig(max_depth_m=1.2,robot_radius_m=.2),quality_evidence=lambda **k:None)
   for profile in ['ros_viewpoint_exploration_v4_unbounded',None]:
    kw=dict(coverage_strategy='total_diverse',admission_profile=profile,attempted_camera_mask=rng.random(shape)<.1)
    a=original.generate_options(m,current,seed%4,None if profile else 20,(*current,seed%4),**kw)
    b=candidate(m,current,seed%4,None if profile else 20,(*current,seed%4),**kw)
    self.assertTrue(equal(a,b),(seed,profile))
 def test_assets_full_output(self):
  ns=dict(original.__dict__)
  points=np.array([[1.5,1.4,.4],[1.5,1.6,.7],[1.5,1.8,.5]])
  asset=dict(front_axis=np.array([1.,0.]),back_axis=np.array([-1.,0.]),side_axis=np.array([0.,1.]),rear_boundary_xy=np.array([1.5,1.6]),measured_width_m=.6,measured_depth_m=.5,measured_height_m=.7,points=points,support_points=points,group=1,observed_low=points.min(0),observed_high=points.max(0),aabb_center=points.mean(0))
  ns['_geometry_assets']=lambda *a:[asset]
  import types
  ref=types.FunctionType(original.generate_options.__code__,ns,argdefs=original.generate_options.__defaults__);ref.__kwdefaults__=original.generate_options.__kwdefaults__
  candidate=batch.build_pool(original.generate_options,ns)
  for seed in range(5):
   belief=np.full((22,22),-1,np.int8);belief[9:12,6:10]=0;belief[7,7]=1
   m=N(belief=belief,shape=belief.shape,camera_seen=np.zeros_like(belief,bool),keyframes=[],config=VirtualConfig(max_depth_m=1.2,robot_radius_m=.2),quality_evidence=lambda **k:None)
   kw=dict(coverage_strategy='total_diverse',admission_profile='ros_viewpoint_exploration_v4_unbounded')
   self.assertTrue(equal(ref(m,(10,8),seed%4,None,(10,8,0),**kw),candidate(m,(10,8),seed%4,None,(10,8,0),**kw)))
 def test_opt_in_reject(self):
  with self.assertRaises(ValueError):batch.install('unknown')
if __name__=='__main__':unittest.main()
