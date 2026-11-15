import sys,importlib.util,unittest
from pathlib import Path
import numpy as np
sys.path.insert(0,'/home/ouc/nso_robot/workspaces/nso_hybrid_20261029/nso_bridge')
from nso_bridge.joint_engine import VENDOR
from nso_bridge.exact_visibility_counts import ExactVisibilityCounts
from nso_bridge.exact_visibility_native import batch_visibility_counts
from utils.grid_geometry import ray_templates
p=Path(__file__).parents[1]/'nso_bridge/exact_visibility_boundary_v1.py';s=importlib.util.spec_from_file_location('boundary',p);m=importlib.util.module_from_spec(s);s.loader.exec_module(m)
class Tests(unittest.TestCase):
 def test_random_full_tables(self):
  for seed in range(16):
   rng=np.random.default_rng(seed);shape=(int(rng.integers(8,25)),int(rng.integers(8,25)))
   occupied=rng.random(shape)<(.02 if seed%2 else 0)
   query=rng.random(shape)<(.8 if seed%3 else 1.1)
   camera=rng.random(shape)<(.7 if seed%3 else 1.1)
   args=(occupied,query,camera,int(rng.integers(2,7)),int(rng.integers(2,6)),101.3)
   a=ExactVisibilityCounts(*args);b=m.ExactBoundaryVisibilityCounts(*args)
   for key in a.tables:self.assertTrue(np.array_equal(a.tables[key][0],b.tables[key][0]),(seed,key))
 def test_uniform_boundary_against_original_native_every_center(self):
  for shape in [(5,7),(15,16),(30,33)]:
   occupied=np.zeros(shape,bool);query=np.ones(shape,bool)
   b=m.ExactBoundaryVisibilityCounts(occupied,query,query,6,4,101.3)
   positions=np.argwhere(np.ones(shape,bool))
   for (kind,h),(counts,_,radius,angle,_) in b.tables.items():
    offsets,valid=ray_templates(radius,angle,h)
    expected=batch_visibility_counts(occupied,query,positions,offsets,valid).reshape(shape)
    self.assertTrue(np.array_equal(counts,expected),(shape,kind,h))
 def test_zero_query_with_blockers_and_sparse_query(self):
  rng=np.random.default_rng(100)
  for density in [0.,.001,.01]:
   shape=(25,29);occupied=rng.random(shape)<.2;query=rng.random(shape)<density
   args=(occupied,query,query,6,4,101.3);a=ExactVisibilityCounts(*args);b=m.ExactBoundaryVisibilityCounts(*args)
   for key in a.tables:self.assertTrue(np.array_equal(a.tables[key][0],b.tables[key][0]),(density,key))
   if density==0:self.assertEqual(sum(x['actual_native_centers'] for x in b.compute_audit),0)
 def test_optin_required(self):
  with self.assertRaises(ValueError):m.install('other')
if __name__=='__main__':unittest.main()
