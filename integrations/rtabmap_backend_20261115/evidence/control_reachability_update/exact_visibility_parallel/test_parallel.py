import sys,unittest
from pathlib import Path
import numpy as np
W=Path('/home/ouc/nso_robot/workspaces/nso_hybrid_20261029');sys.path.insert(0,str(W/'nso_bridge'));sys.path.insert(0,str(Path(__file__).parent/'candidate'))
from nso_bridge import joint_engine
from nso_bridge.exact_visibility_counts import ExactVisibilityCounts as Original
from exact_visibility_parallel_v1 import ParallelExactVisibilityCounts
class Tests(unittest.TestCase):
 def test_full_tables_and_queries(self):
  rng=np.random.default_rng(20261115)
  for shape,radar,camera,fov in [((13,15),4,3,75),((23,25),6,4,101.346),((5,5),7,3,110),((19,17),4,5,360)]:
   for occupancy in [0.,.1,1.]:
    occ=rng.random(shape)<occupancy;radar_unknown=rng.random(shape)<.4;camera_unknown=rng.random(shape)<.3
    a=Original(occ,radar_unknown,camera_unknown,radar,camera,fov)
    b=ParallelExactVisibilityCounts(occ,radar_unknown,camera_unknown,radar,camera,fov)
    self.assertEqual(list(a.tables),list(b.tables))
    for key in a.tables:
     for i in [0,1,4]:self.assertTrue(np.array_equal(a.tables[key][i],b.tables[key][i]),(shape,occupancy,key,i))
     for cell in np.ndindex(shape):self.assertEqual(a.count(key[0],cell,key[1]),b.count(key[0],cell,key[1]))
 def test_native_inputs_immutable(self):
  occ=np.eye(11,dtype=bool);query=~occ;before=(occ.copy(),query.copy())
  ParallelExactVisibilityCounts(occ,query,query,5,4,75)
  self.assertTrue(np.array_equal(occ,before[0]));self.assertTrue(np.array_equal(query,before[1]))
if __name__=='__main__':unittest.main()
