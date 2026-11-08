import math, tempfile, unittest
from pathlib import Path
import numpy as np
from matched_helpers import select_grid, set_score, packet_grid_evidence
class MatchedTests(unittest.TestCase):
 def test_modes_are_distinct(self):
  base={'score_mode':'S','other':7}
  for m in 'CGS': self.assertEqual(set_score(base,m),{'score_mode':m,'other':7})
  self.assertEqual(base['score_mode'],'S')
  with self.assertRaises(ValueError):set_score(base,'X')
 def test_shared_grid_and_pose_delta(self):
  grid={'initial_actual_base_odom_pose':[1,2,3.1],'floor_odom_z_m':-.12,'return_anchor':[200,200,0]}
  result,delta=select_grid([1.3,2.4,-3.1],99,{'calibration':{'initial_grid':grid}},lambda *a,**k: self.fail('must reuse'))
  self.assertEqual(result,grid);self.assertIsNot(result,grid);self.assertAlmostEqual(delta['xy_m'],.5);self.assertAlmostEqual(delta['yaw_rad'],2*math.pi-6.2)
 def test_grid_packet(self):
  with tempfile.TemporaryDirectory() as d:
   p=Path(d)/'p.npz';np.savez(p,joint_belief_grid=np.array([[-1,0,1],[1,0,1]],dtype=np.int8))
   x=packet_grid_evidence(p);self.assertEqual(x['counts'],{'-1':1,'0':2,'1':3});self.assertEqual(x['shape'],[2,3])
if __name__=='__main__':unittest.main()
