import sys,unittest
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parent.parent))
import parts_audit
import replacement as r
class Tests(unittest.TestCase):
 def test_coarse_exact_old_sampler(self):
  p,a=r.box_quadrature([.2,.3,.4],[.83,.39,1.14],.22);old,w,_=parts_audit.box([.2,.3,.4],[.83,.39,1.14]);np.testing.assert_array_equal(p,old);np.testing.assert_array_equal(a,w)
 def test_both_resolutions_preserve_component_area(self):
  size=np.array([.83,.39,1.14]);expected=2*(size[0]*size[1]+size[0]*size[2]+size[1]*size[2])
  for step in [.22,.06]:self.assertAlmostEqual(r.box_quadrature([0,0,0],size,step)[1].sum(),expected)
 def test_measured_only_can_restore_but_free_stays_zero(self):
  b=np.array([1.,2.,3.]);free=np.array([False,True,False]);native=np.array([True,True,False]);np.testing.assert_array_equal(r.weights(b,free,native),[1,0,0])
 def test_no_category_dependence_and_mass_linearity(self):
  f=np.array([False,True]);k=np.array([True,True]);np.testing.assert_allclose(r.weights([.1,.9],f,k)+r.weights([.4,.6],f,k),r.weights([.5,1.5],f,k))
 def test_no_gain_outside_declared_base(self):
  rng=np.random.default_rng(1003);b=rng.random(20);f=rng.random(20)>.5;k=rng.random(20)>.5;self.assertTrue(np.all(r.weights(b,f,k)<=b))
 def test_invalid_shapes_or_values_fail(self):
  with self.assertRaises(ValueError):r.weights([1,2],[0,1],np.ones(2,bool))
  with self.assertRaises(ValueError):r.weights([np.nan],np.zeros(1,bool),np.ones(1,bool))
  with self.assertRaises(ValueError):r.box_quadrature([0,0,0],[1,-1,1])
if __name__=='__main__':unittest.main()
