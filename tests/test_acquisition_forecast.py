import unittest
import numpy as np
from nso import acquisition_forecast as f
class Tests(unittest.TestCase):
 def fixture(self):
  k=np.array([[30.,0,15.5],[0,30.,11.5],[0,0,1.]]);p=np.eye(4);d=np.full((24,32),1.,np.float32)
  return f.DepthFrame(d,p,k,'acquired')
 def test_repeated_same_depth_recovers_nothing_new(self):
  p=self.fixture();future=f.DepthFrame(p.depth_m.copy(),p.world_from_camera.copy(),p.intrinsic.copy(),'predicted_hypothesis')
  r=f.forecast([p],[future],np.array([[0,0,1.],[.1,.1,1.]]),np.ones(2));self.assertEqual(r['receipt']['predicted_gross_new_m2'],0.)
 def test_no_future_exactly_zero(self):
  r=f.forecast([self.fixture()],[],np.array([[0,0,1.]]),np.ones(1));self.assertEqual(r['receipt']['predicted_gross_new_m2'],0.);self.assertEqual(r['receipt']['predicted_lost_m2'],0.)
 def test_no_acquired_map_then_surface_can_appear(self):
  p=self.fixture();future=f.DepthFrame(p.depth_m,p.world_from_camera,p.intrinsic,'predicted_hypothesis');r=f.forecast([],[future],np.array([[0,0,1.]]),np.ones(1));self.assertEqual(r['receipt']['predicted_gross_new_m2'],1.)
 def test_real_future_frame_provenance_rejected(self):
  with self.assertRaises(ValueError):f.forecast([self.fixture()],[self.fixture()],np.zeros((1,3)),np.ones(1))
 def test_renderer_matches_original_depth_range(self):
  from types import SimpleNamespace
  class Scene:
   def cast_rays(self,*args,**kwargs):return {'t_hit':SimpleNamespace(numpy=lambda:np.array([.149,.15,.2,8.,8.001],np.float32))}
  r=f.render(Scene(),[np.eye(4)],np.eye(3),5,1)[0]
  np.testing.assert_array_equal(r.depth_m,np.array([[0,0,.2,8.,0]],np.float32))
 def test_invalid_points_or_mass_rejected(self):
  with self.assertRaises(ValueError):f.forecast([],[],np.array([[np.nan,0,0]]),np.ones(1))
  with self.assertRaises(ValueError):f.forecast([],[],np.zeros((1,3)),np.array([-1.]))
 def test_zero_basis_and_empty_map(self):
  r=f.forecast([],[],np.empty((0,3)),np.empty(0));self.assertEqual(r['receipt']['predicted_gross_new_m2'],0)
 def test_prior_mass_linear_same_geometry(self):
  p=self.fixture();future=f.DepthFrame(p.depth_m,p.world_from_camera,p.intrinsic,'predicted_hypothesis');r=f.forecast([],[future],np.array([[0,0,1.]]),np.array([.25]));self.assertEqual(r['receipt']['predicted_gross_new_m2'],.25)

if __name__=="__main__":unittest.main()
