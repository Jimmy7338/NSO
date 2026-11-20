import unittest
from types import SimpleNamespace
import numpy as np
import open3d as o3d
import render_observed_free as R
K=np.array([[100.,0,1],[0,100.,1],[0,0,1.]])
P=np.eye(4)
def history(depth):return SimpleNamespace(depth_m=np.full((3,3),depth,np.float32),intrinsic=K,world_from_camera=P)
def plane(z):
 mesh=o3d.geometry.TriangleMesh();mesh.vertices=o3d.utility.Vector3dVector([[-1,-1,z],[1,-1,z],[1,1,z],[-1,1,z]])
 mesh.triangles=o3d.utility.Vector3iVector([[0,1,2],[0,2,3]]);return mesh
def scene(*depths):
 mesh=o3d.geometry.TriangleMesh()
 for d in depths:mesh+=plane(d)
 s=o3d.t.geometry.RaycastingScene(nthreads=1);s.add_triangles(o3d.t.geometry.TriangleMesh.from_legacy(mesh));return s
class TestRender(unittest.TestCase):
 def render(self,hyp,frames,measured=None):return R.corrected_depth(hyp,measured,P,K,3,3,frames)
 def test_no_history_retains_first(self):
  d,a=self.render(scene(1,3),[]);np.testing.assert_allclose(d,1);self.assertEqual(a['strict_free_first_hits'],0)
 def test_free_foreground_skipped(self):
  d,a=self.render(scene(1,3),[history(3)]);np.testing.assert_allclose(d,3);self.assertEqual(a['strict_free_first_hits'],9)
 def test_measured_support_vetoes_free(self):
  d,a=self.render(scene(1,3),[history(3),history(1)]);np.testing.assert_allclose(d,1);self.assertEqual(a['strict_free_first_hits'],0)
 def test_native_is_never_erased(self):
  d,a=self.render(scene(1,3),[history(3)],scene(2));np.testing.assert_allclose(d,2);self.assertEqual(a['measured_scene_wins'],9)
 def test_all_free_hypothesis_is_no_hit(self):
  d,a=self.render(scene(1),[history(3)]);np.testing.assert_array_equal(d,0)
 def test_invalid_measured_depth_not_free(self):
  d,a=self.render(scene(1,3),[history(0)]);np.testing.assert_allclose(d,1)
 def test_sensor_near_plane_contract(self):
  d,_=self.render(scene(.17),[]);np.testing.assert_allclose(d,.17);d,_=self.render(scene(.14),[]);np.testing.assert_array_equal(d,0)
 def test_fitted_layout_includes_exact_near_boundary(self):
  d,_=self.render(scene(.15),[]);self.assertTrue(np.all(d>0));np.testing.assert_allclose(d,.15)
 def test_empty_listing_cannot_reveal(self):
  underlying=scene(1)
  class Empty:
   def cast_rays(self,*a,**kw):return underlying.cast_rays(*a,**kw)
   def list_intersections(self,*a,**kw):return {'ray_ids':o3d.core.Tensor([],dtype=o3d.core.Dtype.UInt32),'t_hit':o3d.core.Tensor([],dtype=o3d.core.Dtype.Float32)}
  d,a=self.render(Empty(),[history(3)]);np.testing.assert_allclose(d,1);self.assertEqual(a['empty_listing_fallbacks'],9)
if __name__=='__main__':unittest.main()
