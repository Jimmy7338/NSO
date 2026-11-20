import unittest
from types import SimpleNamespace
import numpy as np
import open3d as o3d
import measured_novelty as m

def scene():
 mesh=o3d.geometry.TriangleMesh();mesh.vertices=o3d.utility.Vector3dVector([[0,0,0],[2,0,0],[0,2,0]])
 mesh.triangles=o3d.utility.Vector3iVector([[0,1,2]])
 s=o3d.t.geometry.RaycastingScene(nthreads=1);s.add_triangles(o3d.t.geometry.TriangleMesh.from_legacy(mesh));return s

class NoveltyTests(unittest.TestCase):
 def test_seen_unseen_and_closed_boundary(self):
  k,d=m.unmeasured_mask(np.array([[.2,.2,z]for z in [0.,.049,.05,.051,.2]]),scene())
  np.testing.assert_array_equal(k,[False,False,False,True,True])
 def test_empty_mesh_and_points(self):
  k,d=m.unmeasured_mask(np.array([[1,2,3.]]),None);self.assertTrue(k[0]);self.assertTrue(np.isinf(d[0]))
  self.assertEqual(len(m.unmeasured_mask(np.empty((0,3)),scene())[0]),0)
 def test_revisit_zero_unseen_retained(self):
  model=SimpleNamespace(points=np.array([[.2,.2,0],[.3,.3,0],[.2,.2,.2]]),weights=np.array([2.,3.,4.]),audit={})
  m.apply(model,scene());np.testing.assert_array_equal(model.weights,[0,0,4])
 def test_existing_history_zeros_and_posterior_basis_preserved(self):
  model=SimpleNamespace(points=np.array([[.2,.2,.2],[.3,.3,0.]]),weights=np.array([0.,3.]),base_weights=np.array([4.,3.]),history_keep=np.array([False,True]),audit={})
  p=model.points.copy();b=model.base_weights.copy();h=model.history_keep.copy()
  m.apply(model,scene());np.testing.assert_array_equal(model.weights,[0,0]);np.testing.assert_array_equal(model.points,p);np.testing.assert_array_equal(model.base_weights,b);np.testing.assert_array_equal(model.history_keep,h)
 def test_G_S_share_mask_and_linear_mass(self):
  pts=np.array([[.2,.2,0],[.2,.2,.2]])
  g=SimpleNamespace(points=pts,weights=np.array([.5,.5]),audit={});s=SimpleNamespace(points=pts,weights=np.array([.1,.9]),audit={})
  kg,_,_=m.apply(g,scene());ks,_,_=m.apply(s,scene());np.testing.assert_array_equal(kg,ks)
  np.testing.assert_allclose(g.weights,[0,.5]);np.testing.assert_allclose(s.weights,[0,.9])
 def test_route_union_commutes_with_shared_mask(self):
  a=np.array([[1,2,0],[2,1,4.]]) ;k=np.array([True,False,True]);np.testing.assert_array_equal(m.consume(a,k).max(0),m.consume(a.max(0),k))
 def test_removed_area_exact_and_idempotent(self):
  model=SimpleNamespace(points=np.array([[.2,.2,0],[.2,.2,.2]]),weights=np.array([.25,.75]),audit={})
  _,_,r=m.apply(model,scene());self.assertEqual(r['consumed_expected_area_m2'],.25)
  _,_,r=m.apply(model,scene());self.assertEqual(r['consumed_expected_area_m2'],0.)
 def test_reject_invalid(self):
  for p in [np.array([[np.nan,0,0]]),np.zeros((3,2))]:
   with self.assertRaises(ValueError):m.unmeasured_mask(p,None)
  with self.assertRaises(ValueError):m.unmeasured_mask(np.empty((0,3)),None,-1)
  with self.assertRaises(ValueError):m.consume(np.ones(3),np.ones(3))
  with self.assertRaises(ValueError):m.consume(np.array([-1.]),np.ones(1,bool))
 def test_gain_never_increases(self):
  rng=np.random.default_rng(61120);a=rng.random((10,100));k=rng.random(100)>.5;b=m.consume(a,k)
  self.assertTrue(np.all(b<=a));np.testing.assert_allclose(a.sum(1)-b.sum(1),a[:,~k].sum(1))
if __name__=='__main__':unittest.main()
