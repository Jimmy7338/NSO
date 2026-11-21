"""Independent exactness/process isolation tests; no candidate score access."""
from pathlib import Path
import sys,unittest,multiprocessing as mp,os
from unittest.mock import patch
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import fork_forecast as f

def terminate_child(*args,**kwargs):os._exit(7)
def frame(z,provenance='acquired',x=0.):
 k=np.array([[30.,0,15.5],[0,30.,11.5],[0,0,1.]]);p=np.eye(4);p[0,3]=x;return f.core.DepthFrame(np.full((24,32),z,np.float32),p,k,provenance)
class ForkTests(unittest.TestCase):
 def setUp(self):
  self.past=[frame(1.)];self.future=[frame(.85,'predicted_hypothesis',.1)];self.points=np.array([[x,y,z]for z in(.85,1.,1.3)for x in(-.15,0,.15,.3)for y in(-.1,.1)]);self.area=np.linspace(.1,.8,len(self.points))
 def test_exact_plane_vs_complete_replay(self):
  snapshot=f.ForkSnapshot(self.past);got=snapshot.forecast(self.future,self.points,self.area);expected=f.core.forecast(self.past,self.future,self.points,self.area)
  for key in ['gain','loss','before_distance','after_distance']:np.testing.assert_array_equal(got[key],expected[key])
  self.assertEqual(got['receipt']['after_mesh_hashes']['vertices'],expected['receipt']['after_vertices_sha256']);self.assertEqual(got['receipt']['after_mesh_hashes']['triangles'],expected['receipt']['after_triangles_sha256']);snapshot.verify_parent_unchanged()
 def test_child_changes_volume_but_parent_and_frames_unchanged(self):
  snapshot=f.ForkSnapshot(self.past);before=snapshot.before_hashes;hashes=[f.core.frame_hash(x)for x in self.future];got=snapshot.forecast(self.future,self.points,self.area)
  self.assertNotEqual(before['vertices'],got['receipt']['after_mesh_hashes']['vertices']);self.assertEqual(snapshot.verify_parent_unchanged()['calls'],1);self.assertEqual(hashes,[f.core.frame_hash(x)for x in self.future])
 def test_empty_future_zero_gross_and_loss(self):
  snapshot=f.ForkSnapshot(self.past);got=snapshot.forecast([],self.points,self.area);np.testing.assert_array_equal(got['before_distance'],got['after_distance']);self.assertEqual(got['gain'].sum(),0);self.assertEqual(got['loss'].sum(),0);self.assertEqual(snapshot.before_hashes,got['receipt']['after_mesh_hashes']);snapshot.verify_parent_unchanged()
 def test_child_exception_propagates_without_mutation(self):
  snapshot=f.ForkSnapshot(self.past)
  with patch.object(f,'append_frames',side_effect=ValueError('test-only child failure')):
   with self.assertRaisesRegex(RuntimeError,'test-only child failure'):snapshot.forecast(self.future,self.points,self.area)
  self.assertEqual(snapshot.verify_parent_unchanged()['calls'],0)
 def test_child_abnormal_exit_propagates_and_reaped(self):
  snapshot=f.ForkSnapshot(self.past);children={x.pid for x in mp.active_children()}
  with patch.object(f,'append_frames',side_effect=terminate_child):
   with self.assertRaisesRegex(RuntimeError,'exited without result'):snapshot.forecast(self.future,self.points,self.area)
  self.assertEqual(children,{x.pid for x in mp.active_children()});snapshot.verify_parent_unchanged()
 def test_ten_alternating_children_stable_isolated_reaped(self):
  snapshot=f.ForkSnapshot(self.past);other=[frame(1.3,'predicted_hypothesis',-.15)];seen={};children={x.pid for x in mp.active_children()}
  for i in range(10):
   key=i%2;got=snapshot.forecast(self.future if key==0 else other,self.points,self.area)
   if key in seen:
    for field in ['gain','loss','after_distance']:np.testing.assert_array_equal(got[field],seen[key][field])
    self.assertEqual(got['receipt']['after_mesh_hashes'],seen[key]['receipt']['after_mesh_hashes'])
   else:seen[key]=got
   self.assertEqual(snapshot.verify_parent_unchanged()['calls'],i+1);self.assertEqual(children,{x.pid for x in mp.active_children()})
  self.assertNotEqual(seen[0]['receipt']['after_mesh_hashes'],seen[1]['receipt']['after_mesh_hashes'])
 def test_invalid_inputs_rejected_before_spawn(self):
  snapshot=f.ForkSnapshot(self.past)
  with self.assertRaises(ValueError):snapshot.forecast(self.past,self.points,self.area)
  bad=self.points.copy();bad[0,0]=np.nan
  with self.assertRaises(ValueError):snapshot.forecast(self.future,bad,self.area)
  with self.assertRaises(ValueError):snapshot.forecast(self.future,self.points,-self.area)
  with self.assertRaises(ValueError):snapshot.forecast(self.future,self.points,self.area,before_distance=np.array([0.]))
  self.assertEqual(snapshot.calls,0)
 def test_empty_basis_and_empty_past(self):
  snapshot=f.ForkSnapshot([]);got=snapshot.forecast(self.future,np.empty((0,3)),np.empty(0));self.assertEqual(got['gain'].size,0);snapshot.verify_parent_unchanged()
  got=snapshot.forecast(self.future,self.points,self.area);expected=f.core.forecast([],self.future,self.points,self.area);np.testing.assert_array_equal(got['gain'],expected['gain']);self.assertTrue(np.isinf(got['before_distance']).all());snapshot.verify_parent_unchanged()
 def test_cached_before_distances_exact(self):
  snapshot=f.ForkSnapshot(self.past);a=snapshot.forecast(self.future,self.points,self.area);b=snapshot.forecast(self.future,self.points,self.area,before_distance=snapshot.before_distance(self.points))
  for field in ['gain','loss','before_distance','after_distance']:np.testing.assert_array_equal(a[field],b[field])
  snapshot.verify_parent_unchanged()
if __name__=='__main__':unittest.main()
