import os
for name in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ[name]='1'
from pathlib import Path
from types import SimpleNamespace
import sys,unittest
SOURCE=Path('/home/ouc/nso_robot/analyses/reproduction_20261104/simulation_publication_stage/experiments/unknown_exploration_20261104/source')
sys.path.insert(0,str(SOURCE))
import numpy as np
import open3d as o3d
import full_posterior_forecast as F
from env.virtual3d import camera_pose,VirtualConfig

def fixture(**limits):
 c=VirtualConfig(width_px=40,height_px=30,depth_sigma_m=0,dropout=0,camera_height_m=.8,max_depth_m=8)
 mesh=o3d.geometry.TriangleMesh.create_box(1,.65,1.4).translate((1.5,2.8,0));scene=F.CORE.ray_scene(mesh)
 K=np.array([[20.,0,19.5],[0,20.,14.5],[0,0,1.]])
 pose=camera_pose((25,10),0,c,30)
 future=F.CORE.render(scene,[pose],K,40,30)[0]
 frame=F.CORE.DepthFrame(future.depth_m,future.world_from_camera,K,'acquired')
 native=F.CORE.integrate([frame]);points=np.array([[2,2.8,.7],[2,3.45,.7],[1.5,3.125,.7],[2.5,3.125,.7],[2,3.125,0],[2,3.125,1.4]])
 area=np.array([1.4,1.4,.91,.91,.65,.65]);allpoints=np.tile(points,(3,1));areas=np.tile(area,3)
 history=F.RENDER.REPAIR.classify_history(allpoints,[frame]);slices=[(0,i,0,i*6,(i+1)*6)for i in range(3)]
 data=dict(points=allpoints,areas=areas,hypothesis_ids=np.repeat(np.arange(3),6),history=history,slices=slices)
 item=dict(variants={'uncertainty':{'particles':[dict(depth_m=.65,lower_m=2.8)]*3}},geometry={(.65,2.8):[dict(mesh=mesh,scene=scene)]})
 bank=SimpleNamespace(objects=[item],materialize=lambda mode:data)
 mapper=SimpleNamespace(config=c,shape=(30,40),keyframes=[frame],frames=1,mesh=lambda:native)
 models={}
 for arm,mass in [('G',[.3,.7,1e-18]),('S',[.8,.2,1e-20])]:
  pm=np.repeat(mass,6);base=areas*pm;keep=~(history['strict_free']|history['measured_surface'])
  models[arm]=SimpleNamespace(bank=bank,points=allpoints,hypothesis_ids=data['hypothesis_ids'],base_weights=base,weights=base*keep,history_keep=keep,objects=[dict(joint_probability_mass=[[m,0]for m in mass])],audit={'extent_mode':'uncertainty'})
 engine=F.FullPosteriorForecast(mapper,models,[frame],**limits)
 return engine

class TestForecast(unittest.TestCase):
 def test_full_positive_mass_and_exact_sequence_cache(self):
  e=fixture();r=e.acquisition([(17,18,3)])
  self.assertEqual(len(r['terms']),3)
  self.assertEqual(r['terms'][2]['posterior_mass']['G'],1e-18)
  self.assertEqual(r['counts']['after_volume_fusions'],1)
  self.assertEqual(r['counts']['after_volume_cache_hits'],2)
  self.assertEqual(r['counts']['rendered_depth_frames'],1)
  self.assertTrue(e.assert_unchanged()['no_positive_probability_pruning'])
 def test_uncached_core_identity(self):
  e=fixture();states=[(17,18,3),(17,17,3)];r=e.acquisition(states)
  key,shape=e._geometry(0,0,0);fs=[e._future(key,shape['scene'],s)[0]for s in states]
  raw=F.CORE.forecast(e.past,fs,e.data['points'][:6],e.data['areas'][:6]*(~e.data['history']['strict_free'][:6]))
  for arm in ('G','S'):
   expected=sum(m['posterior_mass'][arm]for m in r['terms'])*float(raw['gain'].sum())
   self.assertAlmostEqual(r['gross_m2'][arm],expected,places=14)
 def test_fork_and_fresh_replay_are_exact(self):
  a=fixture(backend='fork');b=fixture(backend='replay');states=[(17,18,3),(17,17,3)]
  self.assertEqual(a.acquisition(states)['gross_m2'],b.acquisition(states)['gross_m2']);self.assertEqual(a.acquisition(states)['lost_m2'],b.acquisition(states)['lost_m2']);a.assert_unchanged();b.assert_unchanged()
 def test_cache_eviction_is_not_an_approximation(self):
  a=fixture(depth_cache_limit=1,after_cache_limit=1);b=fixture(depth_cache_limit=100,after_cache_limit=100)
  for states in [[(17,18,3),(17,17,3)],[(20,4,1)],[(17,18,3),(17,16,3)]]:
   self.assertEqual(a.acquisition(states)['gross_m2'],b.acquisition(states)['gross_m2'])
   self.assertEqual(a.acquisition(states)['lost_m2'],b.acquisition(states)['lost_m2'])
 def test_native_ray_cache_is_bit_exact(self):
  e=fixture();key,shape=e._geometry(0,0,0)
  for s in [(17,18,3),(20,4,1),(17,17,3)]:
   cached=e._future(key,shape['scene'],s)[0]
   original,_=F.RENDER.corrected_depth(shape['scene'],e.before_ray,cached.world_from_camera,cached.intrinsic,e.mapper.config.width_px,e.mapper.config.height_px,e.history)
   np.testing.assert_array_equal(cached.depth_m,original)
  # A second hypothesis geometry key still uses identical native ray queries.
  e._future(key+'different_hypothesis',shape['scene'],(17,18,3))
  self.assertGreaterEqual(e.counts['native_ray_cache_hits'],1);self.assertEqual(e.counts['native_ray_casts'],3)
 def test_same_geometry_different_poses_are_separate_depth_entries(self):
  e=fixture();key,shape=e._geometry(0,0,0)
  a=e._future(key,shape['scene'],(17,18,3));b=e._future(key,shape['scene'],(17,17,3))
  self.assertNotEqual(a[1],b[1]);self.assertEqual(e.counts['rendered_depth_frames'],2)
 def test_same_depth_different_pose_cannot_collide_in_volume_cache(self):
  e=fixture();f=e.past[0];pose=f.world_from_camera.copy();pose[0,3]+=.2
  a=F.CORE.DepthFrame(f.depth_m, f.world_from_camera, f.intrinsic, 'predicted_hypothesis')
  b=F.CORE.DepthFrame(f.depth_m, pose, f.intrinsic, 'predicted_hypothesis')
  _,ka=e._after([(a,F.CORE.frame_hash(a),{})]);_,kb=e._after([(b,F.CORE.frame_hash(b),{})])
  self.assertNotEqual(ka,kb);self.assertEqual(e.counts['after_volume_fusions'],2)
 def test_duplicate_frames_are_not_dropped(self):
  e=fixture();key,shape=e._geometry(0,0,0);f=e._future(key,shape['scene'],(17,18,3))
  _,one=e._after([f]);_,two=e._after([f,f]);self.assertNotEqual(one,two)
 def test_empty_future_has_zero_new_and_lost(self):
  r=fixture().acquisition([]);self.assertEqual(r['gross_m2'],dict(G=0.,S=0.));self.assertEqual(r['lost_m2'],dict(G=0.,S=0.))
 def test_same_route_reuses_all_terms(self):
  e=fixture();a=e.acquisition([(17,18,3)]);counts=dict(e.counts);b=e.acquisition([(17,18,3)]);self.assertIs(a,b);self.assertEqual(e.counts,counts)
 def test_sparse_history_not_accepted_as_full_past(self):
  e=fixture();e.mapper.frames=2
  with self.assertRaisesRegex(ValueError,'all acquired'):F.FullPosteriorForecast(e.mapper,e.models,e.past)
 def test_mutation_detected(self):
  e=fixture();e.models['G'].weights[0]+=1
  with self.assertRaises(AssertionError):e.assert_unchanged()
if __name__=='__main__':unittest.main()
