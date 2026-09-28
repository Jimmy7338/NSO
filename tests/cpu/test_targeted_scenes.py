"""Large scene structure, functional cues and observation-boundary tests."""
import json
from pathlib import Path
import tempfile
import unittest
import numpy as np
from env.targeted_scenes import generate_scene,FAMILIES,CONDITIONS
from env.grid_exploration import GridConfig,GridExplorationEnv
from nso.grid_mechanisms import MechanismPolicy
from scripts.eval_targeted_scenes import run_episode
from utils.cpu_protocol import geometry_hash


class TargetedTests(unittest.TestCase):
    def test_all_families_scales_connected_and_seeded(self):
        hashes=set()
        for family in FAMILIES:
            for size in (192,256):
                for seed in (41,42):
                    scene=generate_scene(family,size,seed)
                    np.testing.assert_array_equal(scene.occupancy,generate_scene(family,size,seed).occupancy)
                    h=geometry_hash(scene.occupancy);self.assertNotIn(h,hashes);hashes.add(h)
                    self.assertGreater(scene.descriptors['reachable_area_m2'],500)
                    self.assertEqual(scene.descriptors['extent_m'],size*.25)
                    self.assertEqual(len(scene.rooms),2*(size//32-1))
                    self.assertTrue(all(r['reachable_area_m2']>0 for r in scene.rooms))
                    large=[r['reachable_area_m2'] for r in scene.rooms if r['productive']]
                    small=[r['reachable_area_m2'] for r in scene.rooms if not r['productive']]
                    self.assertGreater(min(large),max(small)*2)
                    for field in scene.semantic_fields.values():
                        self.assertTrue(np.isfinite(field).all());self.assertGreaterEqual(field.min(),0);self.assertLessEqual(field.max(),1)

    def test_shuffle_preserves_support_and_histogram(self):
        scene=generate_scene('office_spine',192,41)
        aligned=scene.semantic_fields['aligned'];shuffled=scene.semantic_fields['shuffled']
        np.testing.assert_array_equal(aligned>0,shuffled>0)
        np.testing.assert_array_equal(np.sort(aligned.ravel()),np.sort(shuffled.ravel()))
        self.assertFalse(np.array_equal(aligned,shuffled))
        self.assertTrue(np.all(scene.semantic_fields['absent']==0))
        for room in scene.rooms:
            expected=.9 if room['productive'] else .1
            self.assertAlmostEqual(aligned[tuple(room['entrance'])],expected,places=6)

    def test_hidden_semantic_and_room_annotations_not_policy_inputs(self):
        scene=generate_scene('office_spine',192,41)
        config=GridConfig(resolution_m=.25,robot_radius_m=.2,sensor_range_m=3)
        field=scene.semantic_fields['aligned'];env=GridExplorationEnv(scene.occupancy,config,field)
        obs=env.reset(start=scene.entrances[0],heading=1)
        self.assertFalse(hasattr(obs,'rooms'));self.assertFalse(hasattr(obs,'room_labels'))
        self.assertTrue(np.isnan(obs.semantic[~obs.visible]).all())
        altered=field.copy();altered[~obs.visible]=1-altered[~obs.visible]
        other=GridExplorationEnv(scene.occupancy,config,altered).reset(start=scene.entrances[0],heading=1)
        a=MechanismPolicy(config,'gain_semantic_structure');b=MechanismPolicy(config,'gain_semantic_structure')
        self.assertEqual(a.act(obs),b.act(other))
        self.assertEqual(a.selection_audit,b.selection_audit)

    def test_geometry_ignores_semantic_condition(self):
        scene=generate_scene('office_spine',192,41)
        config=GridConfig(resolution_m=.25,robot_radius_m=.2,sensor_range_m=3,max_steps=50)
        a=GridExplorationEnv(scene.occupancy,config,scene.semantic_fields['aligned'])
        b=GridExplorationEnv(scene.occupancy,config,scene.semantic_fields['shuffled'])
        x=a.reset(start=scene.entrances[0],heading=1);y=b.reset(start=scene.entrances[0],heading=1)
        p=MechanismPolicy(config,'gain_control');q=MechanismPolicy(config,'gain_control')
        for _ in range(50):
            u,v=p.act(x),q.act(y);self.assertEqual(u,v)
            x,done=a.step(u.action);y,other=b.step(v.action)
            p.observe_outcome(x,done);q.observe_outcome(y,other)
            if done:break
        self.assertEqual(p.semantic_updates,0)

    def test_episode_metrics_and_pairing(self):
        scene=generate_scene('office_spine',192,41)
        config=dict(environment=dict(resolution_m=.25,robot_radius_m=.2,sensor_range_m=3,sensor_fov_deg=90,max_steps=24),max_candidates=24,mechanisms={})
        with tempfile.TemporaryDirectory() as d:
            results=[]
            for case in ('geometry','semantic_aligned'):
                path=Path(d)/case;r=run_episode(scene,config,case,0,path);results.append(r)
                with np.load(path/'observations.npz') as z:
                    h,w=z['occupancy'].shape
                    visible=np.unpackbits(z['visible_packed'],axis=1)[:,:h*w].reshape(-1,h,w).any(0)
                    self.assertAlmostEqual(r['coverage_ratio'],np.count_nonzero(visible&z['reachable'])/z['reachable'].sum())
                self.assertEqual(r['steps'],24)
            self.assertEqual(results[0]['initial_pose'],results[1]['initial_pose'])
            self.assertEqual(results[0]['ground_truth_sha256'],results[1]['ground_truth_sha256'])

if __name__=='__main__':unittest.main()
