"""Optional 3D geometry/metric/data-contract checks; run in .venv-3d."""
import os
os.environ.setdefault('OMP_NUM_THREADS','2')
import tempfile
from pathlib import Path
import unittest
import numpy as np
from env.virtual3d import VirtualConfig,VirtualWorld,camera_pose
from utils.rgbd_contract import RGBDFrame
from utils.reconstruction_metrics import ReconstructionEvaluator
from nso.mapping3d import SensorMapper
from nso.quality_coverage3d import QualityCoveragePolicy
from env.grid_exploration import GridConfig


class Virtual3DTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.config=VirtualConfig(max_steps=12)
        cls.world=VirtualWorld(cls.config,seed=71)

    def test_camera_frame_is_proper_rotation_and_points_forward(self):
        for heading,forward in enumerate(([0,1,0],[1,0,0],[0,-1,0],[-1,0,0])):
            pose=camera_pose((15,5),heading,self.config,30)
            np.testing.assert_allclose(pose[:3,2],forward)
            self.assertAlmostEqual(np.linalg.det(pose[:3,:3]),1.)

    def test_depth_backprojection_hits_true_surface(self):
        world=VirtualWorld(VirtualConfig(depth_sigma_m=0,dropout=0),seed=71)
        points,_=world.sense().points()
        import open3d as o3d
        from utils.reconstruction_metrics import ray_scene
        distances=ray_scene(world.mesh).compute_distance(o3d.core.Tensor(points.astype(np.float32)),nthreads=1).numpy()
        self.assertLess(float(distances.max()),1e-5)

    def test_frame_roundtrip_preserves_depth_and_pose(self):
        frame=self.world.sense()
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'frame.npz';frame.save(path);restored=RGBDFrame.load(path)
            np.testing.assert_array_equal(restored.depth_m,frame.depth_m)
            np.testing.assert_array_equal(restored.world_from_camera,frame.world_from_camera)

    def test_invalid_pose_and_negative_depth_are_rejected(self):
        from dataclasses import replace
        frame=self.world.sense();pose=frame.world_from_camera.copy();pose[:3,:3]*=2
        with self.assertRaises(ValueError):replace(frame,world_from_camera=pose).validate()
        with self.assertRaises(ValueError):replace(frame,depth_m=-np.ones_like(frame.depth_m)).validate()

    def test_semantic_swap_does_not_change_depth_or_scan(self):
        a=VirtualWorld(self.config,71,semantic_condition='aligned')
        b=VirtualWorld(self.config,71,semantic_condition='shuffled')
        changed=False
        for heading in range(4):
            a.heading=b.heading=heading
            np.testing.assert_array_equal(a.sense().depth_m,b.sense().depth_m)
            np.testing.assert_array_equal(a.scan().ranges_m,b.scan().ranges_m)
            changed|=bool(np.any(a.sense().semantic!=b.sense().semantic))
        self.assertTrue(changed)

    def test_unseen_surface_lowers_recall_even_with_high_precision(self):
        mapper=SensorMapper(self.world.shape,self.config);mapper.update(self.world.sense(),self.world.scan())
        evaluator=ReconstructionEvaluator(self.world,count=6000)
        partial=evaluator.evaluate(mapper.mesh(),.5)
        complete=evaluator.evaluate(self.world.mesh,.5)
        self.assertGreater(partial['precision_05cm'],.95)
        self.assertLess(partial['recall_05cm'],.5)
        self.assertAlmostEqual(complete['f1_05cm'],1.)
        self.assertAlmostEqual(complete['joint_05cm'],.5)

    def test_duplicate_view_does_not_increase_angular_diversity(self):
        mapper=SensorMapper(self.world.shape,self.config);frame=self.world.sense();scan=self.world.scan()
        mapper.update(frame,scan);first=mapper.evidence()
        mapper.update(frame,scan);second=mapper.evidence()
        np.testing.assert_array_equal(first[1],second[1])

    def test_geometry_quality_does_not_depend_on_semantic_labels(self):
        mapper=SensorMapper(self.world.shape,self.config);mapper.update(self.world.sense(),self.world.scan())
        obs=mapper.observation(self.world.position,self.world.heading,0)
        grid=GridConfig(resolution_m=.2,robot_radius_m=.2,sensor_range_m=4,sensor_fov_deg=360,max_steps=12)
        a=QualityCoveragePolicy(grid,self.config,'geometric_quality');b=QualityCoveragePolicy(grid,self.config,'geometric_quality')
        p,v,l=mapper.evidence();a.set_surface_evidence((p,v,l));b.set_surface_evidence((p,v,np.zeros_like(l)))
        self.assertEqual(a.act(obs),b.act(obs))


if __name__=='__main__':unittest.main()
