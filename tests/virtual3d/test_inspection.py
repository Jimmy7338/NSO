"""Checks for causal controls, geometry units and fixed-reference metrics."""
from dataclasses import replace
import unittest
import numpy as np
import open3d as o3d
from env.virtual3d import VirtualConfig
from nso.mapping3d import SensorMapper
from utils.inspection_benchmark import InspectionWorld,TargetEvaluator,angular_proxy


class InspectionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.config = VirtualConfig(depth_sigma_m=0,dropout=0)
        cls.world = InspectionWorld(cls.config,101,'shelf')
        cls.pose,cls.cell = cls.world.pose(1.4,-90)

    def test_noiseless_depth_is_axial_and_hits_scene(self):
        frame = self.world.capture(self.pose,0,'none')
        points,_ = frame.points()
        distances = self.world.scene.compute_distance(o3d.core.Tensor(points.astype(np.float32)),nthreads=1).numpy()
        self.assertLess(float(distances.max()),2e-5)

    def test_persistent_and_fresh_noise_are_distinct(self):
        a = self.world.capture(self.pose,4,'fixed_01')
        b = self.world.capture(self.pose,5,'fixed_01')
        np.testing.assert_array_equal(a.depth_m,b.depth_m)
        x = self.world.capture(self.pose,4,'iid_01')
        y = self.world.capture(self.pose,5,'iid_01')
        self.assertGreater(float(np.abs(x.depth_m-y.depth_m).mean()),.005)

    def test_shared_prefix_produces_identical_reconstruction(self):
        a,b = SensorMapper(self.world.shape,self.config),SensorMapper(self.world.shape,self.config)
        for i in range(4):
            frame = self.world.capture(self.pose,i,'iid_01')
            a.update(frame)
            b.update(frame)
        np.testing.assert_array_equal(np.asarray(a.mesh().vertices),np.asarray(b.mesh().vertices))
        np.testing.assert_array_equal(np.asarray(a.mesh().triangles),np.asarray(b.mesh().triangles))

    def test_proxy_requires_no_truth_and_has_no_same_view_reward(self):
        mapper = SensorMapper(self.world.shape,self.config)
        frame = self.world.capture(self.pose,0,'none')
        mapper.update(frame)
        self.assertEqual(angular_proxy(mapper,self.pose,self.cell)['angular'],0.)
        pose,cell = self.world.pose(1.4,90)
        self.assertGreater(angular_proxy(mapper,pose,cell)['ungated'],0.)

    def test_full_truth_scores_perfectly_on_fixed_reference(self):
        evaluator = TargetEvaluator(self.world,6000,12000)
        result = evaluator.evaluate(self.world.target)
        self.assertGreater(len(evaluator.reference),1000)
        self.assertAlmostEqual(result['f1_01cm'],1.)


if __name__=='__main__':
    unittest.main()
