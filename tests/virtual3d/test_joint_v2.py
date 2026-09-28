"""Information-boundary, sensor parity and quality-gain checks for v2."""
from types import SimpleNamespace
import unittest
import numpy as np
from env.virtual3d_v2 import VirtualConfigV2,VirtualWorldV2
from env.grid_exploration import GridConfig
from nso.mapping3d_v2 import QualityMapperV2
from nso.joint_planner_v2 import JointPlannerV2
from nso.camera_mapping_v2 import CameraQualityMapperV2


class JointV2Tests(unittest.TestCase):
    def test_lidar_coverage_does_not_imply_camera_coverage(self):
        w=VirtualWorldV2(VirtualConfigV2(),215)
        m=CameraQualityMapperV2(w.shape,w.config)
        m.update(w.sense(),w.scan())
        self.assertTrue(np.any((m.belief!=-1)&~m.camera_seen))
        old=m.camera_seen.copy()
        w.heading=(w.heading+2)%4
        m.update(w.sense(),w.scan())
        self.assertTrue(np.all(m.camera_seen[old]))
        self.assertGreater(int(m.camera_seen.sum()),int(old.sum()))

    def test_larger_world_is_connected_and_start_safe(self):
        for layout in ('rooms','warehouse'):
            world=VirtualWorldV2(VirtualConfigV2(),211,layout)
            self.assertTrue(world.reachable[world.start])
            self.assertEqual(int(world.reachable.sum()),int((~world._blocked).sum()))
            self.assertEqual(world.shape,(60,80))

    def test_semantics_do_not_change_rgb_depth_pose_or_lidar(self):
        c=VirtualConfigV2()
        a=VirtualWorldV2(c,211,semantic_condition='aligned')
        b=VirtualWorldV2(c,211,semantic_condition='shuffled')
        changed=False
        for heading in range(4):
            a.heading=b.heading=heading
            x,y=a.sense(),b.sense()
            for name in ('depth_m','color_rgb','world_from_camera','intrinsic'):
                np.testing.assert_array_equal(getattr(x,name),getattr(y,name))
            np.testing.assert_array_equal(a.scan().ranges_m,b.scan().ranges_m)
            changed|=bool(np.any(x.semantic!=y.semantic))
        self.assertTrue(changed)

    def test_precision_signal_can_reward_near_view_without_angle_novelty(self):
        c=VirtualConfigV2();p=JointPlannerV2(GridConfig(),c,semantic=False)
        p.mapper=SimpleNamespace(shape=(30,40));p.ray=None
        p.quality=dict(point=np.array([[4.,3.,.8]]),normal=np.array([[0.,-1.,0.]]),
            bits=np.array([255]),label=np.array([3]),n=np.array([4]),information=np.array([.25]),
            residual=np.array([0.]),normal_dispersion=np.array([0.]))
        near=p._quality_gain((19,19),0)
        far=p._quality_gain((26,19),0)
        self.assertEqual(near[1],0.)
        self.assertGreater(near[2],far[2])

    def test_geometric_variant_ignores_class_values(self):
        c=VirtualConfigV2(width_m=12,height_m=10,max_steps=50)
        w=VirtualWorldV2(c,212);m=QualityMapperV2(w.shape,c)
        m.update(w.sense(),w.scan());obs=m.observation(w.position,w.heading,0)
        grid=GridConfig(max_steps=50)
        a=JointPlannerV2(grid,c,semantic=False);a.set_mapping(m)
        first=a.act(obs)
        for q in m.quality.values():q['label']=2 if q['label']==3 else 3
        b=JointPlannerV2(grid,c,semantic=False);b.set_mapping(m)
        second=b.act(obs)
        self.assertEqual(first,second)


if __name__=='__main__':unittest.main()
