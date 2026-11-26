"""Physical joint input tests; no ROS/device or motion access."""
from pathlib import Path
import sys
import unittest
import numpy as np

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'runtime'))
from joint_observation import conservative_belief,freeze_joint_grid,canonical_sensor_arrays,navigation_map_source


class JointObservationTests(unittest.TestCase):
    def test_map_source_defaults_and_explicit_isolated_source(self):
        self.assertEqual(navigation_map_source({}),
            dict(topic='/nso_visual/map',publisher='/slam_gmapping',map_frame='map'))
        declared=dict(topic='/nso_pair/map',publisher='/nso_pair/gmapping',map_frame='nso_pair_map')
        resolved=navigation_map_source({'navigation_map_source':declared})
        self.assertEqual(resolved,declared)
        resolved['map_frame']='changed'
        self.assertEqual(declared['map_frame'],'nso_pair_map')

    def test_incomplete_map_source_cannot_silently_inherit_old_map(self):
        for source in ({'topic':'/nso_pair/map'},
                       dict(topic='map',publisher='/nso_pair/gmapping',map_frame='nso_pair_map')):
            with self.assertRaises(ValueError):
                navigation_map_source({'navigation_map_source':source})

    def test_unknown_occupied_intermediate_and_top_down_rows(self):
        raw=np.zeros((8,8),np.int8);raw[7,0]=-1;raw[0,7]=100;raw[0,0]=50
        original=raw.copy()
        actual=conservative_belief(raw,.05,np.eye(4),[2,2])
        np.testing.assert_array_equal(actual,[[-1,0],[-1,1]])
        np.testing.assert_array_equal(raw,original)

    def test_rotated_origin_and_outside_extent_do_not_become_free(self):
        raw=np.zeros((20,20),np.int8);raw[0,0]=-1;raw[19,19]=100
        T=np.eye(4);T[:2,:2]=[[0,-1],[1,0]];T[0,3]=1.
        actual=conservative_belief(raw,.05,T,[5,5])
        self.assertEqual(actual[0,0],1);self.assertEqual(actual[4,4],-1)
        self.assertEqual(int(np.count_nonzero(actual==0)),23)
        extended=conservative_belief(np.zeros((4,4),np.int8),.05,np.eye(4),[2,2])
        np.testing.assert_array_equal(extended,[[-1,-1],[0,-1]])

    def test_oblique_overlap_uses_all_intersected_fine_cells(self):
        raw=np.zeros((80,80),np.int8);angle=np.pi/4;c,s=np.cos(angle),np.sin(angle)
        T=np.eye(4);T[:2,:2]=[[c,-s],[s,c]];T[:2,3]=[2,2]-T[:2,:2]@[2,2]
        free=conservative_belief(raw,.05,T,[20,20])
        self.assertEqual(free[0,0],-1);self.assertEqual(free[10,10],0)
        raw[40,40]=100;blocked=conservative_belief(raw,.05,T,[20,20])
        self.assertGreater(np.count_nonzero(blocked==1),0)
        self.assertFalse(np.any((free==-1)&(blocked==0)))

    def test_initial_frame_is_frozen_to_real_north_heading_and_cell_center(self):
        base=np.array([16.5,3.,-.7]);grid=freeze_joint_grid(base)
        T=np.asarray(grid['planning_from_odom']);p=T@[base[0],base[1],-.233,1.]
        np.testing.assert_allclose(p,[40.1,39.9,0.,1.],atol=1e-12)
        self.assertAlmostEqual(base[2]+np.arctan2(T[1,0],T[0,0]),np.pi/2)
        self.assertEqual(grid['return_anchor'],[200,200,0])

    def test_measured_pixels_actual_K_and_independent_scan_timestamp_are_retained(self):
        depth=np.ones((8,8),np.float32);depth[0,0]=np.inf;depth[4,4]=5.
        rgb=np.arange(8*8*3,dtype=np.uint8).reshape(8,8,3)
        K=np.array([[10.,0,3.2],[0,11.,3.8],[0,0,1.]])
        source=dict(depth_m=depth,rgb_raw=rgb,intrinsic=K,scan_ranges_raw=np.array([.1,1.,np.inf,26.]))
        receipt=dict(source_geometry=dict(scan=dict(range_min_m=.15,range_max_m=25.,
            angle_min_rad=-3.14,angle_increment_rad=.001)),
            historical_tf=dict(odom_from_laser=dict(matrix=np.eye(4).tolist())),
            headers=dict(depth=dict(stamp_ns=1000000000),scan=dict(stamp_ns=1040000000)))
        result=canonical_sensor_arrays(source,receipt,pixel_stride=4,max_depth_m=4.)
        np.testing.assert_array_equal(result['color_rgb'],rgb[::4,::4])
        np.testing.assert_allclose(result['intrinsic'][:2],K[:2]/4)
        self.assertEqual(result['depth'][0,0],0.);self.assertEqual(result['depth'][1,1],0.)
        self.assertTrue(np.isinf(depth[0,0]));self.assertEqual(depth[4,4],5.)
        self.assertEqual(result['rgbd_timestamp_s'],1.);self.assertEqual(result['scan_timestamp_s'],1.04)
        self.assertTrue(np.isnan(result['ranges'][[0,2,3]]).all())
        self.assertEqual(result['ranges'][1],1.)


if __name__=='__main__':unittest.main()
