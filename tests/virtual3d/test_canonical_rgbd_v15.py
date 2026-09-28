import unittest
import numpy as np
from env.canonical_rgbd_v15 import render_axial_depth, render_marker_rgb, CanonicalRGBDInspectionWorldV15
from env.virtual3d_inspection_v4 import InspectionConfigV4, read_inspection_markers_rgb


class CanonicalRGBDTests(unittest.TestCase):
    def setUp(self):
        self.k = np.array([[3.,0.,2.],[0.,3.,2.],[0.,0.,1.]])
        self.pose = np.eye(4)

    def depth(self, boxes, maximum=5.):
        return render_axial_depth(boxes,self.k,self.pose,5,5,maximum)[0]

    def test_off_axis_depth_is_axial_not_euclidean_range(self):
        depth = self.depth([[-10,-10,2,20,20,1]])
        np.testing.assert_allclose(depth,2.,rtol=0,atol=3e-7)

    def test_front_slab_and_full_solid_match_but_open_geometry_differs(self):
        full = self.depth([[-10,-10,2,20,20,1]])
        panel = [[-10,-10,2,20,20,.08],[-10,-10,2.92,20,20,.08]]
        np.testing.assert_array_equal(full,self.depth(panel))
        np.testing.assert_array_equal(full,self.depth(panel[::-1]))
        self.assertFalse(np.array_equal(full,self.depth(panel[1:])))

    def test_near_far_empty_and_foreground_occlusion(self):
        self.assertFalse(self.depth([]).any())
        self.assertFalse(self.depth([[-10,-10,.1,20,20,.01]]).any())
        self.assertFalse(self.depth([[-10,-10,6,20,20,1]]).any())
        depth = self.depth([[-10,-10,3,20,20,1],[-10,-10,1,20,20,.5]])
        np.testing.assert_allclose(depth,1.,atol=2e-7,rtol=0)

    def test_markers_require_visible_hit_and_are_decoded_from_rgb(self):
        depth = np.ones((1,3),np.float32)
        points = np.array([[[0.,2.,1.],[0.,1.,1.],[2.,2.,1.]]])
        marker=dict(x0=-.5,x1=.5,y=2.,z0=.9,z1=1.2,physical_class=3)
        rgb = render_marker_rgb(depth,points,[marker])
        np.testing.assert_array_equal(read_inspection_markers_rgb(rgb),[[3,0,0]])
        depth[0,0]=0
        self.assertFalse(read_inspection_markers_rgb(render_marker_rgb(depth,points,[marker])).any())

    def test_noise_configuration_is_not_silently_ignored(self):
        with self.assertRaisesRegex(ValueError,'zero'):
            CanonicalRGBDInspectionWorldV15(InspectionConfigV4(depth_sigma_m=.01,semantic_source='rgb_marker'))


if __name__=='__main__':unittest.main()
