import unittest
from types import SimpleNamespace
import numpy as np
import open3d as o3d
from env.virtual3d import camera_pose
from nso.competition_candidates_v8_1 import aperture_support
from nso.visibility_corrected_scores_v17 import ObservedMeshAperture


class VisibilityScoreTests(unittest.TestCase):
    def test_known_occluder_masks_potential_without_reading_class(self):
        config=SimpleNamespace(resolution_m=.2,camera_height_m=.8,fov_deg=90.,max_depth_m=5.,
            width_px=96,height_px=72,truncation_m=.12)
        asset=dict(rear_boundary_xy=np.array([2.1,2.]),back_axis=np.array([0.,1.]),
            side_axis=np.array([1.,0.]),measured_width_m=1.,observed_low=np.array([1.6,2.,.3]),
            observed_high=np.array([2.6,2.,1.3]),class_vote=1.)
        pose=camera_pose((12,10),2,config,30)
        raw=aperture_support(asset,pose,config);self.assertGreater(raw.sum(),0.)
        wall=o3d.geometry.TriangleMesh.create_box(3.,.2,2.).translate([.6,2.6,0.])
        model=ObservedMeshAperture(wall,config)
        np.testing.assert_array_equal(model.support(asset,pose),np.zeros(25))
        asset['class_vote']=-1.
        np.testing.assert_array_equal(model.support(asset,pose),np.zeros(25))
        # Unknown geometry does not certify visibility, and is not invented as a wall.
        empty=ObservedMeshAperture(o3d.geometry.TriangleMesh(),config)
        np.testing.assert_array_equal(empty.support(asset,pose),raw)


if __name__=='__main__':unittest.main()
