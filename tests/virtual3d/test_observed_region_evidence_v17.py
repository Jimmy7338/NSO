import unittest
from types import SimpleNamespace
import numpy as np
import open3d as o3d
from nso.observed_region_evidence_v17 import link_region,actual_depth_evidence,predicted_mesh_evidence


def region(group,index,keys):
    return dict(group=group,asset_index=index,keys=keys,coordinate_epoch='fixed_world_pose')


class RegionEvidenceTests(unittest.TestCase):
    def test_correspondence_uses_keys_instead_of_index_or_label(self):
        anchor=region(1,0,[(1,2,3),(2,2,3)])
        current=[region(1,0,[(9,9,9)]),region(8,3,[(1,2,3),(2,2,3),(3,2,3)])]
        current[0]['class_vote']=1;current[1]['class_vote']=-1
        result=link_region(anchor,current)
        self.assertEqual(result['matched_asset_index'],3)
        current[0]['class_vote']=-1;current[1]['class_vote']=1
        self.assertEqual(link_region(anchor,current),result)

    def test_split_merge_missing_and_rejected_asset_do_not_get_forced_matches(self):
        anchor=region(1,0,[(1,0,0),(2,0,0)])
        self.assertEqual(link_region(anchor,[region(3,0,[(1,0,0)]),region(4,1,[(2,0,0)])])['status'],'split_ambiguous')
        self.assertEqual(link_region(anchor,[region(3,0,[(1,0,0),(4,0,0)])],
            [region(2,1,[(4,0,0)])])['status'],'merged_ambiguous')
        self.assertEqual(link_region(anchor,[])['status'],'no_observed_overlap')
        self.assertEqual(link_region(anchor,[region(3,None,[(1,0,0)])])['status'],'region_not_candidate_asset')

    def test_epoch_change_is_rejected(self):
        anchor=region(1,0,[(1,0,0)]);other=region(2,1,[(1,0,0)]);other['coordinate_epoch']='corrected_pose_graph'
        with self.assertRaises(ValueError):link_region(anchor,[other])

    def test_observed_occluder_and_actual_depth_are_distinct_evidence(self):
        config=SimpleNamespace(max_depth_m=5.,truncation_m=.12)
        intrinsic=np.array([[10.,0.,4.],[0.,10.,4.],[0.,0.,1.]])
        mesh=o3d.geometry.TriangleMesh.create_box(.4,.4,.2).translate([-.2,-.2,1.])
        points=np.array([[0.,0.,3.]])
        before=predicted_mesh_evidence(points,np.eye(4),intrinsic,(9,9),config,mesh,[])
        self.assertEqual(before['known_surface_before_point'],1)
        self.assertEqual(before['occluder_key_not_in_target_region'],1)
        self.assertFalse(before['future_frame_used'])
        frame=SimpleNamespace(world_from_camera=np.eye(4),intrinsic=intrinsic,depth_m=np.ones((9,9)))
        actual=actual_depth_evidence(points,frame,config)
        self.assertEqual(actual['actual_depth_before_point'],1);self.assertEqual(actual['consistent'],0)
        frame.depth_m[:]=3.
        self.assertEqual(actual_depth_evidence(points,frame,config)['consistent'],1)
        self.assertEqual(predicted_mesh_evidence(points,np.eye(4),intrinsic,(9,9),config,mesh,[]),before)
        own=predicted_mesh_evidence(points,np.eye(4),intrinsic,(9,9),config,mesh,[(0,0,6)])
        self.assertEqual(own['occluder_key_in_target_region'],1)


if __name__=='__main__':unittest.main()
