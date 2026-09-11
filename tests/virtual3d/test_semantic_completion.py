"""Semantic information must affect hypotheses, never measured geometry."""
from dataclasses import replace
import unittest
from unittest.mock import patch
from types import SimpleNamespace
import open3d as o3d
from utils.inspection_benchmark import look_at_xy
from utils.reconstruction_metrics import ray_scene
import numpy as np
from env.virtual3d import VirtualConfig
from utils.inspection_benchmark import InspectionWorld
from nso.semantic_completion_v3 import SemanticHistoryMapperV3,ObjectCompletionModel,ObjectJointPlannerV3
from nso.camera_joint_planner_v2 import CameraJointPlannerV2
from env.grid_exploration import GridConfig


class SemanticCompletionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.world=InspectionWorld(VirtualConfig(voxel_m=.03),223,'box')
        pose,_=cls.world.pose(1.4,-90)
        cls.frame=cls.world.capture(pose,0,'none')

    def mapper(self,labels):
        mapper=SemanticHistoryMapperV3((30,40),self.world.config)
        mapper.update(replace(self.frame,semantic=labels))
        return mapper

    def test_labels_do_not_change_reconstruction(self):
        labels=self.frame.semantic
        a=self.mapper(labels);b=self.mapper(np.where(labels==2,3,labels).astype(np.uint8))
        before=np.asarray(a.mesh().vertices).copy()
        ObjectCompletionModel(a,True)
        np.testing.assert_array_equal(before,np.asarray(a.mesh().vertices))
        np.testing.assert_array_equal(np.asarray(a.mesh().vertices),np.asarray(b.mesh().vertices))
        np.testing.assert_array_equal(np.asarray(a.mesh().triangles),np.asarray(b.mesh().triangles))

    def test_geometry_has_identical_candidates_and_hypotheses_after_label_swap(self):
        labels=self.frame.semantic
        a=ObjectCompletionModel(self.mapper(labels),False)
        b=ObjectCompletionModel(self.mapper(np.where(labels==2,3,labels).astype(np.uint8)),False)
        self.assertGreater(len(a.objects),0)
        np.testing.assert_array_equal(a.points,b.points)
        np.testing.assert_array_equal(a.weights,b.weights)
        safe=np.ones((30,40),bool);distances=np.ones((30,40))
        self.assertEqual(a.candidates(safe,distances),b.candidates(safe,distances))
        self.assertLessEqual(len(a.candidates(safe,distances)),20)

    def test_missing_labels_restore_common_shape_mixture(self):
        mapper=self.mapper(np.zeros_like(self.frame.semantic))
        a=ObjectCompletionModel(mapper,True);b=ObjectCompletionModel(mapper,False)
        np.testing.assert_array_equal(a.points,b.points)
        np.testing.assert_array_equal(a.weights,b.weights)

    def test_binary_objectness_ignores_fine_category_swap(self):
        labels=self.frame.semantic
        a=ObjectCompletionModel(self.mapper(labels),True,fine_categories=False)
        swapped=np.where(labels==2,3,labels).astype(np.uint8)
        b=ObjectCompletionModel(self.mapper(swapped),True,fine_categories=False)
        np.testing.assert_array_equal(a.points,b.points)
        np.testing.assert_array_equal(a.weights,b.weights)

    def test_geometry_updates_shape_belief_from_observed_solid_faces(self):
        mapper=SemanticHistoryMapperV3((30,40),self.world.config)
        for index,azimuth in enumerate((-90,0,90,180)):
            pose,_=self.world.pose(1.4,azimuth)
            frame=self.world.capture(pose,index,'none')
            mapper.update(replace(frame,semantic=np.zeros_like(frame.semantic)))
        model=ObjectCompletionModel(mapper,False)
        self.assertGreater(len(model.objects),0)
        target=min(model.objects,key=lambda obj:np.linalg.norm(obj['center'][:2]-[4.,3.]))
        self.assertLess(target['shelf_probability'],.5)

    def test_unseen_completion_survives_saturated_observed_quality(self):
        planner=ObjectJointPlannerV3(GridConfig(),self.world.config)
        planner.quality={'point':np.zeros((2,3))}
        # A perfectly reconstructed observed subset can still have unseen faces.
        gain=np.array([0.,0.,.2,.3])
        self.assertEqual(CameraJointPlannerV2._recall_increment(planner,gain,2,.4,.1,1.),0.)
        self.assertAlmostEqual(planner._recall_increment(gain,2,.4,.1,1.),.25)

    def test_hidden_back_face_has_no_predicted_visible_area(self):
        model=ObjectCompletionModel.__new__(ObjectCompletionModel)
        model.mapper=SimpleNamespace(config=self.world.config,shape=(30,40))
        model.points=np.array([[3.5,3.,.85],[4.5,3.,.85]])
        model.weights=np.ones(2);model.hypothesis_ids=np.zeros(2,dtype=int)
        mesh=o3d.geometry.TriangleMesh.create_box(1.,1.,1.5).translate((3.5,2.5,0.))
        model.hypothesis_rays=[ray_scene(mesh)]
        pose=look_at_xy(np.array([2.,3.,.85]),np.array([4.,3.,.85]))
        with patch('nso.semantic_completion_v3.camera_pose',return_value=pose):
            gain=model.gain((15,10),0,None)
        np.testing.assert_array_equal(gain,[1.,0.])

    def test_semantics_change_prior_weights_with_same_geometric_proposals(self):
        labels=self.frame.semantic
        a=ObjectCompletionModel(self.mapper(labels),True)
        b=ObjectCompletionModel(self.mapper(np.where(labels==2,3,labels).astype(np.uint8)),True)
        np.testing.assert_array_equal(a.points,b.points)
        self.assertGreater(float(np.abs(a.weights-b.weights).sum()),.1)
        safe=np.ones((30,40),bool);distances=np.ones((30,40))
        self.assertEqual(a.candidates(safe,distances),b.candidates(safe,distances))


if __name__=='__main__':unittest.main()
