"""Focused reproductions of candidate omissions, without changing the planner.

These establish visibility/candidate contracts, not improvements in navigation.
The second fixture is a rounded sensor-derived development example; its point
is an observed target cue and the empty ray scene isolates heading selection.
"""
import unittest
from types import SimpleNamespace
import numpy as np
import open3d as o3d
from env.virtual3d import VirtualConfig, camera_pose
from nso.semantic_completion_v3 import ObjectCompletionModel


class CandidateAuditFindings(unittest.TestCase):
    def model(self, objects, points, shape=(66, 46)):
        model=ObjectCompletionModel.__new__(ObjectCompletionModel)
        model.mapper=SimpleNamespace(shape=shape, config=VirtualConfig())
        model.objects=[{'center':np.asarray(point,float)} for point in objects]
        model.points=np.asarray(points,float).reshape(-1,3)
        model.weights=np.ones(len(points));model.hypothesis_ids=np.zeros(len(points),int)
        model.hypothesis_rays=[o3d.t.geometry.RaycastingScene(nthreads=1)]
        return model

    def test_world_bearing_and_camera_axes_agree(self):
        directions=np.array([[0,1],[1,0],[0,-1],[-1,0]])
        for expected,delta in enumerate(directions):
            heading=int(np.argmax(directions@delta))
            self.assertEqual(expected,heading)
            np.testing.assert_array_equal(camera_pose((10,10),heading,VirtualConfig(),30)[:2,2],delta)

    def test_nearest_cluster_heading_drops_another_visible_target(self):
        model=self.model([[4.56,9.99,.72],[2.77,5.07,.75],[1.951,2.550,.75],[4.59,3.28,.71]],
                         [[4.59,3.28,.8]])
        cell=(48,14);xy=camera_pose(cell,0,model.mapper.config,66)[:2,3]
        nearest=min(model.objects,key=lambda obj:np.linalg.norm(obj['center'][:2]-xy))
        heading=int(np.argmax(np.array([[0,1],[1,0],[0,-1],[-1,0]])@(nearest['center'][:2]-xy)))
        self.assertEqual(heading,2)
        self.assertEqual(float(model.gain(cell,heading,None).sum()),0.)
        self.assertEqual(float(model.gain(cell,1,None).sum()),1.)

    def test_two_metre_ring_omits_in_range_safe_standoff(self):
        model=self.model([[4.57,9.99,.72]],[[4.57,9.99,.8]])
        safe=np.ones(model.mapper.shape,bool);distances=np.zeros(model.mapper.shape)
        cell=(29,26)
        proposals=model.candidates(safe,distances,limit=10**9)
        self.assertNotIn(cell,proposals)
        self.assertGreater(float(model.gain(cell,0,None).sum()),0.)
        pose=camera_pose(cell,0,model.mapper.config,66)
        self.assertGreater(np.linalg.norm(model.points[0,:2]-pose[:2,3]),2.)
        self.assertLess(float(((model.points-pose[:3,3])@pose[:3,:3])[0,2]),model.mapper.config.max_depth_m)


if __name__=='__main__':unittest.main()
