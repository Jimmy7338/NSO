"""Information controls and sensor/GT boundaries, not planner effectiveness."""
from dataclasses import fields, replace
import unittest
import numpy as np

from env.virtual3d_inspection_v4 import (InspectionConfigV4, InspectionWorldV4,
                                        read_inspection_markers_rgb, MARKER_COLORS,
                                        union_surface_from_boxes)
from utils.rgbd_contract import RGBDFrame
from utils.reconstruction_metrics import ReconstructionEvaluator


class InspectionSceneTests(unittest.TestCase):
    def config(self,**kwargs):
        return InspectionConfigV4(bays_per_side=1,depth_sigma_m=0.,dropout=0.,**kwargs)

    def test_small_cells_are_connected_and_inspection_views_are_reachable(self):
        world=InspectionWorldV4(self.config(),seed=711)
        self.assertEqual(world.shape,(66,46))
        self.assertFalse(world._blocked[world.start])
        np.testing.assert_array_equal(world.reachable,~world._blocked)
        self.assertEqual(len(world.objects),2)
        for item in world.inspection_truth:
            for key in ('front_pose','rear_pose'):
                self.assertTrue(world.reachable[item[key][0]])
        for _ in range(8):
            frame,collision,done=world.step('forward')
            self.assertFalse(collision);self.assertFalse(done);frame.validate()

    def test_visible_label_interventions_leave_all_physical_inputs_identical(self):
        config=replace(self.config(),depth_sigma_m=.01,dropout=.01,pose_noise_m=.01)
        worlds=[InspectionWorldV4(config,seed=712,semantic_condition=mode)
                for mode in ('aligned','shuffled','absent')]
        base=worlds[0]
        poses=[(base.start,1),base.inspection_truth[0]['front_pose'],base.inspection_truth[0]['rear_pose']]
        for other in worlds[1:]:
            for attr in ('vertices','triangles','vertex_colors'):
                np.testing.assert_array_equal(np.asarray(getattr(base.mesh,attr)),np.asarray(getattr(other.mesh,attr)))
            np.testing.assert_array_equal(base.triangle_classes,other.triangle_classes)
            np.testing.assert_array_equal(base.occupancy,other.occupancy)
        saw_object=False
        for step,(position,heading) in enumerate(poses):
            for world in worlds:world.position=position;world.heading=heading;world.step_count=step
            frames=[world.sense() for world in worlds];scans=[world.scan() for world in worlds]
            for frame in frames:
                frame.validate()
                self.assertEqual(set(frame.__dict__),{field.name for field in fields(RGBDFrame)})
            for frame,scan in zip(frames[1:],scans[1:]):
                for attr in ('depth_m','color_rgb','intrinsic','world_from_camera'):
                    np.testing.assert_array_equal(getattr(frames[0],attr),getattr(frame,attr))
                np.testing.assert_array_equal(scans[0].ranges_m,scan.ranges_m)
                np.testing.assert_array_equal(scans[0].world_from_laser,scan.world_from_laser)
            expected=np.where(frames[0].semantic==2,3,np.where(frames[0].semantic==3,2,frames[0].semantic))
            np.testing.assert_array_equal(frames[1].semantic,expected)
            self.assertFalse(frames[2].semantic.any())
            saw_object |= bool(np.any(frames[0].semantic>1))
        self.assertTrue(saw_object)

    def test_front_face_is_ambiguous_and_rear_depth_reveals_structure(self):
        low=InspectionWorldV4(replace(self.config(),complex_fraction=0.),seed=713)
        high=InspectionWorldV4(replace(self.config(),complex_fraction=1.),seed=713)
        for key in ('front_pose','rear_pose'):
            for world in (low,high):world.position,world.heading=world.inspection_truth[0][key]
            a,b=low.sense(),high.sense()
            if key=='front_pose':
                # Different triangle trees can differ by float32 ray roundoff.
                np.testing.assert_allclose(a.depth_m,b.depth_m,rtol=0,atol=1e-6)
                np.testing.assert_array_equal(a.color_rgb,b.color_rgb)
            else:
                self.assertGreater(np.count_nonzero(np.abs(a.depth_m-b.depth_m)>.01),100)

    def test_open_control_reveals_geometry_from_front(self):
        config=replace(self.config(),observation_condition='open')
        worlds=[InspectionWorldV4(replace(config,complex_fraction=p),seed=714) for p in (0.,1.)]
        for world in worlds:world.position,world.heading=world.inspection_truth[0]['front_pose']
        a,b=[world.sense() for world in worlds]
        self.assertGreater(np.count_nonzero(np.abs(a.depth_m-b.depth_m)>.01),100)

    def test_relation_controls_preserve_truth_and_nonsemantic_observations(self):
        worlds=[InspectionWorldV4(replace(self.config(),semantic_relation=relation),seed=715)
                for relation in ('correlated','independent','reversed')]
        base=worlds[0]
        for world in worlds:
            world.position,world.heading=world.inspection_truth[0]['front_pose']
        first=base.sense()
        for world in worlds[1:]:
            np.testing.assert_array_equal(np.asarray(base.mesh.vertices),np.asarray(world.mesh.vertices))
            np.testing.assert_array_equal(np.asarray(base.mesh.triangles),np.asarray(world.mesh.triangles))
            np.testing.assert_array_equal(base.triangle_classes,world.triangle_classes)
            other=world.sense()
            np.testing.assert_array_equal(first.depth_m,other.depth_m)
            np.testing.assert_array_equal(first.color_rgb,other.color_rgb)
        for item in worlds[2].inspection_truth:
            self.assertEqual(item['visible_category'],5-item['true_category'])

    def test_shape_prior_mismatch_preserves_front_but_changes_hidden_parts(self):
        config=replace(self.config(),complex_fraction=1.)
        worlds=[InspectionWorldV4(replace(config,hidden_parts=kind),seed=716)
                for kind in ('shelves','vertical_baffles')]
        for key in ('front_pose','rear_pose'):
            for world in worlds:world.position,world.heading=world.inspection_truth[0][key]
            a,b=[world.sense() for world in worlds]
            if key=='front_pose':np.testing.assert_allclose(a.depth_m,b.depth_m,rtol=0,atol=1e-6)
            else:self.assertGreater(np.count_nonzero(np.abs(a.depth_m-b.depth_m)>.01),100)

    def test_evaluator_includes_both_object_types_and_whole_scene(self):
        world=InspectionWorldV4(self.config(),seed=717)
        evaluator=ReconstructionEvaluator(world,count=4000,seed=2026)
        self.assertEqual(set(np.unique(evaluator.classes)),{1,2,3})
        self.assertGreater(len(evaluator.reference),100)
        complex_object=next(item for item in world.objects if item['category']==3)
        x,y,_,sx,sy,_=complex_object['box'];points=evaluator.reference
        interior=(evaluator.classes==3)&(points[:,0]>x+.12)&(points[:,0]<x+sx-.12)
        interior&=(points[:,1]>y+.15)&(points[:,1]<y+sy-.15)&(points[:,2]>.12)
        self.assertGreater(int(interior.sum()),0)  # actual hidden shelf surfaces in reference
        metrics=evaluator.evaluate(world.mesh,coverage=1.,thresholds=(.02,.05))
        self.assertAlmostEqual(metrics['precision_02cm'],1.)
        self.assertAlmostEqual(metrics['recall_02cm'],1.)
        self.assertAlmostEqual(metrics['f1_05cm'],1.)
        # Repeated evaluator construction is independent of current robot pose.
        world.position,world.heading=world.inspection_truth[-1]['rear_pose']
        repeated=ReconstructionEvaluator(world,count=4000,seed=2026)
        np.testing.assert_array_equal(evaluator.reference,repeated.reference)

    def test_rgb_only_reader_labels_visible_marker_pixels_without_gt_semantics(self):
        config=replace(self.config(),appearance='marked',semantic_source='rgb_marker')
        world=InspectionWorldV4(config,seed=719)
        for item in world.inspection_truth:
            world.position,world.heading=item['front_pose']
            frame=world.sense();decoded=read_inspection_markers_rgb(frame.color_rgb)
            self.assertGreater(np.count_nonzero(decoded),10)
            self.assertEqual(set(np.unique(decoded))-{0},{item['visible_category']})
            np.testing.assert_array_equal(frame.semantic,decoded)
            # Erasing every simulator semantic hit does not change RGB inference.
            world._geometry_labels={gid:1 for gid in world._geometry_labels}
            np.testing.assert_array_equal(world.sense().semantic,decoded)
        pixels=np.full((3,4,3),153,np.uint8)
        pixels[0,0]=MARKER_COLORS[2];pixels[2,3]=MARKER_COLORS[3]
        result=read_inspection_markers_rgb(pixels)
        self.assertEqual(np.count_nonzero(result),2)
        self.assertEqual((result[0,0],result[2,3]),(2,3))

    def test_marked_counterfactual_labels_do_not_change_rgb_or_depth(self):
        config=replace(self.config(),appearance='marked',semantic_source='rgb_marker')
        worlds=[InspectionWorldV4(config,seed=720,semantic_condition=condition)
                for condition in ('aligned','shuffled','absent')]
        for world in worlds:world.position,world.heading=world.inspection_truth[0]['front_pose']
        frames=[world.sense() for world in worlds]
        for frame in frames[1:]:
            np.testing.assert_array_equal(frame.color_rgb,frames[0].color_rgb)
            np.testing.assert_array_equal(frame.depth_m,frames[0].depth_m)
        self.assertGreater(np.count_nonzero(frames[0].semantic),10)
        expected=np.where(frames[0].semantic==2,3,np.where(frames[0].semantic==3,2,0))
        np.testing.assert_array_equal(frames[1].semantic,expected)
        self.assertFalse(frames[2].semantic.any())
        gray=InspectionWorldV4(replace(config,appearance='gray'),seed=720)
        gray.position,gray.heading=gray.inspection_truth[0]['front_pose']
        np.testing.assert_array_equal(np.asarray(gray.mesh.vertices),np.asarray(worlds[0].mesh.vertices))
        np.testing.assert_array_equal(np.asarray(gray.mesh.triangles),np.asarray(worlds[0].mesh.triangles))
        np.testing.assert_array_equal(gray.sense().depth_m,frames[0].depth_m)
        self.assertFalse(gray.sense().semantic.any())

    def test_marked_relation_control_changes_identity_without_changing_hidden_geometry(self):
        config=replace(self.config(),appearance='marked',semantic_source='rgb_marker')
        worlds=[InspectionWorldV4(replace(config,semantic_relation=relation),seed=721)
                for relation in ('correlated','independent','reversed')]
        for world in worlds:
            world.position,world.heading=world.inspection_truth[0]['front_pose']
            frame=world.sense();labels=read_inspection_markers_rgb(frame.color_rgb)
            self.assertEqual(set(np.unique(labels))-{0},{world.inspection_truth[0]['visible_category']})
            np.testing.assert_array_equal(np.asarray(world.mesh.vertices),np.asarray(worlds[0].mesh.vertices))
            np.testing.assert_array_equal(frame.depth_m,worlds[0].sense().depth_m)
        a,b=worlds[0].sense(),worlds[2].sense()
        self.assertFalse(np.array_equal(a.color_rgb,b.color_rgb))

    def test_long_configuration_constructs_without_navigation_run(self):
        config=InspectionConfigV4(bays_per_side=4,transfer_length_m=8.)
        world=InspectionWorldV4(config,seed=718)
        self.assertAlmostEqual(world.width,47.6)
        self.assertAlmostEqual(world.height,13.2)
        self.assertEqual(len(world.inspection_truth),8)
        self.assertEqual(int(world.reachable.sum()),int((~world._blocked).sum()))

    def test_invalid_layout_rejected(self):
        for settings in ({'doorway_width_m':.4},{'transfer_length_m':-.2},
                         {'semantic_relation':'oracle'},{'bay_width_m':4.9},
                         {'complex_fraction':1.5}):
            with self.subTest(settings=settings),self.assertRaises(ValueError):
                InspectionConfigV4(**settings)

    def test_axis_aligned_union_matches_analytic_areas_and_volumes(self):
        cube=(0,0,0,1,1,1,2)
        cases=[([cube],6.,1.),([cube,cube],6.,1.),
               ([cube,(.3,.3,.3,.4,.4,.4,2)],6.,1.),
               ([cube,(.5,0,0,1,1,1,2)],8.,1.5),
               ([cube,(1,0,0,1,1,1,2)],10.,2.),
               ([(0,0,0,2,1,1,2),(.5,-.5,0,1,2,1,2)],14.,3.)]
        for primitives,area,volume in cases:
            with self.subTest(primitives=primitives):
                mesh,classes,audit=union_surface_from_boxes(primitives)
                self.assertAlmostEqual(mesh.get_surface_area(),area,places=8)
                self.assertAlmostEqual(audit['union_volume_m3'],volume,places=8)
                self.assertTrue(mesh.is_watertight())
                self.assertTrue(mesh.is_orientable())
                self.assertEqual(len(classes),len(mesh.triangles))
                keys={tuple(sorted(triangle)) for triangle in np.asarray(mesh.triangles)}
                self.assertEqual(len(keys),len(mesh.triangles))

    def test_all_scene_faces_are_exterior_of_physical_primitive_union(self):
        world=InspectionWorldV4(self.config(),seed=722)
        vertices=np.asarray(world.mesh.vertices);triangles=np.asarray(world.mesh.triangles)
        xyz=vertices[triangles];centers=xyz.mean(axis=1)
        normals=np.cross(xyz[:,1]-xyz[:,0],xyz[:,2]-xyz[:,0])
        normals/=np.linalg.norm(normals,axis=1)[:,None]
        boxes=np.asarray(world._solid_primitives);lo=boxes[:,:3];hi=lo+boxes[:,3:6]
        def inside(points):
            return np.any(np.all((points[:,None,:]>=lo-1e-10)&(points[:,None,:]<=hi+1e-10),axis=2),axis=1)
        self.assertTrue(inside(centers-1e-6*normals).all())
        self.assertFalse(inside(centers+1e-6*normals).any())
        self.assertTrue(world.mesh.is_watertight())
        self.assertEqual(len(np.unique(vertices,axis=0)),len(vertices))
        self.assertEqual(len({tuple(sorted(t)) for t in triangles}),len(triangles))

    def test_union_surface_matches_sensor_first_hits(self):
        import open3d as o3d
        from env.virtual3d import camera_pose
        from utils.reconstruction_metrics import ray_scene
        world=InspectionWorldV4(self.config(),seed=723)
        exterior=ray_scene(world.mesh)
        for item in world.inspection_truth:
            for key in ('front_pose','rear_pose'):
                position,heading=item[key]
                pose=camera_pose(position,heading,world.config,world.shape[0])
                v,u=np.mgrid[:world.config.height_px,:world.config.width_px]
                direction=np.stack([(u-world.intrinsic[0,2])/world.intrinsic[0,0],
                                    (v-world.intrinsic[1,2])/world.intrinsic[1,1],np.ones_like(u)],axis=-1)@pose[:3,:3].T
                rays=np.empty((*u.shape,6),np.float32);rays[...,:3]=pose[:3,3];rays[...,3:]=direction
                tensor=o3d.core.Tensor(rays)
                expected=world._ray.cast_rays(tensor,nthreads=1)['t_hit'].numpy()
                actual=exterior.cast_rays(tensor,nthreads=1)['t_hit'].numpy()
                np.testing.assert_array_equal(np.isfinite(actual),np.isfinite(expected))
                valid=np.isfinite(actual)
                np.testing.assert_allclose(actual[valid],expected[valid],rtol=0,atol=2e-5)

    def test_conflicting_surface_class_volume_overlap_is_rejected(self):
        with self.assertRaisesRegex(ValueError,'ambiguous surface ownership'):
            union_surface_from_boxes([(0,0,0,1,1,1,2),(.5,0,0,1,1,1,3)])


if __name__=='__main__':unittest.main()
