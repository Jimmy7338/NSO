"""Evaluator visibility/accounting tests, not planning efficacy evidence."""
from types import SimpleNamespace
import unittest

import numpy as np
import open3d as o3d

from utils.counterfactual_surface_visibility import (
    reference_visible, surface_increment, surface_stage_increments)
from utils.reconstruction_metrics import ray_scene


def plane(z, half_width=3.):
    vertices = [[-half_width, -half_width, z], [half_width, -half_width, z],
                [half_width, half_width, z], [-half_width, half_width, z]]
    return o3d.geometry.TriangleMesh(o3d.utility.Vector3dVector(vertices),
        o3d.utility.Vector3iVector([[0, 1, 2], [0, 2, 3]]))


def frame(depth=None, pose=None):
    # Deliberately no RGB/semantic properties: evaluator must not require them.
    return SimpleNamespace(depth_m=np.full((5, 5), 2.) if depth is None else depth,
        intrinsic=np.array([[2., 0., 2.], [0., 2., 2.], [0., 0., 1.]]),
        world_from_camera=np.eye(4) if pose is None else pose)


class SurfaceVisibilityTests(unittest.TestCase):
    def test_visible_surface_and_background_occlusion_use_first_truth_hit(self):
        truth = ray_scene(plane(2.) + plane(3.))
        points = np.array([[0., 0., 2.], [0., 0., 3.], [.7, .1, 2.]])
        mask = reference_visible(points, frame(), truth, max_depth_m=4.)
        np.testing.assert_array_equal(mask, [True, False, True])

    def test_zero_nan_and_infinite_depth_do_not_receive_visibility(self):
        truth = ray_scene(plane(2.))
        points = np.array([[-1., 0., 2.], [0., 0., 2.], [1., 0., 2.], [0., 1., 2.]])
        depth = np.full((5, 5), 2.)
        depth[2, 1], depth[2, 2], depth[2, 3] = 0., np.nan, np.inf
        np.testing.assert_array_equal(reference_visible(points, frame(depth), truth,
                                      max_depth_m=4.), [False, False, False, True])

    def test_projection_near_plane_range_and_pixel_support(self):
        truth = ray_scene(plane(2., half_width=10.))
        points = np.array([[0., 0., 2.], [2.49, 0., 2.], [2.5, 0., 2.],
                           [0., 3., 2.], [0., 0., -.1], [0., 0., .15]])
        mask = reference_visible(points, frame(), truth, max_depth_m=2.)
        np.testing.assert_array_equal(mask, [True, True, False, False, False, False])
        # Off-axis point exceeds 2m Euclidean range but has valid axial z=2m.
        self.assertGreater(np.linalg.norm(points[1]), 2.)
        self.assertFalse(reference_visible(points[:1], frame(), truth, max_depth_m=1.9).any())

    def test_explicit_physical_pose_overrides_erroneous_estimated_pose(self):
        estimated = np.eye(4)
        estimated[0, 3] = 100.
        points = np.array([[0., 0., 2.]])
        truth = ray_scene(plane(2.))
        self.assertFalse(reference_visible(points, frame(pose=estimated), truth, max_depth_m=4.)[0])
        self.assertTrue(reference_visible(points, frame(pose=estimated), truth,
                       physical_world_from_camera=np.eye(4), max_depth_m=4.)[0])

    def test_physical_rotation_and_translation_are_respected(self):
        pose = np.eye(4)
        pose[:3, :3] = [[0., 0., 1.], [0., 1., 0.], [-1., 0., 0.]]
        pose[:3, 3] = [3., 2., 1.]
        mesh = plane(2.).transform(pose)
        point = np.array([[5., 2., 1.]])
        self.assertTrue(reference_visible(point, frame(), ray_scene(mesh), pose, 4.)[0])

    def test_existing_relative_first_hit_tolerance_is_preserved(self):
        truth = ray_scene(plane(2.))
        points = np.array([[0., 0., 2.0001], [0., 0., 2.001]])
        np.testing.assert_array_equal(reference_visible(points, frame(), truth,
                                      max_depth_m=4.), [True, False])

    def test_depth_agreement_is_not_a_reconstruction_test(self):
        truth = ray_scene(plane(2.))
        # Positive in-range noisy pixel is valid; no 3D-accuracy credit is implied.
        self.assertTrue(reference_visible(np.array([[0., 0., 2.]]),
                        frame(np.full((5, 5), 2.08)), truth, max_depth_m=4.)[0])
        self.assertFalse(reference_visible(np.array([[0., 0., 2.]]),
                         frame(np.full((5, 5), 5.)), truth, max_depth_m=4.)[0])

    def test_empty_reference_needs_no_raycast(self):
        mask = reference_visible(np.empty((0, 3)), frame(), None, max_depth_m=4.)
        self.assertEqual(mask.shape, (0,))
        self.assertEqual(mask.dtype, np.bool_)

    def test_bad_reference_pose_range_and_calibration_are_rejected(self):
        truth = ray_scene(plane(2.))
        with self.assertRaises(ValueError):
            reference_visible([[np.nan, 0., 2.]], frame(), truth)
        bad_pose = np.eye(4); bad_pose[0, 0] = 2.
        with self.assertRaises(ValueError):
            reference_visible([[0., 0., 2.]], frame(), truth, bad_pose)
        for maximum in (0., .15, np.nan, np.inf):
            with self.assertRaises(ValueError):
                reference_visible([[0., 0., 2.]], frame(), truth, max_depth_m=maximum)
        bad_frame = frame(); bad_frame.intrinsic[0, 0] = 0.
        with self.assertRaises(ValueError):
            reference_visible([[0., 0., 2.]], bad_frame, truth)

    def test_area_weights_and_seen_union_are_not_renormalised(self):
        before = np.array([True, False, False, False])
        future = np.array([True, True, False, True])
        self.assertAlmostEqual(surface_increment(before, future, [4., 1., 9., 2.]), 3.)
        # Ten initial physical samples, only four retained: each still weighs 0.2.
        self.assertAlmostEqual(surface_increment(before, future, 2. / 10), .4)
        self.assertEqual(surface_increment(before | future, future, 2. / 10), 0.)

    def test_stage_decomposition_telescopes_without_mutating_inputs(self):
        prefix = np.array([True, False, False, False])
        outbound = np.array([True, True, False, False])
        endpoint = np.array([False, True, True, False])
        returning = np.array([True, False, True, True])
        original = [value.copy() for value in (prefix, outbound, endpoint, returning)]
        result = surface_stage_increments(prefix, [outbound, endpoint, returning], [7., 1., 2., 3.])
        self.assertEqual(result['increments'], [1., 2., 3.])
        self.assertEqual(result['total'], 6.)
        self.assertEqual(sum(result['increments']), result['total'])
        np.testing.assert_array_equal(result['union_seen'], np.ones(4, bool))
        reverse = surface_stage_increments(prefix, [returning, endpoint, outbound], [7., 1., 2., 3.])
        self.assertEqual(reverse['total'], result['total'])
        for actual, expected in zip((prefix, outbound, endpoint, returning), original):
            np.testing.assert_array_equal(actual, expected)

    def test_accounting_rejects_mismatched_reference_or_invalid_weights(self):
        prefix = np.array([True, False])
        for weights in ([-1., 1.], [1., np.nan], [1.], np.inf):
            with self.assertRaises(ValueError):
                surface_increment(prefix, ~prefix, weights)
        with self.assertRaises(ValueError):
            surface_increment(prefix, np.array([True]), 1.)
        with self.assertRaises(ValueError):
            surface_increment([1, 0], [0, 1], 1.)
        with self.assertRaises(ValueError):
            surface_stage_increments(prefix, [np.array([True])], 1.)
        with self.assertRaises(ValueError):
            surface_increment(np.empty(0, bool), np.empty(0, bool), np.inf)
        empty = surface_stage_increments(prefix, [], 1.)
        self.assertEqual(empty['total'], 0.)
        np.testing.assert_array_equal(empty['union_seen'], prefix)


if __name__ == '__main__':
    unittest.main()
