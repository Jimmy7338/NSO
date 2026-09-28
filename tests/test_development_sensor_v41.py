"""Only analytical triangles and pure motion fixtures; no DEV World bypass."""
from dataclasses import fields
import math
from pathlib import Path
import unittest
from unittest.mock import patch

import numpy as np

from env.development_sensor_v41 import (DevelopmentSensorV41, GIB, ResourceGateBlocked,
    add_relative_depth_noise, camera_transform_xyyaw, create_development_sensor,
    first_hit_triangles, paid_motion_transition, propose_motion, render_planar_scan,
    render_rgbd_arrays, return_pose_matches, runtime_counts_v41, storage_report_v41,
    swept_circle_collision, wrap_yaw)
from nso.instance_belief_v40 import PaidRGBDObservationV40


def z_plane(depth):
    return (np.asarray([[-10., -10., depth], [10., -10., depth], [10., 10., depth], [-10., 10., depth]]),
            np.asarray([[0, 2, 1], [0, 3, 2]], np.int32))


def x_plane(distance):
    return (np.asarray([[distance, -10., -2.], [distance, 10., -2.], [distance, 10., 2.], [distance, -10., 2.]]),
            np.asarray([[0, 1, 2], [0, 2, 3]], np.int32))


def combine(*meshes):
    vertices, triangles, count = [], [], 0
    for v, t in meshes:
        vertices.append(v); triangles.append(t+count); count += len(v)
    return np.concatenate(vertices), np.concatenate(triangles)


def fixture_render(v, t, **kwargs):
    defaults = dict(intrinsic=np.eye(3), world_from_camera=np.eye(4), width=2, height=1,
                    minimum_depth_m=.1, maximum_depth_m=4., relative_sigma=0., noise_seed=7, step=0)
    defaults.update(kwargs)
    return render_rgbd_arrays(v, t, **defaults)


def marker():
    return {"instance_id": 3, "category": "synthetic_fixture_class", "center_world_m": [0., 0., 2.],
            "normal_world": [0., 0., -1.], "u_world": [1., 0., 0.], "v_world": [0., -1., 0.],
            "width_m": .4, "height_m": .4, "rgb": [40, 100, 220]}


class DevelopmentSensorV41Tests(unittest.TestCase):
    def test_camera_optical_axes_and_yaw_convention(self):
        t = camera_transform_xyyaw([1., 2., 0.], .9)
        np.testing.assert_allclose(t[:3, 0], [0., -1., 0.])
        np.testing.assert_allclose(t[:3, 1], [0., 0., -1.])
        np.testing.assert_allclose(t[:3, 2], [1., 0., 0.])
        np.testing.assert_allclose(t[:3, 3], [1., 2., .9])
        left = camera_transform_xyyaw([0., 0., math.pi/2])
        np.testing.assert_allclose(left[:3, 2], [0., 1., 0.], atol=1e-12)
        self.assertAlmostEqual(np.linalg.det(t[:3, :3]), 1.)

    def test_continuous_forward_and_thirty_degree_turns(self):
        pose = propose_motion([1., 2., 0.], "turn_left")
        self.assertAlmostEqual(pose[2], math.pi/6)
        pose = propose_motion(pose, "forward")
        np.testing.assert_allclose(pose[:2], [1+.25*math.cos(math.pi/6), 2+.125])
        pose = propose_motion(pose, "turn_right")
        self.assertAlmostEqual(pose[2], 0.)

    def test_twelve_turns_return_yaw_without_teleport(self):
        start = [1.1, 2.2, .07]
        pose = start
        for _ in range(12):
            pose = propose_motion(pose, "turn_left")
        self.assertTrue(return_pose_matches(pose, start))
        self.assertFalse(return_pose_matches([1.1, 2.2, .07+math.pi/6], start))
        self.assertFalse(return_pose_matches([1.2, 2.2, .07], start))

    def test_diagonal_swept_collision_detects_middle_not_only_endpoints(self):
        square = [[.9, .9], [1.1, .9], [1.1, 1.1], [.9, 1.1]]
        self.assertTrue(swept_circle_collision([0., 0.], [2., 2.], [square], .1))
        self.assertFalse(swept_circle_collision([0., 0.], [0., 0.], [square], .1))
        self.assertFalse(swept_circle_collision([2., 2.], [2., 2.], [square], .1))

    def test_capsule_radius_corner_clearance_and_tangency(self):
        square = [[1., 1.], [2., 1.], [2., 2.], [1., 2.]]
        self.assertFalse(swept_circle_collision([0., .85], [3., .85], [square], .1))
        self.assertTrue(swept_circle_collision([0., .85], [3., .85], [square], .2))
        self.assertTrue(swept_circle_collision([2.2, 1.5], [2.2, 1.5], [square], .2))
        self.assertTrue(swept_circle_collision([1.5, 1.5], [1.5, 1.5], [square], 0.))

    def test_paid_collision_attempt_costs_one_and_preserves_pose(self):
        obstacle = [[.3, -.5], [.4, -.5], [.4, .5], [.3, .5]]
        receipt = paid_motion_transition([0., 0., 0.], "forward", paid_step=0, max_actions=3,
            start_pose=[0., 0., 0.], obstacle_polygons=[obstacle], radius_m=.1)
        self.assertTrue(receipt["collision"])
        self.assertEqual(receipt["paid_step"], 1)
        self.assertEqual(receipt["action_cost"], 1)
        np.testing.assert_array_equal(receipt["pose_xyyaw_rad"], [0., 0., 0.])

    def test_turn_observe_and_forward_have_same_unit_charge(self):
        for action in ("forward", "turn_left", "turn_right", "observe"):
            receipt = paid_motion_transition([0., 0., 0.], action, paid_step=2, max_actions=3,
                start_pose=[0., 0., 0.], obstacle_polygons=[])
            self.assertEqual((receipt["paid_step"], receipt["action_cost"]), (3, 1))
        with self.assertRaisesRegex(ValueError, "exhausted"):
            paid_motion_transition([0., 0., 0.], "observe", paid_step=3, max_actions=3,
                start_pose=[0., 0., 0.], obstacle_polygons=[])

    def test_stop_is_not_a_free_capture_action(self):
        with self.assertRaises(ValueError):
            propose_motion([0., 0., 0.], "stop")

    def test_first_hit_is_nearest_and_independent_of_triangle_chunks(self):
        v, t = combine(z_plane(3.), z_plane(1.))
        rays = np.asarray([[0., 0., 1.], [1., 0., 0.]])
        first = first_hit_triangles([0., 0., 0.], rays, v, t, max_distance_m=4., triangle_chunk=1, ray_chunk=1)
        second = first_hit_triangles([0., 0., 0.], rays, v, t, max_distance_m=4., triangle_chunk=32)
        np.testing.assert_array_equal(first[0], second[0])
        np.testing.assert_array_equal(first[1], second[1])
        self.assertEqual(first[0][0], 1.)
        self.assertTrue(math.isinf(first[0][1]))
        self.assertEqual(first[1][1], -1)

    def test_ray_direction_contract_and_range(self):
        v, t = z_plane(2.)
        with self.assertRaises(ValueError):
            first_hit_triangles([0., 0., 0.], [[0., 0., 2.]], v, t, max_distance_m=4.)
        distance, identity = first_hit_triangles([0., 0., 0.], [[0., 0., 1.]], v, t, max_distance_m=1.)
        self.assertTrue(math.isinf(distance[0])); self.assertEqual(identity[0], -1)

    def test_rgbd_range_is_optical_z_not_radial_distance(self):
        v, t = z_plane(3.9)
        _, depth = fixture_render(v, t)
        np.testing.assert_allclose(depth, [[3.9, 3.9]], atol=1e-6)
        # Pixel x=1 is at radial distance 3.9*sqrt(2)>4, still valid axial depth.
        self.assertGreater(3.9*math.sqrt(2), 4.)

    def test_near_surface_occludes_far_surface_even_when_depth_clipped(self):
        v, t = combine(z_plane(.05), z_plane(2.))
        rgb, depth = fixture_render(v, t)
        self.assertFalse(depth.any()); self.assertFalse(rgb.any())
        rgb, depth = fixture_render(*z_plane(4.1))
        self.assertFalse(depth.any()); self.assertFalse(rgb.any())

    def test_marker_only_colors_its_existing_face_patch(self):
        v, t = z_plane(2.)
        rgb, depth = fixture_render(v, t, marker_patches=[marker()], triangle_instance_id=np.asarray([3, 3]))
        np.testing.assert_array_equal(rgb[0, 0], [40, 100, 220])
        np.testing.assert_array_equal(rgb[0, 1], [165, 165, 165])
        np.testing.assert_array_equal(depth, [[2., 2.]])

    def test_hidden_decal_not_visible_through_foreground(self):
        v, t = combine(z_plane(1.), z_plane(2.))
        rgb, _ = fixture_render(v, t, marker_patches=[marker()], triangle_instance_id=np.asarray([-1, -1, 3, 3]))
        np.testing.assert_array_equal(rgb, np.full((1, 2, 3), 165, np.uint8))

    def test_wrong_owner_or_floating_patch_cannot_paint_semantic_color(self):
        v, t = z_plane(2.)
        rgb, _ = fixture_render(v, t, marker_patches=[marker()], triangle_instance_id=np.asarray([7, 7]))
        np.testing.assert_array_equal(rgb[0, 0], [165, 165, 165])
        floating = marker(); floating["center_world_m"][2] = 1.99
        rgb, _ = fixture_render(v, t, marker_patches=[floating], triangle_instance_id=np.asarray([3, 3]))
        np.testing.assert_array_equal(rgb[0, 0], [165, 165, 165])

    def test_noise_is_seeded_per_paid_step_and_does_not_change_global_rng(self):
        clean = np.full((8, 16), 2.)
        state = np.random.get_state()
        a = add_relative_depth_noise(clean, relative_sigma=.01, noise_seed=9, step=4)
        b = add_relative_depth_noise(clean, relative_sigma=.01, noise_seed=9, step=4)
        c = add_relative_depth_noise(clean, relative_sigma=.01, noise_seed=9, step=5)
        d = add_relative_depth_noise(clean, relative_sigma=.01, noise_seed=10, step=4)
        np.testing.assert_array_equal(a, b)
        self.assertFalse(np.array_equal(a, c)); self.assertFalse(np.array_equal(a, d))
        current = np.random.get_state()
        self.assertEqual(state[0], current[0]); np.testing.assert_array_equal(state[1], current[1])
        self.assertEqual(state[2:], current[2:])

    def test_noise_preserves_invalid_pixels_and_declared_range(self):
        clean = np.asarray([[0., .05, 2., 5.]])
        depth = add_relative_depth_noise(clean, relative_sigma=5., noise_seed=7, step=0)
        self.assertEqual(depth[0, 0], 0.); self.assertEqual(depth[0, 1], 0.); self.assertEqual(depth[0, 3], 0.)
        self.assertTrue(np.all((depth == 0) | ((depth >= .1) & (depth <= 4.))))

    def test_laser_reports_radial_range_and_360_degree_no_hit(self):
        scan = render_planar_scan(*x_plane(2.), [0., 0., 0.], number_of_rays=8)
        self.assertAlmostEqual(float(scan.ranges_m[4]), 2.)
        self.assertAlmostEqual(float(scan.ranges_m[5]), 2*math.sqrt(2), places=6)
        self.assertEqual(scan.ranges_m[0], 8.)
        self.assertAlmostEqual(scan.angle_increment_rad, math.pi/4)
        self.assertFalse(scan.ranges_m.flags.writeable)

    def test_laser_pose_rotation_is_independent_of_camera_axes(self):
        scan = render_planar_scan(*x_plane(2.), [1., 0., math.pi/2], number_of_rays=4)
        np.testing.assert_allclose(scan.world_from_laser[:3, 0], [0., 1., 0.], atol=1e-12)
        np.testing.assert_allclose(scan.world_from_laser[:3, 3], [1., 0., .3])
        self.assertAlmostEqual(float(scan.ranges_m[1]), 1.)

    def test_paid_observation_whitelist_contains_no_renderer_identity(self):
        rgb, depth = fixture_render(*z_plane(2.))
        observation = PaidRGBDObservationV40("analytic-fixture-only", 0, rgb, depth, np.eye(3), np.eye(4))
        self.assertEqual({field.name for field in fields(observation)},
            {"frame_id", "paid_step", "rgb", "depth_m", "intrinsic", "world_from_camera"})
        self.assertFalse(observation.rgb.flags.writeable)
        self.assertFalse(observation.depth_m.flags.writeable)

    def test_constructor_and_factory_block_before_asset_access(self):
        before = runtime_counts_v41()
        # Astronomical declared peak guarantees failure without substituting
        # the real disk gate or constructing an actual development World.
        with patch("env.development_sensor_v41.load_json_strict", side_effect=AssertionError("asset opened before gate")):
            for constructor in (DevelopmentSensorV41, create_development_sensor):
                with self.assertRaises(ResourceGateBlocked) as context:
                    constructor("must-not-open/DEV_A_00", {}, episode_id="fixture-gate", noise_seed=1,
                                persistent_output_root=Path(__file__).resolve().parents[1], expected_batch_peak_bytes=1 << 80)
                self.assertEqual(context.exception.report["status"], "blocked_before_world_creation")
        after = runtime_counts_v41()
        for key in ("worlds_created", "rgbd_frames", "scans", "paid_actions"):
            self.assertEqual(before[key], after[key])
        self.assertEqual(after["blocked_before_world_creation"], before["blocked_before_world_creation"]+2)

    def test_storage_threshold_and_ram_are_not_bypass_options(self):
        report = storage_report_v41(Path(__file__).resolve().parents[1], 1*GIB)
        self.assertEqual(report["required_free_bytes"], 10*GIB)
        larger = storage_report_v41(Path(__file__).resolve().parents[1], 6*GIB)
        self.assertEqual(larger["required_free_bytes"], 14*GIB)
        ram = storage_report_v41("/dev/shm", 0)
        self.assertFalse(ram["passed"])
        self.assertFalse(ram["non_ram_filesystem"])

    def test_helpers_do_not_increment_study_world_or_frame_counts(self):
        before = runtime_counts_v41()
        fixture_render(*z_plane(2.))
        render_planar_scan(*x_plane(2.), [0., 0., 0.], number_of_rays=4)
        self.assertEqual(before, runtime_counts_v41())

    def test_invalid_calibration_or_polygon_rejected(self):
        v, t = z_plane(2.)
        with self.assertRaises(ValueError):
            fixture_render(v, t, intrinsic=np.zeros((3, 3)))
        with self.assertRaises(ValueError):
            swept_circle_collision([0., 0.], [1., 1.], [[[0., 0.], [1., 1.], [0., 1.], [1., 0.]]])


if __name__ == "__main__":
    unittest.main()
