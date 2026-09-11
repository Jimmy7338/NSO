import unittest
from types import SimpleNamespace
import numpy as np
from env.virtual3d import camera_pose
from env.virtual3d_v2 import VirtualConfigV2
from nso.response_features_v7 import state_patch_descriptor, observing_camera, observed_patches
from nso.response_candidates_v7 import candidate_routes
from utils.grid_geometry import inflated_obstacles
from utils.rgbd_contract import RGBDFrame


class MeasuredPatch:
    def __init__(self, category=2):
        self.config = VirtualConfigV2(width_m=6.4, height_m=6.4, voxel_m=.03,
                                      dropout=0., depth_sigma_m=.01)
        self.shape = (32, 32)
        self.belief = np.zeros(self.shape, np.int8)
        self.belief[[0, -1], :] = 1; self.belief[:, [0, -1]] = 1
        self.belief[21:25, 9:16] = 1
        self.camera_seen = np.zeros(self.shape, bool)
        points = np.array([[x, 2.1, z] for x in np.linspace(2., 2.8, 7)
                           for z in np.linspace(.3, 1.4, 8)])
        self.q = {'point': points, 'normal': np.tile([0., -1., 0.], (len(points), 1)),
                  'bits': np.full(len(points), 1 << 6), 'label': np.full(len(points), category)}
        self.position = (16, 11)
        pose = camera_pose(self.position, 2, self.config, 32)
        f = self.config.width_px / 2
        intrinsic = np.array([[f, 0, (self.config.width_px-1)/2],
                              [0, f, (self.config.height_px-1)/2], [0, 0, 1]])
        self.keyframes = [RGBDFrame(0., np.ones((72, 96), np.float32),
             np.zeros((72, 96, 3), np.uint8), intrinsic, pose, np.zeros((72, 96), np.uint8))]

    def quality_evidence(self, max_points=10000):
        return self.q


class PhysicalDirectionTests(unittest.TestCase):
    def test_front_side_back_and_seen_direction_have_physical_values(self):
        c = VirtualConfigV2(width_m=6.4, height_m=6.4)
        patch = {'points': np.array([[2.1, 2.1, .8]]),
                 'normal_out_xy': np.array([0., 1.]), 'bits': np.array([1 << 6])}
        front = state_patch_descriptor(patch, camera_pose((16, 10), 2, c, 32), c)[0]
        side = state_patch_descriptor(patch, camera_pose((21, 15), 3, c, 32), c)[0]
        back = state_patch_descriptor(patch, camera_pose((26, 10), 0, c, 32), c)[0]
        factor = 16/17
        np.testing.assert_allclose(front, [factor, 0, 0, 0], atol=1e-7)
        np.testing.assert_allclose(side, [0, factor, 0, factor], atol=1e-7)
        np.testing.assert_allclose(back, [0, 0, factor, factor], atol=1e-7)

    def test_direction_novelty_uses_each_surface_point(self):
        c = VirtualConfigV2(width_m=6.4, height_m=6.4)
        patch = {'points': np.array([[2.1, 2.05, .8], [2.1, 2.15, .8]]),
                 'normal_out_xy': np.array([1., 0.]), 'bits': np.array([1 << 4, 1 << 3])}
        d = state_patch_descriptor(patch, camera_pose((21, 15), 3, c, 32), c)
        np.testing.assert_array_equal(d[:, 3], [0, 0])

    def test_normal_orientation_uses_a_depth_supported_view(self):
        m = MeasuredPatch()
        good = m.keyframes[0]
        bad = RGBDFrame(good.timestamp_s, np.zeros_like(good.depth_m), good.color_rgb,
                        good.intrinsic, good.world_from_camera, good.semantic)
        bad_pose = bad.world_from_camera.copy(); bad_pose[0, 3] += 1.
        bad = RGBDFrame(0., bad.depth_m, bad.color_rgb, bad.intrinsic, bad_pose, bad.semantic)
        m.keyframes = [bad, good]
        position, count = observing_camera(m, m.q['point'])
        np.testing.assert_array_equal(position, good.world_from_camera[:3, 3])
        self.assertGreater(count, 0)

    def test_absent_depth_support_cannot_assign_directional_response(self):
        m = MeasuredPatch(); frame = m.keyframes[0]
        m.keyframes = [RGBDFrame(0., np.zeros_like(frame.depth_m), frame.color_rgb,
                                frame.intrinsic, frame.world_from_camera, frame.semantic)]
        patches = observed_patches(m)
        self.assertTrue(patches)
        for patch in patches:
            self.assertFalse(patch['orientation_available'])
            d = state_patch_descriptor(patch, frame.world_from_camera, m.config)
            np.testing.assert_array_equal(d[:, :3], np.zeros_like(d[:, :3]))
        _, audit = candidate_routes(m, SimpleNamespace(position=m.position, heading=2))
        self.assertEqual(audit['missing_direction_roles'], ['left', 'right', 'back', 'front'])


class RouteContractTests(unittest.TestCase):
    def test_labels_cannot_change_shared_routes_and_paid_footprint(self):
        outputs = []
        for category in (2, 3, 0):
            m = MeasuredPatch(category)
            obs = SimpleNamespace(position=m.position, heading=2)
            before = m.belief.copy()
            routes, audit = candidate_routes(m, obs)
            outputs.append((routes, audit))
            np.testing.assert_array_equal(before, m.belief)
            self.assertEqual(len(routes), 6)
            safe = ~inflated_obstacles(before != 0, m.config.robot_radius_m/.2)
            for r in routes:
                self.assertEqual(r['states'][0], r['states'][-1])
                self.assertEqual(r['cost'], len(r['actions']))
                self.assertLessEqual(r['cost'], 48)
                self.assertTrue(all(safe[tuple(s[:2])] for s in r['states']))
        self.assertEqual(outputs[0], outputs[1])
        self.assertEqual(outputs[0], outputs[2])

    def test_unknown_start_footprint_is_not_cleared(self):
        m = MeasuredPatch(); m.belief[m.position[0], m.position[1]+1] = -1
        routes, audit = candidate_routes(m, SimpleNamespace(position=m.position, heading=2))
        self.assertEqual(routes, [])
        self.assertEqual(audit['status'], 'initial_footprint_unknown_or_blocked')


if __name__ == '__main__':
    unittest.main()
