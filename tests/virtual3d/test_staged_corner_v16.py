from copy import deepcopy
from types import SimpleNamespace
import unittest
import numpy as np
from env.virtual3d import camera_pose
from nso.axis_history_view_v16 import AxisHistoryViewV16
from nso.staged_corner_views_v16 import select_corner_view, observed_support_fraction


def fixture():
    config = SimpleNamespace(resolution_m=.2, robot_radius_m=.2,
        camera_height_m=.8, fov_deg=90., max_depth_m=4., width_px=96, height_px=72)
    mapper = SimpleNamespace(shape=(30, 30), config=config, belief=np.zeros((30, 30), np.int8))
    points = np.array([[x, 3.1, z] for x in np.linspace(2.1, 3.1, 11) for z in (.6, .8, 1.)])
    asset = dict(aabb_center=np.array([2.6, 3.1, .8]), front_axis=np.array([0., -1.]),
        side_axis=np.array([-1., 0.]), measured_depth_m=0., measured_width_m=1., points=points)
    pool = []
    for r in range(8, 22):
        for c in range(6, 21):
            for heading in range(4):
                pose = (r, c, heading)
                pool.append(dict(state=len(pool), pose=pose, cost=30,
                    camera=camera_pose(pose[:2], heading, config, 30)))
    return mapper, asset, pool


class StagedCornerTests(unittest.TestCase):
    def test_both_visible_face_ends_with_no_label_dependency(self):
        mapper, asset, pool = fixture()
        results = []
        for sign in (-1, 1):
            row, audit = select_corner_view(asset, sign, pool, set(), (22, 13, 0), mapper)
            self.assertIsNotNone(row)
            self.assertLessEqual(audit['nearest_error_m'], .35)
            self.assertGreater(audit['measured_grid_visible_fraction'], 0.)
            self.assertFalse(audit['hidden_surface_visibility_claimed'])
            altered = deepcopy(asset); altered.update(class_vote=99., marked_points=0, hidden_depth=100.)
            other, other_audit = select_corner_view(altered, sign, pool, set(), (22, 13, 0), mapper)
            self.assertEqual(row['pose'], other['pose']); self.assertEqual(audit, other_audit)
            results.append(row['camera'][:2, 3])
        self.assertGreater(results[0][0], 3.1)
        self.assertLess(results[1][0], 2.1)
        self.assertTrue(all(xy[1] < 3.1 for xy in results))

    def test_unknown_and_occluded_support_are_rejected(self):
        mapper, asset, pool = fixture()
        row, _ = select_corner_view(asset, -1, pool, set(), (22, 13, 0), mapper)
        self.assertGreater(observed_support_fraction(asset['points'], row, mapper), 0.)
        # A complete intervening horizontal barrier blocks the actual surface.
        for occupancy in (-1, 1):
            mapper.belief[15, :] = occupancy
            self.assertEqual(observed_support_fraction(asset['points'], row, mapper), 0.)
        mapper.belief[:] = 0
        # Use a straight ray for the endpoint case. A diagonal ray into a
        # completely occupied row can correctly hit its adjacent cell first.
        straight = dict(row)
        straight['camera'] = camera_pose(row['pose'][:2], 0, mapper.config, 30)
        endpoint = np.array([[straight['camera'][0, 3], 3.1, .8]])
        mapper.belief[14, :] = 1
        self.assertEqual(observed_support_fraction(endpoint, straight, mapper), 1.)

    def test_out_of_frustum_and_unreachable_roles_are_not_filled(self):
        mapper, asset, pool = fixture()
        row, _ = select_corner_view(asset, 1, pool, set(), (22, 13, 0), mapper)
        away = deepcopy(row)
        away['camera'] = camera_pose(row['pose'][:2], (row['pose'][2]+2) % 4, mapper.config, 30)
        self.assertEqual(observed_support_fraction(asset['points'], away, mapper), 0.)
        empty, audit = select_corner_view(asset, 1, [], set(), (22, 13, 0), mapper)
        self.assertIsNone(empty); self.assertNotEqual(audit['reason'], 'selected')
        empty, _ = select_corner_view(asset, 1, [row], {row['state']}, (22, 13, 0), mapper)
        self.assertIsNone(empty)
        empty, _ = select_corner_view(asset, 1, [row], set(), row['pose'], mapper)
        self.assertIsNone(empty)

    def test_complete_axis_history_does_not_mutate_retained_keyframes(self):
        frames = [SimpleNamespace(timestamp_s=float(i)) for i in range(3)]
        mapper = SimpleNamespace(keyframes=[frames[0]], frames=3)
        view = AxisHistoryViewV16(mapper, frames)
        self.assertEqual(len(view.keyframes), 3); self.assertEqual(view.frames, 3)
        self.assertEqual(mapper.keyframes, [frames[0]])
        for invalid in ([], frames[::-1], [frames[0], frames[0]]):
            with self.assertRaises(ValueError): AxisHistoryViewV16(mapper, invalid)


if __name__ == '__main__': unittest.main()
