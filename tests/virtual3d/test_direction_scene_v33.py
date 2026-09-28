"""Geometry-only V33 checks; no information solver or visibility reward table."""
from collections import deque
import hashlib
import itertools
import json
from pathlib import Path
import unittest
import numpy as np

from nso.direction_scene_v33 import (build_direction_scene_v33, graph_v33,
    prefix_states_v33, rotate_xy_v33, exact_asset_faces_v33, swept_safe_v33)
from nso.box_union_geometry_v30 import BoxV30, union_exterior_faces_v30, surface_audit_v30, union_volume_v30


ROOT = Path(__file__).resolve().parents[2]
CONFIG = ROOT/'configs/virtual3d/v33_direction_scene_20260917.json'
CONFIG_SHA = '4bdcf84cab68e205e160f25182cf2ce09cc221fea9a5593d70b57cf03c213390'


class DirectionSceneV33Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.config = json.loads(CONFIG.read_text())

    def test_config_is_frozen_and_reproducible_with_public_type_only_cues(self):
        self.assertEqual(hashlib.sha256(CONFIG.read_bytes()).hexdigest(), CONFIG_SHA)
        self.assertEqual(self.config, build_direction_scene_v33())
        self.assertEqual(len(self.config['parents']), 2)
        self.assertAlmostEqual(self.config['geometric_observation']['vertical_fov_deg'], 73.73979529168804)
        for parent in self.config['parents']:
            self.assertEqual(parent['total_action_budget'], 42)
            self.assertEqual(parent['prefix_paid_actions'], 18)
            self.assertEqual(parent['remaining_action_budget'], 24)
            for hypothesis in parent['hypotheses']:
                self.assertEqual(set(hypothesis['semantic_cue']), {'type'})
                self.assertIn(hypothesis['semantic_cue']['type'], ('type_A', 'type_B'))
                self.assertEqual(len(hypothesis['assets']), 1)
        a, b = self.config['parents']
        self.assertNotEqual(a['geometry_facts']['body_end_local_y'], b['geometry_facts']['body_end_local_y'])
        self.assertNotEqual(a['geometry_facts']['ribs_local_y'], b['geometry_facts']['ribs_local_y'])
        self.assertEqual([a['device_frame']['quarter_turns_ccw'], b['device_frame']['quarter_turns_ccw']], [0, 1])

    def test_navigation_contains_every_safe_rectangle_center_in_both_hypotheses(self):
        for parent, expected_count in zip(self.config['parents'], (24, 28)):
            grid = parent['local_grid']; turns = parent['device_frame']['quarter_turns_ccw']
            actual = set(map(tuple, parent['nav_cells']))
            self.assertEqual(len(actual), expected_count)
            for hypothesis in parent['hypotheses']:
                boxes = hypothesis['assets'][0]['boxes']
                expected = set()
                for x in range(grid['x_min'], grid['x_max']+1):
                    for y in range(grid['y_min'], grid['y_max']+1):
                        p = np.asarray(rotate_xy_v33((x, y), turns), float)
                        distances = []
                        for box in boxes:
                            nearest = np.clip(p, [box[0], box[2]], [box[1], box[3]])
                            distances.append(float(np.dot(p-nearest, p-nearest)))
                        if min(distances) >= .2**2-1e-7:
                            expected.add(tuple(p))
                self.assertEqual(actual, expected, (parent['id'], hypothesis['id']))
            # In particular, legal inner side-lane centres must not be discarded
            # merely because the original explanatory ring did not include them.
            self.assertIn(rotate_xy_v33((2, 2), turns), actual)

    def test_full_action_graph_connected_safe_and_paid_prefix_returns_exactly(self):
        for parent, expected_poses in zip(self.config['parents'], (96, 112)):
            poses, edges, anchor = graph_v33(parent)
            self.assertEqual(len(poses), expected_poses)
            seen, queue = {anchor}, deque([anchor])
            while queue:
                node = queue.popleft()
                for _, following in edges[node]:
                    if following not in seen:
                        seen.add(following); queue.append(following)
            self.assertEqual(len(seen), len(poses))
            for state, row in zip(poses, edges):
                self.assertTrue({'left', 'right'}.issubset(dict(row)))
                for action, next_index in row:
                    following = poses[next_index]
                    if action == 'forward':
                        self.assertEqual(sum(abs(a-b) for a, b in zip(state[:2], following[:2])), 1)
                        for h in parent['hypotheses']:
                            self.assertTrue(swept_safe_v33(state[:2], following[:2], h['assets'][0]['boxes']))
                    else:
                        self.assertEqual(state[:2], following[:2])
            prefix = prefix_states_v33(parent)
            self.assertEqual(len(prefix), 19)
            self.assertEqual(prefix[0], prefix[-1])
            self.assertEqual(prefix[0], tuple(parent['anchor']))

    def test_closed_exact_material_union_and_mirror_equal_reference_area(self):
        parent_volumes = []
        for parent in self.config['parents']:
            summaries = []
            for hypothesis in parent['hypotheses']:
                asset = hypothesis['assets'][0]
                audit = surface_audit_v30(exact_asset_faces_v33(asset))
                self.assertTrue(audit['closed_oriented_two_manifold'], audit)
                volume = union_volume_v30([BoxV30(tuple(b), 0) for b in asset['boxes']])
                self.assertAlmostEqual(audit['signed_volume'], volume, places=10)
                area = sum(f.area for f in exact_asset_faces_v33(asset, vertical_only=True))
                summaries.append((volume, area))
                # The wide front guard is part of the target asset, not an
                # excluded background object used to hide its area cost.
                self.assertEqual(parent['background_boxes'], [])
                self.assertGreater(area, 5.5*2.)
            np.testing.assert_allclose(summaries[0], summaries[1], atol=1e-10)
            parent_volumes.append(summaries[0][0])
        self.assertNotAlmostEqual(parent_volumes[0], parent_volumes[1])

    def test_complete_footprint_and_laser_cross_section_do_not_encode_type(self):
        for parent in self.config['parents']:
            footprints, radar = [], []
            for hypothesis in parent['hypotheses']:
                boxes = hypothesis['assets'][0]['boxes']
                flat = [BoxV30(tuple(b[:4])+(.0, 1.), 0) for b in boxes]
                footprints.append([(f.bounds, f.axis, f.sign) for f in union_exterior_faces_v30(flat)])
                radar.append(sorted(tuple(b[:4]) for b in boxes if b[4] <= .25 <= b[5]))
                self.assertTrue(all(b[4] == .4 for b in boxes[3:]))
            self.assertEqual(footprints[0], footprints[1])
            self.assertEqual(radar[0], radar[1])

    def test_all_prefix_origins_have_continuous_shield_shadow_on_hidden_ribs(self):
        # Projection onto a fixed positive-depth plane is linear-fractional;
        # extrema over each axis-aligned box occur at corners. Thus this is a
        # box-shadow containment check, not a sparse visibility reward estimate.
        for parent in self.config['parents']:
            turns = parent['device_frame']['quarter_turns_ccw']
            prefix = prefix_states_v33(parent)
            for pose in prefix:
                ox, oy = rotate_xy_v33(pose[:2], -turns)
                self.assertEqual(oy, 0)
                for hypothesis in parent['hypotheses']:
                    for box in hypothesis['assets'][0]['boxes'][3:]:
                        global_corners = itertools.product(box[:2], box[2:4], box[4:6])
                        for gx, gy, z in global_corners:
                            x, y = rotate_xy_v33((gx, gy), -turns)
                            self.assertGreater(y, 1.5)
                            scale = 1.1/y
                            u = ox+(x-ox)*scale
                            projected_z = .9+(z-.9)*scale
                            self.assertGreater(u, -2.75)
                            self.assertLess(u, 2.75)
                            self.assertGreater(projected_z, 0.)
                            self.assertLess(projected_z, 2.)
                # Pose rotation changes FOV but cannot reveal a rib behind an
                # occluder that blocks every origin-to-rib ray.
            seed = parent['hypotheses'][0]['assets'][0]['front_seed_xyz']
            local_seed = (*rotate_xy_v33(seed[:2], -turns), seed[2])
            np.testing.assert_allclose(local_seed, [0., 1.1, .9])
            self.assertEqual(seed, parent['hypotheses'][1]['assets'][0]['front_seed_xyz'])


if __name__ == '__main__':
    unittest.main()
