"""Feasibility contracts using synthetic measured maps, not future outcomes."""
from collections import deque
from copy import deepcopy
from types import SimpleNamespace
import unittest

import numpy as np

from env.virtual3d import camera_pose
from nso.hierarchical_options_v10 import generate_options
from utils.rgbd_contract import RGBDFrame


class MeasuredMapperFixture:
    def __init__(self, shape=(11, 15)):
        self.shape = shape
        self.config = SimpleNamespace(resolution_m=.2, robot_radius_m=.2,
                                      camera_height_m=.8, fov_deg=90., max_depth_m=4.,
                                      width_px=96, height_px=72, truncation_m=.12)
        self.belief = np.zeros(shape, dtype=np.int8)
        self.camera_seen = np.ones(shape, dtype=bool)
        self.keyframes = []
        self._q = None

    def quality_evidence(self, max_points=10000):
        return self._q

    @property
    def world(self):
        raise AssertionError('candidate generator must not read a world')

    @property
    def ground_truth(self):
        raise AssertionError('candidate generator must not read truth')


def known_safe_direct(mapper):
    # Exact direct stencil for this fixture's radius/resolution=1. No production
    # inflation or graph helper is used by the assertions below.
    free = mapper.belief == 0
    padded = np.pad(free, 1, constant_values=False)
    safe = np.ones_like(free)
    for dr in range(3):
        for dc in range(3): safe &= padded[dr:dr+free.shape[0], dc:dc+free.shape[1]]
    return safe


def transition(state, action):
    r, c, h = state
    if action == 'left': return r, c, (h-1) % 4
    if action == 'right': return r, c, (h+1) % 4
    if action != 'forward': raise AssertionError(action)
    dr, dc = ((-1, 0), (0, 1), (1, 0), (0, -1))[h]
    return r+dr, c+dc, h


def bfs_cost(safe, start, end):
    queue = deque([(start, 0)]); visited = {start}
    while queue:
        state, cost = queue.popleft()
        if state == end: return cost
        for action in ('left', 'right', 'forward'):
            nxt = transition(state, action)
            if (0 <= nxt[0] < safe.shape[0] and 0 <= nxt[1] < safe.shape[1]
                    and safe[nxt[:2]] and nxt not in visited):
                visited.add(nxt); queue.append((nxt, cost+1))
    return None


def objects_fixture(count):
    mapper = MeasuredMapperFixture((28, 46))
    points, labels = [], []
    centers = (1.5, 4.5, 7.5)[:count]
    for index, cx in enumerate(centers):
        horizontal = [(x, 3.1) for x in np.linspace(cx-.4, cx+.4, 5)]
        horizontal += [(cx+sign*.4, y) for sign in (-1, 1) for y in (3.3, 3.5, 3.7, 3.9)]
        cloud = [[x, y, z] for x, y in horizontal for z in np.linspace(.3, 1.5, 8)]
        points.extend(cloud); labels.extend([2+index % 2] * len(cloud))
        for r in range(mapper.shape[0]):
            for c in range(mapper.shape[1]):
                x, y = (c+.5)*.2, (mapper.shape[0]-r-.5)*.2
                if abs(x-cx) <= .4+1e-12 and 3.1-1e-12 <= y <= 3.9+1e-12:
                    mapper.belief[r, c] = 1
        position = (mapper.shape[0]-1-int(np.floor(.7/.2)), int(np.floor(cx/.2)))
        intrinsic = np.array([[48., 0., 47.5], [0., 48., 35.5], [0., 0., 1.]])
        mapper.keyframes.append(RGBDFrame(float(index), np.full((72, 96), 2.4),
                                          np.zeros((72, 96, 3), np.uint8), intrinsic,
                                          camera_pose(position, 0, mapper.config, mapper.shape[0]),
                                          np.zeros((72, 96), np.uint8)))
    if count:
        mapper._q = {'point': np.asarray(points), 'normal': np.tile([0., -1., 0.], (len(points), 1)),
                     'label': np.asarray(labels), 'bits': np.zeros(len(points), np.uint8)}
    current = (mapper.shape[0]-1-int(np.floor(.7/.2)), int(np.floor(4.5/.2)), 0)
    return mapper, current


class HierarchicalOptionsV10Tests(unittest.TestCase):
    def assert_options_valid(self, mapper, current, anchor, budget, routes):
        safe = known_safe_direct(mapper)
        self.assertEqual([r['candidate_id'] for r in routes], list(range(len(routes))))
        self.assertEqual(len({tuple(r['pose']) for r in routes}), len(routes))
        for route in routes:
            with self.subTest(role=route['group']):
                states = [tuple(state) for state in route['states']]
                self.assertEqual(states[0], current)
                self.assertEqual(states[-1], anchor)
                self.assertEqual(tuple(route['return_anchor']), anchor)
                self.assertGreater(route['outbound_cost'], 0)
                self.assertNotEqual(tuple(route['pose']), current)
                self.assertEqual(route['cost'], route['outbound_cost']+route['return_cost'])
                self.assertEqual(route['cost'], len(route['actions']))
                self.assertEqual(route['actions'], route['outbound_actions']+route['return_actions'])
                self.assertEqual(route['arrival_action'], route['outbound_cost'])
                self.assertEqual(states[route['arrival_action']], tuple(route['pose']))
                self.assertEqual(route['return_states'][0], route['pose'])
                self.assertEqual(route['return_states'][-1], list(anchor))
                self.assertEqual(route['states'], route['outbound_states']+route['return_states'][1:])
                self.assertLessEqual(route['cost'], budget)
                for state in states:
                    self.assertTrue(safe[state[:2]])
                for first, action, second in zip(states, route['actions'], states[1:]):
                    self.assertEqual(transition(first, action), second)
                expected_out = bfs_cost(safe, current, tuple(route['pose']))
                expected_return = bfs_cost(safe, tuple(route['pose']), anchor)
                self.assertEqual(route['outbound_cost'], expected_out)
                self.assertEqual(route['return_cost'], expected_return)

    def test_same_cell_new_heading_is_paid_and_anchor_heading_is_restored(self):
        mapper = MeasuredMapperFixture()
        current, anchor = (5, 7, 0), (5, 7, 3)
        routes, audit = generate_options(mapper, current[:2], current[2], 1, anchor)
        self.assertEqual(len(routes), 1)
        self.assertEqual(routes[0]['pose'], list(anchor))
        self.assertEqual(routes[0]['outbound_actions'], ['left'])
        self.assertEqual(routes[0]['return_actions'], [])
        self.assert_options_valid(mapper, current, anchor, 1, routes)
        self.assertEqual(audit['minimum_current_return_cost'], 1)
        no_routes, audit = generate_options(mapper, current[:2], current[2], 0, anchor)
        self.assertEqual(no_routes, [])
        self.assertEqual(audit['status'], 'current_return_exceeds_remaining_budget')

    def test_zero_action_target_is_never_inserted_to_fill_a_pool(self):
        mapper = MeasuredMapperFixture(); current = (5, 7, 0)
        for budget, reason in ((0, 'no_paid_action_budget'), (1, 'no_paid_option_with_reserved_return')):
            routes, audit = generate_options(mapper, current[:2], current[2], budget, current)
            self.assertEqual(routes, [])
            self.assertEqual(audit['status'], reason)
        routes, _ = generate_options(mapper, current[:2], current[2], 2, current)
        self.assertTrue(routes)
        self.assert_options_valid(mapper, current, current, 2, routes)
        self.assertTrue(all(r['outbound_cost'] == r['return_cost'] == 1 for r in routes))

    def test_fixed_anchor_is_not_replaced_by_current_position(self):
        mapper = MeasuredMapperFixture(); current, anchor = (5, 9, 1), (5, 3, 2)
        minimum = bfs_cost(known_safe_direct(mapper), current, anchor)
        routes, audit = generate_options(mapper, current[:2], current[2], minimum-1, anchor)
        self.assertEqual(routes, [])
        self.assertEqual(audit['status'], 'current_return_exceeds_remaining_budget')
        routes, _ = generate_options(mapper, current[:2], current[2], minimum+4, anchor)
        self.assertTrue(routes)
        self.assert_options_valid(mapper, current, anchor, minimum+4, routes)
        self.assertTrue(all(r['states'][-1] != list(current) for r in routes))

    def test_latest_unknown_footprint_and_disconnected_anchor_refuse_options(self):
        mapper = MeasuredMapperFixture(); current, anchor = (5, 3, 0), (5, 11, 0)
        mapper.belief[4, 3] = -1
        routes, audit = generate_options(mapper, current[:2], current[2], 60, anchor)
        self.assertEqual(routes, []); self.assertEqual(audit['status'], 'current_footprint_not_known_safe')
        mapper.belief[4, 3] = 0; mapper.belief[:, 7] = 1
        routes, audit = generate_options(mapper, current[:2], current[2], 60, anchor)
        self.assertEqual(routes, []); self.assertEqual(audit['status'], 'anchor_disconnected_in_observed_map')

    def test_no_object_fallback_sees_unknown_but_never_routes_through_it(self):
        mapper = MeasuredMapperFixture(); current = (5, 4, 0)
        mapper.belief[:, 10:] = -1
        mapper.camera_seen[:, 10:] = False
        routes, audit = generate_options(mapper, current[:2], current[2], 12, current)
        self.assertEqual(audit['observed_asset_count'], 0)
        self.assertTrue(routes)
        self.assertTrue(audit['common_role_audit'][0]['positive_unknown_proxy'])
        self.assertTrue(all(route['asset_index'] is None for route in routes))
        self.assert_options_valid(mapper, current, current, 12, routes)

    def test_zero_one_three_observed_objects_and_budget_above_48(self):
        for count in (0, 1, 3):
            with self.subTest(count=count):
                mapper, current = objects_fixture(count)
                routes, audit = generate_options(mapper, current[:2], current[2], 100, current)
                self.assertEqual(audit['observed_asset_count'], count)
                self.assertTrue(routes)
                self.assert_options_valid(mapper, current, current, 100, routes)
                if count:
                    self.assertEqual({r['asset_index'] for r in routes if r['asset_index'] is not None}, set(range(count)))
                    self.assertTrue(any(r['cost'] > 48 for r in routes))
                else:
                    self.assertEqual([r['group'] for r in routes], ['coverage_anchor'])

    def test_swapped_and_missing_labels_do_not_change_any_route_or_geometry_audit(self):
        mapper, current = objects_fixture(3)
        original, original_audit = generate_options(mapper, current[:2], current[2], 100, current)
        labels = mapper._q['label'].copy()
        for changed in (np.where(labels == 2, 3, 2), np.zeros_like(labels)):
            mapper._q['label'] = changed
            routes, audit = generate_options(mapper, current[:2], current[2], 100, current)
            self.assertEqual(routes, original)
            self.assertEqual(audit, original_audit)

    def test_candidate_cap_records_omission_and_never_adds_fake_object_views(self):
        mapper, current = objects_fixture(3)
        routes, audit = generate_options(mapper, current[:2], current[2], 100, current, max_candidates=2)
        self.assertLessEqual(len(routes), 2)
        self.assertTrue(audit['omitted_by_cap'])
        self.assert_options_valid(mapper, current, current, 100, routes)
        small, small_audit = generate_options(mapper, current[:2], current[2], 2, current)
        self.assertTrue(small)
        self.assertTrue(small_audit['missing_roles'])
        self.assertTrue(all(r['asset_index'] is None for r in small))
        self.assert_options_valid(mapper, current, current, 2, small)

    def test_input_immutability_and_invalid_budget(self):
        mapper, current = objects_fixture(1)
        previous = deepcopy((mapper.belief, mapper.camera_seen, mapper._q))
        generate_options(mapper, current[:2], current[2], 80, current)
        for before, after in zip(previous[:2], (mapper.belief, mapper.camera_seen)):
            np.testing.assert_array_equal(before, after)
        for key in mapper._q:
            np.testing.assert_array_equal(previous[2][key], mapper._q[key])
        for budget in (-1, 1.5, True):
            with self.assertRaises(ValueError):
                generate_options(mapper, current[:2], current[2], budget, current)


if __name__ == '__main__':
    unittest.main()
