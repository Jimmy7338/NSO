"""Small observed-grid behavior tests, with no simulator or evaluator truth."""
from copy import deepcopy
from types import SimpleNamespace
import unittest
import numpy as np
from nso.facility_candidates_v20 import ObservedRouteSpaceV20
from nso.execution_guard_v8 import ObservedExecutionGuard


def mapper_for(belief, radius=0., max_depth=1.8):
    belief = np.asarray(belief, dtype=np.int8).copy()
    return SimpleNamespace(belief=belief, shape=belief.shape,
        config=SimpleNamespace(resolution_m=.2, robot_radius_m=radius, max_depth_m=max_depth),
        semantic=np.zeros(belief.shape, np.uint8))


class FacilityCandidatesV20Tests(unittest.TestCase):
    def test_directed_cost_reserves_home_heading_and_unknown_blocks_motion(self):
        mapper = mapper_for(np.zeros((15, 15), np.int8))
        anchor = (7, 7, 0)
        space = ObservedRouteSpaceV20(mapper, anchor[:2], anchor[2], anchor, 6)
        route = space.route((7, 8, 1))
        self.assertEqual((route['outbound_cost'], route['return_cost'], route['cost']), (2, 4, 6))
        self.assertEqual(route['return_states'][-1], list(anchor))
        self.assertEqual(len(route['actions']), route['cost'])
        tight = ObservedRouteSpaceV20(mapper, anchor[:2], anchor[2], anchor, 5)
        self.assertIsNone(tight.route((7, 8, 1)))
        mapper.belief[:, 8] = -1
        split = ObservedRouteSpaceV20(mapper, (7, 4), 1, (7, 4, 1), 100)
        self.assertTrue(split.available)
        self.assertIsNone(split.route((7, 8, 1)))
        self.assertIsNone(split.route((7, 12, 1)))

    def test_refresh_keeps_intention_metadata_and_replans_around_new_obstacle(self):
        mapper = mapper_for(np.zeros((15, 17), np.int8))
        anchor, goal = (7, 4, 0), (7, 10, 1)
        original = ObservedRouteSpaceV20(mapper, anchor[:2], anchor[2], anchor, 40).route(
            goal, candidate_id=19, group='coverage_v20_7')
        original.update(option_id='retained-intention', selection_call_id=31,
                        selected_map_version=1, intention={'target_id': 'observed-frontier-2'})
        before = deepcopy(original)
        mapper.belief[7, 8] = 1
        latest = ObservedRouteSpaceV20(mapper, (7, 6), 1, anchor, 36)
        refreshed = latest.refresh(original)
        self.assertIsNotNone(refreshed)
        for key in ('candidate_id', 'group', 'option_id', 'selection_call_id', 'intention', 'pose'):
            self.assertEqual(refreshed[key], before[key])
        self.assertEqual(original, before)
        self.assertEqual(refreshed['outbound_states'][0], [7, 6, 1])
        self.assertEqual(refreshed['outbound_states'][-1], list(goal))
        self.assertEqual(refreshed['return_states'][-1], list(anchor))
        self.assertNotIn([7, 8], [state[:2] for state in refreshed['states']])
        self.assertEqual(refreshed['outbound_cost']+refreshed['return_cost'], refreshed['cost'])
        self.assertLessEqual(refreshed['cost'], 36)

    def test_unknown_prediction_is_unique_and_bound_to_the_route_map_snapshot(self):
        belief = np.zeros((21, 25), np.int8); belief[6:11, 14:19] = -1
        mapper = mapper_for(belief, max_depth=2.)
        space = ObservedRouteSpaceV20(mapper, (10, 10), 0, (10, 10, 0), 80)
        repeated = dict(outbound_states=[[10, 10, 0], [10, 10, 1], [10, 10, 2], [10, 10, 3]])
        mask = space.route_mask(repeated)
        np.testing.assert_array_equal(mask, space.visibility((10, 10)))
        self.assertGreater(int(mask.sum()), 0)
        self.assertFalse(mask[belief != -1].any())
        with self.assertRaises(ValueError):
            space.visibility((10, 10))[0, 0] = True
        mapper.belief[:] = 0
        np.testing.assert_array_equal(space.route_mask(repeated), mask)
        fresh = ObservedRouteSpaceV20(mapper, (10, 10), 0, (10, 10, 0), 80)
        self.assertFalse(fresh.route_mask(repeated).any())

    def test_frontier_groups_precede_spatial_fill_and_class_values_have_no_effect(self):
        belief = np.zeros((29, 39), np.int8)
        belief[[0, -1], :] = 1; belief[:, [0, -1]] = 1
        belief[7:13, 3:7] = -1; belief[16:22, 31:35] = -1
        mapper = mapper_for(belief, radius=.2, max_depth=1.8)
        start = (14, 19, 0)
        space = ObservedRouteSpaceV20(mapper, start[:2], start[2], start, 160)
        routes, audit = space.coverage_candidates()
        self.assertGreaterEqual(len(audit['frontier_groups']), 2)
        self.assertGreaterEqual(len(audit['represented_frontier_groups']), 2)
        self.assertEqual(audit['selected'][0]['selection_reason'], 'rate_anchor')
        reasons = [row['selection_reason'] for row in audit['selected']]
        if 'total_anchor' in reasons:
            self.assertEqual(reasons[1], 'total_anchor')
        if 'new_frontier_group' in reasons and 'spatial_fill' in reasons:
            self.assertLess(max(i for i, value in enumerate(reasons) if value == 'new_frontier_group'),
                            min(i for i, value in enumerate(reasons) if value == 'spatial_fill'))
        self.assertFalse(audit['first_translation_used_for_selection'])
        self.assertFalse(audit['frontier_groups_are_room_labels'])
        self.assertLessEqual(len(routes), 8)
        self.assertEqual(len({tuple(route['pose'][:2]) for route in routes}), len(routes))
        self.assertTrue(audit['same_cell_rotation_excluded'])
        self.assertNotIn(start[:2], [tuple(route['pose'][:2]) for route in routes])
        for index, selected in enumerate(audit['selected']):
            if selected['selection_reason'] == 'total_anchor':
                continue
            for earlier in audit['selected'][:index]:
                self.assertGreaterEqual(np.linalg.norm(np.subtract(selected['pose'][:2], earlier['pose'][:2]))*.2, 1.6)
        guard = ObservedExecutionGuard(.2, .2)
        for route in routes:
            for pose in route['states']:
                self.assertTrue(guard.safe_grid(mapper.belief)[tuple(pose[:2])])
        mapper.semantic[:] = 3
        changed, changed_audit = ObservedRouteSpaceV20(mapper, start[:2], start[2], start, 160).coverage_candidates()
        self.assertEqual(routes, changed)
        self.assertEqual(audit, changed_audit)

    def test_bounds_and_small_slot_limit_are_enforced(self):
        belief = np.zeros((13, 13), np.int8); belief[1:4, 8:11] = -1
        mapper = mapper_for(belief)
        space = ObservedRouteSpaceV20(mapper, (6, 6), 0, (6, 6, 0), 60)
        self.assertIsNone(space.route((13, 2, 0)))
        with self.assertRaises(ValueError):
            space.route((-1, 2, 0))
        with self.assertRaises(ValueError):
            space.route((1, 2, 4))
        with self.assertRaises(ValueError):
            ObservedRouteSpaceV20(mapper, (6, 6), 0, (6, 6, 0), 2.5)
        routes, audit = space.coverage_candidates(slots=1)
        self.assertEqual(len(routes), 1)
        self.assertEqual(audit['selected'][0]['selection_reason'], 'rate_anchor')
        outside = ObservedRouteSpaceV20(mapper, (20, 20), 0, (6, 6, 0), 10)
        self.assertFalse(outside.available)

    def test_rotation_only_space_has_no_pure_coverage_candidate(self):
        belief = np.full((7, 7), -1, np.int8); belief[3, 3] = 0
        mapper = mapper_for(belief)
        space = ObservedRouteSpaceV20(mapper, (3, 3), 0, (3, 3, 0), 8)
        self.assertTrue(space.available)
        self.assertGreater(int(space.visibility((3, 3)).sum()), 0)
        rotation = space.route((3, 3, 1), group='inspection_rotation')
        self.assertEqual(rotation['cost'], 2)
        routes, audit = space.coverage_candidates()
        self.assertEqual(routes, [])
        self.assertEqual(audit['status'], 'no_unknown_gain')
        self.assertTrue(audit['same_cell_rotation_excluded'])


if __name__ == '__main__':
    unittest.main()
