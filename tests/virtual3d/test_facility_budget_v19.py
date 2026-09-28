"""Non-tautological checks for relaxed visit bounds and coverage-budget contracts."""
from collections import deque
from types import SimpleNamespace

import numpy as np
import unittest

from utils.facility_budget_v19 import (advance_pose, conservative_asset_regions,
    exact_region_walk_lower_bound, paired_budget, region_tour_lower_bound, scan_known_mask, shortest_actions)
from utils.grid_geometry import DIRECTIONS
from utils.rgbd_contract import PlanarScan


def exact_closed_walk(safe, start, regions):
    """Brute-force physical (cell, visited-bitset) walk, independent of region DP."""
    def bits(cell):
        return sum(1 << i for i, region in enumerate(regions) if region[cell])
    initial = (*start, bits(start)); queue = deque([(initial, 0)]); seen = {initial}
    while queue:
        (r, c, mask), distance = queue.popleft()
        if (r, c) == start and mask == (1 << len(regions)) - 1:
            return distance
        for dr, dc in DIRECTIONS:
            nr, nc = r + dr, c + dc
            if 0 <= nr < safe.shape[0] and 0 <= nc < safe.shape[1] and safe[nr, nc]:
                nxt = nr, nc, mask | bits((nr, nc))
                if nxt not in seen:
                    seen.add(nxt); queue.append((nxt, distance + 1))
    raise ValueError('no physical tour')


def test_region_relaxation_is_lower_than_physical_tour_with_distinct_entry_exit():
    safe = np.ones((3, 8), bool)
    wide = np.zeros_like(safe); wide[1, 1:7] = True
    end = np.zeros_like(safe); end[1, 7] = True
    regions = [wide, end]
    bound = region_tour_lower_bound(safe, (1, 0), regions)['lower_bound_translation_actions']
    actual = exact_closed_walk(safe, (1, 0), regions)
    assert bound == 9 < actual == 14
    assert exact_region_walk_lower_bound(safe, (1, 0), regions)['lower_bound_translation_actions'] == actual
    singleton = np.zeros_like(safe); singleton[1, 3] = True
    assert region_tour_lower_bound(safe, (1, 0), [singleton, end])['lower_bound_translation_actions'] == actual


def test_region_bound_obstacles_and_overlapping_regions_do_not_overestimate():
    safe = np.ones((5, 7), bool); safe[1:4, 3] = False
    a = np.zeros_like(safe); a[0:2, 1:3] = True
    b = np.zeros_like(safe); b[0:2, 2:5] = True
    c = np.zeros_like(safe); c[3:5, 5] = True
    regions = [a, b, c]
    result = region_tour_lower_bound(safe, (2, 0), regions)
    assert result['lower_bound_translation_actions'] <= exact_closed_walk(safe, (2, 0), regions)
    assert exact_region_walk_lower_bound(safe, (2, 0), regions)['lower_bound_translation_actions'] == exact_closed_walk(safe, (2, 0), regions)
    with unittest.TestCase().assertRaisesRegex(ValueError, 'empty safe'):
        region_tour_lower_bound(safe, (2, 0), [np.zeros_like(safe)])


def test_paired_budget_uses_hardest_coverage_and_easiest_all_visit_bound():
    result = paired_budget([100, 120], [200, 190])
    assert result['budget'] == 144
    assert result['paired_coverage_witness_actions'] == 120
    assert result['paired_all_asset_lower_bound'] == 190
    assert paired_budget([110, 120], [119, 150])['status'] == 'failed'
    assert paired_budget([110, 120], [120, 150])['budget'] is None


def test_containing_box_region_and_scan_support_ignore_quality_scores():
    world = SimpleNamespace(shape=(5, 5), reachable=np.ones((5, 5), bool),
        config=SimpleNamespace(resolution_m=1.),
        objects=[dict(id=0, evaluation_bounds=[[2., 2., 0.], [3., 3., 2.]])])
    regions, audit = conservative_asset_regions(world, 1.)
    assert regions[0][2, 1] and regions[0][2, 3]
    assert not regions[0][2, 0]
    assert audit[0]['safe_region_cells'] == 9
    pose = np.eye(4); pose[:3, 3] = [.5, .5, .25]
    scan = PlanarScan(0., np.array([2.]), 0., 1., 3., pose)
    known = scan_known_mask(scan, (5, 5), 1.)
    expected = np.zeros((5, 5), bool); expected[4, :3] = True
    np.testing.assert_array_equal(known, expected)
    safe = np.ones((4, 4), bool); safe[1, 1] = False
    state = (2, 0, 0)
    for action in shortest_actions(safe, state, (0, 2), 3):
        state = advance_pose(state, action)
        assert safe[state[:2]]
    assert state == (0, 2, 3)


class FacilityBudgetTests(unittest.TestCase):
    test_region_relaxation = staticmethod(test_region_relaxation_is_lower_than_physical_tour_with_distinct_entry_exit)
    test_obstacles_and_overlap = staticmethod(test_region_bound_obstacles_and_overlapping_regions_do_not_overestimate)
    test_paired_budget = staticmethod(test_paired_budget_uses_hardest_coverage_and_easiest_all_visit_bound)
    test_bounds_and_scan = staticmethod(test_containing_box_region_and_scan_support_ignore_quality_scores)


if __name__ == '__main__':
    unittest.main()
