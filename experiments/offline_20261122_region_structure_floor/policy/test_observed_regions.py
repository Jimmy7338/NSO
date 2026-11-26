"""Synthetic observed-map contracts for the regional selector."""

import json
import unittest

import numpy as np

from observed_regions import RegionalPlanner
from utils.grid_geometry import inflated_obstacles


def two_rooms():
    observed = np.zeros((29, 31), dtype=np.int8)
    observed[[0,-1],:] = observed[:,[0,-1]] = 1
    observed[:,15] = 1
    observed[11:18,15] = 0
    observed[2:10,23:30] = -1
    return observed


class RegionalContract(unittest.TestCase):
    def test_entrance_proposes_observed_safe_views_and_ranks_certified_route(self):
        grid = two_rooms()
        policy = RegionalPlanner()
        proposals = policy.proposals(grid,(14,10,1),0,.2)
        safe = ~inflated_obstacles(grid != 0,1)
        self.assertTrue(proposals)
        self.assertTrue(all(safe[tuple(row['pose'][:2])] for row in proposals))
        self.assertTrue(all(row['pose'][2] in (1,3) for row in proposals if row['entrance_id']))
        offers = [dict(pose=row['pose'],outbound_cost=19,return_cost=22) for row in proposals]
        selected, audit = policy.rank(grid,(14,10,1),0,.2,[],offers)
        self.assertTrue(any(selected is row for row in offers))
        self.assertTrue(audit['chosen']['unvisited_region'])
        self.assertEqual(audit['chosen']['origin'],'new_regional_pose')
        self.assertEqual(audit['ranking'][0]['pose'],selected['pose'])
        json.dumps(audit,allow_nan=False)
        selected, audit = policy.rank(grid,(14,10,1),0,.2,[],[])
        self.assertIsNone(selected)
        self.assertEqual(audit['reason'],'no_qualified_feasible_regional_view')

    def test_visit_history_crossing_and_blockage(self):
        grid = two_rooms()
        policy = RegionalPlanner()
        policy.proposals(grid,(14,10,1),0,.2)
        _, first = policy.rank(grid,(14,10,1),0,.2,[],[])
        for paid, col in enumerate(range(11,22),1):
            policy.observe(grid,(14,col,1),paid)
        _, crossed = policy.rank(grid,(14,21,1),11,.2,[],[])
        self.assertEqual(crossed['entrances'][0]['crossed_paid_steps'],[6])
        self.assertEqual(crossed['entrances'][0]['status'],'traversed')
        self.assertEqual(crossed['visible_unvisited_region_ids'],[])
        grid[11:18,15] = 1
        policy.proposals(grid,(14,21,1),11,.2)
        _, blocked = policy.rank(grid,(14,21,1),11,.2,[],[])
        self.assertEqual(blocked['entrances'][0]['entrance_id'],first['entrances'][0]['entrance_id'])
        self.assertEqual(blocked['entrances'][0]['status'],'blocked')
        self.assertEqual(blocked['reason'],'no_credible_observed_entrance')

    def test_unknown_boundary_is_not_door_evidence(self):
        grid = two_rooms()
        grid[11:18,15] = -1
        policy = RegionalPlanner()
        self.assertEqual(policy.proposals(grid,(14,10,1),0,.2),[])
        self.assertIsNone(policy.rank(grid,(14,10,1),0,.2,[],[])[0])


if __name__=='__main__':
    unittest.main()
