import unittest
from copy import deepcopy
from nso.observed_continuation_v17 import select_continuation


class ContinuationTests(unittest.TestCase):
    def setUp(self):
        self.anchor = dict(keys=[(1, 2, 3)], coordinate_epoch='fixed_world_pose')
        self.regions = [dict(self.anchor, group=8, asset_index=2)]
        self.routes = [dict(candidate_id=7, cost=3, group='coverage_total_0', asset_index=None),
                       dict(candidate_id=11, cost=5, group='asset_2_aperture_view', asset_index=2)]
        self.rows = [dict(candidate_id=7, corrected_aperture=[0, 0, 0]),
                     dict(candidate_id=11, corrected_aperture=[0, 0, .2])]

    def choose(self, regions=None, rows=None, scores=None, peers=()):
        return select_continuation(self.anchor, self.regions if regions is None else regions,
            peers, self.routes, [10, 4] if scores is None else scores, self.rows if rows is None else rows)

    def test_committed_stage_tracks_region_despite_index_change(self):
        result = self.choose()
        self.assertEqual(result['candidate_id'], 11)
        self.assertEqual(result['status'], 'continue_observed_region')
        self.assertFalse(result['motion_authorized'])
        self.regions[0]['class_vote'] = -1
        self.assertEqual(self.choose(), result)

    def test_rejected_missing_and_merged_region_fall_back_without_target_credit(self):
        for regions, peers in [([], ()), ([dict(self.regions[0], asset_index=None)], ()),
                               (self.regions, [self.anchor])]:
            result = self.choose(regions=regions, peers=peers)
            self.assertEqual(result['candidate_id'], 7)
            self.assertEqual(result['status'], 'common_geometry_fallback')
            self.assertFalse(result['target_specific_credit_allowed'])

    def test_known_blocked_view_falls_back_and_no_positive_score_requests_return(self):
        rows = deepcopy(self.rows); rows[1]['corrected_aperture'][2] = 0
        self.assertEqual(self.choose(rows=rows)['candidate_id'], 7)
        self.assertEqual(self.choose(scores=[0, -1])['status'], 'request_guarded_return')

    def test_misaligned_scores_are_rejected(self):
        rows = deepcopy(self.rows); rows[1]['candidate_id'] = 99
        with self.assertRaises(ValueError): self.choose(rows=rows)


if __name__ == '__main__': unittest.main()
