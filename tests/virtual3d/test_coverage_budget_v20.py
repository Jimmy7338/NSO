from copy import deepcopy
import json
import unittest

import numpy as np

from nso.coverage_budget_v20 import (ObservedCoverageLedgerV20, assess_coverage_budget,
    coverage_sets, scan_hit_mask, union_unknown_mask)
from utils.rgbd_contract import PlanarScan


class CoverageBudgetV20Tests(unittest.TestCase):
    def test_sensor_and_path_overlap_count_once_and_known_camera_cells_count_zero(self):
        belief=np.full((4,5),-1,np.int8);belief[0,2]=0;belief[0,3]=1
        radar=np.zeros_like(belief,bool);camera=radar.copy();roi=np.ones_like(radar)
        radar[1,1]=radar[1,2]=radar[0,2]=True
        camera[1,1]=camera[2,2]=camera[0,2]=camera[0,3]=True
        roi[2,2]=False
        actual=union_unknown_mask(belief,radar,camera,radar,task_mask=roi)
        self.assertEqual(int(actual.sum()),2)
        self.assertFalse(actual[0,2]);self.assertFalse(actual[0,3]);self.assertFalse(actual[2,2])
        np.testing.assert_array_equal(actual,union_unknown_mask(belief,camera,radar,task_mask=roi))

    def test_only_distinct_actual_scan_evidence_shrinks_possible_region(self):
        belief=np.full((15,15),-1,np.int8);belief[5:10,3:8]=0;belief[:,10]=1
        hits=np.zeros_like(belief,bool);hits[:,10]=True
        ledger=ObservedCoverageLedgerV20(belief.shape,(7,5),1.)
        initial=ledger.start(belief,radar_hits=hits)
        ledger.observe(belief,1,True)
        self.assertEqual(ledger.snapshot()['P_count'],initial['P_count'])
        with self.assertRaisesRegex(ValueError,'exactly once'):
            ledger.observe(belief,1,True,radar_hits=hits)
        self.assertEqual(ledger.snapshot()['confirmed_hit_cells'],0)
        ledger.observe(belief,2,True,radar_hits=hits)
        final=ledger.snapshot()
        self.assertEqual(final['confirmed_hit_cells'],15)
        self.assertLess(final['P_count'],initial['P_count'])
        self.assertEqual(final['S_count'],initial['S_count'])
        sets=coverage_sets(belief,(7,5),1.,confirmed_occupied=hits)
        self.assertFalse(np.any(sets['safe'] & ~sets['possible']))
        self.assertFalse(final['true_coverage_guaranteed'])

    def test_zero_gain_steps_and_short_final_prefix_stay_in_rate_denominator(self):
        belief=np.full((10,10),-1,np.int8);belief[4:7,4:7]=0
        ledger=ObservedCoverageLedgerV20(belief.shape,(5,5),0.)
        self.assertIsNone(ledger.start(belief)['conservative_rate'])
        predicted=np.zeros_like(belief,bool);predicted[3,5]=True
        observed=belief.copy();observed[3,5]=0
        event=ledger.observe(observed,1,True,predicted_mask=predicted)
        self.assertEqual(event['actual_new_known_cells'],1)
        self.assertEqual(event['realized_predicted_cells'],1)
        for action in range(2,6):ledger.observe(observed,action,True,predicted_mask=predicted)
        snapshot=ledger.snapshot()
        self.assertEqual(len(snapshot['coverage_prefixes']),1)
        self.assertEqual(snapshot['coverage_prefixes'][0]['paid_actions'],5)
        self.assertAlmostEqual(snapshot['conservative_rate'],.1)
        self.assertEqual(snapshot['union_yield'],2/3)
        observed[2,5]=0;ledger.observe(observed,6,True)
        pending=ledger.snapshot()['pending_prefix'];self.assertEqual(pending['paid_actions'],1)
        ledger.flush_prefix()
        self.assertIsNone(ledger.snapshot()['pending_prefix'])
        self.assertEqual(sum(x['paid_actions'] for x in ledger.snapshot()['coverage_prefixes']),6)
        self.assertAlmostEqual(ledger.snapshot()['conservative_rate'],1/6)
        # Losing those two observed safe cells must offset their earlier gains.
        ledger.observe(belief,7,True)
        self.assertEqual(ledger.snapshot()['conservative_rate'],0.)
        self.assertEqual(ledger.events[-1]['safe_lost_cells'],2)

    def test_quality_with_slack_is_allowed_but_long_observation_spends_coverage_reserve(self):
        allowed=assess_coverage_budget(120,90,20,20,20,4.)
        self.assertTrue(allowed['allowed']);self.assertEqual(allowed['coverage_reserve_actions'],25)
        self.assertEqual(allowed['total_required_actions'],70)
        refused=assess_coverage_budget(120,90,40,30,0,4.)
        self.assertFalse(refused['allowed']);self.assertEqual(refused['reason'],'insufficient_predicted_coverage_reserve')
        self.assertEqual(refused['total_required_actions'],105)
        for rate in (None,0.):
            unavailable=assess_coverage_budget(120,90,20,20,20,rate)
            self.assertEqual(unavailable['reason'],'coverage_rate_unavailable')
            self.assertIsNone(unavailable['coverage_reserve_actions'])
            json.dumps(unavailable,allow_nan=False)
        achieved=assess_coverage_budget(0,7,1,1,0,None)
        self.assertTrue(achieved['allowed']);self.assertEqual(achieved['coverage_reserve_actions'],0)

    def test_class_annotations_cannot_change_budget_and_predictions_do_not_mutate_map(self):
        belief=np.full((10,10),-1,np.int8);belief[4:7,4:7]=0
        ledger=ObservedCoverageLedgerV20(belief.shape,(5,5),0.)
        ledger.start(belief)
        seen=belief.copy();seen[3,5]=0;ledger.observe(seen,1,True)
        predicted=np.zeros_like(belief,bool);predicted[2,5]=True
        before=deepcopy(ledger.snapshot());before_belief=ledger.belief.copy()
        route=dict(outbound_cost=2,return_cost=2,class_vote=1.)
        positive=ledger.assess_route(route,predicted,500)
        route['class_vote']=-1.
        self.assertEqual(positive,ledger.assess_route(route,predicted,500))
        self.assertEqual(before,ledger.snapshot())
        np.testing.assert_array_equal(before_belief,ledger.belief)
        self.assertEqual(positive['prediction']['predicted_union_cells'],1)
        self.assertFalse(positive['movement_authorized'])

    def test_actual_planar_scan_and_dictionary_share_endpoint_contract(self):
        pose=np.eye(4);pose[:3,3]=[.5,.5,.3]
        scan=PlanarScan(0.,np.asarray([.4,.4,5.,np.inf]),0.,0.,5.,pose)
        actual=scan_hit_mask(scan,(10,10),.2)
        expected=np.zeros((10,10),bool);expected[7,4]=True
        np.testing.assert_array_equal(actual,expected)
        np.testing.assert_array_equal(actual,scan_hit_mask(scan.__dict__,(10,10),.2))
        self.assertEqual(int(actual.sum()),1)


if __name__=='__main__':unittest.main()
