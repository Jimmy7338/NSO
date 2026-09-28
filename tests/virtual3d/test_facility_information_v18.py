from copy import deepcopy
import unittest
from scripts.analyze_facility_v18 import parent_value


def fixture():
    records=[]
    for assignment,votes,rewards in [('one',[1,-1],[.6,.8,.4]),('two',[-1,1],[.6,.4,.8])]:
        for option,reward in zip(('continue_coverage','observe_asset_A','observe_asset_B'),rewards):
            records.append(dict(case=dict(assignment=assignment,option=option,observed_votes=votes),
                after={'2026':dict(eligible=True,**{'05cm':dict(joint_asset=reward)})}))
    return records


class FacilityInformationTests(unittest.TestCase):
    def test_conditioned_value_requires_observable_group_difference(self):
        records=fixture();result=parent_value(records,'2026')
        self.assertTrue(result['screen_passed'])
        self.assertAlmostEqual(result['relative_information'],1/3)
        for r in records:r['case']['observed_votes']=[0,0]
        result=parent_value(records,'2026')
        self.assertFalse(result['screen_passed'])
        self.assertEqual(result['relative_information'],0.)

    def test_missing_case_rejected_and_infeasible_options_are_not_silently_discarded(self):
        records=fixture()
        with self.assertRaises(ValueError):parent_value(records[:-1],'2026')
        for r in records:r['after']['2026']['eligible']=False
        result=parent_value(records,'2026')
        self.assertEqual(result['status'],'no_common_feasible_baseline')
        self.assertEqual(sum(len(x) for x in result['unfiltered_rewards'].values()),6)
        self.assertFalse(result['screen_passed'])


if __name__=='__main__':unittest.main()
