from copy import deepcopy
import unittest

from scripts.analyze_facility_v19 import OPTIONS, REFERENCES, parent_value, validate_metric, validate_record


def fixture():
    records=[]
    for assignment,votes,rewards in [('one',[1.,-1.],[.6,.8,.4]),('two',[-1.,1.],[.6,.4,.8])]:
        for option,reward in zip(OPTIONS,rewards):
            records.append(dict(case=dict(assignment=assignment,option=option,observed_votes=votes),
                after={'2026':dict(eligible=True,**{'05cm':dict(joint_external=reward)})}))
    return records


def metric(*,coverage=.9,failed=False):
    instances=[]
    for i in range(6):
        instances.append(dict(id=i,missing=False,**{tag:dict(precision=.8,recall=.8,f1=.8) for tag in ('02cm','05cm','10cm')}))
    return dict(mission_asset_count=6,instances=instances,coverage_2d=coverage,returned=True,failed=failed,
        collisions=0,eligible=coverage>=.8 and not failed,
        **{tag:dict(external_macro_f1=.8,asset_macro_f1=.8,joint_external=coverage*.8,joint_asset=coverage*.8)
           for tag in ('02cm','05cm','10cm')})


class FacilityInformationV19Tests(unittest.TestCase):
    def test_observed_information_requires_actual_group_difference(self):
        records=fixture();result=parent_value(records,'2026')
        self.assertTrue(result['screen_passed']);self.assertAlmostEqual(result['relative_information'],1/3)
        self.assertAlmostEqual(result['absolute_information'],.2)
        self.assertTrue(result['conditional_choice_differs_from_common'])
        for row in records:row['case']['observed_votes']=[0.,0.]
        result=parent_value(records,'2026')
        self.assertEqual(result['relative_information'],0.);self.assertFalse(result['screen_passed'])

    def test_duplicates_missing_cells_and_option_dependent_groups_are_rejected(self):
        records=fixture()
        for broken in (records[:-1],records+[deepcopy(records[0])]):
            with self.assertRaises(ValueError):parent_value(broken,'2026')
        records[1]['case']['observed_votes']=[-1.,1.]
        with self.assertRaisesRegex(ValueError,'depend on the chosen option'):parent_value(records,'2026')

    def test_failed_options_stay_visible_and_five_percent_is_strict(self):
        records=fixture()
        for row in records:row['after']['2026']['eligible']=False
        result=parent_value(records,'2026')
        self.assertEqual(result['status'],'no_common_feasible_baseline')
        self.assertEqual(sum(len(x) for x in result['unfiltered_rewards'].values()),6)
        self.assertFalse(result['screen_passed'])
        records=fixture()
        for row in records:
            option=row['case']['option'];assignment=row['case']['assignment']
            preferred=(assignment=='one' and option=='observe_asset_A') or (assignment=='two' and option=='observe_asset_B')
            row['after']['2026']['05cm']['joint_external']=.4 if option=='continue_coverage' else .42 if preferred else .36
        result=parent_value(records,'2026')
        self.assertAlmostEqual(result['relative_information'],.05)
        self.assertFalse(result['screen_passed'])

    def test_failed_eligibility_conflicts_and_curve_loss_are_rejected(self):
        invalid=metric(failed=True);invalid['eligible']=True
        with self.assertRaisesRegex(ValueError,'contradicts eligibility'):validate_metric(invalid)
        after=metric();before=metric(coverage=.4)
        row=dict(case=dict(budget=30),paid_actions=20,actions=[dict(collision=False) for _ in range(20)],
            primitive_budget_compliant=True,collisions=0,
            termination=dict(failed=False,returned_to_anchor=True,final_action_id=20),
            before={ref:deepcopy(before) for ref in REFERENCES},after={ref:deepcopy(after) for ref in REFERENCES},
            curve_action_stride=20,quality_curve=[dict(action_id=0,metrics=deepcopy(before)),dict(action_id=20,metrics=deepcopy(after))])
        validate_record(row)
        broken=deepcopy(row);broken['termination']['failed']=True
        with self.assertRaisesRegex(ValueError,'terminal failure'):validate_record(broken)
        broken=deepcopy(row);broken['quality_curve'].pop()
        with self.assertRaisesRegex(ValueError,'curve is incomplete'):validate_record(broken)


if __name__=='__main__':unittest.main()
