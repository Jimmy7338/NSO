"""Finite-candidate oracle gap must not manufacture value or count duplicates."""
from copy import deepcopy
import unittest
from nso.information_value_v13 import summarize_paired_rewards


def fixture(values):
    protocol=dict(parents=["P"],arrangements=["a","b"],
        eligible_pairs=[dict(parent="P",action_id=0,candidates_per_arrangement=2)],
        numerical_positive_tolerance=1e-12,expected_candidate_outcomes=4,
        development_training_screen=dict(minimum_geometry_blind_oracle_mean=1e-12,
        minimum_mean_value_relative_to_geometry_blind_oracle=.005,minimum_positive_parent_count=1))
    rows=[dict(parent="P",checkpoint_action=0,arrangement=a,candidate_id=i,joint_gain=value)
        for a,line in zip(["a","b"],values) for i,value in enumerate(line)]
    return protocol,rows


class InformationValueTests(unittest.TestCase):
    def test_shared_optimum_has_zero_value_even_when_rewards_differ(self):
        p,r=fixture([[2.,1.],[4.,0.]])
        result=summarize_paired_rewards(p,r)
        self.assertEqual(result["mean_information_value"],0.)
        self.assertFalse(result["numerical_information_screen_passed"])

    def test_crossed_optima_have_expected_value(self):
        p,r=fixture([[3.,1.],[1.,3.]])
        result=summarize_paired_rewards(p,r)
        self.assertEqual(result["mean_information_value"],1.)
        self.assertEqual(result["relative_information_value"],.5)

    def test_duplicate_and_incomplete_data_are_rejected(self):
        p,r=fixture([[3.,1.],[1.,3.]])
        for bad in (r+[r[0]],r[:-1]):
            with self.assertRaises(ValueError): summarize_paired_rewards(p,bad)

    def test_nonfinite_and_extra_outcomes_are_rejected(self):
        p,r=fixture([[3.,1.],[1.,3.]])
        bad=deepcopy(r);bad[0]["joint_gain"]=float("nan")
        with self.assertRaises(ValueError): summarize_paired_rewards(p,bad)
        bad=r+[dict(parent="unlisted",checkpoint_action=0,arrangement="a",candidate_id=0,joint_gain=1.)]
        with self.assertRaises(ValueError): summarize_paired_rewards(p,bad)

    def test_zero_blind_return_does_not_produce_infinite_relative_gain(self):
        p,r=fixture([[1.,-1.],[-1.,1.]])
        result=summarize_paired_rewards(p,r)
        self.assertIsNone(result["relative_information_value"])
        self.assertFalse(result["numerical_information_screen_passed"])


if __name__=="__main__":unittest.main()
