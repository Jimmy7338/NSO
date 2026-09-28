"""Synthetic saved-score checks; no scenes, controller, mapper or World."""
from copy import deepcopy
import unittest

from scripts import analyze_article_direct_oracle_20260928 as audit


def fixture(discovery_b=0.):
    targets=[{'node':'a','heading':0},{'node':'b','heading':0}]
    vectors=[[4.,0.,0.,0.],[0.,4.,4.,4.]]
    probability=[.25]*4
    candidates=[dict(candidate={},view_id=t['node']+':0',structure_new_surface_area_m2=v,
        expected_new_surface_area_m2=sum(v)/4,fallback=False,repeated_view_excluded=False)
        for t,v in zip(targets,vectors)]
    direct=[dict(target=t,total_cost=1,expected_gain=sum(v)/4+(discovery_b if t['node']=='b' else 0),
        score=sum(v)/4+(discovery_b if t['node']=='b' else 0)) for t,v in zip(targets,vectors)]
    selection=dict(direct_options=direct,remaining=10,
        selected=dict(kind='direct',target=targets[1]),
        instance_candidate_allocations=[dict(instance_id='i',candidates=targets)],
        forecasts=[dict(instance_id='i',structure_names=list(audit.NAMES),
            structure_probabilities=probability,candidates=candidates)])
    return dict(decision=dict(global_selection=selection),controller_evidence=dict(
        structure_belief=dict(instances=[dict(instance_id='i',structure_probabilities=probability)])))


class DirectOracleTests(unittest.TestCase):
    def test_single_structure_substitution_exposes_switch_without_changing_baseline(self):
        step=fixture();before=deepcopy(step)
        result=audit.analyze_replan(step,dict(inspection_weight=1,discovery_weight=1),set())
        self.assertEqual(step,before)
        self.assertEqual(result['baseline_direct_target'],'b:0')
        self.assertTrue(result['any_single_instance_pure_h_flip'])
        self.assertEqual(result['instances'][0]['changed_hypotheses'],[0])
        self.assertEqual(result['hypotheses'][0]['counterfactual_target'],'a:0')
        self.assertEqual(result['hypotheses'][0]['baseline_target_rank_after'],2)

    def test_positive_structure_variation_can_be_rank_invariant(self):
        result=audit.analyze_replan(fixture(10.),dict(inspection_weight=1,discovery_weight=1),set())
        self.assertTrue(result['any_positive_area'])
        self.assertTrue(result['any_structure_dependent_area'])
        self.assertFalse(result['any_single_instance_pure_h_flip'])
        self.assertEqual(result['instances'][0]['distinct_instance_area_argmax'],2)

    def test_allocated_missing_forecast_is_never_assumed_zero(self):
        step=fixture();step['decision']['global_selection']['forecasts'][0]['candidates'].pop()
        with self.assertRaisesRegex(ValueError,'inventory'):
            audit.analyze_replan(step,dict(inspection_weight=1,discovery_weight=1),set())

    def test_zero_only_for_explicit_unallocated_targets(self):
        step=fixture();s=step['decision']['global_selection']
        s['instance_candidate_allocations'][0]['candidates'].pop()
        s['forecasts'][0]['candidates'].pop()
        # b is now a recorded discovery-only candidate, with the same saved total.
        result=audit.analyze_replan(step,dict(inspection_weight=1,discovery_weight=1),set())
        self.assertEqual(result['candidate_snapshot'][1]['structure_areas']['i'],[0.]*4)
        self.assertEqual(result['candidate_snapshot'][1]['inferred_discovery_remainder'],3.)

    def test_reconstruction_or_visited_discovery_mismatch_rejected(self):
        step=fixture();step['decision']['global_selection']['direct_options'][0]['score']=17.
        with self.assertRaisesRegex(ValueError,'gain/cost'):
            audit.analyze_replan(step,dict(inspection_weight=1,discovery_weight=1),set())
        with self.assertRaisesRegex(ValueError,'visited XY'):
            audit.analyze_replan(fixture(10.),dict(inspection_weight=1,discovery_weight=1),{'b'})

    def test_exact_unique_target_tie_rule_and_no_positive_gain(self):
        self.assertEqual(audit.best({('a',0):1.,('b',0):1.+.5e-12})['target'],('a',0))
        self.assertEqual(audit.best({('a',0):0.,('b',0):0.})['target'],None)


if __name__=='__main__':unittest.main()
