import copy
import importlib.util
from pathlib import Path
import unittest
import numpy as np

from evaluate_temperature import (TEMPERATURES, aggregate, class_metrics,
                                  fallback, numerical_components, select)


def implementation():
    path=Path(__file__).resolve().parents[1]/'model.py'
    spec=importlib.util.spec_from_file_location('B9_test_reference',path)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module);return module


class EvaluatorTests(unittest.TestCase):
    def test_independent_formula_matches_frozen_method_and_T1(self):
        rng=np.random.default_rng(48392);m=implementation()
        for n in (1,7,19):
            q=rng.random(n);q/=q.sum();e=rng.uniform(0,.02,(n,2))
            conditional,L=numerical_components(e,q)
            for t in TEMPERATURES:
                for target in (0,1):
                    for semantic in (False,True):
                        metric=class_metrics(L,target,t,semantic)
                        prior=([.1,.9]if target else[.9,.1])if semantic else[.5,.5]
                        actual=m.posterior(e,q,prior,t)
                        np.testing.assert_allclose(conditional,actual['conditional_geometry_mass'],rtol=0,atol=1e-14)
                        np.testing.assert_allclose(conditional*np.asarray(metric['shape_probability_mass'])[None,:],actual['joint_probability_mass'],rtol=0,atol=1e-13)

    def test_extreme_wrong_NLL_is_not_clipped(self):
        self.assertEqual(class_metrics(1000.,0,1.)['nll'],1000.)
        self.assertEqual(class_metrics(-1000.,1,1.)['nll'],1000.)
        self.assertTrue(class_metrics(1000.,0,1.)['high_confidence_wrong'])
        self.assertFalse(class_metrics(1000.,1,1.)['high_confidence_wrong'])

    def test_neutral_exact_tie_half_accuracy_and_binary_Brier(self):
        row=class_metrics(0.,1,4.)
        self.assertEqual(row['accuracy'],.5);self.assertEqual(row['binary_brier'],.25)
        self.assertTrue(row['exact_tie']);self.assertIsNone(row['predicted_class'])
        self.assertAlmostEqual(row['nll'],np.log(2))

    def test_controlled_correct_prior_is_applied_after_temperature(self):
        for target in (0,1):
            neutral=class_metrics(6.,target,8.)
            semantic=class_metrics(6.,target,8.,True)
            self.assertAlmostEqual(semantic['logit']-neutral['logit'],(2*target-1)*np.log(9))
            self.assertTrue(semantic['correct_category_prior_supplied'])

    def test_fallback_class_and_depth_are_temperature_independent(self):
        track,d,q,conditional,L=fallback('multiple tracks')
        self.assertEqual(L,0.);self.assertEqual(track['depth_axis_world'],[0.,1.,0.])
        for t in TEMPERATURES:
            for semantic in (False,True):
                c=class_metrics(L,1,t,semantic)
                np.testing.assert_allclose(conditional@np.asarray(c['shape_probability_mass']),q,rtol=0,atol=1e-15)
                self.assertAlmostEqual(c['shape_probability_mass'][1],.9 if semantic else .5)

    def test_selector_uses_only_complete_calibration_objects(self):
        rows=[dict(split='calibration',object_id=str(o),condition=c,log_evidence_difference=4.,target=o)
              for o in (0,1)for c in ('front','front_side','around','occluded')]
        self.assertEqual(select(rows)['temperature'],128.)
        self.assertEqual(select([dict(r,log_evidence_difference=0.)for r in rows])['temperature'],1.)
        self.assertEqual(select(rows)['temperature'],implementation().select_temperature(rows)['temperature'])
        for bad in (rows[:-1],rows+[rows[0]],[dict(r,split='test')for r in rows]):
            with self.assertRaises(ValueError):select(bad)

    def test_aggregation_retains_fallback_and_object_units(self):
        rows=[]
        for o,n in [('a',4),('b',4)]:
            for i in range(n):rows.append(dict(object_id=o,class_metrics=class_metrics(0.,0,1.),depth_metrics=dict(crps_m=.1 if o=='a'else.3,median_absolute_error_m=.2),evaluator_prior_fallback=o=='b'))
        a=aggregate(rows)
        self.assertEqual(a['objects'],2);self.assertEqual(a['cases'],8);self.assertEqual(a['fallback_cases'],4)
        self.assertAlmostEqual(a['marginal_depth_crps_m'],.2)


if __name__=='__main__':unittest.main(verbosity=2)
