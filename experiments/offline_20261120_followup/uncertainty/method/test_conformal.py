import copy
import math
import unittest

import numpy as np

from conformal import (CONDITIONS, NUMERICAL_SCALE_FLOOR_M, apply_to_distribution,
                       calibrate_object_intervals, finite_sample_rank,
                       fit_training_scale)


def rows(count, prefix='cal', target=.54, lower=.5, upper=.5):
    return [dict(object_id=f'{prefix}-{i}', condition=c, target_extent_m=target,
                 median_m=(lower+upper)/2, lower_m=lower, upper_m=upper,
                 fallback=False) for i in range(count) for c in CONDITIONS]


def fit():
    return fit_training_scale(rows(12, 'train', target=.52))


class ConformalTests(unittest.TestCase):
    def test_scale_is_equal_object_median_not_frame_replication(self):
        source=rows(3,'train')
        for row in source:
            i=int(row['object_id'].split('-')[1]);row['target_extent_m']=.5+[.01,.04,.12][i]
        trained=fit_training_scale(source)
        self.assertAlmostEqual(trained['s0_m'],.04)
        self.assertEqual(trained['objects'],3)
        with self.assertRaises(ValueError):fit_training_scale(source+[source[0]])

    def test_training_and_calibration_objects_must_be_disjoint(self):
        with self.assertRaises(ValueError):calibrate_object_intervals(rows(24,'train'),fit())

    def test_missing_condition_is_not_silently_dropped(self):
        with self.assertRaises(ValueError):calibrate_object_intervals(rows(24)[:-1],fit())

    def test_group_score_is_max_across_dependent_conditions(self):
        source=rows(24,target=.51)
        for row in source:
            if row['condition']=='occluded':row['target_extent_m']=.6
        cal=calibrate_object_intervals(source,fit())
        self.assertEqual(cal['calibration_objects'],24)
        self.assertEqual(cal['order_statistic_rank'],24)
        self.assertAlmostEqual(cal['q'],5.)
        self.assertEqual({r['score']for r in cal['per_object']},{cal['q']})

    def test_finite_rank_boundary_and_unbounded_small_sample(self):
        self.assertEqual(finite_sample_rank(19),19)
        self.assertEqual(finite_sample_rank(18),19)
        cal=calibrate_object_intervals(rows(18),fit())
        self.assertFalse(cal['finite']);self.assertIsNone(cal['q'])
        output=apply_to_distribution([.5],[1.],cal)
        self.assertEqual(output['calibrated_interval']['lower_m'],0.)
        self.assertIsNone(output['calibrated_interval']['upper_m'])
        self.assertIsNone(output['calibrated_interval']['width_m'])
        bad=copy.deepcopy(cal);bad['finite']=True;bad['q']=10.
        with self.assertRaises(ValueError):apply_to_distribution([.5],[1.],bad)

    def test_opposing_face_point_mass_retains_measured_residual_interval(self):
        cal=calibrate_object_intervals(rows(24,target=.54),fit())
        output=apply_to_distribution([.5],[1.],cal)
        self.assertEqual(output['base_interval']['width_m'],0.)
        self.assertAlmostEqual(output['calibrated_interval']['lower_m'],.46)
        self.assertAlmostEqual(output['calibrated_interval']['upper_m'],.54)
        self.assertFalse(output['posterior_distribution_changed'])

    def test_raw_distribution_is_unchanged_and_application_has_no_truth(self):
        cal=calibrate_object_intervals(rows(24),fit())
        x=[.2,.4,.7,1.2];p=[.1,.3,.5,.1];x0=x.copy();p0=p.copy()
        result=apply_to_distribution(x,p,cal)
        self.assertEqual(result['raw_distribution'],dict(depth_values_m=x0,probability_mass=p0))
        self.assertEqual(x,x0);self.assertEqual(p,p0)
        self.assertFalse(result['actor_or_gain_changed'])
        self.assertNotIn('calibrated_probability_mass',result)

    def test_intervals_adapt_to_base_halfwidth_without_shrinking(self):
        cal=calibrate_object_intervals(rows(24),fit())
        narrow=apply_to_distribution([.5],[1.],cal)
        broad=apply_to_distribution([.2,1.4],[.5,.5],cal)
        self.assertGreater(broad['correction']['scale_m'],narrow['correction']['scale_m'])
        self.assertGreaterEqual(broad['calibrated_interval']['width_m'],broad['base_interval']['width_m'])

    def test_fallbacks_remain_and_upper_not_clipped_to_original_prior(self):
        source=rows(24,target=1.8,lower=.2,upper=1.4)
        for row in source:row['fallback']=True
        cal=calibrate_object_intervals(source,fit())
        self.assertEqual(cal['fallback_cases'],96)
        result=apply_to_distribution([.2,1.4],[.5,.5],cal)
        self.assertAlmostEqual(result['calibrated_interval']['upper_m'],1.8)
        self.assertEqual(result['calibrated_interval']['lower_m'],0.)

    def test_permutation_and_duplicate_particle_representation_invariance(self):
        source=rows(24)
        a=calibrate_object_intervals(source,fit())
        b=calibrate_object_intervals(source[::-1],fit())
        self.assertEqual(a,b)
        normal=apply_to_distribution([.5],[1.],a)
        duplicate=apply_to_distribution([.5,.5],[.3,.7],a)
        self.assertEqual(normal['base_interval'],duplicate['base_interval'])
        self.assertEqual(normal['calibrated_interval'],duplicate['calibrated_interval'])

    def test_zero_training_residual_has_explicit_numerical_floor(self):
        trained=fit_training_scale(rows(12,'train',target=.5))
        self.assertEqual(trained['s0_m'],NUMERICAL_SCALE_FLOOR_M)
        self.assertTrue(trained['numerical_floor_applied'])

    def test_finite_exchangeable_rank_example_covers_19_of_20(self):
        # All 20 possible choices of held-out object are checked, not sampled.
        all_targets=[.5+i*.01 for i in range(20)];covered=0
        for heldout,truth in enumerate(all_targets):
            source=[]
            for i,target in enumerate(all_targets):
                if i==heldout:continue
                source+=rows(1,f'cal-{i}',target=target)
            cal=calibrate_object_intervals(source,fit())
            bound=apply_to_distribution([.5],[1.],cal)['calibrated_interval']
            covered+=bound['lower_m']-1e-12<=truth<=bound['upper_m']+1e-12
        self.assertEqual(covered,19)

    def test_nonfinite_invalid_inputs_are_rejected(self):
        for alpha in (0,1,float('nan')):
            with self.assertRaises(ValueError):finite_sample_rank(24,alpha)
        bad=rows(24);bad[0]['target_extent_m']=float('nan')
        with self.assertRaises(ValueError):calibrate_object_intervals(bad,fit())
        with self.assertRaises(ValueError):calibrate_object_intervals(rows(24),fit(),alpha=.1)
        cal=calibrate_object_intervals(rows(24),fit())
        for x,p in [([.5],[float('nan')]),([-.5],[1.]),([.5],[0.])]:
            with self.assertRaises(ValueError):apply_to_distribution(x,p,cal)


if __name__=='__main__':
    unittest.main(verbosity=2)
