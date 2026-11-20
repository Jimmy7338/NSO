import tempfile
import unittest
from pathlib import Path

import numpy as np
from evaluate_stage import actual_increment, checked, sha, summarize_prediction


class EvaluationTests(unittest.TestCase):
    def test_partition_separates_geometry_old_new_unrealized_and_loss(self):
        result = summarize_prediction(np.array([1., 2., 3., 4., 0.]),
                                      np.array([.2, .05, .01, .01, .01]),
                                      np.array([.1, .04, .2, .2, .2]),
                                      np.array([.1, .2, .05, .2, .01]))
        parts = result['partition']
        self.assertEqual([parts[k]['weighted_template_area_m2'] for k in
                          ['false_geometry', 'true_already_before',
                           'true_previously_unmatched_now_matched',
                           'true_previously_unmatched_still_unrealized']], [1., 2., 3., 4.])
        self.assertEqual(result['previously_correct_now_lost_weighted_template_area_m2'], 2.)
        self.assertAlmostEqual(result['source_supported_weight_fraction'], .9)

    def test_all_invisible_is_not_success(self):
        result = summarize_prediction(np.zeros(3), np.zeros(3), np.zeros(3), np.zeros(3))
        self.assertIsNone(result['source_supported_weight_fraction'])
        self.assertIsNone(result['realization_fraction_of_source_supported_unmatched_weight'])

    def test_missing_after_does_not_invent_realization(self):
        result = summarize_prediction(np.ones(1), np.zeros(1), np.ones(1))
        self.assertNotIn('realization_fraction_of_source_supported_unmatched_weight', result)
        self.assertEqual(result['partition']['true_previously_unmatched']['predicted_points'], 1)

    def test_invalid_weight_rejected(self):
        for w in [[-1.], [np.nan], [np.inf]]:
            with self.assertRaises(ValueError):
                summarize_prediction(np.array(w), np.zeros(1), np.zeros(1))

    def test_actual_net_preserves_lost_area(self):
        result = actual_increment([.2, .2, .01, .01], [.01, .01, .2, .01], 100., 1000)
        self.assertEqual(result['gross_new_samples'], 2)
        self.assertEqual(result['lost_samples'], 1)
        self.assertAlmostEqual(result['net_correct_surface_area_estimate_m2'], .1)
        self.assertAlmostEqual(result['recall_delta_percentage_points'], 25.)

    def test_input_hash_rejection(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d)/'sample'
            p.write_bytes(b'original')
            entry = {'path': str(p), 'sha256': sha(p)}
            checked(entry, {})
            p.write_bytes(b'changed')
            with self.assertRaises(ValueError):
                checked(entry, {})


if __name__ == '__main__':
    unittest.main()
