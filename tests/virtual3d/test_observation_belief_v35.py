"""V35 saved-packet intervention tests: zero new world/TSDF/main calls."""
import json
import math
from pathlib import Path
import unittest

import numpy as np

from nso.observation_belief_v35 import (
    CONFIG_V35, ObservationBeliefV35, PublicTemplatesV35,
)

ROOT = Path(__file__).resolve().parents[2]
PIXELS = ROOT / 'audit_results/v34_pixel_information_20260918'
ROUTES = ROOT / 'audit_results/v34_fixed_measurement_20260918'


def substitute_rgb(rgb):
    output = rgb.copy()
    a = np.all(rgb == np.array([40, 100, 220], np.uint8), axis=-1)
    b = np.all(rgb == np.array([220, 60, 40], np.uint8), axis=-1)
    output[a] = (220, 60, 40); output[b] = (40, 100, 220)
    return output


class ObservationBeliefTestsV35(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        geometry = json.loads((ROOT / 'audit_results/v33_direction_information_r1_20260917/P00_geometry.json').read_text())
        # Only public graph poses are passed to the online class.
        cls.templates = PublicTemplatesV35.from_saved(PIXELS, 'P00', geometry['tables'][0]['poses'])
        cls.saved = []
        for case in ('case00', 'case01'):
            trace = json.loads((ROUTES / case / 'trace.json').read_text())
            sequence = []
            for row in trace:
                with np.load(ROUTES / case / 'packets' / f"{row['paid']:03}.npz", allow_pickle=False) as packet:
                    sequence.append(dict(pose=tuple(row['pose_v33']), step=row['paid'],
                                         depth=packet['frame__depth_m'].copy(),
                                         rgb=packet['frame__color_rgb'].copy(),
                                         ranges=packet['scan__ranges_m'].copy()))
            cls.saved.append(sequence)

    def test_paid_prefix_geometry_is_identical_and_uninformative(self):
        beliefs = [ObservationBeliefV35(self.templates, 'G') for _ in (0, 1)]
        for pair in zip(self.saved[0][:19], self.saved[1][:19]):
            np.testing.assert_array_equal(pair[0]['depth'], pair[1]['depth'])
            np.testing.assert_array_equal(pair[0]['ranges'], pair[1]['ranges'])
            for belief, packet in zip(beliefs, pair):
                report = belief.update(**packet)
                self.assertEqual(report['geometry_log_odds'], 0.)
                self.assertEqual(report['semantic_log_odds'], 0.)
                self.assertEqual(report['probabilities'], [.5, .5])
                self.assertIsNone(report['rgb_pixel_counts'])

    def test_class_needs_two_xy_and_is_one_global_prior(self):
        for hypothesis in (0, 1):
            belief = ObservationBeliefV35(self.templates, 'S')
            first = belief.update(**self.saved[hypothesis][0])
            self.assertEqual(first['semantic_log_odds'], 0.)
            seen_qualified = False
            for packet in self.saved[hypothesis][1:19]:
                report = belief.update(**packet)
                if report['class_distinct_xy'][str(2 + hypothesis)] >= 2:
                    self.assertAlmostEqual(report['probabilities'][hypothesis], .9)
                    seen_qualified = True
            self.assertTrue(seen_qualified)
            self.assertAlmostEqual(abs(belief.semantic_log_odds), math.log(9.))

    def test_no_cue_prefix_falls_back_to_common_prior(self):
        belief = ObservationBeliefV35(self.templates, 'S')
        for source in self.saved[0][:19]:
            packet = dict(source); packet['rgb'] = np.full_like(packet['rgb'], 127)
            self.assertEqual(belief.update(**packet)['probabilities'], [.5, .5])

    def test_class_substitution_changes_prior_with_same_geometry(self):
        a = ObservationBeliefV35(self.templates, 'S')
        b = ObservationBeliefV35(self.templates, 'S')
        for packet in self.saved[0][:19]:
            ra = a.update(**packet)
            rb = b.update(**dict(packet, rgb=substitute_rgb(packet['rgb'])))
            self.assertEqual(ra['geometry_log_odds'], rb['geometry_log_odds'])
        self.assertAlmostEqual(a.probabilities[0], .9)
        self.assertAlmostEqual(b.probabilities[0], .1)

    def test_saved_new_geometry_corrects_swapped_prior(self):
        for hypothesis in (0, 1):
            belief = ObservationBeliefV35(self.templates, 'swapped')
            for packet in self.saved[hypothesis][:19]:
                belief.update(**packet)
            self.assertAlmostEqual(belief.probabilities[hypothesis], .1)
            corrected_at = None
            for packet in self.saved[hypothesis][19:]:
                report = belief.update(**packet)
                if report['probabilities'][hypothesis] > .5 and corrected_at is None:
                    corrected_at = packet['step']
                    self.assertNotEqual(report['geometry_applied_log_odds'], 0.)
            self.assertIsNotNone(corrected_at)
            self.assertGreater(belief.probabilities[hypothesis], .99)

    def test_feedback_ablation_retains_wrong_prior(self):
        for hypothesis in (0, 1):
            belief = ObservationBeliefV35(self.templates, 'swapped_no_feedback')
            for packet in self.saved[hypothesis]:
                report = belief.update(**packet)
                self.assertEqual(report['geometry_log_odds'], 0.)
                self.assertEqual(report['geometry_applied_log_odds'], 0.)
            self.assertAlmostEqual(belief.probabilities[hypothesis], .1)

    def test_repeated_pose_never_amplifies_geometry_or_semantics(self):
        belief = ObservationBeliefV35(self.templates, 'S')
        witness = None
        for packet in self.saved[0]:
            report = belief.update(**packet)
            if report['geometry_applied_log_odds'] > 0:
                witness = packet; break
        self.assertIsNotNone(witness)
        initial = (belief.geometry_log_odds, belief.semantic_log_odds, belief.probabilities)
        for offset in range(1, 9):
            report = belief.update(**dict(witness, step=witness['step'] + offset))
            self.assertFalse(report['new_geometry_pose'])
            self.assertEqual(report['geometry_applied_log_odds'], 0.)
            self.assertEqual((belief.geometry_log_odds, belief.semantic_log_odds, belief.probabilities), initial)

    def test_conflicting_supported_classes_clear_prior(self):
        belief = ObservationBeliefV35(self.templates, 'S')
        for packet in self.saved[0][:19]:
            belief.update(**packet)
        self.assertAlmostEqual(belief.probabilities[0], .9)
        packet = self.saved[0][0]
        report = belief.update(**dict(packet, step=19, rgb=substitute_rgb(packet['rgb'])))
        self.assertTrue(report['class_conflict'])
        self.assertEqual(report['semantic_log_odds'], 0.)
        self.assertEqual(report['probabilities'], [.5, .5])

    def test_forecast_is_soft_and_requires_both_templates(self):
        forecasts = [self.templates.template_information(node) for node in range(len(self.templates.poses))]
        self.assertTrue(any(row['informative'] for row in forecasts))
        for row in forecasts:
            self.assertIn(row['correct_probability'], (.5, .99))
            if row['informative']:
                self.assertGreaterEqual(row['public_clean_template_log_odds'][0], math.log(99.))
                self.assertLessEqual(row['public_clean_template_log_odds'][1], -math.log(99.))
        for packet in self.saved[0][:19]:
            node = self.templates.node_for_pose[packet['pose']]
            self.assertFalse(self.templates.template_information(node)['informative'])

    def test_two_supported_classes_in_one_frame_clear_existing_prior(self):
        for mode in ('S', 'swapped', 'swapped_no_feedback'):
            belief = ObservationBeliefV35(self.templates, mode)
            for packet in self.saved[0][:19]:
                belief.update(**packet)
            self.assertAlmostEqual(abs(belief.semantic_log_odds), math.log(9.))
            packet = self.saved[0][0]
            mixed = packet['rgb'].copy()
            mixed[:4, :4] = (220, 60, 40)
            report = belief.update(**dict(packet, step=19, rgb=mixed))
            self.assertTrue(report['class_conflict'])
            self.assertIsNone(report['observed_class_after_intervention'])
            self.assertEqual(report['semantic_log_odds'], 0.)
            self.assertEqual(report['probabilities'], [.5, .5])

    def test_bad_inputs_are_rejected_before_belief_changes(self):
        belief = ObservationBeliefV35(self.templates, 'S')
        packet = self.saved[0][0]
        with self.assertRaises(ValueError):
            belief.update(**dict(packet, depth=np.full_like(packet['depth'], np.nan)))
        with self.assertRaises(ValueError):
            belief.update(**dict(packet, rgb=packet['rgb'].astype(float)))
        with self.assertRaises(ValueError):
            belief.update(**dict(packet, step=-1))
        self.assertEqual(belief.updates, 0)
        self.assertEqual(belief.probabilities, (.5, .5))
        self.assertFalse(CONFIG_V35['calibrated_probability'])


if __name__ == '__main__':
    unittest.main()
