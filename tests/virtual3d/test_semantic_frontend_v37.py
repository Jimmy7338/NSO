"""V37 contract regressions; synthetic arrays only, no model/world execution."""
import copy
import json
import unittest

import numpy as np

from nso.semantic_frontend_v37 import (
    PRIOR_SCHEMA, normalize_detections, resolve_configuration_prior,
    validate_prior_registry, validate_rgb_image,
)


class SemanticFrontendContractTestsV37(unittest.TestCase):
    def setUp(self):
        self.rgb = np.zeros((12, 16, 3), np.uint8)
        self.rgb[0, 0] = [10, 20, 30]

    def normalize(self, **changes):
        arguments = dict(rgb=self.rgb, boxes_xyxy=[[1, 2, 5, 8], [8, 3, 14, 10]],
                         scores=[.73, .99], class_ids=[56, 2],
                         source_taxonomy='coco80', target_taxonomy='custom52')
        arguments.update(changes)
        return normalize_detections(**arguments)

    def registry(self):
        # A declared fixture, not empirical calibration or a usable task prior.
        return {'schema': PRIOR_SCHEMA, 'configuration_ids': ['h0', 'h1'],
                'entries': {'coco80:56': {'class_name': 'chair',
                    'configuration_probabilities': [.65, .35],
                    'evidence': {'reference': 'unit-test fixture only', 'sha256': 'a' * 64,
                                 'scope': 'not real evidence', 'independent_of_evaluation': True}}}}

    def test_chair_car_mapping_preserves_boxes_scores_and_source_index(self):
        result = self.normalize()
        self.assertEqual(result['input_count'], 2)
        self.assertEqual(len(result['detections']), 1)
        chair = result['detections'][0]
        self.assertEqual((chair['source_index'], chair['source_class_id'], chair['target_class_id']), (0, 56, 0))
        self.assertEqual(chair['box_xyxy'], [1., 2., 5., 8.])
        self.assertEqual(chair['detector_score'], .73)
        self.assertEqual(result['rejected_detections'][0]['source_class_name'], 'car')
        reversed_rows = self.normalize(class_ids=[2, 56])
        self.assertEqual(reversed_rows['detections'][0]['source_index'], 1)
        self.assertEqual(reversed_rows['detections'][0]['box_xyxy'], [8., 3., 14., 10.])
        self.assertEqual(reversed_rows['detections'][0]['detector_score'], .99)
        json.dumps(result, allow_nan=False)

    def test_numeric_ids_two_three_do_not_become_artificial_marker_types(self):
        for taxonomy, expected_names in [('coco80', ['car', 'motorcycle']),
                                         ('custom52', ['bed', 'dining table'])]:
            result = self.normalize(source_taxonomy=taxonomy, target_taxonomy=taxonomy, class_ids=[2, 3])
            self.assertEqual([row['source_class_name'] for row in result['detections']], expected_names)
            for row in result['detections']:
                self.assertEqual(resolve_configuration_prior(row)['probabilities'], [.5, .5])
        with self.assertRaises(ValueError):
            self.normalize(source_taxonomy='type_A_B')
        with self.assertRaises(ValueError):
            self.normalize(target_taxonomy='type_A_B')

    def test_unknown_low_confidence_and_unsupported_all_fall_back(self):
        for changes, reason in [({'class_ids': [-1, 999]}, 'unknown_source_class'),
                                ({'scores': [.01, .01]}, 'low_detector_score'),
                                ({'class_ids': [2, 3]}, 'unsupported_target_class')]:
            result = self.normalize(**changes)
            self.assertFalse(result['detections'])
            row = result['rejected_detections'][0]
            self.assertEqual(row['rejection_reason'], reason)
            prior = resolve_configuration_prior(row, registry=self.registry())
            self.assertEqual(prior['probabilities'], [.5, .5])
            self.assertTrue(prior['geometry_fallback'])

    def test_high_detector_score_is_not_a_configuration_probability(self):
        row = self.normalize(scores=[.999, .99])['detections'][0]
        self.assertEqual(resolve_configuration_prior(row)['probabilities'], [.5, .5])
        prior = resolve_configuration_prior(row, registry=self.registry())
        self.assertEqual(prior['probabilities'], [.65, .35])
        self.assertFalse(prior['detector_score_used_as_probability'])
        self.assertFalse(prior['calibrated_probability'])

    def test_registry_cannot_alias_custom_id_to_coco_id_or_configuration_order(self):
        row = self.normalize(source_taxonomy='custom52', class_ids=[0, 1])['detections'][0]
        self.assertEqual(resolve_configuration_prior(row, registry=self.registry())['probabilities'], [.5, .5])
        with self.assertRaises(ValueError):
            resolve_configuration_prior(row, registry=self.registry(), configuration_ids=('h1', 'h0'))
        registry = self.registry()
        registry['entries']['custom52:2'] = registry['entries'].pop('coco80:56')
        with self.assertRaises(ValueError):
            validate_prior_registry(registry)

    def test_registry_requires_independent_evidence_and_valid_distribution(self):
        for change in ('evidence', 'independence', 'sha', 'probabilities', 'namespace'):
            registry = self.registry()
            entry = registry['entries']['coco80:56']
            if change == 'evidence':
                del entry['evidence']
            elif change == 'independence':
                entry['evidence']['independent_of_evaluation'] = False
            elif change == 'sha':
                entry['evidence']['sha256'] = 'missing'
            elif change == 'probabilities':
                entry['configuration_probabilities'] = [.9, .9]
            else:
                registry['entries']['56'] = registry['entries'].pop('coco80:56')
            with self.subTest(change=change), self.assertRaises(ValueError):
                validate_prior_registry(registry)

    def test_repeated_calls_never_accumulate_and_do_not_mutate_registry(self):
        row = self.normalize()['detections'][0]
        registry = self.registry()
        original = copy.deepcopy(registry)
        first = resolve_configuration_prior(row, registry=registry)
        for _ in range(10):
            self.assertEqual(resolve_configuration_prior(row, registry=registry), first)
        first['evidence']['reference'] = 'changed by caller'
        first['probabilities'][0] = 0.
        self.assertEqual(registry, original)

    def test_rgb_channels_are_not_swapped_and_hash_binds_shape(self):
        original = self.rgb.copy()
        record = validate_rgb_image(self.rgb)
        self.assertEqual(record['encoding'], 'rgb8')
        np.testing.assert_array_equal(self.rgb, original)
        self.assertNotEqual(record['sha256'], validate_rgb_image(self.rgb[..., ::-1])['sha256'])
        self.assertNotEqual(record['sha256'], validate_rgb_image(self.rgb.reshape(16, 12, 3))['sha256'])
        self.assertEqual(record['sha256'], validate_rgb_image(self.rgb.copy(order='F'))['sha256'])

    def test_bad_pixels_detection_shapes_and_values_are_rejected(self):
        invalid = [dict(rgb=self.rgb.astype(float)), dict(rgb=self.rgb[..., 0]),
                   dict(rgb=np.zeros((0, 3, 3), np.uint8)),
                   dict(boxes_xyxy=[[1, 2, 5, 8]]), dict(scores=[.5]),
                   dict(scores=[float('nan'), .5]), dict(scores=[1.1, .5]),
                   dict(class_ids=[56.2, 2]), dict(class_ids=[True, False]),
                   dict(class_ids=[True, 2]), dict(source_taxonomy=[]),
                   dict(class_ids=[-2, 2]), dict(class_ids=[float('inf'), 2]),
                   dict(boxes_xyxy=[[1, 2, 17, 8], [8, 3, 14, 10]]),
                   dict(boxes_xyxy=[[5, 2, 1, 8], [8, 3, 14, 10]]),
                   dict(boxes_xyxy=[[1, -1, 5, 8], [8, 3, 14, 10]]),
                   dict(min_score=True)]
        for changes in invalid:
            with self.subTest(changes=list(changes)), self.assertRaises(ValueError):
                self.normalize(**changes)

    def test_empty_detections_are_valid_without_model_execution(self):
        result = self.normalize(boxes_xyxy=np.empty((0, 4)), scores=[], class_ids=[])
        self.assertEqual(result['detections'], [])
        self.assertEqual(result['rejected_detections'], [])
        self.assertFalse(result['model_execution_performed'])
        self.assertFalse(result['image_detection_alignment_verified'])


if __name__ == '__main__':
    unittest.main()
