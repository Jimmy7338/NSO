"""Adapter boundary regressions using an explicitly identified fake backend."""
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from nso.cpu_semantic_detector_v37 import CPUSemanticDetectorV37, PREDICTION_SETTINGS
from nso.semantic_frontend_v37 import TAXONOMIES, validate_rgb_image


class FakeBackend:
    names = dict(enumerate(TAXONOMIES['coco80']))

    def __init__(self):
        self.calls = []
        self.bad_alignment = False
        self.empty = False

    def predict(self, **kwargs):
        self.calls.append(kwargs)
        bgr = kwargs['source']
        original = bgr.copy()
        if self.bad_alignment:
            original[0, 0] ^= 255
        boxes = (SimpleNamespace(xyxy=np.empty((0, 4)), conf=np.empty((0,)), cls=np.empty((0,)))
                 if self.empty else SimpleNamespace(
                     xyxy=np.array([[0, 0, 2, 2], [2, 2, 4, 4]], dtype=np.float32),
                     conf=np.array([.9, .8], dtype=np.float32), cls=np.array([2, 56], dtype=np.float32)))
        return [SimpleNamespace(orig_img=original, orig_shape=bgr.shape[:2], boxes=boxes)]


class CPUSemanticDetectorTestsV37(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        root = Path(self.directory.name)
        self.content = b'unit test checkpoint fixture; never a real model'
        self.sha256 = hashlib.sha256(self.content).hexdigest()
        self.object_path = root / self.sha256
        self.object_path.write_bytes(self.content)
        self.alias = root / 'local_fixture.pt'
        self.alias.symlink_to(self.object_path)
        self.backend = FakeBackend()
        self.loaded_paths = []
        self.rgb = np.full((4, 4, 3), [10, 20, 30], dtype=np.uint8)

    def factory(self, checkpoint_path):
        self.loaded_paths.append(checkpoint_path)
        return self.backend, {'fixture_only': True}

    def detector(self, **changes):
        arguments = dict(checkpoint_path=self.alias, expected_sha256=self.sha256,
                         target_taxonomy='custom52', _backend_factory=self.factory)
        arguments.update(changes)
        return CPUSemanticDetectorV37(**arguments)

    def test_missing_mismatched_and_extensionless_checkpoints_never_load_backend(self):
        for change, exception in [({'checkpoint_path': self.alias.parent / 'missing.pt'}, FileNotFoundError),
                                  ({'expected_sha256': '0' * 64}, ValueError),
                                  ({'checkpoint_path': self.object_path}, ValueError),
                                  ({'expected_sha256': None}, ValueError)]:
            with self.subTest(change=change), self.assertRaises(exception):
                self.detector(**change)
        self.assertEqual(self.loaded_paths, [])

    def test_rgb_conversion_alias_and_original_rows_survive_backend_boundary(self):
        detector = self.detector()
        result = detector.detect(self.rgb)
        self.assertEqual(self.loaded_paths, [str(self.alias)])
        call = self.backend.calls[0]
        self.assertEqual(set(call), {'source', *PREDICTION_SETTINGS})
        self.assertIsInstance(call['source'], np.ndarray)
        self.assertTrue(call['source'].flags.c_contiguous)
        np.testing.assert_array_equal(call['source'][0, 0], [30, 20, 10])
        np.testing.assert_array_equal(self.rgb[0, 0], [10, 20, 30])
        self.assertEqual(result['image'], validate_rgb_image(self.rgb))
        self.assertEqual(result['model']['checkpoint_sha256'], self.sha256)
        self.assertEqual(result['detections'][0]['source_index'], 1)
        self.assertEqual(result['detections'][0]['source_class_name'], 'chair')
        self.assertEqual(result['detections'][0]['box_xyxy'], [2., 2., 4., 4.])
        self.assertEqual(result['rejected_detections'][0]['source_class_name'], 'car')
        self.assertFalse(result['input_filename_used_as_feature'])
        self.assertFalse(result['model_execution_performed'])
        self.assertEqual(result['runtime']['execution_backend'], 'injected_test_backend')
        self.assertFalse(result['natural_device_accuracy_measured'])
        self.assertTrue(all(p['probabilities'] == [.5, .5] for p in result['configuration_priors']))
        self.assertFalse(any(p['evidence_scientifically_validated_here'] for p in result['configuration_priors']))
        json.dumps(result, allow_nan=False)

    def test_wrong_namespace_bad_pixels_and_misaligned_result_fail_closed(self):
        self.backend.names = {0: 'type_A', 1: 'type_B'}
        with self.assertRaisesRegex(ValueError, 'coco80 namespace'):
            self.detector()
        self.backend.names = dict(enumerate(TAXONOMIES['coco80']))
        detector = self.detector()
        for invalid in [str(self.alias), self.rgb.astype(float), self.rgb[..., 0]]:
            with self.subTest(input_type=type(invalid).__name__), self.assertRaises(ValueError):
                detector.detect(invalid)
        self.assertEqual(self.backend.calls, [])
        self.backend.bad_alignment = True
        with self.assertRaisesRegex(ValueError, 'image alignment'):
            detector.detect(self.rgb)

    def test_empty_output_has_no_invented_detections_or_configuration_evidence(self):
        self.backend.empty = True
        result = self.detector().detect(self.rgb)
        self.assertEqual(result['input_count'], 0)
        self.assertEqual(result['detections'], [])
        self.assertEqual(result['rejected_detections'], [])
        self.assertEqual(result['configuration_priors'], [])
        self.assertFalse(result['upstream_suppressed_proposals_available'])
        self.assertFalse(result['model_execution_performed'])

    def test_unpinned_installed_version_is_rejected_before_model_constructor(self):
        # The fake imported module has no YOLO constructor. A wrong version
        # must fail before import/use of Torch or checkpoint deserialization.
        with patch.dict('sys.modules', {'ultralytics': SimpleNamespace(__version__='0.0.0')}), \
                patch.dict('os.environ', {'YOLO_CONFIG_DIR': self.directory.name}), \
                self.assertRaisesRegex(RuntimeError, '8.3.200 required'):
            self.detector(_backend_factory=None)


if __name__ == '__main__':
    unittest.main()
