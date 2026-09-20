"""Local, pinned CPU detector adapter for the V37 frontend contract.

This executes a detector on supplied RGB pixels. It supplies no natural-device
configuration registry and measures no industrial accuracy or planning gain.
Only a separately reviewed registry can turn a natural class into a task prior.
"""
import copy
import hashlib
import os
from pathlib import Path
import re
import time

import numpy as np

from nso.semantic_frontend_v37 import (
    TAXONOMIES, normalize_detections, resolve_configuration_prior,
    validate_prior_registry, validate_rgb_image,
)


SCHEMA = 'cpu_semantic_detector_v37/1'
ULTRALYTICS_VERSION = '8.3.200'
PREDICTION_SETTINGS = {
    'conf': 0.25, 'iou': 0.7, 'imgsz': 640, 'device': 'cpu',
    'half': False, 'verbose': False, 'save': False, 'stream': False,
    'augment': False, 'agnostic_nms': False, 'max_det': 300, 'rect': False,
}


def _sha256(path):
    digest = hashlib.sha256()
    with open(path, 'rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def _load_pinned_backend(checkpoint_path):
    # These must precede the lazy import. The runner owns a bounded local
    # configuration directory and may also guard sockets during execution.
    if not os.environ.get('YOLO_CONFIG_DIR'):
        raise ValueError('set YOLO_CONFIG_DIR to an explicit local runtime directory')
    os.environ['YOLO_OFFLINE'] = 'true'
    os.environ['YOLO_AUTOINSTALL'] = 'false'
    os.environ['YOLO_VERBOSE'] = 'false'
    import ultralytics
    if ultralytics.__version__ != ULTRALYTICS_VERSION:
        raise RuntimeError(f'ultralytics {ULTRALYTICS_VERSION} required')
    import torch
    import cv2
    from ultralytics.utils import torch_utils
    # Disable the package's optional remote synchronization before model use.
    ultralytics.settings.update({'sync': False})
    # select_device('cpu') resets Torch's thread count using this imported
    # constant on the first prediction; setting Torch alone is insufficient.
    torch_utils.NUM_THREADS = 1
    torch.set_num_threads(1)
    if torch.get_num_interop_threads() != 1:
        torch.set_num_interop_threads(1)
    model = ultralytics.YOLO(checkpoint_path, task='detect')
    model.to('cpu')
    model.model.float()
    return model, {
        'ultralytics_version': ultralytics.__version__,
        'torch_version': torch.__version__, 'torch_num_threads': torch.get_num_threads(),
        'torch_num_interop_threads': torch.get_num_interop_threads(),
        'cv2_num_threads': cv2.getNumThreads(),
        'thread_scope': 'Torch intraop/inter-op limits; not an entire-process thread count',
        'ultralytics_num_threads_runtime_override': 1,
        'installed_package_source_modified': False,
        'device': 'cpu', 'dtype': 'float32', 'offline_requested': True,
        'autoinstall_enabled': False, 'remote_sync_enabled': False,
    }


def _numpy(value):
    if hasattr(value, 'detach'):
        value = value.detach().cpu().numpy()
    return np.asarray(value)


class CPUSemanticDetectorV37:
    """Load one verified local checkpoint and accept RGB arrays only.

    ``_backend_factory`` is an explicit unit-test seam. Its records are marked
    as injected, with model_execution_performed=False; they cannot be mistaken
    for the actual pinned detector's execution records.
    """

    def __init__(self, checkpoint_path, expected_sha256, *, target_taxonomy='coco80',
                 registry=None, configuration_ids=('h0', 'h1'), _backend_factory=None):
        if not isinstance(expected_sha256, str) or re.fullmatch('[0-9a-f]{64}', expected_sha256) is None:
            raise ValueError('explicit lowercase checkpoint SHA256 required')
        if not isinstance(checkpoint_path, (str, os.PathLike)):
            raise ValueError('explicit local checkpoint path required')
        # Keep the .pt alias: resolving an LFS symlink to an extensionless object
        # makes the backend treat the checkpoint as an unsupported model format.
        path = Path(os.path.abspath(os.fspath(checkpoint_path)))
        if path.suffix != '.pt':
            raise ValueError('local checkpoint alias must have a .pt suffix')
        if not path.is_file():
            raise FileNotFoundError(f'local checkpoint does not exist: {path}')
        actual_sha256 = _sha256(path)
        if actual_sha256 != expected_sha256:
            raise ValueError('local checkpoint SHA256 does not match the frozen expected value')
        if not isinstance(target_taxonomy, str) or target_taxonomy not in TAXONOMIES:
            raise ValueError('target_taxonomy must be coco80 or custom52')
        labels = list(configuration_ids)
        if (len(labels) < 2 or any(not isinstance(label, str) or not label.strip() for label in labels)
                or len(set(labels)) != len(labels)):
            raise ValueError('at least two unique configuration IDs required')
        if registry is not None:
            validate_prior_registry(registry)
            if registry['configuration_ids'] != labels:
                raise ValueError('registry configuration order must match the requested hypotheses')
        self._checkpoint_path = str(path)
        self._checkpoint_sha256 = actual_sha256
        self._target_taxonomy = target_taxonomy
        self._registry = copy.deepcopy(registry)
        self._configuration_ids = tuple(labels)
        self._injected = _backend_factory is not None
        factory = _load_pinned_backend if _backend_factory is None else _backend_factory
        self._model, self._runtime = factory(self._checkpoint_path)
        names = self._model.names
        names = tuple(names.get(index) for index in range(80)) if isinstance(names, dict) else tuple(names)
        if names != TAXONOMIES['coco80']:
            raise ValueError('checkpoint class names must exactly match the coco80 namespace')
        self._runtime = dict(self._runtime)
        self._runtime['execution_backend'] = ('injected_test_backend' if self._injected
                                              else 'ultralytics_pinned_cpu')

    def detect(self, rgb):
        """Infer from RGB pixels and retain original IDs plus uniform fallback.

        The runtime includes host wall time, not a throughput or accuracy claim.
        Model scores are uncalibrated; low-score proposals suppressed by YOLO
        itself are not observable rejection records in this adapter.
        """
        validate_rgb_image(rgb)
        # Ultralytics NumPy input is HWC BGR. Use a copy so the backend cannot
        # mutate caller-owned RGB or receive a negative-stride channel view.
        bgr = np.ascontiguousarray(rgb[..., ::-1]).copy()
        started = time.perf_counter()
        outputs = self._model.predict(source=bgr, **PREDICTION_SETTINGS)
        elapsed = time.perf_counter() - started
        if not self._injected:
            import torch
            import cv2
            predictor = self._model.predictor.model
            if (torch.get_num_threads() != 1 or torch.get_num_interop_threads() != 1 or
                    predictor.device.type != 'cpu' or predictor.fp16 or
                    any(parameter.device.type != 'cpu' or parameter.dtype != torch.float32
                        for parameter in self._model.model.parameters())):
                raise RuntimeError('actual detector runtime must remain CPU float32 with one thread')
            self._runtime.update(torch_num_threads=torch.get_num_threads(),
                                 torch_num_interop_threads=torch.get_num_interop_threads(),
                                 cv2_num_threads=cv2.getNumThreads())
        if not isinstance(outputs, (list, tuple)) or len(outputs) != 1:
            raise ValueError('one RGB input must produce exactly one detector result')
        output = outputs[0]
        if (tuple(output.orig_shape) != tuple(rgb.shape[:2]) or
                not np.array_equal(output.orig_img, rgb[..., ::-1])):
            raise ValueError('detector result does not preserve the supplied image alignment')
        if output.boxes is None:
            raise ValueError('a detection checkpoint must return a boxes object, including when empty')
        result = normalize_detections(
            rgb=rgb, boxes_xyxy=_numpy(output.boxes.xyxy),
            scores=_numpy(output.boxes.conf), class_ids=_numpy(output.boxes.cls),
            source_taxonomy='coco80', target_taxonomy=self._target_taxonomy,
            min_score=PREDICTION_SETTINGS['conf'])
        result.update({
            'adapter_schema': SCHEMA,
            'model_execution_performed': not self._injected,
            'image_detection_alignment_verified': True,
            'model': {'checkpoint_sha256': self._checkpoint_sha256,
                      'format': 'pytorch_pt', 'source_taxonomy': 'coco80'},
            'runtime': dict(self._runtime, prediction_wall_seconds=elapsed,
                            prediction_settings=dict(PREDICTION_SETTINGS)),
            'pixel_conversion': 'caller RGB -> contiguous BGR ndarray -> backend RGB tensor',
            'input_filename_used_as_feature': False,
            'detector_score_calibration': 'uncalibrated detector score',
            'upstream_score_suppression_threshold': PREDICTION_SETTINGS['conf'],
            'upstream_suppressed_proposals_available': False,
            'natural_device_accuracy_measured': False,
            'planning_gain_measured': False,
            'configuration_priors': [],
        })
        rows = sorted(result['detections'] + result['rejected_detections'],
                      key=lambda row: row['source_index'])
        for row in rows:
            prior = resolve_configuration_prior(row, registry=self._registry,
                                                configuration_ids=self._configuration_ids)
            # resolve_configuration_prior itself performs no detector execution.
            # Keep that flag false inside this nested, stateless prior record.
            result['configuration_priors'].append({'source_index': row['source_index'], **prior})
        return result
