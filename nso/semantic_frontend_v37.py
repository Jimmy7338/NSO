"""CPU contract for externally supplied semantic detections; no model execution.

The caller supplies RGB pixels and detections from those pixels. This module
does not run a detector, infer device identity, verify image/detection alignment,
or connect natural classes to the V35 artificial marker IDs. A detector adapter
must explicitly handle its own RGB/BGR convention before supplying results.

Registry evidence declarations are checked structurally, not scientifically
validated here. A registry must be reviewed and frozen independently of the
evaluation set before use. Its distributions are task priors, never calibrated
detector probabilities or repeatable observation likelihoods.
"""
import hashlib
import math
import re

import numpy as np

from semantic.class_mapping import (
    COCO_CLASS_NAMES, CUSTOM_SEMANTIC_LABELS, YOLO_TO_CUSTOM_MAPPING,
)


TAXONOMIES = {'coco80': tuple(COCO_CLASS_NAMES),
              'custom52': tuple(CUSTOM_SEMANTIC_LABELS)}
SCHEMA = 'semantic_frontend_v37/1'
PRIOR_SCHEMA = 'semantic_configuration_prior_registry_v37/1'


def validate_rgb_image(rgb):
    """Hash explicitly declared RGB HWC uint8 pixels without changing channels."""
    if not isinstance(rgb, np.ndarray) or rgb.dtype != np.uint8:
        raise ValueError('RGB must be a numpy uint8 array')
    if rgb.ndim != 3 or rgb.shape[2] != 3 or min(rgb.shape[:2]) <= 0:
        raise ValueError('nonempty HWC RGB with exactly three channels required')
    header = f'rgb8:{rgb.shape[0]}:{rgb.shape[1]}:3\n'.encode('ascii')
    digest = hashlib.sha256(header + np.ascontiguousarray(rgb).tobytes()).hexdigest()
    return {'encoding': 'rgb8', 'shape': list(rgb.shape), 'sha256': digest,
            'hash_definition': 'ASCII rgb8:H:W:3 newline followed by C-order pixels',
            'channels_changed': False}


def _unit_scalar(value, name):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, float, np.integer, np.floating)):
        raise ValueError(f'{name} must be a number in [0,1]')
    result = float(value)
    if not math.isfinite(result) or not 0 <= result <= 1:
        raise ValueError(f'{name} must be finite and in [0,1]')
    return result


def _numeric(value, shape, name):
    array = np.asarray(value)
    if array.shape != shape or array.dtype.kind not in 'fiu' or not np.isfinite(array).all():
        raise ValueError(f'{name} must be a finite numeric array with shape {shape}')
    return array


def normalize_detections(*, rgb, boxes_xyxy, scores, class_ids, source_taxonomy,
                         target_taxonomy=None, min_score=0.2):
    """Return JSON-compatible accepted/rejected records retaining source indices.

    Pixel boxes use half-open bounds: 0 <= x1 < x2 <= width (likewise y).
    Integral detector class floats are accepted; unknown IDs (-1 or IDs beyond
    the declared taxonomy) are retained as rejected records. Only coco80 to
    custom52 conversion is supported; neither namespace is a V35 marker type.
    """
    image = validate_rgb_image(rgb)
    if not isinstance(source_taxonomy, str) or source_taxonomy not in TAXONOMIES:
        raise ValueError('explicit source_taxonomy coco80 or custom52 required')
    target_taxonomy = source_taxonomy if target_taxonomy is None else target_taxonomy
    if not isinstance(target_taxonomy, str) or target_taxonomy not in TAXONOMIES or (source_taxonomy != target_taxonomy and
                                            (source_taxonomy, target_taxonomy) != ('coco80', 'custom52')):
        raise ValueError('only identity or explicit coco80 to custom52 mapping is supported')
    threshold = _unit_scalar(min_score, 'min_score')
    classes = np.asarray(class_ids)
    if classes.ndim != 1:
        raise ValueError('class_ids must be one dimensional')
    if any(isinstance(value, (bool, np.bool_)) for value in np.asarray(class_ids, dtype=object)):
        raise ValueError('boolean class IDs are not permitted')
    count = len(classes)
    classes = _numeric(classes, (count,), 'class_ids')
    if (classes < -1).any() or (classes > 2**31 - 1).any() or not np.equal(classes, np.floor(classes)).all():
        raise ValueError('class_ids must be integers between -1 and 2**31-1')
    classes = classes.astype(np.int64)
    boxes = _numeric(boxes_xyxy, (count, 4), 'boxes_xyxy')
    confidence = _numeric(scores, (count,), 'scores')
    if (confidence < 0).any() or (confidence > 1).any():
        raise ValueError('scores must be in [0,1]')
    height, width = rgb.shape[:2]
    if ((boxes < 0).any() or (boxes[:, [0, 2]] > width).any() or
            (boxes[:, [1, 3]] > height).any() or
            (boxes[:, 0] >= boxes[:, 2]).any() or (boxes[:, 1] >= boxes[:, 3]).any()):
        raise ValueError('nonempty xyxy boxes must lie within the supplied RGB image')

    source_names = TAXONOMIES[source_taxonomy]
    records = []
    for index, raw_id in enumerate(classes):
        raw_id = int(raw_id)
        known = 0 <= raw_id < len(source_names)
        mapped_id = raw_id if known else -1
        if known and target_taxonomy != source_taxonomy:
            mapped_id = YOLO_TO_CUSTOM_MAPPING.get(raw_id, -1)
        reason = ('unknown_source_class' if not known else
                  'unsupported_target_class' if mapped_id < 0 else
                  'low_detector_score' if confidence[index] < threshold else None)
        records.append({
            'source_index': index, 'source_taxonomy': source_taxonomy,
            'source_class_id': raw_id,
            'source_class_name': source_names[raw_id] if known else None,
            'target_taxonomy': target_taxonomy,
            'target_class_id': mapped_id if mapped_id >= 0 else None,
            'target_class_name': TAXONOMIES[target_taxonomy][mapped_id] if mapped_id >= 0 else None,
            'box_xyxy': [float(x) for x in boxes[index]],
            'detector_score': float(confidence[index]),
            'min_detector_score': threshold,
            'accepted': reason is None, 'rejection_reason': reason,
        })
    # One decision per original row; never apply a shortened class mask to boxes.
    return {'schema': SCHEMA, 'image': image, 'source_taxonomy': source_taxonomy,
            'target_taxonomy': target_taxonomy, 'min_score': threshold,
            'input_count': count,
            'detections': [row for row in records if row['accepted']],
            'rejected_detections': [row for row in records if not row['accepted']],
            'model_execution_performed': False,
            'detector_score_is_configuration_probability': False,
            'image_detection_alignment_verified': False}


def validate_prior_registry(registry):
    """Validate a JSON registry contract; this does not authenticate its claims.

    Entries are keyed by ORIGINAL taxonomy and ID (e.g. ``coco80:56``), with
    matching class_name, configuration_probabilities, and evidence fields:
    reference, sha256, scope, independent_of_evaluation=True. No registry is
    bundled because the current project has no verified natural-device prior.
    """
    if not isinstance(registry, dict) or registry.get('schema') != PRIOR_SCHEMA:
        raise ValueError('explicit V37 prior registry schema required')
    labels = registry.get('configuration_ids')
    if (not isinstance(labels, list) or len(labels) < 2 or
            any(not isinstance(x, str) or not x.strip() for x in labels) or len(set(labels)) != len(labels)):
        raise ValueError('at least two unique explicit configuration_ids required')
    entries = registry.get('entries')
    if not isinstance(entries, dict):
        raise ValueError('registry entries must be a mapping')
    for key, entry in entries.items():
        if not isinstance(key, str) or ':' not in key or not isinstance(entry, dict):
            raise ValueError('entry key must explicitly include source taxonomy and ID')
        taxonomy, text_id = key.split(':', 1)
        if taxonomy not in TAXONOMIES or not text_id.isdecimal():
            raise ValueError('entry must use a supported natural-class namespace')
        class_id = int(text_id)
        if key != f'{taxonomy}:{class_id}' or not 0 <= class_id < len(TAXONOMIES[taxonomy]):
            raise ValueError('registry class ID is outside its taxonomy')
        if entry.get('class_name') != TAXONOMIES[taxonomy][class_id]:
            raise ValueError('registry class_name disagrees with the declared taxonomy')
        probabilities = entry.get('configuration_probabilities')
        if not isinstance(probabilities, list) or len(probabilities) != len(labels):
            raise ValueError('one probability per explicit configuration required')
        values = [_unit_scalar(value, 'configuration probability') for value in probabilities]
        if not math.isclose(sum(values), 1., rel_tol=0., abs_tol=1e-9):
            raise ValueError('configuration probabilities must sum to one')
        evidence = entry.get('evidence')
        if (not isinstance(evidence, dict) or evidence.get('independent_of_evaluation') is not True or
                any(not isinstance(evidence.get(field), str) or not evidence[field].strip()
                    for field in ('reference', 'scope')) or
                not isinstance(evidence.get('sha256'), str) or
                re.fullmatch('[0-9a-f]{64}', evidence['sha256']) is None):
            raise ValueError('independent evidence reference, scope and SHA256 declaration required')
    return registry


def resolve_configuration_prior(detection, *, registry=None, configuration_ids=('h0', 'h1')):
    """Resolve ONE semantic prior, statelessly; never multiply repeated calls.

    Unknown/unsupported/low-score records and absent registry entries yield a
    uniform prior. Detector confidence gates acceptance only; a valid registry
    supplies the entire prior distribution. The consumer must keep this as one
    replaceable prior, not add it per frame as independent evidence.
    """
    labels = list(configuration_ids)
    if (len(labels) < 2 or any(not isinstance(x, str) or not x.strip() for x in labels)
            or len(set(labels)) != len(labels)):
        raise ValueError('at least two unique configuration IDs required')
    if not isinstance(detection, dict):
        raise ValueError('one normalized detection record required')
    taxonomy = detection.get('source_taxonomy')
    class_id = detection.get('source_class_id')
    if (not isinstance(taxonomy, str) or taxonomy not in TAXONOMIES or
            isinstance(class_id, bool) or not isinstance(class_id, int)):
        raise ValueError('normalized original class namespace and integer ID required')
    score = _unit_scalar(detection.get('detector_score'), 'detector_score')
    minimum_score = _unit_scalar(detection.get('min_detector_score'), 'min_detector_score')
    known = 0 <= class_id < len(TAXONOMIES[taxonomy])
    if known and detection.get('source_class_name') != TAXONOMIES[taxonomy][class_id]:
        raise ValueError('normalized source name/ID mismatch')
    if not isinstance(detection.get('accepted'), bool):
        raise ValueError('normalized acceptance decision required')
    reason = 'no_prior_registry'
    evidence = None
    values = [1. / len(labels)] * len(labels)
    if registry is not None:
        validate_prior_registry(registry)
        if registry['configuration_ids'] != labels:
            raise ValueError('registry configuration order must match the requested hypotheses')
    if score < minimum_score:
        reason = 'low_detector_score'
    elif not known or not detection['accepted'] or detection.get('rejection_reason') is not None:
        reason = detection.get('rejection_reason') or 'unknown_or_rejected_detection'
    elif registry is not None:
        entry = registry['entries'].get(f'{taxonomy}:{class_id}')
        reason = 'class_has_no_independent_prior'
        if entry is not None:
            values = [float(value) for value in entry['configuration_probabilities']]
            evidence = dict(entry['evidence'])
            reason = 'explicit_registry_prior'
    return {'configuration_ids': labels, 'probabilities': values, 'reason': reason,
            'geometry_fallback': reason != 'explicit_registry_prior',
            'source_taxonomy': taxonomy, 'source_class_id': class_id,
            'detector_score': score, 'detector_score_used_as_probability': False,
            'evidence': evidence, 'evidence_scientifically_validated_here': False,
            'calibrated_probability': False,
            'repetition_policy': 'replace one semantic prior; never accumulate per frame',
            'model_execution_performed': False}
