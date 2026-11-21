"""Load declared public task data into the unmodified V35 classes; no ROS.

Manifest schema ``nso_task_bundle_v35/1`` requires task_id, scope,
execution_allowed, geometry and sensor_predictions file references (path,
sha256), hypotheses (two descriptions with source_ids), physical_sources,
prefix_actions, total_budget, binding and calibration. References resolve
relative to the manifest; explicit absolute source paths are also supported.
Geometry JSON contains the two original SavedPotentialModelV34 ``tables``.
Prediction NPZ contains poses, depth, ranges, calibration_sha256 and
binding_sha256. The two hashes use canonical_json_sha256 below.

No geometry is invented, graph edge is removed, candidate is scored or
observation is acquired here. A hash establishes file identity, not the
accuracy of a claimed physical model. Synthetic unit fixtures remain marked
non-executable and must never be treated as physical task assets.
"""
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import re
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'vendor_nso'))
from nso.cpu_four_modules_v35 import CPUFourModuleControllerV35
from nso.observation_belief_v35 import PublicTemplatesV35
from nso.pixel_information_v34 import SavedPotentialModelV34


SCHEMA = 'nso_task_bundle_v35/1'


def canonical_json_sha256(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'),
                                    ensure_ascii=False, allow_nan=False).encode()).hexdigest()


def _require(ok, reason):
    if not ok:
        raise ValueError(reason)


def _text(value):
    return isinstance(value, str) and bool(value.strip())


def _file_reference(reference, directory):
    _require(isinstance(reference, dict) and _text(reference.get('path')),
             'file reference requires an explicit path')
    expected = reference.get('sha256')
    _require(isinstance(expected, str) and re.fullmatch('[0-9a-f]{64}', expected),
             'file reference requires a SHA256')
    path = (directory / reference['path']).resolve()
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    _require(digest.hexdigest() == expected, 'sha256 mismatch: ' + str(path))
    return path, dict(path=str(path), sha256=expected)


def _matrix(value, shape, name, rigid=False):
    array = np.asarray(value, dtype=float)
    _require(array.shape == shape and np.isfinite(array).all(), name + ' must be explicit finite matrix')
    if rigid:
        rotation = array[:3, :3]
        _require(np.allclose(array[3], [0, 0, 0, 1], rtol=0, atol=1e-8)
                 and np.allclose(rotation.T @ rotation, np.eye(3), rtol=0, atol=1e-6)
                 and np.isclose(np.linalg.det(rotation), 1., rtol=0, atol=1e-6),
                 name + ' must be rigid')
    return array


def _indices(values, size, name):
    _require(type(size) is int and size > 0, name + ' requires positive source size')
    _require(isinstance(values, list) and bool(values)
             and all(type(v) is int and 0 <= v < size for v in values)
             and len(set(values)) == len(values), name + ' requires explicit unique source indices')


def _observation_contract(binding, calibration):
    _require(isinstance(binding, dict) and isinstance(calibration, dict),
             'explicit binding and calibration required')
    _require(_text(binding.get('graph_frame_id')) and _text(binding.get('physical_frame_id')),
             'binding requires both frame IDs')
    _matrix(binding.get('translation_xy_m'), (2,), 'translation_xy_m')
    _require(isinstance(binding.get('yaw_rad'), (int, float))
             and np.isfinite(binding['yaw_rad']), 'binding requires finite yaw_rad')
    _require(binding.get('meters_per_grid_unit') == 1.0
             and not isinstance(binding['meters_per_grid_unit'], bool),
             'original V35 actions require one metre per grid unit')
    _require(calibration.get('schema') == 'nso_real_sensor_calibration/1',
             'unsupported calibration schema')
    rgbd, scan, extrinsics = (calibration.get(k) for k in ('rgbd', 'scan', 'extrinsics'))
    _require(all(isinstance(v, dict) for v in (rgbd, scan, extrinsics)),
             'explicit RGBD, scan and extrinsic specifications required')
    _indices(rgbd.get('row_indices'), rgbd.get('height'), 'RGBD rows')
    _indices(rgbd.get('column_indices'), rgbd.get('width'), 'RGBD columns')
    intrinsic = _matrix(rgbd.get('intrinsic_3x3'), (3, 3), 'intrinsic_3x3')
    _require(intrinsic[0, 0] > 0 and intrinsic[1, 1] > 0
             and np.allclose(intrinsic[2], [0, 0, 1]) and abs(np.linalg.det(intrinsic)) > 1e-12,
             'invalid pinhole intrinsic')
    _require(_text(rgbd.get('optical_frame')) and rgbd.get('depth_units') == 'm'
             and rgbd.get('invalid_depth_policy') == 'nonfinite_or_nonpositive_to_zero'
             and rgbd.get('rgb_policy') == 'original_rgb8_at_selected_pixels',
             'RGBD contract must preserve measured RGB and explicit missing depth')
    _indices(scan.get('indices'), scan.get('range_count'), 'scan')
    numeric = [scan.get(k) for k in ('angle_min_rad', 'angle_increment_rad', 'range_min_m', 'range_max_m')]
    _require(all(isinstance(v, (int, float)) and not isinstance(v, bool) and np.isfinite(v)
                 for v in numeric), 'explicit finite scan calibration required')
    _require(_text(scan.get('frame')) and numeric[1] != 0 and 0 <= numeric[2] < numeric[3]
             and scan.get('invalid_range_policy') == 'nonfinite_or_outside_sensor_limits_to_zero',
             'invalid measured scan contract')
    for name in ('base_from_camera', 'base_from_laser'):
        _matrix(extrinsics.get(name), (4, 4), name, rigid=True)
    return len(rgbd['row_indices']), len(rgbd['column_indices']), len(scan['indices'])


def _validate_models(models, tables, prefix, budget):
    _require(type(budget) is int and budget > 0, 'explicit positive action budget required')
    _require(isinstance(prefix, list), 'explicit prefix_actions required')
    first, second = models
    _require((first.poses, first.edges, first.anchor, first.prefix_nodes)
             == (second.poses, second.edges, second.anchor, second.prefix_nodes),
             'both public templates must share graph, anchor and prefix')
    _require(0 <= first.anchor < len(first.poses), 'anchor outside graph')
    direction = ((0, 1), (1, 0), (0, -1), (-1, 0))
    for table, model in zip(tables, models):
        _require(len(model.edges) == len(model.poses) == len(model.observed_masks),
                 'one edge row and exposure mask required per pose')
        _require(bool(model.floor_cells) and all(np.isfinite(v) and v > 0 for v in model.area.values()),
                 'nonempty floor cells and positive declared surface areas required')
        for node, (pose, row) in enumerate(zip(model.poses, table['edges'])):
            x, y, heading = pose
            _require(len(dict(row)) == len(row), 'ambiguous action edge')
            for action, destination in row:
                _require(type(destination) is int and 0 <= destination < len(model.poses),
                         'invalid graph destination')
                if action == 'forward':
                    dx, dy = direction[heading]
                    expected = (x + dx, y + dy, heading)
                else:
                    _require(action in ('left', 'right'), 'unknown discrete action')
                    expected = (x, y, (heading + (1 if action == 'right' else -1)) % 4)
                _require(model.poses[destination] == expected,
                         'graph edge changes the original V35 discrete action')
            model.terminal(model.observed_masks[node])
    cursor, nodes = first.anchor, [first.anchor]
    for action in prefix:
        _require(action in dict(first.edges[cursor]), 'prefix action outside public graph')
        cursor = dict(first.edges[cursor])[action]
        nodes.append(cursor)
    _require(cursor == first.anchor and len(prefix) <= budget
             and tuple(nodes) == first.prefix_nodes, 'prefix must match saved tables and restore anchor')


@dataclass(frozen=True)
class TaskBundle:
    models: tuple
    templates: PublicTemplatesV35
    prefix_actions: tuple
    total_budget: int
    receipt: dict
    calibration: dict
    binding: dict

    @property
    def execution_allowed(self):
        return self.receipt['execution_allowed']

    @property
    def observation_contract(self):
        return dict(binding=self.binding, calibration=self.calibration)

    def controller(self, mode='G'):
        return CPUFourModuleControllerV35(self.models, self.templates, self.prefix_actions,
                                         mode=mode, total_budget=self.total_budget)


def load_task(path):
    """Load frozen assets; perform no planning, collection or physical execution."""
    path = Path(path).resolve()
    raw = path.read_bytes()
    manifest = json.loads(raw)
    _require(manifest.get('schema') == SCHEMA and _text(manifest.get('task_id')),
             'explicit V35 task manifest required')
    scope, allowed = manifest.get('scope'), manifest.get('execution_allowed')
    _require(scope in ('physical_experiment', 'synthetic_unit_test') and type(allowed) is bool,
             'explicit task scope and execution_allowed required')
    _require(scope != 'synthetic_unit_test' or not allowed,
             'synthetic unit fixtures cannot allow physical execution')
    sources = manifest.get('physical_sources')
    _require(isinstance(sources, list) and (scope != 'physical_experiment' or bool(sources)),
             'physical_experiment requires hashed physical_sources')
    source_receipts = {}
    for source in sources:
        _require(isinstance(source, dict) and _text(source.get('id'))
                 and _text(source.get('kind')) and _text(source.get('description')),
                 'physical source requires id, kind and description')
        _require(source['id'] not in source_receipts, 'duplicate physical source id')
        _, receipt = _file_reference(source, path.parent)
        source_receipts[source['id']] = dict(receipt, kind=source['kind'], description=source['description'])
    hypotheses = manifest.get('hypotheses')
    _require(isinstance(hypotheses, list) and len(hypotheses) == 2,
             'two explicit public hypothesis descriptions required')
    for hypothesis in hypotheses:
        _require(isinstance(hypothesis, dict) and _text(hypothesis.get('description'))
                 and isinstance(hypothesis.get('source_ids'), list)
                 and set(hypothesis['source_ids']) <= set(source_receipts),
                 'hypothesis must cite declared sources')
        _require(scope != 'physical_experiment' or bool(hypothesis['source_ids']),
                 'physical hypotheses require measured or external model sources')
    calibration, binding = manifest.get('calibration'), manifest.get('binding')
    height, width, rays = _observation_contract(binding, calibration)
    calibration_sha = canonical_json_sha256(calibration)
    binding_sha = canonical_json_sha256(binding)
    geometry_path, geometry_receipt = _file_reference(manifest.get('geometry'), path.parent)
    prediction_path, prediction_receipt = _file_reference(manifest.get('sensor_predictions'), path.parent)
    tables = json.loads(geometry_path.read_bytes()).get('tables')
    _require(isinstance(tables, list) and len(tables) == 2, 'two original public model tables required')
    models = tuple(SavedPotentialModelV34(t, ['unused'] * len(t['poses'])) for t in tables)
    with np.load(prediction_path, allow_pickle=False) as predictions:
        _require(set(('poses', 'depth', 'ranges', 'calibration_sha256', 'binding_sha256')) <= set(predictions.files),
                 'paired predictions require poses, depth, ranges and contract hashes')
        _require(predictions['calibration_sha256'].shape == ()
                 and str(predictions['calibration_sha256'].item()) == calibration_sha,
                 'prediction calibration_sha256 differs')
        _require(predictions['binding_sha256'].shape == ()
                 and str(predictions['binding_sha256'].item()) == binding_sha,
                 'prediction binding_sha256 differs')
        poses = predictions['poses']
        _require(poses.dtype.kind in 'iu' and np.array_equal(poses, np.asarray(models[0].poses)),
                 'prediction node order differs from public graph')
        templates = PublicTemplatesV35(poses, predictions['depth'], predictions['ranges'],
                                      provenance=dict(scope=scope, manifest_sha256=hashlib.sha256(raw).hexdigest()))
    _require(templates.depth.shape[2:] == (height, width) and templates.ranges.shape[2:] == (rays,),
             'prediction shapes differ from selected sensor samples')
    prefix, budget = manifest.get('prefix_actions'), manifest.get('total_budget')
    _validate_models(models, tables, prefix, budget)
    receipt = dict(schema=SCHEMA, task_id=manifest['task_id'], scope=scope, execution_allowed=allowed,
                   manifest=dict(path=str(path), sha256=hashlib.sha256(raw).hexdigest()),
                   geometry=geometry_receipt, sensor_predictions=prediction_receipt,
                   physical_sources=source_receipts, hypotheses=hypotheses,
                   calibration_sha256=calibration_sha, binding_sha256=binding_sha,
                   physical_model_accuracy_verified=False, planning_rules_modified=False)
    return TaskBundle(models, templates, tuple(prefix), budget, receipt, calibration, binding)
