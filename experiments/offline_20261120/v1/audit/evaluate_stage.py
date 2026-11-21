#!/usr/bin/env python3
"""Evaluate saved measured meshes and predicted points; never execute a policy.

Inputs are a hash-addressed JSON stage manifest. Source geometry is loaded only
here, after replay/candidate execution. No registration or mesh repair is used.
"""
import argparse
import hashlib
import importlib.util
import json
import os
import sys
import time
from pathlib import Path

for _key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[_key] = '1'

import numpy as np
import open3d as o3d

B = Path('/home/ouc/nso_robot/analyses/dual_template_exploration_20261119')
REFERENCE_ROOT = B / 'audit/references'
TOLERANCE = .05


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def checked(entry, input_hashes):
    path = Path(entry['path']).resolve()
    actual = sha(path)
    if actual != entry['sha256']:
        raise ValueError(f'input hash differs: {path}')
    input_hashes[str(path)] = actual
    return path


def ray_scene(mesh):
    if not len(mesh.triangles):
        return None
    scene = o3d.t.geometry.RaycastingScene(nthreads=1)
    scene.add_triangles(o3d.t.geometry.TriangleMesh.from_legacy(mesh))
    return scene


def distance(scene, points):
    if scene is None:
        return np.full(len(points), np.inf)
    if not len(points):
        return np.empty(0)
    return scene.compute_distance(o3d.core.Tensor(np.asarray(points, np.float32)), nthreads=1).numpy()


def summarize_prediction(weights, source_distance, before, after=None):
    """Same-point four-way partition, with no claim of unique physical area."""
    weights = np.asarray(weights, float)
    source_distance = np.asarray(source_distance)
    before = np.asarray(before)
    if weights.ndim != 1 or source_distance.shape != weights.shape or before.shape != weights.shape:
        raise ValueError('prediction weights/distances are not aligned vectors')
    if np.any(~np.isfinite(weights)) or np.any(weights < 0):
        raise ValueError('weights must be finite and nonnegative')
    if np.any(np.isnan(source_distance)) or np.any(np.isnan(before)):
        raise ValueError('distance array contains NaN')
    active = weights > 0
    real = source_distance <= TOLERANCE
    was = before <= TOLERANCE
    masks = {'false_geometry': active & ~real, 'true_already_before': active & real & was}
    if after is None:
        masks['true_previously_unmatched'] = active & real & ~was
    else:
        after = np.asarray(after)
        if after.shape != weights.shape or np.any(np.isnan(after)):
            raise ValueError('after distance array invalid')
        now = after <= TOLERANCE
        masks.update(true_previously_unmatched_now_matched=active & real & ~was & now,
                     true_previously_unmatched_still_unrealized=active & real & ~was & ~now)
    assert np.array_equal(sum(x.astype(int) for x in masks.values()), active.astype(int))
    total = float(weights.sum())
    supported = float(weights[active & real].sum())
    available = float(weights[active & real & ~was].sum())
    rows = {key: {'predicted_points': int(mask.sum()),
                  'weighted_template_area_m2': float(weights[mask].sum()),
                  'fraction_of_active_weight': float(weights[mask].sum() / total) if total else None}
            for key, mask in masks.items()}
    result = {'active_predicted_points': int(active.sum()), 'total_weighted_template_area_m2': total,
              'source_supported_weighted_template_area_m2': supported,
              'source_supported_weight_fraction': supported / total if total else None,
              'source_supported_previously_unmatched_weight_m2': available,
              'partition': rows}
    if after is not None:
        realized = rows['true_previously_unmatched_now_matched']['weighted_template_area_m2']
        result.update(realization_fraction_of_source_supported_unmatched_weight=realized / available if available else None,
                      previously_correct_now_lost_weighted_template_area_m2=float(weights[active & real & was & ~now].sum()))
    return result


def actual_increment(before, after, source_area_m2, drawn_samples):
    """Paired source samples: gross new, lost and net must all be reported."""
    before, after = np.asarray(before), np.asarray(after)
    if before.shape != after.shape:
        raise ValueError('fixed reference changed between checkpoints')
    was, now = before <= TOLERANCE, after <= TOLERANCE
    unit = float(source_area_m2) / int(drawn_samples)
    new, lost = int((~was & now).sum()), int((was & ~now).sum())
    n = len(before)
    return {'fixed_visible_samples': n, 'before_matched_samples': int(was.sum()),
            'after_matched_samples': int(now.sum()), 'gross_new_samples': new, 'lost_samples': lost,
            'net_new_samples': new-lost, 'source_area_per_drawn_sample_m2': unit,
            'gross_new_correct_surface_area_estimate_m2': new*unit,
            'lost_correct_surface_area_estimate_m2': lost*unit,
            'net_correct_surface_area_estimate_m2': (new-lost)*unit,
            'recall_delta_percentage_points': 100.*(new-lost)/n if n else None}


class Evaluator:
    def __init__(self):
        spec = importlib.util.spec_from_file_location('frozen_B_surface_metrics', B/'audit/surface_metrics.py')
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        self.metrics = mod.evaluate_surface
        self.helper_paths = [B/'audit/surface_metrics.py', mod.OLD_EVALUATION/'surface_reference.py']
        self.refs = {}
        self.meshes = {}
        self.inputs = {}

    def reference(self, layout):
        if layout not in ('shelf_left', 'shelf_right'):
            raise ValueError('unknown frozen reference layout')
        if layout not in self.refs:
            base = REFERENCE_ROOT/layout
            receipt = json.loads((base/'RECEIPT.json').read_text())
            for name, key in [('fixed_reference.npz', 'reference_npz_sha256'), ('source_mesh.ply', 'source_mesh_sha256'), ('PROTOCOL.json', 'protocol_sha256')]:
                checked({'path': str(base/name), 'sha256': receipt[key]}, self.inputs)
            for path, expected in receipt['source_inputs_sha256'].items():
                checked({'path': path, 'sha256': expected}, self.inputs)
            self.inputs[str(base/'RECEIPT.json')] = sha(base/'RECEIPT.json')
            points = dict(np.load(base/'fixed_reference.npz', allow_pickle=False))
            source = o3d.io.read_triangle_mesh(str(base/'source_mesh.ply'))
            self.refs[layout] = {'receipt': receipt, 'points': points, 'mesh': source, 'ray': ray_scene(source)}
        return self.refs[layout]

    def mesh(self, entry, layout):
        path = checked(entry, self.inputs)
        key = (entry['sha256'], layout)
        if key not in self.meshes:
            ref = self.reference(layout)
            mesh = o3d.io.read_triangle_mesh(str(path))
            scene = ray_scene(mesh)
            self.meshes[key] = {'ray': scene, 'metrics': self.metrics(mesh, ref['mesh'], ref['points']),
                                'reference_distances': {name: distance(scene, ref['points'][name])
                                                        for name in ('all', 'open31', 'closed118')}}
        return self.meshes[key]

    def stage(self, row, default_layout=None):
        layout = row.get('layout', default_layout)
        ref = self.reference(layout)
        result = {key: row[key] for key in ('stage_id', 'variant', 'state_action', 'candidate_id', 'pair_key', 'outbound_actions') if key in row}
        result['layout'] = layout
        if 'execution_receipt' not in row:
            raise ValueError('execution receipt required')
        receipt_path = checked(row['execution_receipt'], self.inputs)
        execution = json.loads(receipt_path.read_text())
        result['execution_receipt'] = {'path': str(receipt_path), 'sha256': row['execution_receipt']['sha256'],
                                       'status': execution.get('status')}
        before = self.mesh(row['before_mesh'], layout)
        after = self.mesh(row['after_mesh'], layout) if row.get('after_mesh') else None
        result['before_native_TSDF_metrics'] = before['metrics']
        if after is not None:
            result['after_native_TSDF_metrics'] = after['metrics']
            result['actual_fixed_reference_increment'] = {}
            for name in ('all', 'open31', 'closed118'):
                meta = ref['receipt']['metadata'][name]
                result['actual_fixed_reference_increment'][name] = actual_increment(
                    before['reference_distances'][name], after['reference_distances'][name],
                    meta['sampled_source_area_m2'], meta['drawn_samples'])
        if row.get('predictions'):
            npz = checked(row['predictions'], self.inputs)
            arrays = dict(np.load(npz, allow_pickle=False))
            result['prediction_groups'] = {}
            for group in row['prediction_groups']:
                points = np.asarray(arrays[group['points_key']])
                weights = np.asarray(arrays[group['weights_key']])
                if points.shape != (len(weights), 3) or not np.isfinite(points).all():
                    raise ValueError('invalid prediction coordinates')
                source_d = distance(ref['ray'], points)
                before_d = distance(before['ray'], points)
                after_d = distance(after['ray'], points) if after is not None else None
                primary = summarize_prediction(weights, source_d, before_d, after_d)
                if len(points):
                    closest = ref['ray'].compute_closest_points(o3d.core.Tensor(points.astype(np.float32)), nthreads=1)['points'].numpy()
                    projected_before = distance(before['ray'], closest)
                    projected_after = distance(after['ray'], closest) if after is not None else None
                else:
                    projected_before = np.empty(0)
                    projected_after = np.empty(0) if after is not None else None
                result['prediction_groups'][group['name']] = {
                    'primary_predicted_point_match': primary,
                    'secondary_closest_source_point_match': summarize_prediction(weights, source_d, projected_before, projected_after),
                    'points_sha256': hashlib.sha256(points.tobytes()).hexdigest(),
                    'weights_sha256': hashlib.sha256(weights.tobytes()).hexdigest()}
        return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    manifest_hash = sha(args.manifest)
    manifest = json.loads(args.manifest.read_text())
    if manifest['mode'] not in ('fixed_replay', 'candidate_branch'):
        raise ValueError('unknown evaluation mode')
    evaluator = Evaluator()
    stages = [evaluator.stage(row, manifest.get('layout')) for row in manifest['stages']]
    for path, expected in evaluator.inputs.items():
        if sha(path) != expected:
            raise ValueError(f'input changed during evaluation: {path}')
    if sha(args.manifest) != manifest_hash:
        raise ValueError('stage manifest changed during evaluation')
    result = {'status': 'completed_frozen_reference_stage_evaluation', 'mode': manifest['mode'],
              'manifest_sha256': manifest_hash, 'stages': stages,
              'input_sha256': evaluator.inputs,
              'evaluator_sha256': {str(p): sha(p) for p in [Path(__file__), *evaluator.helper_paths]},
              'threshold_m': TOLERANCE, 'no_registration': True, 'no_actor_executed': True,
              'source_truth_loaded_only_by_evaluator': True, 'elapsed_s': time.monotonic()-started,
              'limits': ['Hypothesis weights are alternative template surface integrals, not unique physical area.',
                         'Actual surface increments use the same fixed independent source samples, including lost area.',
                         'Checkpoints/candidate branches from one trajectory are dependent; no independent-trial confidence intervals.',
                         'Source-supported prediction alone does not establish a semantic closed-loop efficiency benefit.']}
    (args.output/'result.json').write_text(json.dumps(result, indent=2, allow_nan=False)+'\n')
    print(json.dumps({'status': result['status'], 'stages': len(stages), 'output': str(args.output), 'elapsed_s': result['elapsed_s']}))


if __name__ == '__main__':
    main()
