"""Remeasure correct-free coverage, native-mesh F1, and J without running an actor."""

import hashlib
import json
import math
import os
import sys
from pathlib import Path

for variable in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ.setdefault(variable, '1')

import numpy as np
import open3d as o3d

ARCHIVE = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ARCHIVE / 'semantic_scene_redesign_20261119/exploration_evaluation'))
from surface_reference import evaluate_surface


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    with Path(path).open('rb') as file:
        return hashlib.file_digest(file, 'sha256').hexdigest()


def verified_file(ref):
    path = Path(ref['path'])
    if sha(path) != ref['sha256']:
        raise ValueError(f'Frozen evidence changed: {path}')
    return path


def verify_attempt(attempt):
    manifest_path = attempt / 'artifact_manifest.json'
    manifest = read(manifest_path)
    for name, checksum in manifest['files_sha256'].items():
        verified_file({'path': attempt / name, 'sha256': checksum})
    return manifest


def count_free(belief, reachable):
    if belief.shape != reachable.shape or reachable.dtype != np.bool_ or not reachable.any():
        raise ValueError('Belief and reachable mask do not describe the same nonempty grid')
    if not np.isin(belief, (-1, 0, 1)).all():
        raise ValueError('Expected a ternary occupancy grid')
    correct = int(np.count_nonzero((belief == 0) & reachable))
    total = int(reachable.sum())
    return correct, total


def measure(attempt, reference):
    """Read an immutable terminal attempt and evaluator-only reference bundle."""
    attempt = Path(attempt)
    manifest = verify_attempt(attempt)
    required = {'steps.json', 'result.json', 'final_belief.npz', 'final_tsdf_mesh.ply'}
    if not required <= set(manifest['files_sha256']):
        raise ValueError(f'Missing terminal artifacts in {attempt}')
    r = read(attempt / 'result.json')
    steps = read(attempt / 'steps.json')
    with np.load(verified_file(reference['reachable']), allow_pickle=False) as data:
        reachable = data['reachable']
    with np.load(attempt / 'final_belief.npz', allow_pickle=False) as data:
        free, total = count_free(data['belief'], reachable)
        terminal_pose = [*map(int, data['position']), int(data['heading'])]
    budget = r['paid_action_budget']
    if r['paid_actions'] != len(steps) or not 0 <= len(steps) <= budget:
        raise ValueError(f'Step count does not match the declared cap: {attempt}')
    assert [step['step'] for step in steps] == list(range(1, len(steps) + 1))
    assert not steps or terminal_pose == [*steps[-1]['position'], steps[-1]['heading']]
    assert math.isclose(free/total, r.get('free_correct_fraction', free/total), abs_tol=1e-12, rel_tol=0)
    assert r['collisions'] == sum(bool(step['collision']) for step in steps)
    termination = r['termination']
    returned = terminal_pose == reference['anchor']
    assert termination['returned_to_anchor'] == returned
    assert r['initial_packet_sha256'] == reference['initial_packet_sha256']

    source = o3d.io.read_triangle_mesh(str(verified_file(reference['source_mesh'])))
    native = o3d.io.read_triangle_mesh(str(attempt / 'final_tsdf_mesh.ply'))
    with np.load(verified_file(reference['fixed_samples']), allow_pickle=False) as data:
        points = np.asarray(data['all'], dtype=np.float32)
        target_groups = {'open31': ('open31', 'shelf'), 'closed115_fronts': ('closed118', 'cabinet')}
        groups = {}
        for label, aliases in target_groups.items():
            key = next((alias for alias in aliases if alias in data), None)
            groups[label] = np.asarray(data[key], dtype=np.float32) if key else np.empty((0, 3), np.float32)
    if len(source.triangles) == 0 or len(points) == 0 or points.ndim != 2 or points.shape[1] != 3:
        raise ValueError('Empty or invalid whole-scene reference')
    surface = evaluate_surface(native, source, {'all': points, **groups}, thresholds=(.05,))
    f1 = surface['f1_05cm']
    return dict(path=str(attempt), status=r['status'], C_free=free/total, F1_scene_05cm=f1,
                J=free/total*f1, p_scene_05cm=surface['precision_05cm'], r_scene_05cm=surface['recall_05cm'],
                correct_free_cells=free, reachable_cells=total,
                actions=len(steps), budget=budget, returned=returned, collisions=r['collisions'],
                stop_reason=termination['reason'], failed=bool(termination.get('failed', False)),
                final_pose=terminal_pose, initial_packet_sha256=r['initial_packet_sha256'],
                source_hashes={'manifest': sha(attempt / 'artifact_manifest.json'),
                               'reference': reference,
                               'metric_source': sha(Path(sys.modules['surface_reference'].__file__))})
