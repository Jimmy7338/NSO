#!/usr/bin/env python3
"""Seal sensor-only paired prefixes, shared candidates and V7 feature inputs.

No future branch is executed and no reconstruction evaluator is constructed.
Training parents have frozen legacy predictions only; calibration parents also
require a previously fitted and frozen V7 model bank.
"""
import os
os.environ.setdefault('OMP_NUM_THREADS', '1')
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import sys
import time
import zipfile
import numpy as np
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from env.virtual3d_response_v7 import (create_response_world, collect_response_prefix,
    get_response_context, RESPONSE_FAMILIES, DEFAULT_CONTEXT_MANIFEST)
from nso.response_candidates_v7 import candidate_routes
from nso.response_features_v7 import response_features, ResponseFeatureConfig
from nso.conditional_response_v7 import HistoryFeatures, ResponseModelBank
from nso.counterfactual_view_scoring import score_routes, CounterfactualScoreConfig
from scripts.eval_counterfactual_views import remap, grid_config, write_json
from utils.cpu_protocol import file_hash


def read(path):
    return json.loads(path.read_text())


def array_hash(values):
    digest = hashlib.sha256()
    for name, value in values:
        value = np.ascontiguousarray(value)
        digest.update(json.dumps([name, value.dtype.str, value.shape]).encode())
        digest.update(value.tobytes())
    return digest.hexdigest()


def records_from_prefix(raw, shape, config):
    rows = []; moves = 0
    directions = np.array([[0, 1], [1, 0], [0, -1], [-1, 0]])
    for r in raw:
        pose = r['frame'].world_from_camera
        position = [shape[0] - 1 - int(np.floor(pose[1, 3] / config.resolution_m)),
                    int(np.floor(pose[0, 3] / config.resolution_m))]
        heading = int(np.argmax(directions @ pose[:2, 2]))
        moves += r['action'] == 'forward'
        rows.append({'step': r['step'], 'position': position, 'heading': heading,
                     'collision': bool(r['collision']), 'collisions': 0, 'moves': int(moves),
                     'action': r['action'] or 'reset', 'done': bool(r['done'])})
    return rows


def prepare(context_id, output, bank_path=None):
    context = get_response_context(context_id)
    if context.role == 'calibration' and bank_path is None:
        raise ValueError('calibration prefixes require a previously frozen trained model bank')
    if output.exists():
        raise FileExistsError(output)
    output.mkdir(parents=True)
    started = time.time()
    # Freeze a concrete inventory; unrelated future scripts are not dependencies.
    names = sorted({str(p.relative_to(ROOT)) for folder in ('env', 'nso', 'utils')
                    for p in (ROOT / folder).glob('*.py')} | {
        'scripts/prepare_response_v7.py', 'scripts/eval_counterfactual_views.py',
        str(DEFAULT_CONTEXT_MANIFEST.relative_to(ROOT)), 'requirements-3d.lock.txt'})
    hashes = {name: file_hash(ROOT / name) for name in names}
    with zipfile.ZipFile(output / 'sources.zip', 'w', zipfile.ZIP_DEFLATED) as archive:
        for name in names:
            archive.write(ROOT / name, name)
    bank = None if bank_path is None else ResponseModelBank.from_dict(read(bank_path))
    metadata = {'status': 'running', 'context': asdict(context), 'source_sha256': hashes,
                'scope': 'paired prefix and candidate construction, no future outcomes',
                'bank_sha256': None if bank_path is None else file_hash(bank_path),
                'new_physical_branches': 0, 'ground_truth_used_by_planner': False,
                'feature_config': asdict(ResponseFeatureConfig())}
    write_json(output / 'metadata.json', metadata)
    pair = []; summaries = []
    for family in RESPONSE_FAMILIES:
        folder = output / family; folder.mkdir()
        prefix = folder / 'prefix'; prefix.mkdir(); (prefix / 'frames').mkdir(); (prefix / 'scans').mkdir()
        world = create_response_world(context_id, family)
        c = world.config
        raw = collect_response_prefix(world)
        frames = [r['frame'] for r in raw]; scans = [r['scan'] for r in raw]
        rows = records_from_prefix(raw, world.shape, c)
        write_json(folder / 'fixture.json', {'context': asdict(context), 'family': family, 'config': asdict(c)})
        for i, (frame, scan) in enumerate(zip(frames, scans)):
            frame.save(prefix / 'frames' / f'{i:04d}.npz'); scan.save(prefix / 'scans' / f'{i:04d}.npz')
        write_json(prefix / 'records.json', rows)
        mappers = {mode: remap(frames, scans, rows, c, world.shape, mode) for mode in ('aligned', 'shuffled', 'absent')}
        mapper = mappers['aligned']; last = rows[-1]
        obs = mapper.observation(tuple(last['position']), last['heading'], last['step'], False)
        routes, candidate_audit = candidate_routes(mapper, obs)
        write_json(folder / 'candidates.json', routes)
        write_json(folder / 'candidate_audit.json', candidate_audit)
        features, feature_audit = response_features(mapper, routes)
        # Check the actual sensor interpretation routes, not only sign algebra.
        actual_x, _ = response_features(mappers['shuffled'], routes)
        actual_m, _ = response_features(mappers['absent'], routes)
        np.testing.assert_array_equal(features['X'], actual_x['S'])
        np.testing.assert_array_equal(features['M'], actual_m['G'])
        np.testing.assert_array_equal(features['M'], actual_m['M'])
        np.savez_compressed(folder / 'features.npz', **features)
        write_json(folder / 'feature_audit.json', feature_audit)
        legacy = score_routes(mappers, routes, CounterfactualScoreConfig(grid_config(c), c))
        write_json(folder / 'legacy_predictions.json', legacy)
        if bank is not None:
            h = HistoryFeatures(f'{context_id}/{family}', context_id, features)
            prediction = bank.predict(h)
            write_json(folder / 'predictions.json', {name: values.tolist() for name, values in prediction.items()})
            write_json(folder / 'choices.json', bank.select(h, [r['cost'] for r in routes]))
        q = mapper.quality_evidence(max_points=10**9)
        mesh = mapper.mesh()
        geom = array_hash([('belief', mapper.belief), ('camera_seen', mapper.camera_seen),
            ('vertices', np.asarray(mesh.vertices)), ('triangles', np.asarray(mesh.triangles)),
            ('normals', np.asarray(mesh.vertex_normals))] + ([] if q is None else [(k, q[k]) for k in sorted(q) if k != 'label']))
        cue_roles = [r['role'] for r in candidate_audit['available_roles'] if r.get('patch') is not None
                     and feature_audit['patches'][r['patch']]['marked_points'] > 0]
        summary = {'family': family, 'prefix_frames': len(frames), 'paid_prefix_actions': len(frames)-1,
                   'candidates': len(routes), 'costs': [r['cost'] for r in routes],
                   'marked_patches': sum(p['marked_points'] > 0 for p in feature_audit['patches']),
                   'marked_direction_roles': cue_roles, 'nonlabel_geometry_sha256': geom,
                   'actual_swapped_and_absent_features_verified': True}
        write_json(folder / 'preparation.json', summary); summaries.append(summary)
        pair.append((frames, scans, rows, routes, features, geom))
        print('prepared', context_id, family, summary, flush=True)
        del mappers, mapper
    first, second = pair
    for fa, fb, sa, sb in zip(first[0], second[0], first[1], second[1]):
        for key in ('depth_m', 'intrinsic', 'world_from_camera', 'timestamp_s'):
            np.testing.assert_array_equal(getattr(fa, key), getattr(fb, key))
        for key in sa.__dataclass_fields__:
            np.testing.assert_array_equal(getattr(sa, key), getattr(sb, key))
        nonmarker = (fa.semantic == 0) & (fb.semantic == 0)
        np.testing.assert_array_equal(fa.color_rgb[nonmarker], fb.color_rgb[nonmarker])
        np.testing.assert_array_equal(fa.semantic > 0, fb.semantic > 0)
    assert first[2] == second[2] and first[3] == second[3]
    assert first[5] == second[5], 'paired measured geometry differs'
    for name in ('G', 'O', 'N', 'M', 'G_capacity'):
        np.testing.assert_array_equal(first[4][name], second[4][name])
    np.testing.assert_array_equal(first[4]['S'][:, :8], second[4]['S'][:, :8])
    np.testing.assert_array_equal(first[4]['S'][:, 8:], -second[4]['S'][:, 8:])
    sufficient = all(s['candidates'] == 6 and len(set(s['marked_direction_roles'])) >= 2 for s in summaries)
    write_json(output / 'pair_audit.json', {'geometry_prefix_exact': True, 'shared_candidates_exact': True,
              'nonsemantic_features_exact': True, 'only_S_conditional_sign_differs': True,
              'candidate_structure_gate_passed': sufficient, 'families': summaries,
              'candidate_outcomes_computed': False})
    assert all(file_hash(ROOT / name) == sha for name, sha in hashes.items()), 'source changed during preparation'
    metadata.update(status='complete', candidate_structure_gate_passed=sufficient, elapsed_s=time.time()-started)
    write_json(output / 'metadata.json', metadata)
    write_json(output / 'artifact_hashes.json', {str(p.relative_to(output)): file_hash(p)
                                                for p in output.rglob('*') if p.is_file()})
    print('pair prepared', context_id, 'structure_gate', sufficient, flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--context', required=True); p.add_argument('--output', type=Path, required=True)
    p.add_argument('--bank', type=Path)
    args = p.parse_args()
    try:
        prepare(args.context, args.output, args.bank)
    except Exception as exc:
        metadata_path = args.output / 'metadata.json'
        if metadata_path.exists():
            metadata = read(metadata_path)
            if metadata.get('status') == 'running':
                metadata.update(status='failed_preparation', error=f'{type(exc).__name__}: {exc}',
                                new_physical_branches=0)
                write_json(metadata_path, metadata)
        raise
