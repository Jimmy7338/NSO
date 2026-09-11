#!/usr/bin/env python3
"""Apply already frozen response models to expanded development route pools.

No fitting or parameter search is performed. New candidate outcomes are read
only after every feature matrix, prediction and choice has been written.
"""
import os
os.environ.setdefault('OMP_NUM_THREADS', '1')
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
import argparse
import copy
import json
from pathlib import Path
import sys
import time
import zipfile
import numpy as np
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from env.virtual3d_inspection_v4 import InspectionConfigV4
from nso.counterfactual_view_scoring import _mapper_snapshot
from nso.semantic_view_response_v6 import response_features, RelativeResponseRidge
from scripts.eval_counterfactual_views import remap, write_json
from utils.cpu_protocol import file_hash
from utils.rgbd_contract import RGBDFrame, PlanarScan


def read(path):
    return json.loads(path.read_text())


def run(source, supplement, probe, output):
    for folder in (source, supplement):
        assert read(folder / 'verification.json')['status'] == 'passed_full'
    protocol = read(probe / 'protocol.json')
    assert protocol['status'] == 'complete' and protocol['fixed_alpha'] == 1.
    output.mkdir(parents=True, exist_ok=False)
    started = time.time()
    inputs = {}
    manifests = {folder: read(folder / 'artifact_hashes.json') for folder in (source, supplement)}

    def checked(path, folder=None):
        sha = file_hash(path)
        if folder is not None:
            assert manifests[folder][str(path.relative_to(folder))] == sha
        inputs[str(path.resolve())] = sha
        return path

    models_raw = read(checked(probe / 'models.json'))
    models = {(r['held_out_seed'], r['name']): RelativeResponseRidge(
        np.array(r['coefficient']), np.array(r['scale']), r['alpha']) for r in models_raw}
    assert all(r['train_seeds'] == [752 if r['held_out_seed'] == 751 else 751] for r in models_raw)
    names = sorted(set(protocol['source_sha256']) | {'scripts/transfer_frozen_response_v6.py'})
    hashes = {name: file_hash(ROOT / name) for name in names}
    for name, sha in protocol['source_sha256'].items():
        assert hashes[name] == sha, name
    with zipfile.ZipFile(output / 'sources.zip', 'w', zipfile.ZIP_DEFLATED) as archive:
        for name in names:
            archive.write(ROOT / name, name)
    metadata = {'status': 'running', 'scope': 'frozen-model transfer to extra routes in already viewed development worlds',
                'source': str(source.resolve()), 'supplement': str(supplement.resolve()),
                'models_source': str(probe.resolve()), 'source_sha256': hashes,
                'refitted': False, 'new_physical_branches': 0, 'independent_confirmation': False,
                'predictions_relative_to_union_pool': True, 'original_gates_changed': False}
    write_json(output / 'metadata.json', metadata)
    pending = []
    for h in read(checked(source / 'summary.json', source))['histories']:
        fixture = source / h['fixture']
        index = h['history_step']
        history_name = f'history_{index:04d}'
        original = fixture / history_name
        extra = supplement / h['fixture'] / history_name
        routes_old = read(checked(original / 'candidates.json', source))
        routes_extra = read(checked(extra / 'candidates.json', supplement))
        routes = routes_old + routes_extra
        assert [r['candidate_id'] for r in routes] == list(range(len(routes)))
        assert len({tuple(r['pose']) for r in routes}) == len(routes)
        prediction_old = read(checked(original / 'predictions.json', source))
        prediction_extra = read(checked(extra / 'predictions.json', supplement))
        assert prediction_old['objects'] == prediction_extra['objects']
        assert prediction_old['invariants']['mapper_hashes'] == prediction_extra['invariants']['mapper_hashes']
        prediction = copy.deepcopy(prediction_old)
        prediction['candidates'] += prediction_extra['candidates']
        c = InspectionConfigV4(**read(checked(fixture / 'fixture.json', source))['environment'])
        rows = read(checked(fixture / 'prefix/records.json', source))[:index + 1]
        frames = [RGBDFrame.load(checked(fixture / 'prefix/frames' / f'{i:04d}.npz', source)) for i in range(index + 1)]
        scans = [PlanarScan.load(checked(fixture / 'prefix/scans' / f'{i:04d}.npz', source)) for i in range(index + 1)]
        mapper = remap(frames, scans, rows, c, (round(c.height_m / c.resolution_m), round(c.width_m / c.resolution_m)))
        assert _mapper_snapshot(mapper)['hashes'] == prediction['invariants']['mapper_hashes']['aligned']
        features, audit = response_features(mapper, routes, prediction)
        old_features = np.load(checked(probe / h['fixture'] / history_name / 'features.npz'))
        for name in features:
            np.testing.assert_array_equal(features[name][:len(routes_old)], old_features[name])
        old_predictions = read(checked(probe / h['fixture'] / history_name / 'predictions.json'))
        target = output / h['fixture'] / history_name
        target.mkdir(parents=True)
        np.savez_compressed(target / 'features.npz', **features)
        write_json(target / 'feature_audit.json', audit)
        predictions, choices = {}, {}
        for name in ('G', 'O', 'S', 'X', 'M', 'N', 'G_shared', 'O_shared'):
            model_name = 'S' if name in ('X', 'G_shared', 'O_shared') else 'G' if name == 'M' else name
            model = models[(h['seed'], model_name)]
            feature_name = name.split('_')[0]
            scores = model.predict(features[feature_name])
            original_scores = model.predict(features[feature_name][:len(routes_old)])
            np.testing.assert_allclose(original_scores, old_predictions[name], rtol=0, atol=1e-12)
            # A larger candidate mean shifts all old scores equally, preserving their order.
            delta = scores[:len(routes_old)] - original_scores
            np.testing.assert_allclose(delta, np.broadcast_to(delta[0], delta.shape), rtol=0, atol=1e-12)
            chosen = min(range(len(routes)), key=lambda i: (-scores[i, 0], routes[i]['cost'], i))
            predictions[name] = scores.tolist()
            choices[name] = routes[chosen]['candidate_id']
        write_json(target / 'predictions.json', predictions)
        write_json(target / 'choices.json', choices)
        pending.append((h, original, extra, choices))
        print('sealed', h['fixture'], index, len(routes), flush=True)
        del mapper
    sealed = {str(p.relative_to(output)): file_hash(p) for p in output.rglob('*') if p.is_file()}
    write_json(output / 'prediction_seal.json', sealed)
    # No outcome table from either route pool has been opened before this point.
    choices_all = []
    for h, original, extra, choices in pending:
        rows = read(checked(original / 'outcomes.json', source)) + read(checked(extra / 'outcomes.json', supplement))
        outcomes = {r['candidate_id']: r for r in rows}
        assert len(outcomes) == len(rows)
        for name, cid in choices.items():
            result = outcomes[cid]
            choices_all.append({**{k: h[k] for k in ('fixture', 'history_step', 'seed', 'depth_sigma_m')},
                'scorer': name, 'chosen': cid, 'new_candidate_selected': cid >= 12,
                **{key: result[key] for key in ('area_per_action', 'new_area_m2', 'f1_gain_05cm', 'coverage_gain_m2')},
                'area_rate_regret': max(r['area_per_action'] for r in rows) - result['area_per_action']})
    averages = {}
    for name in ('G', 'O', 'S', 'X', 'M', 'N', 'G_shared', 'O_shared'):
        rows = [r for r in choices_all if r['scorer'] == name]
        averages[name] = {key: float(np.mean([r[key] for r in rows])) for key in (
            'area_per_action', 'new_area_m2', 'f1_gain_05cm', 'coverage_gain_m2', 'area_rate_regret')}
        averages[name]['new_choices'] = sum(r['new_candidate_selected'] for r in rows)
    assert all(file_hash(ROOT / name) == sha for name, sha in hashes.items())
    assert all(file_hash(Path(path)) == sha for path, sha in inputs.items())
    assert all(file_hash(output / path) == sha for path, sha in sealed.items())
    write_json(output / 'choices.json', choices_all)
    write_json(output / 'summary.json', {'scorer_averages': averages, 'histories': len(pending), 'candidate_routes': 88})
    write_json(output / 'input_hashes.json', inputs)
    metadata.update(status='complete', elapsed_s=time.time() - started, input_hashes_checked=len(inputs))
    # Initial metadata remains covered by the prediction seal; completion lives separately.
    write_json(output / 'completion.json', metadata)
    print(json.dumps(averages, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('source', 'supplement', 'probe', 'output'):
        parser.add_argument('--' + name, type=Path, required=True)
    args = parser.parse_args()
    run(args.source, args.supplement, args.probe, args.output)
