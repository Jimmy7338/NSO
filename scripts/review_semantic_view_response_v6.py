#!/usr/bin/env python3
"""Read-only independent SVD refit and artifact review of the frozen V6 probe.

Uses recorded measured-support descriptors; does not regenerate their raw-frame
geometry, execute physical branches, tune parameters, or modify probe artifacts.
Only the explicitly requested new review JSON is written.
"""
import os
os.environ.setdefault('OMP_NUM_THREADS', '1')
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
import argparse
import hashlib
import json
from pathlib import Path
import zipfile

import numpy as np
from scipy.stats import spearmanr


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def independent_fit(histories, name):
    """SVD solve, independently of the production normal-equation solver."""
    xc = [h['x'][name] - h['x'][name].mean(axis=0) for h in histories]
    yc = [h['y'] - h['y'].mean(axis=0) for h in histories]
    scale = np.sqrt(np.mean(np.concatenate(xc) ** 2, axis=0))
    scale[scale < 1e-8] = 1.
    design = np.concatenate([x / scale / np.sqrt(len(x)) for x in xc])
    target = np.concatenate([y / np.sqrt(len(y)) for y in yc])
    u, singular, vt = np.linalg.svd(design, full_matrices=False)
    coefficient = (vt.T * (singular / (singular ** 2 + 1.))) @ u.T @ target
    common, conditional = design[:, :8], design[:, 8:]
    residual = conditional - common @ np.linalg.lstsq(common, conditional, rcond=None)[0]
    denominator = np.linalg.norm(conditional)
    stats = {
        'rows': len(design), 'histories': len(histories), 'columns': design.shape[1],
        'active_columns': int(np.count_nonzero(np.linalg.norm(design, axis=0) > 1e-10)),
        'design_rank': int(np.linalg.matrix_rank(design)),
        'common_rank': int(np.linalg.matrix_rank(common)),
        'ridge_effective_df_per_target': float(np.sum(singular ** 2 / (singular ** 2 + 1.))),
        'singular_values': singular.tolist(),
        'conditional_residual_norm_fraction': float(np.linalg.norm(residual) / denominator) if denominator else None,
        'loss': 'sum_history(mean_candidates(squared_error)) + 1 * squared_coefficient_norm',
        'coefficient': coefficient.tolist(), 'scale': scale.tolist(),
    }
    return coefficient, scale, stats


def review(run, output):
    consumed = {}

    def read(path):
        consumed[str(path.resolve())] = sha(path)
        return json.loads(path.read_text())

    protocol = read(run / 'protocol.json')
    assert protocol['status'] == 'complete' and protocol['fixed_alpha'] == 1.
    source = Path(protocol['source'])
    original_hashes = read(source / 'artifact_hashes.json')
    source_summary = read(source / 'summary.json')
    source_meta = read(source / 'metadata.json')
    source_verification = read(source / 'verification.json')
    assert source_meta['status'] == 'complete'
    assert source_verification['status'] == 'passed_full'
    for name in ('metadata.json', 'summary.json'):
        assert original_hashes[name] == sha(source / name)
    input_hashes = read(run / 'input_hashes.json')
    for filename, expected in input_hashes.items():
        path = Path(filename)
        assert sha(path) == expected, filename
        assert original_hashes[str(path.relative_to(source))] == expected, filename
    consumed[str((run / 'sources.zip').resolve())] = sha(run / 'sources.zip')
    with zipfile.ZipFile(run / 'sources.zip') as archive:
        for name, expected in protocol['source_sha256'].items():
            assert hashlib.sha256(archive.read(name)).hexdigest() == expected, name

    histories = []
    features_checked = 0
    for h in source_summary['histories']:
        assert h['seed'] in (751, 752)
        directory = run / h['fixture'] / f"history_{h['history_step']:04d}"
        feature_path = directory / 'features.npz'
        consumed[str(feature_path.resolve())] = sha(feature_path)
        with np.load(feature_path) as arrays:
            matrices = {name: arrays[name].copy() for name in arrays.files}
        audit = read(directory / 'feature_audit.json')
        environment = read(source / h['fixture'] / 'fixture.json')['environment']
        old = read(source / h['predictions'])
        routes = read((source / h['predictions']).parent / 'candidates.json')
        outcomes = {r['candidate_id']: r for r in read(source / h['outcomes'])}
        marked = any(sum(o['label_counts'].get(str(k), 0) for k in (2, 3)) > 0 for o in old['objects']['G'])
        marker_domain = environment['semantic_source'] == 'rgb_marker' and environment['appearance'] == 'marked'
        candidates = {c['candidate_id']: c for c in old['candidates']}
        rebuilt = {name: [] for name in matrices}
        for index, route in enumerate(routes):
            cost = route['cost']
            row = candidates[route['candidate_id']]['scores']['G']
            base = [row['radar_coverage_m2'] / cost, row['camera_coverage_m2'] / cost,
                    row['observed_quality_gain_per_point'] / cost, 1. / cost]
            descriptor = np.array(audit['object_descriptors'][index])
            for name in matrices:
                total, conditional = np.zeros(4), np.zeros(4)
                if name != 'N':
                    for j, f in enumerate(descriptor):
                        weight = .25 if name in ('O', 'S', 'X') and marker_domain and marked and not old['objects']['G'][j]['marker_supported'] else 1.
                        prior_name = 'G' if name in ('G', 'O') else name
                        p = old['objects'][prior_name][j]['shelf_probability']
                        total += weight * f
                        conditional += weight * (2. * p - 1.) * f
                rebuilt[name].append(np.r_[base, total, conditional])
        for name, matrix in matrices.items():
            np.testing.assert_array_equal(matrix, rebuilt[name])
            np.testing.assert_array_equal(matrix[:, :4], matrices['G'][:, :4])
            assert np.isfinite(matrix).all()
            features_checked += matrix.size
        np.testing.assert_array_equal(matrices['S'][:, :8], matrices['O'][:, :8])
        np.testing.assert_array_equal(matrices['S'][:, :8], matrices['X'][:, :8])
        np.testing.assert_array_equal(matrices['G'], matrices['M'])
        assert not matrices['N'][:, 4:].any()
        histories.append({'h': h, 'x': matrices, 'routes': routes, 'outcomes': outcomes,
                          'y': np.array([[outcomes[r['candidate_id']][k] for k in protocol['targets']] for r in routes]),
                          'scores': read(directory / 'predictions.json')})

    saved_models = read(run / 'models.json')
    saved_choices = read(run / 'choices.json')
    saved_summary = read(run / 'summary.json')
    model_lookup = {(m['held_out_seed'], m['name']): m for m in saved_models}
    choice_lookup = {(c['fixture'], c['history_step'], c['scorer']): c for c in saved_choices}
    fit_reports, choice_reports = [], []
    max_coefficient_error = max_prediction_error = 0.
    for held in (751, 752):
        train = [h for h in histories if h['h']['seed'] != held]
        test = [h for h in histories if h['h']['seed'] == held]
        assert {h['h']['seed'] for h in train} == {1503 - held}
        fitted = {}
        for name in ('G', 'O', 'S', 'N'):
            coefficient, scale, stats = independent_fit(train, name)
            saved = model_lookup[held, name]
            assert saved['train_seeds'] == [1503 - held] and saved['alpha'] == 1.
            np.testing.assert_allclose(scale, saved['scale'], rtol=1e-13, atol=1e-14)
            np.testing.assert_allclose(coefficient, saved['coefficient'], rtol=1e-10, atol=1e-12)
            max_coefficient_error = max(max_coefficient_error, float(np.max(np.abs(coefficient - saved['coefficient']))))
            stats.update(held_out_seed=held, train_seed=1503-held, model=name,
                         train_histories=[f"{h['h']['fixture']}/{h['h']['history_step']}" for h in train])
            fit_reports.append(stats)
            fitted[name] = (coefficient, scale)
        for h in test:
            report = {**h['h'], 'choices': {}, 'S_vs_comparators': {}}
            indices = {}
            for name in ('G', 'O', 'S', 'X', 'M', 'N', 'G_shared', 'O_shared'):
                model_name = 'S' if name in ('X', 'G_shared', 'O_shared') else 'G' if name == 'M' else name
                coefficient, scale = fitted[model_name]
                x = h['x'][name.split('_')[0]]
                predicted = (x - x.mean(axis=0)) / scale @ coefficient
                archived = np.array(h['scores'][name])
                np.testing.assert_allclose(predicted, archived, rtol=1e-10, atol=1e-12)
                max_prediction_error = max(max_prediction_error, float(np.max(np.abs(predicted - archived))))
                # Preserve exact archived arithmetic in tie breaking; refit must
                # agree on all strict winners outside numerical tolerance.
                ranks = sorted(range(len(archived)), key=lambda i: (-archived[i, 0], h['routes'][i]['cost'], i))
                j = ranks[0]
                assert predicted[j, 0] >= predicted[:, 0].max() - 1e-12
                cid = h['routes'][j]['candidate_id']
                saved = choice_lookup[h['h']['fixture'], h['h']['history_step'], name]
                assert saved['chosen'] == cid and saved['train_seed'] == 1503 - held
                outcome = h['outcomes'][cid]
                for key in ('area_per_action', 'f1_gain_05cm', 'coverage_gain_m2', 'new_area_m2'):
                    assert saved[key] == outcome[key]
                actual = h['y'][:, 0]
                pairs = [(a, b) for a in range(len(actual)) for b in range(a + 1, len(actual)) if actual[a] != actual[b]]
                accuracy = np.mean([float(np.sign(archived[a, 0] - archived[b, 0]) == np.sign(actual[a] - actual[b]))
                                    if archived[a, 0] != archived[b, 0] else .5 for a, b in pairs])
                np.testing.assert_allclose(accuracy, saved['pairwise_accuracy'], atol=1e-15)
                np.testing.assert_allclose(spearmanr(archived[:, 0], actual).statistic, saved['spearman'], atol=1e-15)
                np.testing.assert_allclose(actual.max() - actual[j], saved['area_rate_regret'], atol=1e-15)
                indices[name] = j
                report['choices'][name] = {'candidate': cid, 'cost': h['routes'][j]['cost'],
                    'area_rate': float(actual[j]), 'f1_gain': float(h['y'][j, 1]),
                    'predicted_area_margin': float(archived[j, 0] - archived[ranks[1], 0]),
                    'relative_area_rmse': float(np.sqrt(np.mean((archived[:, 0] - (actual - actual.mean())) ** 2))),
                    'relative_f1_rmse': float(np.sqrt(np.mean((archived[:, 1] - (h['y'][:, 1] - h['y'][:, 1].mean())) ** 2)))}
            coefficient, scale = fitted['S']
            for other in ('G', 'O', 'N', 'X', 'G_shared', 'O_shared'):
                j, k = indices['S'], indices[other]
                term = (h['x']['S'][j] - h['x']['S'][k]) / scale * coefficient[:, 0]
                report['S_vs_comparators'][other] = {
                    'same_choice': j == k, 'actual_area_rate_difference': float(h['y'][j, 0] - h['y'][k, 0]),
                    'actual_f1_difference': float(h['y'][j, 1] - h['y'][k, 1]),
                    'frozen_S_prediction_difference_blocks': {
                        'common_route': float(term[:4].sum()), 'object_support': float(term[4:8].sum()),
                        'class_conditioned': float(term[8:].sum())},
                    'warning': 'S-score decomposition at two candidates, not causal attribution of separately retrained models'}
            class_scores = (h['x']['S'][:, 8:] - h['x']['S'][:, 8:].mean(axis=0)) / scale[8:] @ coefficient[8:, 0]
            report['S_class_score_range'] = float(np.ptp(class_scores))
            report['S_total_score_range'] = float(np.ptp(np.array(h['scores']['S'])[:, 0]))
            report['S_vs_X_max_prediction_change'] = float(np.max(np.abs(np.array(h['scores']['S'])[:, 0] - np.array(h['scores']['X'])[:, 0])))
            choice_reports.append(report)
    for name, metrics in saved_summary['metrics'].items():
        selected = [c for c in saved_choices if c['scorer'] == name]
        for key in ('area_per_action', 'f1_gain_05cm', 'coverage_gain_m2', 'area_rate_regret'):
            np.testing.assert_allclose(np.mean([c[key] for c in selected]), metrics[key], atol=1e-15)
        for seed in (751, 752):
            for key, value in metrics['per_seed'][str(seed)].items():
                np.testing.assert_allclose(np.mean([c[key] for c in selected if c['seed'] == seed]), value, atol=1e-15)
    for path, expected in consumed.items():
        assert sha(Path(path)) == expected, path
    result = {
        'status': 'passed_bounded_independent_review', 'run': str(run.resolve()),
        'review_script_sha256': sha(Path(__file__)), 'consumed_artifact_hashes': consumed,
        'source_archive_members_verified': len(protocol['source_sha256']),
        'original_input_hashes_verified': len(input_hashes), 'feature_scalars_reassembled': features_checked,
        'fold_fits_independently_recomputed': len(fit_reports), 'choice_rows_verified': len(saved_choices),
        'max_coefficient_absolute_error': max_coefficient_error, 'max_prediction_absolute_error': max_prediction_error,
        'raw_descriptor_geometry_recomputed': False, 'new_physical_branches': 0,
        'held_out_targets_supplied_to_refit': False, 'fitting_method': 'independent NumPy SVD, alpha fixed at 1',
        'fit_diagnostics': fit_reports, 'history_diagnostics': choice_reports,
        'metrics': saved_summary['metrics'], 'independent_confirmation': False, 'gates_passed': False,
        'limits': [
            'Features reconstructed from stored measured-support descriptors plus original prefix-only predictions; raw RGB-D descriptor extraction statically reviewed, not rerun.',
            'Training refit uses only the other geometry seed; original execution loaded all labels before fits, so this is code/data-flow isolation, not prospective process blinding.',
            'Two already-viewed geometry seeds cannot establish out-of-distribution calibration or statistical significance.',
            'Response mixing still inherits uncalibrated V3 shelf_probability, even though new directional features do not use completion box dimensions.',
            'Support/FOV descriptors omit scene occlusion and are not unique physical exterior area.',
            'Equal alpha and feature count do not imply equal effective capacity; full-route thesis superiority is untested.',
        ],
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open('x') as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
        f.write('\n')
    print(json.dumps({k: result[k] for k in ('status', 'feature_scalars_reassembled', 'fold_fits_independently_recomputed',
        'choice_rows_verified', 'max_coefficient_absolute_error', 'max_prediction_absolute_error')}, ensure_ascii=False))
    for row in fit_reports:
        print(row['held_out_seed'], row['model'], 'rank', row['design_rank'], 'df', row['ridge_effective_df_per_target'],
              'conditional residual', row['conditional_residual_norm_fraction'])


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--run', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    review(args.run.resolve(), args.output.resolve())
