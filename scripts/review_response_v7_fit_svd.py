#!/usr/bin/env python3
"""Independent NumPy SVD review of one sealed eight-parent V7 T fit.

Does not import the fitted model, fitting entrypoint, simulator, or mapper.
Reads T targets only; never optimizes a new grid or opens calibration outcomes.
"""
import os
for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[key] = '1'
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import time
import traceback
import numpy as np


def read(path): return json.loads(Path(path).read_text())
def write(path, value): Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n')
def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def centered_design(keys, x, y):
    return (np.vstack([(x[k] - x[k].mean(0)) / np.sqrt(x[k].shape[0]) for k in keys]),
            np.vstack([(y[k] - y[k].mean(0)) / np.sqrt(y[k].shape[0]) for k in keys]))


def independent_svd_fit(keys, x, y, alpha):
    matrix, target = centered_design(keys, x, y)
    left, singular, right_t = np.linalg.svd(matrix, full_matrices=False)
    projected = left.T.dot(target)
    beta = right_t.T.dot(projected * (singular / (singular * singular + alpha))[:, None])
    residual = matrix.T.dot(matrix.dot(beta) - target) + alpha * beta
    report = {'rows': len(matrix), 'histories': len(keys), 'columns': matrix.shape[1],
              'design_rank': int(np.linalg.matrix_rank(matrix)), 'singular_values': singular.tolist(),
              'effective_df_per_target': float(np.sum(singular * singular / (singular * singular + alpha))),
              'active_columns': int(np.count_nonzero(np.any(matrix != 0, axis=0))),
              'normal_equation_max_residual': float(np.abs(residual).max())}
    return beta, report


def run(args):
    output = args.output.resolve(); output.mkdir(parents=True, exist_ok=False)
    fit = args.fit.resolve(); runs = args.runs.resolve(); started = time.monotonic()
    hashes, differences = {}, {}

    def use(path, expected=None):
        path = Path(path).resolve(); digest = sha(path)
        if expected is not None: assert digest == expected, f'changed input: {path}'
        assert str(path) not in hashes or hashes[str(path)] == digest
        hashes[str(path)] = digest
        return path

    def compare(name, actual, expected):
        a, b = np.asarray(actual), np.asarray(expected)
        assert a.shape == b.shape, name
        diff = float(np.max(np.abs(a - b))) if a.size else 0.
        differences[name] = diff
        np.testing.assert_allclose(a, b, rtol=1e-11, atol=1e-12, err_msg=name)

    try:
        fit_manifest = read(use(fit / 'artifact_hashes.json'))
        for name, digest in fit_manifest.items():
            path = (fit / name).resolve(); assert path.is_relative_to(fit)
            use(path, digest)
        meta = read(fit / 'metadata.json'); assert meta['status'] == 'complete' and meta['stage'] == 'fit'
        bank = read(fit / 'model_bank.json'); assert sha(fit / 'model_bank.json') == meta['model_bank_sha256']
        trained_inputs = read(fit / 'input_hashes.json')
        assert bank['_training_contract']['train_contexts'] == [f'T{i}' for i in range(8)]
        assert set(bank['models']) == {'G', 'O', 'S', 'N', 'G_capacity'}
        features, targets, costs, results, parents, bindings = {}, {}, {}, {}, {}, []
        for cid in (f'T{i}' for i in range(8)):
            folder = runs / cid
            manifest_path = use(folder / 'artifact_hashes.json', trained_inputs[str(folder / 'artifact_hashes.json')])
            manifest = read(manifest_path)
            runmeta = read(use(folder / 'metadata.json', manifest['metadata.json']))
            replay = read(use(folder / 'verification.json', trained_inputs[str(folder / 'verification.json')]))
            assert runmeta['context']['context_id'] == cid and runmeta['context']['role'] == 'train'
            assert runmeta['status'] == 'complete' and runmeta['physical_branches'] == 12
            assert replay['status'] == 'passed_full' and replay['passed_full'] is True
            assert replay['branches_checked'] == replay['branches_total'] == 12 and not replay['partial']
            assert replay['run'] == str(folder) and replay['artifact_manifest_sha256'] == sha(manifest_path)
            assert replay['source_sha256'] == runmeta['source_sha256']
            bindings.append({'context': cid, 'run': str(folder), 'manifest_sha256': sha(manifest_path),
                             'verification_sha256': sha(folder / 'verification.json')})
            for family in ('storage_shelves', 'ventilation_baffles'):
                history = cid + '/' + family; path = folder / family
                assets = {}
                for name in ('features.npz', 'candidates.json', 'outcomes.json'):
                    target = use(path / name, manifest[family + '/' + name])
                    assert sha(target) == trained_inputs[str(target)]
                    assets[name] = target
                with np.load(assets['features.npz']) as data: features[history] = {k: data[k].copy() for k in data.files}
                route = read(assets['candidates.json']); outcome = read(assets['outcomes.json'])
                assert [r['candidate_id'] for r in route] == list(range(6))
                by_id = {r['candidate_id']: r for r in outcome}; assert set(by_id) == set(range(6))
                results[history] = [by_id[i] for i in range(6)]; costs[history] = np.array([r['cost'] for r in route])
                parents[history] = cid
                for r in results[history]:
                    assert r['paid_actions'] > 0
                    compare(history + '/area_rate/' + str(r['candidate_id']), r['new_area_m2'] / r['paid_actions'], r['area_per_action'])
                    compare(history + '/f1/' + str(r['candidate_id']), r['after']['f1_05cm'] - r['before']['f1_05cm'], r['f1_gain_05cm'])
                targets[history] = np.array([[r['area_per_action'], r['f1_gain_05cm']] for r in results[history]])
                arrays = features[history]
                assert set(arrays) == {'G', 'O', 'S', 'N', 'G_capacity', 'X', 'M'}
                assert all(a.shape == (6, 12) and np.isfinite(a).all() for a in arrays.values())
                assert np.array_equal(arrays['M'], arrays['G'])
                assert np.array_equal(arrays['X'][:, :8], arrays['S'][:, :8])
                assert np.array_equal(arrays['X'][:, 8:], -arrays['S'][:, 8:])
        keys = sorted(features); context_ids = sorted(set(parents.values())); coefficients, models, full_cv = {}, {}, {}
        for method in ('G', 'O', 'S', 'N', 'G_capacity'):
            x = {k: features[k][method] for k in keys}; recorded = bank['models'][method]
            assert recorded['center_history'] is True
            cv = []
            for alpha in (.1, 1., 10.):
                losses, folds = [], []
                saved_cv = next(r for r in recorded['cv'] if r['alpha'] == alpha)
                for held in context_ids:
                    train_keys = [k for k in keys if parents[k] != held]
                    valid_keys = [k for k in keys if parents[k] == held]
                    beta, report = independent_svd_fit(train_keys, x, targets, alpha)
                    saved_fold = next(r for r in saved_cv['folds'] if r['held_out_context'] == held)
                    assert saved_fold['train_histories'] == train_keys and saved_fold['validation_histories'] == valid_keys
                    assert saved_fold['train_contexts'] == [c for c in context_ids if c != held]
                    for diagnostic, value in saved_fold['design'].items():
                        compare(f'{method}/{alpha}/{held}/design/{diagnostic}', report[diagnostic], value)
                    history_losses = {}
                    for k in valid_keys:
                        predicted = (x[k] - x[k].mean(0)).dot(beta)
                        truth = targets[k] - targets[k].mean(0)
                        mse = float(np.mean((predicted[:, 0] - truth[:, 0]) ** 2))
                        compare(f'{method}/{alpha}/{k}/validation_mse', mse, saved_fold['history_mse'][k])
                        losses.append(mse); history_losses[k] = mse
                    folds.append({'held_out_context': held, 'train_histories': train_keys,
                                  'validation_histories': valid_keys, 'history_mse': history_losses,
                                  'normal_equation_max_residual': report['normal_equation_max_residual']})
                mean_loss = float(np.mean(losses))
                compare(f'{method}/{alpha}/mean_cv', mean_loss, saved_cv['mean_history_validation_mse'])
                cv.append({'alpha': alpha, 'mean_history_validation_mse': mean_loss, 'folds': folds})
            minimum = min(c['mean_history_validation_mse'] for c in cv)
            tolerance = 1e-12 * max(1., abs(minimum))
            alpha = max(c['alpha'] for c in cv if c['mean_history_validation_mse'] <= minimum + tolerance)
            assert alpha == recorded['alpha'], method
            beta, report = independent_svd_fit(keys, x, targets, alpha)
            compare(method + '/coefficient', beta, recorded['coefficient'])
            for name in ('rows', 'histories', 'columns', 'design_rank', 'singular_values', 'effective_df_per_target', 'active_columns'):
                compare(method + '/full/' + name, report[name], recorded['diagnostics'][name])
            assert recorded['diagnostics']['parent_contexts'] == context_ids
            capacity = report['effective_df_per_target'] <= 4. + 1e-12
            assert capacity == recorded['diagnostics']['capacity_ok'] and capacity
            coefficients[method] = beta; models[method] = {'alpha': alpha, **report, 'capacity_ok': capacity}
            full_cv[method] = cv
        saved_choices = read(fit / 'training_choices.json')
        saved_by_key = {(r['history'], r['scorer']): r for r in saved_choices['rows']}
        assert len(saved_by_key) == 16 * 9
        rows, arrays = [], {}
        for history in keys:
            channel_map = {method: (method, method) for method in coefficients}
            channel_map.update(M=('G', 'M'), X=('S', 'X'), G_shared=('S', 'G'), O_shared=('S', 'O'))
            for scorer, (model, feature) in channel_map.items():
                x = features[history][feature]
                prediction = (x - x.mean(0)).dot(coefficients[model])
                selected = int(np.lexsort((np.arange(6), costs[history], -prediction[:, 0]))[0])
                recorded = saved_by_key[(history, scorer)]
                assert selected == recorded['candidate_id'], (history, scorer)
                outcome = results[history][selected]
                assert recorded['failure'] == outcome['failure']
                for key in ('paid_actions', 'planned_actions', 'area_per_action', 'f1_gain_05cm'):
                    compare(f'{history}/{scorer}/chosen/{key}', outcome[key], recorded[key])
                rows.append({'history': history, 'scorer': scorer, 'candidate_id': selected,
                             'failure': outcome['failure'], 'paid_actions': outcome['paid_actions'],
                             'planned_actions': outcome['planned_actions']})
                arrays[history.replace('/', '__') + '__' + scorer] = prediction
        counts = dict(Counter(r['scorer'] for r in rows if r['failure'] is not None))
        assert counts == saved_choices['selected_failures_by_scorer']
        guard = read(fit / 'guard.json'); raw = np.vstack([features[k]['G'][:, :8] for k in keys])
        # The stored field names belong to the fixed T-only OOD contract.
        minimum, maximum = raw.min(0), raw.max(0)
        margin = np.maximum(.25 * (maximum - minimum), 1e-6)
        compare('guard/lower', minimum - margin, guard['lower'])
        compare('guard/upper', maximum + margin, guard['upper'])
        for path, digest in hashes.items(): assert sha(path) == digest, f'input changed during review: {path}'
        np.savez_compressed(output / 'independent_predictions.npz', **arrays)
        write(output / 'independent_cv.json', full_cv); write(output / 'independent_choices.json', rows)
        write(output / 'numerical_differences.json', differences)
        summary = {'status': 'passed_independent_T_svd_review', 'fit': str(fit), 'run_root': str(runs),
                   'fit_manifest_sha256': sha(fit / 'artifact_hashes.json'), 'model_bank_sha256': sha(fit / 'model_bank.json'),
                   'models': models, 'parent_contexts': 8, 'histories': 16, 'target_rows': 96,
                   'cv_parent_folds': 120, 'selection_checks': len(rows),
                   'failed_training_branches_retained': sum(r['failure'] is not None for values in results.values() for r in values),
                   'selected_failure_counts': counts, 'max_absolute_difference': max(differences.values()),
                   'all_original_selections_equal': True, 'C_outcomes_read': False,
                   'production_fit_or_model_imported': False, 'new_hyperparameters_or_targets_used': False,
                   'independent_confirmation_of_semantic_advantage': False, 'bindings': bindings,
                   'runtime_seconds': time.monotonic() - started, 'script_sha256': sha(__file__)}
        write(output / 'summary.json', summary)
        (output / 'FIT_SVD_REVIEW.md').write_text(
            '# V7 T bank 独立 SVD 数值复核\n\n'
            '从原 T 的封存特征和全部 outcome 表独立重写 NumPy 中心化、history 权重、SVD、父 context CV 和动作选择；'
            '没有导入生产拟合器或模型类，没有读取 C outcomes。\n\n'
            f"8 个父 context、16 个 history、96 个候选，5 个模型 × 3 个 alpha × 8 折全部匹配；144 个 scorer/history 选择全部一致。最大数值差 {summary['max_absolute_difference']:.3g}。\n\n"
            '全部 4 条执行失败保留；训练内动作选择及正则通过只能验证拟合产物计算正确，不能证明语义收益，也不能消除旧执行器缺少最新地图守卫的问题。'
            'C 仍应等执行策略版本决定后再放行。输入哈希、每折误差与独立预测另附。\n')
        print(json.dumps({k: v for k, v in summary.items() if k not in ('models', 'bindings')}, ensure_ascii=False), flush=True)
    except Exception:
        write(output / 'failure.json', {'status': 'failed', 'traceback': traceback.format_exc()})
        raise
    finally:
        write(output / 'input_hashes.json', hashes)
        write(output / 'artifact_hashes.json', {p.name: sha(p) for p in output.iterdir() if p.is_file() and p.name != 'artifact_hashes.json'})


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--fit', type=Path, required=True)
    parser.add_argument('--runs', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    run(parser.parse_args())
