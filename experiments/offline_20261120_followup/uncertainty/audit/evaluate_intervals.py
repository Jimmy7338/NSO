#!/usr/bin/env python3
"""Split-separated independent evaluation of unchanged B8 PMFs and B9 sets."""
import argparse
import csv
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import sys
import traceback
from datetime import datetime, timezone
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
B8 = ROOT.parents[1] / 'dual_template_uncertainty_20261120'
sys.path.insert(0, str(B8 / 'audit'))
import evaluate_calibration as base
import distribution_metrics as dm
sys.path.insert(0, str(ROOT / 'method'))
import conformal
CONDITIONS = tuple(conformal.CONDITIONS)
SIGMA = .01
ALPHA = .05

def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def read(path): return json.loads(Path(path).read_text())
def resource(path): return dict(path=str(Path(path).resolve()), sha256=sha(path))
def write(path, obj): Path(path).write_text(json.dumps(obj, indent=2, allow_nan=False)+'\n')
def now(): return datetime.now(timezone.utc).isoformat()
def load_resource(obj):
    assert sha(obj['path']) == obj['sha256'], obj['path']
    return read(obj['path'])

def validate_inputs(dataset, index):
    objects = dataset['objects']; cases = dataset['cases']
    assert len(objects) == 48 and len(cases) == 192
    by = {o['object_id']: o for o in objects}; assert len(by) == 48
    assert {o['split'] for o in objects} == {'calibration', 'test'}
    for split in ('calibration', 'test'):
        assert sum(o['split'] == split for o in objects) == 24
    cids = set()
    for case in cases:
        assert case['case_id'] not in cids; cids.add(case['case_id'])
        assert case['split'] == by[case['object_id']]['split']
    for obj in objects:
        subset = [c for c in cases if c['object_id'] == obj['object_id']]
        assert len(subset) == 4 and {c['condition'] for c in subset} == set(CONDITIONS)
        assert set(obj['cases']) == {c['case_id'] for c in subset}
    assert index['status'] in ('completed_isolated_banks', 'completed_with_failures')
    assert not index['pending_case_ids']
    assert len(index['cases']) == 192
    assert {c['case_id'] for c in index['cases']} == cids
    assert len({c['case_id'] for c in index['cases']}) == 192

def install_truth_boundary(dataset, split):
    """Trusted-code provenance audit, not an operating-system security sandbox."""
    truth = {Path(c['truth']['path']).resolve(): c['split'] for c in dataset['cases']}
    opened = set(); blocked = []
    forbidden = {(ROOT/'generator/generator_spec.json').resolve(),
                 (ROOT/'generator/spec.py').resolve()}
    def hook(event, args):
        if event != 'open' or not args or not isinstance(args[0], (str, bytes, os.PathLike)): return
        p = Path(os.fsdecode(args[0])).resolve()
        if p in forbidden or (p in truth and truth[p] != split):
            blocked.append(str(p)); raise PermissionError(f'Wrong evaluation split: {p}')
        if p in truth: opened.add(str(p))
    sys.addaudithook(hook)
    return opened, blocked

def frozen_paths():
    paths = [Path(__file__), ROOT/'audit/test_evaluator.py', ROOT/'audit/EVALUATION_SPEC.md',
             ROOT/'audit/validation.json', ROOT/'audit/counterfactual_check.json', ROOT/'method/validation.json',
             ROOT/'generator/freeze.json', ROOT/'generator/dataset_manifest.json',
             ROOT/'generator/protocol.json', ROOT/'inference/inference_freeze.json',
             ROOT/'inference/inference_index.json',
             B8/'audit/evaluate_calibration.py', B8/'audit/distribution_metrics.py', B8/'audit/check_counterfactual.py',
             B8/'audit/calibration_attempt_001/selected_scale.json']
    validation = read(ROOT/'method/validation.json')
    paths += [Path(p) for p in validation['files_sha256']]
    paths.append(Path(validation['external_training_source']['path']))
    return sorted(set(p.resolve() for p in paths))

def freeze():
    path = ROOT/'audit/freeze.json'; assert not path.exists(), 'Existing evaluator freeze is immutable'
    val = read(ROOT/'method/validation.json')
    for p, digest in val['files_sha256'].items(): assert sha(p) == digest
    load_resource(val['external_training_source'])
    assert read(ROOT/'audit/counterfactual_check.json')['status'] == 'passed'
    assert read(B8/'audit/calibration_attempt_001/selected_scale.json')['sigma_m'] == SIGMA
    write(path, dict(status='frozen_before_new_interval_calibration_or_test', created_utc=now(),
         files_sha256={str(p): sha(p) for p in frozen_paths()}, sigma_m=SIGMA,
         calibration_objects=24, test_objects=24, cases_per_object=4,
         original_PMF_or_actor_changed=False, test_release_required=True))
    return resource(path)

def check_freeze():
    receipt = read(ROOT/'audit/freeze.json')
    for p, digest in receipt['files_sha256'].items(): assert sha(p) == digest, p
    return resource(ROOT/'audit/freeze.json')

def rows_for_method(rows):
    return [dict(object_id=r['object_id'], condition=r['condition'],
         target_extent_m=r['target']['projected_true_extent_m'],
         median_m=r['metrics']['median_m'],
         lower_m=r['metrics']['intervals']['0.95']['lower_m'],
         upper_m=r['metrics']['intervals']['0.95']['upper_m'],
         fallback=r['evaluator_prior_fallback']) for r in rows]

def interval_metrics(interval, target):
    lower = float(interval['lower_m']); upper = interval['upper_m']
    if upper is None:
        assert lower == 0 and interval.get('finite') is False
        return dict(covered=target >= lower, width_m=None, interval_score_m=None, unbounded=True)
    upper = float(upper)
    assert math.isfinite(lower) and math.isfinite(upper) and 0 <= lower <= upper
    score = upper-lower + (2/ALPHA)*max(lower-target, target-upper, 0.)
    return dict(covered=lower-1e-12 <= target <= upper+1e-12, width_m=upper-lower,
                interval_score_m=score, unbounded=False)

def attach_intervals(rows, calibration):
    out = []
    for row in rows:
        m = row['metrics']; target = row['target']['projected_true_extent_m']
        applied = conformal.apply_to_distribution(m['depth_values_m'], m['depth_probability_mass'], calibration)
        assert applied['raw_distribution']['depth_values_m'] == m['depth_values_m']
        assert applied['raw_distribution']['probability_mass'] == m['depth_probability_mass']
        assert not applied['posterior_distribution_changed'] and not applied['actor_or_gain_changed']
        for key in ('lower_m', 'upper_m', 'width_m'):
            assert applied['base_interval'][key] == m['intervals']['0.95'][key], key
        out.append(dict(**row, intervals=applied,
             base_interval_metrics=interval_metrics(applied['base_interval'], target),
             calibrated_interval_metrics=interval_metrics(applied['calibrated_interval'], target)))
    return out

def aggregate(items, kind):
    assert items
    by = {}
    for row in items: by.setdefault(row['object_id'], []).append(row)
    def avg(xs): return float(sum(xs)/len(xs))
    metrics = [r[kind+'_interval_metrics'] for r in items]
    unbounded = sum(m['unbounded'] for m in metrics)
    complete = {oid: rs for oid, rs in by.items() if len(rs) == 4 and {r['condition'] for r in rs} == set(CONDITIONS)}
    result = dict(objects=len(by), cases=len(items), covered_cases=sum(m['covered'] for m in metrics),
         equal_object_coverage_fraction=avg([avg([float(r[kind+'_interval_metrics']['covered']) for r in rs]) for rs in by.values()]),
         unbounded_cases=unbounded,
         equal_object_mean_width_m=None if unbounded else avg([avg([r[kind+'_interval_metrics']['width_m'] for r in rs]) for rs in by.values()]),
         equal_object_mean_interval_score_m=None if unbounded else avg([avg([r[kind+'_interval_metrics']['interval_score_m'] for r in rs]) for rs in by.values()]),
         simultaneous_complete_objects=len(complete),
         simultaneous_covered_objects=sum(all(r[kind+'_interval_metrics']['covered'] for r in rs) for rs in complete.values()))
    result['simultaneous_object_coverage_fraction'] = (result['simultaneous_covered_objects']/len(complete)) if complete else None
    return result

def summarize_intervals(rows):
    by_object = {}
    for r in rows: by_object.setdefault(r['object_id'], []).append(r)
    assert all(len(rs) == 4 and {r['condition'] for r in rs} == set(CONDITIONS) for rs in by_object.values())
    groups = {'all_conditions': rows}
    groups.update({'condition_'+c: [r for r in rows if r['condition']==c] for c in CONDITIONS})
    for field in ('family', 'shape_regime'):
        for value in sorted({r['geometry_group'][field] for r in rows}):
            groups[field+'_'+value] = [r for r in rows if r['geometry_group'][field]==value]
    valid = [r for r in rows if not r['evaluator_prior_fallback']]
    if valid: groups['valid_only_secondary'] = valid
    return {name: {kind: aggregate(items, kind) for kind in ('base', 'calibrated')} for name, items in groups.items()}

def annotate_groups(rows):
    cache = {}
    for row in rows:
        oid = row['object_id']
        if oid not in cache:
            record = load_resource(row['truth'])['construction']
            cache[oid] = {k: record[k] for k in ('family', 'shape_regime')}
        row['geometry_group'] = cache[oid]

def check_exported_posterior(cases):
    checked = 0
    for c in cases:
        if c['failure']: continue
        track = c['track']; assert track['sigma_m'] == SIGMA
        expected = dm.joint_posterior(c['energy'], c['prior'], SIGMA)
        np.testing.assert_allclose(track['joint_probability_mass'], expected, rtol=1e-12, atol=1e-14)
        checked += 1
    return checked

def emit_csv(path, summary):
    fields = ['group', 'method', 'objects', 'cases', 'covered_cases', 'equal_object_coverage_fraction',
              'simultaneous_complete_objects', 'simultaneous_covered_objects', 'simultaneous_object_coverage_fraction',
              'equal_object_mean_width_m', 'equal_object_mean_interval_score_m', 'unbounded_cases']
    with Path(path).open('w') as f:
        writer = csv.DictWriter(f, fieldnames=fields); writer.writeheader()
        for group, methods in summary.items():
            for method, values in methods.items(): writer.writerow(dict(group=group, method=method, **values))

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=['freeze', 'calibrate', 'evaluate'])
    parser.add_argument('--output', type=Path)
    parser.add_argument('--calibration', type=Path)
    parser.add_argument('--test-release', type=Path)
    args = parser.parse_args()
    if args.mode == 'freeze': print(json.dumps(freeze())); return
    receipt = check_freeze()
    dataset_path = ROOT/'generator/dataset_manifest.json'; index_path = ROOT/'inference/inference_index.json'
    dataset = read(dataset_path); index = read(index_path); validate_inputs(dataset, index)
    assert args.output and not args.output.exists(), 'Each attempt uses a fresh output directory'
    args.output.mkdir(parents=True)
    split = 'calibration' if args.mode == 'calibrate' else 'test'
    opened, blocked = install_truth_boundary(dataset, split)
    shared = dict(freeze=receipt, dataset=resource(dataset_path), predictions=resource(index_path),
         sigma_m=SIGMA, alpha=ALPHA, independent_objects=24, dependent_cases=96,
         source_distribution='frozen IID procedural geometry mixture; exact-pose clean first-hit RGB-D',
         posterior_or_policy_changed=False, audit_boundary='Python trusted-code provenance audit, not OS sandbox')
    try:
        if args.mode == 'evaluate':
            assert args.calibration and args.test_release
            calibration_receipt = read(args.calibration)
            assert calibration_receipt['status'] == 'frozen_calibration_before_test'
            calibration = load_resource(calibration_receipt['calibrated_config'])
            assert calibration_receipt['freeze'] == receipt
            assert set(calibration['calibration_object_ids']).isdisjoint(o['object_id'] for o in dataset['objects'] if o['split']=='test')
            release = read(args.test_release)
            assert release['status'] == 'protocols_frozen_test_evaluation_released'
            for r in release['frozen_protocols']: load_resource(r)
            shared.update(calibration_receipt=resource(args.calibration), test_release=resource(args.test_release),
                          calibration_loaded_before_test_truth=True)
        cases, failures, truth_paths = base.load_cases(dataset, index, split)
        assert len(cases) == 96
        verified = check_exported_posterior(cases)
        rows = base.case_metrics(cases, SIGMA)
        if args.mode == 'calibrate':
            training_path = ROOT/'method/training_fit.json'; training = read(training_path)
            # Receipt was already frozen; no new test or B8 validation/heldout input is loaded.
            derived = conformal.fit_training_scale(training['raw_training_scalar_inputs'])
            assert derived['s0_m'] == training['s0_m'] and derived['training_rows_sha256'] == training['training_rows_sha256']
            calibration = conformal.calibrate_object_intervals(rows_for_method(rows['posterior']), training)
            assert calibration['calibration_objects'] == 24 and calibration['order_statistic_rank'] == 24
            write(args.output/'calibrated_config.json', calibration)
            write(args.output/'calibration_rows.json', dict(rows=rows, failures=failures, opened_truth_paths=truth_paths, **shared))
            check_freeze()
            write(args.output/'calibration_receipt.json', dict(status='frozen_calibration_before_test', created_utc=now(),
                  calibrated_config=resource(args.output/'calibrated_config.json'), calibration_rows=resource(args.output/'calibration_rows.json'),
                  training_fit=resource(training_path), opened_truth_splits=['calibration'], opened_truth_paths=sorted(opened),
                  blocked_paths=blocked, posterior_comparisons_verified=verified, failures=failures, **shared))
            print(json.dumps(dict(status='calibrated', q=calibration['q'], s0_m=calibration['s0_m'], fallback_cases=len(failures))))
        else:
            intervals = attach_intervals(rows['posterior'], calibration); annotate_groups(intervals)
            interval_summary = summarize_intervals(intervals)
            out = dict(status='completed_once_frozen_test_evaluation', created_utc=now(), interval_summary=interval_summary,
                  original_distribution_summary=base.summarize(rows), failures=failures, fallback_cases=len(failures),
                  rows=intervals, distribution_rows=rows, opened_truth_splits=['test'], opened_truth_paths=sorted(opened),
                  blocked_paths=blocked, posterior_comparisons_verified=verified,
                  interpretation='Corrected sets are report-only. CRPS belongs to the unchanged PMF. This is not a planning or policy benefit.', **shared)
            check_freeze(); write(args.output/'summary.json', out); emit_csv(args.output/'interval_results.csv', interval_summary)
            print(json.dumps(dict(status=out['status'], all_conditions=interval_summary['all_conditions'], fallback_cases=len(failures))))
    except Exception as exc:
        write(args.output/'failure.json', dict(status='evaluation_failed_retained', error=repr(exc), traceback=traceback.format_exc(),
              opened_truth_paths=sorted(opened), blocked_paths=blocked, **shared)); raise

if __name__ == '__main__': main()
