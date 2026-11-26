#!/usr/bin/env python3
"""Freeze, execute, and select the two weighting objectives for the 192-action study."""

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import csv
import fcntl
import hashlib
from itertools import product
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time


ROOT = Path(__file__).resolve().parent
RUNNER = ROOT / 'runner'
PRIOR = ROOT.parent / 'same_goal_cold_holdout_20261121'
SOURCE = ROOT.parent / 'same_goal_route_choice_20261121'
CLASS_GRID = (('G', 'zero'), ('G_route', 'decision'), ('G_perf', 'performance'), ('S', 'zero'),
              ('X', 'zero'), ('S_perf', 'performance'))
NATIVE_FREEZE = PRIOR / 'runner/holdout_freeze.json'
SPLITS = ('development', 'holdout')
ZERO = {'unknown_weight': 0.0, 'action_cost': 0.0}
PROTOCOL = ROOT / 'PROTOCOL.json'
RESULTS = ROOT / 'measurement/results'
WEIGHTS = ROOT / 'fit_weights.json'
PARITY = RUNNER / 'local_parity/receipt.json'
PREFLIGHT = ROOT / 'execution/preflight_attempt_001'


def read(path):
    return json.loads(path.read_text())


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def write_once(path, contents):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x') as stream:
        json.dump(contents, stream, indent=2, ensure_ascii=False, allow_nan=False)
        stream.write('\n')


def freeze_path(split):
    return ROOT / ('dev_freeze.json' if split == 'development' else 'holdout_freeze.json')


def verify_files(files):
    for path, expected in files.items():
        if sha(path) != expected:
            raise ValueError(f'Frozen input changed: {path}')


def verify_parity():
    parity = read(PARITY)
    assert parity['status'] == 'passed' and parity['holdout_policy_actions'] == 0
    bound = dict(parity['sources_sha256'])
    worker = RUNNER / 'weighted_run.py'
    before = PREFLIGHT / 'weighted_run.before.py'
    surviving_bound = bound.pop(str(worker))
    assert surviving_bound == sha(before)
    verify_files(bound)
    original = 'if any(ORIGINAL_SHA(path) != expected for path, expected in frozen["files_sha256"].items()):'
    repaired = 'if any(ORIGINAL_SHA(Path(path)) != expected for path, expected in frozen["files_sha256"].items()):'
    old_code = before.read_text()
    assert old_code.count(original) == 1 and old_code.replace(original, repaired) == worker.read_text()
    receipt = read(PREFLIGHT / 'repair_receipt.json')
    assert receipt['before_weighted_run_sha256'] == surviving_bound
    assert receipt['after_weighted_run_sha256'] == sha(worker)
    verify_files(receipt['failed_attempt_source_hashes'])


def geometrical_grid(protocol):
    config = protocol['local_score']
    for u, c in product(config['unknown_weight'], config['action_cost_m2_per_charged_action']):
        yield ('G' if u == c == 0 else f'G_u{round(100*u):03d}_c{round(100*c):03d}',
               {'unknown_weight': u, 'action_cost': c})


def schedule(split, protocol, fit=None):
    if split == 'development':
        layouts = protocol['development_layouts']
        modes = [('S', 'S', ZERO), *((a, 'G', w) for a, w in geometrical_grid(protocol))]
    else:
        layouts = protocol['heldout_layouts']
        modes = [(a, protocol['heldout_modes'][a]['prior'], ZERO if key == 'zero' else fit[key])
                 for a, key in CLASS_GRID]
    jobs = []
    parents = {}
    for arm, prior, weights in modes:
        tuple_id = (prior, weights['unknown_weight'], weights['action_cost'])
        source = parents.setdefault(tuple_id, arm)
        for layout in layouts:
            jobs.append({'split': split, 'layout': layout, 'arm': arm, 'prior': prior,
                         **{key: weights[key] for key in ZERO}, 'literal': source == arm,
                         'source_arm': source})
    return jobs


def seal(split):
    protocol = read(PROTOCOL)
    assert protocol['native_budget'] == protocol['capture_paid'][-1] == 192
    actor_path = ROOT / 'scenes' / f'actor_freeze_{split}.json'
    evaluator_path = ROOT / 'scenes' / f'evaluator_freeze_{split}.json'
    actor, evaluator = read(actor_path), read(evaluator_path)
    assert actor['actors_run'] == evaluator['actors_run'] == 0
    assert actor['budget'] == evaluator['budget'] == protocol['native_budget']
    verify_files(actor['files_sha256'])
    verify_files(evaluator['files_sha256'])
    expected = protocol['development_layouts' if split == 'development' else 'heldout_layouts']
    assert actor['layouts'] == evaluator['layouts'] and [row['layout'] for row in actor['layouts']] == expected
    if (RUNNER / 'runs' / split).exists():
        raise ValueError(f'A cold episode exists before {split} was sealed')
    development_fit = None
    fit_digest = None
    if split == 'holdout':
        development_fit = read(WEIGHTS)
        fit_digest = sha(WEIGHTS)
        assert development_fit['protocol_sha256'] == sha(PROTOCOL)
        assert development_fit['has_not_used_holdout'] and sha(freeze_path('development')) == development_fit['dev_freeze_sha256']
        verify_files(development_fit['fit_inputs_sha256'])
    else:
        assert not (RUNNER / 'runs/holdout').exists()
    auxiliary = [PARITY, ROOT / 'decision/historical_regression.json', ROOT / 'decision/fit_routes.py',
                 ROOT / 'measurement/evaluate.py', ROOT / 'scenes/actor_freeze_holdout.json',
                 ROOT / 'publication/preregistration_receipt.json', PREFLIGHT / 'repair_receipt.json']
    verify_parity()
    history = read(ROOT / 'decision/historical_regression.json')
    assert history['status'] == 'verified_two_historical_regressions_from_saved_states'
    preregistration = read(ROOT / 'publication/preregistration_receipt.json')
    assert preregistration['protocol_sha256'] == sha(PROTOCOL) and preregistration['verified']
    inherited = read(NATIVE_FREEZE)['files_sha256']
    # Reuse original runner/geometry code hashes without importing its scenes or old experiments' inputs.
    code_bindings = {p: h for p, h in inherited.items() if Path(p).suffix in ('.py', '.so')
                     or p.endswith('/selected_temperature.json') or p.endswith('/execution/protocol.json')}
    verify_files(code_bindings)
    actor_paths = [PROTOCOL, Path(__file__), RUNNER / 'weighted_run.py', RUNNER / 'weighted_score.py', actor_path,
                   SOURCE / 'PROTOCOL.json', SOURCE / 'route_choice.py', SOURCE / 'runner/full_run.py',
                   PRIOR / 'runner/cold_run.py']
    actor_bindings = code_bindings | actor['files_sha256'] | {str(p): sha(p) for p in actor_paths}
    if any('evaluator_truth' in path or 'evaluator_references' in path for path in actor_bindings):
        raise ValueError('Evaluator files entered the actors’ input ledger')
    jobs = schedule(split, protocol, development_fit)
    assert len(jobs) == (20 if split == 'development' else 24)
    path = freeze_path(split)
    write_once(path, {'status': 'sealed_before_cold_launch', 'split': split, 'frozen_unix_s': time.time(),
                      'protocol_sha256': sha(PROTOCOL), 'native_budget': protocol['native_budget'],
                      'capture_paid': protocol['capture_paid'], 'frozen_online_fields': protocol['frozen_online_fields'],
                      'weight_freeze_sha256': fit_digest, 'actor_scene_freeze_sha256': sha(actor_path),
                      'evaluator_scene_freeze_sha256': sha(evaluator_path), 'schedule': jobs,
                      'files_sha256': actor_bindings, 'world_files_sha256': actor['files_sha256'],
                      'supervisor_only_evidence_sha256': {str(p): sha(p) for p in [evaluator_path, *auxiliary]}})
    return path


def paths(job):
    name = ROOT / 'execution' / job['split'] / job['layout'] / job['arm']
    return name, RUNNER / 'runs' / job['split'] / job['layout'] / job['arm']


def run_job(job, freeze, resume):
    logdir, run = paths(job)
    logdir.mkdir(parents=True, exist_ok=True)
    status = logdir / 'launch.json'
    log = logdir / 'worker.log'
    with (logdir / 'lease.lock').open('a') as lease:
        fcntl.flock(lease, fcntl.LOCK_EX)
        present = [p for p in (run / 'manifest.json', run / 'failure.json') if p.is_file()]
        if present:
            if not resume or len(present) != 1:
                raise ValueError(f'Unexpected existing cold result {run}')
            result = read(present[0])
            assert result['runner_freeze_sha256'] == sha(freeze)
            if present[0].name == 'manifest.json':
                verify_files({str(run / p): h for p, h in result['files_sha256'].items()})
            return job | {'returncode': int(present[0].name == 'failure.json'), 'verified_existing': True,
                          'source_sha256': sha(present[0])}
        if any(p.exists() for p in (run, status, log)):
            if not resume:
                raise ValueError(f'Unfinished attempt exists at {run}; use frozen resume')
            previous = logdir / 'interruptions' / str(time.time_ns())
            previous.mkdir(parents=True)
            for item in (run, status, log):
                if item.exists():
                    item.rename(previous / item.name)
        cmd = [sys.executable, str(RUNNER / 'weighted_run.py'), job['split'], job['layout'], job['arm'],
               '--prior', job['prior'], '--unknown-weight', str(job['unknown_weight']),
               '--action-cost', str(job['action_cost']), '--freeze', str(freeze)]
        env = dict(os.environ, OPENBLAS_NUM_THREADS='1', OMP_NUM_THREADS='1', MKL_NUM_THREADS='1',
                   NUMEXPR_NUM_THREADS='1', PYTHONUNBUFFERED='1')
        with log.open('x') as stream:
            worker = subprocess.Popen(cmd, stdout=stream, stderr=subprocess.STDOUT, env=env, pass_fds=(lease.fileno(),))
            write_once(status, job | {'pid': worker.pid, 'started_unix_s': time.time(), 'command': cmd})
            exit_code = worker.wait()
        result_file = run / ('manifest.json' if exit_code == 0 else 'failure.json')
        return job | {'pid': worker.pid, 'returncode': exit_code, 'exit_unix_s': time.time(),
                      'log_sha256': sha(log), 'source_sha256': sha(result_file) if result_file.exists() else None}


def emit_alias(job, frozen, receipts):
    _, output = paths(job)
    source_job = next(row for row in frozen['schedule'] if row['layout'] == job['layout']
                      and row['arm'] == job['source_arm'] and row['literal'])
    _, source_dir = paths(source_job)
    manifest, failure = source_dir / 'manifest.json', source_dir / 'failure.json'
    values = {key: job[key] for key in ('split', 'layout', 'arm', 'source_arm', 'prior', *ZERO)} | {
        'schema': 'identical_frozen_configuration_alias_v1', 'run_freeze_sha256': sha(freeze_path(job['split'])),
        'source_manifest_sha256': sha(manifest) if manifest.is_file() else None,
        'source_failure_sha256': sha(failure) if failure.is_file() else None,
    }
    target = output / 'alias.json'
    if target.exists():
        assert read(target) == values
    else:
        write_once(target, values)
    return job | {'returncode': receipts[job['layout'], job['source_arm']]['returncode'],
                  'source_sha256': sha(target)}


def run(split, workers, resume):
    path = freeze_path(split)
    frozen = read(path)
    verify_files(frozen['files_sha256'])
    verify_files(frozen['supervisor_only_evidence_sha256'])
    if split == 'holdout':
        assert sha(WEIGHTS) == frozen['weight_freeze_sha256']
    expected = frozen['schedule']
    receipt_path = ROOT / 'execution' / split / 'phase_receipt.json'
    receipt_path.parent.mkdir(parents=True, exist_ok=True)
    with (receipt_path.parent / 'phase.lock').open('a') as phase_lock:
        fcntl.flock(phase_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if receipt_path.exists():
            raise ValueError(f'Phase has evidence of termination: {receipt_path}')
        started = time.monotonic()
        available_cpu = len(os.sched_getaffinity(0))
        requests = min(workers, available_cpu, 8)
        literal = {}
        with ThreadPoolExecutor(max_workers=requests) as pool:
            futures = [pool.submit(run_job, job, path, resume) for job in expected if job['literal']]
            for future in as_completed(futures):
                row = future.result()
                literal[row['layout'], row['arm']] = row
                print(json.dumps(row, ensure_ascii=False), flush=True)
        records = [literal[j['layout'], j['arm']] if j['literal'] else emit_alias(j, frozen, literal) for j in expected]
        status = 'completed' if all(job['returncode'] == 0 for job in records) else 'retained_failures'
        write_once(receipt_path, {'status': status, 'split': split, 'workers': requests,
                                  'freeze_sha256': sha(path), 'elapsed_s': time.monotonic() - started, 'jobs': records})
    return receipt_path


def fit():
    protocol = read(PROTOCOL)
    dev_freeze = freeze_path('development')
    exit_path = ROOT / 'execution/development/phase_receipt.json'
    exit_receipt = read(exit_path)
    assert exit_receipt['freeze_sha256'] == sha(dev_freeze)
    assert not (RUNNER / 'runs/holdout').exists()
    decision_path = ROOT / 'decision/route_fit.json'
    decision = read(decision_path)
    csv_path = RESULTS / 'development/episodes.csv'
    with csv_path.open(newline='') as stream:
        rows = list(csv.DictReader(stream))
    def signature(row):
        return row['layout'], row.get('arm', row.get('mode'))
    indexed = {signature(row): row for row in rows}
    expected = {(job['layout'], job['arm']) for job in schedule('development', protocol)}
    if len(rows) != len(expected) or set(indexed) != expected:
        raise ValueError('Developer evaluation does not cover every frozen literal run')

    rankings = []
    for arm, weights in geometrical_grid(protocol):
        runs = [indexed[layout, arm] for layout in protocol['development_layouts']]
        measured = [row for row in runs if row['J'] and row['actual_paid']]
        values = [float(row['J']) if row in measured else 0.0 for row in runs]
        if any(int(row['actual_paid']) > protocol['native_budget'] or not math.isfinite(float(row['J'])) for row in measured):
            raise ValueError(f'Invalid development budget or metric at {arm}')
        returned = sum(row['returned'].lower() == 'true' for row in measured)
        collisions = sum(int(row['collisions']) for row in measured)
        rankings.append({'arm': arm, **weights, 'measured': len(measured), 'returns': returned,
                         'collisions': collisions, 'eligible': len(measured) == returned == len(runs) and collisions == 0,
                         'mean_terminal_J': sum(values) / len(runs), 'worst_terminal_J': min(values),
                         'J_by_layout': {row['layout']: float(row['J']) if row in measured else None for row in runs}})
    eligible = [row for row in rankings if row['eligible']]
    capacity = eligible or rankings
    if not eligible:
        best_prefix = max((r['measured'], r['returns'], -r['collisions']) for r in capacity)
        capacity = [r for r in capacity if (r['measured'], r['returns'], -r['collisions']) == best_prefix]
    for name in ('mean_terminal_J', 'worst_terminal_J'):
        best = max(row[name] for row in capacity)
        capacity = [row for row in capacity if best - row[name] <= 1e-9]
    winner = min(capacity, key=lambda row: (row['unknown_weight'] / .25 + row['action_cost'] / .20,
                                           row['unknown_weight'], row['action_cost']))
    inputs = [decision_path, csv_path, dev_freeze, exit_path]
    write_once(WEIGHTS, {'status': 'two_developer_weight_fits_frozen_before_holdout', 'sealed_unix_s': time.time(),
                         'has_not_used_holdout': True, 'protocol_sha256': sha(PROTOCOL), 'dev_freeze_sha256': sha(dev_freeze),
                         'fit_inputs_sha256': {str(p): sha(p) for p in inputs},
                         'decision': {key: decision['decision'][key] for key in ZERO},
                         'performance': {key: winner[key] for key in ZERO}, 'performance_fit_arm': winner['arm'],
                         'no_reliable_developer_configuration': not eligible, 'developer_performance_grid': rankings})
    return WEIGHTS


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('operation', choices=('seal', 'run', 'fit'))
    parser.add_argument('split', nargs='?', choices=SPLITS)
    parser.add_argument('--workers', type=int, choices=range(1, 9), default=8)
    parser.add_argument('--resume', action='store_true')
    options = parser.parse_args()
    if options.operation == 'fit':
        result = fit()
    elif options.split is None:
        parser.error('seal and run require a split')
    elif options.operation == 'seal':
        result = seal(options.split)
    else:
        result = run(options.split, options.workers, options.resume)
    print(result, flush=True)


if __name__ == '__main__':
    main()
