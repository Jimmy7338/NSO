#!/usr/bin/env python3
"""Seal and run the four native branches of the declared route intervention."""

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time


ROOT = Path(__file__).resolve().parent
PRIOR = ROOT.parent / 'weighted_geometric_control_20261122'
PROTOCOL = ROOT / 'PROTOCOL.json'
MEASUREMENT = ROOT / 'evaluation/MEASUREMENT_PROTOCOL.json'
FREEZE = ROOT / 'RUN_FREEZE.json'
EXECUTION = ROOT / 'execution'
RUNS = ROOT / 'runner/runs'
REPLAYS = ROOT / 'runner/replays'
ARMS = ('category', 'geometry')


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def write_new(path, value):
    with path.open('x') as stream:
        json.dump(value, stream, indent=2, allow_nan=False, ensure_ascii=False)
        stream.write('\n')


def hash_files(paths):
    return {str(path.resolve()): sha(path) for path in sorted(set(paths))}


def verify_files(files):
    for path, expected in files.items():
        if sha(path) != expected:
            raise ValueError(f'Frozen file has changed: {path}')


def scheduled(protocol):
    return [dict(layout=case['layout'], arm=arm, route_id=case[f'{arm}_route'])
            for case in protocol['cases'] for arm in ARMS]


def seal():
    if RUNS.exists():
        raise ValueError('A branch has already started before this seal')
    protocol = read(PROTOCOL)
    measurement = read(MEASUREMENT)
    archive = read(PRIOR / 'publication/receipt.json')
    if archive['head'] != protocol['source_commit'] or archive['status'] != 'verified_published_draft':
        raise ValueError('Prior publication receipt does not bind the requested commit')
    inherited = read(PRIOR / 'holdout_freeze.json')
    verify_files(inherited['files_sha256'])
    if inherited['native_budget'] != protocol['native_budget_including_return']:
        raise ValueError('Protocol changed the native action budget')
    weights_file = PRIOR / 'fit_weights.json'
    assert inherited['weight_freeze_sha256'] == sha(weights_file)
    assert read(weights_file)['performance'] == {
        key: protocol['continuation_policy'][key] for key in ('unknown_weight', 'action_cost')}

    anchors_path = ROOT / 'source/frozen_anchors.json'
    evaluator_bindings = ROOT / 'source/evaluator_bindings.json'
    anchors = read(anchors_path)
    identity = read(evaluator_bindings)
    assert anchors['commit'] == identity['commit'] == protocol['source_commit']
    for case in protocol['cases']:
        layout = case['layout']
        anchor = anchors['scenes'][layout]
        measured_case = measurement['cases'][layout]
        assert anchor['paid_before'] == case['paid_before'] == measured_case['fork_paid']
        assert anchor['selection_call_id'] == case['selection_call_id'] == measured_case['selection_call_id']
        assert identity['scenes'][layout]['target_facility_key'] == measured_case['primary_target']
        assert anchor['nominal_paid_arrival'] == case['nominal_paid_arrival']
        routes = {arm: anchor['routes'][arm] for arm in ARMS}
        for arm, route in routes.items():
            assert route['route_id'] == case[f'{arm}_route'] and route['admitted']
            assert route['outbound_cost'] == case['equal_nominal_outbound_actions']
        assert routes['category']['unknown_union_cells'] - routes['geometry']['unknown_union_cells'] == (
            case['category_minus_geometry_forecast_unknown_cells'])
    source_files = [anchors_path, ROOT / 'source/remote_readback.json']
    source_manifests = [PRIOR / 'runner/runs/holdout' / case['layout'] / 'S' for case in protocol['cases']]
    expectations = ('attempt.json', 'manifest.json', 'steps.json', 'local_choices.json', 'global_choices.json',
                    'initial_raw_rgbd_scan_sha256.json', 'observed_belief_trace.npz')
    history = [path / member for path in source_manifests for member in expectations]
    for path in source_manifests:
        manifest = read(path / 'manifest.json')
        assert manifest['study_prior'] == 'S' and manifest['global_mode'] == 'G'
        fork_paid = next(case['paid_before'] for case in protocol['cases'] if case['layout'] == path.parent.name)
        checkpoint_slots = ('start', *(f'paid_{paid}' for paid in inherited['capture_paid'] if 0 < paid <= fork_paid))
        checkpoint_members = [f'{slot}/{name}' for slot in checkpoint_slots
                              for name in ('checkpoint.json', 'belief.npz', 'native_mesh.ply')]
        history.extend(path / member for member in checkpoint_members)
        for member in (*expectations, *checkpoint_members):
            if member != 'manifest.json' and sha(path / member) != manifest['files_sha256'][member]:
                raise ValueError(f'Prior S evidence has changed: {path / member}')
    actor_local = [PROTOCOL, weights_file, PRIOR / 'holdout_freeze.json', PRIOR / 'runner/weighted_run.py',
                   PRIOR / 'runner/weighted_score.py', *source_files, *history]
    actor_local.extend((ROOT / 'runner').glob('*.py'))
    actor = inherited['files_sha256'] | hash_files(actor_local)
    if any('evaluator_references' in path or 'evaluator_truth' in path for path in actor):
        raise ValueError('Evaluator truth entered the actor input list')

    truth_root = PRIOR / 'scenes/evaluator_references'
    truths = [truth_root / case['layout'] / name for case in protocol['cases']
              for name in ('RECEIPT.json', 'evaluator_truth.npz', 'fixed_reference.npz', 'source_mesh.ply')]
    earlier_eval = ROOT.parent / 'same_goal_route_choice_20261121/measurement'
    offline_only = hash_files([Path(__file__), MEASUREMENT, evaluator_bindings, ROOT / 'source/seal_source.py',
                               PRIOR / 'measurement/evaluate.py', *truths,
                               *earlier_eval.glob('*.py'), *(ROOT / 'evaluation').glob('*.py')])
    offline_only |= {str(ROOT.parent / path): digest
                     for path, digest in measurement['metric_and_reference_bindings_sha256'].items()}
    for case in measurement['cases'].values():
        offline_only |= {str(ROOT.parent / path): digest for path, digest in case['published_input_sha256'].items()}
    verify_files(offline_only)
    write_new(FREEZE, dict(status='sealed_before_native_injected_branches',
                           frozen_unix_s=time.time(), protocol_sha256=sha(PROTOCOL),
                           measurement_protocol_sha256=sha(MEASUREMENT),
                           prior_publication_receipt_sha256=sha(PRIOR / 'publication/receipt.json'),
                           inherited_holdout_freeze_sha256=sha(PRIOR / 'holdout_freeze.json'),
                           scope=measurement.get('schema'), schedule=scheduled(protocol),
                           actor_files_sha256=actor, evaluator_files_sha256=offline_only))
    print(FREEZE)


def actor_command(layout, arm, out, capture_at=None, original_run=None):
    command = [sys.executable, str(ROOT / 'runner/run.py'), layout, arm,
               '--freeze', str(FREEZE), '--out', str(out)]
    if capture_at is not None:
        command.extend(['--capture-at', str(capture_at), '--original-run', str(original_run)])
    return command


def worker_environment():
    return dict(os.environ, OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1',
                NUMEXPR_NUM_THREADS='1', PYTHONUNBUFFERED='1', PYTHONDONTWRITEBYTECODE='1')


def run():
    frozen = read(FREEZE)
    assert frozen['protocol_sha256'] == sha(PROTOCOL)
    assert frozen['measurement_protocol_sha256'] == sha(MEASUREMENT)
    assert frozen['schedule'] == scheduled(read(PROTOCOL))
    verify_files(frozen['actor_files_sha256'])
    verify_files(frozen['evaluator_files_sha256'])
    EXECUTION.mkdir(exist_ok=False)
    processes = []
    environment = worker_environment()
    for job in frozen['schedule']:
        layout, arm = job['layout'], job['arm']
        output = RUNS / layout / arm
        log_path = EXECUTION / layout / f'{arm}.log'
        log_path.parent.mkdir(exist_ok=True, parents=True)
        command = actor_command(layout, arm, output)
        started = time.time()
        with log_path.open('x') as log:
            process = subprocess.Popen(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, env=environment)
        launch = job | {'pid': process.pid, 'started_unix_s': started, 'command': command}
        write_new(log_path.with_suffix('.launch.json'), launch)
        processes.append((process, launch, output, log_path))

    finished = []
    while processes:
        for item in processes[:]:
            process, launch, output, log_path = item
            status = process.poll()
            if status is None:
                continue
            record = output / ('manifest.json' if status == 0 else 'failure.json')
            outcome = launch | {'retcode': status, 'observed_exit_unix_s': time.time(), 'log_sha256': sha(log_path),
                                'record_path': str(record), 'record_sha256': sha(record) if record.exists() else None}
            write_new(log_path.with_suffix('.exit.json'), outcome)
            finished.append(outcome)
            processes.remove(item)
            print(launch['layout'], launch['arm'], 'process exit', status, flush=True)
        if processes:
            time.sleep(1.0)
    write_new(EXECUTION/'phase_receipt.json', dict(run_freeze_sha256=sha(FREEZE), jobs=finished))
    if any(job['retcode'] != 0 or job['record_sha256'] is None for job in finished):
        raise SystemExit('At least one branch has a technical failure. Its unfiltered receipt and log are retained.')


def replay_missing():
    phase = read(EXECUTION / 'phase_receipt.json')
    assert phase['run_freeze_sha256'] == sha(FREEZE)
    output_manifest = EXECUTION / 'paired_clock_receipt.json'
    records = []
    for case in read(PROTOCOL)['cases']:
        layout = case['layout']
        files = {arm: RUNS / layout / arm / 'manifest.json' for arm in ARMS}
        results = {arm: read(path) for arm, path in files.items()}
        if any(result['status'] != 'completed_execution' for result in results.values()):
            raise ValueError(f'Run must provide a native terminal before diagnostic replay: {layout}')
        paid = max(result['route_end']['actual_paid'] for result in results.values())
        for arm, result in results.items():
            archive = files[arm].parent
            record = dict(layout=layout, arm=arm, requested_paid=paid, native_manifest_sha256=sha(files[arm]))
            match = next((name for name, receipt in result['checkpoints'].items()
                          if receipt['status'] == 'captured' and receipt['actual_paid'] == paid), None)
            if match is not None:
                records.append(record | {'status': 'native_capture', 'capture': match})
                continue
            if result['paid_actions'] < paid:
                records.append(record | {'status': 'native_terminal_before_clock', 'terminal_paid': result['paid_actions']})
                continue
            output = REPLAYS/layout/arm/f'paid_{paid}'
            log = EXECUTION/layout/f'{arm}_capture_at_{paid}.log'
            command = actor_command(layout, arm, output, paid, archive)
            print(layout, arm, 'verified clock capture', paid, flush=True)
            with log.open('x') as stream:
                status = subprocess.run(command, cwd=ROOT, stdout=stream, stderr=subprocess.STDOUT,
                                        env=worker_environment(), check=False)
            if status.returncode:
                raise RuntimeError(f'Clock capture failed with status {status.returncode}: {log}')
            manifest = output/'manifest.json'
            replay = read(manifest)
            assert replay['original_full_trace_verified'] and replay['extra_capture_paid'] == paid
            assert replay['verified_against_original_manifest_sha256'] == record['native_manifest_sha256']
            records.append(record | {'status': 'exact_capture_replay', 'receipt_path': str(manifest),
                                      'receipt_sha256': sha(manifest), 'log_sha256': sha(log), 'command': command})
    write_new(output_manifest, {'run_freeze_sha256': sha(FREEZE), 'captures': records})
    print(output_manifest)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('phase', choices=('seal', 'run', 'replay-missing'))
    args = parser.parse_args()
    {'seal': seal, 'run': run, 'replay-missing': replay_missing}[args.phase]()


if __name__ == '__main__':
    main()
