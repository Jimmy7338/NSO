#!/usr/bin/env python3
"""Verify and stage the completed four-building evaluation as one release."""

import argparse
import gzip
import hashlib
import json
from pathlib import Path
import shutil
import zipfile


ROOT = Path(__file__).resolve().parent
RUNNER = ROOT / 'runner'
METRICS = ROOT / 'measurement/results'
PREFIX = 'experiments/offline_20261121_same_goal_cold_holdout'


def read(path):
    return json.loads(path.read_text())


def write(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n')


def sha(path, algorithm='sha256'):
    with path.open('rb') as source:
        return hashlib.file_digest(source, algorithm).hexdigest()


def upload_item(path, payload):
    body = path.read_bytes()
    git_object = b'blob ' + str(len(body)).encode() + b'\0' + body
    return {'stage_path': str(path), 'repo_path': f'{PREFIX}/{path.relative_to(payload).as_posix()}',
            'size': len(body), 'sha256': hashlib.sha256(body).hexdigest(),
            'git_sha1': hashlib.sha1(git_object).hexdigest()}


def verified_phase(phase, layouts, modes, expected_freeze):
    launch = read(RUNNER / f'{phase}_launch.json')
    receipt = read(RUNNER / f'{phase}_supervisor_receipt.json')
    expected = {(layout, arm) for layout in layouts for arm in modes}
    assert {(job['layout'], job['arm']) for job in launch['expected_jobs']} == expected
    assert {(job['layout'], job['arm']) for job in receipt['jobs']} == expected
    assert len(receipt['jobs']) == len(expected)
    assert receipt['runner_freeze_sha256'] == launch['runner_freeze_sha256'] == expected_freeze
    for job in receipt['jobs']:
        assert sha(RUNNER / 'logs' / f'{job["layout"]}_{job["arm"]}.log') == job['log_sha256']
    return receipt['jobs']


def run_bundle(directory, archive):
    anchor = directory / 'manifest.json'
    if not anchor.is_file():
        anchor = directory / 'failure.json'
    files = {item.relative_to(directory).as_posix(): item for item in directory.rglob('*') if item.is_file()}
    retained = {}
    native = {}
    with zipfile.ZipFile(archive, 'x', compression=zipfile.ZIP_DEFLATED, compresslevel=8) as bundle:
        for name, path in sorted(files.items()):
            digest = sha(path)
            if path.suffix in ('.ply', '.png', '.npy'):
                native[name] = {'sha256': digest, 'bytes': path.stat().st_size}
            else:
                entry = zipfile.ZipInfo(name, (2026, 11, 21, 0, 0, 0))
                bundle.writestr(entry, path.read_bytes(), compress_type=zipfile.ZIP_DEFLATED, compresslevel=8)
                retained[name] = digest
    with zipfile.ZipFile(archive) as bundle:
        assert bundle.testzip() is None
        assert all(hashlib.sha256(bundle.read(name)).hexdigest() == digest for name, digest in retained.items())
    if anchor.is_file() and anchor.name == 'manifest.json':
        declared = read(anchor)['files_sha256']
        for name, digest in declared.items():
            assert (retained.get(name) or native[name]['sha256']) == digest
    return {'run_directory': str(directory), 'archive': archive.name, 'archive_sha256': sha(archive),
            'archived_files_sha256': retained, 'native_artifacts_on_host': native}


def phase_starts(jobs):
    return [read(RUNNER / 'worker_pids' / f'{job["layout"]}_{job["arm"]}.json')['started_unix_s'] for job in jobs
            if (RUNNER / 'worker_pids' / f'{job["layout"]}_{job["arm"]}.json').is_file()]


def audit(protocol):
    layouts = protocol['layouts']
    episode_rows = read(METRICS / 'episodes.json')['rows']
    pair_rows = read(METRICS / 'pair_causal_audit.json')['rows']
    indexed = {(row['layout'], row['mode']): row for row in episode_rows}
    pairs = {row['layout']: row for row in pair_rows}
    contrasts = read(METRICS / 'contrasts.json')['rows']
    checkpoints = read(METRICS / 'checkpoints.json')['rows']
    gate = read(METRICS / 'engineering_gate.json')
    for evidence_path, digest in gate['input_evidence_sha256'].items():
        assert sha(METRICS / evidence_path) == digest
    freeze_path = RUNNER / 'holdout_freeze.json'
    freeze = read(freeze_path)
    evaluator_path = ROOT / 'measurement/EVALUATOR_FREEZE.json'
    evaluator = read(evaluator_path)
    for path, digest in (evaluator['inputs_sha256'] | freeze['files_sha256']).items():
        assert sha(Path(path)) == digest
    proof = read(ROOT / 'diagnostics/regression_verification.json')
    assert proof['status'] == 'passed_independent_readonly_regression_review'
    assert {row['layout'] for row in evaluator['layouts']} == set(layouts)
    assert len(indexed) == len(episode_rows) == 16
    assert set(indexed) == {(layout, mode) for layout in layouts for mode in protocol['modes']}
    assert len(pairs) == len(pair_rows) == 4 and set(pairs) == set(layouts)
    assert len(contrasts) == 12 and {(row['layout'], row['comparison']) for row in contrasts} == {
        (layout, f'fullS - {mode}') for layout in layouts for mode in protocol['modes'] if mode != 'fullS'
    }
    prior, baselines = (verified_phase(phase, layouts, modes, sha(freeze_path))
                        for phase, modes in zip(('primary', 'baselines'), protocol['batches'], strict=True))
    assert len(phase_starts(baselines)) == 8
    assert max(job['completion_unix_s'] for job in prior) <= min(phase_starts(baselines))

    for row in indexed.values():
        if row['status'] != 'completed_execution':
            continue
        directory = RUNNER / 'runs' / row['layout'] / row['mode']
        manifest = read(directory / 'manifest.json')
        assert row['source_manifest_sha256'] == sha(directory / 'manifest.json')
        assert (row['actual_paid'], row['returned'], row['collisions'], row['stop_reason']) == (
            manifest['paid_actions'], manifest['returned'], manifest['collisions'],
            manifest['observed_native_termination']['reason'],
        )
        selected = [point for point in checkpoints if point['layout'] == row['layout'] and point['mode'] == row['mode']]
        expected = {'start', 'terminal'} | {f'paid_{p}' for p in protocol['capture_paid'][1:]}
        assert len(selected) == len(expected) and {point['checkpoint'] for point in selected} == expected
        terminal = next(point for point in selected if point['checkpoint'] == 'terminal')
        assert terminal['actual_paid'] == row['actual_paid'] <= protocol['native_budget']
        assert abs(row['J'] - row['C_free'] * row['F1_scene_05cm']) <= 1e-10
        assert abs(terminal['J'] - row['J']) <= 1e-10
        for point in selected:
            if not point['checkpoint'].startswith('paid_'):
                continue
            paid = int(point['checkpoint'].removeprefix('paid_'))
            assert point['requested_paid'] == paid
            if paid <= terminal['actual_paid']:
                assert point['status'] == 'captured' and point['actual_paid'] == paid
            else:
                assert point['status'] == 'unavailable' and point.get('J') is None
    for row in contrasts:
        mode = row['comparison'].removeprefix('fullS - ')
        category, control = (indexed[(row['layout'], arm)] for arm in ('fullS', mode))
        if row['status'] == 'measured':
            for metric in ('C_free', 'F1_scene_05cm', 'J'):
                assert abs(row[f'delta_{metric}_pp'] - 100 * (category[metric] - control[metric])) <= 1e-10
        else:
            assert category.get('J') is None or control.get('J') is None
    criterion = protocol['evaluation']['practical_gate']
    primary_pairs = [row for row in contrasts if row['comparison'] == 'fullS - fullG']
    positive = [row['layout'] for row in primary_pairs if row.get('delta_J', 0) > criterion['threshold_delta_J_fraction_strict_gt']]
    gate_data = gate['engineering_gate']
    assert gate_data['status'] == 'evaluated' and gate_data['qualifying_buildings'] == positive
    assert gate_data['pass'] == (len(positive) >= criterion['minimum_buildings']
                                 and gate_data['maintained_all_achieved_fullG_returns']
                                 and gate_data['no_increased_collisions_against_fullG']
                                 and gate_data['all_episodes_within_budget'])
    return {
        'status': 'verified_evaluation_archive',
        'protocol_sha256': sha(ROOT / 'PROTOCOL.json'), 'runner_freeze_sha256': sha(freeze_path),
        'evaluator_freeze_sha256': sha(evaluator_path), 'independent_unit': 'building',
        'short_regression_verification': proof, 'baseline_batch_launched_after_all_eight_primary_exits': True,
        'registered_runs': len(indexed),
        'finalized_measured_runs': sum(row['status'] == 'completed_execution' for row in indexed.values()),
        'missing_endpoints': [list(key) for key, row in indexed.items() if row.get('J') is None],
        'measured_action_collisions': sum(row.get('collisions', 0) for row in indexed.values()),
        'measured_episode_returns': sum(row.get('returned', False) for row in indexed.values()),
        'measured_paid_actions': sum(row.get('actual_paid', 0) for row in indexed.values()),
        'engineering_gate': gate_data,
    }


def stage(directory):
    protocol = read(ROOT / 'PROTOCOL.json')
    audit_record = audit(protocol)
    payload = directory / 'payload'
    payload.mkdir(parents=True, exist_ok=False)
    selections = [ROOT / name for name in ('README.md', 'PROTOCOL.json', 'report.py', 'stage_publication.py')]
    selections += [item for section in ('results', 'diagnostics', 'measurement/results', 'runner/regression',
                                       'runner/cold_inputs', 'runner/worker_pids', 'runner/logs')
                   for item in (ROOT / section).glob('*') if item.is_file()]
    selections += [item for item in (ROOT / 'measurement').glob('*')
                   if item.is_file() and item.suffix in ('.json', '.py', '.csv')]
    selections += [item for item in RUNNER.glob('*') if item.is_file() and item.suffix in ('.py', '.json')]
    for source in sorted(set(selections)):
        destination = payload / source.relative_to(ROOT)
        destination.parent.mkdir(parents=True, exist_ok=True)
        if source == METRICS / 'coverage_curve.json':
            body = source.read_bytes()
            packed = destination.with_suffix('.json.gz')
            packed.write_bytes(gzip.compress(body, compresslevel=7, mtime=0))
            assert gzip.decompress(packed.read_bytes()) == body
        else:
            shutil.copyfile(source, destination)
    bundles = []
    for name in protocol['layouts']:
        for mode in protocol['modes']:
            destination = payload / 'episodes' / f'{name}_{mode}.zip'
            destination.parent.mkdir(exist_ok=True)
            bundles.append(run_bundle(RUNNER / 'runs' / name / mode, destination))
    interruptions = RUNNER / 'interrupted_attempts'
    recovery = {}
    if interruptions.is_dir():
        archive = payload / 'episodes/interrupted_attempts.zip'
        recovery = run_bundle(interruptions, archive)
    write(payload / 'completion_audit.json', audit_record)
    write(payload / 'EPISODE_INDEX.json', {'bundles': bundles, 'interrupted_attempts': recovery})
    upload = [upload_item(path, payload) for path in sorted(payload.rglob('*')) if path.is_file()]
    write(directory / 'upload.json', upload)
    print(f'Staged {len(upload)} blobs, {sum(item["size"] for item in upload):,} bytes')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--destination', type=Path, default=ROOT / 'publication')
    stage(parser.parse_args().destination)
