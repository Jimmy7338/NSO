#!/usr/bin/env python3
"""Seal the paper-ready evidence and compact native-run receipts for publication."""

import gzip
import hashlib
import json
from pathlib import Path
import shutil
import zipfile

from study import PROTOCOL, ROOT, RUNNER, WEIGHTS, freeze_path, read, sha, verify_files, write_once


DESTINATION = ROOT / 'publication'
PREFIX = 'experiments/offline_20261122_weighted_geometric_control'


def verify_matrix(split):
    sealed = freeze_path(split)
    freeze = read(sealed)
    verify_files(freeze['files_sha256'] | freeze['supervisor_only_evidence_sha256'])
    expected = {(job['layout'], job['arm']): job for job in freeze['schedule']}
    phase = read(ROOT / 'execution' / split / 'phase_receipt.json')
    assert phase['freeze_sha256'] == sha(sealed)
    assert len(phase['jobs']) == len(expected)
    assert {(job['layout'], job['arm']) for job in phase['jobs']} == set(expected)
    metrics = read(ROOT / 'measurement/results' / split / 'episodes.json')
    assert metrics['run_freeze_sha256'] == sha(sealed)
    episodes = {(row['layout'], row['arm']): row for row in metrics['rows']}
    assert len(metrics['rows']) == len(episodes) == len(expected) and set(episodes) == set(expected)
    sources = []
    for person in phase['jobs']:
        layout, arm = person['layout'], person['arm']
        row = episodes[layout, arm]
        path = RUNNER / 'runs' / split / layout / arm
        if person['literal']:
            terminal = path / ('manifest.json' if person['returncode'] == 0 else 'failure.json')
            assert terminal.is_file(), f'Execution has no retained terminal or failure record: {path}'
            launch = read(ROOT / 'execution' / split / layout / arm / 'launch.json')
            assert all(launch[key] == value for key, value in expected[layout, arm].items())
            assert freeze['frozen_unix_s'] <= launch['started_unix_s']
        else:
            terminal = path / 'alias.json'
            alias = read(terminal)
            assert alias['source_arm'] == person['source_arm'] and alias['run_freeze_sha256'] == sha(sealed)
        assert sha(terminal) == person['source_sha256']
        if row['status'] == 'completed_execution':
            assert row['actual_paid'] <= 192 and abs(row['J'] - row['C_free'] * row['F1_scene_05cm']) <= 1e-10
            archive = path.parent / row['physical_source_arm']
            assert sha(archive / 'manifest.json') == row['source_manifest_sha256']
            assert sha(archive / 'terminal/native_mesh.ply') == row['terminal_mesh_sha256']
            assert sha(archive / 'steps.json') == row['charged_steps_sha256']
        sources.append((path, person['literal']))
    return phase, episodes, sources


def pack(archive, target, *, retain_native_meshes=False, excluded_names=()):
    visible, retained_here = {}, {}
    with zipfile.ZipFile(target, 'x', compression=zipfile.ZIP_DEFLATED, compresslevel=7) as zipped:
        for path in sorted(p for p in archive.rglob('*') if p.is_file()):
            if path.name in excluded_names:
                continue
            relative = path.relative_to(archive).as_posix()
            record = {'sha256': sha(path), 'bytes': path.stat().st_size}
            if retain_native_meshes and path.suffix == '.ply':
                retained_here[relative] = record
            else:
                zipped.write(path, relative)
                visible[relative] = record
    with zipfile.ZipFile(target) as zipped:
        assert not zipped.testzip()
        assert all(hashlib.sha256(zipped.read(name)).hexdigest() == record['sha256'] for name, record in visible.items())
    return {'original_path': str(archive), 'archive': target.name, 'archive_sha256': sha(target),
            'released_files': visible, 'native_meshes_on_workstation': retained_here}


def stage():
    dev, dev_rows, dev_sources = verify_matrix('development')
    heldout, rows, sources = verify_matrix('holdout')
    weights = read(WEIGHTS)
    assert weights['has_not_used_holdout'] and heldout['freeze_sha256'] == sha(freeze_path('holdout'))
    holdout_freeze = read(freeze_path('holdout'))
    assert weights['sealed_unix_s'] <= holdout_freeze['frozen_unix_s']
    assert holdout_freeze['weight_freeze_sha256'] == sha(WEIGHTS)
    decisions = read(ROOT / 'decision/heldout_decision_audit.json')
    validate_decision = {str(ROOT / 'decision' / file): digest for file, digest in decisions['output_files_sha256'].items()}
    verify_files(validate_decision)
    assert decisions['frozen_weight_file_sha256'] == sha(WEIGHTS)
    status = read(ROOT / 'results/engineering_verdict.json')
    assert status['route_replication_gate'] == decisions['route_replication_engineering_gate']

    release = DESTINATION / 'payload'
    release.mkdir(parents=True, exist_ok=False)
    files = [ROOT / name for name in ('README.md','study.py','pipeline.py','report.py','archive.py',
             'dev_freeze.json','holdout_freeze.json','fit_weights.json','runner/weighted_run.py',
             'runner/weighted_score.py','runner/local_parity.py','runner/local_parity/receipt.json',
             'scenes/prepare_scenes.py','scenes/seal_scenes.py','scenes/reference_protocol.json',
             'measurement/evaluate.py','publication/preregistration_receipt.json')]
    for location in ('decision', 'results', 'measurement/results/development','measurement/results/holdout',
                     'runner/local_parity', 'execution/preflight_attempt_001'):
        files.extend(p for p in (ROOT / location).glob('*') if p.is_file() and not p.name.endswith('.pyc'))
    files.extend(ROOT / 'execution' / s / 'phase_receipt.json' for s in ('development','holdout'))
    for path in sorted(set(files)):
        destination = release / path.relative_to(ROOT)
        destination.parent.mkdir(parents=True, exist_ok=True)
        if path.name in ('paths.json','paths.csv','checkpoints.json'):
            content = path.read_bytes()
            packed = destination.with_suffix(destination.suffix + '.gz')
            packed.write_bytes(gzip.compress(content, compresslevel=7, mtime=0))
            assert gzip.decompress(packed.read_bytes()) == content
        else:
            shutil.copyfile(path, destination)

    output = release / 'run_evidence'
    output.mkdir()
    bundles = []
    for original, literal in [*dev_sources,*sources]:
        name = '_'.join(original.parts[-3:]) + '.zip'
        bundles.append(pack(original, output / name, retain_native_meshes=True) | {'literal_native_execution': literal})
    archived_failed = ROOT / 'execution/preflight_attempt_001'
    bundles.append(pack(archived_failed, output / 'first_preaction_attempt.zip'))
    for split in ('development', 'holdout'):
        bundles.append(pack(ROOT / 'execution' / split, output / f'{split}_launches.zip',
                            excluded_names=('lease.lock', 'phase.lock')))
    for fixture in ('prepared', 'precomputed_assets', 'evaluator_references'):
        bundles.append(pack(ROOT / 'scenes' / fixture, output / f'scenes_{fixture}.zip'))
    write_once(release / 'EVIDENCE_INDEX.json', {'bundles': bundles})
    write_once(release / 'completion_audit.json', {'status':'verified_frozen_192_execution',
               'preregistration': read(ROOT/'publication/preregistration_receipt.json')['head'],
               'dev_registered': len(dev_rows), 'holdout_registered': len(rows),
               'dev_observed_endpoints': sum(r['status']=='completed_execution' for r in dev_rows.values()),
               'holdout_observed_endpoints': sum(r['status']=='completed_execution' for r in rows.values()),
               'holdout_literal_runs': sum(literal for _, literal in sources),
               'development_freeze_sha256': dev['freeze_sha256'], 'heldout_freeze_sha256': heldout['freeze_sha256'],
               'weight_freeze_sha256': sha(WEIGHTS), 'source_protocol_sha256': sha(PROTOCOL),
               'complete_four_building_evidence': len({layout for layout, _ in rows}) == 4,
               'engineering_verdict': status})
    manifest = []
    for file in sorted(p for p in release.rglob('*') if p.is_file()):
        payload = file.read_bytes()
        manifest.append({'stage_path': str(file), 'repo_path': PREFIX+'/'+file.relative_to(release).as_posix(),
                         'size': len(payload), 'sha256': hashlib.sha256(payload).hexdigest(),
                         'git_sha1': hashlib.sha1(f'blob {len(payload)}\0'.encode()+payload).hexdigest()})
    write_once(DESTINATION / 'upload.json', manifest)
    print(len(manifest), 'artifacts staged for publication')


if __name__ == '__main__':
    stage()
