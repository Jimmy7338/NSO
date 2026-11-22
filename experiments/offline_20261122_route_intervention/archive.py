#!/usr/bin/env python3
"""Verify the executed study and stage immutable compact publication artifacts."""

import csv
import hashlib
import math
import shutil
import zipfile

from study import ARMS, EXECUTION, FREEZE, MEASUREMENT, PROTOCOL, REPLAYS, ROOT, RUNS, read, sha, verify_files, write_new


PREFIX = 'experiments/offline_20261122_route_intervention'
PUBLICATION = ROOT/'publication'
EVALUATED = ROOT/'evaluation/results'


def verify_run(directory, freeze_digest):
    record = directory/'manifest.json'
    data = read(record)
    assert data['status'] == 'completed_execution' and data['runner_freeze_sha256'] == freeze_digest
    for name, digest in data['files_sha256'].items():
        assert sha(directory/name) == digest, directory/name
    assert data['paid_actions'] <= data['native_budget'] == 192
    assert data['actual_closed_loop'] and not data['evaluator_inputs_read_by_actor']
    at = data['checkpoints']['route_end']['actual_paid']
    assert at == data['route_end']['actual_paid'] <= data['paid_actions']
    assert data['checkpoints']['terminal']['actual_paid'] == data['paid_actions']
    return data


def verify_study():
    frozen = read(FREEZE)
    freeze_digest = sha(FREEZE)
    verify_files(frozen['actor_files_sha256'])
    verify_files(frozen['evaluator_files_sha256'])
    phase = read(EXECUTION/'phase_receipt.json')
    clocks = read(EXECUTION/'paired_clock_receipt.json')
    measured = read(EVALUATED/'measured.json')
    assert phase['run_freeze_sha256'] == clocks['run_freeze_sha256'] == measured['runtime_freeze_sha256'] == freeze_digest
    assert len(phase['jobs']) == len(clocks['captures']) == len(frozen['schedule']) == 4
    assert measured['measurement_protocol_sha256'] == sha(MEASUREMENT)
    assert measured['evaluator_sha256'] == sha(ROOT/'evaluation/evaluate.py')
    assert frozen['protocol_sha256'] == sha(PROTOCOL)
    for relative, digest in measured['surface_mask_files_sha256'].items():
        assert sha(EVALUATED/relative) == digest
    records = {(row['layout'], row['arm']): row for row in phase['jobs']}
    captured = {(row['layout'], row['arm']): row for row in clocks['captures']}
    endpoints = {}
    with (EVALUATED/'paired_primary.csv').open(newline='') as stream:
        table = list(csv.DictReader(stream))
    assert len(table) == len(measured['cases']) == 2
    for item in frozen['schedule']:
        layout, arm, route = item['layout'], item['arm'], item['route_id']
        key = layout, arm
        directory = RUNS/layout/arm
        job = records[key]
        assert job['retcode'] == 0 and job['route_id'] == route and job['started_unix_s'] >= frozen['frozen_unix_s']
        receipt_digest = sha(directory/'manifest.json')
        assert receipt_digest == job['record_sha256'] == captured[key]['native_manifest_sha256']
        endpoint = verify_run(directory, freeze_digest)
        assert read(directory/'intervention.json')['route_id'] == route
        assert receipt_digest == measured['cases'][layout]['route_geometry'][arm]['run_manifest_or_failure_sha256']
        endpoints[key] = endpoint

    for row in table:
        layout = row['layout']
        case = measured['cases'][layout]
        target = case['primary_target']
        branches = [endpoints[layout, arm] for arm in ARMS]
        dirs = [RUNS/layout/arm for arm in ARMS]
        assert sha(dirs[0]/'fork_ready.json') == sha(dirs[1]/'fork_ready.json')
        max_clock = max(branch['route_end']['actual_paid'] for branch in branches)
        assert max_clock == case['shared_route_evaluation_paid'] == int(row['equal_clock_paid'])
        for arm, branch in zip(ARMS, branches, strict=True):
            clock_row = captured[layout, arm]
            assert clock_row['requested_paid'] == max_clock
            if clock_row['status'] == 'exact_capture_replay':
                replay = REPLAYS/layout/arm/f'paid_{max_clock}'
                assert sha(replay/'manifest.json') == clock_row['receipt_sha256']
                recovered = verify_run(replay, freeze_digest)
                assert recovered['original_full_trace_verified'] and recovered['extra_capture_paid'] == max_clock
            if clock_row['status'] == 'native_terminal_before_clock':
                assert branch['paid_actions'] < max_clock
            else:
                match = case['arms'][arm]['equal_clock']
                assert match['receipt']['actual_paid'] == max_clock
            terminal = case['arms'][arm]['terminal']
            assert math.isclose(terminal['J'], terminal['C_free'] * terminal['F1_scene_05cm'], abs_tol=1e-12)
        for stamp in ('equal_clock', 'terminal'):
            differences = case['pair'].get(stamp)
            if differences is not None:
                category, geometry = [differences[target][f'{arm}_only_new']['area_m2'] for arm in ARMS]
                assert math.isclose(category - geometry, float(row[f'{stamp}_category_minus_geometry_m2']), abs_tol=1e-12)
    return endpoints, clocks, measured


def archive_tree(directory, target, exclude_meshes=False):
    included, local = {}, {}
    with zipfile.ZipFile(target, 'x', compression=zipfile.ZIP_DEFLATED, compresslevel=7) as bundle:
        for path in sorted(item for item in directory.rglob('*') if item.is_file()):
            relative = path.relative_to(directory).as_posix()
            info = {'bytes': path.stat().st_size, 'sha256': sha(path)}
            if exclude_meshes and path.suffix == '.ply':
                local[relative] = info
            else:
                bundle.write(path, relative)
                included[relative] = info
    with zipfile.ZipFile(target) as bundle:
        assert bundle.testzip() is None
        assert all(hashlib.sha256(bundle.read(relative)).hexdigest() == item['sha256']
                   for relative, item in included.items())
    return {'archive': target.name, 'archive_sha256': sha(target), 'native_directory': str(directory),
            'included_files': included, 'native_meshes_on_workstation': local}


def stage():
    endpoints, clocks, measured = verify_study()
    prereg = read(PUBLICATION/'preregistration_receipt.json')
    assert prereg['verified'] and prereg['base'] == read(PROTOCOL)['source_commit']
    payload = PUBLICATION/'payload'
    payload.mkdir(exist_ok=False)
    files = [ROOT/name for name in ('README.md', 'figures.py', 'summarize.py', 'archive.py',
                                   'publication/preregistration_receipt.json')]
    files.extend(path for directory in (EVALUATED, ROOT/'results') for path in directory.rglob('*') if path.is_file())
    for path in files:
        target = payload/path.relative_to(ROOT)
        target.parent.mkdir(exist_ok=True, parents=True)
        shutil.copyfile(path, target)

    archives = payload/'run_evidence'
    archives.mkdir()
    bundles = [archive_tree(RUNS/layout/arm, archives/f'{layout}_{arm}.zip', exclude_meshes=True)
               | {'layout': layout, 'arm': arm} for layout, arm in endpoints]
    for record in clocks['captures']:
        if record['status'] == 'exact_capture_replay':
            case, arm, paid = [record[key] for key in ('layout', 'arm', 'requested_paid')]
            bundle = archive_tree(REPLAYS/case/arm/f'paid_{paid}', archives/f'{case}_{arm}_capture_{paid}.zip', exclude_meshes=True)
            bundles.append(bundle | {'layout': case, 'arm': arm, 'diagnostic_capture_paid': paid})
    bundles.append(archive_tree(EXECUTION, archives/'launch_and_clock_receipts.zip'))
    write_new(payload/'EVIDENCE_INDEX.json', {'source_commit': prereg['base'], 'preregistered_commit': prereg['head'],
              'runtime_freeze_sha256': sha(FREEZE), 'bundles': bundles,
              'retained_or_missing_clock_receipts': clocks['captures'],
              'complete': len(endpoints) == 4, 'observational_units': list(measured['cases'])})

    listing = []
    for path in sorted(item for item in payload.rglob('*') if item.is_file()):
        data = path.read_bytes()
        listing.append({'stage_path': str(path), 'repo_path': PREFIX+'/'+path.relative_to(payload).as_posix(),
                        'size': len(data), 'sha256': hashlib.sha256(data).hexdigest(),
                        'git_sha1': hashlib.sha1(f'blob {len(data)}\0'.encode()+data).hexdigest()})
    write_new(PUBLICATION/'upload.json', listing)
    print(f'Staged {len(listing)} verified artifacts.')


if __name__ == '__main__':
    stage()
