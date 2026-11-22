#!/usr/bin/env python3
"""Audit the three-arm comparison and package only the new evidence."""

import csv
import hashlib
import math
import shutil
import zipfile

from study import EXECUTION, FREEZE, MEASUREMENT, PROTOCOL, ROOT, RUN, branches, read, sha, verify, write


PUBLICATION = ROOT/'publication'
EVAL = ROOT/'evaluation/results'
PREFIX = 'experiments/offline_20261122_route_top_geometry'
ARMS = ('category','geometry_equal_cost','geometry_top')


def audit():
    frozen = read(FREEZE)
    freeze_hash = sha(FREEZE)
    verify(frozen['actor_files_sha256'])
    verify(frozen['evaluator_files_sha256'])
    assert frozen['protocol_sha256']==sha(PROTOCOL) and frozen['measurement_protocol_sha256']==sha(MEASUREMENT)
    exit_record = read(EXECUTION/'exit.json')
    measured = read(EVAL/'measured.json')
    captures = read(EXECUTION/'paired_clock_receipt.json')['captures']
    assert exit_record['exit_code']==0 and exit_record['runner_freeze_sha256']==freeze_hash
    assert frozen['frozen_unix_s'] <= exit_record['started_unix_s'] < exit_record['exit_unix_s']
    assert sha(EXECUTION/'native.log')==exit_record['log_sha256']
    assert exit_record['result_sha256']==sha(RUN/'manifest.json')
    assert (measured.get('runtime_freeze_sha256') or measured.get('run_freeze_sha256') or measured.get('freeze_sha256'))==freeze_hash
    assert set(measured['arms'])==set(ARMS) and len(captures)==4
    by_name = branches()
    records = {arm:read(directory/'manifest.json') for arm,directory in by_name.items()}
    for arm, directory in by_name.items():
        manifest = records[arm]
        assert manifest['status']=='completed_execution' and manifest['native_budget']==192 and manifest['actual_closed_loop']
        assert manifest['paid_actions'] <= 192 and not manifest['evaluator_inputs_read_by_actor']
        assert sha(directory/'fork_ready.json')==frozen['source_fork_ready_sha256']
        verify({str(directory/name):digest for name,digest in manifest['files_sha256'].items()})
        if arm!='geometry_top':
            assert frozen['reused_manifest_sha256'][arm]==sha(directory/'manifest.json')
        assert manifest['paid_actions']==measured['arms'][arm]['terminal']['receipt']['actual_paid']
        terminal=measured['arms'][arm]['terminal']
        assert math.isclose(terminal['J'],terminal['C_free']*terminal['F1_scene_05cm'],abs_tol=1e-12)
    selection = read(RUN/'intervention.json')
    assert selection['route_id']=='native' and selection['no_forced_route'] is True and selection['selected_g_perf_ordinal']==1
    assert records['geometry_top']['runner_freeze_sha256']==freeze_hash
    table = EVAL/'three_arms.csv'
    with table.open(newline='') as stream:
        headings = csv.DictReader(stream).fieldnames
    assert set(ARMS)<=set(headings)
    for key, arm in (('category_minus_top','geometry_top'),('category_minus_equal_cost','geometry_equal_cost')):
        pair = measured['comparisons'][key]
        clock = max(records[name]['route_end']['actual_paid'] for name in ('category',arm))
        assert pair['common_paid']==clock
        for name in ('category',arm):
            capture=next(row for row in captures if row['comparison']==arm and row['arm']==name)
            assert capture['paid']==clock and capture['trace_receipt_sha256']==sha(by_name[name]/'manifest.json')
            if capture['status']=='native_capture':
                assert records[name]['checkpoints'][capture['checkpoint']]['actual_paid']==clock
            elif capture['status']=='native_terminal_before_clock':
                assert records[name]['paid_actions'] < clock
            else:
                assert sha(ROOT/capture['output']/'manifest.json')==capture['result_sha256']
        for phase in ('common','terminal'):
            area = pair.get(f'{phase}_surface')
            if area and area.get('category_minus_comparator_new_m2') is not None:
                left,right = (area[side]['area_m2'] for side in ('category_only_new','comparator_only_new'))
                assert math.isclose(left-right,area['category_minus_comparator_new_m2'],abs_tol=1e-12)
    return by_name, records, measured, captures


def bundle(directory, output, include_meshes=False):
    archive, retained = {}, {}
    with zipfile.ZipFile(output,'x',compression=zipfile.ZIP_DEFLATED,compresslevel=7) as handle:
        for path in sorted(path for path in directory.rglob('*') if path.is_file()):
            relative = path.relative_to(directory).as_posix()
            record = {'sha256':sha(path),'bytes':path.stat().st_size}
            if path.suffix=='.ply' and not include_meshes:
                retained[relative]=record
            else:
                handle.write(path,relative)
                archive[relative]=record
    with zipfile.ZipFile(output) as handle:
        assert handle.testzip() is None
        assert all(hashlib.sha256(handle.read(name)).hexdigest()==item['sha256'] for name,item in archive.items())
    return {'path':str(directory),'archive':output.name,'archive_sha256':sha(output),
            'included':archive,'original_meshes':retained}


def main():
    by_name, records, measured, captures = audit()
    pre = read(PUBLICATION/'preregistration_receipt.json')
    payload = PUBLICATION/'payload'
    payload.mkdir(exist_ok=False)
    files = [ROOT/name for name in ('README.md','summarize.py','archive.py','results/manuscript_zh.md',
                                   'publication/preregistration_receipt.json')]
    files.extend(path for path in EVAL.rglob('*') if path.is_file())
    files.extend([RUN/'manifest.json',RUN/'intervention.json',RUN/'fork_ready.json',
                 EXECUTION/'launch.json',EXECUTION/'exit.json',EXECUTION/'paired_clock_receipt.json'])
    for path in files:
        target=payload/path.relative_to(ROOT)
        target.parent.mkdir(parents=True,exist_ok=True)
        shutil.copyfile(path,target)
    artifacts=payload/'run_evidence'
    artifacts.mkdir()
    collected=[bundle(RUN,artifacts/'native.zip')]
    for diagnostic in {row['output'] for row in captures if row['status']=='exact_capture_replay'}:
        directory=ROOT/diagnostic
        collected.append(bundle(directory,artifacts/f'replay_{directory.parent.name}_{directory.name}.zip'))
    old_receipts={arm:{'run_manifest_sha256':sha(by_name[arm]/'manifest.json'), 'path':str(by_name[arm]),
                      'original_meshes':{name:digest for name,digest in records[arm]['files_sha256'].items()
                                         if name.endswith('.ply')}} for arm in ARMS if arm!='geometry_top'}
    write(payload/'EVIDENCE_INDEX.json',{'source_commit':pre['base'],'preregistered_commit':pre['head'],
          'new_literal_closed_loops':1,'run_freeze_sha256':sha(FREEZE),'native_and_diagnostic_bundles':collected,
          'reused_literal_arms':old_receipts,'paired_clocks':captures,'results_sha256':sha(EVAL/'measured.json'),
          'surface_masks_sha256':sha(EVAL/'surface_masks.npz'),'status':'complete_three_arm_verified'})
    listing=[]
    for path in sorted(path for path in payload.rglob('*') if path.is_file()):
        data=path.read_bytes()
        listing.append({'stage_path':str(path),'repo_path':PREFIX+'/'+path.relative_to(payload).as_posix(),
                        'size':len(data),'sha256':hashlib.sha256(data).hexdigest(),
                        'git_sha1':hashlib.sha1(f'blob {len(data)}\0'.encode()+data).hexdigest()})
    write(PUBLICATION/'upload.json',listing)
    print(f'Staged {len(listing)} verified release files.')


if __name__=='__main__':
    main()
