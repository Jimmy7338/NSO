#!/usr/bin/env python3
"""Freeze and launch exactly the four declared single-choice continuations."""

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
from pathlib import Path
import subprocess
import sys
import time

sys.dont_write_bytecode = True
from continue_witness import (AUDIT_REPLAY, AUDIT_STUDY, DEVELOPER, HEAD, HERE, MODES, PROTOCOL,
                              STUDY, TARGETS, read, sha, witness_inputs)

PROTOCOL_SHA = '220a852f3d34cb82383e80ef3d361c4962711a7f39b485c02505fd3cb730ded9'
SOURCE_SHA = 'e99e30c030329506582a0a06ad7efa8ceebeb899bc8a7bb0a4d86ba8b7998de4'


def write_once(path, data):
    with path.open('x') as stream:
        json.dump(data,stream,indent=2,allow_nan=False)
        stream.write('\n')


def freeze():
    protocol = read(PROTOCOL)
    assert protocol['frozen_before_any_new_trajectory'] and protocol['source_commit'] == HEAD
    assert sha(PROTOCOL) == PROTOCOL_SHA and protocol['total_paid_action_budget'] == 256
    assert tuple(protocol['layouts_in_order']) == tuple(TARGETS) and tuple(protocol['modes']) == MODES
    assert protocol['maximum_continuations'] == 4 and not (HERE / 'runs').exists()
    source = STUDY / 'SOURCE_FREEZE.json'
    assert sha(source) == SOURCE_SHA
    bound = read(source)
    assert bound['source_commit'] == HEAD
    assert all(sha(Path(path)) == digest for path,digest in bound['files_sha256'].items())
    files = [PROTOCOL, source, Path(__file__), HERE / 'continue_witness.py',
             AUDIT_STUDY / 'publication/receipt.json', AUDIT_STUDY / 'PROTOCOL.json',
             AUDIT_STUDY / 'exhaustive.py', AUDIT_REPLAY / 'freeze.json', AUDIT_REPLAY / 'replay.py',
             AUDIT_REPLAY / 'supervisor_receipt.json', DEVELOPER / 'PROTOCOL.json',
             DEVELOPER / 'runner/full_run.py', DEVELOPER / 'runner/development_freeze.json',
             DEVELOPER / 'route_choice.py']
    schedule = []
    checkpoints = {}
    bindings = {}
    for layout,target in TARGETS.items():
        path,witness,archives = witness_inputs(layout)
        plan = protocol['pairs'][layout]
        assert path == AUDIT_STUDY / plan['source_state']
        assert target['paid'] == plan['paid_before'] and target['call_id'] == plan['global_call_id']
        assert target['comparison_paid'] == plan['equal_paid_checkpoint']
        assert witness['state_id'] == plan['state_id']
        checkpoints[layout] = plan['checkpoints']
        assert list(sorted(set(checkpoints[layout]))) == checkpoints[layout]
        assert checkpoints[layout][0] == target['paid'] and checkpoints[layout][-1] == 256
        bindings[layout] = dict(state_id=witness['state_id'], witness_file=str(path), witness_sha256=sha(path),
                                paid_before=plan['paid_before'], global_call_id=plan['global_call_id'])
        files += [path, path.parents[1] / 'manifest.json']
        for archived in archives.values():
            files.append(archived['directory'] / 'manifest.json')
            files.extend(archived['directory'] / key for key in archived['manifest']['files_sha256'])
        schedule.extend(dict(layout=layout, mode=mode) for mode in MODES)
    write_once(HERE / 'freeze.json', dict(status='frozen_before_any_new_trajectory', source_commit=HEAD,
        frozen_unix_s=time.time(), schedule=schedule, capture_paid=checkpoints, witness_bindings=bindings,
        files_sha256={str(path):sha(path) for path in sorted(set(files))}))


def one(job):
    layout,mode = job['layout'],job['mode']
    log = HERE / 'logs' / f'{layout}_{mode}.log'
    with log.open('x') as stream:
        process = subprocess.run([sys.executable,str(HERE / 'continue_witness.py'),layout,mode],
                                 stdout=stream,stderr=subprocess.STDOUT,check=False)
    return dict(**job,returncode=process.returncode,completion_unix_s=time.time(),log_sha256=sha(log))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--freeze',action='store_true')
    args = parser.parse_args()
    if args.freeze:
        freeze()
        return 0
    frozen = read(HERE / 'freeze.json')
    assert all(sha(Path(path)) == digest for path,digest in frozen['files_sha256'].items())
    (HERE / 'logs').mkdir()
    started = time.monotonic()
    jobs = []
    with ThreadPoolExecutor(max_workers=4) as workers:
        futures = [workers.submit(one,job) for job in frozen['schedule']]
        for future in as_completed(futures):
            job = future.result()
            jobs.append(job)
            print(json.dumps(job),flush=True)
    success = all(job['returncode'] == 0 for job in jobs)
    jobs.sort(key=lambda row: frozen['schedule'].index({k:row[k] for k in ('layout','mode')}))
    write_once(HERE / 'supervisor_receipt.json', dict(status='completed' if success else 'failed',
        jobs=jobs, runner_freeze_sha256=sha(HERE/'freeze.json'), elapsed_s=time.monotonic()-started))
    return 0 if success else 1


if __name__ == '__main__':
    raise SystemExit(main())
