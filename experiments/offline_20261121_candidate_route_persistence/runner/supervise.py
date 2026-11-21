#!/usr/bin/env python3
"""Freeze and execute just the five declared neutral-G continuations."""

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
from pathlib import Path
import subprocess
import sys
import time

sys.dont_write_bytecode = True
from continue_route import (DIAGNOSTIC_PAID, HERE, MIDPOINT, NATIVE_CAP, PRIOR, PRIOR_HEAD,
                            PRIOR_RUNNER, PROTOCOL, SCHEDULE, read, recover_inputs, sha)


def write_once(path, data):
    with path.open('x') as stream:
        json.dump(data, stream, indent=2)
        stream.write('\n')


def freeze(protocol):
    assert not (HERE / 'runs').exists()
    assert sha(PROTOCOL) == '28565997c4f60cc1cb1dccf4161a969a2e4ee083ed17ca504e20e7e125be926f'
    assert protocol['frozen_before_new_terminal_outcomes'] and protocol['source_commit'] == PRIOR_HEAD
    assert protocol['native_total_paid_cap'] == NATIVE_CAP
    assert tuple(n for n in DIAGNOSTIC_PAID if n != MIDPOINT) == tuple(protocol['optional_diagnostic_checkpoints'])
    assert tuple((layout, candidate) for layout, ids in protocol['new_routes'].items() for candidate in ids) == SCHEDULE
    assert Path(protocol['source_local']).resolve() == PRIOR
    published = PRIOR / 'publication/receipt.json'
    assert read(published)['head'] == PRIOR_HEAD and read(published)['earlier_experiments_unchanged']
    paths = [PROTOCOL, Path(__file__), HERE / 'continue_route.py', published,
             PRIOR / 'PROTOCOL.json', PRIOR_RUNNER / 'execute.py', PRIOR_RUNNER / 'freeze.json',
             PRIOR_RUNNER / 'supervisor_receipt.json']
    short_recovery = {}
    for layout, candidate in SCHEDULE:
        source = recover_inputs(layout, candidate)
        paths.append(source['directory'] / 'manifest.json')
        paths.extend(source['directory'] / path for path in source['files'])
        short_recovery[f'{layout}/{candidate}'] = {name: shot['actual_paid']
                                                   for name, shot in source['checkpoints'].items()}
    receipt = dict(status='frozen_before_five_new_continuations', at_unix_s=time.time(),
                   new_schedule=[dict(layout=layout, candidate_id=candidate) for layout, candidate in SCHEDULE],
                   archived_comparators=protocol['archived_full_routes'], source_commit=PRIOR_HEAD,
                   native_cap=NATIVE_CAP, diagnostics=list(DIAGNOSTIC_PAID),
                   short_snapshot_paid=short_recovery,
                   files_sha256={str(path): sha(path) for path in sorted(set(paths))})
    write_once(HERE / 'freeze.json', receipt)


def run_one(layout, candidate):
    path = HERE / 'logs' / f'{layout}_{candidate}.log'
    with path.open('x') as stream:
        status = subprocess.run([sys.executable, str(HERE / 'continue_route.py'), layout, str(candidate)],
                                stdout=stream, stderr=subprocess.STDOUT, check=False)
    return dict(layout=layout, candidate_id=candidate, process_returncode=status.returncode,
                log_sha256=sha(path), completion_unix_s=time.time())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--freeze', action='store_true')
    parser.add_argument('--workers', type=int, default=5, choices=(3, 4, 5))
    args = parser.parse_args()
    protocol = read(PROTOCOL)
    if args.freeze:
        freeze(protocol)
        return
    frozen = read(HERE / 'freeze.json')
    assert all(sha(Path(path)) == expected for path, expected in frozen['files_sha256'].items())
    assert tuple((row['layout'], row['candidate_id']) for row in frozen['new_schedule']) == SCHEDULE
    (HERE / 'logs').mkdir()
    begin = time.monotonic()
    with ThreadPoolExecutor(max_workers=args.workers) as workers:
        active = [workers.submit(run_one, layout, candidate) for layout, candidate in SCHEDULE]
        finished = []
        for completed in as_completed(active):
            result = completed.result()
            finished.append(result)
            print(json.dumps(result), flush=True)
    results = sorted(finished, key=lambda row: SCHEDULE.index((row['layout'], row['candidate_id'])))
    success = all(row['process_returncode'] == 0 for row in results)
    write_once(HERE / 'supervisor_receipt.json', dict(status='completed' if success else 'failed',
        jobs=results, workers=args.workers, elapsed_s=time.monotonic() - begin,
        runner_freeze_sha256=sha(HERE / 'freeze.json')))
    return 0 if success else 1


if __name__ == '__main__':
    raise SystemExit(main())
