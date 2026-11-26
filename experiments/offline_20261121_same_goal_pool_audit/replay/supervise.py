#!/usr/bin/env python3
"""Freeze and run exactly four prescribed archive recoveries."""

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
from pathlib import Path
import subprocess
import sys
import time

sys.dont_write_bytecode = True
from replay import (B11, COUNTS, HEAD, HERE, PROTOCOL, SOURCE, SOURCE_RUNNER, STUDY,
                    YIELD, read, read_archive, sha)

PROTOCOL_SHA = '3d2d4d8241f8ab6a99ec6f785b76a31fa9f4f3e41ea923c510e347e617b6dfeb'
CONSUMER = STUDY / 'exhaustive.py'


def write_once(path, value):
    with path.open('x') as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write('\n')


def freeze():
    protocol = read(PROTOCOL)
    assert sha(PROTOCOL) == PROTOCOL_SHA and protocol['source_commit'] == HEAD
    assert protocol['frozen_before_expanded_scores']
    assert Path(protocol['source_local']).resolve() == SOURCE
    scope = protocol['scope']
    assert scope['states_per_layout'] == COUNTS and scope['layouts_in_order'] == list(COUNTS)
    assert scope['common_states'] == sum(COUNTS.values()) == 25
    assert not (HERE / 'runs').exists()
    prior_freeze = SOURCE_RUNNER / 'development_freeze.json'
    protected = read(prior_freeze)['files_sha256']
    if any(sha(Path(path)) != digest for path, digest in protected.items()):
        raise ValueError('Published actor source differs from the original developer freeze')
    source_receipt = STUDY / 'SOURCE_FREEZE.json'
    source_freeze = read(source_receipt)
    assert source_freeze['source_commit'] == HEAD
    assert source_freeze['status'] == 'verified_source_archive_read_only'
    assert all(sha(Path(path)) == digest for path, digest in source_freeze['files_sha256'].items())
    previous_local = read(SOURCE / 'PROTOCOL.json')['local_route']
    assert all(value == previous_local[key] for key, value in protocol['frozen_online_fields'].items())
    files = [PROTOCOL, source_receipt, CONSUMER, HERE / 'replay.py', Path(__file__), SOURCE / 'PROTOCOL.json',
             SOURCE / 'publication/receipt.json', SOURCE / 'route_choice.py', prior_freeze,
             SOURCE_RUNNER / 'full_run.py', SOURCE_RUNNER / 'development_supervisor_receipt.json',
             B11 / 'freeze.json', B11 / 'runtime_binding.py', YIELD / 'execute.py']
    independent_index = STUDY / 'verification/archived_states.json'
    files.append(independent_index)
    state_sets = {}
    for layout in COUNTS:
        archives, state_sets[layout] = read_archive(layout)
        for record in archives.values():
            files.extend(record['directory'] / name for name in record['manifest']['files_sha256'])
            files.append(record['directory'] / 'manifest.json')
    derived_rows = [(row['layout'], row['paid_before'], row['source_decision_ordinal'], row['global_call_id'],
                     row['original_source_G_SHA'], row['original_source_S_SHA'])
                    for states in state_sets.values() for row in states]
    verified_rows = [(row['layout'], row['paid_before'], row['ordinal'], row['global_call_id'],
                      row['original_row_G_sha256'], row['original_row_S_sha256'])
                     for row in read(independent_index)['rows']]
    assert derived_rows == verified_rows
    write_once(HERE / 'freeze.json', dict(status='frozen_before_any_expanded_ranking',
        source_commit=HEAD, paid_inputs_are_archived_only=True, counterfactual_actions_authorized=False,
        frozen_unix_s=time.time(), states=state_sets, consumer_path=str(CONSUMER),
        files_sha256={str(path): sha(path) for path in sorted(set(files))}))


def one(layout):
    path = HERE / 'logs' / f'{layout}.log'
    with path.open('x') as stream:
        status = subprocess.run([sys.executable, str(HERE / 'replay.py'), layout], stdout=stream,
                                stderr=subprocess.STDOUT, check=False)
    return dict(layout=layout, process_returncode=status.returncode, log_sha256=sha(path),
                completion_unix_s=time.time())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--freeze', action='store_true')
    args = parser.parse_args()
    if args.freeze:
        freeze()
        return 0
    frozen = read(HERE / 'freeze.json')
    assert list(frozen['states']) == list(COUNTS)
    assert all(sha(Path(path)) == digest for path, digest in frozen['files_sha256'].items())
    (HERE / 'logs').mkdir()
    started = time.monotonic()
    jobs = []
    with ThreadPoolExecutor(max_workers=4) as workers:
        pending = [workers.submit(one, layout) for layout in COUNTS]
        for completed in as_completed(pending):
            row = completed.result()
            jobs.append(row)
            print(json.dumps(row), flush=True)
    jobs.sort(key=lambda j: list(COUNTS).index(j['layout']))
    success = all(job['process_returncode'] == 0 for job in jobs)
    write_once(HERE / 'supervisor_receipt.json', dict(status='completed' if success else 'failed', jobs=jobs,
                    replay_freeze_sha256=sha(HERE / 'freeze.json'), elapsed_s=time.monotonic()-started))
    return 0 if success else 1


if __name__ == '__main__':
    raise SystemExit(main())
