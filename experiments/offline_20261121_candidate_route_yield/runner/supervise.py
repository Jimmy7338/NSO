#!/usr/bin/env python3
"""Freeze and run the sixteen declared candidates on up to four CPU workers."""

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time

HERE = Path(__file__).resolve().parent
PROTOCOL = HERE.parent / 'PROTOCOL.json'


def read(path):
    return json.loads(path.read_text())


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_once(path, data):
    with path.open('x') as stream:
        json.dump(data, stream, indent=2)
        stream.write('\n')


def freeze(tasks):
    paths = [PROTOCOL, HERE / 'execute.py', Path(__file__)]
    write_once(HERE / 'freeze.json', dict(status='frozen_before_any_new_route_outcome',
        frozen_unix_s=time.time(), tasks=[dict(layout=x, candidate_id=y) for x, y in tasks],
        files_sha256={str(path): sha(path) for path in paths}))


def run_task(layout, candidate_id):
    logfile = HERE / 'logs' / f'{layout}_{candidate_id}.log'
    with logfile.open('x') as log:
        result = subprocess.run([sys.executable, str(HERE / 'execute.py'), layout, str(candidate_id)],
                                stdout=log, stderr=subprocess.STDOUT, check=False)
    return dict(layout=layout, candidate_id=candidate_id, process_returncode=result.returncode,
                stdout_sha256=sha(logfile), elapsed_logged_unix_s=time.time())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--freeze', action='store_true')
    args = parser.parse_args()
    protocol = read(PROTOCOL)
    tasks = [(x, y) for x in protocol['layouts_in_order'] for y in protocol['candidates_in_order']]
    if args.freeze:
        freeze(tasks)
        return
    frozen = read(HERE / 'freeze.json')
    assert all(sha(Path(path)) == expected for path, expected in frozen['files_sha256'].items())
    assert frozen['tasks'] == [dict(layout=x, candidate_id=y) for x, y in tasks]
    (HERE / 'logs').mkdir()
    # Spread parent/placement pairs over the same process pool.
    round_robin = [(layout, candidate_id) for candidate_id in protocol['candidates_in_order']
                   for layout in protocol['layouts_in_order']]
    with ThreadPoolExecutor(max_workers=4) as pool:
        futures = [pool.submit(run_task, *task) for task in round_robin]
        statuses = []
        for future in as_completed(futures):
            result = future.result()
            statuses.append(result)
            print(json.dumps(result), flush=True)
    receipt = dict(status='completed' if all(s['process_returncode'] == 0 for s in statuses) else 'failed',
                   freeze_sha256=sha(HERE / 'freeze.json'), jobs=sorted(statuses, key=lambda row: (
                       protocol['layouts_in_order'].index(row['layout']), row['candidate_id'])))
    write_once(HERE / 'supervisor_receipt.json', receipt)
    return 0 if receipt['status'] == 'completed' else 1


if __name__ == '__main__':
    raise SystemExit(main())
