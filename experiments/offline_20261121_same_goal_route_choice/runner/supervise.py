#!/usr/bin/env python3
"""Freeze one declared matrix and execute its complete episodes."""

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
from pathlib import Path
import subprocess
import sys
import time

sys.dont_write_bytecode = True
from full_run import (ARM_MODES, B11, HERE, HOLDOUT, MECHANISM, METHOD, PROTOCOL,
                      ROOT, STRICT, STUDY, YIELD, read, sha)


def schedule_of(protocol, stage):
    group = protocol['development' if stage == 'development' else 'heldout']
    return [(layout, arm) for layout in group['layouts'] for arm in group['modes']]


def write_once(path, payload):
    with path.open('x') as stream:
        json.dump(payload, stream, indent=2)
        stream.write('\n')


def freeze(protocol, stage):
    frozen_path = HERE / f'{stage}_freeze.json'
    assert protocol['declared_before_developer_policy_runs']
    assert sha(PROTOCOL) == '6682840e52662c348c4296573ccee9fbb3baf6aaa594f1ea65b67110bef0c64c'
    assert not (HERE / 'runs' / stage).exists()
    schedule = schedule_of(protocol, stage)
    paths = [PROTOCOL, Path(__file__), HERE / 'full_run.py', METHOD,
             YIELD / 'execute.py', YIELD.parent / 'PROTOCOL.json', B11 / 'freeze.json',
             B11 / 'runtime_binding.py']
    if stage == 'development':
        assert len(schedule) == len(set(schedule)) == 8
        assert {arm for _, arm in schedule} == {'newG', 'newS'}
        prior = ROOT / 'candidate_route_persistence_20261121/publication/receipt.json'
        assert read(prior)['head'] == protocol['source_commit']
        paths += [prior, MECHANISM / 'run.py', MECHANISM / 'freeze.json']
        world_files = [MECHANISM.parent / 'scenes/mechanism_world.py',
                       MECHANISM.parent / 'scenes/freeze.json']
        for layout in protocol['development']['layouts']:
            base = MECHANISM / 'layouts' / layout
            world_files += list((base / 'prepared').glob('*'))
            paths += [base / 'runs/G/attempt_001/intervention.json']
    else:
        release_path = STUDY / 'measurement/development_decision.json'
        release = read(release_path)
        assert release['advance_holdout'] is True
        assert release['runner_freeze_sha256'] == sha(HERE / 'development_freeze.json')
        assert len(schedule) == len(set(schedule)) == 16
        assert {arm for _, arm in schedule} == set(ARM_MODES)
        world_files = [HOLDOUT / 'scene.py', HOLDOUT / 'freeze.json']
        for layout in protocol['heldout']['layouts']:
            base = HOLDOUT / 'prepared' / layout
            world_files.extend(base.glob('*'))
        assert all(sha(Path(path)) == checksum for path, checksum in read(HOLDOUT / 'freeze.json')['files_sha256'].items())
        paths += [release_path, STRICT / 'controller.py', STRICT / 'strict_policy.py', STRICT / 'freeze.json']
    whole_files = sorted(set(paths + world_files))
    write_once(frozen_path, dict(status='frozen_before_any_new_paid_episode', stage=stage,
                    at_unix_s=time.time(),
                    schedule=[dict(layout=layout, arm=arm) for layout, arm in schedule],
                    world_files_sha256={str(path): sha(path) for path in sorted(set(world_files))},
                    files_sha256={str(path): sha(path) for path in whole_files}))


def run_one(stage, layout, arm):
    path = HERE / 'logs' / stage / f'{layout}_{arm}.log'
    with path.open('x') as stream:
        status = subprocess.run([sys.executable, str(HERE / 'full_run.py'), stage, layout, arm],
                                stdout=stream, stderr=subprocess.STDOUT, check=False)
    return dict(layout=layout, arm=arm, process_returncode=status.returncode,
                log_sha256=sha(path), completion_unix_s=time.time())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', choices=('development', 'holdout'))
    parser.add_argument('--freeze', action='store_true')
    parser.add_argument('--workers', type=int, default=8, choices=range(3, 9))
    args = parser.parse_args()
    protocol = read(PROTOCOL)
    if args.freeze:
        freeze(protocol, args.stage)
        return
    frozen_path = HERE / f'{args.stage}_freeze.json'
    frozen = read(frozen_path)
    assert all(sha(Path(path)) == expected for path, expected in frozen['files_sha256'].items())
    schedule = schedule_of(protocol, args.stage)
    assert [(row['layout'], row['arm']) for row in frozen['schedule']] == schedule
    (HERE / 'logs' / args.stage).mkdir(parents=True)
    begin = time.monotonic()
    with ThreadPoolExecutor(max_workers=args.workers) as workers:
        futures = [workers.submit(run_one, args.stage, layout, arm) for layout, arm in schedule]
        done = []
        for future in as_completed(futures):
            result = future.result()
            done.append(result)
            print(json.dumps(result), flush=True)
    results = sorted(done, key=lambda row: schedule.index((row['layout'], row['arm'])))
    success = all(row['process_returncode'] == 0 for row in results)
    write_once(HERE / f'{args.stage}_supervisor_receipt.json', dict(
        status='completed' if success else 'failed', stage=args.stage,
        jobs=results, workers=args.workers, elapsed_s=time.monotonic() - begin,
        runner_freeze_sha256=sha(frozen_path)))
    return 0 if success else 1


if __name__ == '__main__':
    raise SystemExit(main())
