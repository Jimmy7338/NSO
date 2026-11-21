#!/usr/bin/env python3
"""Execute the root-authorized frozen four-run matrix with three workers."""
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import argparse
import datetime
import hashlib
import json
import os
import subprocess
import threading
import time

HERE = Path(__file__).resolve().parent
BASE = HERE.parents[2]
LOCK = threading.Lock()

def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()

def write(p, data):
    temp = p.with_suffix(p.suffix + '.tmp')
    temp.write_text(json.dumps(data, indent=2, allow_nan=False) + '\n')
    temp.replace(p)

def resource(p):
    return dict(path=str(p), sha256=sha(p))

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--readiness',required=True,type=Path);args=parser.parse_args()
    output = HERE / 'batch_attempt_001'
    output.mkdir(exist_ok=False)
    plan = json.loads((HERE / 'launch_plan.json').read_text())
    readiness = args.readiness.resolve()
    data = json.loads(readiness.read_text())
    assert data['eligible_for_predeclared_four_policy_runs'] is True
    assert data['complete_prefixes']==6 and data['reused_actual_branches']==66
    assert data['actual_selected_target_changes_vs_T1']>0
    assert plan['total_planned_runs'] == len(plan['commands']) == 4
    entries = [dict(layout=row['layout'], arm=row['arm'], status='pending', argv=row['argv']) for row in plan['commands']]
    state = dict(status='running_authorized_frozen_matrix', started_unix_s=time.time(),
        concurrency=3, library_threads_per_process=1, runs=entries)
    auth = dict(status='authorized_by_root_after_complete_fixed_replay_review',
        created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        authorization='Root explicitly directed G/S at frozen T16 on both existing layouts after complete six-prefix review and any actual target change;3 concurrent runs, unchanged sigma/candidates/budget/return, retain all failures.',
        readiness=resource(readiness),
        launch_plan=resource(HERE / 'launch_plan.json'), frozen_protocol=resource(HERE / 'protocol.json'),
        launcher=resource(Path(__file__).resolve()), stop_policy='No new time limit or early stopping; frozen policy controls termination.',
        entries=plan['commands'])
    write(output / 'authorization.json', auth)
    write(output / 'status.json', state)
    env = os.environ.copy()
    for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS'):
        env[key] = '1'
    env['PYTHONUNBUFFERED'] = '1'

    def run(entry):
        key = entry['layout'] + '__' + entry['arm']
        root = HERE / 'layouts' / entry['layout'] / 'runs' / entry['arm']
        prior = set(root.glob('attempt_*')) if root.exists() else set()
        started = time.monotonic()
        with (output / (key + '.log')).open('wb') as log:
            proc = subprocess.Popen(entry['argv'], stdout=log, stderr=subprocess.STDOUT, env=env, cwd=str(HERE))
            with LOCK:
                entry.update(status='running', pid=proc.pid, started_unix_s=time.time(), log=str(output / (key + '.log')))
                write(output / 'status.json', state)
                print(json.dumps(dict(event='started', layout=entry['layout'], arm=entry['arm'], pid=proc.pid)), flush=True)
            code = proc.wait()
        new_attempts = sorted(set(root.glob('attempt_*')) - prior) if root.exists() else []
        summaries = []
        for p in new_attempts:
            for name in ('result.json', 'failure.json'):
                q = p / name
                if q.exists():
                    summaries.append(dict(receipt=resource(q), outcome=json.loads(q.read_text())))
        success = code == 0 and any(x['outcome'].get('status') == 'completed_execution' for x in summaries)
        with LOCK:
            entry.update(status='completed' if success else 'failed', returncode=code, elapsed_s=time.monotonic()-started,
                finished_unix_s=time.time(), attempt_paths=[str(p) for p in new_attempts], outcomes=summaries)
            write(output / 'status.json', state)
            print(json.dumps(dict(event=entry['status'], layout=entry['layout'], arm=entry['arm'], returncode=code,
                elapsed_s=entry['elapsed_s'], completed=sum(r['status']=='completed' for r in entries),
                failed=sum(r['status']=='failed' for r in entries))), flush=True)

    with ThreadPoolExecutor(max_workers=3) as pool:
        list(pool.map(run, entries))
    state.update(status='all_four_attempts_finished', finished_unix_s=time.time(),
        completed_runs=sum(r['status']=='completed' for r in entries), failed_runs=sum(r['status']=='failed' for r in entries))
    write(output / 'status.json', state)
    print(json.dumps({k: state[k] for k in ('status', 'completed_runs', 'failed_runs')}), flush=True)

if __name__ == '__main__':
    main()
