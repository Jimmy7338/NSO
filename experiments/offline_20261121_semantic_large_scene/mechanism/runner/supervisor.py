"""Detached controller for the fixed twelve-episode matrix."""

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time

HERE = Path(__file__).resolve().parent


def write(path, data):
    tmp = path.with_suffix('.tmp')
    tmp.write_text(json.dumps(data, indent=2) + '\n')
    tmp.replace(path)


def supervise(batch, concurrent):
    from run import verify
    verify()
    protocol = json.loads((HERE / 'protocol.json').read_text())
    pending = [dict(layout=layout, arm=arm) for layout in protocol['layouts'] for arm in protocol['arms']]
    running, completed = [], []
    while pending or running:
        while pending and len(running) < concurrent:
            row = pending.pop(0)
            log = batch / f"{row['layout']}_{row['arm']}.log"
            command = [sys.executable, str(HERE / 'run.py'), 'run', '--layout', row['layout'], '--arm', row['arm']]
            with log.open('wb') as out:
                proc = subprocess.Popen(command, stdout=out, stderr=subprocess.STDOUT, start_new_session=True)
            row.update(pid=proc.pid, started=time.time(), log=str(log))
            running.append((proc, row))
        for proc, row in list(running):
            exitcode = proc.poll()
            if exitcode is not None:
                completed.append(dict(row, exitcode=exitcode, finished=time.time()))
                running.remove((proc, row))
        write(batch / 'status.json', dict(status='running' if pending or running else 'finished',
             supervisor_pid=os.getpid(), jobs_pending=pending, jobs_running=[row for _, row in running],
             jobs_completed=completed, numerical_library_threads=1, concurrency=concurrent))
        if running:
            time.sleep(1)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=('launch', 'supervise'))
    parser.add_argument('--concurrent', type=int, choices=(1, 2, 3, 4), default=2)
    parser.add_argument('--batch', type=Path, default=HERE / 'batch_001')
    args = parser.parse_args()
    if args.command == 'supervise':
        supervise(args.batch, args.concurrent)
        return
    args.batch.mkdir(exist_ok=False)
    with (args.batch / 'supervisor.log').open('wb') as out:
        proc = subprocess.Popen([sys.executable, str(Path(__file__)), 'supervise', '--batch', str(args.batch),
                                 '--concurrent', str(args.concurrent)], stdout=out, stderr=subprocess.STDOUT,
                                start_new_session=True)
    write(args.batch / 'launch.json', dict(supervisor_pid=proc.pid, launched=time.time()))
    print(json.dumps(dict(batch=str(args.batch), supervisor_pid=proc.pid)))


if __name__ == '__main__':
    main()
