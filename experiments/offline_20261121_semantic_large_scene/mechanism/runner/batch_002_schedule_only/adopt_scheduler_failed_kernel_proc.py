"""Scheduling-only adoption of two running jobs into a four-worker queue.

Never signals an actor. The original scientific runner and its sealed source,
logs, and attempt manifests stay in place. There is no automatic actor retry.
"""

import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

HERE = Path(__file__).resolve().parent
ORIGINAL = HERE / 'batch_001'
ADOPTION = HERE / 'batch_002_schedule_only'


def read(path):
    return json.loads(path.read_text())


def write(path, value):
    tmp = path.with_suffix('.tmp')
    tmp.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    tmp.replace(path)


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def proc_state(pid):
    try:
        stat = Path(f'/proc/{pid}/stat').read_text().rsplit(')', 1)[1].split()
        return dict(status=stat[0], start_ticks=int(stat[19]))
    except FileNotFoundError:
        return None


def argv(pid):
    return [part.decode() for part in Path(f'/proc/{pid}/cmdline').read_bytes().split(b'\0') if part]


def key(row):
    return row['layout'], row['arm']


def attempt_dir(row):
    return HERE / 'layouts' / row['layout'] / 'runs' / row['arm'] / 'attempt_001'


def attempts(row):
    return sorted(attempt_dir(row).parent.glob('attempt_*'))


def terminal(row):
    """Terminal state from authoritative artifacts, separate from exit code."""
    out = attempt_dir(row)
    if (out / 'failure.json').is_file():
        return 'failure_artifact'
    manifest = out / 'artifact_manifest.json'
    if manifest.is_file() and 'intervention.json' in read(manifest)['files_sha256']:
        return 'completed_execution_artifact'
    return 'no_terminal_artifact'


def snapshot_old():
    launch = read(ORIGINAL / 'launch.json')
    pid = launch['supervisor_pid']
    proc = proc_state(pid)
    assert proc and proc['status'] == 'T' and str(HERE / 'supervisor.py') in argv(pid)
    s = read(ORIGINAL / 'status.json')
    assert s['supervisor_pid'] == pid and s['concurrency'] == 2 and s['status'] == 'running'
    new, live, done = s['jobs_pending'], s['jobs_running'], s['jobs_completed']
    frozen = read(HERE / 'freeze.json')
    assert sorted(key(r) for r in new + live + done) == sorted(key(r) for r in frozen['trial_schedule'])
    direct = Path(f'/proc/{pid}/task/{pid}/children').read_text().split()
    assert {int(c) for c in direct} == {r['pid'] for r in live}, 'Atomic status does not match child set'
    for r in live:
        p = proc_state(r['pid'])
        assert p and p['status'] != 'Z'  # old process will account for these under its own schedule on rollback
        cmd = argv(r['pid'])
        assert cmd[1:] == [str(HERE / 'run.py'), 'run', '--layout', r['layout'], '--arm', r['arm']]
        assert os.getsid(r['pid']) == r['pid'], 'Actor must be in its own session'
        assert attempts(r) == [attempt_dir(r)]
        assert terminal(r) == 'no_terminal_artifact'
        r.update(process_start_ticks=p['start_ticks'], scheduling_origin='adopted_prior_live_actor',
                 actor_signaled=False)
    for r in new:
        assert not attempts(r), (r, 'Pending actor already ran')
    for r in done:
        assert attempts(r) == [attempt_dir(r)]
        r['terminal_artifact'] = terminal(r)
        assert r['exitcode'] != 0 or r['terminal_artifact'] == 'completed_execution_artifact'
    return s


def take_over():
    from run import verify
    verify()
    ADOPTION.mkdir(exist_ok=False)
    pid = read(ORIGINAL / 'launch.json')['supervisor_pid']
    state = proc_state(pid)
    assert state and state['status'] not in ('Z', 'T') and str(HERE / 'supervisor.py') in argv(pid)
    handed_off = False
    try:
        # Only this process signal targets the scheduler PID. Every simulator
        # child is a separate session and is identified explicitly below.
        os.kill(pid, signal.SIGSTOP)
        for _ in range(50):
            stopped = proc_state(pid)
            if stopped and stopped['status'] == 'T':
                break
            time.sleep(.02)
        snapshot = snapshot_old()
        write(ADOPTION / 'original_scheduler_snapshot.json', snapshot)
        write(ADOPTION / 'adoption.json', dict(status='validated_scheduler_partition', old_scheduler_pid=pid,
              scheduler_start_ticks=state['start_ticks'], new_scheduler_pid=os.getpid(),
              original_scientific_freeze_sha256=sha(HERE / 'freeze.json'), scheduling_script_sha256=sha(Path(__file__)),
              original_status_sha256=sha(ORIGINAL / 'status.json'),
              snapshot_sha256=sha(ADOPTION / 'original_scheduler_snapshot.json'),
              reason='E1 actor jobs ended; two additional single-thread slots are available.',
              admitted_concurrency=4, no_scientific_run_changes=True, actor_signals=[]))
        assert proc_state(pid) == dict(status='T', start_ticks=state['start_ticks'])
        os.kill(pid, signal.SIGKILL)  # single scheduling PID, never process group or actor
        handed_off = True
    finally:
        p = proc_state(pid)
        if not handed_off and p and p['start_ticks'] == state['start_ticks'] and p['status'] == 'T':
            os.kill(pid, signal.SIGCONT)
    return snapshot


def supervisor(snapshot):
    prior = snapshot['jobs_running']
    pending = snapshot['jobs_pending']
    completed = snapshot['jobs_completed']
    running = []
    while pending or prior or running:
        for row in prior[:]:
            p = proc_state(row['pid'])
            if p is None or p['status'] == 'Z' or p['start_ticks'] != row['process_start_ticks']:
                completed.append(dict(row, exitcode=None, exit_status='unavailable_not_process_parent',
                                      terminal_artifact=terminal(row), finished=time.time()))
                prior.remove(row)
        for proc, row in running[:]:
            code = proc.poll()
            if code is not None:
                completed.append(dict(row, exitcode=code, terminal_artifact=terminal(row), finished=time.time()))
                running.remove((proc, row))
        while pending and len(running) + len(prior) < 4:
            row = pending.pop(0)
            if attempts(row):
                completed.append(dict(row, exitcode=None, terminal_artifact='not_launched_existing_attempt_conflict'))
                continue
            log = ADOPTION / f"{row['layout']}_{row['arm']}.log"
            with log.open('xb') as output:
                proc = subprocess.Popen([sys.executable, str(HERE / 'run.py'), 'run', '--layout', row['layout'],
                                         '--arm', row['arm']], stdout=output, stderr=subprocess.STDOUT,
                                        start_new_session=True)
            running.append((proc, dict(row, pid=proc.pid, log=str(log), started=time.time(),
                                       scheduling_origin='new_four_worker_slot')))
        working = prior + [row for _, row in running]
        write(ADOPTION / 'status.json', dict(status='running' if pending or working else 'finished',
              jobs_pending=pending, jobs_running=working, jobs_completed=completed, concurrency=4,
              numerical_library_threads=1, supervisor_pid=os.getpid(), original_batch_immutable=True))
        if pending or working:
            time.sleep(1.)


if __name__ == '__main__':
    supervisor(take_over())
