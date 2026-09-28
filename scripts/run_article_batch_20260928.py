#!/usr/bin/env python3
"""Schedule never-started frozen article slots; never retry an attempt.

This administrative wrapper is outside the scientific source closure. The
frozen runner remains responsible for its own reservations, limits and records.
Review and byte-exact archival are deliberately separate postprocessing steps.
"""
import argparse
from contextlib import contextmanager
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import re
import shutil
import signal
import subprocess
import time

ROOT = Path(__file__).resolve().parents[1]
GIB = 1024**3
EPISODE_ALLOWANCE = 64*1024**2
SCRATCH_ALLOWANCE = 256*1024**2
CAPS = {'development': 12, 'main': 96, 'ablation': 24}
RUNNER = 'scripts/run_article_experiment_20260928.py'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def save(path, value):
    """Atomic administrative record, never an experimental ledger."""
    path = Path(path)
    temporary = path.with_name(path.name+'.new')
    with temporary.open('x') as stream:
        json.dump(value, stream, ensure_ascii=False, sort_keys=True, indent=2)
        stream.write('\n'); stream.flush(); os.fsync(stream.fileno())
    os.replace(temporary, path)
    fd = os.open(path.parent, os.O_RDONLY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def within(root, relative):
    value = Path(relative)
    if value.is_absolute() or '..' in value.parts:
        raise ValueError('relative contained path required')
    result = (root/value).resolve()
    if not result.is_relative_to(root):
        raise ValueError('path escapes repository')
    return result


def load_protocol(path, root, expected_sha=None):
    content = Path(path).read_bytes()
    digest = hashlib.sha256(content).hexdigest()
    if expected_sha is not None and digest != expected_sha:
        raise ValueError('protocol changed after batch start')
    protocol = json.loads(content)
    if (protocol.get('schema') != 'article.experiment_protocol.v1'
            or protocol.get('status') != 'frozen'
            or protocol.get('phase') not in CAPS):
        raise ValueError('frozen article protocol required')
    slots = protocol.get('slots', {})
    if not isinstance(slots, dict) or not 1 <= len(slots) <= CAPS[protocol['phase']]:
        raise ValueError('invalid finite declared slots')
    if any(not re.fullmatch(r'[A-Za-z0-9_.-]{1,80}', key) for key in slots):
        raise ValueError('invalid run identifier')
    output = within(root, protocol['output_relative_path'])
    if (Path(protocol['output_relative_path']).parts[:2]
            != ('audit_results', 'article_stage_20260928')
            or len(Path(protocol['output_relative_path']).parts) != 3):
        raise ValueError('isolated article phase output required')
    if protocol['reserve_bytes'] < GIB or not 2*1024**2 <= protocol['maximum_episode_bytes'] <= EPISODE_ALLOWANCE:
        raise ValueError('resource contract differs')
    pins = protocol.get('source_sha256', {})
    if not isinstance(pins, dict) or RUNNER not in pins:
        raise ValueError('frozen runner source pin required')
    for name, pin in pins.items():
        if sha(within(root, name)) != pin:
            raise ValueError('frozen source changed: '+name)
    archive = within(root, protocol['source_archive'])
    if sha(archive) != protocol['source_archive_sha256']:
        raise ValueError('frozen source archive changed')
    return protocol, digest, output


@contextmanager
def exclusive(path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('a+b') as stream:
        try:
            fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise RuntimeError('another batch holds this protocol lock') from error
        try:
            yield
        finally:
            fcntl.flock(stream, fcntl.LOCK_UN)


def ledger_state(phase_output, protocol, protocol_sha):
    """Read-only, fail-closed view of both runner-owned atomic JSON records."""
    path = phase_output/'start_ledger.json'
    local = {}
    if path.exists():
        data = read(path)
        expected = dict(schema='article.start_ledger.v1', protocol_sha256=protocol_sha,
                        phase=protocol['phase'], slots=protocol['slots'])
        if any(data.get(k) != v for k, v in expected.items()):
            raise ValueError('local ledger/protocol binding differs')
        for row in data['entries']:
            run_id = row['run_id']
            if run_id not in protocol['slots'] or run_id in local or not isinstance(row.get('status'), str):
                raise ValueError('unknown/duplicate local ledger entry')
            local[run_id] = row
    registry_path = phase_output.parent/'execution_registry.json'
    registry = read(registry_path) if registry_path.exists() else dict(
        schema='article.execution_registry.v1', phase_caps=CAPS, entries=[])
    if registry.get('schema') != 'article.execution_registry.v1' or registry.get('phase_caps') != CAPS:
        raise ValueError('global registry contract differs')
    seen = set()
    blocked = set()
    phase_count = 0
    for row in registry['entries']:
        if (row.get('phase') not in CAPS or not isinstance(row.get('run_id'), str)
                or not isinstance(row.get('ledger'), str)
                or not isinstance(row.get('protocol_sha256'), str)):
            raise ValueError('unknown global attempt; refusing new launches')
        identity = (row['ledger'], row['run_id'])
        if identity in seen:
            raise ValueError('duplicate global reservation')
        seen.add(identity)
        if row['phase'] == protocol['phase']:
            phase_count += 1
            # Conservatively reject reused run IDs even in a revised output path.
            if row['run_id'] in protocol['slots']:
                blocked.add(row['run_id'])
    return local, blocked, phase_count


def child_environment():
    env = dict(os.environ)
    for name in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS',
                 'NUMEXPR_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS', 'BLIS_NUM_THREADS'):
        env[name] = '1'
    env['PYTHONDONTWRITEBYTECODE'] = '1'
    env['PYTHONUNBUFFERED'] = '1'
    return env


def run_batch(protocol_path, output, *, workers=2, max_wall_seconds=None,
              root=ROOT, command_factory=None, preflight=True,
              free_bytes=None, poll_seconds=.5):
    """Injection hooks exist only for tiny subprocess tests, never CLI options."""
    if type(workers) is not int or not 1 <= workers <= 2:
        raise ValueError('workers must be 1 or 2')
    if max_wall_seconds is not None and (not math.isfinite(max_wall_seconds) or max_wall_seconds <= 0):
        raise ValueError('positive finite launch deadline required')
    root = Path(root).resolve(); protocol_path = Path(protocol_path).resolve()
    output = Path(output).resolve()
    protocol, pin, phase_output = load_protocol(protocol_path, root)
    if output == phase_output or output.is_relative_to(phase_output):
        raise ValueError('batch records must be outside the scientific phase directory')
    control = phase_output.parent/'batch_control'
    control.mkdir(parents=True, exist_ok=True)
    began = time.monotonic()
    env = child_environment()
    python = root/'.venv-3d/bin/python'
    if command_factory is None:
        if not python.exists():
            raise ValueError('cached .venv-3d interpreter unavailable')
        command_factory = lambda run_id: [str(python), '-B', str(root/RUNNER),
                                         '--protocol', str(protocol_path), '--run-id', run_id]
    free_bytes = free_bytes or (lambda: shutil.disk_usage(root).free)
    with exclusive(control/(pin+'.lock')):
        output.mkdir(parents=True, exist_ok=False)
        claim_path = control/(pin+'.launch_claims.json')
        claims = read(claim_path) if claim_path.exists() else dict(
            schema='article.batch_launch_claims.v1', protocol_sha256=pin, entries=[])
        if claims.get('schema') != 'article.batch_launch_claims.v1' or claims.get('protocol_sha256') != pin:
            raise ValueError('batch claim binding differs')
        claimed = {row['run_id'] for row in claims['entries']}
        if len(claimed) != len(claims['entries']) or not claimed <= set(protocol['slots']):
            raise ValueError('unknown/duplicate administrative launch claim')
        report = dict(schema='article.batch_schedule.v1', protocol=str(protocol_path),
            protocol_sha256=pin, scheduler_sha256=sha(Path(__file__)),
            phase=protocol['phase'], started_unix_s=time.time(), workers=workers,
            max_wall_seconds=max_wall_seconds, scratch_allowance_bytes=SCRATCH_ALLOWANCE,
            per_active_episode_allowance_bytes=EPISODE_ALLOWANCE,
            status='running', runs=[], review_and_archive='separate root postprocessing')
        save(output/'batch.json', report)
        active = {}; halted = None; interrupted = []
        old_handlers = {}
        def stop_launches(number, frame):
            interrupted.append(signal.Signals(number).name)
        for number in (signal.SIGINT, signal.SIGTERM):
            old_handlers[number] = signal.signal(number, stop_launches)
        pending = list(sorted(protocol['slots']))
        try:
            if preflight:
                # The frozen validator/source-closure walker does not construct a World.
                code = ('import json,sys; from nso.article_experiment_v1 import validate_protocol,source_hashes; '
                        'p=validate_protocol(json.load(open(sys.argv[1]))); '
                        'assert source_hashes()==p["source_sha256"],"source closure changed"')
                with (output/'preflight.log').open('xb') as stream:
                    checked = subprocess.run([str(python), '-B', '-c', code, str(protocol_path)],
                        cwd=root, env=env, stdout=stream, stderr=subprocess.STDOUT, timeout=90)
                if checked.returncode:
                    raise ValueError('frozen protocol preflight failed; see preflight.log')
            while pending or active:
                for run_id, (process, stream, row) in list(active.items()):
                    code = process.poll()
                    if code is None:
                        continue
                    stream.close(); row.update(returncode=code, ended_unix_s=time.time(), status='process_finished')
                    try:
                        local, _, _ = ledger_state(phase_output, protocol, pin)
                        row['runner_status'] = local.get(run_id, {}).get('status', 'no_local_reservation')
                    except (ValueError, KeyError, OSError) as error:
                        row['ledger_read_error'] = str(error)
                        halted = 'ledger_validation_failed'
                    del active[run_id]; save(output/'batch.json', report)
                if interrupted:
                    halted = 'signal:'+interrupted[-1]
                if max_wall_seconds is not None and time.monotonic()-began >= max_wall_seconds:
                    halted = halted or 'launch_deadline'
                while pending and len(active) < workers and halted is None:
                    if interrupted or (max_wall_seconds is not None and time.monotonic()-began >= max_wall_seconds):
                        halted = 'signal:'+interrupted[-1] if interrupted else 'launch_deadline'
                        break
                    try:
                        load_protocol(protocol_path, root, pin)
                        local, blocked, phase_count = ledger_state(phase_output, protocol, pin)
                    except (ValueError, KeyError, OSError, json.JSONDecodeError) as error:
                        halted = 'validation_failed'; report['validation_error'] = str(error); break
                    run_id = pending.pop(0)
                    reason = ('existing_local:'+local[run_id]['status'] if run_id in local else
                              'existing_global_reservation' if run_id in blocked else
                              'previous_scheduler_launch' if run_id in claimed else
                              'retained_episode_path' if (phase_output/'episodes'/run_id).exists() else
                              'retained_failure_path' if (phase_output/'failures'/(run_id+'.json')).exists() else None)
                    if reason:
                        report['runs'].append(dict(run_id=run_id, status='skipped', reason=reason))
                        save(output/'batch.json', report); continue
                    not_yet_reserved = sum(key not in blocked for key in active)
                    if phase_count+not_yet_reserved >= CAPS[protocol['phase']]:
                        pending.insert(0, run_id); halted = 'cumulative_phase_cap'; break
                    needed = max(GIB, protocol['reserve_bytes'])+(len(active)+1)*EPISODE_ALLOWANCE+SCRATCH_ALLOWANCE
                    available = int(free_bytes())
                    if available < needed:
                        pending.insert(0, run_id); halted = 'resource_blocked'
                        report['resource_gate'] = dict(free_bytes=available, required_bytes=needed,
                            active_processes=len(active), candidate=run_id)
                        break
                    row = dict(run_id=run_id, status='launch_claimed', started_unix_s=time.time(),
                               log=run_id+'.log', free_bytes=available, required_bytes=needed)
                    # Durable before Popen: failure before runner reservation still cannot retry.
                    claims['entries'].append(dict(run_id=run_id, batch=str(output),
                        claimed_unix_s=row['started_unix_s']))
                    save(claim_path, claims); claimed.add(run_id)
                    report['runs'].append(row); save(output/'batch.json', report)
                    stream = (output/row['log']).open('xb')
                    try:
                        process = subprocess.Popen(command_factory(run_id), cwd=root, env=env,
                            stdout=stream, stderr=subprocess.STDOUT, start_new_session=True)
                    except Exception as error:
                        stream.close(); row.update(status='spawn_failed', error=str(error),
                                                   ended_unix_s=time.time())
                    else:
                        row.update(status='running', pid=process.pid)
                        active[run_id] = (process, stream, row)
                    save(output/'batch.json', report)
                if halted:
                    report['status'] = 'draining'; report['stop_reason'] = halted
                    report.setdefault('never_launched', list(pending)); pending.clear()
                    save(output/'batch.json', report)
                if active:
                    time.sleep(poll_seconds)
            report['status'] = 'stopped' if halted else 'complete'
        except Exception as error:
            # Unexpected scheduler failure also drains children; no automatic restart.
            report.update(status='scheduler_failed', error=str(error), never_launched=list(pending))
            for run_id, (process, stream, row) in active.items():
                row.update(returncode=process.wait(), ended_unix_s=time.time(), status='process_finished')
                stream.close()
        finally:
            for number, handler in old_handlers.items():
                signal.signal(number, handler)
            report['failed_subprocesses'] = sum(
                row['status'] == 'spawn_failed' or row.get('returncode', 0) != 0
                for row in report['runs'])
            if report['status'] == 'complete' and report['failed_subprocesses']:
                report['status'] = 'complete_with_failures'
            report['ended_unix_s'] = time.time()
            report['elapsed_s'] = time.monotonic()-began
            save(output/'batch.json', report)
        return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--protocol', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True, help='fresh administrative record directory')
    parser.add_argument('--workers', type=int, choices=(1, 2), default=2)
    parser.add_argument('--max-wall-seconds', type=float,
                        help='stop new launches at this elapsed time; finish active children')
    args = parser.parse_args()
    result = run_batch(args.protocol, args.output, workers=args.workers,
                       max_wall_seconds=args.max_wall_seconds)
    print(json.dumps(result, ensure_ascii=False))
    if result['status'] != 'complete':
        raise SystemExit(1)


if __name__ == '__main__':
    main()
