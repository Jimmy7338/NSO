"""Bounded, resumable orchestration of the unchanged V43/V44 development slots.

Only the existing V44 subprocess owns World creation and the V43 start ledger.
Saved-observation recomputation is not a new-World replay. No quality threshold
selects whether the other member of a predeclared G/S pair is attempted.
"""
import fcntl
import hashlib
import json
import os
from pathlib import Path
import selectors
import signal
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = 'configs/virtual3d/v43_runtime_protocol_20260921.json'
STATE_ROOT = 'audit_results/v45_development_batch_20260921'
REFERENCE_ROOT = 'audit_results/v44_evidence_pipeline_20260921/references'
REFERENCES = {
    'DEV_A_00': '96d00c04384aefbecd55a6a81dcdfc16a432fae47e8c2761204388b5a4cd4326',
    'DEV_C_00': 'a4c99ff26ef19fde5be5cd5c67fd00e6ea4152e2acc42e432f0c5e33ade8af4c',
}
ORDER = ('R2_A_diagnostic', 'R3_A_G', 'R3_A_S', 'R4_C_G', 'R4_C_S')
COMPLETE = {'controller_stop', 'controller_blocked', 'budget_exhausted',
            'stopped_without_confirmed_return'}
MAX_STREAM = 256 * 1024
MAX_METADATA = 32 * 1024**2
MAX_ATTEMPTS = 8
HARD_TIMEOUT_S = 900.0


def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _json(path):
    path = Path(path)
    if path.is_symlink() or path.stat().st_size > 2*1024**2:
        raise ValueError('plain bounded JSON file required')
    def unique(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError('duplicate JSON field: '+key)
            result[key] = value
        return result
    return json.loads(path.read_text(), object_pairs_hook=unique,
                      parse_constant=lambda value: (_ for _ in ()).throw(ValueError(value)))


def _bytes(value):
    return (json.dumps(value, sort_keys=True, indent=2, allow_nan=False)+'\n').encode()


def _write(path, content, *, metadata_root):
    path, metadata_root = Path(path), Path(metadata_root)
    if not path.resolve().is_relative_to(metadata_root.resolve()):
        raise ValueError('metadata output escapes its root')
    used = sum(p.stat().st_size for p in metadata_root.rglob('*')
               if p.is_file() and not p.is_relative_to(metadata_root/'episodes'))
    ceiling = MAX_METADATA if path.name == 'summary.json' else MAX_METADATA-MAX_STREAM
    if len(content) > 2*1024**2 or used+len(content) > ceiling:
        raise ValueError('batch metadata byte allowance exceeded')
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('xb') as stream:
        stream.write(content); stream.flush(); os.fsync(stream.fileno())
    parent_fd = os.open(path.parent, os.O_RDONLY)
    try:
        os.fsync(parent_fd)
    finally:
        os.close(parent_fd)


def _save(path, value, *, metadata_root):
    _write(path, _bytes(value), metadata_root=metadata_root)


def _resource(output, peak):
    from env.development_sensor_v41 import storage_report_v41
    return storage_report_v41(output, peak)


def source_names_v45(root=ROOT):
    names = [PROTOCOL, 'configs/virtual3d/v40_scene_protocol_20260920.json',
        'nso/instance_belief_v40.py', 'nso/observed_instances_v41.py',
        'nso/observed_residual_v41.py', 'nso/primitive_navigation_v41.py',
        'nso/development_geometry_v40.py', 'nso/surface_evaluation_v40.py',
        'nso/observed_mapper_v42.py', 'nso/view_quality_v42.py',
        'env/development_sensor_v41.py', 'nso/scene_contract_v40.py', 'utils/rgbd_contract.py',
        'nso/evidence_writer_v44.py', 'nso/saved_replay_v44.py', 'nso/offline_evaluation_v44.py',
        'nso/development_batch_v45.py', 'scripts/run_development_v43.py',
        'scripts/run_development_v44.py', 'scripts/replay_episode_v44.py',
        'scripts/evaluate_episode_v44.py', 'scripts/run_development_batch_v45.py',
        'scripts/verify_development_surface_pipeline_v40.py']
    names += [str(p.relative_to(root)) for p in sorted((Path(root)/'nso').glob('*v43.py'))]
    return sorted(set(names))


def _references(root):
    records = {}
    for asset, expected in REFERENCES.items():
        directory = root/REFERENCE_ROOT/asset
        if _sha(directory/'manifest.json') != expected:
            raise ValueError('presealed reference manifest changed: '+asset)
        manifest = _json(directory/'manifest.json')
        if manifest.get('asset_id') != asset or set(manifest['files']) != {
                'candidate_views.json', 'coverage_domain.npz', 'reference.json', 'surface.npz'}:
            raise ValueError('reference inventory mismatch')
        for name, row in manifest['files'].items():
            path = directory/name
            if path.is_symlink() or path.stat().st_size != row['bytes'] or _sha(path) != row['sha256']:
                raise ValueError('presealed reference content changed: '+asset+'/'+name)
        records[asset] = dict(root=str(directory), manifest_sha256=expected)
    return records


def _ledger(root, protocol):
    path = root/protocol['ledger_relative_path']
    if not path.exists():
        return {}
    data = _json(path)
    if (data.get('schema') != 'v43.development_start_ledger.v1'
            or data.get('maximum_slots') != 5 or len(data.get('entries', [])) > 5):
        raise ValueError('unchanged V43 five-slot ledger required')
    rows = {row['run_id']: row for row in data['entries']}
    if len(rows) != len(data['entries']) or set(rows)-set(ORDER):
        raise ValueError('start ledger duplicate or undeclared run')
    return rows


def bounded_subprocess_v45(command, *, cwd, timeout_s=HARD_TIMEOUT_S,
                           maximum_stream_bytes=MAX_STREAM, file_limit_bytes=32*1024**2):
    """Drain both pipes and await process exit under one deadline; retain prefix."""
    import resource
    def limits():
        resource.setrlimit(resource.RLIMIT_FSIZE, (file_limit_bytes, file_limit_bytes))
    started = time.monotonic()
    capture = {'stdout': bytearray(), 'stderr': bytearray()}
    try:
        proc = subprocess.Popen(command, cwd=cwd, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            start_new_session=True, preexec_fn=limits,
            env=dict(os.environ, OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1',
                     MKL_NUM_THREADS='1', PYTHONDONTWRITEBYTECODE='1'))
    except Exception as exc:
        return dict(returncode=None, status='spawn_failed', stdout=b'', stderr=b'',
                    elapsed_s=time.monotonic()-started, error=dict(type=type(exc).__name__, message=str(exc)))
    reason = None; error = None; killed_at = None
    selector = selectors.DefaultSelector()
    def terminate():
        try:
            os.killpg(proc.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
    try:
        for name in capture:
            stream = getattr(proc, name)
            os.set_blocking(stream.fileno(), False)
            selector.register(stream, selectors.EVENT_READ, name)
        # Closing both pipes does not mean the process has finished.
        while selector.get_map() or proc.poll() is None:
            if killed_at is None and time.monotonic()-started >= timeout_s:
                reason = 'hard_timeout'
            if reason is not None and killed_at is None:
                terminate(); killed_at = time.monotonic()
            if killed_at is not None and time.monotonic()-killed_at > 1.:
                break  # A detached descendant must not keep inherited pipes open forever.
            for key, _ in selector.select(.05):
                chunk = os.read(key.fileobj.fileno(), 65536)
                if not chunk:
                    selector.unregister(key.fileobj); key.fileobj.close(); continue
                room = maximum_stream_bytes-len(capture[key.data])
                capture[key.data].extend(chunk[:room])
                if len(chunk) > room and reason is None:
                    reason = 'output_limit'
    except Exception as exc:
        reason = 'subprocess_io_error'
        error = dict(type=type(exc).__name__, message=str(exc))
        terminate()
    finally:
        for key in list(selector.get_map().values()):
            selector.unregister(key.fileobj); key.fileobj.close()
        selector.close()
        if proc.poll() is None:
            terminate()
        try:
            returncode = proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            returncode = None
            reason = reason or 'process_exit_unconfirmed'
    return dict(returncode=returncode, status=reason or 'exited',
                stdout=bytes(capture['stdout']), stderr=bytes(capture['stderr']),
                elapsed_s=time.monotonic()-started, error=error,
                process_exit_confirmed=returncode is not None)


def _command(command, directory, *, output_root, root, runner, evaluation=False):
    """A started.json without command.json is an interrupted call, never retried."""
    if not directory.resolve().is_relative_to(output_root.resolve()):
        raise ValueError('subprocess record directory escapes output root')
    if directory.exists():
        receipt = _json(directory/'command.json')  # Partial output fails closed.
        if receipt.get('command') != command or receipt.get('status') != 'exited':
            raise ValueError('retained subprocess cannot be retried or rebound')
        for name, row in receipt['files'].items():
            if _sha(directory/name) != row['sha256']:
                raise ValueError('retained subprocess output changed')
        return receipt
    directory.mkdir(parents=True, exist_ok=False)
    _save(directory/'started.json', dict(command=command, automatic_retry=False,
        hard_timeout_s=HARD_TIMEOUT_S, maximum_stream_bytes=MAX_STREAM), metadata_root=output_root)
    try:
        result = runner(command, cwd=root, timeout_s=HARD_TIMEOUT_S,
                        maximum_stream_bytes=MAX_STREAM,
                        file_limit_bytes=MAX_STREAM if evaluation else 32*1024**2)
    except Exception as exc:
        result = dict(status='subprocess_adapter_error', returncode=None, stdout=b'', stderr=b'',
            error=dict(type=type(exc).__name__, message=str(exc)), capture_available=False)
    files = {}
    for name in ('stdout', 'stderr'):
        content = result.pop(name)
        if not isinstance(content, bytes) or len(content) > MAX_STREAM:
            raise ValueError('subprocess adapter violated stream bound')
        filename = name+'.txt'
        _write(directory/filename, content, metadata_root=output_root)
        files[filename] = dict(bytes=len(content), sha256=hashlib.sha256(content).hexdigest())
    if evaluation and (directory/'result.json').is_file():
        path = directory/'result.json'
        if path.is_symlink() or path.stat().st_size > MAX_STREAM:
            raise ValueError('evaluation output violates its bounded plain-file contract')
        files['result.json'] = dict(bytes=path.stat().st_size, sha256=_sha(path))
    receipt = dict(result, command=command, files=files, automatic_retry=False)
    _save(directory/'command.json', receipt, metadata_root=output_root)
    return receipt


def _parse_stdout(directory):
    text = (directory/'stdout.txt').read_text()
    # Native libraries may print preceding diagnostics; the CLI emits JSON last.
    lines = [line for line in text.splitlines() if line.strip()]
    if not lines:
        raise ValueError('subprocess did not produce a JSON result')
    return json.loads(lines[-1], parse_constant=lambda x: (_ for _ in ()).throw(ValueError(x)))


def run_batch_v45(output_root, preflight_only=False):
    return _run_batch_v45(output_root, preflight_only=preflight_only,
                         root=ROOT, resource_probe=_resource, runner=bounded_subprocess_v45)


def _run_batch_v45(output_root, *, preflight_only=False, root=ROOT,
                   resource_probe=_resource, runner=bounded_subprocess_v45):
    """Private dependency injection is for finite subprocess fakes, never World."""
    root, output_root = Path(root).resolve(), Path(output_root).resolve()
    protocol = _json(root/PROTOCOL)
    if tuple(protocol['slots']) != ORDER or protocol['maximum_new_world_slots'] != 5:
        raise ValueError('unchanged five-slot protocol required')
    resource_report = resource_probe(output_root, protocol['expected_batch_peak_bytes'])
    result = dict(schema='v45.development_batch.v1', status='blocked_before_world_creation',
        resource=resource_report, output_root=str(output_root), runs=[],
        automatic_retry=False, new_world_replays=0, semantic_performance_claim=False,
        resource_gate_precedes_side_effects=True, preflight_only=bool(preflight_only))
    if not resource_report['passed']:
        return result
    # The permanent launch journal prevents a timeout before V44 reservation
    # from becoming a silent retry merely by selecting a different output root.
    state_root = root/STATE_ROOT
    journal_probe = resource_probe(state_root, 0)
    # This repository-local directory contains only bounded launch metadata.
    # It may be on a different filesystem from an externally supplied episode
    # output. Requiring a second 10 GiB here would block that valid remedy.
    journal_required = 2*MAX_METADATA
    journal_passed = (journal_probe.get('non_ram_filesystem') is True
                      and journal_probe.get('free_bytes', 0) >= journal_required)
    state_resource = dict(journal_probe, passed=journal_passed,
        required_free_bytes=journal_required, expected_batch_peak_bytes=MAX_METADATA,
        purpose='permanent bounded metadata journal only; never a World output path',
        status='metadata_resources_ready' if journal_passed else 'blocked_metadata_storage',
        reason=None if journal_passed else 'metadata_store_nonpersistent_or_insufficient_space')
    result['launch_journal_resource'] = state_resource
    if not state_resource['passed']:
        result['status'] = 'blocked_before_launch_journal_creation'
        return result
    references = _references(root)
    sources = {name: _sha(root/name) for name in source_names_v45(root)}
    result.update(references=references, source_sha256=sources,
                  protocol_sha256=_sha(root/PROTOCOL), start_ledger=str(root/protocol['ledger_relative_path']))
    ledger = _ledger(root, protocol)
    if preflight_only:
        result.update(status='ready_without_world_creation', existing_ledger_runs=list(ledger))
        return result
    output_root.mkdir(parents=True, exist_ok=True)
    state_root.mkdir(parents=True, exist_ok=True)
    # A nonblocking process lock prevents two orchestrators from racing a slot.
    with (state_root/'batch.lock').open('a+b') as lock:
        try:
            fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            result['status'] = 'batch_already_running'; return result
        attempts = output_root/'orchestrator'
        attempts.mkdir(exist_ok=True)
        occupied = {p.name for p in attempts.iterdir()}
        index = next((n for n in range(MAX_ATTEMPTS) if f'attempt_{n:03d}' not in occupied), None)
        if index is None:
            result['status'] = 'orchestration_attempt_cap_exhausted'; return result
        attempt = attempts/f'attempt_{index:03d}'
        attempt.mkdir()
        result['summary_path'] = str(attempt/'summary.json')
        _save(attempt/'started.json', result, metadata_root=output_root)
        result['status'] = 'batch_completed'
        try:
            for name, sha in sources.items():
                _write(attempt/'source'/name, (root/name).read_bytes(), metadata_root=output_root)
                if _sha(attempt/'source'/name) != sha:
                    raise ValueError('source changed during snapshot')
            for run_id in ORDER:
                if any(_sha(root/name) != sha for name, sha in sources.items()):
                    raise ValueError('sources changed before next slot')
                _references(root)
                slot = protocol['slots'][run_id]
                row = dict(run_id=run_id, episode_root=str(output_root/'episodes'/run_id),
                    episode_status=None, world_created=False, verification_result_path=None,
                    evaluation_result_path=None, episode_manifest_sha256=None,
                    verification_result_sha256=None, evaluation_result_sha256=None,
                    launch_receipt_path=None, launch_receipt_sha256=None, orchestration_status='not_started')
                result['runs'].append(row)
                ledger = _ledger(root, protocol)
                claim = state_root/'launches'/f'{run_id}.json'
                if claim.exists():
                    row.update(launch_receipt_path=str(claim), launch_receipt_sha256=_sha(claim), world_created=None)
                entry = ledger.get(run_id)
                if entry is not None:
                    row['episode_root'] = entry.get('metadata', {}).get('output', row['episode_root'])
                    row.update(episode_status=entry.get('status'), world_created=entry.get('world_created', False))
                    if entry['status'] not in COMPLETE:
                        raise ValueError('reserved, unfinished, or technically failed slot cannot restart: '+run_id)
                elif claim.exists():
                    raise ValueError('retained launch without completed V43 reservation cannot restart: '+run_id)
                else:
                    episode = Path(row['episode_root'])
                    if episode.exists():
                        raise ValueError('orphan episode directory cannot be reused')
                    fresh = resource_probe(output_root, protocol['expected_batch_peak_bytes'])
                    if not fresh['passed']:
                        row['resource'] = fresh
                        result['status'] = 'blocked_before_next_slot'; break
                    claim.parent.mkdir(exist_ok=True)
                    _save(claim, dict(run_id=run_id, output=str(episode), source_sha256=sources,
                        protocol_sha256=result['protocol_sha256'], automatic_retry=False), metadata_root=state_root)
                    row.update(launch_receipt_path=str(claim), launch_receipt_sha256=_sha(claim), world_created=None)
                    command = [sys.executable, '-B', str(root/'scripts/run_development_v44.py'),
                        '--run-id', run_id, '--output-root', str(output_root/'episodes')]
                    launched = _command(command, output_root/'reviews'/run_id/'episode_command',
                        output_root=output_root, root=root, runner=runner)
                    row['episode_command_path'] = str(output_root/'reviews'/run_id/'episode_command'/'command.json')
                    if launched['status'] != 'exited':
                        row['orchestration_status'] = launched['status']
                        raise ValueError('interrupted subprocess retained; no safe retry: '+run_id)
                    entry = _ledger(root, protocol).get(run_id)
                    if entry is None:
                        raise ValueError('subprocess exited without a durable V43 reservation')
                    row.update(episode_status=entry.get('status'), world_created=entry.get('world_created', False))
                episode = Path(row['episode_root']).resolve()
                terminal = _json(episode/'result.json')
                if (entry.get('status') not in COMPLETE or entry.get('status') != terminal.get('status')
                        or entry.get('world_created') is not True
                        or entry.get('result_sha256') != _sha(episode/'result.json')
                        or entry.get('metadata', {}).get('slot') != slot):
                    raise ValueError('complete episode does not bind the durable V43 ledger')
                episode_sha = _sha(episode/'artifact_manifest.json')
                row['episode_manifest_sha256'] = episode_sha
                review = output_root/'reviews'/run_id
                binding = dict(run_id=run_id, episode_root=str(episode), episode_manifest_sha256=episode_sha,
                               reference=references[slot['asset_id']], source_sha256=sources)
                if (review/'binding.json').exists():
                    if _json(review/'binding.json') != binding:
                        raise ValueError('postprocessing inputs changed since first binding')
                else:
                    _save(review/'binding.json', binding, metadata_root=output_root)
                replay_dir = review/'verification'
                command = [sys.executable, '-B', str(root/'scripts/replay_episode_v44.py'),
                           str(episode), '--expected-manifest-sha256', episode_sha]
                replay_call = _command(command, replay_dir, output_root=output_root, root=root, runner=runner)
                replay = _parse_stdout(replay_dir)
                row['verification_result_path'] = str(replay_dir/'result.json')
                if not (replay_dir/'result.json').exists():
                    _save(replay_dir/'result.json', replay, metadata_root=output_root)
                elif (replay_dir/'result.json').read_bytes() != _bytes(replay):
                    raise ValueError('retained verification JSON differs from pinned stdout')
                row['verification_result_sha256'] = _sha(replay_dir/'result.json')
                if (replay_call['status'] != 'exited' or replay_call['returncode'] != 0
                        or replay.get('status') != 'verified' or replay.get('prediction_verified') is not True
                        or replay.get('source_manifest_sha256') != episode_sha
                        or replay.get('eligible_study_episode') is not True
                        or replay.get('source_unchanged_after_verification') is not True):
                    raise ValueError('saved-observation verification did not pass')
                if run_id == ORDER[0] and replay.get('source_task_success') is not True:
                    raise ValueError('R2 diagnostic did not return home; autonomous slots remain unattempted')
                evaluation_dir = review/'evaluation'
                evaluation_path = evaluation_dir/'result.json'
                reference = references[slot['asset_id']]
                command = [sys.executable, '-B', str(root/'scripts/evaluate_episode_v44.py'), 'evaluate',
                    '--episode', str(episode), '--reference', reference['root'],
                    '--reference-manifest-sha256', reference['manifest_sha256'],
                    '--episode-manifest-sha256', episode_sha, '--output', str(evaluation_path)]
                evaluation_call = _command(command, evaluation_dir, output_root=output_root,
                                            root=root, runner=runner, evaluation=True)
                row['evaluation_result_path'] = str(evaluation_path)
                evaluation = _json(evaluation_path)
                row['evaluation_result_sha256'] = _sha(evaluation_path)
                if (evaluation_call['status'] != 'exited' or evaluation_call['returncode'] != 0
                        or _parse_stdout(evaluation_dir) != evaluation
                        or evaluation.get('episode_manifest_sha256') != episode_sha
                        or evaluation.get('reference_manifest_sha256') != reference['manifest_sha256']
                        or evaluation.get('run_id') != run_id
                        or evaluation.get('original_episode_status') != terminal['status']):
                    raise ValueError('offline evaluation failed or did not bind frozen inputs')
                row['world_created'] = True
                row['orchestration_status'] = ('diagnostic_verified' if run_id == ORDER[0] else
                    'development_endpoint_recorded' if replay.get('source_task_success') else
                    'non_success_endpoint_recorded')
                row['episode_manifest_sha256'] = episode_sha
                row['verification_result_sha256'] = _sha(replay_dir/'result.json')
                row['evaluation_result_sha256'] = _sha(evaluation_path)
                # Failure to complete the task is retained, but does not cancel
                # the predeclared paired method. Technical failures do stop.
            changed = [name for name, sha in sources.items() if _sha(root/name) != sha]
            if changed:
                raise ValueError('sources changed during batch: '+','.join(changed))
        except Exception as exc:
            result['status'] = 'batch_stopped_for_audit'
            result['error'] = dict(type=type(exc).__name__, message=str(exc))
            if result['runs'] and result['runs'][-1]['orchestration_status'] == 'not_started':
                result['runs'][-1]['orchestration_status'] = 'audit_required_no_retry'
        try:
            final_ledger = _ledger(root, protocol)
            result['unreserved_runs'] = [run_id for run_id in ORDER if run_id not in final_ledger]
            result['launches_without_reservation'] = [run_id for run_id in result['unreserved_runs']
                if (state_root/'launches'/f'{run_id}.json').exists()]
            result['unattempted_runs'] = [run_id for run_id in result['unreserved_runs']
                if run_id not in result['launches_without_reservation']]
        except Exception as exc:
            result['status'] = 'batch_stopped_for_audit'
            result['final_ledger_error'] = dict(type=type(exc).__name__, message=str(exc))
            result['unreserved_runs'] = result['launches_without_reservation'] = result['unattempted_runs'] = None
        result['all_predeclared_runs_accounted'] = len(result['runs']) == len(ORDER)
        _save(attempt/'summary.json', result, metadata_root=output_root)
        return result
