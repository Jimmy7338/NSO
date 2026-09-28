"""Read-only development reporting with all reserved slots retained.

No simulator, mapper, policy execution, quality evaluation or metric imputation
occurs here. Paired descriptive deltas require independently verified endpoints.
"""
import hashlib
import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = 'configs/virtual3d/v43_runtime_protocol_20260921.json'
LAUNCH_ROOT = 'audit_results/v45_development_batch_20260921/launches'
REFERENCE_SHA = {
    'DEV_A_00': '96d00c04384aefbecd55a6a81dcdfc16a432fae47e8c2761204388b5a4cd4326',
    'DEV_C_00': 'a4c99ff26ef19fde5be5cd5c67fd00e6ea4152e2acc42e432f0c5e33ade8af4c',
}


def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _read(path):
    def pairs(rows):
        result = {}
        for key, value in rows:
            if key in result:
                raise ValueError('duplicate JSON field: '+key)
            result[key] = value
        return result
    def invalid(value):
        raise ValueError('nonfinite JSON constant: '+value)
    path = Path(path)
    if path.stat().st_size > 8*1024**2:
        raise ValueError('batch record exceeds 8 MiB bound')
    return json.loads(path.read_text(), object_pairs_hook=pairs, parse_constant=invalid)


def _ratio(value, name):
    if (type(value) not in (int, float) or not math.isfinite(value)
            or not 0 <= value <= 1):
        raise ValueError('finite [0,1] ratio required: '+name)
    return float(value)


def _verified_endpoint(entry, run, slot, source_root):
    """Recheck receipt identity and original episode; never trust summary scores."""
    from nso.saved_replay_v44 import load_saved_episode_v44
    root = Path(entry['metadata']['output']).resolve(strict=True)
    verification_path = Path(run['verification_result_path']).resolve(strict=True)
    evaluation_path = Path(run['evaluation_result_path']).resolve(strict=True)
    verification, evaluation = _read(verification_path), _read(evaluation_path)
    digest = run.get('episode_manifest_sha256')
    if (digest != _sha(root/'artifact_manifest.json')
            or run.get('verification_result_sha256') != _sha(verification_path)
            or run.get('evaluation_result_sha256') != _sha(evaluation_path)):
        raise ValueError('endpoint or child result differs from batch-pinned SHA256')
    for path, script in ((verification_path, 'replay_episode_v44.py'),
                         (evaluation_path, 'evaluate_episode_v44.py')):
        command = _read(path.parent/'command.json')
        if (command.get('status') != 'exited' or command.get('returncode') != 0
                or len(command.get('command', [])) < 3
                or Path(command['command'][2]).resolve() != source_root/'scripts'/script):
            raise ValueError('successful independent child command required')
        for name in ('stdout.txt', 'stderr.txt'):
            stream_path = path.parent/name
            row = command['files'][name]
            if stream_path.stat().st_size != row['bytes'] or _sha(stream_path) != row['sha256']:
                raise ValueError('child command stream hash mismatch')
        lines = [line for line in (path.parent/'stdout.txt').read_text().splitlines() if line.strip()]
        if not lines or json.loads(lines[-1]) != _read(path):
            raise ValueError('saved result differs from independent child stdout')
    episode = load_saved_episode_v44(root, source_root=source_root, expected_manifest_sha256=digest)
    if (not episode.eligible_study_episode or episode.started['run_id'] != entry['run_id']
            or episode.started['slot'] != slot or episode.status != entry['status']):
        raise ValueError('endpoint is not the reserved development episode')
    if (verification.get('schema') != 'v44.saved_observation_verification.v1'
            or verification.get('status') != 'verified'
            or verification.get('prediction_verified') is not True
            or verification.get('eligible_study_episode') is not True
            or verification.get('externally_pinned_manifest') is not True
            or verification.get('source_unchanged_after_verification') is not True
            or verification.get('source_manifest_sha256') != digest
            or verification.get('source_episode_status') != episode.status
            or verification.get('source_task_success') != episode.task_success
            or verification.get('frames_verified') != len(episode.frames)
            or verification.get('physical_actions') != 0 or verification.get('new_worlds') != 0):
        raise ValueError('independent saved-observation verification does not match endpoint')
    if (evaluation.get('schema') != 'v44.saved_prediction_offline_evaluation.v1'
            or evaluation.get('run_id') != entry['run_id'] or evaluation.get('asset_id') != slot['asset_id']
            or evaluation.get('mode') != slot['mode']
            or evaluation.get('original_episode_status') != episode.status
            or evaluation.get('episode_manifest_sha256') != digest
            or evaluation.get('reference_manifest_sha256') != REFERENCE_SHA[slot['asset_id']]
            or evaluation.get('externally_pinned_episode_manifest') is not True
            or evaluation.get('evaluation_source_sha256') != _sha(source_root/'nso/offline_evaluation_v44.py')
            or evaluation.get('all_task_instances_in_macro_denominator') is not True
            or evaluation.get('prediction_roi_cropped') is not False
            or evaluation.get('task_success') != episode.task_success
            or evaluation.get('eligible_successful_development_endpoint') != (
                episode.task_success and slot['mode'] != 'diagnostic')):
        raise ValueError('offline evaluation identity, source, or metric contract mismatch')
    expected_prediction = {name: _sha(root/'prediction'/name) for name in ('mesh.npz', 'occupancy.npz', 'mapper.json')}
    if evaluation.get('input_prediction_sha256') != expected_prediction:
        raise ValueError('evaluation was produced from a different prediction')
    metrics = {key: _ratio(evaluation['metrics'][key], key) for key in ('C_nav', 'Q', 'J_nav')}
    if (not math.isclose(metrics['J_nav'], metrics['C_nav']*metrics['Q'], rel_tol=0, abs_tol=1e-12)
            or metrics['C_nav'] != _ratio(evaluation['coverage']['C_nav'], 'coverage.C_nav')):
        raise ValueError('joint score or coverage component is inconsistent')
    return dict(metrics=metrics, task_success=episode.task_success,
        executed_paid_actions=episode.result['executed_paid_actions'],
        collisions=episode.result['collisions'], reference_manifest_sha256=evaluation['reference_manifest_sha256'],
        public_contract_sha256={name: _sha(root/name) for name in
            ('public_graph.json', 'public_spec.json', 'public_workspace.json')},
        source_sha256=episode.manifest['source_sha256'],
        provenance={str(path): _sha(path) for path in
            (root/'artifact_manifest.json', verification_path, evaluation_path)})


def _paired_rows(rows):
    pairs = []
    for asset in ('DEV_A_00', 'DEV_C_00'):
        selected = {row['mode']: row for row in rows if row['asset_id'] == asset and row['mode'] in ('G', 'S')}
        g, s = selected['G'], selected['S']
        pair = dict(asset_id=asset, G_run_id=g['run_id'], S_run_id=s['run_id'],
            G_status=g['state'], S_status=s['state'], delta_S_minus_G=None,
            both_task_successful=False, independent_test_parent=False)
        if g['state'] == s['state'] == 'verified_endpoint':
            if (g['endpoint']['reference_manifest_sha256'] != s['endpoint']['reference_manifest_sha256']
                    or g['endpoint']['public_contract_sha256'] != s['endpoint']['public_contract_sha256']
                    or g['endpoint']['source_sha256'] != s['endpoint']['source_sha256']
                    or g['noise_seed'] != s['noise_seed']):
                pair['comparison_status'] = 'incompatible_paired_contracts'
            else:
                pair['comparison_status'] = 'descriptive_development_pair'
                pair['both_task_successful'] = g['endpoint']['task_success'] and s['endpoint']['task_success']
                pair['delta_S_minus_G'] = {key: s['endpoint']['metrics'][key]-g['endpoint']['metrics'][key]
                                          for key in ('C_nav', 'Q', 'J_nav')}
        else:
            pair['comparison_status'] = 'not_available_without_two_verified_endpoints'
        pairs.append(pair)
    return pairs


def report_batch_v45(batch_summary_path, *, source_root=ROOT):
    source_root = Path(source_root).resolve()
    summary_path = Path(batch_summary_path).resolve(strict=True)
    summary = _read(summary_path)
    if summary.get('schema') != 'v45.development_batch.v1' or not isinstance(summary.get('runs'), list):
        raise ValueError('V45 batch summary required')
    protocol = _read(source_root/PROTOCOL)
    ledger_path = source_root/protocol['ledger_relative_path']
    if ledger_path.exists():
        ledger = _read(ledger_path)
        if ledger.get('schema') != 'v43.development_start_ledger.v1' or ledger.get('maximum_slots') != 5:
            raise ValueError('original five-slot start ledger required')
        entries = ledger['entries']
    else:
        entries = []
    slots = protocol['slots']
    if (len(entries) > 5 or len({entry['run_id'] for entry in entries}) != len(entries)
            or any(entry['run_id'] not in slots for entry in entries)):
        raise ValueError('duplicate or undeclared ledger reservation')
    if (len({run['run_id'] for run in summary['runs']}) != len(summary['runs'])
            or any(run['run_id'] not in slots for run in summary['runs'])):
        raise ValueError('duplicate or undeclared orchestration row')
    by_id = {entry['run_id']: entry for entry in entries}
    runs = {run['run_id']: run for run in summary['runs']}
    batch_sources_verified, batch_source_error = False, None
    sources = summary.get('source_sha256')
    if sources is not None:
        try:
            from nso.development_batch_v45 import source_names_v45
            if not isinstance(sources, dict) or set(sources) != set(source_names_v45(source_root)):
                raise ValueError('complete batch source closure required')
            if any(_sha(source_root/name) != digest for name, digest in sources.items()):
                raise ValueError('batch source changed after child verification')
            batch_sources_verified = True
        except (ValueError, OSError, KeyError) as exc:
            batch_source_error = str(exc)
    rows = []
    for run_id, slot in slots.items():
        row = dict(run_id=run_id, asset_id=slot['asset_id'], mode=slot['mode'], noise_seed=slot['noise_seed'],
            state='not_started', reserved=False, launch_recorded=False, world_created=False,
            episode_status=None, endpoint=None)
        entry, run = by_id.get(run_id), runs.get(run_id, {})
        row['orchestration_status'] = run.get('orchestration_status', 'not_reported')
        claim = source_root/LAUNCH_ROOT/(run_id+'.json')
        if claim.exists():
            try:
                launch = _read(claim)
                if launch.get('run_id') != run_id or launch.get('automatic_retry') is not False:
                    raise ValueError('invalid permanent launch receipt')
                if run.get('launch_receipt_sha256') is not None and run['launch_receipt_sha256'] != _sha(claim):
                    raise ValueError('launch receipt differs from batch SHA256')
                row.update(launch_recorded=True, launch_provenance={str(claim): _sha(claim)})
            except (ValueError, OSError, KeyError) as exc:
                row.update(state='integrity_error', error=str(exc))
                rows.append(row)
                continue
        if entry is None:
            if row['launch_recorded']:
                row['state'] = 'launch_without_reservation_no_retry'
                row['world_created'] = None
            if run.get('world_created') is True or run.get('evaluation_result_path'):
                row.update(state='integrity_error', error='derived outcome has no original start reservation')
        else:
            row.update(reserved=True, world_created=entry.get('world_created') is True,
                       episode_status=entry['status'], state='terminal_without_verified_endpoint')
            if entry['status'] == 'reserved_before_factory':
                row['state'] = 'unfinished_reservation_no_retry'
                row['world_created'] = None
            else:
                try:
                    if entry.get('metadata', {}).get('slot') != slot:
                        raise ValueError('reservation slot identity changed')
                    episode_root = Path(entry['metadata']['output']).resolve(strict=True)
                    terminal = [episode_root/name for name in ('result.json', 'attempt_failure.json', 'integrity_failure.json')
                                if (episode_root/name).is_file() and _sha(episode_root/name) == entry['result_sha256']]
                    if len(terminal) != 1 or _read(terminal[0]).get('status') != entry['status']:
                        raise ValueError('reserved terminal record is missing or changed')
                    row['terminal_provenance'] = {str(terminal[0]): _sha(terminal[0])}
                    if run.get('verification_result_path') and run.get('evaluation_result_path'):
                        if not batch_sources_verified:
                            raise ValueError('batch verifier/evaluator source closure is not verified')
                        row['endpoint'] = _verified_endpoint(entry, run, slot, source_root)
                        row['state'] = 'verified_endpoint'
                except (ValueError, OSError, KeyError, TypeError) as exc:
                    row.update(state='integrity_error', endpoint=None,
                               error=type(exc).__name__+': '+str(exc))
        rows.append(row)
    counts = {state: sum(row['state'] == state for row in rows) for state in
        ('not_started', 'launch_without_reservation_no_retry', 'unfinished_reservation_no_retry',
         'terminal_without_verified_endpoint', 'verified_endpoint', 'integrity_error')}
    pairs = _paired_rows(rows)
    return dict(schema='v45.development_batch_report.v1',
        status='report_with_integrity_errors' if counts['integrity_error'] or batch_source_error else 'reported',
        planned_slots=5, reserved_slots=sum(row['reserved'] for row in rows),
        recorded_launches=sum(row['launch_recorded'] for row in rows),
        batch_sources_verified=batch_sources_verified, batch_source_error=batch_source_error,
        diagnostic_slots=1, autonomous_slots=4, rows=rows, counts=counts, paired_deltas=pairs,
        verified_autonomous_endpoints=sum(row['state'] == 'verified_endpoint' and row['mode'] != 'diagnostic' for row in rows),
        inference='descriptive development pairs only; no confidence interval, p-value, or independent-test claim',
        missing_scores_imputed=False, failures_excluded_from_rows=False,
        semantic_superiority_claim_added=False,
        input_sha256={str(summary_path): _sha(summary_path), str(source_root/PROTOCOL): _sha(source_root/PROTOCOL),
                      **({str(ledger_path): _sha(ledger_path)} if ledger_path.exists() else {})},
        actual_worlds_created_here=0, sensor_queries_here=0, tsdf_integrations_here=0)
