#!/usr/bin/env python3
"""Record the case-0 receipt-write failure and authorize one identical replay."""
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess

ROOT = Path(__file__).resolve().parents[1]
BATCH = ROOT/'audit_results/observed_autonomous_v27_20260916'
CASE = BATCH/'case_00'
OUTPUT = ROOT/'audit_results/observed_autonomous_v27_replay_recovery_20260916'
RUNNER = ROOT/'scripts/run_observed_autonomous_v27.py'
PROTOCOL = ROOT/'docs/research/V27_AUTONOMOUS_REPLAY_RECOVERY_20260916.md'


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    temporary = path.with_name(path.name+'.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False)+'\n')
    os.replace(temporary, path)


def verify_inventory(folder):
    inventory = read(folder/'artifact_hashes.json')
    actual = {str(path.relative_to(folder)) for path in folder.rglob('*')
              if path.is_file() and path.name != 'artifact_hashes.json'}
    if actual != set(inventory):
        raise ValueError('case artifact set changed')
    for name, expected in inventory.items():
        if sha(folder/name) != expected:
            raise ValueError('case artifact changed: '+name)
    return inventory


def main():
    if OUTPUT.exists():
        raise ValueError('recovery evidence already exists')
    active = subprocess.run(['pgrep', '-f', 'run_observed_autonomous_v27.py'],
                            capture_output=True, text=True)
    pids = [int(row) for row in active.stdout.split() if int(row) != os.getpid()]
    if pids:
        raise ValueError('autonomous process still active: '+str(pids))
    manifest = read(BATCH/'manifest.json')
    case = manifest['cases'][0]
    if not (manifest['status'] == 'running' and case['physical_status'] == 'complete'
            and case['replay_status'] == 'running' and manifest['main_attempts_started'] == 1):
        raise ValueError('unexpected stale replay state')
    if sha(RUNNER) != manifest['source_sha256']['scripts/run_observed_autonomous_v27.py']:
        raise ValueError('frozen runner changed')
    inventory = verify_inventory(CASE)
    result = read(CASE/'result.json')
    if result['paid_actions'] != 400 or (CASE/'verification.json').exists():
        raise ValueError('main result or failed replay stage differs')
    if shutil.disk_usage(ROOT).free < 104*1024**2:
        raise OSError('one replay requires 40 MiB capacity plus 64 MiB reserve')
    protected = [Path('/tmp/tmp.C2BDwPWLuR/ota_file/scanner.rootfs.ext4'),
                 Path('/tmp/tmp.sveChZoiCT/ota_file/scanner.rootfs.ext4')]
    if [path.stat().st_size for path in protected] != [548552704, 548552704]:
        raise ValueError('protected OTA evidence changed')
    OUTPUT.mkdir()
    write(OUTPUT/'manifest_before_recovery.json', manifest)
    failure = dict(status='terminal_transcription_of_failed_verification_receipt_write',
        case=0, replay=True, tool_session_id=63950, exit_code=1,
        last_logged_action=400, paid_replay_actions_executed=400,
        terminal_error='OSError: 64 MiB real free-space reserve would be crossed',
        failed_stage='verification.json write after terminal comparison',
        formal_replay_verification_passed=False, quota_refunded=False,
        main_result_unchanged=True, case_inventory_sha256=sha(CASE/'artifact_hashes.json'),
        main_result_sha256=sha(CASE/'result.json'), frozen_runner_sha256=sha(RUNNER),
        source='Codex unified exec session 63950 terminal output; no collector receipt could be written')
    write(OUTPUT/'failed_replay.json', failure)
    cleanup = dict(status='capacity_restored', free_bytes_after=shutil.disk_usage(ROOT).free,
        removed_path='/root/viscanner.udp/build', removed_kind='regenerable build output',
        last_observed_update='2026-09-08 16:30:21 +0800', active_processes_before_removal=[],
        recently_created_other_build_preserved='/root/viscanner.arm-refact-audit/build',
        protected_ota_sizes={str(path): path.stat().st_size for path in protected},
        experiment_data_removed=False, unique_evidence_removed=False)
    write(OUTPUT/'capacity_recovery.json', cleanup)
    history = dict(case=0, retry_limit=1, failed_replay_paid_actions=400,
        failed_replay_session_id=63950, formal_replay_verification_passed=False,
        failure_record=str((OUTPUT/'failed_replay.json').relative_to(ROOT)),
        failure_record_sha256=sha(OUTPUT/'failed_replay.json'),
        recovery_protocol=str(PROTOCOL.relative_to(ROOT)), recovery_protocol_sha256=sha(PROTOCOL),
        manifest_before_sha256=sha(OUTPUT/'manifest_before_recovery.json'),
        case_inventory_sha256=sha(CASE/'artifact_hashes.json'),
        frozen_runner_sha256=sha(RUNNER), free_bytes_before_retry=shutil.disk_usage(ROOT).free)
    updated = json.loads(json.dumps(manifest))
    updated['cases'][0]['replay_status'] = 'unstarted'
    updated.setdefault('operational_recovery_history', []).append(history)
    write(BATCH/'manifest.json', updated)
    write(OUTPUT/'manifest_after_recovery.json', updated)
    write(OUTPUT/'result.json', dict(status='one_identical_case0_replay_retry_authorized',
        failed_replay=history, changed_manifest_fields=['cases[0].replay_status',
        'operational_recovery_history'], main_attempts_unchanged=True,
        frozen_sources_unchanged=True, case_inventory_reverified=True))
    verify_inventory(CASE)
    files = {str(path.relative_to(OUTPUT)): sha(path) for path in OUTPUT.rglob('*')
             if path.is_file() and path.name != 'artifact_hashes.json'}
    write(OUTPUT/'artifact_hashes.json', files)
    print(json.dumps(dict(status='recovered', free_bytes=shutil.disk_usage(ROOT).free,
                          failed_replay_paid_actions=400)), flush=True)


if __name__ == '__main__':
    main()
