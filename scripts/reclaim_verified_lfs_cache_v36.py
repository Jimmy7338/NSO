#!/usr/bin/env python3
"""Remove one recoverable model download cache after full read-only remote verification."""
import hashlib
from pathlib import Path
import shutil
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from nso.research_evidence_v31 import read, sha, write, write_bytes, freeze, seal, verify_inventory

OID = 'f5cf9c5fbcaafa703ec6c6332fbbe14bf2f374c6fdd0945e01cd06e73766a0bd'
SIZE = 77268959
POINTER = 'trained_models/stage1_slam_local/model_best.slam'
CACHE = ROOT / '.git/lfs/objects' / OID[:2] / OID[2:4] / OID
OUTPUT = ROOT / 'audit_results/v36_lfs_cache_recovery_20260918'


def main():
    if OUTPUT.exists(): raise FileExistsError('exclusive-create recovery record')
    receipt = read('/tmp/nso_v36_remote_lfs_receipt.json')
    assert receipt['status'] == 'passed' and receipt['operation'] == 'read_only_remote_download_streamed_to_hash_not_disk'
    assert receipt['repository'] == 'https://github.com/Jimmy7338/NSO.git'
    assert receipt['oid'] == receipt['sha256'] == OID and receipt['bytes'] == SIZE
    assert receipt['uploaded_model_bytes'] == 0
    pointer = (ROOT / POINTER).read_bytes()
    assert pointer == subprocess.check_output(['git', 'show', 'HEAD:' + POINTER], cwd=ROOT)
    assert ('oid sha256:' + OID) in pointer.decode() and ('size ' + str(SIZE)) in pointer.decode()
    assert CACHE.is_file() and not CACHE.is_symlink() and CACHE.stat().st_size == SIZE
    assert sha(CACHE) == OID
    before = shutil.disk_usage(ROOT).free
    OUTPUT.mkdir()
    write(OUTPUT, OUTPUT / 'remote_read_verification.json', receipt)
    write_bytes(OUTPUT, OUTPUT / 'verify_remote_readonly.py', Path('/tmp/nso_v36_verify_remote_lfs.py').read_bytes())
    write(OUTPUT, OUTPUT / 'before.json', dict(cache=str(CACHE.relative_to(ROOT)), oid=OID,
        size=SIZE, allocated_bytes=CACHE.stat().st_blocks * 512, link_count=CACHE.stat().st_nlink,
        preserved_tracked_pointer=POINTER, pointer_sha256=hashlib.sha256(pointer).hexdigest(),
        all_experiment_evidence_paths_preserved=True, free_bytes=before,
        rejected_action='automatic review rejected optional historical-model upload verification; no upload executed',
        safer_alternative='read-only remote model streamed through SHA256 with zero uploaded model bytes'))
    freeze(OUTPUT, [Path(__file__), ROOT / POINTER, OUTPUT / 'verify_remote_readonly.py'],
        deleted_cache_is_not_a_frozen_live_input=True,
        recovery_command='git lfs fetch origin main --include=trained_models/stage1_slam_local/model_best.slam --exclude=')
    CACHE.unlink()
    assert (ROOT / POINTER).read_bytes() == pointer and not CACHE.exists()
    write(OUTPUT, OUTPUT / 'result.json', dict(status='complete', cached_model_bytes_removed=SIZE,
        cache_path=str(CACHE.relative_to(ROOT)), model_oid=OID, remote_full_bytes_sha256_verified=True,
        tracked_pointer_preserved=True, experiment_files_removed=0, source_files_removed=0,
        uploaded_model_bytes=0, model_requires_fetch_before_future_neural_use=True,
        free_bytes_after=shutil.disk_usage(ROOT).free,
        observed_free_bytes_gain=shutil.disk_usage(ROOT).free-before))
    seal(OUTPUT); verify_inventory(OUTPUT)
    print(read(OUTPUT / 'result.json'))


if __name__ == '__main__':
    main()
