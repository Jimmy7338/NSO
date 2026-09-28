#!/usr/bin/env python3
"""Recover disk from four remote-verified historical LFS download caches only."""
import json
from pathlib import Path
import shutil
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from nso.research_evidence_v31 import freeze, read, seal, sha, verify_inventory, write

OUTPUT = ROOT / 'audit_results/v37_lfs_cache_recovery_20260920'
EXPECTED = {
    'trained_models/stage2_paper_global/model_best.slam': ('038430df7aafadc54c4ae04ce4270bf6471520f61c6ee2ab6cfd077826fb0e8f', 77268959),
    'trained_models/stage2_paper_global/model_best.local': ('859b1a65d163d9362eb2039a2e4101727b8eba31d7eac0babfbc3b10ac22ce1f', 54318285),
    'pretrained_models/model_best.slam': ('1155bcd9856f721355a2e85c4f4c86724fe7cfb5fe3af3428a8a3ce3f5d32181', 77246019),
    'trained_models/stage1_slam_local/model_best.local': ('4646884d7ef064d7f079e0911abb37260b0a4ec269ba74dac0a1299fc49dbf09', 54318285),
}


def main():
    if OUTPUT.exists():
        raise FileExistsError('exclusive-create recovery; no overwrite')
    candidates = read('/tmp/nso_v37_lfs_four_candidates.json')['candidates']
    assert {row['pointer'] for row in candidates} == set(EXPECTED)
    receipts = []
    for row in candidates:
        oid, size = EXPECTED[row['pointer']]
        assert row['oid'] == oid and row['bytes'] == size
        expected_cache = '.git/lfs/objects/' + oid[:2] + '/' + oid[2:4] + '/' + oid
        assert row['cache_path'] == expected_cache
        cache = ROOT / expected_cache
        assert cache.is_file() and not cache.is_symlink() and cache.stat().st_nlink == 1
        assert cache.stat().st_size == size and sha(cache) == oid
        pointer = (ROOT / row['pointer']).read_bytes()
        assert pointer == subprocess.check_output(['git', 'show', 'HEAD:' + row['pointer']], cwd=ROOT)
        assert ('oid sha256:' + oid) in pointer.decode() and ('size ' + str(size)) in pointer.decode()
        receipt = read(row['remote_receipt'])
        assert receipt['status'] == 'passed' and receipt['operation'] == 'read_only_remote_download_streamed_to_hash_not_disk'
        assert receipt['repository'] == 'https://github.com/Jimmy7338/NSO.git'
        assert receipt['oid'] == receipt['sha256'] == oid and receipt['bytes'] == size
        assert receipt['uploaded_model_bytes'] == 0 and not receipt['credentials_or_signed_URL_saved']
        receipts.append(receipt)
    free_before = shutil.disk_usage(ROOT).free
    before = dict(candidates=candidates, receipts=receipts, free_bytes_before=free_before,
                  scope='Recover known remote-retrievable download caches; preserve all tracked pointers and experiment files',
                  emergency_bootstrap_note='Preexisting free space below 64MiB; write at most16KiB receipt before first cache unlink, then restore normal64MiB reserve.')
    payload = (json.dumps(before, indent=2) + '\n').encode()
    assert len(payload) <= 16 * 1024 and free_before > len(payload) + 1024**2
    OUTPUT.mkdir()
    with (OUTPUT / 'before.json').open('xb') as stream:
        stream.write(payload)
    for row in candidates:
        (ROOT / row['cache_path']).unlink()
    assert shutil.disk_usage(ROOT).free > 64 * 1024**2
    for row in candidates:
        assert not (ROOT / row['cache_path']).exists()
        assert sha(ROOT / row['pointer']) == row['pointer_sha256']
    freeze(OUTPUT, [Path(__file__)] + [ROOT / name for name in EXPECTED],
           deleted_cache_paths_not_frozen_live_inputs=True,
           recover_commands=['git lfs fetch origin main --include=' + name + ' --exclude=' for name in EXPECTED])
    result = dict(status='completed', cache_files_removed=4,
                  cached_model_bytes_removed=sum(size for _, size in EXPECTED.values()),
                  remote_full_SHA256_verified=True, experiment_files_removed=0,
                  tracked_pointers_preserved=True, current_yolo_cache_preserved=True,
                  uploaded_model_bytes=0, free_bytes_after=shutil.disk_usage(ROOT).free,
                  future_neural_use_requires_lfs_fetch=list(EXPECTED))
    write(OUTPUT, OUTPUT / 'result.json', result)
    seal(OUTPUT)
    verify_inventory(OUTPUT)
    print(json.dumps(result))


if __name__ == '__main__':
    main()
