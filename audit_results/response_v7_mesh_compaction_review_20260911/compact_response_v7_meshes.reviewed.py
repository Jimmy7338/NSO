#!/usr/bin/env python3
"""Archive array identities before removing replayed, reproducible TSDF meshes.

Raw sensors, references, prefix meshes and route 0 meshes are always retained.
This implements the storage policy fixed before the V7 candidate acquisitions.
"""
import argparse
import hashlib
import json
from pathlib import Path
import time
import numpy as np

if not __debug__:
    raise RuntimeError('mesh compaction and validation require enabled assertions; do not use Python -O')

FAMILIES = ('storage_shelves', 'ventilation_baffles')
REMOVABLE = {f'{family}/candidate_{candidate:03d}/{stage}_mesh.npz'
             for family in FAMILIES for candidate in range(1, 6)
             for stage in ('arrival', 'final')}


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path, data):
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False) + '\n')
    temporary.replace(path)


def array_identity(array):
    array = np.ascontiguousarray(array)
    return {'dtype': array.dtype.str, 'shape': list(array.shape), 'nbytes': array.nbytes,
            'sha256': hashlib.sha256(array.tobytes()).hexdigest()}


def validate_run_assets(run):
    """Validate original assets, with only a fully witnessed mesh exception."""
    run = Path(run)
    manifest = read(run / 'artifact_hashes.json')
    compact_path = run / 'compaction_manifest.json'
    compact = read(compact_path) if compact_path.exists() else None
    # A full byte-exact restoration restores the original manifest contract.
    # Subsequent replay may then replace verification.json without invalidating
    # the historical compaction witness. A planned/partial removal never gets
    # this exception, even if interrupted before its first unlink.
    if compact is not None and compact.get('status') == 'complete' and all((run / name).is_file() for name in manifest):
        for name, digest in manifest.items():
            assert sha(run / name) == digest, f'asset hash mismatch: {run / name}'
        return {'original_manifest_entries': len(manifest), 'archived_meshes': 0,
                'fully_restored_original_manifest': True,
                'artifact_manifest_sha256': sha(run / 'artifact_hashes.json')}
    removed = {}
    if compact is not None:
        assert compact['schema_version'] == 'response_v7_mesh_compaction/1'
        assert compact['status'] == 'complete'
        assert compact['artifact_manifest_sha256'] == sha(run / 'artifact_hashes.json')
        assert compact['verification_sha256'] == sha(run / 'verification.json')
        verification = read(run / 'verification.json')
        assert verification['status'] == 'passed_full' and verification['passed_full']
        assert verification['artifact_manifest_sha256'] == compact['artifact_manifest_sha256']
        assert verification['branches_checked'] == verification['branches_total'] == 12
        assert not verification['partial'] and verification['raw_hashes_rechecked_after_replay']
        removed = compact['meshes']
        assert set(removed) == REMOVABLE, 'only the fixed uniform mesh set is removable'
        assert set(compact['deleted']) == REMOVABLE
        assert compact['source_archive_sha256'] == sha(run / 'sources.zip')
        assert verification['source_archive_sha256'] == compact['source_archive_sha256']
        assert compact['raw_observations_removed'] == 0
        for name, item in removed.items():
            assert item['file_sha256'] == manifest[name]
            assert set(item['arrays']) == {'vertices', 'triangles'}, 'unexpected mesh schema'
            for identity in item['arrays'].values():
                assert len(identity['sha256']) == 64 and identity['nbytes'] >= 0
    checked = 0
    for name, digest in manifest.items():
        path = run / name
        if path.is_file():
            assert sha(path) == digest, f'asset hash mismatch: {path}'
        else:
            assert name in removed, f'missing non-archived asset: {path}'
        checked += 1
    return {'original_manifest_entries': checked, 'archived_meshes': len(removed),
            'artifact_manifest_sha256': sha(run / 'artifact_hashes.json')}


def compact(run):
    run = run.resolve()
    target = run / 'compaction_manifest.json'
    if target.exists():
        raise FileExistsError(target)
    validate_run_assets(run)
    metadata = read(run / 'metadata.json')
    verification = read(run / 'verification.json')
    assert metadata['status'] == 'complete' and metadata['failures'] == 0
    assert verification['status'] == 'passed_full' and verification['passed_full']
    assert not verification['partial'] and verification['max_branches'] is None
    assert verification['branches_checked'] == verification['branches_total'] == 12
    assert verification['raw_hashes_rechecked_after_replay']
    assert verification['artifact_manifest_sha256'] == sha(run / 'artifact_hashes.json')
    assert verification['source_archive_sha256'] == sha(run / 'sources.zip')
    assert verification['source_sha256'] == metadata['source_sha256']
    config = read(run / 'config.json')
    assert config['storage']['mesh_removal_only_after_independent_replay']
    assert config['storage']['mesh_fast_inspection_route'] == 0
    original = read(run / 'artifact_hashes.json')
    inventory = {}
    for name in sorted(REMOVABLE):
        path = run / name
        assert sha(path) == original[name]
        with np.load(path, allow_pickle=False) as arrays:
            identities = {key: array_identity(arrays[key]) for key in arrays.files}
        assert set(identities) == {'vertices', 'triangles'}
        inventory[name] = {'file_sha256': original[name], 'file_bytes': path.stat().st_size,
                           'arrays': identities}
    record = {'schema_version': 'response_v7_mesh_compaction/1', 'status': 'planned',
              'run': str(run), 'policy': 'all arrival/final meshes for candidate 1..5 in both families',
              'artifact_manifest_sha256': sha(run / 'artifact_hashes.json'),
              'verification_sha256': sha(run / 'verification.json'),
              'source_archive_sha256': sha(run / 'sources.zip'), 'tool_sha256': sha(__file__),
              'raw_observations_removed': 0, 'prefix_and_candidate_0_meshes_retained': True,
              'meshes': inventory, 'deleted': [], 'started_unix_s': time.time()}
    # All array identities are durable before the first unlink. A partial
    # operation stays visibly incomplete; it is not accepted by validation.
    write(target, record)
    for name in sorted(inventory):
        path = run / name
        assert sha(path) == inventory[name]['file_sha256']
        path.unlink()
        record['deleted'].append(name)
        write(target, record)
    record.update(status='complete', completed_unix_s=time.time(),
                  saved_bytes=sum(row['file_bytes'] for row in inventory.values()))
    write(target, record)
    verified = validate_run_assets(run)
    print(json.dumps({'run': str(run), 'saved_bytes': record['saved_bytes'], **verified}), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--validate-only', action='store_true')
    args = parser.parse_args()
    if args.validate_only:
        print(json.dumps(validate_run_assets(args.run)), flush=True)
    else:
        compact(args.run)
