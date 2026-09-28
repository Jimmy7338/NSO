#!/usr/bin/env python3
"""Exact-byte storage deduplication of one approved, completed-run manifest."""
import argparse
from collections import Counter
import fcntl
import gzip
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import shutil
import stat
import time
import uuid
import zipfile

ROOT = Path(__file__).resolve().parents[1]
AUDIT = ROOT / 'audit_results'
ESTIMATE = AUDIT / 'completed_sensor_duplicate_estimate_20260910.json'
ALLOWED_RUNS = frozenset('''inspection_v4_integration_smoke_20260910
joint_v2_balance_development_20260910 joint_v2_camera_development_20260910
joint_v2_development_20260910 joint_v2_gated_development_20260910 joint_v2_smoke_20260910
joint_v3_channels_generic_development_20260910 joint_v3_channels_occluded_development_20260910
joint_v3_fit_development_20260910 joint_v3_labels_development_20260910
joint_v3_objectness_development_20260910 joint_v3_occluded_development_20260910
joint_v3_posterior_development_20260910 joint_v3_replication_development_20260910
joint_v3_self_visibility_development_20260910 joint_v4_ablation_verified_20260910
joint_v4_feedback_development_20260910 virtual3d_pilot_v1_20260910
virtual3d_smoke_20260910'''.split())


class SkipPair(Exception):
    pass


def sha256(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')


def signature(s):
    # ctime/link-count necessarily change when creating a hard link; data mtime
    # and the remaining identity/ownership fields must remain stable.
    return (s.st_dev, s.st_ino, s.st_size, s.st_mtime_ns, s.st_mode, s.st_uid, s.st_gid)


def snapshot(s):
    return dict(dev=s.st_dev, inode=s.st_ino, size=s.st_size, mtime_ns=s.st_mtime_ns,
                ctime_ns=s.st_ctime_ns, mode=stat.S_IMODE(s.st_mode), uid=s.st_uid,
                gid=s.st_gid, links=s.st_nlink, allocated_bytes=s.st_blocks * 512)


def checked_path(value):
    if not isinstance(value, str) or '\\' in value:
        raise SkipPair('invalid path text')
    parts = PurePosixPath(value).parts
    if (PurePosixPath(value).is_absolute() or str(PurePosixPath(value)) != value
            or any(part in ('.', '..') for part in parts)
            or len(parts) not in (4, 5) or parts[0] != 'eval_results'
            or parts[1] not in ALLOWED_RUNS):
        raise SkipPair('path outside explicit completed-run allowlist')
    if len(parts) == 4:
        if parts[3] not in ('final_mesh.npz', 'maps.npz'):
            raise SkipPair('not an allowed terminal payload')
    elif parts[3] not in ('frames', 'scans') or not re.fullmatch(r'\d{4,}\.npz', parts[4]):
        raise SkipPair('not an allowed numbered sensor payload')
    current = ROOT
    for part in parts:
        current /= part
        s = current.lstat()
        if stat.S_ISLNK(s.st_mode):
            raise SkipPair('symbolic link in payload path')
    if not stat.S_ISREG(s.st_mode):
        raise SkipPair('payload is not a regular file')
    return current, parts[1]


def writable_inodes():
    result = set()
    inaccessible = []
    for entry in os.scandir('/proc'):
        if not entry.name.isdigit():
            continue
        try:
            descriptors = list(os.scandir(Path(entry.path) / 'fd'))
        except (FileNotFoundError, ProcessLookupError):
            continue
        except PermissionError:
            inaccessible.append(entry.name)
            continue
        for descriptor in descriptors:
            try:
                info = (Path(entry.path) / 'fdinfo' / descriptor.name).read_text()
                flags = next(int(line.split()[1], 8) for line in info.splitlines() if line.startswith('flags:'))
                if flags & os.O_ACCMODE:
                    s = os.stat(descriptor.path)
                    if stat.S_ISREG(s.st_mode):
                        result.add((s.st_dev, s.st_ino))
            except (FileNotFoundError, ProcessLookupError, StopIteration):
                continue
            except PermissionError:
                inaccessible.append(entry.name)
    return result, sorted(set(inaccessible))


def hash_fd(fd):
    os.lseek(fd, 0, os.SEEK_SET)
    digest = hashlib.sha256()
    while block := os.read(fd, 1024 * 1024):
        digest.update(block)
    return digest.hexdigest()


def run_states():
    states = {}
    for name in sorted(ALLOWED_RUNS):
        run = ROOT / 'eval_results' / name
        metadata = run / 'run_metadata.json'
        if run.is_symlink() or metadata.is_symlink() or not metadata.is_file():
            raise ValueError(f'unsafe or absent run metadata: {name}')
        raw = metadata.read_bytes()
        if json.loads(raw).get('status') != 'complete':
            raise ValueError(f'run is not complete: {name}')
        archive_path = run / 'sources.zip'
        if archive_path.is_symlink():
            raise ValueError('symbolic-link source archive')
        hits = []
        with zipfile.ZipFile(archive_path) as archive:
            for source in archive.namelist():
                if source.endswith('.py') and re.search(rb'st_mtime|mtime_ns|getmtime', archive.read(source)):
                    hits.append(source)
        states[name] = dict(metadata_sha256=hashlib.sha256(raw).hexdigest(),
                            metadata_signature=signature(metadata.stat()), source_archive_sha256=sha256(archive_path),
                            archived_mtime_dependencies=hits)
    return states


def verify_run(name, states):
    path = ROOT / 'eval_results' / name / 'run_metadata.json'
    state = states[name]
    if signature(path.lstat()) != tuple(state['metadata_signature']):
        raise SkipPair('run metadata changed after completed-state check')
    if state['archived_mtime_dependencies']:
        raise SkipPair('archived code may depend on payload mtime')


def replace_pair(pair, states, writers, apply):
    canonical, canonical_run = checked_path(pair['canonical'])
    duplicate, duplicate_run = checked_path(pair['duplicate'])
    verify_run(canonical_run, states)
    verify_run(duplicate_run, states)
    expected_size, expected_hash = pair['bytes'], pair['sha256']
    fds = []
    backup = duplicate.with_name('.storage-dedup-backup-' + uuid.uuid4().hex + '.tmp')
    linked = duplicate.with_name('.storage-dedup-link-' + uuid.uuid4().hex + '.tmp')
    replaced = False
    succeeded = False
    try:
        for path in (canonical, duplicate):
            fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW)
            fds.append(fd)
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        a, b = (os.fstat(fd) for fd in fds)
        if (a.st_dev, a.st_ino) == (b.st_dev, b.st_ino):
            if a.st_size != expected_size or hash_fd(fds[0]) != expected_hash:
                raise SkipPair('already linked payload hash mismatch')
            return dict(status='already_linked', canonical_before=snapshot(a), duplicate_before=snapshot(b))
        if a.st_dev != b.st_dev:
            raise SkipPair('different filesystems')
        if (a.st_mode, a.st_uid, a.st_gid) != (b.st_mode, b.st_uid, b.st_gid):
            raise SkipPair('permission or ownership mismatch')
        if (a.st_dev, a.st_ino) in writers or (b.st_dev, b.st_ino) in writers:
            raise SkipPair('open writable file descriptor detected')
        if a.st_size != expected_size or b.st_size != expected_size:
            raise SkipPair('manifest size mismatch')
        if hash_fd(fds[0]) != expected_hash or hash_fd(fds[1]) != expected_hash:
            raise SkipPair('manifest SHA256 mismatch')
        if any(signature(os.fstat(fd)) != signature(before) or signature(path.lstat()) != signature(before)
               for fd, path, before in zip(fds, (canonical, duplicate), (a, b))):
            raise SkipPair('payload changed during hash check')
        result = dict(status='verified_only', canonical_before=snapshot(a), duplicate_before=snapshot(b),
                      expected_data_blocks_released=(b.st_blocks * 512 if b.st_nlink == 1 else 0))
        if not apply:
            return result
        # Keep an original-inode backup until post-replacement validation passes.
        # A concurrent reader holding the old inode retains a readable descriptor.
        os.link(duplicate, backup, follow_symlinks=False)
        os.link(canonical, linked, follow_symlinks=False)
        if any(signature(os.fstat(fd)) != signature(before) or signature(path.lstat()) != signature(before)
               for fd, path, before in zip(fds, (canonical, duplicate), (a, b))):
            raise SkipPair('payload changed before atomic replacement')
        verify_run(canonical_run, states)
        verify_run(duplicate_run, states)
        os.replace(linked, duplicate)
        replaced = True
        post = duplicate.lstat()
        if ((post.st_dev, post.st_ino) != (a.st_dev, a.st_ino) or post.st_nlink < 2
                or post.st_size != expected_size or sha256(duplicate) != expected_hash
                or hash_fd(fds[0]) != expected_hash):
            raise SkipPair('post-replacement inode or hash validation failed')
        if signature(os.fstat(fds[1])) != signature(b):
            raise SkipPair('original duplicate changed during replacement')
        backup.unlink()
        succeeded = True
        result.update(status='linked', after=snapshot(post), bytes_preserved=True,
                      expected_data_blocks_released=(b.st_blocks * 512 if os.fstat(fds[1]).st_nlink == 0 else 0))
        return result
    finally:
        if backup.exists():
            if replaced and not succeeded:
                os.replace(backup, duplicate)
            else:
                backup.unlink()
        if linked.exists():
            linked.unlink()
        for fd in reversed(fds):
            os.close(fd)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--apply', action='store_true', help='Perform approved exact-byte hard-link storage deduplication.')
    args = parser.parse_args()
    output = args.output.absolute()
    if (output.parent != AUDIT or not output.name.startswith('completed_sensor_storage_dedup_')
            or output.exists() or AUDIT.is_symlink()):
        raise ValueError('output must be a new approved audit_results/completed_sensor_storage_dedup_* directory')
    output.mkdir()
    lock = (AUDIT / '.completed_sensor_storage_dedup.lock').open('a')
    fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    compressed = Path(str(ESTIMATE) + '.gz')
    if ESTIMATE.exists():
        if ESTIMATE.is_symlink():
            raise ValueError('symbolic-link estimate forbidden')
        before_estimate = ESTIMATE.stat()
        raw = ESTIMATE.read_bytes()
    else:
        if compressed.is_symlink():
            raise ValueError('symbolic-link compressed estimate forbidden')
        raw = gzip.decompress(compressed.read_bytes())
        before_estimate = None
    input_hash = hashlib.sha256(raw).hexdigest()
    estimate = json.loads(raw)
    if estimate.get('action') != 'estimate_only' or estimate.get('scope') != 'read-only exact duplicate estimate in completed-run immutable raw frame/scan/mesh/map files':
        raise ValueError('not the approved duplicate estimate schema')
    pairs = estimate['pairs']
    duplicate_paths = {r['duplicate'] for r in pairs}
    if len(duplicate_paths) != len(pairs) or duplicate_paths & {r['canonical'] for r in pairs}:
        raise ValueError('duplicate targets or canonical replacement chains forbidden')
    if any(not isinstance(r['bytes'], int) or r['bytes'] <= 0 or not re.fullmatch('[0-9a-f]{64}', r['sha256']) for r in pairs):
        raise ValueError('invalid estimated payload size or digest')
    states = run_states()
    free_before = shutil.disk_usage(ROOT).free
    started = time.monotonic()
    summary = dict(status='running', operation='Storage deduplication only; original experiment paths and bytes retained',
                   apply=args.apply, source_script_sha256=sha256(Path(__file__)), input_estimate_sha256=input_hash,
                   input_estimate_uncompressed_bytes=len(raw), expected_pairs=len(pairs),
                   estimated_duplicate_payload_bytes=estimate['bytes'], runs=states,
                   allowed_payloads=['frames/number.npz', 'scans/number.npz', 'final_mesh.npz', 'maps.npz'],
                   active_counterfactual_runs_excluded=True, inode_mtime_policy='Linked path adopts canonical inode/mtime; original metadata retained in action manifest',
                   integrity_mtime_audit='No mtime APIs in 19 frozen source archives. Current mtime usages are training log watchdogs and live JPEG/PNG preview, outside eligible NPZ payloads.',
                   writable_descriptor_policy='Scan /proc writable file descriptors initially and every 250 pairs; nonblocking advisory exclusive locks and size/mtime/hash checks bracket every replacement.',
                   canonical_paths_never_replaced=True, free_bytes_before=free_before)
    write_json(output / 'summary.json', summary)
    if args.apply and before_estimate is not None:
        if compressed.exists():
            if compressed.is_symlink() or hashlib.sha256(gzip.decompress(compressed.read_bytes())).hexdigest() != input_hash:
                raise ValueError('existing compressed estimate differs')
        else:
            with compressed.open('xb') as stream:
                with gzip.GzipFile(fileobj=stream, mode='wb', mtime=0, compresslevel=6) as archive:
                    archive.write(raw)
        if (hashlib.sha256(gzip.decompress(compressed.read_bytes())).hexdigest() != input_hash
                or signature(ESTIMATE.lstat()) != signature(before_estimate) or sha256(ESTIMATE) != input_hash):
            raise ValueError('estimate changed or compressed archive failed verification')
        summary.update(input_estimate_compressed_path=str(compressed.relative_to(ROOT)),
                       input_estimate_compressed_sha256=sha256(compressed),
                       input_estimate_compressed_bytes=compressed.stat().st_size,
                       redundant_uncompressed_estimate_removed=True)
        write_json(output / 'summary.json', summary)
        ESTIMATE.unlink()
    counts = Counter()
    skip_reasons = Counter()
    linked_bytes = released_blocks = 0
    action_path = output / 'actions.jsonl.gz'
    try:
        with action_path.open('xb') as file_stream:
            with gzip.GzipFile(fileobj=file_stream, mode='wb', mtime=0, compresslevel=3) as journal:
                for index, pair in enumerate(pairs):
                    if index % 250 == 0:
                        writers, inaccessible = writable_inodes()
                    try:
                        if inaccessible:
                            raise SkipPair('cannot inspect all process writable descriptors')
                        outcome = replace_pair(pair, states, writers, args.apply)
                    except (SkipPair, OSError) as error:
                        outcome = dict(status='skipped', reason=str(error))
                        skip_reasons[str(error)] += 1
                    counts[outcome['status']] += 1
                    if outcome['status'] == 'linked':
                        linked_bytes += pair['bytes']
                        released_blocks += outcome['expected_data_blocks_released']
                    journal.write((json.dumps(dict(index=index, **pair, **outcome), separators=(',', ':')) + '\n').encode())
                    if (index + 1) % 5000 == 0:
                        journal.flush()
                        summary.update(processed=index + 1, counts=dict(counts), skip_reasons=dict(skip_reasons),
                                       linked_payload_bytes=linked_bytes, expected_released_data_block_bytes=released_blocks,
                                       elapsed_s=time.monotonic() - started)
                        write_json(output / 'summary.json', summary)
                        print(index + 1, '/', len(pairs), dict(counts), 'linked MiB', round(linked_bytes / 2**20, 1), flush=True)
        # Completed metadata and canonical payload paths must remain unchanged.
        for name, state in states.items():
            verify_run(name, states)
            if sha256(ROOT / 'eval_results' / name / 'run_metadata.json') != state['metadata_sha256']:
                raise AssertionError('run metadata bytes changed')
        if sha256(Path(__file__)) != summary['source_script_sha256']:
            raise AssertionError('deduplication script changed during execution')
        summary.update(status='complete', processed=len(pairs), counts=dict(counts), skip_reasons=dict(skip_reasons),
                       linked_payload_bytes=linked_bytes, expected_released_data_block_bytes=released_blocks,
                       free_bytes_after=shutil.disk_usage(ROOT).free, elapsed_s=time.monotonic() - started,
                       action_manifest_sha256=sha256(action_path), action_manifest_compressed_bytes=action_path.stat().st_size,
                       unique_payloads_removed=0, original_experiment_paths_removed=0,
                       run_metadata_bytes_unchanged=True, limitation='Hard links share an inode; completed payloads must remain immutable. '
                       'Observed free-space change also includes concurrent writers and open readers retaining replaced inodes.')
        write_json(output / 'summary.json', summary)
    except Exception as error:
        summary.update(status='failed', error=repr(error), counts=dict(counts), elapsed_s=time.monotonic() - started)
        write_json(output / 'summary.json', summary)
        raise


if __name__ == '__main__':
    main()
