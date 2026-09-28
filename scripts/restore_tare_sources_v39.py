#!/usr/bin/env python3
"""Restore the exact official TARE source/OR-Tools snapshot, without building it.

Existing trees are verified in place. Cold restores default to task-owned tmpfs
and expose the expected repository path through a symlink. No research freeze,
system package, native planner binary, or unrelated directory is overwritten.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import shutil
import tarfile
import tempfile
import time
import urllib.request

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / 'docs/research/official_baseline_source_manifest.json'
MANIFEST_SHA = 'ff702f088a13d8c5f39bd18b8287bc94f4154d65bf04babe0d2e1be3a72fa447'
COMMIT = '44500592b86138257273e0cab264e6a847ccefc7'
ARCHIVE_SHA = 'a3adfcab09a669a2a1b9bf0aaec734171cc46ce0fa5cc16a3b54fc51fdb36024'
ARCHIVE_BYTES = 38270178
MAX_EXPANDED_BYTES = 160 * 1024**2
RAM_ROOT = Path('/dev/shm/nso_v39_tare_sources')
EXPECTED_LINKS = {
    'src/tare_planner/or-tools/lib/libortools.so': 'libortools.so.9',
    'src/tare_planner/or-tools/lib/libortools.so.9': 'libortools.so.9.8.3296',
}
PROVENANCE_FILES = {
    'upstream_package.xml': 'src/tare_planner/package.xml',
    'upstream_README.md': 'README.md',
    'upstream_ortools_README.md': 'src/tare_planner/or-tools/README.md',
}


def sha_file(path):
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        while True:
            block = stream.read(1024**2)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def guard(storage, required=0):
    memory = int(next(x.split()[1] for x in Path('/proc/meminfo').read_text().splitlines()
                      if x.startswith('MemAvailable:'))) * 1024
    disk = shutil.disk_usage(ROOT).free
    if disk < 64 * 1024**2 or memory < 2 * 1024**3:
        raise RuntimeError('64 MiB persistent / 2 GiB available-memory reserve would be violated')
    if storage == 'ram':
        free = shutil.disk_usage('/dev/shm').free
        if free < 512 * 1024**2 + required:
            raise RuntimeError('insufficient tmpfs: restore must preserve 512 MiB after declared allocation')
    elif disk < 64 * 1024**2 + required:
        raise RuntimeError('insufficient persistent space; use the default --storage ram')
    return dict(persistent_free_bytes=disk, memory_available_bytes=memory,
                tmpfs_free_bytes=shutil.disk_usage('/dev/shm').free)


def pinned_record():
    if sha_file(MANIFEST) != MANIFEST_SHA:
        raise RuntimeError('pinned official source manifest changed')
    record = next(x for x in json.loads(MANIFEST.read_text())['sources'] if x['id'] == 'tare_official')
    if (record['commit'] != COMMIT or record['archive_sha256'] != ARCHIVE_SHA
            or record['archive_bytes'] != ARCHIVE_BYTES):
        raise RuntimeError('official source snapshot identity mismatch')
    return record


def verify_tree(tree, record):
    checked_bytes = 0
    for relative, expected in record['files_sha256'].items():
        path = tree / relative
        if not path.is_file() or path.is_symlink() or sha_file(path) != expected:
            raise RuntimeError('official source file missing or changed: ' + str(path))
        checked_bytes += path.stat().st_size
    for relative, expected in EXPECTED_LINKS.items():
        path = tree / relative
        if not path.is_symlink() or os.readlink(str(path)) != expected:
            raise RuntimeError('OR-Tools SONAME symlink mismatch: ' + str(path))
    return dict(files_checked=len(record['files_sha256']), checked_file_bytes=checked_bytes,
                soname_symlinks_checked=EXPECTED_LINKS)


def extract_verified(archive, output, record):
    prefix = 'tare_planner-' + COMMIT
    omitted = record['omitted_external_workspace_links']
    with tarfile.open(str(archive), 'r:gz') as source:
        selected, total = [], 0
        for member in source.getmembers():
            path = PurePosixPath(member.name)
            if path.is_absolute() or '..' in path.parts or not path.parts or path.parts[0] != prefix:
                raise RuntimeError('archive member escapes pinned root')
            if member.name in omitted:
                if not (member.issym() or member.islnk()) or member.linkname != omitted[member.name]:
                    raise RuntimeError('omitted upstream workspace symlink differs')
                continue
            if not (member.isfile() or member.isdir() or member.issym() or member.islnk()):
                raise RuntimeError('unexpected archive member type')
            if member.issym() or member.islnk():
                link = PurePosixPath(member.linkname)
                if link.is_absolute():
                    raise RuntimeError('unexpected absolute archive link')
                target = path.parent / link if member.issym() else link
                resolved = PurePosixPath(os.path.normpath(str(target)))
                if not resolved.parts or resolved.parts[0] != prefix or '..' in resolved.parts:
                    raise RuntimeError('archive link escapes pinned root')
            if member.isfile():
                total += member.size
            selected.append(member)
        if total > MAX_EXPANDED_BYTES:
            raise RuntimeError('expanded source exceeds declared space budget')
        # Every member and link has been confined to this new private tree;
        # use data_filter too when available, while retaining Python 3.8 support.
        if hasattr(tarfile, 'data_filter'):
            source.extractall(str(output), members=selected, filter='data')
        else:
            source.extractall(str(output), members=selected)
    return output / prefix


def restore(record, storage):
    target = ROOT / record['directory']
    if target.exists():
        return target, False, verify_tree(target, record)
    ram_target = RAM_ROOT / 'tare_official'
    if target.is_symlink() and os.readlink(str(target)) != str(ram_target):
        raise RuntimeError('refusing to replace an unrelated dangling source symlink')
    guard(storage, ARCHIVE_BYTES + MAX_EXPANDED_BYTES)
    if storage == 'ram':
        if RAM_ROOT.is_symlink():
            raise RuntimeError('task tmpfs root cannot be a symlink')
        RAM_ROOT.mkdir(exist_ok=True)
        stage_parent, destination = RAM_ROOT, ram_target
    else:
        target.parent.mkdir(parents=True, exist_ok=True)
        stage_parent, destination = target.parent, target
    downloaded = False
    if destination.exists():
        verification = verify_tree(destination, record)
    else:
        with tempfile.TemporaryDirectory(prefix='.tare-source-v39-', dir=str(stage_parent)) as temporary:
            temporary = Path(temporary)
            archive = temporary / 'source.tar.gz'
            digest, size = hashlib.sha256(), 0
            request = urllib.request.Request(record['archive_url'], headers={'User-Agent': 'NSO-pinned-TARE-source/39'})
            with urllib.request.urlopen(request, timeout=60) as response, archive.open('wb') as stream:
                while True:
                    chunk = response.read(1024**2)
                    if not chunk:
                        break
                    size += len(chunk)
                    if size > ARCHIVE_BYTES:
                        raise RuntimeError('archive exceeds pinned exact byte count')
                    stream.write(chunk)
                    digest.update(chunk)
            if size != ARCHIVE_BYTES or digest.hexdigest() != ARCHIVE_SHA:
                raise RuntimeError('official archive exact SHA or size mismatch')
            extracted = extract_verified(archive, temporary / 'tree', record)
            verification = verify_tree(extracted, record)
            if destination == target and target.is_symlink():
                target.unlink()  # matching task-owned dangling link, validated above
            extracted.rename(destination)
            downloaded = True
    if storage == 'ram':
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.is_symlink():
            target.unlink()  # only the matching, already verified task-owned dangling link
        target.symlink_to(ram_target, target_is_directory=True)
    guard(storage)
    return target, downloaded, verification


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--storage', choices=('ram', 'repository'), default='ram')
    parser.add_argument('--receipt', type=Path, default=ROOT / 'tmp/v39_tare_sources_bootstrap/receipt.json')
    args = parser.parse_args()
    started = time.monotonic()
    record = pinned_record()
    guard(args.storage)
    tree, downloaded, checked = restore(record, args.storage)
    args.receipt.parent.mkdir(parents=True, exist_ok=True)
    origins = {}
    for name, relative in PROVENANCE_FILES.items():
        path = args.receipt.parent / name
        data = (tree / relative).read_bytes()
        path.write_bytes(data)
        origins[name] = dict(upstream_relative_path=relative, sha256=hashlib.sha256(data).hexdigest())
    result = dict(status='restored_and_verified' if downloaded else 'existing_tree_verified',
        downloaded_archive=downloaded, repository=record['repository'], commit=COMMIT,
        archive_url=record['archive_url'], archive_sha256=ARCHIVE_SHA, archive_bytes=ARCHIVE_BYTES,
        pinned_manifest=str(MANIFEST.relative_to(ROOT)), pinned_manifest_sha256=MANIFEST_SHA,
        tree=str(tree), resolved_tree=str(tree.resolve()), verification=checked,
        omitted_external_workspace_links=record['omitted_external_workspace_links'],
        provenance_files=origins, package_declared_license='BSD',
        standalone_LICENSE_COPYING_NOTICE_present_in_pinned_manifest=False,
        license_note='Preserve upstream package.xml and README verbatim; do not invent a BSD variant. '
                     'Bundled OR-Tools README notes separate SCIP terms. Full third-party artifacts are '
                     'restored from the official pinned archive rather than re-licensed as this project.',
        script_sha256=sha_file(Path(__file__)), elapsed_seconds=time.monotonic() - started,
        resources=guard(args.storage), native_code_executed=False, sources_compiled=False,
        new_worlds=0, new_sensor_renders=0, new_TSDF=0, new_Q=0)
    args.receipt.write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps(result, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
