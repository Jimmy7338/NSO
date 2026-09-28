#!/usr/bin/env python3
"""Rebuild the pinned, isolated Focal/Noetic TARE runtime after tmpfs loss.

Only official signed repository metadata and the already audited exact package
versions/hashes are accepted. No host packages, containers or HOME are changed.
Run with the task's authorized permission to write its /dev/shm directory.
"""
import argparse
import gzip
import hashlib
import json
import lzma
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
import urllib.request

ROOT = Path(__file__).resolve().parents[1]
RECEIPTS = ROOT / 'audit_results/v39_tare_runtime_preflight_20260920'
PACKAGE_RECEIPT_SHA = 'fe78696d1544c0927ab514915a2ca69ee550a45870383a17198a393bb1fb88b3'
ROS_KEY_SHA = '4a91c49af0d6f0016108b93698782b596c27ccd836937e18e0e36c3347dc602f'
NODE_SHA = '59fd9b818f4cc6a2edc3e5bf4a8641e2dad92b19ba1d824f1427624aecfd6f03'


def digest(data):
    return hashlib.sha256(data).hexdigest()


def checked_read(path, expected):
    data = path.read_bytes()
    if digest(data) != expected:
        raise RuntimeError('frozen artifact hash mismatch: ' + str(path))
    return data


def guard(runtime):
    available = int(next(x.split()[1] for x in Path('/proc/meminfo').read_text().splitlines()
                         if x.startswith('MemAvailable:'))) * 1024
    owned_bytes = sum(p.stat().st_size for p in runtime.rglob('*')
                      if p.is_file() and not p.is_symlink()) if runtime.exists() else 0
    result = dict(memory_available_bytes=available,
                  tmpfs_free_bytes=shutil.disk_usage('/dev/shm').free,
                  persistent_free_bytes=shutil.disk_usage(ROOT).free,
                  owned_file_bytes=owned_bytes)
    if (available < 2 * 1024**3 or result['tmpfs_free_bytes'] < 512 * 1024**2
            or result['persistent_free_bytes'] < 64 * 1024**2
            or owned_bytes > 1.5 * 1024**3):
        raise RuntimeError('runtime resource reserve violated: ' + json.dumps(result))
    return result


def command(argv, environment=None, timeout=60):
    process = subprocess.run(argv, env=environment, capture_output=True, text=True, timeout=timeout)
    if process.returncode:
        raise RuntimeError('command failed: %r\n%s\n%s' % (argv, process.stdout, process.stderr))
    return process


def download(runtime, url, limit=32 * 1024**2):
    guard(runtime)
    request = urllib.request.Request(url, headers={'User-Agent': 'NSO-pinned-TARE-runtime/39'})
    with urllib.request.urlopen(request, timeout=45) as response:
        data = response.read(limit + 1)
    if len(data) > limit:
        raise RuntimeError('download exceeds explicit size bound: ' + url)
    return data


def package_rows(text):
    for paragraph in text.split('\n\n'):
        fields = {}
        for line in paragraph.splitlines():
            if line and not line[0].isspace() and ': ' in line:
                key, value = line.split(': ', 1)
                fields[key] = value
        if 'Package' in fields:
            yield fields


def verified_metadata(runtime, frozen):
    metadata = runtime / 'metadata'
    key = checked_read(RECEIPTS / 'ros.asc', ROS_KEY_SHA)
    (metadata / 'ros.asc').write_bytes(key)
    command(['gpg', '--batch', '--yes', '--dearmor', '--output', str(metadata / 'ros.gpg'),
             str(metadata / 'ros.asc')])
    repositories = [
        ('ros', 'http://packages.ros.org/ros/ubuntu', metadata / 'ros.gpg',
         ['main/binary-amd64/Packages.gz']),
        ('ubuntu', 'https://archive.ubuntu.com/ubuntu',
         Path('/usr/share/keyrings/ubuntu-archive-keyring.gpg'),
         ['main/binary-amd64/Packages.xz', 'universe/binary-amd64/Packages.xz']),
    ]
    wanted = {row['package'] for row in frozen}
    selected, receipts = {}, []
    for label, base, keyring, indexes in repositories:
        release = download(runtime, base + '/dists/focal/Release', 1024**2)
        signature = download(runtime, base + '/dists/focal/Release.gpg', 65536)
        release_path, signature_path = metadata / (label + '.Release'), metadata / (label + '.Release.gpg')
        release_path.write_bytes(release)
        signature_path.write_bytes(signature)
        verification = command(['gpgv', '--keyring', str(keyring), str(signature_path), str(release_path)])
        checks = {}
        in_sha256 = False
        for line in release.decode().splitlines():
            if line == 'SHA256:':
                in_sha256 = True
            elif in_sha256 and line.startswith(' '):
                sha, size, name = line.split()
                checks[name] = (sha, int(size))
            elif in_sha256:
                break
        for index in indexes:
            data = download(runtime, base + '/dists/focal/' + index)
            if (digest(data), len(data)) != checks[index]:
                raise RuntimeError('signed package index hash or size mismatch: ' + index)
            unpacked = gzip.decompress(data) if index.endswith('.gz') else lzma.decompress(data)
            for row in package_rows(unpacked.decode()):
                if row['Package'] in wanted:
                    row.update(repo_base=base, index_label=label)
                    selected[row['Package']] = row
            receipts.append(dict(repository=base, distribution='focal', index=index,
                                 bytes=len(data), sha256=digest(data), release_sha256=digest(release),
                                 signature_verification=verification.stdout + verification.stderr))
    for pin in frozen:
        row = selected.get(pin['package'])
        if (row is None or row['Version'] != pin['version'] or row['SHA256'] != pin['sha256']
                or int(row['Size']) != pin['download_bytes']
                or row['repo_base'] + '/' + row['Filename'] != pin['url']):
            raise RuntimeError('repository no longer provides frozen exact package; refusing substitution: '
                               + pin['package'])
    (metadata / 'selected_packages.json').write_text(json.dumps(selected, indent=2) + '\n')
    (metadata / 'index_receipts.json').write_text(json.dumps(receipts, indent=2) + '\n')
    return selected


def install(runtime, pins, force):
    receipt_path = runtime / 'installed_packages.json'
    previous = json.loads(receipt_path.read_text()) if receipt_path.exists() else []
    existing = {row['package']: row for row in previous}
    installed = []
    for pin in pins:
        if not force and existing.get(pin['package']) == pin:
            installed.append(pin)
            continue
        data = download(runtime, pin['url'])
        if digest(data) != pin['sha256'] or len(data) != pin['download_bytes']:
            raise RuntimeError('frozen deb hash or size mismatch: ' + pin['package'])
        temporary = runtime / (pin['package'] + '.deb')
        temporary.write_bytes(data)
        try:
            command(['dpkg-deb', '-x', str(temporary), str(runtime / 'runtime')])
        finally:
            temporary.unlink()
        installed.append(pin)
        receipt_path.write_text(json.dumps(installed, indent=2) + '\n')
        guard(runtime)
        print(json.dumps(dict(status='package_restored', package=pin['package'], version=pin['version'])), flush=True)
    receipt_path.write_text(json.dumps(installed, indent=2) + '\n')


def configure(runtime):
    prefix = runtime / 'runtime'
    original = ROOT / 'audit_results/tare_native_node_attempt_20260911/tare_planner_node'
    node = runtime / 'tare_planner_node'
    node.write_bytes(checked_read(original, NODE_SHA))
    node.chmod(0o755)
    configuration = dict(
        PYTHONHOME=str(prefix / 'usr'),
        PYTHONPATH=str(prefix / 'opt/ros/noetic/lib/python3/dist-packages') + ':'
        + str(prefix / 'usr/lib/python3/dist-packages'), PYTHONDONTWRITEBYTECODE='1',
        LD_LIBRARY_PATH=':'.join(map(str, [prefix / 'opt/ros/noetic/lib', prefix / 'usr/lib/x86_64-linux-gnu',
            prefix / 'lib/x86_64-linux-gnu', ROOT / 'third_party/official_baselines/tare_official/src/tare_planner/or-tools/lib'])),
        ROS_MASTER_URI='http://127.0.0.1:11339', ROS_IP='127.0.0.1',
        ROS_HOME=str(runtime / 'ros_home'), ROS_LOG_DIR=str(runtime / 'logs'))
    (runtime / 'environment.json').write_text(json.dumps(configuration, indent=2) + '\n')
    return configuration


def verify(runtime, smoke, port):
    checked_read(runtime / 'tare_planner_node', NODE_SHA)
    environment = dict(os.environ)
    environment.update(json.loads((runtime / 'environment.json').read_text()))
    dependency_check = command(['ldd', str(runtime / 'tare_planner_node')], environment)
    if 'not found' in dependency_check.stdout + dependency_check.stderr:
        raise RuntimeError('unresolved native dependency')
    (runtime / 'ldd_current.txt').write_text(dependency_check.stdout)
    python_check = command([str(runtime / 'runtime/usr/bin/python3.8'), '-B', '-c',
        'import sys,rospy,rosmaster,rosgraph,numpy,yaml;'
        'from sensor_msgs.msg import PointCloud2;from nav_msgs.msg import Odometry;'
        'print(sys.version);print("Noetic Python imports OK")'], environment)
    (runtime / 'python_import_check.txt').write_text(python_check.stdout)
    if smoke:
        check = command([sys.executable, '-B', str(ROOT / 'scripts/tare_ros_bridge_v39.py'),
                         '--runtime', str(runtime), '--smoke-only', '--receipt',
                         str(runtime / 'restored_node_smoke_receipt.json'), '--port', str(port)], timeout=60)
        (runtime / 'restored_node_smoke.stdout.jsonl').write_text(check.stdout)
        (runtime / 'restored_node_smoke.stderr.txt').write_text(check.stderr)
    return dict(native_ldd_all_resolved=True, python_noetic_imports=True,
                zero_sensor_full_node_smoke=smoke, resources=guard(runtime))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--runtime', type=Path, default=Path('/dev/shm/nso_v39_tare'))
    parser.add_argument('--verify-only', action='store_true', help='offline native/import/zero-sensor checks only')
    parser.add_argument('--force-reextract', action='store_true', help='re-extract pinned packages even if receipts exist')
    parser.add_argument('--skip-node-smoke', action='store_true')
    parser.add_argument('--port', type=int, default=11339, help='private master port for the zero-sensor smoke')
    args = parser.parse_args()
    runtime = args.runtime.absolute()
    if runtime.parent != Path('/dev/shm') or not runtime.name.startswith('nso_v39_tare') or runtime.is_symlink():
        raise RuntimeError('only the task-owned /dev/shm/nso_v39_tare namespace is supported')
    started = time.monotonic()
    pins = json.loads(checked_read(RECEIPTS / 'installed_packages.json', PACKAGE_RECEIPT_SHA))
    guard(runtime)
    if not args.verify_only:
        for directory in [runtime, runtime / 'runtime', runtime / 'metadata', runtime / 'logs', runtime / 'ros_home']:
            if directory.is_symlink():
                raise RuntimeError('refusing symlinked extraction/configuration directory')
            directory.mkdir(exist_ok=True)
        verified_metadata(runtime, pins)
        install(runtime, pins, args.force_reextract)
        configure(runtime)
    result = verify(runtime, not args.skip_node_smoke, args.port)
    result.update(status='restored_and_verified' if not args.verify_only else 'verified',
                  exact_package_count=len(pins), elapsed_seconds=time.monotonic() - started,
                  node_sha256=NODE_SHA, runtime=str(runtime), new_worlds=0, new_sensor_renders=0,
                  new_TSDF=0, new_Q=0, host_packages_changed=False)
    (runtime / 'restore_verification.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result), flush=True)


if __name__ == '__main__':
    main()
