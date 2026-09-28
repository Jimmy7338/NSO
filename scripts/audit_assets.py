#!/usr/bin/env python3
"""Inventory local NSO research assets without loading pickle/model payloads."""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import gzip
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import platform
import re
import subprocess

SCAN_DIRS = ('pretrained_models', 'trained_models', 'noise_models', 'data',
             'eval_results', 'tmp')


def sha256_file(path):
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def inspect_asset(path, root):
    entry = {'path': str(path.relative_to(root))}
    if path.is_symlink():
        entry['symlink_target'] = os.readlink(path)
        if not path.exists():
            return dict(entry, status='broken_symlink')
    entry['size_bytes'] = path.stat().st_size
    entry['sha256'] = sha256_file(path)
    with path.open('rb') as stream:
        prefix = stream.read(1024)
    if prefix.startswith(b'version https://git-lfs.github.com/spec/v1'):
        oid = re.search(rb'oid sha256:([a-f0-9]{64})', prefix)
        size = re.search(rb'\nsize (\d+)', prefix)
        entry.update(status='lfs_pointer_not_materialized',
                     lfs_oid=oid.group(1).decode() if oid else None,
                     expected_size_bytes=int(size.group(1)) if size else None)
    else:
        entry['status'] = 'present_unvalidated'
    if path.name.endswith('.json.gz') and entry['status'] == 'present_unvalidated':
        try:
            with gzip.open(path, 'rt') as stream:
                data = json.load(stream)
            episodes = data.get('episodes', [])
            entry.update(status='episode_json_valid', episode_count=len(episodes),
                         scene_ids=sorted({e['scene_id'] for e in episodes if 'scene_id' in e}))
        except (OSError, ValueError, TypeError, AttributeError) as exc:
            entry.update(status='invalid_episode_json', error=str(exc))
    return entry


def audit(root, output_path=None):
    root = Path(root).resolve()
    paths = set()
    for name in SCAN_DIRS:
        directory = root / name
        if directory.is_symlink() and not directory.exists():
            paths.add(directory)
        if directory.exists():
            paths.update(p for p in directory.rglob('*') if p.is_file() or p.is_symlink())
    paths.update(p for p in root.iterdir() if p.suffix in ('.pt', '.zip', '.pdf', '.tex'))
    assets = []
    for path in sorted(paths):
        if '__pycache__' in path.parts or (output_path and path.resolve() == output_path.resolve()):
            continue
        if path.is_dir():
            continue
        assets.append(inspect_asset(path, root))
    devices = [str(p) for pattern in ('nvidia*', 'dri/render*') for p in Path('/dev').glob(pattern)]
    try:
        revision = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=root, text=True).strip()
    except (OSError, subprocess.CalledProcessError):
        revision = None
    scene_assets = [a for a in assets if Path(a['path']).suffix in ('.glb', '.ply', '.navmesh')]
    weights = [a for a in assets if 'model_best.' in a['path'] or a['path'] == 'yolov8n.pt']
    return {'generated_at': datetime.now(timezone.utc).isoformat(), 'root': str(root),
            'code_revision': revision,
            'scope': 'local files only; no remote availability or weight loadability inferred',
            'environment': {'python': platform.python_version(), 'cpu_count': os.cpu_count(),
                            'visible_graphics_devices': devices,
                            'dependencies': {m: bool(importlib.util.find_spec(m)) for m in
                                             ('numpy', 'scipy', 'matplotlib', 'torch', 'cv2', 'habitat', 'habitat_sim')}},
            'summary': {'asset_count': len(assets),
                        'status_counts': dict(Counter(a['status'] for a in assets)),
                        'weight_file_count': len(weights),
                        'unmaterialized_weight_count': sum(a['status'] == 'lfs_pointer_not_materialized' for a in weights),
                        'scene_asset_count': len(scene_assets),
                        'rgbd_or_map_cache_files': [a['path'] for a in assets if Path(a['path']).suffix in ('.npz', '.npy', '.h5')]},
            'assets': assets}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f'Refusing to overwrite an audit: {args.output}')
    result = audit(args.root, args.output)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps(result['summary'], ensure_ascii=False))


if __name__ == '__main__':
    main()
