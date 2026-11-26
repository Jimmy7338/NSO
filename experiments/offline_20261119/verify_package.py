#!/usr/bin/env python3
"""Verify packaged evidence and source bytes without loading actors or assets."""
from pathlib import Path
import hashlib
import json
import zipfile

root = Path(__file__).resolve().parent
sha = lambda data: hashlib.sha256(data).hexdigest()
manifest = json.loads((root / 'RELEASE_MANIFEST.json').read_text())
for entry in manifest['files']:
    path = root / entry['path']
    assert path.is_file() and path.stat().st_size == entry['bytes'] and sha(path.read_bytes()) == entry['sha256'], entry['path']
index = json.loads((root / 'ARCHIVE_INDEX.json').read_text())
members = 0
for name, info in index.items():
    path = root / 'archives' / name
    assert sha(path.read_bytes()) == info['sha256']
    with zipfile.ZipFile(path) as archive:
        assert set(archive.namelist()) == {r['member'] for r in info['members']}
        for row in info['members']:
            data = archive.read(row['member'])
            assert len(data) == row['bytes'] and sha(data) == row['sha256'], row['member']
            assert not row['member'].startswith('/') and '..' not in Path(row['member']).parts
            members += 1
print(json.dumps(dict(status='all_packaged_bytes_verified', payload_files=len(manifest['files']),
                      archived_files=members, actor_executed=False, raw_assets_required=False)))
