#!/usr/bin/env python3
"""Verify exact packaged bytes; never load an actor or third-party geometry."""
from pathlib import Path
import hashlib
import json
import zipfile

root = Path(__file__).resolve().parent
sha = lambda b: hashlib.sha256(b).hexdigest()
manifest = json.loads((root/'RELEASE_MANIFEST.json').read_text())
for row in manifest['files']:
    p = root/row['path']
    assert p.is_file() and p.stat().st_size == row['bytes'] and sha(p.read_bytes()) == row['sha256'], row['path']
index = json.loads((root/'ARCHIVE_INDEX.json').read_text())
members = 0
for name, entry in index.items():
    p = root/'archives'/name
    assert sha(p.read_bytes()) == entry['sha256']
    with zipfile.ZipFile(p) as z:
        assert set(z.namelist()) == {r['member'] for r in entry['members']}
        for row in entry['members']:
            assert not row['member'].startswith('/') and '..' not in Path(row['member']).parts
            data = z.read(row['member'])
            assert len(data) == row['bytes'] and sha(data) == row['sha256'], row['member']
            members += 1
print(json.dumps(dict(status='all_packaged_bytes_verified', payload_files=len(manifest['files']),
                      archived_files=members, actor_executed=False, raw_assets_required=False)))
