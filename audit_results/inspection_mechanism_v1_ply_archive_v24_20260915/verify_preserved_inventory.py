#!/usr/bin/env python3
"""Verify all old inventory bytes without restoring archived derived meshes."""
import sys
sys.dont_write_bytecode=True
import gzip,hashlib,json,os
from pathlib import Path
from datetime import datetime,timezone
FOLDER=Path(__file__).resolve().parent
ROOT=FOLDER.parents[1]
RUN=ROOT/'eval_results/inspection_mechanism_v1_20260910'
def sha(p):
    with p.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def main():
    manifest=json.loads((FOLDER/'manifest.json').read_text())
    for name,h in manifest['input_sha256'].items():assert sha(RUN/name)==h,name
    inventory=json.loads((RUN/'artifacts_sha256.json').read_text())
    with gzip.open(FOLDER/'target_inventory.json.gz','rt') as f:targets={r['path']:r for r in json.load(f)}
    assert len(targets)==464
    found=set();preserved=archived=total_bytes=0
    for relative,wanted in inventory.items():
        p=RUN/relative;key=str(p.relative_to(ROOT))
        if key in targets:
            row=targets[key];found.add(key)
            a=FOLDER/'archives'/(key+'.gz')
            assert a.is_file() and not a.is_symlink() and a.stat().st_nlink==1,key
            assert sha(a)==row['gzip_sha256'],key
            data=gzip.decompress(a.read_bytes())
            assert hashlib.sha256(data).hexdigest()==wanted==row['sha256'],key
            assert len(data)==row['file_bytes'] and not p.exists(),key
            total_bytes+=len(data);archived+=1
        else:
            assert p.is_file() and sha(p)==wanted,key
            preserved+=1
    assert found==set(targets)
    report=dict(status='passed',utc=datetime.now(timezone.utc).isoformat(),original_inventory_entries=len(inventory),
        original_paths_preserved_sha256_matches=preserved,archived_original_contents_sha256_matches=archived,
        archived_original_bytes=total_bytes,original_metadata_sha256_unchanged=True,
        all_original_inventory_bytes_accounted_for=True,raw_and_non_target_inventory_files_unchanged=True,
        verified_source_sha256=sha(Path(__file__).resolve()))
    temp=FOLDER/'preserved_inventory_verification.json.tmp'
    with temp.open('x') as f:json.dump(report,f,indent=2);f.write('\n');f.flush();os.fsync(f.fileno())
    os.replace(temp,FOLDER/'preserved_inventory_verification.json')
    print(json.dumps(report))
if __name__=='__main__':main()
