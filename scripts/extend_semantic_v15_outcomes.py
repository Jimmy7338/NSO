#!/usr/bin/env python3
"""Extend verified V15 smoke to the frozen universe; lossless log storage."""
import os
os.environ.setdefault('OMP_NUM_THREADS','1')
os.environ.setdefault('OPENBLAS_NUM_THREADS','1')
import argparse
from datetime import datetime, timezone
import gzip
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys
import zipfile
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from scripts.collect_semantic_v15_outcomes import (
    branch_name, run_branch, seal, verify_inputs, verify_sources)
from scripts.collect_semantic_gain_v13_history import sha,write


def compact_verified_logs(folder):
    """Preserve exact JSON bytes, including formatting; never quantize data."""
    if json.loads((folder/'verification.json').read_text())['status']!='passed':
        raise ValueError('only independently replayed branch logs may be compressed')
    index={}
    for name in ('all_decisions.json','module_calls.json','runtime_audit.json','terminal.json'):
        path=folder/name;raw=path.read_bytes();target=folder/(name+'.gz')
        with target.open('xb') as f:f.write(gzip.compress(raw,compresslevel=6,mtime=0))
        recovered=gzip.decompress(target.read_bytes())
        if recovered!=raw:raise ValueError('lossless log round trip failed')
        index[name]=dict(stored_file=target.name,raw_sha256=hashlib.sha256(raw).hexdigest(),
            compressed_sha256=sha(target),raw_bytes=len(raw),compressed_bytes=target.stat().st_size)
    # Publish recovery identities before removing redundant uncompressed copies.
    write(folder/'archive_index.json',dict(schema='lossless_json_logs/1',files=index,
        recovery='gzip.decompress(stored_file); verify raw_sha256 before restoring original path'))
    for name in index:(folder/name).unlink()
    return sum(r['raw_bytes']-r['compressed_bytes'] for r in index.values())


def main():
    p=argparse.ArgumentParser();p.add_argument('--smoke',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    sm=json.loads((a.smoke/'manifest.json').read_text())
    summary=json.loads((a.smoke/'summary.json').read_text())
    if not (sm['status']=='complete' and summary['outcomes']==summary['replayed']==4
            and summary['collisions']==summary['failed']==summary['budget_violations']==0
            and summary['returns']==4):
        raise ValueError('completed safe four-branch smoke and replay required')
    verify_sources(sm);verify_inputs(sm['protocol'])
    for name,h in json.loads((a.smoke/'artifact_hashes.json').read_text()).items():
        if sha(a.smoke/name)!=h:raise ValueError('smoke evidence changed')
    protocol=sm['protocol']
    declarations=[dict(structure_seed=c['structure_seed'],action_id=s['action_id'],candidate_id=cid)
        for s in protocol['states'] for c in s['certificates'] for cid in c['candidate_ids']]
    if len(declarations)!=protocol['expected_unique_candidate_outcomes']:raise ValueError('universe mismatch')
    frozen={**sm['source_sha256'],str(Path(__file__).resolve().relative_to(ROOT)):sha(Path(__file__))}
    for name in ('scripts/analyze_semantic_v15_outcomes.py','nso/information_value_v15.py',
                 'tests/virtual3d/test_information_value_v15.py'):
        frozen[name]=sha(ROOT/name)
    m=dict(status='running',protocol=protocol,source_sha256=frozen,declarations=declarations,
        created_utc=datetime.now(timezone.utc).isoformat(),training_allowed=False,
        reused_smoke=str(a.smoke),smoke_inventory_sha256=sha(a.smoke/'artifact_hashes.json'),
        storage='After independent replay, four verbose JSON logs use exact-byte gzip with raw/compressed hashes; byte-identical prefix meshes share an inode. Physical packet and mesh contents unchanged.',
        completed=[])
    a.output.mkdir(parents=True,exist_ok=False);write(a.output/'manifest.json',m)
    with zipfile.ZipFile(a.output/'sources.zip','x',zipfile.ZIP_DEFLATED) as z:
        for name in frozen:z.write(ROOT/name,name)
    old={branch_name(d) for d in sm['declarations']};rows=[];reused=[];saved=0;prefixes={};mesh_saved=0
    try:
        for index,d in enumerate(declarations):
            name=branch_name(d);folder=a.output/name
            if shutil.disk_usage(a.output).free < 48 * 1024**2:
                raise RuntimeError('less than 48 MiB free before next branch; completed evidence retained')
            if name in old:
                original=a.smoke/name
                if json.loads((original/'verification.json').read_text())['status']!='passed':
                    raise ValueError('reuse lacks physical replay')
                shutil.copytree(original,folder)
                row=json.loads((folder/'result.json').read_text())
                if any(row[k]!=d[k] for k in d):raise ValueError('reuse declaration mismatch')
                reused.append(name)
            else:
                row=run_branch(a.output,protocol,d)
                subprocess.run([sys.executable,str(ROOT/'scripts/collect_semantic_v15_outcomes.py'),
                    '--output',str(a.output),'--replay-index',str(index)],check=True)
            saved+=compact_verified_logs(folder)
            prefix=folder/'prefix_mesh.npz';key=sha(prefix)
            if key in prefixes:
                prior=prefixes[key]
                if prefix.read_bytes()!=prior.read_bytes():raise ValueError('prefix digest collision')
                temporary=folder/'prefix_mesh.link.npz'
                os.link(prior,temporary);mesh_saved+=prefix.stat().st_size
                os.replace(temporary,prefix)
            else:prefixes[key]=prefix
            rows.append(row);write(a.output/'partial.json',rows)
            m['completed'].append(name);write(a.output/'manifest.json',m)
            print(json.dumps(dict(completed=len(rows),expected=len(declarations),branch=name,
                reused=name in old,returned=row['termination']['returned_to_anchor'],
                joint=row['after']['2026']['joint_05cm'])),flush=True)
        verify_sources(m);verify_inputs(protocol)
        write(a.output/'summary.json',dict(status='complete',outcomes=len(rows),replayed=len(rows),
            reused=len(reused),newly_executed=len(rows)-len(reused),reused_branches=reused,
            lossless_log_bytes_saved=saved,duplicate_prefix_bytes_saved=mesh_saved,
            collisions=sum(r['collisions'] for r in rows),
            failed=sum(r['termination']['failed'] for r in rows),
            returns=sum(r['termination']['returned_to_anchor'] for r in rows),
            budget_violations=sum(r['total_paid_actions']>r['total_budget'] for r in rows),
            first_targets_reached=sum(r['first_target_reached'] for r in rows),
            parent_groups=1,training_allowed=False,semantic_efficacy_proven=False))
        m['status']='complete'
    except Exception as error:m.update(status='failed',error=repr(error));raise
    finally:write(a.output/'manifest.json',m);seal(a.output)


if __name__=='__main__':main()
