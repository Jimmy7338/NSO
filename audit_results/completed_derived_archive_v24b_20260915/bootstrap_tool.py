import os,sys,pathlib,json,hashlib,stat,importlib.util,uuid,shutil,subprocess
sys.dont_write_bytecode=True
r=pathlib.Path('/root/NSO');spec=importlib.util.spec_from_file_location('arc',r/'scripts/archive_completed_derived_v24b.py');m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
base=r/'audit_results/facility_choice_v24_shape_prefix_20260915';inv=json.loads((base/'artifact_hashes.json').read_text());assert len(inv)==24
assert json.loads((base/'manifest.json').read_text())['status']=='complete'
for rel,h in inv.items():assert m.sha(base/rel)==h
sets=sorted(str(p.relative_to(base)) for p in base.rglob('*') if p.is_file());original_inv_sha=m.sha(base/'artifact_hashes.json');groups=json.loads(pathlib.Path('/dev/shm/nso_v24b_bootstrap_dups.json').read_text());jobs=[]
for group in groups:
 files=group['files'];anchor=r/files[0]['path']
 for row in files[1:]:jobs.append((anchor,row,group['sha256']))
jobs.sort(key=lambda x:-x[1]['allocated']);active=m.writers();receipts=[];free0=shutil.disk_usage(r).free
for anchor,row,h in jobs:
 p=r/row['path'];s=p.lstat();a=anchor.lstat()
 assert stat.S_ISREG(s.st_mode) and s.st_nlink==1 and s.st_ino==row['ino'] and s.st_size==row['size'] and not p.is_symlink() and not anchor.is_symlink()
 assert (s.st_dev,s.st_ino) not in active and (a.st_dev,a.st_ino) not in active and m.sha(p)==h and m.sha(anchor)==h
 original=m.meta(s);tmp=p.with_name(p.name+'.dedup-'+uuid.uuid4().hex);os.link(anchor,tmp,follow_symlinks=False)
 assert m.sha(tmp)==h and m.signature(p.lstat())==m.signature(s)
 os.replace(tmp,p);m.syncdir(p.parent);assert p.stat().st_ino==anchor.stat().st_ino and m.sha(p)==h
 receipts.append(dict(path=row['path'],anchor=str(anchor.relative_to(r)),sha256=h,original=original,anchor_metadata=m.meta(a),released_allocated_bytes=s.st_blocks*512,mtime_change_explicit=True))
 # First atomic replacement frees enough space for the small durable receipt.
 audit=r/'audit_results/completed_derived_archive_v24b_20260915';audit.mkdir(exist_ok=True)
 m.writejson(audit/'bootstrap_exact_dedup.json',dict(status='deduplicating',rows=receipts,original_inventory_sha256=original_inv_sha,original_artifact_paths=sets,available_before=free0))
for rel,h in inv.items():assert m.sha(base/rel)==h
assert sets==sorted(str(p.relative_to(base)) for p in base.rglob('*') if p.is_file()) and m.sha(base/'artifact_hashes.json')==original_inv_sha
m.writejson(audit/'bootstrap_exact_dedup.json',dict(status='verified',rows=receipts,original_inventory_sha256=original_inv_sha,original_artifact_paths=sets,available_before=free0,available_after=shutil.disk_usage(r).free,released_allocated_bytes=sum(x['released_allocated_bytes'] for x in receipts),all_24_original_hashes_and_file_set_unchanged=True,scope='Root-authorized one-off exact-byte dedup of 11 completed shape-output NPZ paths; all paths retained, original differing mtimes recorded; never modify shared anchors in place.'))
shutil.copyfile('/dev/shm/nso_v24b_pyc_cleanup.json',audit/'bootstrap_pyc_cleanup.json');shutil.copyfile(__file__,audit/'bootstrap_tool.py')
print('bootstrap duplicate paths',len(receipts),'available',shutil.disk_usage(r).free,flush=True)
m.prepare(audit/'bootstrap_v9_decisions',pathlib.Path('/dev/shm/nso_v24b_bootstrap_plan.json'));m.apply(audit/'bootstrap_v9_decisions')
m.prepare(audit/'pilot_single_link_ply',pathlib.Path('/dev/shm/nso_v24b_pilot_plan.json'));m.apply(audit/'pilot_single_link_ply')
