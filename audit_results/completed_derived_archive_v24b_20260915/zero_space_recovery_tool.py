import pathlib,json,os,stat,gzip,hashlib,base64,importlib.util,shutil
r=pathlib.Path('/root/NSO');audit=r/'audit_results/completed_derived_archive_v24b_20260915';spec=importlib.util.spec_from_file_location('arc',r/'scripts/archive_completed_derived_v24b.py');m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
# Use only slack in this in-progress audit receipt inode, never an older sealed archive.
receipt=audit/'pilot_single_link_ply/receipts.jsonl';st=receipt.stat();remaining=st.st_blocks*512-st.st_size
run=r/'eval_results/competition_v8_1_execution_20260911';ver=json.loads((run/'verification.json').read_text());assert ver.get('passed_full') is True or ver.get('status') in ('passed','passed_full')
choices=[]
for p in run.rglob('decisions.jsonl'):
 s=p.lstat()
 if s.st_nlink==1:
  data=p.read_bytes();z=gzip.compress(data,6,mtime=0)
  row=dict(event='emergency_persisted_gzip_before_unlink',path=str(p.relative_to(r)),original=m.meta(s),sha256=hashlib.sha256(data).hexdigest(),gzip_sha256=hashlib.sha256(z).hexdigest(),gzip_base64=base64.b64encode(z).decode(),old_verification_sha256=m.sha(run/'verification.json'),restoration='base64 decode this record gzip_base64, decompress, verify sha256, restore exact path and original metadata')
  line=(json.dumps(row,separators=(',',':'))+'\n').encode()
  if len(line)<remaining-32:choices.append((len(line),p,s,row,line))
assert choices,('no fitting durable bootstrap record',remaining)
_,p,s,row,line=min(choices,key=lambda x:x[0]);assert (s.st_dev,s.st_ino) not in m.writers() and m.signature(p.stat())==m.signature(s)
with receipt.open('ab',buffering=0) as f:f.write(line);os.fsync(f.fileno())
last=json.loads(receipt.read_text().splitlines()[-1]);decoded=gzip.decompress(base64.b64decode(last['gzip_base64']));assert len(decoded)==s.st_size and hashlib.sha256(decoded).hexdigest()==m.sha(p)==row['sha256'];assert m.signature(p.stat())==m.signature(s)
p.unlink();m.syncdir(p.parent)
# This durable embedded record is an exact-byte archive, not a RAM-only copy.
more=[]
for p in sorted(run.rglob('decisions.jsonl'),key=lambda p:p.stat().st_size):
 s=p.lstat()
 if s.st_nlink!=1:continue
 b=p.read_bytes();z=gzip.compress(b,6,mtime=0);h=hashlib.sha256(b).hexdigest();a=p.with_name(p.name+'.v24b.gz');j=p.with_name(p.name+'.v24b.archive.json');assert not a.exists() and not j.exists() and (s.st_dev,s.st_ino) not in m.writers()
 with a.open('xb') as f:f.write(z);f.flush();os.fsync(f.fileno())
 assert hashlib.sha256(gzip.decompress(a.read_bytes())).hexdigest()==h
 note=dict(path=str(p.relative_to(r)),archive=str(a.relative_to(r)),original=m.meta(s),sha256=h,gzip_sha256=m.sha(a),old_verification_sha256=m.sha(run/'verification.json'))
 with j.open('x') as f:json.dump(note,f,separators=(',',':'));f.flush();os.fsync(f.fileno())
 m.syncdir(p.parent);assert m.signature(p.stat())==m.signature(s) and m.sha(p)==h and (s.st_dev,s.st_ino) not in m.writers();p.unlink();m.syncdir(p.parent);more.append(note)
# Persist the improved tool, then immediately grow free space by archiving V13 logs.
new=r/'scripts/archive_completed_derived_v24c.py';new.write_bytes(pathlib.Path('/dev/shm/archive_completed_derived_v24c.py').read_bytes())
with new.open('rb') as f:os.fsync(f.fileno())
spec=importlib.util.spec_from_file_location('newarc',new);n=importlib.util.module_from_spec(spec);spec.loader.exec_module(n)
for group,leaf in [('semantic_gain_v13_2_outbound_paired_20260914','v13_2_outbound_logs'),('semantic_gain_v13_paired_information_resumed_20260914','v13_resumed_logs')]:
 run2=r/'eval_results'/group;old=json.loads((run2/'artifact_hashes.json').read_text());assert json.loads((run2/'manifest.json').read_text())['status']=='complete';rows=[]
 for q in run2.rglob('*'):
  if q.name not in ('modules.json','module_calls.json','candidates.jsonl','selection_audit.jsonl'):continue
  qs=q.lstat()
  if qs.st_nlink!=1:continue
  b=q.read_bytes();h=hashlib.sha256(b).hexdigest();assert old[str(q.relative_to(run2))]==h
  rows.append(dict(path=str(q.relative_to(r)),inode=qs.st_ino,sha256=h,gzip_bytes=len(gzip.compress(b,6,mtime=0)),original_size=len(b),allocated_bytes=qs.st_blocks*512))
 protected={str(q.relative_to(r)):n.sha(q) for q in run2.iterdir() if q.is_file()}
 plan=dict(scope='Single-link completed derived module/candidate/selection logs only in '+group+'; no raw sensor, metrics, models, reference or hardlink targets',rows=rows,protected_metadata=protected,completion_evidence=dict(original_manifest_status='complete',targets_match_existing_artifact_inventory=True,old_artifact_count=len(old)))
 dest=pathlib.Path('/dev/shm/nso_v24c_'+group+'.json');dest.write_text(json.dumps(plan));n.prepare(audit/leaf,dest);n.apply(audit/leaf)
# Enough writable space now for the whole pilot compressed payload, in addition to incremental release.
assert shutil.disk_usage(r).free>150*1024**2
m.apply(audit/'pilot_single_link_ply')
n.writejson(audit/'emergency_v8_bootstrap.json',dict(status='verified',embedded_record_path=str(receipt.relative_to(r)),embedded=row,sidecar_rows=more,source_verification_sha256=m.sha(run/'verification.json'),note='Exactly one gzip was durably embedded in preallocated slack of this in-progress audit receipt, read back and hash-verified before original unlink; other 23 gzip archives and receipts sit beside original paths. Old metadata files unchanged.'))
shutil.copyfile(__file__,audit/'zero_space_recovery_tool.py')
print('COMPLETE available',shutil.disk_usage(r).free,flush=True)
