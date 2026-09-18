#!/usr/bin/env python3
"""Restore the fixed V24b completed-derived archive to its exact original paths.
Defaults to read-only verification. Mutations require --restore or --rearchive,
plus a new receipt path outside the sealed archive. Original compressed bytes stay.
"""
import os,sys
sys.dont_write_bytecode=True
os.environ['PYTHONDONTWRITEBYTECODE']='1'
import argparse,base64,fcntl,gzip,hashlib,io,json,pathlib,shutil,stat,uuid
ROOT=pathlib.Path('/root/NSO')
DEFAULT=ROOT/'audit_results/completed_derived_archive_v24b_20260915'
CHUNK=128*1024
RESERVE=8*1024**2

def sha(p):
 fd=os.open(p,os.O_RDONLY|os.O_NOFOLLOW|getattr(os,'O_NOATIME',0))
 with os.fdopen(fd,'rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def plain(p):
 p=pathlib.Path(p).absolute()
 if not p.is_relative_to(ROOT):raise ValueError('outside NSO project')
 q=ROOT
 for part in p.relative_to(ROOT).parts:
  q/=part
  if q.is_symlink():raise ValueError('symlink path')
 return p
def syncdir(p):
 fd=os.open(p,os.O_RDONLY|os.O_DIRECTORY)
 try:os.fsync(fd)
 finally:os.close(fd)
def source(row):
 if 'archive' in row:
  p=plain(ROOT/row['archive'])
  if sha(p)!=row['gzip_sha256']:raise ValueError('compressed archive hash mismatch')
  return gzip.open(p,'rb')
 container=plain(ROOT/row['embedded_container']);obj=json.loads(container.read_text());b=base64.b64decode(obj['embedded']['gzip_base64'])
 if hashlib.sha256(b).hexdigest()!=row['gzip_sha256']:raise ValueError('embedded compressed hash mismatch')
 return gzip.GzipFile(fileobj=io.BytesIO(b),mode='rb')
def verify_row(row):
 h=hashlib.sha256();n=0
 with source(row) as f:
  while True:
   b=f.read(CHUNK)
   if not b:break
   h.update(b);n+=len(b)
 if n!=row['original']['size'] or h.hexdigest()!=row['sha256']:raise ValueError('original byte mismatch')
 p=plain(ROOT/row['path'])
 if p.exists() and sha(p)!=row['sha256']:raise ValueError('existing original differs; refusing overwrite')
 return p.exists()
def load(folder):
 seal=json.loads((folder/'restoration_integrity.json').read_text())
 if sha(folder/'restoration_manifest.json.gz')!=seal['restoration_manifest_sha256']:raise ValueError('restoration inventory changed')
 if sha(pathlib.Path(__file__))!=seal['restore_tool_sha256']:raise ValueError('restorer source changed')
 with gzip.open(folder/'restoration_manifest.json.gz','rt') as f:rows=json.load(f)
 if len(rows)!=375 or len({x['path'] for x in rows})!=375:raise ValueError('fixed target count differs')
 for row in rows:plain(ROOT/row['path'])
 return rows
def writers():
 out=set()
 for p in pathlib.Path('/proc').iterdir():
  if not p.name.isdigit():continue
  try:fds=list((p/'fd').iterdir())
  except FileNotFoundError:continue
  for fd in fds:
   try:
    flags=next(int(x.split()[1],8) for x in (p/'fdinfo'/fd.name).read_text().splitlines() if x.startswith('flags:'))
    if flags&os.O_ACCMODE:
     s=fd.stat()
     if stat.S_ISREG(s.st_mode):out.add((s.st_dev,s.st_ino))
   except (FileNotFoundError,ProcessLookupError,StopIteration):continue
 return out
def sig(s):return (s.st_dev,s.st_ino,s.st_size,s.st_mode,s.st_uid,s.st_gid,s.st_nlink,s.st_mtime_ns)
def event(p,obj):
 with p.open('a') as f:f.write(json.dumps(obj,separators=(',',':'))+'\n');f.flush();os.fsync(f.fileno())
def restore(row,receipt):
 p=plain(ROOT/row['path'])
 if verify_row(row):event(receipt,dict(event='already_present_identical',path=row['path']));return
 o=row['original']
 if shutil.disk_usage(ROOT).free<o['allocated_bytes']+RESERVE:raise OSError('not enough actually writable space; retain 8 MiB')
 p.parent.mkdir(parents=True,exist_ok=True);t=p.with_name(p.name+'.restore-'+uuid.uuid4().hex)
 try:
  with source(row) as inp,t.open('xb') as out:
   shutil.copyfileobj(inp,out,CHUNK);out.flush();os.fsync(out.fileno())
  if sha(t)!=row['sha256']:raise ValueError('restored byte mismatch')
  os.chown(t,o['uid'],o['gid']);os.chmod(t,o['mode'])
  for n,b in row.get('xattrs',{}).items():os.setxattr(t,n,base64.b64decode(b),follow_symlinks=False)
  os.utime(t,ns=(o['atime_ns'],o['mtime_ns']))
  with t.open('rb') as f:os.fsync(f.fileno())
  os.link(t,p,follow_symlinks=False);t.unlink();syncdir(p.parent)
  assert sha(p)==row['sha256'] and p.stat().st_mtime_ns==o['mtime_ns']
  event(receipt,dict(event='restored',path=row['path'],sha256=row['sha256'],bytes=o['size'],mode=o['mode'],mtime_ns=o['mtime_ns']))
 finally:
  if t.exists():t.unlink()
def rearchive(row,receipt):
 p=plain(ROOT/row['path']);present=verify_row(row)
 if not present:event(receipt,dict(event='already_archived',path=row['path']));return
 fd=os.open(p,os.O_RDONLY|os.O_NOFOLLOW)
 try:
  fcntl.flock(fd,fcntl.LOCK_EX|fcntl.LOCK_NB);s=os.fstat(fd)
  if not stat.S_ISREG(s.st_mode) or s.st_nlink!=1 or (s.st_dev,s.st_ino) in writers():raise ValueError('unsafe/open/shared original')
  if sha(p)!=row['sha256'] or sig(os.fstat(fd))!=sig(s) or sig(p.lstat())!=sig(s):raise ValueError('original changed')
  event(receipt,dict(event='verified_archive_before_reunlink',path=row['path'],sha256=row['sha256'],current_inode=s.st_ino))
  p.unlink();syncdir(p.parent);event(receipt,dict(event='rearchived',path=row['path']))
 finally:os.close(fd)
def main():
 ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--archive',type=pathlib.Path,default=DEFAULT);g=ap.add_mutually_exclusive_group();g.add_argument('--restore',action='store_true');g.add_argument('--rearchive',action='store_true');ap.add_argument('--only',action='append');ap.add_argument('--receipt',type=pathlib.Path);a=ap.parse_args();folder=plain(a.archive);rows=load(folder)
 if a.only:
  wanted=set(a.only);selected=[r for r in rows if r['path'] in wanted]
  if len(selected)!=len(wanted):raise ValueError('each --only must exactly match an inventoried relative path')
  rows=selected
 if a.restore or a.rearchive:
  if not a.receipt:raise ValueError('--receipt required outside sealed archive')
  receipt=plain(a.receipt)
  if receipt.is_relative_to(folder):raise ValueError('do not change sealed archive contents')
  receipt.parent.mkdir(parents=True,exist_ok=True)
  with receipt.open('x') as f:f.write('');f.flush();os.fsync(f.fileno())
  try:
   for row in rows:(restore if a.restore else rearchive)(row,receipt)
  except BaseException as e:event(receipt,dict(event='failure',error=repr(e)));raise
 present=sum(verify_row(r) for r in rows)
 print(json.dumps(dict(status='verified',targets=len(rows),originals_present=present,all_original_bytes_recoverable=True,required_additional_bytes_to_restore_missing=sum(r['original']['allocated_bytes'] for r in rows if not (ROOT/r['path']).exists())+RESERVE,available_bytes=shutil.disk_usage(ROOT).free)),flush=True)
if __name__=='__main__':main()
