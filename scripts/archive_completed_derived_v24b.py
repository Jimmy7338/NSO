#!/usr/bin/env python3
"""Lossless archive of an explicitly frozen completed NSO derived-file inventory.
No model, reference, raw sensor, or earlier archive is an eligible target.
"""
import os,sys
sys.dont_write_bytecode=True
os.environ['PYTHONDONTWRITEBYTECODE']='1'
import argparse,base64,fcntl,gzip,hashlib,json,pathlib,shutil,stat,time,uuid
ROOT=pathlib.Path('/root/NSO')
RESERVE=8*1024**2
CHUNK=128*1024

def sha(p):
 with open(p,'rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def syncdir(p):
 fd=os.open(p,os.O_RDONLY|os.O_DIRECTORY)
 try:os.fsync(fd)
 finally:os.close(fd)
def physical_free():
 v=os.statvfs(ROOT);return (v.f_bfree if os.geteuid()==0 else v.f_bavail)*v.f_frsize
def require(n=0):
 if physical_free()<RESERVE+n:raise OSError('insufficient physical free space, preserving 8 MiB')
def writejson(p,obj):
 data=(json.dumps(obj,ensure_ascii=False,indent=2)+'\n').encode();require(len(data)+4096)
 t=p.with_name(p.name+'.tmp-'+uuid.uuid4().hex)
 try:
  with t.open('xb') as f:f.write(data);f.flush();os.fsync(f.fileno())
  os.replace(t,p);syncdir(p.parent)
 finally:
  if t.exists():t.unlink()
def plain(p):
 p=pathlib.Path(p)
 if not p.is_relative_to(ROOT):raise ValueError('outside NSO')
 q=ROOT
 for part in p.relative_to(ROOT).parts:
  q/=part
  if q.is_symlink():raise ValueError('symlink path: '+str(q))
 return p
def meta(s):
 return dict(dev=s.st_dev,inode=s.st_ino,size=s.st_size,mode=stat.S_IMODE(s.st_mode),uid=s.st_uid,gid=s.st_gid,nlink=s.st_nlink,atime_ns=s.st_atime_ns,mtime_ns=s.st_mtime_ns,ctime_ns=s.st_ctime_ns,allocated_bytes=s.st_blocks*512)
def signature(s):return (s.st_dev,s.st_ino,s.st_size,s.st_mode,s.st_uid,s.st_gid,s.st_nlink,s.st_mtime_ns)
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
def event(folder,obj):
 require(16384)
 with (folder/'receipts.jsonl').open('a') as f:f.write(json.dumps(obj,separators=(',',':'))+'\n');f.flush();os.fsync(f.fileno())
def archive_path(folder,row):return plain(folder/'archives'/(row['path']+'.gz'))
def check_gzip(p,row):
 if not stat.S_ISREG(p.lstat().st_mode) or p.stat().st_nlink!=1:raise ValueError('archive not single-link regular')
 h=hashlib.sha256();n=0
 with gzip.open(p,'rb') as f:
  while True:
   b=f.read(CHUNK)
   if not b:break
   h.update(b);n+=len(b)
 if n!=row['original']['size'] or h.hexdigest()!=row['sha256']:raise ValueError('decompressed bytes differ')
 return sha(p)
def seal(folder):
 writejson(folder/'artifact_hashes.json',{str(p.relative_to(folder)):sha(p) for p in sorted(folder.rglob('*')) if p.is_file() and p.name!='artifact_hashes.json'})
def load(folder):
 m=json.loads((folder/'manifest.json').read_text())
 if sha(folder/'target_inventory.json.gz')!=m['inventory_sha256']:raise ValueError('target inventory changed')
 if sha(pathlib.Path(__file__))!=m['tool_sha256']:raise ValueError('tool changed')
 with gzip.open(folder/'target_inventory.json.gz','rt') as f:rows=json.load(f)
 for name,h in m['protected_metadata'].items():
  if sha(plain(ROOT/name))!=h:raise ValueError('completed run metadata changed')
 if len(rows)!=m['target_count']:raise ValueError('target count changed')
 return m,rows

def prepare(folder,planpath):
 plan=json.loads(planpath.read_text());rows=plan['rows'];active=writers()
 if folder.exists():raise FileExistsError(folder)
 for row in rows:
  p=plain(ROOT/row['path']);s=p.lstat()
  if not stat.S_ISREG(s.st_mode) or s.st_nlink!=1 or (s.st_dev,s.st_ino) in active:raise ValueError('unsafe original '+row['path'])
  if s.st_ino!=row['inode'] or sha(p)!=row['sha256']:raise ValueError('review identity changed')
  row['original']=meta(s)
  row['xattrs']={n:base64.b64encode(os.getxattr(p,n,follow_symlinks=False)).decode() for n in os.listxattr(p,follow_symlinks=False)}
 rows.sort(key=lambda x:x['gzip_bytes'])
 require(1024*1024);folder.mkdir(parents=True)
 inventory=gzip.compress(json.dumps(rows,separators=(',',':')).encode(),6,mtime=0)
 with (folder/'target_inventory.json.gz').open('xb') as f:f.write(inventory);f.flush();os.fsync(f.fileno())
 shutil.copyfile(pathlib.Path(__file__),folder/'archive_tool.py')
 with (folder/'archive_tool.py').open('rb') as f:os.fsync(f.fileno())
 m=dict(status='prepared',created_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),scope=plan['scope'],target_count=len(rows),inventory_sha256=sha(folder/'target_inventory.json.gz'),tool_sha256=sha(pathlib.Path(__file__)),protected_metadata=plan['protected_metadata'],completion_evidence=plan['completion_evidence'],free_before_apply=dict(bavail_bytes=shutil.disk_usage(ROOT).free,physical_free_bytes=physical_free()),metadata_limits='Original uid/gid/mode/atime/mtime/xattrs are restored; inode and ctime are recorded but cannot be reinstated by normal filesystem APIs.',old_manifests_unchanged=True,strict_old_checks_require_restore=True)
 writejson(folder/'manifest.json',m);seal(folder)
 print('prepared',len(rows),flush=True)

def apply(folder):
 m,rows=load(folder);m['status']='archiving';writejson(folder/'manifest.json',m)
 try:
  for i,row in enumerate(rows):
   p=plain(ROOT/row['path']);a=archive_path(folder,row)
   if not p.exists():check_gzip(a,row);continue
   fd=os.open(p,os.O_RDONLY|os.O_NOFOLLOW)
   try:
    fcntl.flock(fd,fcntl.LOCK_EX|fcntl.LOCK_NB);s=os.fstat(fd)
    if s.st_ino!=row['inode'] or not stat.S_ISREG(s.st_mode) or s.st_nlink!=1:raise ValueError('unsafe/changed inode')
    if sha(p)!=row['sha256']:raise ValueError('source hash changed')
    if not a.exists():
     require(row['gzip_bytes']+128*1024);a.parent.mkdir(parents=True,exist_ok=True)
     t=a.with_name(a.name+'.tmp-'+uuid.uuid4().hex)
     try:
      with t.open('xb') as out,os.fdopen(os.dup(fd),'rb') as inp:
       os.lseek(fd,0,os.SEEK_SET)
       with gzip.GzipFile(filename='',mode='wb',fileobj=out,compresslevel=6,mtime=0) as z:
        while True:
         b=inp.read(CHUNK)
         if not b:break
         require(len(b)+16384);z.write(b)
       out.flush();os.fsync(out.fileno())
      ghash=check_gzip(t,row)
      os.link(t,a,follow_symlinks=False);t.unlink();syncdir(a.parent)
     finally:
      if t.exists():t.unlink()
    ghash=check_gzip(a,row)
    if signature(os.fstat(fd))!=signature(s) or signature(p.lstat())!=signature(s) or (s.st_dev,s.st_ino) in writers():raise ValueError('original changed/open writer before unlink')
    event(folder,dict(event='verified_before_unlink',path=row['path'],original_sha256=row['sha256'],gzip_sha256=ghash,gzip_bytes=a.stat().st_size,original=meta(s)))
    p.unlink();syncdir(p.parent)
    event(folder,dict(event='archived',path=row['path'],released_original_allocated_bytes=s.st_blocks*512))
   finally:os.close(fd)
   if (i+1)%10==0:print('archived',i+1,'/',len(rows),'available_MiB',round(shutil.disk_usage(ROOT).free/1024**2,2),flush=True)
  v=verify(folder,False);m['status']='archived';writejson(folder/'manifest.json',m);writejson(folder/'verification.json',v);seal(folder);print(json.dumps(v),flush=True)
 except BaseException as e:
  try:event(folder,dict(event='failure',error=repr(e)));m['status']='interrupted_recoverable';writejson(folder/'manifest.json',m)
  except BaseException:pass
  raise

def verify(folder,publish=True):
 m,rows=load(folder);archived=present=allocated=removed=0
 for row in rows:
  p=plain(ROOT/row['path']);a=archive_path(folder,row)
  if a.exists():check_gzip(a,row);allocated+=a.stat().st_blocks*512
  elif not p.exists():raise ValueError('missing both original and archive')
  if p.exists():
   if sha(p)!=row['sha256']:raise ValueError('original hash differs')
   present+=1
  else:archived+=1;removed+=row['original']['allocated_bytes']
 v=dict(status='verified',target_count=len(rows),archived_original_paths=archived,original_paths_present=present,all_original_bytes_recoverable=True,protected_metadata_unchanged=True,archive_payload_allocated_bytes=allocated,removed_original_allocated_bytes=removed,net_payload_allocated_bytes_released=removed-allocated,available_bytes=shutil.disk_usage(ROOT).free,physical_free_bytes=physical_free(),old_strict_verification_requires_restore=bool(archived))
 if publish:writejson(folder/'verification.json',v);seal(folder)
 return v

def restore(folder,only):
 m,rows=load(folder)
 if only:
  rows=[r for r in rows if r['path']==only]
  if len(rows)!=1:raise ValueError('--only must match exactly one inventoried path')
 for row in rows:
  p=plain(ROOT/row['path']);a=archive_path(folder,row)
  if p.exists():
   if sha(p)!=row['sha256']:raise ValueError('existing original differs; refusing overwrite')
   continue
  check_gzip(a,row);require(row['original']['allocated_bytes']+128*1024);p.parent.mkdir(parents=True,exist_ok=True)
  t=p.with_name(p.name+'.restore-'+uuid.uuid4().hex);o=row['original']
  try:
   with gzip.open(a,'rb') as inp,t.open('xb') as out:shutil.copyfileobj(inp,out,CHUNK);out.flush();os.fsync(out.fileno())
   if sha(t)!=row['sha256']:raise ValueError('restore bytes mismatch')
   os.chown(t,o['uid'],o['gid']);os.chmod(t,o['mode'])
   for n,b in row['xattrs'].items():os.setxattr(t,n,base64.b64decode(b),follow_symlinks=False)
   os.utime(t,ns=(o['atime_ns'],o['mtime_ns']))
   with t.open('rb') as f:os.fsync(f.fileno())
   os.link(t,p,follow_symlinks=False);t.unlink();syncdir(p.parent)
   event(folder,dict(event='restored',path=row['path'],sha256=row['sha256']))
  finally:
   if t.exists():t.unlink()
 print(json.dumps(verify(folder)),flush=True)

def main():
 ap=argparse.ArgumentParser();ap.add_argument('action',choices=['prepare','apply','verify','restore']);ap.add_argument('--archive',type=pathlib.Path,required=True);ap.add_argument('--plan',type=pathlib.Path);ap.add_argument('--only');a=ap.parse_args();folder=plain(a.archive.absolute())
 if a.action=='prepare':prepare(folder,a.plan)
 elif a.action=='apply':apply(folder)
 elif a.action=='verify':print(json.dumps(verify(folder)),flush=True)
 else:restore(folder,a.only)
if __name__=='__main__':main()
