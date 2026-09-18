#!/usr/bin/env python3
"""Byte-exact restore for one frozen linked-PLY archive; default is read-only.
Use --restore/--rearchive --only ORIGINAL [--group] --receipt NEW_PROJECT_PATH.
"""
import sys,os
sys.dont_write_bytecode=True
import argparse,base64,gzip,hashlib,json,pathlib,stat,uuid
ROOT=pathlib.Path('/root/NSO')
DEFAULT=ROOT/'audit_results/virtual3d_pilot_linked_mesh_archive_v25_20260915'
RESERVE=64*1024**2
CHUNK=128*1024

def plain(p):
 p=pathlib.Path(p).absolute()
 if not p.is_relative_to(ROOT):raise ValueError('outside project')
 q=ROOT
 for s in p.relative_to(ROOT).parts:
  q/=s
  if q.is_symlink():raise ValueError('symlink path')
 return p

def sha(p):
 fd=os.open(p,os.O_RDONLY|os.O_NOFOLLOW|os.O_NOATIME)
 with os.fdopen(fd,'rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()

def free():s=os.statvfs(ROOT);return s.f_bavail*s.f_frsize

def syncdir(p):
 fd=os.open(p,os.O_RDONLY|os.O_DIRECTORY)
 try:os.fsync(fd)
 finally:os.close(fd)

def writer_keys():
 keys=set()
 for p in pathlib.Path('/proc').iterdir():
  if not p.name.isdigit():continue
  try:fds=list((p/'fd').iterdir())
  except OSError:continue
  for f in fds:
   try:
    flags=int(next(x.split()[1] for x in (p/'fdinfo'/f.name).read_text().splitlines() if x.startswith('flags:')),8)
    if flags&os.O_ACCMODE:
     s=f.stat()
     if stat.S_ISREG(s.st_mode):keys.add((s.st_dev,s.st_ino))
   except (OSError,StopIteration):continue
 return keys

def event(p,d):
 with p.open('a') as f:f.write(json.dumps(d,separators=(',',':'))+'\n');f.flush();os.fsync(f.fileno())

def load(folder,allow_incomplete=False):
 seal=folder/'restoration_integrity.json'
 if seal.exists():
  d=json.loads(seal.read_bytes());inv=folder/'restoration_inventory.json'
  if sha(inv)!=d['inventory_sha256'] or sha(pathlib.Path(__file__))!=d['restore_tool_sha256']:raise ValueError('inventory/tool changed')
  return json.loads(inv.read_bytes())['groups']
 if not allow_incomplete:raise ValueError('archive not sealed; explicit --allow-incomplete needed for recovery')
 prepared=json.loads((folder/'prepared_inventory.json').read_bytes());groups={g['id']:g for g in prepared['groups']};ready={}
 p=folder/'events.jsonl'
 if p.exists():
  for line in p.read_text().splitlines():
   try:r=json.loads(line)
   except ValueError:continue
   if r.get('event')=='gzip_verified':ready[r['id']]={**groups[r['id']],**r['gzip']}
 if not ready:raise ValueError('no durably verified archive groups')
 return list(ready.values())

def verify(g):
 p=plain(ROOT/g['archive']);h=hashlib.sha256();n=0
 if sha(p)!=g['gzip_sha256']:raise ValueError('gzip byte hash mismatch')
 with gzip.open(p,'rb') as f:
  while b:=f.read(CHUNK):h.update(b);n+=len(b)
 if n!=g['size'] or h.hexdigest()!=g['sha256']:raise ValueError('decompressed byte mismatch')
 present=[]
 for a in g['aliases']:
  q=plain(ROOT/a['path'])
  if q.exists():
   if sha(q)!=g['sha256']:raise ValueError('existing alias differs; no overwrite')
   present.append(q)
 return present

def setmeta(p,a):
 os.chown(p,a['uid'],a['gid']);os.chmod(p,int(a['mode'],8))
 for k,v in a['xattrs'].items():os.setxattr(p,k,base64.b64decode(v),follow_symlinks=False)
 os.utime(p,ns=(a['atime_ns'],a['mtime_ns']))
 fd=os.open(p,os.O_RDONLY|os.O_NOATIME)
 try:os.fsync(fd)
 finally:os.close(fd)

def restore(g,selected,receipt):
 present=verify(g);keys={(p.stat().st_dev,p.stat().st_ino) for p in present}
 if len(keys)>1:raise ValueError('existing aliases have separate inodes; refuse silent relinking')
 if keys&writer_keys():raise ValueError('open writer')
 anchor=present[0] if present else None
 for a in g['aliases']:
  if a['path'] not in selected:continue
  p=plain(ROOT/a['path'])
  if p.exists():event(receipt,{'event':'already_restored','path':a['path']});continue
  p.parent.mkdir(parents=True,exist_ok=True)
  if anchor is None:
   if free()<a['allocated_bytes']+RESERVE:raise OSError('restore needs original bytes plus 64MiB reserve')
   t=p.with_name(p.name+'.restore-'+uuid.uuid4().hex)
   try:
    with gzip.open(plain(ROOT/g['archive']),'rb') as inp,t.open('xb') as out:
     while b:=inp.read(CHUNK):
      if free()<RESERVE+len(b):raise OSError('64MiB reserve')
      out.write(b)
     out.flush();os.fsync(out.fileno())
    if sha(t)!=g['sha256']:raise ValueError('restored SHA mismatch')
    setmeta(t,a);os.link(t,p,follow_symlinks=False);t.unlink();syncdir(p.parent);anchor=p
   finally:
    if t.exists():t.unlink()
  else:
   if free()<RESERVE+8192:raise OSError('64MiB reserve')
   os.link(anchor,p,follow_symlinks=False);syncdir(p.parent)
  assert sha(p)==g['sha256'] and p.stat().st_mtime_ns==a['mtime_ns']
  event(receipt,{'event':'restored','path':a['path'],'sha256':g['sha256'],'inode':p.stat().st_ino,'nlink':p.stat().st_nlink})
 return verify(g)

def rearchive(g,selected,receipt):
 present=verify(g);keys={(p.stat().st_dev,p.stat().st_ino) for p in present}
 if len(keys)>1:raise ValueError('separate existing alias inodes')
 if not present:return []
 s=present[0].stat()
 if s.st_nlink!=len(present):raise ValueError('unlisted hardlink; refuse unlink')
 if keys&writer_keys():raise ValueError('open writer')
 left=len(present)
 for p in present:
  if str(p.relative_to(ROOT)) not in selected:continue
  t=p.lstat()
  if (t.st_dev,t.st_ino,t.st_nlink)!=(s.st_dev,s.st_ino,left):raise ValueError('alias closure changed')
  if sha(p)!=g['sha256']:raise ValueError('alias changed')
  event(receipt,{'event':'verified_before_reunlink','path':str(p.relative_to(ROOT)),'sha256':g['sha256']})
  p.unlink();left-=1;syncdir(p.parent);event(receipt,{'event':'rearchived','path':str(p.relative_to(ROOT))})
 return verify(g)

def main():
 ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--archive',type=pathlib.Path,default=DEFAULT);m=ap.add_mutually_exclusive_group();m.add_argument('--restore',action='store_true');m.add_argument('--rearchive',action='store_true');ap.add_argument('--only',action='append');ap.add_argument('--group',action='store_true');ap.add_argument('--receipt',type=pathlib.Path);ap.add_argument('--allow-incomplete',action='store_true');args=ap.parse_args();folder=plain(args.archive);groups=load(folder,args.allow_incomplete)
 wanted=set(args.only or [a['path'] for g in groups for a in g['aliases']]);known={a['path'] for g in groups for a in g['aliases']}
 if wanted-known:raise ValueError('unknown original path')
 groups=[g for g in groups if any(a['path'] in wanted for a in g['aliases'])]
 if args.group:wanted={a['path'] for g in groups for a in g['aliases']}
 if args.restore or args.rearchive:
  if not args.receipt:raise ValueError('new --receipt required')
  receipt=plain(args.receipt)
  if receipt.is_relative_to(folder):raise ValueError('receipt must be outside sealed archive')
  if free()<RESERVE+65536:raise OSError('64MiB reserve')
  receipt.parent.mkdir(parents=True,exist_ok=True)
  with receipt.open('x') as f:f.flush();os.fsync(f.fileno())
  try:
   for g in groups:(restore if args.restore else rearchive)(g,wanted,receipt)
  except BaseException as e:event(receipt,{'event':'failure','error':repr(e)});raise
 present=sum(len(verify(g)) for g in groups)
 print(json.dumps({'status':'verified','groups':len(groups),'selected_paths':len(wanted),'group_original_paths_present':present,'gzip_preserved':True,'free_bytes':free()}))
if __name__=='__main__':main()
