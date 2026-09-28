#!/usr/bin/env python3
"""Archive exactly the authorized 55-inode/133-alias completed pilot group."""
import sys,os
sys.dont_write_bytecode=True
import base64,gzip,hashlib,json,pathlib,stat,time
from restore_pilot_linked_meshes_v25 import ROOT,DEFAULT,RESERVE,CHUNK,sha,plain,syncdir,writer_keys,event,free
SOURCE=ROOT/'audit_results/v25_runtime_capacity_20260915/remaining_ply_archive_plan.json'
RESTORE=ROOT/'scripts/restore_pilot_linked_meshes_v25.py'

def write(p,d):
 b=json.dumps(d,ensure_ascii=False,indent=2).encode()
 if free()<RESERVE+len(b)+8192:raise OSError('64MiB reserve before receipt')
 with p.open('xb') as f:f.write(b);f.flush();os.fsync(f.fileno())
 return hashlib.sha256(b).hexdigest()

def metadata(p):
 s=p.lstat();assert stat.S_ISREG(s.st_mode)
 return dict(path=str(p.relative_to(ROOT)),size=s.st_size,allocated_bytes=s.st_blocks*512,dev=s.st_dev,inode=s.st_ino,nlink=s.st_nlink,mode=oct(stat.S_IMODE(s.st_mode)),uid=s.st_uid,gid=s.st_gid,atime_ns=s.st_atime_ns,mtime_ns=s.st_mtime_ns,ctime_ns=s.st_ctime_ns,xattrs={n:base64.b64encode(os.getxattr(p,n)).decode() for n in os.listxattr(p)})

def main():
 assert sha(SOURCE)=='565f4bf986c292802280b11dd5192d1264cd1d512cc6984451c23e261151be7e';audit=json.loads(SOURCE.read_bytes());raw=[g for g in audit['proposed_inode_groups'] if all(pathlib.Path(m['path']).parts[1]=='virtual3d_pilot_v1_20260910' for m in g['hardlink_closure_paths'])];assert len(raw)==55 and sum(len(g['hardlink_closure_paths']) for g in raw)==133
 assert not DEFAULT.exists();assert free()>RESERVE+4*1024**2
 groups=[];protected={};start_free=free()
 for i,g in enumerate(raw):
  aliases=[]
  for m in sorted(g['hardlink_closure_paths'],key=lambda x:x['path']):
   p=plain(ROOT/m['path']);now=metadata(p)
   for k in ['size','allocated_bytes','dev','inode','nlink','mode','uid','gid','mtime_ns','ctime_ns']:assert now[k]==m[k],(m['path'],k)
   assert p.suffix=='.ply' and p.name not in ('scene.ply','truth.ply') and 'reference' not in p.name and not p.name.endswith('_truth.ply')
   aliases.append(now)
  assert len(aliases)==aliases[0]['nlink'];assert len({(x['dev'],x['inode']) for x in aliases})==1
  assert sha(ROOT/aliases[0]['path'])==g['sha256']
  groups.append({'id':f'group_{i:03d}','sha256':g['sha256'],'size':aliases[0]['size'],'original_allocated_bytes':aliases[0]['allocated_bytes'],'aliases':aliases,'archive':str((DEFAULT/'blobs'/f'group_{i:03d}.ply.gz').relative_to(ROOT))})
 keys={(g['aliases'][0]['dev'],g['aliases'][0]['inode']) for g in groups};assert not keys&writer_keys()
 for name in ['run_metadata.json','verification.json','config.json','episodes.jsonl']:
  p=ROOT/'eval_results/virtual3d_pilot_v1_20260910'/name;protected[str(p.relative_to(ROOT))]=sha(p)
 assert json.loads((ROOT/'eval_results/virtual3d_pilot_v1_20260910/run_metadata.json').read_bytes())['status']=='complete'
 DEFAULT.mkdir();(DEFAULT/'blobs').mkdir();syncdir(DEFAULT.parent)
 log=DEFAULT/'events.jsonl';log.touch();syncdir(DEFAULT)
 prepared={'status':'prepared_before_any_alias_unlink','audit_sha256':sha(SOURCE),'free_before':start_free,'reserve_bytes':RESERVE,'groups':groups,'protected_completed_receipts':protected,'scope':'Only authorized completed pilot derived PLY; old archive receipts and cleanup ENOSPC evidence unchanged.'};write(DEFAULT/'prepared_inventory.json',prepared)
 done=[]
 try:
  for g in groups:
   aliases=g['aliases'];p=ROOT/aliases[0]['path'];expected=(aliases[0]['dev'],aliases[0]['inode']);assert not {expected}&writer_keys()
   for a in aliases:
    s=(ROOT/a['path']).lstat();assert (s.st_dev,s.st_ino,s.st_nlink,s.st_size,s.st_mtime_ns)==(*expected,len(aliases),g['size'],a['mtime_ns'])
   out=plain(ROOT/g['archive']);temp=out.with_suffix(out.suffix+'.tmp');assert not out.exists() and not temp.exists()
   # Stream into a durable disk gzip; no original alias is removed yet.
   fd=os.open(p,os.O_RDONLY|os.O_NOATIME|os.O_NOFOLLOW)
   try:
    with os.fdopen(fd,'rb') as inp,temp.open('xb') as raw_out:
     with gzip.GzipFile(filename='',mode='wb',fileobj=raw_out,compresslevel=6,mtime=0) as z:
      while b:=inp.read(CHUNK):
       if free()<RESERVE+2*CHUNK:raise OSError('64MiB reserve while compressing')
       z.write(b)
     raw_out.flush();os.fsync(raw_out.fileno())
    h=hashlib.sha256();n=0
    with gzip.open(temp,'rb') as z:
     while b:=z.read(CHUNK):h.update(b);n+=len(b)
    assert n==g['size'] and h.hexdigest()==g['sha256'];os.replace(temp,out);syncdir(out.parent)
   except BaseException:
    # A partial temp is this task's own regenerable output; originals remain.
    if temp.exists():temp.unlink();syncdir(temp.parent)
    raise
   g.update(gzip_sha256=sha(out),gzip_size=out.stat().st_size,gzip_allocated_bytes=out.stat().st_blocks*512)
   event(log,{'event':'gzip_verified','id':g['id'],'gzip':{k:g[k] for k in ['archive','gzip_sha256','gzip_size','gzip_allocated_bytes']}})
   assert sha(p)==g['sha256'];assert not {expected}&writer_keys()
   left=len(aliases)
   for a in aliases:
    q=ROOT/a['path'];s=q.lstat();assert (s.st_dev,s.st_ino,s.st_nlink,s.st_size,s.st_mtime_ns)==(*expected,left,g['size'],a['mtime_ns'])
    event(log,{'event':'verified_before_unlink','id':g['id'],'path':a['path'],'remaining_nlink':left,'sha256':g['sha256']})
    q.unlink();left-=1;syncdir(q.parent);event(log,{'event':'alias_archived','id':g['id'],'path':a['path'],'remaining_nlink':left})
   done.append(g);event(log,{'event':'group_complete','id':g['id'],'free_bytes':free()})
  for p,h in protected.items():assert sha(ROOT/p)==h
  ih=write(DEFAULT/'restoration_inventory.json',{'schema':'linked_ply_archive_v25','groups':groups,'protected_completed_receipts':protected,'note':'Original inode and ctime cannot be recreated; bytes, paths, hardlink relations and supported metadata are recoverable.'})
  write(DEFAULT/'restoration_integrity.json',{'inventory_sha256':ih,'restore_tool_sha256':sha(RESTORE),'archive_tool_sha256':sha(pathlib.Path(__file__))})
  result={'status':'archived_pending_restore_proof','groups':len(done),'original_alias_paths':sum(len(g['aliases']) for g in done),'all_disk_gzip_roundtrip_sha_verified':True,'all_original_paths_absent':all(not (ROOT/a['path']).exists() for g in groups for a in g['aliases']),'original_allocated_bytes':sum(g['original_allocated_bytes'] for g in groups),'gzip_allocated_bytes':sum(g['gzip_allocated_bytes'] for g in groups),'gross_reclaimed_bytes':sum(g['original_allocated_bytes']-g['gzip_allocated_bytes'] for g in groups),'free_before':start_free,'free_after':free(),'protected_completed_receipts_unchanged':True,'reserve_bytes':RESERVE};write(DEFAULT/'capacity_result.json',result);print(json.dumps(result,indent=2))
 except BaseException as e:
  event(log,{'event':'failure','error':repr(e),'groups_complete':len(done),'free_bytes':free()})
  try:write(DEFAULT/'failure.json',{'status':'failed','error':repr(e),'groups_complete':len(done),'recovery':'Use restore tool --allow-incomplete; only durably gzip_verified groups enter recovery.'})
  except OSError:pass
  raise
if __name__=='__main__':main()
