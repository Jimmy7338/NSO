"""Read-only metadata/large-file duplicate screening; never modifies evidence."""
from pathlib import Path
from collections import defaultdict
import os,json,hashlib,subprocess
out=Path('/root/NSO/audit_results/v44_storage_audit_20260921')
sizes=defaultdict(dict)
for scope in ['eval_results','audit_results']:
 for root,dirs,files in os.walk(scope):
  if str(out)==str(Path(root).resolve()):dirs[:]=[];continue
  for name in files:
   p=Path(root)/name
   if p.is_symlink():continue
   s=p.stat()
   if s.st_size:sizes[s.st_size].setdefault((s.st_dev,s.st_ino),[]).append(str(p))
candidates={s:v for s,v in sizes.items() if len(v)>1}
thresholds={str(t):dict(groups=sum(s>=t for s in candidates),logical_reclaim_upper_bound=sum((len(v)-1)*s for s,v in candidates.items() if s>=t)) for t in [1,65536,1048576]}
verified=[]
for size,records in candidates.items():
 if size<1048576:continue
 hashes=defaultdict(list)
 for inode,paths in records.items():
  p=Path(paths[0]);digest=hashlib.sha256(p.read_bytes()).hexdigest();st=p.stat()
  hashes[digest].append(dict(paths=paths,size=size,inode=inode[1],device=inode[0],allocated_bytes=st.st_blocks*512,hard_links=st.st_nlink))
 for sha,ls in hashes.items():
  if len(ls)>1:verified.append(dict(sha256=sha,copies=ls,logical_reclaim_upper_bound=(len(ls)-1)*size,note='hash equal; no hardlink mutation performed; retain all aliases and check complete inode links before reclaim'))
r=dict(scope='cross eval/audit metadata and >=1MiB exact hash duplicate read-only audit',thresholds=thresholds,verified_large_duplicate_groups=verified,verified_large_duplicate_logical_reclaim_upper_bound=sum(x['logical_reclaim_upper_bound'] for x in verified),warning='threshold upper bounds are not measured free space; unverified groups may differ; inode link aliases outside scope may prevent immediate reclaim',git_count_objects=subprocess.run(['git','-c','gc.auto=0','count-objects','-v'],capture_output=True,text=True).stdout,git_maintenance_performed=False)
(out/'duplicate_audit.json').write_text(json.dumps(r,indent=2)+'\n')
print(json.dumps({k:v for k,v in r.items() if k!='verified_large_duplicate_groups'},indent=2));print('verified groups',len(verified))
