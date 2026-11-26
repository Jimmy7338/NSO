"""Publish compact B9 evidence; raw RGB-D and native meshes stay local."""
from pathlib import Path
import collections, hashlib, json, shutil, zipfile

HERE=Path(__file__).resolve().parent;ROOT=HERE.parent;OUT=HERE/'payload'
assert not OUT.exists(),'Use a new explicit publication attempt, never overwrite a sealed bundle'
OUT.mkdir(parents=True)
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(p,d):p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(d,indent=2)+'\n')

allowed={'.py','.json','.jsonl','.csv','.md','.txt','.bib','.log','.pdf','.svg'}
sources=[];raw=[]
for p in sorted(ROOT.rglob('*')):
    if not p.is_file():continue
    rel=p.relative_to(ROOT);parts=rel.parts
    if parts[0] in ('publication','paper_update') or '__pycache__' in parts:continue
    include=p.suffix in allowed
    if p.suffix=='.png':include=parts[0]=='figure'
    if p.suffix=='.npz':include=parts[:2]==('structure','replay_attempt_001') and p.name.endswith('_predictions.npz')
    if include:sources.append((p,rel))
    elif p.suffix in ('.npz','.ply','.glb','.obj'):
        raw.append(dict(path=str(rel),bytes=p.stat().st_size,sha256=sha(p),reason='Raw sensor/mesh or intermediate array omitted; generator/source and complete scalar evidence retained.'))

archive_sources=[];source_index=[]
for p,rel in sources:
    direct=(p.suffix in ('.py','.md','.txt','.bib','.pdf','.svg','.png') or
            (p.stat().st_size<70000 and len(rel.parts)<=3 and p.suffix in ('.json','.csv')))
    source_index.append(dict(path=str(rel),sha256=sha(p),bytes=p.stat().st_size,storage='direct' if direct else 'archive'))
    if direct:
        dest=OUT/rel;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(p,dest)
    else:archive_sources.append((p,rel))

archives=[];groups=collections.defaultdict(list)
for p,rel in archive_sources:groups[rel.parts[0]].append((p,rel))
for group,items in groups.items():
    batches=[];batch=[];size=0
    for item in items:
        n=item[0].stat().st_size
        if batch and size+n>3000000:batches.append(batch);batch=[];size=0
        batch.append(item);size+=n
    if batch:batches.append(batch)
    for i,batch in enumerate(batches,1):
        path=OUT/'archives'/f'{group}_{i:03d}.zip';path.parent.mkdir(exist_ok=True)
        members=[]
        with zipfile.ZipFile(path,'w',zipfile.ZIP_DEFLATED,compresslevel=9) as z:
            for p,rel in batch:
                info=zipfile.ZipInfo(str(rel),date_time=(2026,11,20,0,0,0));info.compress_type=zipfile.ZIP_DEFLATED
                info.external_attr=0o100644<<16;z.writestr(info,p.read_bytes())
                members.append(dict(path=str(rel),bytes=p.stat().st_size,sha256=sha(p)))
        with zipfile.ZipFile(path) as z:
            for m in members:
                data=z.read(m['path']);assert len(data)==m['bytes'] and hashlib.sha256(data).hexdigest()==m['sha256']
        archives.append(dict(path=str(path.relative_to(OUT)),sha256=sha(path),bytes=path.stat().st_size,members=members))

write(OUT/'SOURCE_INDEX.json',dict(sources=source_index,archives=archives))
write(OUT/'OMITTED_ARRAYS.json',dict(role='Local reproducibility inputs; not silently counted as uploaded',files=raw))
(OUT/'README.md').write_text('''# Independent uncertainty and semantic follow-up (B9)

This bundle accompanies the current anonymous NSO manuscript. It preserves
independent geometry calibration/test, complete null and adverse outcomes,
the exact semantic decision diagnosis, structural-temperature revision,
fixed-branch re-scoring, paired virtual rollouts and controlled mechanism
attempts. Physical experiments remain paused.

`STUDY_SCOPE.json` and each sub-study protocol define the evaluated versions.
Object-level interval calibration changes diagnostic sets only; it does not
change the planning probability distribution. Structural temperature changes
the structural mixture, retaining the shape-conditional geometric distribution
and original category prior. Neither change establishes semantic navigation
superiority by itself. Correct-category tests use supplied ideal cues.

The archived JSON/CSV/JSONL retain every recorded condition, candidate and
execution, including ties, failed attempts and negative effects. Source and
member SHA256 values are in `SOURCE_INDEX.json`. ZIP paths reconstruct the B9
study tree. Raw sensor arrays and native/source meshes are omitted and listed
in `OMITTED_ARRAYS.json`; generators and frozen dependencies reproduce them.
Absolute source paths document the original local environment. Prior B8/B6/B5
dependencies and the original acquired66 branches are referenced by hashes
and remain in the earlier evidence bundles; references do not make them new
independent observations. Formal manuscript PDFs and editable sources are
published separately under `docs/thesis/submission_focus_20261108`.
''')
tools=OUT/'tools';tools.mkdir(exist_ok=True);shutil.copyfile(Path(__file__),tools/'build_bundle.py')
files=[dict(local_path=str(p),repository_relative_path=str(p.relative_to(OUT)),bytes=p.stat().st_size,sha256=sha(p)) for p in sorted(OUT.rglob('*')) if p.is_file()]
write(HERE/'content_manifest.json',dict(schema='nso.github_payload_manifest.v1',payload_root=str(OUT),suggested_repository_prefix='experiments/offline_20261120_followup',files=files,file_count=len(files),bytes=sum(x['bytes'] for x in files),remote_upload_performed=False))
write(HERE/'payload_validation.json',dict(status='all_direct_files_and_archive_members_hash_verified',files=len(files),archived_members=sum(len(a['members']) for a in archives),archives=len(archives),source_files=len(sources),raw_arrays_omitted=len(raw)))
print(json.dumps({'files':len(files),'bytes':sum(x['bytes']for x in files),'archives':len(archives),'source_files':len(sources)}))
