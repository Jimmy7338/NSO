"""Publish compact B10 evidence; raw RGB-D and native meshes stay local."""
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
    if p.suffix=='.npz':include=parts[0]=='replay_attempt_001' and p.name.endswith('_predictions.npz')
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
(OUT/'README.md').write_text('# Measured-surface novelty alignment (B10)\n\nThis bundle preserves the declared5cm current-native-TSDF consumption correction to the B9 T16 template observation reward, all six-prefix fixed-replay predictions, complete point-level error partitions, independent12-object/36-route accounting fixtures, and four full virtual G/S executions. The primary task remains unknown exploration plus measured reconstruction. Physical robot experiments remain paused.\n\nThe correction multiplies the original predicted point weights by a category-independent mask from the acquired native mesh before route visibility evaluation. It changes decision reward values; geometry, posterior probabilities, common score terms, candidate generation and the virtual execution contract are retained. It removes only one source of repeated reward. Fitted false surfaces, unreconstructed visible surfaces, and object-versus-whole-scene objective differences remain distinct failure modes.\n\nKnown-geometry fixtures isolate accounting and do not validate fitted geometry, natural detection or NSO navigation. The six inspected development prefixes reuse66 actual candidate endpoints; these are not new independent trials. Full comparisons reuse B9 as the matched baseline and preserve all ties and adverse outcomes.\n\nSOURCE_INDEX.json lists direct and archived evidence with SHA256 hashes. ZIP members reconstruct this study tree. Large raw arrays, native/source meshes and image captures remain local and are listed in OMITTED_ARRAYS.json. Complete sources, protocols, per-condition/candidate scalar records, failures and editable figure data are retained. Prior B3/B8/B9 frozen dependencies are referenced by hash; this compact archive is not a standalone simulator. Formal manuscript PDFs/sources are published under docs/thesis/submission_focus_20261108.\n')
tools=OUT/'tools';tools.mkdir(exist_ok=True);shutil.copyfile(Path(__file__),tools/'build_bundle.py')
files=[dict(local_path=str(p),repository_relative_path=str(p.relative_to(OUT)),bytes=p.stat().st_size,sha256=sha(p)) for p in sorted(OUT.rglob('*')) if p.is_file()]
write(HERE/'content_manifest.json',dict(schema='nso.github_payload_manifest.v1',payload_root=str(OUT),suggested_repository_prefix='experiments/offline_20261120_gain_alignment',files=files,file_count=len(files),bytes=sum(x['bytes'] for x in files),remote_upload_performed=False))
write(HERE/'payload_validation.json',dict(status='all_direct_files_and_archive_members_hash_verified',files=len(files),archived_members=sum(len(a['members']) for a in archives),archives=len(archives),source_files=len(sources),raw_arrays_omitted=len(raw)))
print(json.dumps({'files':len(files),'bytes':sum(x['bytes']for x in files),'archives':len(archives),'source_files':len(sources)}))
