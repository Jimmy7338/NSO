"""Publish compact B11 evidence; raw RGB-D and native meshes stay local."""
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
(OUT/'README.md').write_text('# Paired unseen-layout observation-gain study (B11)\n\nThis evidence package preserves four predeclared constructed room layouts (chain/cycle topologies, two each), eight controlled acquired states and twenty-four actually executed short candidate branches. The staged first12 and final24 tables precede the authorized eight complete visibility/acquisition policy runs. Source CAD assets are reused; these are new placements/topologies, not four natural sensor environments. Physical robot experiments remain paused.\n\nBoth estimators share the frozen B9T16 full geometry/structure posterior, every positive particle, original0.22m surface quadrature, visible ideal instance/category input, common score and execution contract. Only observation-gain prediction changes: repaired historical visibility versus prediction-only RGBD/TSDF reconstruction acquisition. No conditional-mean collapse, positive-mass truncation or coefficient tuning is used. Neutral/class prior rankings share the same three executed branches at each short state.\n\nThe short cost-matched pool is an explicit controlled experiment; complete runs retain all native candidates. Two initially infeasible prefix states are preserved, alongside the cost-only pre-outcome revision selecting their earliest valid scripted successors. See USER_PROTOCOL.txt, STUDY_SCOPE.json, PREFIX_PREPARATION_V2.json and all preparation records. The central0.2m optical opening and inherited zero-radius occupancy-boundary limitation are disclosed; actual robot-room connectivity and reference robot-center reachable space were independently checked.\n\nRaw traces, all ties/negative results/failures, state/branch costs and independent source-surface measurements are retained. The best-candidate comparator means the best of the three actually executed controlled routes, not an unexecuted global optimum. Object gross gains, lost/net areas, scene quality and unknown coverage remain separate outcomes.\n\nExact cached computations and process copy-on-write accelerate hypothetical TSDF fusion while preserving original volume weights and actual measured state. Computational fork failure retries the identical fresh-replay formula without skipping candidates or changing policy stopping rules. Tests and bitwise compatibility records accompany the source.\n\nSOURCE_INDEX.json maps direct and zipped files to their hashes. Extracting ZIP members restores the study tree. Large sensor arrays and native/source geometry remain local with OMITTED_ARRAYS.json hashes; separately obtained licensed assets and earlier frozen B3/B8/B9/B10 dependencies are required for reproduction. Formal updated PDFs and editable manuscript assets are under docs/thesis/submission_focus_20261108.\n')
tools=OUT/'tools';tools.mkdir(exist_ok=True);shutil.copyfile(Path(__file__),tools/'build_bundle.py')
files=[dict(local_path=str(p),repository_relative_path=str(p.relative_to(OUT)),bytes=p.stat().st_size,sha256=sha(p)) for p in sorted(OUT.rglob('*')) if p.is_file()]
write(HERE/'content_manifest.json',dict(schema='nso.github_payload_manifest.v1',payload_root=str(OUT),suggested_repository_prefix='experiments/offline_20261120_paired_unseen',files=files,file_count=len(files),bytes=sum(x['bytes'] for x in files),remote_upload_performed=False))
write(HERE/'payload_validation.json',dict(status='all_direct_files_and_archive_members_hash_verified',files=len(files),archived_members=sum(len(a['members']) for a in archives),archives=len(archives),source_files=len(sources),raw_arrays_omitted=len(raw)))
print(json.dumps({'files':len(files),'bytes':sum(x['bytes']for x in files),'archives':len(archives),'source_files':len(sources)}))
