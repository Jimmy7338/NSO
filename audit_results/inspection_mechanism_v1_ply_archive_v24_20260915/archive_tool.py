#!/usr/bin/env python3
"""Exact-byte archival of one fixed, completed V1 mesh-checkpoint inventory."""
import os
os.environ['PYTHONDONTWRITEBYTECODE']='1'
import sys
sys.dont_write_bytecode=True
import argparse
import fcntl
import gzip
import hashlib
import json
from pathlib import Path
import shutil
import stat
import time
import uuid

ROOT=Path(__file__).resolve().parents[1]
RUN=ROOT/'eval_results/inspection_mechanism_v1_20260910'
REVIEW=ROOT/'audit_results/environment_cleanup_v24_20260915'
DEFAULT_ARCHIVE=ROOT/'audit_results/inspection_mechanism_v1_ply_archive_v24_20260915'
SELF=Path(__file__).resolve()
RESTORER=ROOT/'scripts/restore_inspection_meshes_v24.py'
RESERVE=8*1024**2
INPUTS=('run_metadata.json','verification.json','replay_verification.json','artifacts_sha256.json')


def sha(path):
    with Path(path).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()


def read(path):return json.loads(Path(path).read_text())


def fsync_dir(path):
    fd=os.open(path,os.O_RDONLY|os.O_DIRECTORY)
    try:os.fsync(fd)
    finally:os.close(fd)


def atomic_json(path,value):
    temp=path.with_name(path.name+'.tmp-'+uuid.uuid4().hex)
    with temp.open('x') as f:
        json.dump(value,f,ensure_ascii=False,indent=2);f.write('\n');f.flush();os.fsync(f.fileno())
    os.replace(temp,path);fsync_dir(path.parent)


def plain_path(path):
    if not path.is_relative_to(ROOT):raise ValueError('outside project')
    current=ROOT
    for part in path.relative_to(ROOT).parts:
        current/=part
        if current.is_symlink():raise ValueError('symlink component: '+str(current))
    return path


def writers():
    result=set()
    for p in Path('/proc').iterdir():
        if not p.name.isdigit():continue
        try:fds=list((p/'fd').iterdir())
        except FileNotFoundError:continue
        except PermissionError:raise ValueError('cannot inspect active writers')
        for fd in fds:
            try:
                flags=next(int(x.split()[1],8) for x in (p/'fdinfo'/fd.name).read_text().splitlines() if x.startswith('flags:'))
                if flags&os.O_ACCMODE:
                    s=fd.stat()
                    if stat.S_ISREG(s.st_mode):result.add((s.st_dev,s.st_ino))
            except (FileNotFoundError,ProcessLookupError,StopIteration):continue
            except PermissionError:raise ValueError('cannot inspect descriptor')
    return result


def snap(s):
    return dict(dev=s.st_dev,inode=s.st_ino,size=s.st_size,mtime_ns=s.st_mtime_ns,
        mode=stat.S_IMODE(s.st_mode),uid=s.st_uid,gid=s.st_gid,nlink=s.st_nlink,allocated_bytes=s.st_blocks*512)


def signature(s):
    return (s.st_dev,s.st_ino,s.st_size,s.st_mtime_ns,s.st_mode,s.st_uid,s.st_gid,s.st_nlink)


def receipt(folder,value):
    with (folder/'receipts.jsonl').open('a') as f:
        f.write(json.dumps(value,ensure_ascii=False,separators=(',',':'))+'\n');f.flush();os.fsync(f.fileno())


def seal(folder):
    atomic_json(folder/'artifact_hashes.json',{str(p.relative_to(folder)):sha(p)
        for p in sorted(folder.rglob('*')) if p.is_file() and p!=folder/'artifact_hashes.json'})


def target_paths(folder,row):
    path=ROOT/row['path'];plain_path(path)
    rel=path.relative_to(RUN)
    if len(rel.parts)!=4 or rel.name not in ('001.ply','004.ply','012.ply'):
        raise ValueError('not in fixed checkpoint scope')
    archive=folder/'archives'/(row['path']+'.gz');plain_path(archive)
    return path,archive


def load_frozen(folder):
    plain_path(folder)
    m=read(folder/'manifest.json')
    if sha(folder/'target_inventory.json.gz')!=m['target_inventory_sha256']:raise ValueError('target list changed')
    if sha(SELF)!=m['tool_sha256'] or sha(RESTORER)!=m['restore_tool_sha256']:raise ValueError('tool changed')
    for name,wanted in m['input_sha256'].items():
        if sha(RUN/name)!=wanted:raise ValueError('original evidence metadata changed: '+name)
    if read(RUN/'run_metadata.json')['status']!='complete' or read(RUN/'verification.json')['status']!='passed':raise ValueError('run is not completed and verified')
    with gzip.open(folder/'target_inventory.json.gz','rt') as f:rows=json.load(f)
    if len(rows)!=464 or len({r['path'] for r in rows})!=464:raise ValueError('fixed target count differs')
    for row in rows:target_paths(folder,row)
    return m,rows


def check_archive(archive,row):
    s=archive.lstat()
    if not stat.S_ISREG(s.st_mode) or s.st_nlink!=1:raise ValueError('archive is not a regular single-link file')
    compressed=archive.read_bytes()
    if hashlib.sha256(compressed).hexdigest()!=row['gzip_sha256']:raise ValueError('archive hash mismatch')
    data=gzip.decompress(compressed)
    if len(data)!=row['file_bytes'] or hashlib.sha256(data).hexdigest()!=row['sha256']:raise ValueError('archive does not restore exact original bytes')
    return data


def selected(rows,only):
    if only is None:return rows
    found=[r for r in rows if r['path']==only]
    if len(found)!=1:raise ValueError('--only must exactly match an inventoried project-relative path')
    return found


def prepare(folder):
    plain_path(folder)
    if folder.exists():raise FileExistsError(folder)
    review=read(REVIEW/'mesh_compression_review.json')
    with gzip.open(REVIEW/'mesh_compression_candidates.json.gz','rt') as f:rows=json.load(f)
    if review['status']!='read_only_compression_candidates_not_applied' or len(rows)!=464:raise ValueError('unexpected approved review')
    for name,wanted in review['input_sha256'].items():
        if sha(RUN/name)!=wanted:raise ValueError('review input changed')
    inventory=read(RUN/'artifacts_sha256.json');active=writers()
    for row in rows:
        path,_=target_paths(folder,row);s=path.lstat()
        if not stat.S_ISREG(s.st_mode) or s.st_nlink!=1 or (s.st_dev,s.st_ino) in active:raise ValueError('unsafe original')
        if s.st_ino!=row['inode'] or s.st_mtime_ns!=row['mtime_ns'] or s.st_size!=row['file_bytes']:raise ValueError('review identity changed')
        if inventory[str(path.relative_to(RUN))]!=row['sha256'] or sha(path)!=row['sha256']:raise ValueError('original differs from old inventory')
        row['original_metadata']=snap(s)
    folder.mkdir(parents=True,exist_ok=False)
    packed=gzip.compress(json.dumps(rows,separators=(',',':')).encode(),compresslevel=6,mtime=0)
    with (folder/'target_inventory.json.gz').open('xb') as f:f.write(packed);f.flush();os.fsync(f.fileno())
    shutil.copyfile(SELF,folder/'archive_tool.py');shutil.copyfile(RESTORER,folder/'restore_tool.py')
    m=dict(status='prepared',run=str(RUN),target_count=464,input_sha256=review['input_sha256'],
        target_inventory_sha256=sha(folder/'target_inventory.json.gz'),tool_sha256=sha(SELF),restore_tool_sha256=sha(RESTORER),
        source_candidate_inventory_sha256=sha(REVIEW/'mesh_compression_candidates.json.gz'),python=sys.version,
        scope=review['scope'],original_evidence_content_preserved=True,original_metadata_files_unchanged=True,free_before_apply=None)
    atomic_json(folder/'manifest.json',m);seal(folder)
    print('prepared 464 exact-byte derived mesh targets; none removed',flush=True)


def verify(folder,publish=True):
    m,rows=load_frozen(folder);archived=restored=0;archive_allocated=missing_original_allocated=0
    for row in rows:
        path,archive=target_paths(folder,row)
        if archive.exists():
            check_archive(archive,row);archive_allocated+=archive.stat().st_blocks*512
        elif not path.exists():raise ValueError('both original and archive missing: '+row['path'])
        if path.exists():
            if not stat.S_ISREG(path.lstat().st_mode) or sha(path)!=row['sha256']:raise ValueError('restored/original file changed')
            restored+=1
        else:
            archived+=1;missing_original_allocated+=row['original_metadata']['allocated_bytes']
    report=dict(status='verified',targets=464,archived_missing_originals=archived,original_paths_present=restored,
        archive_allocated_bytes=archive_allocated,removed_original_allocated_bytes=missing_original_allocated,
        net_payload_allocated_bytes_released=missing_original_allocated-archive_allocated,
        all_original_bytes_recoverable=True,input_metadata_unchanged=True,free_bytes=shutil.disk_usage(ROOT).free,
        free_before_apply=m.get('free_before_apply'),legacy_full_checks_require_restore=archived>0)
    if publish:atomic_json(folder/'verification.json',report);seal(folder)
    return report


def apply(folder,only=None):
    m,rows=load_frozen(folder)
    if m.get('free_before_apply') is None:m['free_before_apply']=shutil.disk_usage(ROOT).free
    m['status']='archiving';atomic_json(folder/'manifest.json',m)
    try:
        for count,row in enumerate(selected(rows,only),start=1):
            path,archive=target_paths(folder,row)
            if not path.exists():
                check_archive(archive,row);receipt(folder,dict(event='already_archived_verified',path=row['path']));continue
            fd=os.open(path,os.O_RDONLY|os.O_NOFOLLOW)
            try:
                fcntl.flock(fd,fcntl.LOCK_EX|fcntl.LOCK_NB);before=os.fstat(fd)
                if not stat.S_ISREG(before.st_mode) or before.st_nlink!=1:raise ValueError('unsafe original inode')
                with os.fdopen(os.dup(fd),'rb') as f:data=f.read()
                if len(data)!=row['file_bytes'] or hashlib.sha256(data).hexdigest()!=row['sha256']:raise ValueError('source hash mismatch')
                if archive.exists():check_archive(archive,row)
                else:
                    if shutil.disk_usage(ROOT).free<row['gzip_level6_bytes']+RESERVE:raise OSError('insufficient per-file archive reserve')
                    compressed=gzip.compress(data,compresslevel=6,mtime=0)
                    if hashlib.sha256(compressed).hexdigest()!=row['gzip_sha256']:raise ValueError('compressed bytes differ from reviewed stream')
                    archive.parent.mkdir(parents=True,exist_ok=True)
                    temp=archive.with_name(archive.name+'.tmp-'+uuid.uuid4().hex)
                    try:
                        with temp.open('xb') as f:f.write(compressed);f.flush();os.fsync(f.fileno())
                        check_archive(temp,row)
                        os.link(temp,archive,follow_symlinks=False);temp.unlink();fsync_dir(archive.parent)
                    finally:
                        if temp.exists():temp.unlink()
                check_archive(archive,row)
                if signature(os.fstat(fd))!=signature(before) or signature(path.lstat())!=signature(before):raise ValueError('original changed during archival')
                if (before.st_dev,before.st_ino) in writers():raise ValueError('original has an active writer')
                receipt(folder,dict(event='archive_verified_before_unlink',path=row['path'],original=snap(before),
                    original_sha256=row['sha256'],archive_sha256=row['gzip_sha256']))
                path.unlink();fsync_dir(path.parent)
                receipt(folder,dict(event='original_archived',path=row['path'],released_original_allocated_bytes=before.st_blocks*512))
            finally:os.close(fd)
            if count%50==0:print('archived',count,'of',len(selected(rows,only)),flush=True)
        report=verify(folder,publish=False);m['status']='archived' if report['archived_missing_originals']==464 else 'partially_restored'
        atomic_json(folder/'manifest.json',m);atomic_json(folder/'verification.json',report);seal(folder)
        print(json.dumps(report),flush=True)
    except Exception as error:
        m.update(status='failed',error=repr(error));atomic_json(folder/'manifest.json',m)
        receipt(folder,dict(event='failure',error=repr(error)));seal(folder);raise


def restore(folder,only=None):
    m,rows=load_frozen(folder);targets=selected(rows,only)
    required=sum(r['original_metadata']['allocated_bytes'] for r in targets if not (ROOT/r['path']).exists())+RESERVE
    if shutil.disk_usage(ROOT).free<required:raise OSError('Restore retaining archives needs '+str(required)+' free bytes; use --only for a bounded required file or free more storage first')
    try:
        for row in targets:
            path,archive=target_paths(folder,row);data=check_archive(archive,row)
            if path.exists():
                if not stat.S_ISREG(path.lstat().st_mode) or sha(path)!=row['sha256']:raise ValueError('refusing existing mismatched original')
                continue
            temp=path.with_name(path.name+'.restore-'+uuid.uuid4().hex)
            try:
                with temp.open('xb') as f:f.write(data);f.flush();os.fsync(f.fileno())
                if sha(temp)!=row['sha256']:raise ValueError('restore hash mismatch')
                meta=row['original_metadata'];os.chmod(temp,meta['mode']);os.chown(temp,meta['uid'],meta['gid'])
                os.utime(temp,ns=(meta['mtime_ns'],meta['mtime_ns']))
                with temp.open('rb') as f:os.fsync(f.fileno())
                os.link(temp,path,follow_symlinks=False);temp.unlink();fsync_dir(path.parent)
                if sha(path)!=row['sha256']:raise ValueError('published restoration mismatch')
                receipt(folder,dict(event='restored_exact_original_path',path=row['path'],sha256=row['sha256'],metadata=snap(path.stat())))
            finally:
                if temp.exists():temp.unlink()
        report=verify(folder,publish=False);m['status']='restored' if report['original_paths_present']==464 else 'partially_restored'
        atomic_json(folder/'manifest.json',m);atomic_json(folder/'verification.json',report);seal(folder)
        print(json.dumps(report),flush=True)
    except Exception as error:
        receipt(folder,dict(event='restore_failure',error=repr(error)));seal(folder);raise


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--archive',type=Path,default=DEFAULT_ARCHIVE)
    op=p.add_mutually_exclusive_group(required=True);op.add_argument('--prepare',action='store_true');op.add_argument('--apply',action='store_true');op.add_argument('--verify',action='store_true');p.add_argument('--only')
    a=p.parse_args();folder=a.archive.absolute()
    if a.prepare:
        if a.only:p.error('--prepare cannot use --only')
        prepare(folder)
    elif a.apply:apply(folder,a.only)
    else:
        if a.only:p.error('--verify always verifies all targets')
        print(json.dumps(verify(folder)),flush=True)


if __name__=='__main__':main()
