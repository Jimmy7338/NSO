#!/usr/bin/env python3
"""Lossless storage packaging of completed, independently reviewed episodes.

No original episode is modified or removed. Original step gzip streams are
represented as JSON inside one tar.xz so XZ can reuse a dictionary across steps.
Verification reconstructs every original byte stream and checks original SHA256.
Restoration only writes into a new directory and verifies the original artifact
manifest. There are no simulator, controller, mapper or evaluator imports.
"""
import argparse
from copy import deepcopy
import gzip
import hashlib
import io
import json
import lzma
import os
from pathlib import Path, PurePosixPath
import platform
import shutil
import tarfile
import time
import zlib


META='PACKAGING_MANIFEST.json'
CHUNK=1024**2
GIB=1024**3
MAX_ARCHIVE=100_000_000
MAX_ORIGINAL=64*CHUNK
MAX_LOGICAL=512*CHUNK
MAX_MEMBER=32*CHUNK
MAX_FILES=1024
MAX_MANIFEST=2*CHUNK
MAX_TAR=MAX_LOGICAL+MAX_MANIFEST+(MAX_FILES+1)*1024+10240


def canonical(value):
    return (json.dumps(value,sort_keys=True,separators=(',',':'),ensure_ascii=False,
                       allow_nan=False)+'\n').encode()


def decode(data):
    def unique(items):
        result={}
        for key,value in items:
            if key in result:raise ValueError('duplicate JSON key: '+key)
            result[key]=value
        return result
    return json.loads(data,object_pairs_hook=unique,parse_constant=lambda x:
        (_ for _ in ()).throw(ValueError('nonfinite JSON: '+x)))


def sha_bytes(data):return hashlib.sha256(data).hexdigest()


def sha_file(path):
    digest=hashlib.sha256()
    with Path(path).open('rb') as source:
        while chunk:=source.read(CHUNK):digest.update(chunk)
    return digest.hexdigest()


def relative(name):
    if not isinstance(name,str) or not name or '\\' in name or '\x00' in name:
        raise ValueError('plain relative POSIX path required')
    path=PurePosixPath(name)
    if path.is_absolute() or '..' in path.parts or str(path)!=name:
        raise ValueError('archive path traversal or ambiguous name rejected')
    return path


def regular(path):
    path=Path(path)
    if path.is_symlink() or any(p.is_symlink() for p in path.parents) or not path.is_file():
        raise ValueError('plain file without symlink ancestors required: '+str(path))
    return path


def tree_files(root):
    root=Path(root)
    if root.is_symlink() or any(p.is_symlink() for p in root.parents) or not root.is_dir():
        raise ValueError('plain episode directory required')
    files=[]
    for p in sorted(root.rglob('*')):
        if p.is_symlink():raise ValueError('symlink inside episode rejected')
        if p.is_file():files.append(p)
        elif not p.is_dir():raise ValueError('nonregular episode member rejected')
    if not 1<=len(files)<=MAX_FILES:raise ValueError('bounded episode file count required')
    return files


def reserve_space(path,allowance):
    p=Path(path).absolute()
    while not p.exists():p=p.parent
    free=shutil.disk_usage(p).free
    if free<GIB+allowance:raise ValueError('at least 1 GiB reserve plus operation allowance required')
    return free


def original_manifest_check(root,expected_sha=None):
    root=Path(root);files=tree_files(root)
    manifest_path=regular(root/'artifact_manifest.json')
    if expected_sha is not None and sha_file(manifest_path)!=expected_sha:
        raise ValueError('original artifact manifest pin mismatch')
    manifest=decode(manifest_path.read_bytes())
    if set(str(p.relative_to(root)) for p in files)!=set(manifest['files'])|{'artifact_manifest.json'}:
        raise ValueError('complete original artifact inventory mismatch')
    for name,row in manifest['files'].items():
        relative(name);p=regular(root/name)
        if p.stat().st_size!=row['bytes'] or sha_file(p)!=row['sha256']:
            raise ValueError('original artifact differs from manifest: '+name)
    return manifest


def approved_episode(episode,review_root):
    episode=Path(episode);review_root=Path(review_root)
    ledger=decode(regular(episode.parent.parent/'start_ledger.json').read_bytes())
    entries=[r for r in ledger['entries'] if r['run_id']==episode.name]
    if len(entries)!=1 or entries[0]['status']!='controller_stop' or entries[0].get('qualified') is not True:
        raise ValueError('only a completed qualified episode can be packaged; failed/pending attempts remain untouched')
    entry=entries[0]
    report=decode(regular(review_root/'review.json').read_bytes())
    review_manifest=decode(regular(review_root/'manifest.json').read_bytes())
    if (report.get('status')!='reviewed' or report.get('all_checks_passed') is not True
            or report.get('qualified') is not True or report['run_id']!=episode.name
            or report.get('input_manifest_sha256')!=entry['artifact_manifest_sha256']):
        raise ValueError('matching independent passed review required')
    for name,row in review_manifest['files'].items():
        relative(name);p=regular(review_root/name)
        if p.stat().st_size!=row['bytes'] or sha_file(p)!=row['sha256']:
            raise ValueError('independent review output changed')
    manifest=original_manifest_check(episode,entry['artifact_manifest_sha256'])
    if sha_file(episode/'result.json')!=entry['result_sha256']:
        raise ValueError('terminal result is not bound to the retained ledger')
    return dict(artifact_manifest_sha256=entry['artifact_manifest_sha256'],
        review_sha256=sha_file(review_root/'review.json'),
        review_manifest_sha256=sha_file(review_root/'manifest.json'),
        ledger_entry=deepcopy(entry),original_artifact_manifest=manifest)


def logical_digest(source,*,gzip_reencode=False,output=None,capture=False,limit=MAX_MEMBER):
    logical=hashlib.sha256();original=hashlib.sha256();logical_n=original_n=0
    compressor=zlib.compressobj(level=6,wbits=31) if gzip_reencode else None
    captured=io.BytesIO() if capture else None
    def emit(data):
        nonlocal original_n
        original_n+=len(data);original.update(data)
        if original_n>MAX_MEMBER:raise ValueError('reconstructed member exceeds byte cap')
        if output is not None:output.write(data)
    while chunk:=source.read(CHUNK):
        logical_n+=len(chunk)
        if logical_n>limit:raise ValueError('logical member exceeds byte cap')
        logical.update(chunk)
        if captured is not None:captured.write(chunk)
        emit(compressor.compress(chunk) if compressor else chunk)
    if compressor:emit(compressor.flush())
    return dict(logical_bytes=logical_n,logical_sha256=logical.hexdigest(),
                original_bytes=original_n,original_sha256=original.hexdigest()),(
                None if captured is None else captured.getvalue())


def packaging_manifest(episode,provenance):
    records=[];original_total=logical_total=allocated=0
    encoding=decode((episode/'encoding.json').read_bytes())
    gzip_steps={r['artifact']:r for r in encoding['steps']}
    for p in tree_files(episode):
        name=str(p.relative_to(episode));relative(name)
        if name==META:raise ValueError('reserved packaging metadata name already exists')
        size=p.stat().st_size
        if size>MAX_MEMBER:raise ValueError('original member exceeds byte cap')
        pin=sha_file(p);transformed=name.startswith('steps/') and name.endswith('.json.gz')
        if transformed:
            if name not in gzip_steps or gzip_steps[name]['compression_level']!=6 or gzip_steps[name]['mtime']!=0:
                raise ValueError('step is not declared canonical level-6, mtime-0 gzip')
            with gzip.open(p,'rb') as source:digests,_=logical_digest(source,gzip_reencode=True)
            if (digests['original_bytes']!=size or digests['original_sha256']!=pin
                    or digests['logical_bytes']!=gzip_steps[name]['uncompressed_bytes']):
                raise ValueError('current zlib cannot exactly reconstruct original gzip bytes: '+name)
            stored=name[:-3]
        else:
            digests=dict(original_bytes=size,original_sha256=pin,logical_bytes=size,logical_sha256=pin)
            stored=name
        original_total+=size;logical_total+=digests['logical_bytes'];allocated+=p.stat().st_blocks*512
        if original_total>MAX_ORIGINAL or logical_total>MAX_LOGICAL:
            raise ValueError('total original/logical allowance exceeded')
        records.append(dict(original_path=name,stored_path=stored,
                            transformation='gzip_decode_reencode' if transformed else 'identity',**digests))
    if len({r['stored_path'] for r in records})!=len(records):raise ValueError('packaged path collision')
    return dict(schema='article.lossless_episode_package.v1',episode_id=episode.name,
        artifact_manifest_sha256=provenance['artifact_manifest_sha256'],
        independent_review_sha256=provenance['review_sha256'],
        independent_review_manifest_sha256=provenance['review_manifest_sha256'],
        terminal_ledger_entry=provenance['ledger_entry'],files=records,
        total_original_bytes=original_total,total_logical_bytes=logical_total,
        original_allocated_bytes=allocated,
        reconstruction=dict(gzip_compresslevel=6,gzip_mtime=0,zlib_wbits=31,
            streaming_recipe='zlib.compressobj(level=6,wbits=31); Z_NO_FLUSH chunks; final Z_FINISH',
            python=platform.python_version(),zlib_compile_version=zlib.ZLIB_VERSION,
            zlib_runtime_version=zlib.ZLIB_RUNTIME_VERSION),
        packaging=dict(container='tar.xz',tar_format='ustar',xz_preset=6,xz_check='CRC64',
            cross_step_logical_dictionary=True,original_files_deleted=False),
        limits=dict(maximum_archive_bytes=MAX_ARCHIVE,maximum_original_bytes=MAX_ORIGINAL,
            maximum_logical_bytes=MAX_LOGICAL,maximum_member_bytes=MAX_MEMBER,
            maximum_files=MAX_FILES,minimum_free_bytes=GIB),
        source_sha256=sha_file(Path(__file__)))


class LimitedOutput:
    def __init__(self,stream,maximum):self.stream=stream;self.maximum=maximum;self.bytes=0
    def write(self,data):
        if self.bytes+len(data)>=self.maximum:raise ValueError('archive must remain below 100,000,000 bytes')
        count=self.stream.write(data);self.bytes+=count;return count
    def flush(self):self.stream.flush()


class BoundedXZInput:
    """One XZ stream, bounded decoder memory and total decompressed tar bytes."""
    def __init__(self,path):
        self.source=Path(path).open('rb')
        self.decoder=lzma.LZMADecompressor(format=lzma.FORMAT_XZ,memlimit=128*CHUNK)
        self.total=0;self.closed=False;self.end_checked=False
    def __enter__(self):return self
    def __exit__(self,*args):self.source.close();self.closed=True
    def read(self,size):
        if not isinstance(size,int) or size<0:raise ValueError('bounded read size required')
        pieces=[];remaining=size
        while remaining and not self.decoder.eof:
            compressed=self.source.read(64*1024) if self.decoder.needs_input else b''
            if self.decoder.needs_input and not compressed:
                raise ValueError('truncated XZ stream')
            piece=self.decoder.decompress(compressed,max_length=min(remaining,CHUNK))
            self.total+=len(piece)
            if self.total>MAX_TAR:raise ValueError('decompressed tar exceeds total byte allowance')
            pieces.append(piece);remaining-=len(piece)
        if self.decoder.eof and not self.end_checked:
            if self.decoder.unused_data or self.source.read(1):
                raise ValueError('concatenated streams or trailing compressed data rejected')
            self.end_checked=True
        return b''.join(pieces)


def tar_add(tar,name,size,source):
    member=tarfile.TarInfo(name);member.size=size;member.mode=0o644
    member.uid=member.gid=member.mtime=0;member.uname=member.gname=''
    tar.addfile(member,source)


def pack(episode,review_root,archive):
    began=time.monotonic();episode=Path(episode).absolute();archive=Path(archive).absolute()
    if archive.resolve().is_relative_to(episode.resolve()):raise ValueError('archive must be outside original episode')
    if any(p.is_symlink() for p in archive.parents):raise ValueError('archive symlink ancestors rejected')
    partial=archive.with_name(archive.name+'.partial')
    if archive.exists() or partial.exists():raise FileExistsError('exclusive new archive required')
    free=reserve_space(archive.parent,MAX_ARCHIVE+MAX_MEMBER)
    provenance=approved_episode(episode,Path(review_root))
    manifest=packaging_manifest(episode,provenance);meta=canonical(manifest)
    if len(meta)>MAX_MANIFEST:raise ValueError('bounded packaging metadata required')
    archive.parent.mkdir(parents=True,exist_ok=True)
    with partial.open('xb') as raw:
        bounded=LimitedOutput(raw,MAX_ARCHIVE)
        with lzma.LZMAFile(bounded,'w',format=lzma.FORMAT_XZ,check=lzma.CHECK_CRC64,preset=6) as xz:
            with tarfile.open(fileobj=xz,mode='w|',format=tarfile.USTAR_FORMAT) as tar:
                tar_add(tar,META,len(meta),io.BytesIO(meta))
                for row in manifest['files']:
                    p=regular(episode/row['original_path'])
                    opener=gzip.open if row['transformation']=='gzip_decode_reencode' else open
                    with opener(p,'rb') as source:tar_add(tar,row['stored_path'],row['logical_bytes'],source)
        raw.flush();os.fsync(raw.fileno())
    checked=verify(partial)
    # Re-read the original after packing; a live or altered input cannot pass.
    original_manifest_check(episode,provenance['artifact_manifest_sha256'])
    os.link(partial,archive);partial.unlink()  # Removes only our own completed temporary package.
    report=dict(status='packaged_and_roundtrip_verified',archive=str(archive),archive_sha256=sha_file(archive),
        original_bytes=manifest['total_original_bytes'],original_allocated_bytes=manifest['original_allocated_bytes'],
        logical_bytes=manifest['total_logical_bytes'],archive_bytes=archive.stat().st_size,
        original_to_archive_ratio=manifest['total_original_bytes']/archive.stat().st_size,
        verified_original_files=checked['verified_original_files'],verified_gzip_files=checked['verified_gzip_files'],
        elapsed_s=time.monotonic()-began,verification_s=checked['elapsed_s'],
        free_before_bytes=free,free_after_bytes=shutil.disk_usage(archive.parent).free,
        original_episode_deleted=False,original_episode_modified=False)
    with archive.with_name(archive.name+'.report.json').open('xb') as out:out.write(canonical(report))
    return report


def validate_packaging_manifest(manifest):
    if manifest.get('schema')!='article.lossless_episode_package.v1':raise ValueError('declared package schema required')
    if manifest['reconstruction']['zlib_runtime_version']!=zlib.ZLIB_RUNTIME_VERSION:
        raise ValueError('original zlib runtime required for byte-identical gzip restoration')
    if any(manifest['reconstruction'][k]!=v for k,v in
           [('gzip_compresslevel',6),('gzip_mtime',0),('zlib_wbits',31)]):
        raise ValueError('supported byte-exact gzip recipe required')
    rows=manifest['files']
    if not 1<=len(rows)<=MAX_FILES:raise ValueError('package file count exceeds limit')
    original=set();stored=set()
    for row in rows:
        for key,seen in [('original_path',original),('stored_path',stored)]:
            relative(row[key])
            if row[key] in seen or row[key]==META:raise ValueError('duplicate/reserved package path')
            seen.add(row[key])
        for key in ('original_bytes','logical_bytes'):
            if type(row[key]) is not int or not 0<=row[key]<=MAX_MEMBER:raise ValueError('bounded member sizes required')
        for key in ('original_sha256','logical_sha256'):
            if len(row[key])!=64 or any(c not in '0123456789abcdef' for c in row[key]):raise ValueError('SHA256 required')
        kind=row['transformation']
        if kind=='gzip_decode_reencode':
            if not row['original_path'].startswith('steps/') or not row['original_path'].endswith('.json.gz') or row['stored_path']!=row['original_path'][:-3]:
                raise ValueError('gzip transformation restricted to saved step JSON')
        elif kind!='identity' or row['original_path']!=row['stored_path']:
            raise ValueError('unknown package transformation')
    if (sum(r['original_bytes'] for r in rows)!=manifest['total_original_bytes']
            or sum(r['logical_bytes'] for r in rows)!=manifest['total_logical_bytes']
            or manifest['total_original_bytes']>MAX_ORIGINAL or manifest['total_logical_bytes']>MAX_LOGICAL):
        raise ValueError('bounded total byte declarations required')
    # No member may turn an already declared file into a containing directory.
    for names in (original,stored):
        for name in names:
            if any(str(parent) in names for parent in PurePosixPath(name).parents if str(parent)!='.'):
                raise ValueError('file/directory path collision')
    return rows


def verify(archive,restore_to=None):
    began=time.monotonic();archive=regular(archive)
    if not 0<archive.stat().st_size<MAX_ARCHIVE:raise ValueError('bounded package file required')
    restore=None if restore_to is None else Path(restore_to).absolute()
    if restore is not None:
        if restore.exists():raise FileExistsError('restore requires a wholly new directory')
        if any(p.is_symlink() for p in restore.parents):raise ValueError('restore symlink ancestors rejected')
        reserve_space(restore.parent,MAX_ORIGINAL)
    artifact_data=None;seen=set();gz_count=0
    with BoundedXZInput(archive) as xz:
        with tarfile.open(fileobj=xz,mode='r|') as tar:
            first=tar.next()
            if first is None or first.name!=META or not first.isfile() or not 0<first.size<=MAX_MANIFEST:
                raise ValueError('first member must be bounded regular packaging metadata')
            manifest=decode(tar.extractfile(first).read(MAX_MANIFEST+1));rows=validate_packaging_manifest(manifest)
            by_stored={r['stored_path']:r for r in rows}
            if restore is not None:restore.mkdir(parents=True,exist_ok=False)
            for member in tar:
                # Streaming tar iteration yields the already-read first member.
                if member is first:continue
                relative(member.name)
                if not member.isfile() or member.issym() or member.islnk():raise ValueError('links/nonregular tar members rejected')
                if member.name not in by_stored or member.name in seen:raise ValueError('undeclared/duplicate tar member')
                row=by_stored[member.name]
                if member.size!=row['logical_bytes']:raise ValueError('tar member size differs from packaging metadata')
                seen.add(member.name);source=tar.extractfile(member)
                transformed=row['transformation']=='gzip_decode_reencode';gz_count+=int(transformed)
                target=None
                try:
                    if restore is not None:
                        p=restore/row['original_path'];p.parent.mkdir(parents=True,exist_ok=True)
                        if not p.resolve().is_relative_to(restore.resolve()):raise ValueError('restore path escape')
                        target=p.open('xb')
                    digests,captured=logical_digest(source,gzip_reencode=transformed,output=target,
                        capture=row['original_path']=='artifact_manifest.json')
                finally:
                    if target is not None:target.close()
                if any(digests[k]!=row[k] for k in digests):raise ValueError('roundtrip SHA/size mismatch: '+row['original_path'])
                if captured is not None:artifact_data=captured
            if seen!=set(by_stored):raise ValueError('missing packaged member')
            if artifact_data is None or sha_bytes(artifact_data)!=manifest['artifact_manifest_sha256']:
                raise ValueError('original artifact manifest not restored exactly')
            artifact=decode(artifact_data)
            by_original={r['original_path']:r for r in rows}
            if set(by_original)!=set(artifact['files'])|{'artifact_manifest.json'}:
                raise ValueError('packaging inventory differs from original artifact manifest')
            for name,row in artifact['files'].items():
                packed=by_original[name]
                if row['bytes']!=packed['original_bytes'] or row['sha256']!=packed['original_sha256']:
                    raise ValueError('original scientific manifest mismatch: '+name)
            # Consume the XZ footer too, validating its CRC and truncated streams.
            while xz.read(CHUNK):pass
    if restore is not None:original_manifest_check(restore,manifest['artifact_manifest_sha256'])
    return dict(status='restored_and_verified' if restore is not None else 'roundtrip_verified',
        archive_sha256=sha_file(archive),artifact_manifest_sha256=manifest['artifact_manifest_sha256'],
        verified_original_files=len(seen),verified_gzip_files=gz_count,
        original_bytes=manifest['total_original_bytes'],logical_bytes=manifest['total_logical_bytes'],
        archive_bytes=archive.stat().st_size,restore_to=None if restore is None else str(restore),
        strict_original_manifest_passed=True,elapsed_s=time.monotonic()-began)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);commands=parser.add_subparsers(dest='command',required=True)
    packing=commands.add_parser('pack');packing.add_argument('--episode',type=Path,required=True)
    packing.add_argument('--review',type=Path,required=True);packing.add_argument('--archive',type=Path,required=True)
    checking=commands.add_parser('verify');checking.add_argument('--archive',type=Path,required=True)
    restoring=commands.add_parser('restore');restoring.add_argument('--archive',type=Path,required=True)
    restoring.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    result=(pack(args.episode,args.review,args.archive) if args.command=='pack' else
            verify(args.archive,args.output if args.command=='restore' else None))
    print(json.dumps(result,ensure_ascii=False,allow_nan=False))
