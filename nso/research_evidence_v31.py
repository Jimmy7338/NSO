"""Small offline-audit persistence helpers; no robot or mapping constructors."""
import ast
import hashlib
import io
import json
from pathlib import Path
import shutil
import zipfile
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
CAP=2*1024**2
RESERVE=64*1024**2


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):return json.loads(Path(path).read_text())


def json_value(value):
    if isinstance(value,np.ndarray):return value.tolist()
    if isinstance(value,np.generic):return value.item()
    if isinstance(value,dict):return {str(k):json_value(v) for k,v in value.items()}
    if isinstance(value,(tuple,list)):return [json_value(x) for x in value]
    return value


def write_bytes(root,path,payload):
    root,path=Path(root),Path(path)
    if path.exists():raise FileExistsError(path)
    current=sum(p.stat().st_size for p in root.rglob('*') if p.is_file())
    if current+len(payload)>CAP or shutil.disk_usage(root).free-len(payload)<RESERVE:
        raise RuntimeError('offline audit cap/reserve would be violated')
    with path.open('xb') as stream:stream.write(payload)


def write(root,path,value):
    payload=(json.dumps(json_value(value),ensure_ascii=False,indent=2,allow_nan=False)+'\n').encode()
    write_bytes(root,path,payload)


def closure(paths):
    pending=[Path(p).resolve() for p in paths];seen=set()
    while pending:
        path=pending.pop()
        if path in seen:continue
        seen.add(path)
        if path.suffix!='.py':continue
        for node in ast.walk(ast.parse(path.read_text())):
            modules=[a.name for a in node.names] if isinstance(node,ast.Import) else (
                [node.module] if isinstance(node,ast.ImportFrom) and node.module and not node.level else [])
            for module in modules:
                parts=module.split('.')
                candidates=[ROOT/Path(*parts).with_suffix('.py'),ROOT/Path(*parts)/'__init__.py']
                candidates.extend(ROOT/Path(*parts[:n])/'__init__.py' for n in range(1,len(parts)))
                pending.extend(p for p in candidates if p.is_file() and p.resolve() not in seen)
    return sorted(seen)


def freeze(root,paths,**metadata):
    paths=closure(paths);buffer=io.BytesIO()
    with zipfile.ZipFile(buffer,'w',zipfile.ZIP_DEFLATED,compresslevel=9) as archive:
        for path in paths:archive.write(path,str(path.relative_to(ROOT)))
    write_bytes(root,root/'sources.zip',buffer.getvalue())
    manifest=dict(source_sha256={str(p.relative_to(ROOT)):sha(p) for p in paths},
        source_archive_sha256=sha(root/'sources.zip'),**metadata)
    write(root,root/'manifest.json',manifest)
    return manifest


def verify_sources(root):
    m=read(root/'manifest.json')
    for rel,expected in m['source_sha256'].items():
        if sha(ROOT/rel)!=expected:raise ValueError('frozen source changed: '+rel)
    if sha(root/'sources.zip')!=m['source_archive_sha256']:raise ValueError('source archive changed')
    with zipfile.ZipFile(root/'sources.zip') as archive:
        if set(archive.namelist())!=set(m['source_sha256']):raise ValueError('source archive set differs')
        for rel,expected in m['source_sha256'].items():
            if hashlib.sha256(archive.read(rel)).hexdigest()!=expected:raise ValueError('source archive bytes differ')
    return m


def seal(root):
    write(root,root/'artifact_hashes.json',{str(p.relative_to(root)):sha(p)
        for p in sorted(root.rglob('*')) if p.is_file() and p.name!='artifact_hashes.json'})


def verify_inventory(root,exact=True):
    inventory=read(root/'artifact_hashes.json')
    for rel,expected in inventory.items():
        if sha(root/rel)!=expected:raise ValueError('artifact changed: '+rel)
    actual={str(p.relative_to(root)) for p in root.rglob('*') if p.is_file() and p.name!='artifact_hashes.json'}
    if exact and actual!=set(inventory):raise ValueError('artifact file set differs')
    return inventory
