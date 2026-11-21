"""Stage the paper evidence with small direct files and per-attempt archives."""

import base64
import hashlib
import json
import shutil
import sys
import zipfile
from pathlib import Path


STUDY = Path(__file__).resolve().parents[1]
PUBLICATION = STUDY / 'publication'
REPO_PREFIX = 'experiments/offline_20261121_semantic_large_scene'
TEXT = {'.json', '.jsonl', '.csv', '.md', '.py', '.sh', '.log', '.txt'}
VECTORS = {'.pdf', '.svg'}


def file_digest(data):
    return hashlib.sha256(data).hexdigest()


def prepare():
    result = STUDY / 'evaluation/results'
    decision = json.loads((result / 'decision.json').read_text())
    rows = json.loads((result / 'endpoint_rows.json').read_text())
    assert len(rows) == 12 and all(r['status'] not in ('missing_or_in_progress', 'awaiting_final_intervention_receipt') for r in rows)
    assert decision['status'] == 'stop_and_diagnose' or (STUDY/'confirmatory/terminal_receipt.json').is_file()
    payload = PUBLICATION / 'payload'
    payload.mkdir(parents=True, exist_ok=False)
    archives = {}
    for file in sorted(STUDY.rglob('*')):
        path = file.relative_to(STUDY)
        if not file.is_file() or path.parts[0] == 'publication' or '__pycache__' in path.parts:
            continue
        if path.parts[:2] in [('evaluation', 'intermediate'), ('evaluation','replication')]:
            continue
        if file.suffix not in TEXT | VECTORS:
            continue
        if any(part.startswith('attempt_') for part in path.parts) and file.stat().st_size > 40000:
            attempt_folder = file.parents[next(i for i,p in enumerate(file.parents) if p.name.startswith('attempt_'))]
            archive_key = '__'.join(attempt_folder.relative_to(STUDY).parts)
            archives.setdefault(archive_key, []).append(file)
            continue
        if file.stat().st_size > 170000:
            archives.setdefault(path.parts[0] + '_large_reports', []).append(file)
            continue
        target = payload / path
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(file, target)

    index = {}
    if archives:
        (payload / 'archives').mkdir()
    for archive_key, files in archives.items():
        zipped = payload / 'archives' / (archive_key + '.zip')
        with zipfile.ZipFile(zipped, 'w') as z:
            for file in files:
                relative = str(file.relative_to(STUDY))
                data = file.read_bytes()
                entry = zipfile.ZipInfo(relative, date_time=(2026, 11, 21, 0, 0, 0))
                z.writestr(entry, data, compress_type=zipfile.ZIP_DEFLATED)
                index[relative] = dict(archive=str(zipped.relative_to(payload)), sha256=file_digest(data))
        with zipfile.ZipFile(zipped) as z:
            assert all(file_digest(z.read(p)) == index[p]['sha256'] for p in z.namelist())
    (payload/'archive_index.json').write_text(json.dumps(index, indent=2)+'\n')

    upload = []
    for p in sorted(payload.rglob('*')):
        if not p.is_file():
            continue
        data = p.read_bytes()
        upload.append(dict(path=str(p), repo_path=f'{REPO_PREFIX}/{p.relative_to(payload)}', bytes=len(data),
                           sha256=file_digest(data), binary=p.suffix in {'.zip', '.pdf'},
                           blob_sha=hashlib.sha1(f'blob {len(data)}\0'.encode()+data).hexdigest()))
    (PUBLICATION/'upload.json').write_text(json.dumps(upload, indent=2)+'\n')
    print(json.dumps(dict(files=len(upload), bytes=sum(row['bytes'] for row in upload))))


def chunk(index, offset=0, maximum=None):
    files = json.loads((PUBLICATION/'upload.json').read_text())
    row = files[index]
    data = Path(row['path']).read_bytes()
    assert file_digest(data) == row['sha256']
    data = data[offset:] if maximum is None else data[offset:offset+maximum]
    print(json.dumps(base64.b64encode(data).decode() if row['binary'] else data.decode(), ensure_ascii=False))


if __name__ == '__main__':
    if sys.argv[1] == 'prepare':
        prepare()
    else:
        chunk(*map(int, sys.argv[2:]))
