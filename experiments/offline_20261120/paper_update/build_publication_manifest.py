#!/usr/bin/env python3
"""Seal the current small-paper build closure for the parent's publication step."""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re

HERE = Path(__file__).resolve().parent
PAPER = Path('/home/ouc/nso_robot/workspaces/nso_research_20261029/NSO/docs/thesis/submission_focus_20261108')
PREFIX = 'docs/thesis/submission_focus_20261108'
OLD = HERE.parent.parent/'dual_template_repair_20261119/paper_update/publication_manifest.json'


def main():
    prior = json.loads(OLD.read_text())
    roles = {Path(row['local_path']):row['roles'] for row in prior['files']}
    for name in ('dual_template_geometry_20261120.tex','staged_diagnostic_table_20261120.tex','sampling_policy_table_20261120.tex'):
        roles[PAPER/name] = ['latex_source_dependency']
    for name in ('staged_diagnostic_table_20261120.csv','sampling_policy_table_20261120.csv'):
        roles[PAPER/name] = ['full_precision_table_source']
    # Refuse a manifest with missing recursive TeX or graphics dependencies.
    visited = set()
    def walk(path):
        path = path.resolve()
        if path in visited: return
        visited.add(path)
        assert path in roles and path.exists(), path
        text = path.read_text()
        for name in re.findall(r'\\input\{([^}]+)\}', text):
            p = path.parent/name
            if p.suffix != '.tex': p = p.with_suffix('.tex')
            walk(p)
        for name in re.findall(r'\\includegraphics(?:\[[^]]*\])?\{([^}]+)\}', text):
            p = path.parent/name
            if not p.suffix: p = p.with_suffix('.pdf')
            assert p.resolve() in roles and p.exists(), p
    walk(PAPER/'main.tex'); walk(PAPER/'supplement.tex')
    files = []
    for path, role in sorted(roles.items()):
        data = path.read_bytes()
        files.append(dict(local_path=str(path),repo_path=PREFIX+'/'+str(path.relative_to(PAPER)),
            bytes=len(data),sha256=hashlib.sha256(data).hexdigest(),
            git_blob_sha1=hashlib.sha1(b'blob '+str(len(data)).encode()+b'\0'+data).hexdigest(),roles=role))
    out = dict(schema='nso.paper_publication_manifest.v1',created_utc=datetime.now(timezone.utc).isoformat(),
        status='prepared_local_only_not_pushed',scope='Current main/supp recursive build closure, PDFs, editable figure dependencies and full-precision new table sources; excludes logs, backups, native meshes and raw frames.',
        roots=['main.tex','supplement.tex'],local_root=str(PAPER),repo_prefix=PREFIX,files=files,
        previous_manifest_sha256=hashlib.sha256(OLD.read_bytes()).hexdigest(),
        generator_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    (HERE/'publication_manifest.json').write_text(json.dumps(out,indent=2)+'\n')
    print(json.dumps({'files':len(files),'recursive_tex_files':len(visited),'bytes':sum(x['bytes'] for x in files)}))


if __name__=='__main__': main()
