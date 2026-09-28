#!/usr/bin/env python3
"""Fetch pinned original source trees only; never install or execute their code."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import tarfile
import tempfile
import urllib.request

ROOT = Path(__file__).resolve().parents[1]


def prepare(inventory, output, report):
    output.mkdir(parents=True, exist_ok=True)
    records = []
    for item in json.loads(inventory.read_text())['pinned_baseline_sources']:
        target = output / item['id']
        if target.exists():
            raise FileExistsError(f'refusing to overwrite {target}')
        if shutil.disk_usage(output).free < 700 * 1024**2:
            raise OSError('less than 700 MiB available for bounded source preparation')
        repo = item['url'].split('https://github.com/', 1)[1]
        commit = item['commit']
        url = f'https://codeload.github.com/{repo}/tar.gz/{commit}'
        with tempfile.TemporaryDirectory(prefix='.baseline-source-', dir=output) as temporary:
            temporary = Path(temporary)
            archive = temporary / 'source.tar.gz'
            total = 0
            digest = hashlib.sha256()
            with urllib.request.urlopen(url, timeout=60) as response, archive.open('wb') as stream:
                while chunk := response.read(1024**2):
                    total += len(chunk)
                    if total > 200 * 1024**2:
                        raise OSError('archive exceeded 200 MiB download cap')
                    stream.write(chunk)
                    digest.update(chunk)
            with tarfile.open(archive) as source:
                members = source.getmembers()
                if sum(m.size for m in members if m.isfile()) > 400 * 1024**2:
                    raise OSError('expanded source exceeded 400 MiB cap')
                roots = {m.name.split('/')[0] for m in members}
                if len(roots) != 1 or not next(iter(roots)).endswith(commit):
                    raise ValueError('archive root does not match pinned commit')
                # Catkin repositories may contain src/CMakeLists.txt pointing
                # into /opt/ros on the author's host. Record these external
                # workspace links; never materialize them during extraction.
                external_links = {m.name: m.linkname for m in members
                                  if (m.issym() or m.islnk()) and Path(m.linkname).is_absolute()}
                source.extractall(temporary / 'tree',
                    members=[m for m in members if m.name not in external_links], filter='data')
            extracted = temporary / 'tree' / next(iter(roots))
            extracted.rename(target)
        files = {}
        for path in sorted(target.rglob('*')):
            if path.is_file() and not path.is_symlink():
                files[str(path.relative_to(target))] = hashlib.sha256(path.read_bytes()).hexdigest()
        records.append(dict(id=item['id'], repository=item['url'], commit=commit,
            archive_url=url, archive_sha256=digest.hexdigest(), archive_bytes=total,
            directory=str(target.relative_to(ROOT)), files_sha256=files,
            status='source_prepared_only', build_executed=False, demo_executed=False,
            adaptations_applied=False, omitted_external_workspace_links=external_links,
            workspace_initialization_required=bool(external_links)))
        print(item['id'], commit, len(files), 'source files; not built', flush=True)
        report.parent.mkdir(parents=True, exist_ok=True)
        report.write_text(json.dumps(dict(scope='official source preparation, not baseline reproduction',
            status='complete' if len(records)==len(json.loads(inventory.read_text())['pinned_baseline_sources']) else 'partial',
            sources=records), ensure_ascii=False, indent=2)+'\n')


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--inventory', type=Path, default=ROOT/'docs/research/source_inventory.json')
    p.add_argument('--output', type=Path, default=ROOT/'third_party/official_baselines')
    p.add_argument('--report', type=Path, default=ROOT/'docs/research/official_baseline_source_manifest.json')
    args = p.parse_args()
    prepare(args.inventory, args.output, args.report)
