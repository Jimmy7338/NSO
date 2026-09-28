#!/usr/bin/env python3
"""Build and render the V46 manuscript offline; no simulation or evaluation."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import signal
import subprocess
import time

ROOT = Path(__file__).resolve().parents[1]
PAPER = ROOT / 'docs/thesis/NSO_Manuscript_V46_20260922.tex'
RUNTIME = ROOT / 'tmp/v38-tex'


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write(path, data):
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + '\n')


def source_closure():
    sources, texts = {}, []

    def visit(path):
        name = str(path.relative_to(ROOT))
        if name in sources:
            return
        sources[name] = dict(sha256=sha(path), bytes=path.stat().st_size)
        if path.suffix != '.tex':
            return
        content = path.read_text()
        texts.append(content)
        for ref in re.findall(r'\\input\{([^}]+)\}', content):
            visit(ROOT / ref)
        for ref in re.findall(r'\\includegraphics(?:\[[^]]*\])?\{([^}]+)\}', content):
            visit(ROOT / ref.replace(r'\figpath ', 'docs/thesis/figures/v38/'))

    visit(PAPER)
    tex = '\n'.join(texts)
    labels = re.findall(r'\\label\{([^}]+)\}', tex)
    refs = set(re.findall(r'\\(?:eqref|ref)\{([^}]+)\}', tex))
    cites = {k for group in re.findall(r'\\cite\{([^}]+)\}', tex) for k in group.split(',')}
    bib = set(re.findall(r'\\bibitem\{([^}]+)\}', tex))
    assert len(labels) == len(set(labels)), 'duplicate labels'
    assert refs <= set(labels), refs - set(labels)
    assert cites == bib, (cites - bib, bib - cites)
    return sources, dict(labels=len(labels), citations=len(cites),
                        figures=tex.count(r'\begin{figure}'), tables=tex.count(r'\begin{table}'))


def build(output, allow_downloads=False):
    assert shutil.disk_usage(ROOT).free > 80 * 1024**2
    sources, structure = source_closure()
    output.mkdir(parents=True, exist_ok=False)
    write(output / 'input_sha256.json', sources)
    # Snapshot the editable writing inputs before compilation. Existing figure
    # and table versions are bound separately by their full source hashes.
    for name in sources:
        if name.endswith('.tex'):
            dest = output / 'writing_sources' / name
            dest.parent.mkdir(parents=True, exist_ok=True)
            dest.write_bytes((ROOT / name).read_bytes())
    env = dict(os.environ)
    overrides = dict(TECTONIC_CACHE_DIR=str(RUNTIME / 'cache'),
                     XDG_CACHE_HOME=str(RUNTIME / 'cache'),
                     XDG_CONFIG_HOME=str(RUNTIME / 'config'),
                     SOURCE_DATE_EPOCH='1790035200')
    env.update(overrides)
    command = [str(RUNTIME / 'bin/tectonic'), '--keep-logs',
               '--keep-intermediates', '-Z', f'search-path={ROOT}',
               '--outdir', str(output), str(PAPER)]
    if not allow_downloads:
        command.insert(1, '--only-cached')
    started, reason = time.monotonic(), None
    minimum = shutil.disk_usage(ROOT).free
    stdout = output / 'build_stdout.txt'
    with stdout.open('w') as stream:
        child = subprocess.Popen(command, cwd=ROOT, env=env, stdout=stream,
                                 stderr=subprocess.STDOUT, start_new_session=True)
        while child.poll() is None:
            minimum = min(minimum, shutil.disk_usage(ROOT).free)
            if minimum < 80 * 1024**2:
                reason = '80_MiB_publication_reserve'
            elif time.monotonic() - started > 240:
                reason = '240_second_build_timeout'
            elif stdout.stat().st_size > 8 * 1024**2:
                reason = '8_MiB_build_log_limit'
            if reason:
                os.killpg(child.pid, signal.SIGTERM)
                try:
                    child.wait(timeout=4)
                except subprocess.TimeoutExpired:
                    os.killpg(child.pid, signal.SIGKILL)
                    child.wait()
                break
            time.sleep(.1)
    receipt = dict(command=command, environment_overrides=overrides,
                   exit_code=child.returncode, stop_reason=reason,
                   elapsed_s=time.monotonic()-started, minimum_free_bytes=minimum,
                   downloads_allowed=allow_downloads, tectonic_sha256=sha(RUNTIME/'bin/tectonic'),
                   input_sha256=sha(output/'input_sha256.json'),
                   source_script_sha256=sha(Path(__file__)), structure=structure,
                   new_worlds=0, new_performance_samples=0)
    write(output / 'build_receipt.json', receipt)
    assert child.returncode == 0 and reason is None, stdout.read_text()[-4000:]
    assert sources == source_closure()[0], 'writing inputs changed during build'
    log = (output / f'{PAPER.stem}.log').read_text()
    prohibited = ('Overfull', 'Underfull', 'undefined', 'Missing character',
                  'LaTeX Error', 'Package xeCJK Warning')
    diagnostics = [line for line in log.splitlines() if any(v in line for v in prohibited)]
    write(output / 'typesetting_check.json', dict(passed=not diagnostics,
          prohibited_diagnostics=diagnostics, pdf_sha256=sha(output/f'{PAPER.stem}.pdf')))
    print(json.dumps(dict(**receipt, typesetting_diagnostics=diagnostics), ensure_ascii=False, indent=2))


def render(output):
    import pypdfium2 as pdfium
    from PIL import Image, ImageDraw
    pdf = output / f'{PAPER.stem}.pdf'
    review = output / 'visual'
    review.mkdir(exist_ok=False)
    doc = pdfium.PdfDocument(pdf)
    pages = []
    for i in range(len(doc)):
        page = doc[i]
        textpage = page.get_textpage()
        text = textpage.get_text_bounded()
        pages.append(dict(page=i+1, text=text,
                          text_sha256=hashlib.sha256(text.encode()).hexdigest()))
        page.render(scale=1).to_pil().save(review/f'page_{i+1:02d}.png')
        textpage.close()
        page.close()
    for start in range(0, len(pages), 6):
        canvas = Image.new('RGB', (1260, 1860), '#eeeeee')
        draw = ImageDraw.Draw(canvas)
        for slot, i in enumerate(range(start, min(start+6, len(pages)))):
            with Image.open(review/f'page_{i+1:02d}.png') as img:
                img.thumbnail((600, 586))
                x = slot % 2 * 630 + (630-img.width)//2
                y = slot // 2 * 620 + 25
                canvas.paste(img, (x, y))
                draw.text((slot % 2*630+12, slot//2*620+6), f'Page {i+1}', fill='black')
        canvas.save(review/f'contact_{start+1:02d}.png')
    write(review / 'pages.json', dict(pdf_sha256=sha(pdf), pages=pages))
    assert all('??' not in p['text'] and '\ufffd' not in p['text'] for p in pages)
    print(json.dumps(dict(pages=len(pages), pdf_sha256=sha(pdf), review=str(review))))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=('build', 'render'))
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--allow-downloads', action='store_true',
                        help='allow TeX to cache only required missing resources; default offline')
    args = parser.parse_args()
    output = args.output.resolve()
    assert output.is_relative_to(ROOT), 'publication output must remain in repository'
    build(output, args.allow_downloads) if args.command == 'build' else render(output)
