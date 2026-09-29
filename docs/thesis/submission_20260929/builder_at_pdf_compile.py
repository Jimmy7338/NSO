#!/usr/bin/env python3
"""Prepare a JIRS formatting draft using the unchanged official sn-jnl class.

No simulator/evaluator imports. One main.tex, local figure files, source-preserved
references, and explicit unresolved author fields. No upload or network access.
The optional cached Tectonic preview is not a claim of pdflatex/Snapp validation.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import tempfile
import zipfile

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / 'docs/thesis/ARTICLE_SUBMISSION_EN_20260928.md'
TEMPLATE = ROOT / 'docs/thesis/journal_template_20260929/springer'
UPSTREAM = TEMPLATE / 'upstream/sn-article-template'
RUNTIME = ROOT / 'tmp/thesis-tex-runtime_20260928'
GUIDE = 'https://link.springer.com/journal/10846/submission-guidelines'
READER = 'markdown+tex_math_single_backslash-implicit_figures'
PROTECTED = re.compile(r'```.*?```|`[^`\n]*`|\\\[.*?\\\]|\\\(.*?\\\)|\$\$.*?\$\$|(?<!\\)\$[^$\n]+\$', re.S)
CITATION = re.compile(r'(?<![\\!])\[(\d+(?:\s*[,\-–]\s*\d+)*)\](?!\()')
FIGURE = re.compile(r'!\[[^\]]*\]\((?P<image>[^)]+)\)\s*\n\n\*\*Figure (?P<number>\d+)\.\*\*\s*(?P<caption>[^\n]+)')
TABLE = re.compile(r'^\*\*Table (?P<number>\d+)\.\s*(?P<caption>[^\n]+?)\*\*\n\n(?P<grid>^\|[^\n]+\|\n\|[ :|\-]+\|\n(?:^\|[^\n]+\|(?:\n|$))+)(?:\n(?P<note>†[^\n]+)\n)?', re.M)


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def json_write(path: Path, value) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')


def tex_symbols(value: str) -> str:
    # No custom fonts: standard TeX expressions also compile under pdfLaTeX.
    substitutions = {'−': r'\ensuremath{-}', '×': r'\ensuremath{\times}',
                     '→': r'\ensuremath{\rightarrow}\allowbreak{}',
                     '≤': r'\ensuremath{\leq}', '≥': r'\ensuremath{\geq}',
                     '†': r'\ensuremath{\dagger}', 'ρ': r'\ensuremath{\rho}',
                     'λ': r'\ensuremath{\lambda}', 'Δ': r'\ensuremath{\Delta}',
                     '°': r'\ensuremath{^{\circ}}',
                     '–': '--', '—': '---', '’': "'", '‘': '`',
                     '“': '``', '”': "''", '\u00a0': '~'}
    for old, new in substitutions.items():
        value = value.replace(old, new)
    return value


def pandoc(text: str) -> str:
    out = subprocess.run(['pandoc', '--from', READER, '--to', 'latex',
                          '--wrap=none', '--no-highlight', '--ascii'],
                         input=text, text=True, capture_output=True, check=True,
                         timeout=30).stdout.strip()
    return tex_symbols(out)


def inline(text: str) -> str:
    return pandoc('INLINEPREFIX ' + text).removeprefix('INLINEPREFIX ')


def citation_ids(value: str) -> list[int]:
    ids = []
    for item in value.split(','):
        parts = re.split(r'[-–]', item.strip())
        if len(parts) == 1:
            ids.append(int(parts[0]))
        elif len(parts) == 2 and int(parts[0]) <= int(parts[1]):
            ids.extend(range(int(parts[0]), int(parts[1]) + 1))
        else:
            raise ValueError(f'Invalid reference range: {value}')
    return ids


def convert_citations(content: str, references: dict[int, str]):
    order = []

    def convert(match):
        ids = citation_ids(match[1])
        for key in ids:
            if key not in references:
                raise ValueError(f'Unknown reference {key}')
            if key not in order:
                order.append(key)
        return r'\cite{' + ','.join(f'ref{x}' for x in ids) + '}'

    parts, end = [], 0
    for match in PROTECTED.finditer(content):
        parts.extend([CITATION.sub(convert, content[end:match.start()]), match[0]])
        end = match.end()
    parts.append(CITATION.sub(convert, content[end:]))
    if set(order) != set(references):
        raise ValueError(f'Uncited references: {sorted(set(references) - set(order))}')
    return ''.join(parts), order


def split_source(content: str):
    title = re.search(r'^# ([^\n]+)', content, re.M)
    abstract = re.search(r'^## Abstract\n(.*?)\n\*\*Keywords:\*\*([^\n]+)', content, re.M | re.S)
    if not title or not abstract or '\n## References\n' not in content:
        raise ValueError('Expected title, Abstract, Keywords and References')
    body, reference_text = content[abstract.end():].split('\n## References\n', 1)
    reference_text = reference_text.split('\n## ', 1)[0]
    references = {}
    for match in re.finditer(r'^(\d+)\. ([^\n]+)', reference_text, re.M):
        key = int(match[1])
        if key in references:
            raise ValueError(f'Duplicate reference {key}')
        references[key] = match[2]
    keywords = [x.strip() for x in abstract[2].split(';') if x.strip()]
    abstract_text = abstract[1].strip()
    if not 150 <= len(abstract_text.split()) <= 250:
        raise ValueError('JIRS abstract must contain 150–250 words')
    if not 4 <= len(keywords) <= 6:
        raise ValueError('JIRS requires 4–6 keywords')
    if re.search(r'^#{5,} ', body, re.M):
        raise ValueError('More than three body heading levels')
    return title[1], abstract_text, keywords, body, references


def checked_local(raw: str, source: Path) -> Path:
    path = Path(raw)
    if not path.is_absolute():
        path = source.parent / path
    path = path.resolve()
    if not path.is_relative_to(ROOT) or not path.is_file():
        raise ValueError(f'Unavailable repository attachment: {raw}')
    return path


def source_key(path: Path) -> str:
    return str(path.resolve().relative_to(ROOT))


def verify_sources(sources: dict[str, str]) -> None:
    for name, sha in sources.items():
        if digest(ROOT / name) != sha:
            raise RuntimeError(f'Input changed during preparation/build: {name}')


def prepare(source: Path, destination: Path) -> dict:
    """Prepare in a fresh directory; never modify manuscript or upstream class."""
    if any(destination.iterdir()):
        raise ValueError('Preparation directory must be empty')
    raw = source.read_text()
    title, abstract, keywords, body, references = split_source(raw)
    body, reference_order = convert_citations(body, references)
    sources = {source_key(source): digest(source), source_key(Path(__file__)): digest(Path(__file__))}
    figures, tables, supplements = [], [], []
    supplement_ids = {}

    def copy(path, name):
        if path.stat().st_size > 80 * 1024**2:
            raise ValueError(f'Oversized publication attachment: {path.name}')
        sources[source_key(path)] = digest(path)
        shutil.copyfile(path, destination / name)

    def supplement(match):
        label, target = match[1], match[2]
        if re.match(r'^[a-z]+://|^#', target):
            return match[0]
        path = checked_local(target, source)
        if path.suffix == '.md' and path.with_suffix('.pdf').is_file():
            path = path.with_suffix('.pdf')
        if path not in supplement_ids:
            number = len(supplements) + 1
            filename = f'ESM_{number}{path.suffix}'
            copy(path, filename)
            supplements.append(dict(number=number, file=filename, caption=label, source=source_key(path)))
            supplement_ids[path] = number
        number = supplement_ids[path]
        return f'{label} (Online Resource {number})'

    def figure(match):
        number = int(match['number'])
        original = checked_local(match['image'], source)
        path = original.with_suffix('.pdf')
        if not path.is_file():
            raise ValueError(f'Vector PDF unavailable: {path}')
        name = f'Fig{number}.pdf'
        copy(path, name)
        caption = re.sub(r'\[(?:Vector PDF|PDF|矢量PDF)\]\([^)]+\)\.?', '', match['caption']).strip().rstrip('.')
        caption = re.sub(r'(?<!!)\[([^\]]+)\]\(([^)]+)\)', supplement, caption)
        figures.append(dict(number=number, file=name, source=source_key(path), caption=caption))
        return ('\n```{=latex}\n\\begin{figure}[!htbp]\n\\centering\n'
                f'\\includegraphics[width=\\linewidth,height=.67\\textheight,keepaspectratio]{{{name}}}\n'
                f'\\caption{{{inline(caption)}}}\\label{{fig:{number}}}\n\\end{{figure}}\n```\n')

    body = FIGURE.sub(figure, body)
    if len(figures) != len(re.findall(r'^!\[', raw, re.M)):
        raise ValueError('Unpaired or unsupported figure/caption')
    if [f['number'] for f in figures] != list(range(1, len(figures) + 1)):
        raise ValueError('Figures not numbered in order')
    body = re.sub(r'(?<!!)\[([^\]]+)\]\(([^)]+)\)', supplement, body)

    def table(match):
        lines = match['grid'].strip().splitlines()
        rows = [[c.strip() for c in re.split(r'(?<!\\)\|', line.strip()[1:-1])]
                for line in [lines[0]] + lines[2:]]
        n = len(rows[0])
        if n > 6 or n < 2 or any(len(row) != n for row in rows):
            raise ValueError('Tables require 2–6 consistent columns')
        number = int(match['number'])
        caption = match['caption'].rstrip('.')
        note = match['note']
        weights = {2: [.27, .73], 3: [.16, .32, .52], 4: [.32, .22, .22, .24],
                   5: [.28, .18, .18, .18, .18], 6: [.26, .148, .148, .148, .148, .148]}[n]
        specs = '@{}' + ' '.join(r'>{\raggedright\arraybackslash}p{' + str(w) + r'\JournalTableWidth}' for w in weights) + '@{}'
        result = [r'\begin{table}[!htbp]', f'\\caption{{{inline(caption)}}}\\label{{tab:{number}}}',
                  r'\small\setlength{\tabcolsep}{3pt}',
                  r'\setlength{\JournalTableWidth}{\dimexpr\linewidth-' + str(2 * (n-1)) + r'\tabcolsep\relax}',
                  r'\begin{tabular}{' + specs + '}', r'\toprule']
        for i, row in enumerate(rows):
            result.append(' & '.join(inline(cell.replace('†', r'\textsuperscript{a}') if note else cell)
                                     for cell in row) + r' \\')
            if i == 0:
                result.append(r'\midrule')
        result.extend([r'\bottomrule', r'\end{tabular}'])
        if note:
            result.extend([r'\begin{tablenotes}[flushleft]\footnotesize',
                           r'\item[a] ' + inline(note.removeprefix('†').strip()), r'\end{tablenotes}'])
        result.append(r'\end{table}')
        tables.append(dict(number=number, columns=n, rows=rows, caption=caption, original_note=note))
        # Editable inspection copies; main.tex embeds the same content directly.
        (destination / f'Table{number}.tex').write_text('\n'.join(result) + '\n')
        return '\n```{=latex}\n' + '\n'.join(result) + '\n```\n'

    body = TABLE.sub(table, body)
    if len(tables) != len(re.findall(r'^\*\*Table \d+', raw, re.M)):
        raise ValueError('Unpaired or unsupported table/caption')
    if [t['number'] for t in tables] != list(range(1, len(tables) + 1)):
        raise ValueError('Tables not numbered in order')
    body = re.sub(r'^(#{2,4}) (?:\d+(?:\.\d+)*\s+)?',
                  lambda m: m[1][1:] + ' ', body, flags=re.M)
    body_tex = pandoc(body)
    reference_items = []
    for key in reference_order:
        text = references[key]
        text = re.sub(r'\[[^\]]+\]\((https://doi\.org/[^)]+)\)', lambda m: f'<{m[1]}>', text)
        reference_items.append(r'\bibitem{ref' + str(key) + '} ' + inline(text))
    bibliography = r'\begin{thebibliography}{99}' + '\n' + '\n\n'.join(reference_items) + '\n' + r'\end{thebibliography}'
    (destination / 'references.tex').write_text(bibliography + '\n')
    json_write(destination / 'references.json', [dict(key=f'ref{k}', original_number=k, formatted_number=i+1, source_text=references[k]) for i, k in enumerate(reference_order)])
    manifest = json.loads((TEMPLATE / 'SOURCE_MANIFEST.json').read_text())
    for entry in manifest['upstream_files']:
        if digest(TEMPLATE / entry['path']) != entry['sha256']:
            raise ValueError(f'Official template changed: {entry["path"]}')
    copy(UPSTREAM / 'sn-jnl.cls', 'sn-jnl.cls')
    copy(UPSTREAM / 'bst/sn-mathphys-num.bst', 'sn-mathphys-num.bst')
    copy(UPSTREAM / 'user-manual.pdf', 'template_user_manual.pdf')
    copy(TEMPLATE / 'README.md', 'template_provenance.md')
    copy(TEMPLATE / 'SOURCE_MANIFEST.json', 'template_manifest.json')
    dependencies = TEMPLATE / 'dependencies'
    dependency_manifest = json.loads((dependencies / 'SOURCE_MANIFEST.json').read_text())
    for package in dependency_manifest['packages']:
        if digest(dependencies / (package['name'] + '.zip')) != package['sha256']:
            raise ValueError('Official package archive changed: ' + package['name'])
        for item in package['files']:
            if digest(dependencies / item['path']) != item['sha256']:
                raise ValueError(f'Package source changed: {item["path"]}')
        copy(dependencies / (package['name'] + '.zip'), 'template_dependency_' + package['name'] + '.zip')
    for item in json.loads((dependencies / 'GENERATION_RECEIPT.json').read_text()):
        path = dependencies / 'generated' / item['file']
        if digest(path) != item['sha256']:
            raise ValueError('Generated standard package changed: ' + item['file'])
        copy(path, item['file'])
    for name in ('wrapfig', 'threeparttable'):
        copy(dependencies / 'upstream' / name / (name + '.sty'), name + '.sty')
    copy(dependencies / 'SOURCE_MANIFEST.json', 'template_dependencies_manifest.json')
    copy(dependencies / 'GENERATION_RECEIPT.json', 'template_dependencies_generation.json')
    declarations = r'''\section*{Statements and Declarations}
\subsection*{Funding}
\textit{Author confirmation required:} \underline{\hspace{5cm}}
\subsection*{Competing interests}
\textit{Author confirmation required:} \underline{\hspace{5cm}}
\subsection*{Author contributions}
\textit{Author confirmation required:} \underline{\hspace{5cm}}
\subsection*{Data and code availability}
See the Reproducibility and Data section and the accompanying Online Resources.
The final public version identifier and any access restrictions require author confirmation.
\subsection*{AI assistance disclosure: draft for author confirmation}
OpenAI Codex assisted with manuscript organization and revision and with developing
experiment, analysis, and plotting code. Tool/model versions, the extent of assistance,
and the authors' verification and responsibility statement must be completed before submission.
This draft disclosure does not assert that the authors have already approved the manuscript.
'''
    preamble = r'''% Formatting draft for JIRS; NOT SUBMITTED. Author fields intentionally blank.
% The official sn-jnl.cls is unchanged. No custom fonts. One manuscript TeX.
\documentclass[pdflatex,sn-mathphys-num]{sn-jnl}
\usepackage{graphicx,amsmath,amssymb,booktabs,array,textcomp}
\hypersetup{colorlinks=true,linkcolor=black,citecolor=black,urlcolor=blue,pdfauthor={}}
\setlength{\emergencystretch}{3em}
\newlength{\JournalTableWidth}
\providecommand{\tightlist}{\setlength{\itemsep}{0pt}\setlength{\parskip}{0pt}}
\providecommand{\passthrough}[1]{#1}
\providecommand{\pandocbounded}[1]{#1}
\raggedbottom
\begin{document}
'''
    front = (f'\\title{{{inline(title)}}}\n'
             r'% Author name, affiliation, email and ORCID: deliberately not supplied.' + '\n'
             r'% \author*[1]{\fnm{} \sur{}}\email{}' + '\n'
             r'% \affil*[1]{\orgdiv{},\orgname{},\orgaddress{\city{},\country{}}}' + '\n'
             f'\\abstract{{{pandoc(abstract)}}}\n\\keywords{{{", ".join(inline(k) for k in keywords)}}}\n'
             r'\maketitle' + '\n'
             r'\noindent\textbf{Categories:} (2), (5), (8).\quad\textit{Formatting draft; not submitted.}' + '\n\n')
    if supplements:
        declarations += '\n\\subsection*{Online Resources}\n'
        declarations += '\n\n'.join(f'Online Resource {item["number"]}: {inline(item["caption"])} (\\texttt{{{item["file"].replace("_", r"\_")}}}).' for item in supplements)
    tex = preamble + front + body_tex + '\n\n' + bibliography + '\n\n' + declarations + '\n\\end{document}\n'
    if '/root/' in tex or '\\input{' in tex or '\\include{' in tex:
        raise ValueError('Submission main.tex must be flat and self-contained')
    nonascii = sorted(set(c for c in tex if ord(c) > 127))
    if nonascii:
        raise ValueError(f'Unconverted non-ASCII TeX characters: {nonascii}')
    (destination / 'main.tex').write_text(tex)
    (destination / 'manuscript_source.md').write_text(raw)
    json_write(destination / 'author_fields.json', dict(authors=[], affiliations=[], corresponding_email='', orcid='', funding='', competing_interests='', contributions='', msc_codes=[]))
    (destination / 'README.md').write_text('''# JIRS formatting package — not submitted

Target: Journal of Intelligent & Robotic Systems. This package uses the unchanged
official Springer Nature December 2024 sn-jnl template and the journal guidelines
checked on 2026-09-29. It is a formatting draft, not a claim of acceptance and not
a commitment to publication fees. JIRS is fully open access; the author must
separately decide on APC funding or a verified waiver before submission.

Compile the single main.tex with pdfLaTeX three times, with shell escape disabled:

    pdflatex -no-shell-escape -interaction=nonstopmode -halt-on-error main.tex

Keep all supplied files in one directory. Tables and references are embedded in
main.tex; Table*.tex and references.tex/json are editable inspection copies, not
external inputs required by the main document. Fig*.pdf files preserve full-size
vector figures. Inspect these originals for dense panel details; no claim is made
that every small label in the scaled manuscript is at least 8 pt. ESM_*.pdf files
are the numbered Online Resources. The original template notices and CTAN source
archives retain their own licenses, not the research repository's license.

Before any submission the authors must fill author_fields.json and the commented
author/affiliation/email fields in main.tex, confirm funding, competing interests,
contributions, MSC classification, AI-use details, final public data/version links,
and Online Resource author metadata. They must review and approve the scientific
text and disclosures. Blank fields and draft declarations are intentional; this
package must not be represented as an author-approved submission.

build_receipt.json binds the actual manuscript, figures, supplementary PDFs,
template, references, compiler result and output file hashes. No experiment or
metric recomputation is part of this publication build.
''')
    verify_sources(sources)
    return dict(schema='publication.jirs_format_package.v1', generated_at=datetime.now(timezone.utc).isoformat(),
                journal='Journal of Intelligent & Robotic Systems', official_guide=GUIDE,
                source_sha256=sources, figures=figures, tables=tables, online_resources=supplements,
                reference_count=len(references), reference_order=reference_order,
                abstract_words=len(abstract.split()), keyword_count=len(keywords),
                manuscript_status='formatting_draft_not_submitted', author_confirmation_pending=True,
                new_worlds=0, new_evaluations=0, downloads_allowed=False,
                actual_pdflatex_validation=False,
                pending=['author metadata', 'funding/conflicts/contributions', 'MSC classification',
                         'final data release identifier', 'confirmed AI disclosure',
                         'author metadata in Online Resources', 'APC/waiver decision'])


def build(directory: Path, receipt: dict, engine: str) -> None:
    env = dict(os.environ)
    env.update(TECTONIC_CACHE_DIR=str(RUNTIME / 'cache'), XDG_CACHE_HOME=str(RUNTIME / 'cache'),
               XDG_CONFIG_HOME=str(RUNTIME / 'config'), SOURCE_DATE_EPOCH='1790640000')
    if engine == 'tectonic':
        command = [str(RUNTIME / 'bin/tectonic'), '--only-cached', '--keep-logs', 'main.tex']
        repeats = 1
    else:
        executable = shutil.which('pdflatex')
        if not executable:
            raise RuntimeError('pdflatex is unavailable; no installation or download attempted')
        command = [executable, '-no-shell-escape', '-interaction=nonstopmode', '-halt-on-error', 'main.tex']
        repeats = 3
    outputs = []
    for _ in range(repeats):
        run = subprocess.run(command, cwd=directory, env=env, text=True,
                             stdout=subprocess.PIPE, stderr=subprocess.STDOUT, timeout=240)
        outputs.append(run.stdout)
        if run.returncode:
            (directory / 'build_stdout.txt').write_text('\n'.join(outputs))
            raise RuntimeError('TeX compilation failed; see build_stdout.txt. No automatic network fallback.\n' + run.stdout[-3500:])
    (directory / 'build_stdout.txt').write_text('\n'.join(outputs))
    log = (directory / 'main.log').read_text(errors='replace')
    diagnostics = [line for line in log.splitlines() if any(x in line for x in ('Overfull', 'undefined', 'Missing character', 'LaTeX Error'))]
    receipt.update(command=command, engine=engine, actual_pdflatex_validation=engine == 'pdflatex',
                   compiler_sha256=digest(Path(command[0])),
                   compiler_version=subprocess.run([command[0], '--version'], text=True,
                                                   capture_output=True, check=True).stdout.splitlines()[0],
                   typesetting_diagnostics=diagnostics, pdf_sha256=digest(directory / 'main.pdf'))
    verify_sources(receipt['source_sha256'])
    if diagnostics:
        raise RuntimeError('Typesetting diagnostics require review: ' + repr(diagnostics))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, default=SOURCE)
    parser.add_argument('--output', type=Path, default=ROOT / 'docs/thesis/submission_20260929')
    parser.add_argument('--work', type=Path, default=ROOT / 'tmp/final-journal_20260929')
    parser.add_argument('--build', action='store_true', help='compile after preparation; otherwise prepare source only')
    parser.add_argument('--engine', choices=['tectonic', 'pdflatex'], default='pdflatex')
    parser.add_argument('--render', action='store_true')
    parser.add_argument('--export', action='store_true', help='export verified package and ZIP after successful build')
    args = parser.parse_args()
    if (args.render or args.export) and not args.build:
        parser.error('--render and --export require --build')
    work, output, source = args.work.resolve(), args.output.resolve(), args.source.resolve()
    if not all(p.is_relative_to(ROOT) for p in [work, output, source]):
        raise ValueError('Source and output must be inside the repository')
    if shutil.disk_usage(ROOT).free < 160 * 1024**2:
        raise RuntimeError('Insufficient temporary publication space')
    work.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='stage-', dir=work) as name:
        stage = Path(name)
        receipt = prepare(source, stage)
        try:
            if args.build:
                build(stage, receipt, args.engine)
            if args.render:
                from build_virtual_paper_review_20260928 import render
                receipt['pages'] = render(stage / 'main.pdf', stage)
            receipt['file_sha256'] = {p.name: digest(p) for p in stage.iterdir() if p.is_file()}
            json_write(stage / 'build_receipt.json', receipt)
            # Keep a reproducible prepared copy even without exporting a publication.
            prepared = work / 'prepared'
            if prepared.exists():
                shutil.rmtree(prepared)
            shutil.copytree(stage, prepared)
            if args.export:
                if output.exists() and any(output.iterdir()):
                    raise RuntimeError('Refusing to overwrite an existing export; choose a fresh --output')
                shutil.copytree(stage, output, dirs_exist_ok=True, ignore=shutil.ignore_patterns('visual'))
                with zipfile.ZipFile(output.with_suffix('.zip'), 'w', zipfile.ZIP_DEFLATED) as archive:
                    for path in sorted(output.iterdir()):
                        if path.is_file():
                            archive.write(path, path.name)
            print(json.dumps(dict(prepared=str(prepared), output=str(output) if args.export else None,
                                  figures=len(receipt['figures']), tables=len(receipt['tables']),
                                  pages=receipt.get('pages'), status=receipt['manuscript_status']), ensure_ascii=False))
        except Exception:
            failed = work / 'failed'
            if failed.exists():
                shutil.rmtree(failed)
            shutil.copytree(stage, failed)
            json_write(failed / 'build_receipt.json', receipt)
            raise


if __name__ == '__main__':
    main()
