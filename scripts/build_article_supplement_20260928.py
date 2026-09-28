#!/usr/bin/env python3
"""Typeset the existing English ArticleV1 appendix using cached tools only.

No scientific controller, World, mapper or evaluator is imported. The source
excerpt ends before the Chinese integration draft. Main manuscripts and their
builders are untouched.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'scripts'))
from build_virtual_paper_review_20260928 import PREAMBLE, RUNTIME, digest, pandoc, render

SOURCE = ROOT/'docs/thesis/ARTICLE_MULTI_INSTANCE_METHOD_APPENDIX_20260928.md'
STEM = 'ARTICLE_SUPPLEMENTARY_METHODS_20260928'
TITLE = 'Supplementary Methods: Observed Multi-Instance Planning'
TABLE_TITLES = ('Roles of the four CPU interfaces.',
                'Distinction between the earlier local method and ArticleV1.',
                'Information and planning access of the four methods.')


def inline(value):
    marker = 'SUPPLEMENTINLINE '
    converted = pandoc(marker+value)
    if not converted.startswith(marker):
        raise ValueError('unexpected inline conversion')
    return converted[len(marker):]


def prepare():
    original = SOURCE.read_text()
    boundary = '\n---\n\n# 中文并稿片段：'
    if original.count(boundary) != 1:
        raise ValueError('expected one explicit English/Chinese source boundary')
    excerpt = original.split(boundary, 1)[0].strip()+'\n'
    if re.findall(r'\\tag\{(A\d+)\}', excerpt) != ['A1','A2','A3','A4','A5','A6']:
        raise ValueError('expected all six unchanged labeled formulas')
    content = excerpt.split('\n', 1)[1].lstrip()
    content = re.sub(r'^## ([A-E]\. )', r'# \1', content, flags=re.M)
    tables = []

    def table(match):
        raw = match[0].strip().splitlines()
        rows = [[c.strip() for c in re.split(r'(?<!\\)\|', line.strip()[1:-1])]
                for line in [raw[0]]+raw[2:]]
        columns = len(rows[0])
        if any(len(row) != columns for row in rows) or len(tables) >= 3:
            raise ValueError('unexpected supplementary table')
        weights = {2:[.18,.82],3:[.20,.38,.42],4:[.10,.40,.23,.27]}[columns]
        number = len(tables)+1
        title = TABLE_TITLES[number-1]
        specs = '@{}'+' '.join('>{\\raggedright\\arraybackslash}p{'
            +str(w)+'\\SupplementTableWidth}' for w in weights)+'@{}'
        tex = ['\\par\\noindent\\begin{minipage}{\\linewidth}',
            '\\small\\textbf{Table S'+str(number)+'. '+title+'}\\par\\medskip',
            '\\centering', '\\setlength{\\tabcolsep}{4pt}',
            '\\renewcommand{\\arraystretch}{1.13}',
            '\\setlength{\\SupplementTableWidth}{\\dimexpr\\linewidth-'
                +str(2*(columns-1))+'\\tabcolsep\\relax}',
            '\\begin{tabular}{'+specs+'}', '\\toprule']
        for index,row in enumerate(rows):
            tex.append(' & '.join(inline(cell) for cell in row)+r' \\')
            if index == 0:
                tex.append('\\midrule')
        tex += ['\\bottomrule','\\end{tabular}','\\end{minipage}\\par\\medskip']
        tables.append(dict(number='S'+str(number), title=title, rows=len(rows)-1,
                           columns=columns, source_cells=rows))
        return '\n```{=latex}\n'+'\n'.join(tex)+'\n```\n'

    content = re.sub(r'^\|[^\n]+\|\n\|[ :|\-]+\|\n(?:^\|[^\n]+\|(?:\n|$))+',
                     table, content, flags=re.M)
    body = pandoc(content)
    if len(tables) != 3 or '\\begin{longtable}' in body:
        raise ValueError('all three tables must be converted')
    body = body.replace('\\section{', '\\section*{')
    body = body.replace('\\begin{verbatim}',
        '\\par\\noindent\\begin{minipage}{\\linewidth}\n\\small\n\\begin{verbatim}')
    body = body.replace('\\end{verbatim}', '\\end{verbatim}\n\\end{minipage}\\par')
    preamble = PREAMBLE.replace('面向预算受限设施建档的类别先验与几何反馈主动观测方法', TITLE)
    preamble = preamble.replace('李兆宇', 'Zhaoyu Li')
    preamble = preamble.replace('2026年9月28日\\quad 虚拟实验论文审阅稿',
                               '28 September 2026\\quad ArticleV1 supplementary methods')
    preamble = preamble.replace('\\begin{document}',
        '\\DeclareMathSizes{10.95}{10.95}{8}{7}\n'
        '\\clubpenalty=10000\n\\widowpenalty=10000\n'
        '\\displaywidowpenalty=10000\n\\newlength{\\SupplementTableWidth}\n'
        '\\renewcommand{\\tablename}{Table}\n\\begin{document}')
    return preamble+'\n'+body+'\n\\end{document}\n', excerpt, tables


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=ROOT/'tmp/article-supplement_20260928')
    parser.add_argument('--render', action='store_true')
    parser.add_argument('--export', action='store_true')
    args = parser.parse_args()
    output = args.output.resolve()
    if not output.is_relative_to(ROOT) or shutil.disk_usage(ROOT).free < 100*1024**2:
        raise ValueError('bounded repository build workspace required')
    output.mkdir(parents=True, exist_ok=True)
    sources = {str(p.relative_to(ROOT)):digest(p) for p in
        (SOURCE, Path(__file__).resolve(), ROOT/'scripts/build_virtual_paper_review_20260928.py')}
    tex, excerpt, tables = prepare()
    target = output/f'{STEM}.tex'
    target.write_text(tex)
    (output/'source_english_excerpt.md').write_text(excerpt)
    env = dict(os.environ)
    env.update(TECTONIC_CACHE_DIR=str(RUNTIME/'cache'), XDG_CACHE_HOME=str(RUNTIME/'cache'),
               XDG_CONFIG_HOME=str(RUNTIME/'config'), SOURCE_DATE_EPOCH='1790553600')
    command = [str(RUNTIME/'bin/tectonic'), '--only-cached', '--keep-logs', '-Z',
               f'search-path={ROOT}', '--outdir', str(output), str(target)]
    result = subprocess.run(command, cwd=ROOT, env=env, text=True,
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT, timeout=180)
    (output/'build_stdout.txt').write_text(result.stdout)
    if result.returncode:
        raise RuntimeError(result.stdout[-7000:])
    if any(digest(ROOT/p) != h for p,h in sources.items()):
        raise RuntimeError('build input changed during compilation')
    pdf = output/f'{STEM}.pdf'
    log = (output/f'{STEM}.log').read_text()
    diagnostics = [line for line in log.splitlines() if any(token in line for token in
        ('Overfull', 'undefined', 'Missing character', 'LaTeX Error'))]
    receipt = dict(schema='article.supplementary_methods_build.v1', title=TITLE,
        source_sha256=sources, excerpt_boundary='English only, before Chinese integration draft',
        english_excerpt_sha256=hashlib.sha256(excerpt.encode()).hexdigest(),
        equation_labels=['A1','A2','A3','A4','A5','A6'], tables=tables,
        pdf_sha256=digest(pdf), tex_sha256=digest(target), command=command,
        compiler_sha256=digest(RUNTIME/'bin/tectonic'),
        pandoc_version=subprocess.run(['pandoc','--version'],check=True,text=True,
            capture_output=True,timeout=30).stdout.splitlines()[0],
        downloads_allowed=False, typesetting_diagnostics=diagnostics,
        new_worlds=0, new_policy_runs=0, new_evaluations=0,
        status='supplementary_methods_review_draft')
    if args.render:
        receipt['pages'] = render(pdf, output)
        pages = json.loads((output/'visual/pages.json').read_text())
        extracted = '\n'.join(p['text'] for p in pages)
        receipt['extracted_equation_labels'] = [label for label in receipt['equation_labels']
                                               if '('+label+')' in extracted]
        if receipt['extracted_equation_labels'] != receipt['equation_labels']:
            diagnostics.append('Rendered PDF does not expose all six equation labels')
        receipt['rendered_pages_sha256'] = {p.name:digest(p) for p in sorted(
            (output/'visual').glob('page_*.png'))}
    (output/'build_receipt.json').write_text(json.dumps(receipt,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps(dict(pdf=str(pdf), pages=receipt.get('pages'), diagnostics=diagnostics)))
    if diagnostics:
        raise SystemExit('Fix supplementary typesetting before export')
    if args.export:
        destination = ROOT/'docs/thesis'
        shutil.copyfile(pdf,destination/pdf.name)
        shutil.copyfile(target,destination/target.name)
        shutil.copyfile(output/'build_receipt.json',destination/f'{STEM}.build.json')


if __name__ == '__main__':
    main()
