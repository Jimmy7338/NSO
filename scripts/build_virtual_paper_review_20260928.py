#!/usr/bin/env python3
"""Typeset the virtual application draft from Markdown, using cached Tectonic.

This is a publication-only build: no planner, simulator, fusion or evaluator is
imported. Optional PDF previews use the verified local rendering dependencies.
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
SOURCE = ROOT / 'docs/thesis/VIRTUAL_PAPER_DRAFT_20260928.md'
RUNTIME = ROOT / 'tmp/thesis-tex-runtime_20260928'
STEM = 'VIRTUAL_PAPER_REVIEW_20260928'
READER = 'markdown+tex_math_single_backslash-implicit_figures'

PREAMBLE = r'''\documentclass[11pt,a4paper]{article}
\usepackage[UTF8,fontset=none]{ctex}
\setCJKmainfont{Noto Serif CJK SC}[BoldFont={Noto Sans CJK SC Bold}]
\setCJKsansfont{Noto Sans CJK SC}
\setCJKmonofont{Noto Sans Mono CJK SC}
\setmainfont{Liberation Serif}
\setsansfont{Liberation Sans}
\setmonofont{Liberation Mono}
\usepackage{amsmath,amssymb,graphicx,booktabs,array}
\usepackage{geometry,caption,hyperref,flafter}
\geometry{left=22mm,right=22mm,top=23mm,bottom=24mm}
\hypersetup{colorlinks=true,linkcolor=black,citecolor=black,urlcolor=blue,
 pdftitle={面向预算受限设施建档的类别先验与几何反馈主动观测方法},pdfauthor={李兆宇}}
\captionsetup{font=small,labelfont=bf,labelsep=quad}
\setlength{\parskip}{3pt}
\setlength{\emergencystretch}{3em}
\renewcommand{\topfraction}{0.9}
\renewcommand{\bottomfraction}{0.85}
\renewcommand{\textfraction}{0.07}
\renewcommand{\floatpagefraction}{0.8}
\linespread{1.13}
\urlstyle{same}
\Urlmuskip=0mu plus 2mu
\providecommand{\tightlist}{\setlength{\itemsep}{0pt}\setlength{\parskip}{0pt}}
\providecommand{\passthrough}[1]{#1}
\providecommand{\pandocbounded}[1]{#1}
\title{\textbf{面向预算受限设施建档的类别先验与几何反馈主动观测方法}}
\author{李兆宇}
\date{2026年9月28日\quad 虚拟实验论文审阅稿}
\begin{document}
\maketitle
'''


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def pandoc(text):
    return subprocess.run(
        ['pandoc', '--from', READER, '--to', 'latex', '--wrap=none',
         '--no-highlight'], input=text, text=True, check=True,
        capture_output=True, timeout=30).stdout.strip()


def break_long_code(body):
    """Allow file names and paths in prose to wrap at their separators."""
    def replace(match):
        value = match[1]
        if len(value) < 24 or not ('/' in value or r'\_' in value):
            return match[0]
        plain = value.replace(r'\_', '_')
        if '\\' in plain:
            return match[0]
        return r'\path{' + plain + '}'
    return re.sub(r'\\texttt\{([^{}\n]+)\}', replace, body)


def prepare():
    content = SOURCE.read_text()
    # Repository provenance remains in Markdown; the PDF is the scientific body.
    content = content.split('\n## 仓库来源对应表')[0]
    content = re.sub(r'\A.*?^# [^\n]+\n', '', content, count=1, flags=re.S | re.M)
    abstract = re.search(r'^## 摘要\n(.*?)\n(?=## )', content, flags=re.S | re.M)
    if not abstract:
        raise ValueError('Expected abstract and following section')
    abstract_text = abstract.group(1).strip()
    abstract_text, keywords = abstract_text.split('\n\n**关键词：**', 1)
    content = content[abstract.end():]
    content = re.sub(r'^## (?:\d+\s+)?', '# ', content, flags=re.M)
    content = re.sub(r'^### (?:\d+(?:\.\d+)*\s+)?', '## ', content, flags=re.M)
    sources = {str(SOURCE.relative_to(ROOT)): digest(SOURCE)}
    figures = []

    def figure(match):
        number, caption = int(match['number']), match['caption']
        original = Path(match['image'])
        if not original.is_absolute():
            original = SOURCE.parent / original
        path = original.with_suffix('.pdf')
        if not path.is_file() or not path.resolve().is_relative_to(ROOT):
            raise ValueError(f'Missing local vector figure: {path}')
        name = str(path.relative_to(ROOT))
        sources[name] = digest(path)
        # The vector file is embedded; omit duplicate download links in print.
        caption = re.sub(r'\[矢量PDF\]\([^)]*\)[。.]?', '', caption).strip()
        cap = pandoc(caption)
        figures.append({'number': number, 'path': name, 'caption': caption})
        return ('\n```{=latex}\n\\begin{figure}[!htbp]\n\\centering\n'
                f'\\includegraphics[width=\\linewidth,height=.72\\textheight,keepaspectratio]{{{name}}}\n'
                f'\\caption{{{cap}}}\\label{{fig:virtual-{number}}}\n'
                '\\end{figure}\n```\n')

    pattern = (r'!\[[^\]]*\]\((?P<image>[^)]+)\)\s*\n\n'
               r'\*\*图(?P<number>\d+)\*\*\s*(?P<caption>[^\n]+)')
    content = re.sub(pattern, figure, content)
    if not figures or len(figures) != len(re.findall(r'\*\*图\d+\*\*', SOURCE.read_text())):
        raise ValueError('Every manuscript figure must have one paired caption')
    if [f['number'] for f in figures] != list(range(1, len(figures) + 1)):
        raise ValueError('Figure numbers must follow manuscript order')
    def table(match):
        lines = match[0].strip().splitlines()
        rows = [[c.strip() for c in re.split(r'(?<!\\)\|', line.strip()[1:-1])]
                for line in [lines[0]] + lines[2:]]
        columns = len(rows[0])
        assert all(len(row) == columns for row in rows)
        weights = {2: [.24, .76], 3: [.24, .34, .42],
                   4: [.25]*4, 5: [.20]*5, 6: [.17, .17, .17, .17, .16, .16]}[columns]
        specs = '@{}' + ' '.join('>{\\raggedright\\arraybackslash}p{'
                  + str(w) + '\\ReviewTableWidth}' for w in weights) + '@{}'
        tex = ['\\begin{center}', '\\small', '\\setlength{\\tabcolsep}{4pt}',
               '\\setlength{\\ReviewTableWidth}{\\dimexpr\\linewidth-'
               + str(2*(columns-1)) + '\\tabcolsep\\relax}',
               '\\begin{tabular}{' + specs + '}', '\\toprule']
        for index, row in enumerate(rows):
            tex.append(' & '.join(pandoc('NSOINLINE '+cell).removeprefix('NSOINLINE ')
                                  for cell in row) + r' \\')
            if index == 0: tex.append('\\midrule')
        tex += ['\\bottomrule', '\\end{tabular}', '\\end{center}']
        return '\n```{=latex}\n' + '\n'.join(tex) + '\n```\n'

    content = re.sub(r'^\|[^\n]+\|\n\|[ :|\-]+\|\n(?:^\|[^\n]+\|(?:\n|$))+',
                     table, content, flags=re.M)
    body = break_long_code(pandoc(content))
    # All scientific tables in this short draft fit on one page. Use the cached
    # tabular environment instead of downloading a multipage-table package.
    body = body.replace(r'\begin{longtable}[]', '\\begin{center}\n\\small\n\\begin{tabular}')
    body = body.replace('\\endhead\n', '')
    body = body.replace('\\bottomrule\\noalign{}\n\\endlastfoot\n', '')
    body = body.replace(r'\end{longtable}', '\\bottomrule\n\\end{tabular}\n\\end{center}')
    body = re.sub(r'(\\textbf\{表\d+[^\n]+\}\n\n)(\\begin\{center\}.*?\\end\{center\})',
                  lambda m: '\\par\\noindent\\begin{minipage}{\\linewidth}\n'
                  + m.group(1) + m.group(2) + '\n\\end{minipage}\n', body, flags=re.S)
    # Keep the short algorithm and its title together, without splitting the
    # last execution/recording steps onto the next page.
    body = re.sub(r'(\\textbf\{算法1[^\n]+\}\n\n)(\\begin\{verbatim\}.*?\\end\{verbatim\})',
                  lambda m: '\\par\\noindent\\begin{minipage}{\\linewidth}\n\\small\n'
                  + m.group(1) + m.group(2) + '\n\\end{minipage}\n', body, flags=re.S)
    body = body.replace(r'\section{参考文献}', r'\section*{参考文献}')
    # Flush the saved experimental figures before interpretation, preventing
    # them from being inserted halfway through the final conclusion.
    body = body.replace(r'\section{讨论与适用边界}', '\\clearpage\n\\section{讨论与适用边界}')
    # An explicit unnumbered appendix can be kept in the print draft.
    body = body.replace(r'\section{复现与数据说明}', r'\section*{复现与数据说明}')
    preamble = PREAMBLE.replace('\\begin{document}',
                               '\\newlength{\\ReviewTableWidth}\n\\begin{document}')
    tex = (preamble + '\n\\begin{abstract}\n' + pandoc(abstract_text)
           + '\n\\end{abstract}\n\\noindent\\textbf{关键词：}'
           + pandoc(keywords) + '\n\n' + body + '\n\\end{document}\n')
    sources[str(Path(__file__).resolve().relative_to(ROOT))] = digest(Path(__file__))
    return tex, sources, figures


def render(pdf, output):
    preview_packages = ROOT / 'tmp/thesis-python_20260928/site-packages'
    if preview_packages.is_dir() and str(preview_packages) not in sys.path:
        sys.path.insert(0, str(preview_packages))
    import pypdfium2 as pdfium
    from PIL import Image, ImageDraw
    visual = output / 'visual'
    visual.mkdir(exist_ok=True)
    document = pdfium.PdfDocument(pdf)
    pages = []
    for index in range(len(document)):
        page = document[index]
        textpage = page.get_textpage()
        text = textpage.get_text_bounded()
        page.render(scale=1).to_pil().save(visual / f'page_{index+1:02}.png')
        pages.append({'page': index + 1, 'text': text})
        textpage.close()
        page.close()
    document.close()
    for start in range(0, len(pages), 6):
        sheet = Image.new('RGB', (1260, 1860), '#eeeeee')
        draw = ImageDraw.Draw(sheet)
        for slot, index in enumerate(range(start, min(start+6, len(pages)))):
            with Image.open(visual / f'page_{index+1:02}.png') as im:
                im.thumbnail((600, 585))
                x = slot % 2 * 630 + (630-im.width)//2
                y = slot // 2 * 620 + 26
                sheet.paste(im, (x, y))
                draw.text((slot % 2*630+12, slot//2*620+6), f'Page {index+1}', fill='black')
        sheet.save(visual / f'contact_{start+1:02}.png')
    (visual / 'pages.json').write_text(json.dumps(pages, ensure_ascii=False, indent=2)+'\n')
    return len(pages)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=ROOT/'tmp/virtual-paper-review_20260928')
    parser.add_argument('--render', action='store_true')
    parser.add_argument('--export', action='store_true',
                        help='copy the successful local review PDF, TeX and receipt to docs/thesis')
    args = parser.parse_args()
    output = args.output.resolve()
    if not output.is_relative_to(ROOT):
        raise ValueError('Build output must be inside the repository')
    if shutil.disk_usage(ROOT).free < 100 * 1024**2:
        raise RuntimeError('Insufficient publication workspace')
    output.mkdir(parents=True, exist_ok=True)
    tex, sources, figures = prepare()
    target = output / f'{STEM}.tex'
    target.write_text(tex)
    (output / SOURCE.name).write_bytes(SOURCE.read_bytes())
    env = dict(os.environ)
    env.update(TECTONIC_CACHE_DIR=str(RUNTIME/'cache'), XDG_CACHE_HOME=str(RUNTIME/'cache'),
               XDG_CONFIG_HOME=str(RUNTIME/'config'), SOURCE_DATE_EPOCH='1790553600')
    command = [str(RUNTIME/'bin/tectonic'), '--only-cached', '--keep-logs',
               '-Z', f'search-path={ROOT}', '--outdir', str(output), str(target)]
    run = subprocess.run(command, cwd=ROOT, env=env, text=True,
                         stdout=subprocess.PIPE, stderr=subprocess.STDOUT, timeout=180)
    (output / 'build_stdout.txt').write_text(run.stdout)
    if run.returncode:
        raise RuntimeError(run.stdout[-6000:])
    for name, sha in sources.items():
        if digest(ROOT/name) != sha:
            raise RuntimeError(f'Input changed during build: {name}')
    pdf = output / f'{STEM}.pdf'
    log = (output / f'{STEM}.log').read_text()
    diagnostics = [line for line in log.splitlines() if any(x in line for x in
                   ('Overfull', 'undefined', 'Missing character', 'LaTeX Error'))]
    receipt = dict(source_sha256=sources, figures=figures, pdf_sha256=digest(pdf),
                   tex_sha256=digest(target), command=command, exit_code=run.returncode,
                   downloads_allowed=False, typesetting_diagnostics=diagnostics,
                   new_worlds=0, new_evaluations=0, manuscript_status='review_draft')
    if args.render:
        receipt['pages'] = render(pdf, output)
    (output / 'build_receipt.json').write_text(json.dumps(receipt, ensure_ascii=False, indent=2)+'\n')
    print(json.dumps({'pdf':str(pdf), 'pages':receipt.get('pages'),
                      'figures':len(figures), 'diagnostics':diagnostics},ensure_ascii=False))
    if diagnostics:
        raise SystemExit('Review typesetting diagnostics before publishing this build')
    if args.export:
        destination = ROOT / 'docs/thesis'
        shutil.copyfile(pdf, destination / pdf.name)
        shutil.copyfile(target, destination / target.name)
        shutil.copyfile(output / 'build_receipt.json', destination / f'{STEM}.build.json')


if __name__ == '__main__':
    main()
