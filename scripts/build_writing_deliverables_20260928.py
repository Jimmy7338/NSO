#!/usr/bin/env python3
"""Build the English article and accompanying thesis drafts, without experiments."""
import argparse
from functools import lru_cache
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys

from build_virtual_paper_review_20260928 import (
    ROOT, PREAMBLE, digest, pandoc, render, break_long_code,
)

RUNTIME = ROOT/'tmp/thesis-tex-runtime_20260928'
DOCS = ROOT / 'docs/thesis'
CHAPTERS = [DOCS/'GRADUATION_METHOD_CHAPTER_20260928.md',
            DOCS/'GRADUATION_SIMULATION_CHAPTER_20260928.md']
REFERENCES = DOCS/'GRADUATION_CORE_REFERENCES_20260928.md'
THESIS_CHAPTERS = [DOCS/'GRADUATION_INTRODUCTION_20260928.md',
                   DOCS/'GRADUATION_FOUNDATIONS_20260928.md',
                   *CHAPTERS, DOCS/'GRADUATION_CONCLUSION_20260928.md']
THESIS_REFERENCES = DOCS/'GRADUATION_REFERENCES_20260928.md'
ABSTRACTS = DOCS/'GRADUATION_ABSTRACTS_20260928.md'
APPENDIX = DOCS/'GRADUATION_REPRODUCTION_APPENDIX_20260928.md'
THESIS_TITLE = '面向预算受限设施建档的类别先验与几何反馈主动观测方法'


@lru_cache(maxsize=512)
def inline(text):
    # A caption starting with '(A)' must remain inline, not become a Markdown
    # enumerated list inside a LaTeX caption argument.
    marker = 'NSOINLINE '
    converted = pandoc(marker+text)
    assert converted.startswith(marker),converted
    return converted[len(marker):]


def scientific_body(text):
    return re.split(r'\n## (?:仓库来源对应表|Repository provenance)', text, maxsplit=1)[0]


def combine_chapters(full=False):
    parts = ['# 毕业论文核心章节：系统方法与仿真实验\n\n'
             '本稿为两章核心正文的合并审阅版，章节编号在最终学校模板中统一调整。'
             '学校没有硬性发表要求；当前按文章优先、毕业论文同步扩展推进，'
             '实车验证作为后续补充。\n']
    figure_offset = table_offset = 0
    sources = {}
    if full:
        abstract = scientific_body(ABSTRACTS.read_text()).split('\n',1)[1].strip()
        parts = ['# '+THESIS_TITLE+'\n\n'+abstract+'\n']
        sources[str(ABSTRACTS.relative_to(ROOT))] = digest(ABSTRACTS)
    chapters = THESIS_CHAPTERS if full else CHAPTERS
    references = THESIS_REFERENCES if full else REFERENCES
    for index, source in enumerate(chapters, 1):
        text = scientific_body(source.read_text())
        title = re.search(r'^# (.+)$', text, flags=re.M).group(1)
        title = re.sub(r'[（(]毕设章节草稿[）)]', '', title).strip()
        text = re.sub(r'\A.*?^# [^\n]+\n', '', text, count=1, flags=re.S | re.M).strip()
        if full:
            # The experimental chapter's source-editing note is useful in its
            # standalone draft but is not part of the integrated thesis body.
            text = re.sub(r'^本章(?:依据已经保存的在线实验|整合已有在线实验)[^\n]+\n\s*', '', text)
        figs = [int(x) for x in re.findall(r'^\*\*图(\d+)\*\*', text, flags=re.M)]
        tables = [int(x) for x in re.findall(r'^\*\*表(\d+)', text, flags=re.M)]
        assert figs == list(range(1,len(figs)+1)), (source, figs)
        assert tables == list(range(1,len(tables)+1)), (source, tables)
        text = re.sub(r'图\s*(\d+)', lambda m:'图'+str(int(m[1])+figure_offset), text)
        text = re.sub(r'表\s*(\d+)', lambda m:'表'+str(int(m[1])+table_offset), text)
        # Add one heading level; remove source-local numbering in the PDF later.
        text = re.sub(r'^(#{2,}) ', lambda m:'#'+m[1]+' ', text, flags=re.M)
        parts.append(f'\n## {index} {title}\n\n'+text+'\n')
        figure_offset += len(figs)
        table_offset += len(tables)
        sources[str(source.relative_to(ROOT))] = digest(source)
    if full:
        appendix = APPENDIX.read_text()
        appendix = re.sub(r'^(#+) ', lambda m:'#'+m[1]+' ', appendix, flags=re.M)
        parts.append('\n'+appendix+'\n')
        sources[str(APPENDIX.relative_to(ROOT))] = digest(APPENDIX)
    parts.append('\n## 参考文献\n\n'+references.read_text().split('\n',1)[1].strip()+'\n')
    sources[str(references.relative_to(ROOT))] = digest(references)
    path = DOCS / ('GRADUATION_THESIS_20260928.md' if full else 'GRADUATION_CORE_20260928.md')
    path.write_text('\n'.join(parts))
    return path, sources


def prepare(document):
    if document in ('graduation','thesis'):
        source, sources = combine_chapters(full=document=='thesis')
        stem = 'GRADUATION_THESIS_20260928' if document=='thesis' else 'GRADUATION_CORE_20260928'
    else:
        source = DOCS/'VIRTUAL_PAPER_EN_20260928.md'
        sources, stem = {}, 'VIRTUAL_PAPER_EN_20260928'
    sources[str(source.relative_to(ROOT))] = digest(source)
    content = scientific_body(source.read_text())
    title = re.search(r'^# (.+)$', content, flags=re.M).group(1)
    content = re.sub(r'\A.*?^# [^\n]+\n', '', content, count=1, flags=re.S | re.M)
    abstract_tex = ''
    if document == 'thesis':
        for heading, label, keyword in [('摘要','摘要','关键词：'),
                                        ('Abstract','Abstract','Keywords:')]:
            match = re.search(r'^## '+heading+r'\n(.*?)\n(?=## )',content,flags=re.S|re.M)
            if not match:
                raise ValueError('Missing thesis '+heading)
            abstract, keywords = re.split(r'\n\n\*\*'+keyword+r'\*\*\s*',
                                         match[1].strip(),maxsplit=1)
            abstract_tex += ('\\section*{'+label+'}\n'
                             '\\addcontentsline{toc}{section}{'+label+'}\n'
                             +pandoc(abstract)+'\n\n\\noindent\\textbf{'+keyword+'} '
                             +inline(keywords)+'\n\\clearpage\n')
            content = content[:match.start()]+content[match.end():]
        abstract_tex += ('{\\small\\linespread{.96}\\selectfont\\setlength{\\parskip}{0pt}\\tableofcontents}\n\\clearpage\n'
                         '\\pagenumbering{arabic}\n')
    if document == 'english':
        match = re.search(r'^## Abstract\n(.*?)\n(?=## )', content, flags=re.S | re.M)
        if not match:
            raise ValueError('Expected an English Abstract section')
        abstract, keywords = re.split(r'\n\n\*\*Keywords:\*\*\s*', match[1].strip(), maxsplit=1)
        abstract_tex = ('\\begin{abstract}\n'+pandoc(abstract)+'\n\\end{abstract}\n'
                        '\\noindent\\textbf{Keywords:} '+inline(keywords)+'\n\n')
        content = content[match.end():]
    content = re.sub(r'^(#{2,}) (?:\d+(?:\.\d+)*[.：:]?\s+)?',
                     lambda m:m[1][1:]+' ', content, flags=re.M)
    figures, tables = [], []

    def figure(match):
        number, caption = int(match['number']), match['caption']
        path = Path(match['image'])
        if not path.is_absolute():
            path = source.parent/path
        path = path.with_suffix('.pdf')
        assert path.is_file() and path.resolve().is_relative_to(ROOT),path
        name = str(path.relative_to(ROOT))
        sources[name] = digest(path)
        caption = re.sub(r'\[(?:矢量PDF|Vector PDF|PDF)\]\([^)]*\)[。.]?', '', caption,
                         flags=re.I).strip()
        figures.append({'number':number,'path':name,'caption':caption})
        if document=='thesis':
            return ('\n```{=latex}\n\\begin{figure}[!htbp]\n\\centering\n'
                    +f'\\includegraphics[width=\\linewidth,height=.72\\textheight,keepaspectratio]{{{name}}}\n'
                    +f'\\caption{{{inline(caption)}}}\\label{{fig:writing-{number}}}\n'
                    +'\\end{figure}\n```\n')
        return ('\n```{=latex}\n\\begin{center}\n\\begin{minipage}{\\linewidth}\n'
                '\\centering\n'+f'\\includegraphics[width=\\linewidth,height=.72\\textheight,keepaspectratio]{{{name}}}\n'
                f'\\captionof{{figure}}{{{inline(caption)}}}\\label{{fig:writing-{number}}}\n'
                '\\end{minipage}\n\\end{center}\n```\n')

    content = re.sub(r'!\[[^\]]*\]\((?P<image>[^)]+)\)\s*\n\n'
                     r'\*\*(?:图|Figure\s+)(?P<number>\d+)\.?\*\*\s*(?P<caption>[^\n]+)',
                     figure, content)
    assert [f['number'] for f in figures] == list(range(1,len(figures)+1))
    assert not re.search(r'!\[[^\]]*\]\(',content),'Unpaired figure caption'

    def table(match):
        lines = match[0].strip().splitlines()
        rows = [[c.strip() for c in re.split(r'(?<!\\)\|',line.strip()[1:-1])]
                for line in [lines[0]]+lines[2:]]
        columns = len(rows[0])
        assert all(len(row)==columns for row in rows),rows
        tables.append({'columns':columns,'data_rows':len(rows)-1})
        weights = {2:[.24,.76],3:[.28,.36,.36],4:[.25,.25,.25,.25],
                   5:[.20,.20,.20,.20,.20],6:[.17,.17,.17,.17,.16,.16]}[columns]
        specs = '@{}'+' '.join('>{\\raggedright\\arraybackslash}p{'
                              +str(w)+'\\ReviewTableWidth}' for w in weights)+'@{}'
        lines = ['\\begin{center}','\\small','\\setlength{\\tabcolsep}{4pt}',
                 '\\setlength{\\ReviewTableWidth}{\\dimexpr\\linewidth-'
                 +str(2*(columns-1))+'\\tabcolsep\\relax}',
                 '\\begin{tabular}{'+specs+'}','\\toprule']
        for index,row in enumerate(rows):
            lines.append(' & '.join(inline(c) for c in row)+r' \\')
            if index==0:lines.append('\\midrule')
        lines += ['\\bottomrule','\\end{tabular}','\\end{center}']
        return '\n```{=latex}\n'+'\n'.join(lines)+'\n```\n'

    content = re.sub(r'^\|[^\n]+\|\n\|[ :|\-]+\|\n(?:^\|[^\n]+\|(?:\n|$))+',
                     table, content, flags=re.M)
    body = break_long_code(pandoc(content))
    assert '\\begin{longtable}' not in body, 'Unexpected unconverted table'
    # Short scientific tables and pseudocode stay with their captions.
    body = re.sub(r'(\\textbf\{(?:表\d+|Table\s+\d+)[^\n]+\}\n\n)'
                  r'(\\begin\{center\}.*?\\end\{center\})',
                  lambda m:'\\par\\noindent\\begin{minipage}{\\linewidth}\n'
                  +m[1]+m[2]+'\n\\end{minipage}\n',body,flags=re.S)
    body = re.sub(r'(\\textbf\{(?:算法\d+|Algorithm\s+\d+)[^\n]+\}\n\n)'
                  r'(\\begin\{verbatim\}.*?\\end\{verbatim\})',
                  lambda m:'\\par\\noindent\\begin{minipage}{\\linewidth}\n\\small\n'
                  +m[1]+m[2]+'\n\\end{minipage}\n',body,flags=re.S)
    for heading in ('References','参考文献','Reproducibility and Data','复现与数据说明'):
        body = body.replace('\\section{'+heading+'}','\\section*{'+heading+'}')
    if document=='graduation':
        body = body.replace('\\section{仿真平台与实验分析}',
                            '\\clearpage\n\\section{仿真平台与实验分析}')
    if document=='thesis':
        body = body.replace('\\subsection{适用边界与本章小结}',
                            '\\clearpage\n\\subsection{适用边界与本章小结}')
        body = body.replace('\\section{附录A 复现与材料索引}',
                            '\\clearpage\n\\section*{附录A 复现与材料索引}\n'
                            '\\addcontentsline{toc}{section}{附录A 复现与材料索引}')
        body = re.sub(r'\\subsection\{A\.(\d+) ([^}]+)\}',
                      r'\\subsection*{A.\1 \2}', body)
        body = body.replace('\\section{','\\clearpage\n\\section{')
        body = body.replace('\\section*{参考文献}',
                            '\\clearpage\n\\section*{参考文献}\n'
                            '\\addcontentsline{toc}{section}{参考文献}\n'
                            '\\begingroup\\renewcommand{\\labelenumi}{[\\theenumi]}')
        body += '\n\\endgroup\n'
        # Pandoc sets its own list labels after opening enumerate.
        prefix, bibliography = body.split('\\section*{参考文献}',1)
        bibliography = bibliography.replace('\\def\\labelenumi{\\arabic{enumi}.}',
                                              '\\def\\labelenumi{[\\arabic{enumi}]}')
        body = prefix+'\\section*{参考文献}'+bibliography
        body = body.replace('\\[','\\begin{equation}').replace('\\]','\\end{equation}')
    preamble = PREAMBLE.replace('面向预算受限设施建档的类别先验与几何反馈主动观测方法',inline(title))
    preamble = preamble.replace('\\begin{document}',
                               # Keep nested mathematical indices readable and
                               # use the already cached 7 pt math font family.
                               '\\DeclareMathSizes{10.95}{10.95}{8}{7}\n'
                               '\\clubpenalty=10000\n\\widowpenalty=10000\n'
                               '\\displaywidowpenalty=10000\n'
                               '\\newlength{\\ReviewTableWidth}\n\\begin{document}')
    if document=='english':
        # Reuse the complete cached font/encoding setup, with English names.
        # This also keeps mixed Unicode symbols readable without new bundles.
        preamble = preamble.replace('\\begin{document}',
            '\\renewcommand{\\abstractname}{Abstract}\n'
            '\\renewcommand{\\figurename}{Figure}\n'
            '\\renewcommand{\\tablename}{Table}\n\\begin{document}')
        preamble = preamble.replace('李兆宇','Zhaoyu Li').replace(
            '2026年9月28日\\quad 虚拟实验论文审阅稿','28 September 2026\\quad Review draft')
    elif document=='graduation':
        preamble = preamble.replace('虚拟实验论文审阅稿','毕业论文核心章节审阅稿')
    else:
        preamble = preamble.replace('虚拟实验论文审阅稿','毕业论文初稿（通用审阅版）')
        preamble = preamble.replace('\\begin{document}',
            '\\makeatletter\n'
            '\\def\\@seccntformat#1{\\ifcsname fmt@#1\\endcsname'
            '\\csname fmt@#1\\endcsname\\else\\csname the#1\\endcsname\\quad\\fi}\n'
            '\\def\\fmt@section{第\\thesection 章\\quad}\n\\makeatother\n'
            '\\setcounter{tocdepth}{2}\n\\numberwithin{equation}{section}\n'
            '\\begin{document}')
        preamble = preamble.replace('\\maketitle',
            '\\hypersetup{pageanchor=false}\n\\begin{titlepage}\n\\maketitle\\thispagestyle{empty}\n'
            '\\vfill\\begin{center}基于CPU虚拟实验的设施主动观测与建档\\\\[1em]\n'
            '五章正文及中英文摘要\\\\[1em]供导师审阅与后续学校模板整理'
            '\\end{center}\n\\end{titlepage}\n\\pagenumbering{Roman}\n'
            '\\hypersetup{pageanchor=true}')
    for path in (Path(__file__),ROOT/'scripts/build_virtual_paper_review_20260928.py'):
        sources[str(path.resolve().relative_to(ROOT))]=digest(path)
    return stem, source, preamble+abstract_tex+body+'\n\\end{document}\n', sources, figures, tables


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('document',choices=['graduation','thesis','english'])
    parser.add_argument('--render',action='store_true')
    parser.add_argument('--export',action='store_true')
    args=parser.parse_args()
    assert shutil.disk_usage(ROOT).free>100*1024**2
    stem,source,tex,sources,figures,tables=prepare(args.document)
    output=ROOT/f'tmp/writing-{args.document}_20260928'
    output.mkdir(parents=True,exist_ok=True)
    target=output/f'{stem}.tex';target.write_text(tex)
    (output/source.name).write_bytes(source.read_bytes())
    env=dict(os.environ)
    env.update(TECTONIC_CACHE_DIR=str(RUNTIME/'cache'),XDG_CACHE_HOME=str(RUNTIME/'cache'),
               XDG_CONFIG_HOME=str(RUNTIME/'config'),SOURCE_DATE_EPOCH='1790553600')
    command=[str(RUNTIME/'bin/tectonic'),'--only-cached','--keep-logs','-Z',
             f'search-path={ROOT}','--outdir',str(output),str(target)]
    result=subprocess.run(command,cwd=ROOT,env=env,text=True,stdout=subprocess.PIPE,
                          stderr=subprocess.STDOUT,timeout=180)
    (output/'build_stdout.txt').write_text(result.stdout)
    if result.returncode:raise RuntimeError(result.stdout[-7000:])
    assert all(digest(ROOT/p)==s for p,s in sources.items()),'Build inputs changed'
    log=(output/f'{stem}.log').read_text()
    diagnostics=[line for line in log.splitlines() if any(v in line for v in
                 ('Overfull','undefined','Missing character','LaTeX Error'))]
    pdf=output/f'{stem}.pdf'
    receipt=dict(document=args.document,source_sha256=sources,figures=figures,tables=tables,
                 pdf_sha256=digest(pdf),tex_sha256=digest(target),command=command,
                 compiler_sha256=digest(RUNTIME/'bin/tectonic'),
                 downloads_allowed=False,typesetting_diagnostics=diagnostics,
                 new_worlds=0,new_evaluations=0,status='review_draft')
    if args.render:
        # Small, verified copies keep PDF review independent of the experimental
        # Python environment whose large packages live in host shared memory.
        preview_packages=ROOT/'tmp/thesis-python_20260928/site-packages'
        if preview_packages.is_dir():sys.path.insert(0,str(preview_packages))
        receipt['pages']=render(pdf,output)
    (output/'build_receipt.json').write_text(json.dumps(receipt,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps({'document':args.document,'pages':receipt.get('pages'),
                      'figures':len(figures),'tables':len(tables),'diagnostics':diagnostics},ensure_ascii=False))
    if diagnostics:raise SystemExit('Fix typesetting before exporting')
    if args.export:
        shutil.copyfile(pdf,DOCS/pdf.name);shutil.copyfile(target,DOCS/target.name)
        shutil.copyfile(output/'build_receipt.json',DOCS/f'{stem}.build.json')


if __name__=='__main__':
    main()
