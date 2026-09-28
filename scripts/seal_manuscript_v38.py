#!/usr/bin/env python3
"""Verify and seal manuscript artifacts without calling research execution code."""
import csv
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import zipfile

os.environ.setdefault('PYTHONDONTWRITEBYTECODE','1')
import pypdfium2 as pdfium

ROOT=Path(__file__).resolve().parents[1]
OUTPUT=ROOT/'audit_results/v38_manuscript_release_20260920'
BUILD=ROOT/'tmp/v38-tex/manuscript_release'
DELIVERY=ROOT/'docs/thesis/NSO_Manuscript_V38_20260920.pdf'

def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()
def read(p): return json.loads(p.read_text())
def write(p,d): p.write_text(json.dumps(d,ensure_ascii=False,indent=2)+'\n')

def main():
    assert not OUTPUT.exists(),'Do not overwrite a released manuscript evidence pack'
    assert shutil.disk_usage(ROOT).free>70*1024**2
    checked={}
    def verify(relative,expected):
        assert sha(ROOT/relative)==expected,'Changed input: '+relative
        checked[relative]=expected
    figdir=ROOT/'docs/thesis/figures/v38'
    figures=read(figdir/'manifest.json')
    for relative,expected in figures['source_sha256'].items(): verify(relative,expected)
    for name,expected in figures['outputs'].items(): verify(str((figdir/name).relative_to(ROOT)),expected)
    qualitative=read(ROOT/'docs/thesis/figures/v38_qualitative/provenance.json')
    for relative,record in qualitative['sources'].items(): verify(relative,record['sha256'])
    with (figdir/'online_measurements.csv').open() as f: rows=list(csv.DictReader(f))
    assert len(rows)==48
    for row in rows:
        assert abs(float(row['joint'])-float(row['C_map'])*float(row['f1']))<1e-13
    paper=ROOT/'Semantic_Enhanced_Active_SLAM_Paper.tex'; tex=paper.read_text()
    cites={k for group in re.findall(r'\\cite\{([^}]+)\}',tex) for k in group.split(',')}
    bib=set(re.findall(r'\\bibitem\{([^}]+)\}',tex)); assert cites==bib
    labels=re.findall(r'\\label\{([^}]+)\}',tex); assert len(labels)==len(set(labels))
    refs=set(re.findall(r'\\(?:eqref|ref)\{([^}]+)\}',tex)); assert refs<=set(labels)
    for path in re.findall(r'\\input\{([^}]+)\}',tex): assert (ROOT/path).is_file()
    assert len(re.findall(r'\\begin\{figure\}',tex))==6
    build=read(BUILD/'receipt.json')
    assert build['exit_code']==0 and not build['downloads_allowed'] and build['stop_reason'] is None
    assert build['manuscript_sha256']==sha(paper)
    log=(BUILD/'Semantic_Enhanced_Active_SLAM_Paper.log').read_text()
    prohibited=('Overfull','Underfull','undefined','Missing character','LaTeX Error','Package xeCJK Warning')
    assert not [word for word in prohibited if word in log]
    pdfpath=BUILD/'Semantic_Enhanced_Active_SLAM_Paper.pdf'
    pdf=pdfium.PdfDocument(pdfpath); pages=[]
    for i in range(len(pdf)):
        page=pdf[i]; text=page.get_textpage().get_text_bounded()
        assert '??' not in text and '\ufffd' not in text
        pages.append({'page':i+1,'text_characters':len(text),'text_sha256':hashlib.sha256(text.encode()).hexdigest()})
    assert all(p['text_characters']>150 for p in pages),'Unexpected almost-empty page'
    assert len(pdf)==14, 'Update explicit page review if layout changes'
    assert not DELIVERY.exists()
    shutil.copyfile(pdfpath,DELIVERY)
    OUTPUT.mkdir()
    shutil.copyfile(BUILD/'receipt.json',OUTPUT/'build_receipt.json')
    shutil.copyfile(BUILD/'stdout.txt',OUTPUT/'build_stdout.txt')
    shutil.copyfile(BUILD/'Semantic_Enhanced_Active_SLAM_Paper.log',OUTPUT/'build_log.txt')
    review=ROOT/'tmp/v38-tex/final_visual'
    layout=read(review/'final_layout_comparison.json')
    assert layout['final_pdf_sha256']==sha(DELIVERY)
    assert len(layout['first_13_renders_identical'])==13 and all(layout['first_13_renders_identical'])
    (OUTPUT/'visual_review').mkdir()
    for name in ('contact_01.png','contact_07.png','check_page_11.png','final_page14.png','final_layout_comparison.json'):
        shutil.copyfile(review/name,OUTPUT/'visual_review'/name)
    attempts=[]
    for name in ('manuscript','manuscript_attempt02','manuscript_attempt03','manuscript_attempt04','manuscript_attempt06','manuscript_final'):
        folder=ROOT/'tmp/v38-tex'/name
        if (folder/'receipt.json').exists():
            dest=OUTPUT/'compilation_history'/name; dest.mkdir(parents=True)
            for fn in ('receipt.json','stdout.txt'): shutil.copyfile(folder/fn,dest/fn)
            attempts.append(read(folder/'receipt.json'))
    # Snapshot mutable writing documents: old experiment source files are only hashed.
    mutable=[paper,*sorted((ROOT/'docs/thesis').glob('V38_*')),
             ROOT/'docs/thesis/FIGURE_GUIDE_V38_20260920.md',
             ROOT/'docs/research/V38_EXTERNAL_BASELINE_PROTOCOL_20260920.md',
             *[ROOT/'docs/research'/s for s in ('CURRENT_RESEARCH_STATE.json','GOAL_PROMPT.md','RESEARCH_GOAL_AND_STORY.md','THESIS_CLAIM_EVIDENCE_LEDGER.md')],
             *[ROOT/'scripts'/s for s in ('build_manuscript_v38.py','build_thesis_figures_v38.py','build_thesis_qualitative_v38.py','seal_manuscript_v38.py')]]
    with zipfile.ZipFile(OUTPUT/'writing_sources.zip','x',compression=zipfile.ZIP_DEFLATED) as z:
        for p in mutable:
            checked[str(p.relative_to(ROOT))]=sha(p); z.write(p,str(p.relative_to(ROOT)))
    checked[str(DELIVERY.relative_to(ROOT))]=sha(DELIVERY)
    # PDF and each saved output bind the exact reviewable state, not just the script.
    for folder in (figdir,ROOT/'docs/thesis/figures/v38_qualitative'):
        for p in folder.iterdir():
            if p.is_file(): checked[str(p.relative_to(ROOT))]=sha(p)
    write(OUTPUT/'source_and_artifact_sha256.json',checked)
    result={'status':'manuscript_and_six_figures_complete_external_experiments_not_executed',
            'paper':str(DELIVERY.relative_to(ROOT)),'paper_sha256':sha(DELIVERY),
            'tex_sha256':sha(paper),'pages':len(pdf),'page_checks':pages,
            'figures':6,'online_csv_rows':48,'joint_component_checks':48,
            'defined_and_resolved_citation_keys':sorted(cites),'unique_labels':len(labels),
            'tex_log_prohibited_diagnostics':[],
            'remaining_build_notice':'Tectonic warns that absolute system font paths require matching fonts on another machine; font manifest supplied',
            'independent_review':'Three source/statistics/literature audits plus sampled full-page review; all-page contacts and repaired timeline checked by root; final pages 1-13 pixel-identical to reviewed layout, new page14 visually checked',
            'compilation_history_count':len(attempts),
            'precondition_stop':'Attempt05 did not start compiler because free space was below 76 MiB; only regenerable pyc was then removed',
            'scope':'Manuscript, figure rendering and saved-evidence verification only',
            'new_worlds':0,'new_online_planning_runs':0,'new_TSDF_fusions':0,'new_quality_evaluations':0,
            'historical_main_used':35,'historical_main_limit':36,
            'external_baselines_executed':False,'natural_equipment_efficacy_deferred':True,
            'free_bytes_after':shutil.disk_usage(ROOT).free}
    write(OUTPUT/'result.json',result)
    inventory={str(p.relative_to(OUTPUT)):sha(p) for p in sorted(OUTPUT.rglob('*')) if p.is_file()}
    write(OUTPUT/'artifact_hashes.json',inventory)
    assert sum(p.stat().st_size for p in OUTPUT.rglob('*') if p.is_file())<4*1024**2
    assert shutil.disk_usage(ROOT).free>64*1024**2
    print(json.dumps(result,ensure_ascii=False,indent=2))

if __name__=='__main__': main()
