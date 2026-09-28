#!/usr/bin/env python3
"""Seal the V39 manuscript after a bound PDF visual review; no experiments."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import re
import shutil
import zipfile

ROOT=Path(__file__).resolve().parents[1]
BUILD=ROOT/'tmp/v38-tex/manuscript_v39'
REVIEW=ROOT/'tmp/v38-tex/v39_visual'
OUTPUT=ROOT/'audit_results/v39_manuscript_release_20260920'
DELIVERY=ROOT/'docs/thesis/NSO_Manuscript_V39_20260920.pdf'
PDF=BUILD/'Semantic_Enhanced_Active_SLAM_Paper.pdf'


def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def read(path):return json.loads(path.read_text())
def write(path,value):path.write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n')


def render_review():
    import pypdfium2 as pdfium
    from PIL import Image,ImageDraw
    REVIEW.mkdir(exist_ok=False)
    doc=pdfium.PdfDocument(PDF)
    pages=[]
    for i in range(len(doc)):
        page=doc[i]
        text=page.get_textpage().get_text_bounded()
        pages.append(dict(page=i+1,text=text,text_sha256=hashlib.sha256(text.encode()).hexdigest()))
        img=page.render(scale=1.).to_pil()
        img.save(REVIEW/f'page_{i+1:02d}.png')
    for start in range(0,len(doc),6):
        canvas=Image.new('RGB',(1050,1500),'#eeeeee');draw=ImageDraw.Draw(canvas)
        for slot,i in enumerate(range(start,min(start+6,len(doc)))):
            img=Image.open(REVIEW/f'page_{i+1:02d}.png');img.thumbnail((510,470))
            x=(slot%2)*525+(525-img.width)//2;y=(slot//2)*500+22
            canvas.paste(img,(x,y));draw.text(((slot%2)*525+12,(slot//2)*500+4),f'Page {i+1}',fill='black')
        canvas.save(REVIEW/f'contact_{start+1:02d}.png')
    write(REVIEW/'pages.json',dict(pdf_sha256=sha(PDF),pages=pages))
    print(json.dumps(dict(pages=len(doc),review=str(REVIEW),pdf_sha256=sha(PDF))))


def release():
    assert not OUTPUT.exists() and not DELIVERY.exists()
    assert shutil.disk_usage(ROOT).free>80*1024**2
    paper=ROOT/'Semantic_Enhanced_Active_SLAM_Paper.tex';tex=paper.read_text()
    build=read(BUILD/'receipt.json')
    assert build['exit_code']==0 and not build['downloads_allowed'] and build['stop_reason'] is None
    assert build['manuscript_sha256']==sha(paper)
    log=(BUILD/'Semantic_Enhanced_Active_SLAM_Paper.log').read_text()
    bad=[w for w in ('Overfull','Underfull','undefined','Missing character','LaTeX Error','Package xeCJK Warning') if w in log]
    assert not bad,bad
    cites={k for group in re.findall(r'\\cite\{([^}]+)\}',tex) for k in group.split(',')}
    assert cites==set(re.findall(r'\\bibitem\{([^}]+)\}',tex))
    labels=re.findall(r'\\label\{([^}]+)\}',tex)
    assert len(labels)==len(set(labels))
    assert set(re.findall(r'\\(?:eqref|ref)\{([^}]+)\}',tex))<=set(labels)
    assert 'V39_TARE_RESULTS_INSERTION' not in tex
    for path in re.findall(r'\\input\{([^}]+)\}',tex):assert (ROOT/path).is_file()
    page_info=read(REVIEW/'pages.json');visual=read(REVIEW/'review.json')
    assert page_info['pdf_sha256']==visual['pdf_sha256']==sha(PDF)
    assert visual['all_pages_reviewed'] and visual['passed']
    assert all('??' not in p['text'] and '\ufffd' not in p['text'] and len(p['text'])>100 for p in page_info['pages'])
    checked={}
    def bind(path):checked[str(path.relative_to(ROOT))]=sha(path)
    # Verify source and output manifests made by the independent chart audit.
    for dirname in ('v39_external','v39_tare'):
        folder=ROOT/'docs/thesis/figures'/dirname
        manifest=read(folder/'manifest.json')
        for key in ('source_sha256','input_sha256','outputs_sha256',
                'output_sha256','report_sha256','script_sha256'):
            for rel,expected in manifest.get(key,{}).items():assert sha(ROOT/rel)==expected
        for p in folder.rglob('*'):
            if p.is_file():bind(p)
    for dirname in ('v38','v38_qualitative'):
        for p in (ROOT/'docs/thesis/figures'/dirname).rglob('*'):
            if p.is_file():bind(p)
    tabledir=ROOT/'docs/thesis/tables/v39'
    tables=read(tabledir/'manifest.json')
    for rel,expected in tables['source_sha256'].items():assert sha(ROOT/rel)==expected
    for name,expected in tables['outputs'].items():assert sha(tabledir/name)==expected
    for p in tabledir.iterdir():
        if p.is_file():bind(p)
    # Experimental directories were independently sealed; verify each inventory.
    for dirname in ('v39_external_cpu_20260920','v39_tare_adapter_r1_20260920'):
        folder=ROOT/'audit_results'/dirname
        for rel,expected in read(folder/'final_seal.json').items():
            assert sha(folder/rel)==expected
        bind(folder/'result.json');bind(folder/'final_seal.json')
    core=read(ROOT/'audit_results/v39_external_cpu_20260920/result.json')
    tare=read(ROOT/'audit_results/v39_tare_adapter_r1_20260920/result.json')
    assert len(core['rows'])==16 and len(tare['rows'])==4
    for row in core['rows']+tare['rows']:
        for cm in (2,5,10):assert abs(row[f'J{cm}']-row['C_map']*row[f'Q{cm}'])<1e-13
    shutil.copyfile(PDF,DELIVERY);OUTPUT.mkdir();bind(DELIVERY)
    for source,name in ((BUILD/'receipt.json','build_receipt.json'),(BUILD/'stdout.txt','build_stdout.txt'),
            (BUILD/'Semantic_Enhanced_Active_SLAM_Paper.log','build_log.txt')):
        shutil.copyfile(source,OUTPUT/name)
    # Preserve earlier layout attempts; they made no experiment or metric calls.
    history=OUTPUT/'layout_attempts';history.mkdir()
    for number in (1,2):
        earlier=BUILD.parent/f'manuscript_v39_attempt{number:02d}'
        for name in ('receipt.json','stdout.txt','Semantic_Enhanced_Active_SLAM_Paper.log'):
            if (earlier/name).is_file():
                shutil.copyfile(earlier/name,history/f'attempt{number:02d}_{name}')
    (OUTPUT/'visual_review').mkdir()
    for p in [*sorted(REVIEW.glob('contact_*.png')),REVIEW/'review.json']:
        shutil.copyfile(p,OUTPUT/'visual_review'/p.name)
    mutable=[paper,Path(__file__),ROOT/'scripts/build_manuscript_v38.py',
        ROOT/'scripts/build_external_comparison_v39.py',ROOT/'scripts/summarize_tare_transfer_v39.py',
        ROOT/'scripts/build_manuscript_tables_v39.py',
        *sorted((ROOT/'docs/research').glob('V39_*.md')),
        *sorted((ROOT/'docs/thesis').glob('V39_*.md')),
        *[ROOT/'docs/research'/p for p in ('CURRENT_RESEARCH_STATE.json','GOAL_PROMPT.md',
            'RESEARCH_GOAL_AND_STORY.md','THESIS_CLAIM_EVIDENCE_LEDGER.md')]]
    with zipfile.ZipFile(OUTPUT/'writing_sources.zip','x',compression=zipfile.ZIP_DEFLATED) as z:
        for p in mutable:bind(p);z.write(p,str(p.relative_to(ROOT)))
    write(OUTPUT/'source_and_artifact_sha256.json',checked)
    result=dict(status='complete',paper=str(DELIVERY.relative_to(ROOT)),paper_sha256=sha(DELIVERY),
        tex_sha256=sha(paper),pages=len(page_info['pages']),figures=len(re.findall(r'\\begin\{figure\}',tex)),
        citations=sorted(cites),labels=len(labels),tex_prohibited_diagnostics=bad,
        joint_component_checks=60,L1_main=16,L1_full_policy_replays_passed=16,
        TARE_actual_main=4,TARE_fixed_action_replays_passed=4,TARE_replay_attempts_including_comparator_failure=5,
        TARE_policy_replayed=False,old_v36_main_allocation_unchanged='35/36',
        publication_builder_new_worlds=0,natural_equipment_efficacy_deferred=True,
        no_original_SWAP_or_VISTA_SOTA_claim=True,
        free_bytes_after=shutil.disk_usage(ROOT).free)
    write(OUTPUT/'result.json',result)
    write(OUTPUT/'artifact_hashes.json',{str(p.relative_to(OUTPUT)):sha(p) for p in sorted(OUTPUT.rglob('*')) if p.is_file()})
    print(json.dumps(result,ensure_ascii=False,indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('command',choices=('render-review','release'))
    args=p.parse_args()
    render_review() if args.command=='render-review' else release()
