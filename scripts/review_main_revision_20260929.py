#!/usr/bin/env python3
"""Check the revised publication against its frozen predecessor; no experiments."""
import hashlib
import json
from pathlib import Path
import re
import zipfile

ROOT = Path(__file__).resolve().parents[1]
OLD = ROOT / 'docs/thesis/submission_20260929'
NEW = ROOT / 'docs/thesis/submission_20260929_revised'
SOURCE = ROOT / 'docs/thesis/ARTICLE_SUBMISSION_EN_REVISED_20260929.md'


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    checks = []

    def check(condition, description):
        if not condition:
            raise ValueError(description)
        checks.append(description)

    receipts = {}
    for label, directory in [('previous', OLD), ('revised', NEW)]:
        receipt = json.loads((directory / 'build_receipt.json').read_text())
        receipts[label] = receipt
        for name, expected in receipt['source_sha256'].items():
            check(sha(ROOT / name) == expected, f'{label} source unchanged: {name}')
        for name, expected in receipt['file_sha256'].items():
            check(sha(directory / name) == expected, f'{label} output verified: {name}')
        check(receipt['actual_pdflatex_validation'], f'{label} uses actual pdfLaTeX')
        check(not receipt['typesetting_diagnostics'], f'{label} has no typesetting errors')

    old, new = receipts['previous'], receipts['revised']
    table_mapping = {1: 1, 2: 2, 3: 3, 4: 4, 5: 6}
    for number, previous in table_mapping.items():
        a = next(t for t in new['tables'] if t['number'] == number)
        b = next(t for t in old['tables'] if t['number'] == previous)
        check(a['rows'] == b['rows'], f'Table {number} retains every cell of old Table {previous}')
    check(len(new['tables']) == 5 and len(new['figures']) == 8, 'five tables and eight figures')
    for number in range(1, 9):
        check(sha(NEW / f'Fig{number}.pdf') == sha(OLD / f'Fig{number}.pdf'),
              f'Figure {number} unchanged')
    old_refs = json.loads((OLD / 'references.json').read_text())
    new_refs = json.loads((NEW / 'references.json').read_text())
    refmap = lambda refs: {r['original_number']: r['source_text'] for r in refs}
    check(refmap(old_refs) == refmap(new_refs), 'all fourteen references retained')

    source = SOURCE.read_text()
    headings = re.findall(r'^## ([1-7]) ', source, re.M)
    check(headings == list('1234567'), 'seven consecutively numbered main sections')
    method = source.split('## 4 ', 1)[1].split('## 5 ', 1)[0]
    check(len(re.findall(r'^### 4\.', method, re.M)) == 5, 'five method subsections')
    check(source.count('**Algorithm 1.') == 1, 'one explicit execution algorithm')
    check('−0.6995%' in source and '32 fail at budget 30' in source,
          'larger-budget reversal and low-budget failures retained')
    check('all nine matched comparisons' in source and '36 runs' in source,
          'separate sharing result retained in Discussion')
    check('other three conditions retain the same trajectories' in source,
          'unchanged background-layout outcomes stated')
    check('13.10%' in source and 'In this condition' in source,
          'single-condition gain distinguished from aggregate')
    check('one win and three ties' not in source and 'one win and three losses' not in source,
          'results described using effect size and conditions')

    with zipfile.ZipFile(NEW.with_suffix('.zip')) as archive:
        files = {p.name for p in NEW.iterdir() if p.is_file()}
        check(set(archive.namelist()) == files, 'ZIP contains every publication file exactly once')
        for name in sorted(files):
            check(hashlib.sha256(archive.read(name)).hexdigest() == sha(NEW / name),
                  f'ZIP matches exported file: {name}')

    # Persisted previews permit review on a new device without the old tmp tree.
    pages_path = ROOT / 'docs/thesis/build_review_revision_20260929/pages.json'
    if not pages_path.is_file():
        pages_path = ROOT / 'tmp/final-journal_20260929_revised/prepared/visual/pages.json'
    pages = json.loads(pages_path.read_text())
    algorithm_pages = [p['page'] for p in pages if 'Algorithm 1.' in p['text']]
    check(len(algorithm_pages) == 1, 'Algorithm 1 appears once in rendered PDF')
    algorithm_text = pages[algorithm_pages[0] - 1]['text']
    check('Output: measured map' in algorithm_text, 'algorithm title and final line share a page')
    check(len(pages) == new['pages'], 'rendered page count matches build receipt')

    report = {
        'schema': 'publication.main_revision_review.v1',
        'source': str(SOURCE.relative_to(ROOT)), 'source_sha256': sha(SOURCE),
        'reviewer_script_sha256': sha(Path(__file__)),
        'pdf_sha256': sha(NEW / 'main.pdf'), 'pages': len(pages),
        'figures': len(new['figures']), 'tables': len(new['tables']),
        'algorithm_pages': algorithm_pages, 'passed': len(checks), 'checks': checks,
        'old_package_and_bound_inputs_unchanged': True,
        'new_scientific_runs': 0, 'new_metric_evaluations': 0,
        'scope': 'publication integrity and preserved reported data; not a new scientific validation',
    }
    out = ROOT / 'audit_results/main_revision_20260929'
    out.mkdir(parents=True, exist_ok=True)
    (out / 'review.json').write_text(json.dumps(report, indent=2, ensure_ascii=False) + '\n')
    print(json.dumps({k: report[k] for k in ['passed', 'pages', 'figures', 'tables', 'algorithm_pages']}))


if __name__ == '__main__':
    main()
