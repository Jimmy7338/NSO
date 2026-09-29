#!/usr/bin/env python3
"""Verify the bounded paper/thesis delivery; never rerun a scientific experiment."""
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import zipfile

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'audit_results/final_delivery_20260929'


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    checks = []
    pins = {}

    def check(ok, label):
        checks.append({'check': label, 'passed': bool(ok)})

    def bound(path, expected):
        path = Path(path)
        check(path.is_file() and sha(path) == expected, str(path.relative_to(ROOT)))
        if path.is_file():
            pins[str(path.relative_to(ROOT))] = sha(path)

    def read(rel):
        path = ROOT / rel
        pins[rel] = sha(path)
        return json.loads(path.read_text())

    proc = subprocess.run([sys.executable, '-B', str(ROOT / 'scripts/verify_final_reproduction_index_20260929.py')],
                          cwd=ROOT, text=True, capture_output=True, check=True)
    reproduction = json.loads(proc.stdout)
    check(reproduction['status'] == 'passed' and not reproduction['failures'], 'reproduction index')
    (OUT / 'reproduction_verification.json').write_text(json.dumps(reproduction, indent=2) + '\n')

    science = read('audit_results/final_delivery_20260929/validation_review.json')
    check(science['status'] == 'passed', 'independent scientific review')
    check(science['final_goal_online_tasks_used'] == 8 and science['qualified_tasks'] == 8,
          'exactly eight attempted and qualified tasks')
    check((science['wins'], science['ties'], science['losses']) == (1, 3, 0), 'complete paired outcomes')

    builds = {}
    for stem, pages, figs, tables in [
        ('ARTICLE_SUBMISSION_EN_20260928', 19, 8, 6),
        ('VIRTUAL_PAPER_EN_20260928', 36, 22, 9),
        ('GRADUATION_THESIS_20260928', 57, 25, 18),
    ]:
        receipt = read('docs/thesis/' + stem + '.build.json')
        for rel, digest in receipt['source_sha256'].items():
            bound(ROOT / rel, digest)
        bound(ROOT / 'docs/thesis' / (stem + '.pdf'), receipt['pdf_sha256'])
        bound(ROOT / 'docs/thesis' / (stem + '.tex'), receipt['tex_sha256'])
        check(not receipt['typesetting_diagnostics'], stem + ' typesetting')
        check((receipt['pages'], len(receipt['figures']), len(receipt['tables'])) == (pages, figs, tables),
              stem + ' page/figure/table counts')
        builds[stem] = {'pages': pages, 'figures': figs, 'tables': tables}

    folder = ROOT / 'docs/thesis/submission_20260929'
    receipt = read('docs/thesis/submission_20260929/build_receipt.json')
    for rel, digest in receipt['source_sha256'].items():
        bound(ROOT / rel, digest)
    for rel, digest in receipt['file_sha256'].items():
        bound(folder / rel, digest)
    bound(folder / 'main.pdf', receipt['pdf_sha256'])
    check(not receipt['typesetting_diagnostics'] and receipt['actual_pdflatex_validation'], 'journal pdflatex')
    check(receipt['pages'] == 20 and receipt['reference_count'] == 14, 'journal pages/references')
    check(150 <= receipt['abstract_words'] <= 250 and 4 <= receipt['keyword_count'] <= 6,
          'journal abstract and keyword limits')
    for name in ('cover_letter_draft.md', 'submission_checklist.md'):
        check((folder / name).is_file(), name)
    archive = ROOT / 'docs/thesis/submission_20260929.zip'
    with zipfile.ZipFile(archive) as bundle:
        for item in bundle.infolist():
            if item.is_dir():
                continue
            candidate = folder / item.filename
            if not candidate.is_file():
                candidate = folder / Path(item.filename).name
            check(candidate.is_file() and hashlib.sha256(bundle.read(item)).hexdigest() == sha(candidate),
                  'submission archive: ' + item.filename)
    pins[str(archive.relative_to(ROOT))] = sha(archive)

    folder = ROOT / 'docs/thesis/defense_20260929'
    receipt = read('docs/thesis/defense_20260929/manifest.json')
    for item in receipt['inputs']:
        bound(ROOT / item['path'], item['sha256'])
    for item in receipt['outputs']:
        bound(folder / item['path'], item['sha256'])
    bound(ROOT / 'scripts/build_final_defense_20260929.py', receipt['builder_sha256'])
    check(receipt['office_rendered'] is False, 'Office rendering limit disclosed')

    for rel in [
        'docs/research/FINAL_CONTRIBUTION_EVIDENCE_MATRIX_20260929.md',
        'docs/research/FINAL_CONTENT_REVIEW_20260929.md',
        'docs/research/FINAL_METHOD_REPRODUCTION_INDEX_20260929.md',
        'docs/research/FINAL_JOURNAL_FIT_20260929.md',
        'docs/research/FINAL_DELIVERY_20260929.md',
    ]:
        path = ROOT / rel
        check(path.is_file(), rel)
        if path.is_file():
            pins[rel] = sha(path)
    pins[str(Path(__file__).resolve().relative_to(ROOT))] = sha(Path(__file__).resolve())
    failures = [c for c in checks if not c['passed']]
    result = dict(schema='final.paper_thesis.delivery_verification.v1',
                  status='passed' if not failures else 'failed', checks_total=len(checks),
                  checks_passed=len(checks) - len(failures), failures=failures,
                  reproduction_checks_passed=reproduction['checks_passed'],
                  reviewed_documents=builds, new_online_tasks=0, new_worlds=0,
                  new_tsdf_integrations=0, new_surface_evaluations=0,
                  source_and_artifact_sha256=pins,
                  external_author_confirmation_required=True,
                  school_format='generic_per_user_blank_metadata',
                  scientific_tasks_complete=8, reserved_tasks_used=0)
    (OUT / 'delivery_verification.json').write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps({k: v for k, v in result.items() if k != 'source_and_artifact_sha256'}, indent=2))
    if failures:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
