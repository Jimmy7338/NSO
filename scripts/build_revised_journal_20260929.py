#!/usr/bin/env python3
"""Build the revised article without changing the frozen original package.

This publication-only wrapper reuses the existing journal builder. It changes
the default manuscript, author-review materials and output locations, and binds
its own source to the new receipt. It does not run scientific experiments.
All options supported by build_final_journal_20260929.py remain available.
"""
from __future__ import annotations

from pathlib import Path
import re
import sys

import build_final_journal_20260929 as journal


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / 'docs/thesis/ARTICLE_SUBMISSION_EN_REVISED_20260929.md'
MATERIALS = ROOT / 'docs/thesis/journal_revision_materials_20260929'
OUTPUT = ROOT / 'docs/thesis/submission_20260929_revised'
WORK = ROOT / 'tmp/final-journal_20260929_revised'


def keep_algorithm_together(tex: str) -> str:
    """Keep the one declared execution algorithm and its title on one page."""
    pattern = re.compile(
        r'(\\textbf\{Algorithm 1\. Category-conditioned receding-horizon '
        r'observation\.\}\s*\n\s*)'
        r'(\\begin\{verbatim\}\n.*?\\end\{verbatim\})', re.S)
    matches = list(pattern.finditer(tex))
    if len(matches) != 1:
        raise ValueError('Expected exactly one titled Algorithm 1 verbatim block')
    return pattern.sub(
        lambda match: '\\par\\medskip\\noindent\\begin{minipage}{\\linewidth}\n'
        '\\small\\hrule\\vspace{0.6em}\n'
        + match[1] + match[2]
        + '\n\\vspace{0.4em}\\hrule\n\\end{minipage}\\par\\medskip\n', tex)


def with_defaults(arguments: list[str]) -> list[str]:
    """Keep explicit caller paths; use separate revision paths otherwise."""
    result = list(arguments)
    for option, value in (('--source', SOURCE), ('--output', OUTPUT), ('--work', WORK)):
        if not any(item == option or item.startswith(option + '=') for item in result):
            result.extend((option, str(value)))
    return result


def main() -> None:
    original_prepare = journal.prepare
    journal.SOURCE = SOURCE
    journal.MATERIALS = MATERIALS

    def prepare(source: Path, destination: Path) -> dict:
        receipt = original_prepare(source, destination)
        manuscript = destination / 'main.tex'
        manuscript.write_text(keep_algorithm_together(manuscript.read_text()))
        wrapper = Path(__file__).resolve()
        receipt['source_sha256'][journal.source_key(wrapper)] = journal.digest(wrapper)
        receipt['revision'] = {
            'purpose': 'method exposition and evidence organization',
            'previous_package': 'docs/thesis/submission_20260929',
            'previous_package_modified': False,
            'new_scientific_experiments': 0,
            'algorithm_layout': 'one small-font minipage; verbatim content preserved',
        }
        journal.verify_sources(receipt['source_sha256'])
        return receipt

    journal.prepare = prepare
    sys.argv[1:] = with_defaults(sys.argv[1:])
    journal.main()


if __name__ == '__main__':
    main()
