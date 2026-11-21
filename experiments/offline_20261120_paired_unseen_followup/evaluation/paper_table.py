"""Export the final matched layout table from independently evaluated numeric rows."""

import argparse
import json
from pathlib import Path


def fmt_action(value):
    return '--' if value is None else str(value)


def build_table(summary):
    assert not summary['missing_or_partial'] and not summary['integrity_errors']
    names = {'chain_a': 'Chain A', 'cycle_a': 'Cycle A', 'chain_b': 'Chain B', 'cycle_b': 'Cycle B'}
    arms = {'F_nearest': 'Frontier*', 'F_strict': 'Frontier', 'G_visibility': 'G', 'S_visibility': 'S'}
    lines = []
    for layout, label in names.items():
        for arm, arm_label in arms.items():
            (row,) = summary['rows'][layout][arm]
            fields = [label, arm_label, f"{100*row['coverage']['free_correct_fraction']:.2f}",
                      fmt_action(row['A95']), fmt_action(row['A100']),
                      f"{100*row['object_weighted_recall_5cm']:.2f}",
                      f"{100*row['surface_metrics']['f1_05cm']:.2f}",
                      str(row['paid_actions']), 'Yes' if row['returned'] else 'No']
            lines.append(' & '.join(fields) + r' \\')
            label = ''
        lines.append(r'\addlinespace')
    del lines[-1]
    return '\n'.join([
        r'\begin{table}[tb]', r'\centering', r'\small',
        r'\caption{Paired exploration outcomes. Frontier* is the original implementation with an unconditional current-pose candidate. Frontier restricts candidates to reachable observed-free cells adjacent to unknown cells. G and S use the same visibility forecast and differ in the structural class prior. Coverage is the fraction of reachable free cells correctly mapped as free. A95 and A100 are first actions reaching 95\% and 100\%, with dashes for thresholds unreached by the recorded stop. Object recall is weighted by reference surface area. Recall and scene F1 use a 5\,cm tolerance. The 256-action cap includes return.}',
        r'\label{tab:matched-frontier-prior}', r'\begin{tabular}{llrrrrrrc}', r'\toprule',
        r'Layout & Method & Cov. (\%) & A95 & A100 & Obj. (\%) & F1 (\%) & Actions & Return \\',
        r'\midrule', *lines, r'\bottomrule', r'\end{tabular}', r'\end{table}', '',
    ])


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('summary', type=Path)
    parser.add_argument('table', type=Path)
    args = parser.parse_args()
    args.table.write_text(build_table(json.loads(args.summary.read_text())))
