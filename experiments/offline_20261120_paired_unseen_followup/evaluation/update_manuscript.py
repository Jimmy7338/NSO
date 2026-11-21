"""Build the numerical follow-up subsection and supporting LaTeX table."""

import argparse
import json
from pathlib import Path

from paper_table import build_table

ROOT = Path(__file__).resolve().parents[1]


def read(path):
    return json.loads(path.read_text())


def value(d, key):
    return f'{d[key]:.3f}'


def make_calibration_table(metrics):
    methods = {'raw_visibility_m2': 'Visibility', 'raw_acquisition_m2': 'Acquisition',
               'scaled_acquisition_m2': 'Global scaling', 'corrected_acquisition_m2': 'Support scaling'}
    lines = [r'\begin{table}[tb]', r'\centering', r'\small',
             r'\caption{Four held-out single-object layouts and twelve executed short branches. Area errors and gains are layout means in $\mathrm{m}^2$. Ranking counts agreement among pairs with different measured gross gain. Gross and net columns report the measured area of the highest-ranked branch under each forecast.}',
             r'\label{tab:gain-holdout-followup}', r'\begin{tabular}{lrrrrr}', r'\toprule',
             r'Forecast & MAE & Bias & Order & Gross & Net \\', r'\midrule']
    for method, label in methods.items():
        row = metrics[method]
        fields = [label, value(row, 'mean_layout_mae_m2'), value(row, 'mean_layout_bias_m2'),
                  f"{row['concordant_pairs']}/{row['comparable_pairs']}",
                  value(row, 'mean_layout_actual_selected_gross_m2'), value(row, 'mean_layout_actual_selected_net_m2')]
        lines.append(' & '.join(fields) + r' \\')
    return '\n'.join([*lines, r'\bottomrule', r'\end{tabular}', r'\end{table}', ''])


def update(paper, outcomes, short, coefficients):
    (paper / 'b11_followup_policy_table.tex').write_text(build_table(outcomes))
    metrics = short['metrics']
    assert short['independent_layout_count'] == 4
    assert not short['failures'] and not short['partial_routes']
    assert all(record['measured'] == 12 for record in metrics.values())
    assert coefficients['status'] == 'calibration_coefficients_locked_before_heldout'
    (paper / 'b11_followup_gain_table.tex').write_text(make_calibration_table(metrics))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('paper', type=Path)
    args = parser.parse_args()
    update(args.paper, read(ROOT / 'evaluation/results/summary.json'),
           read(ROOT / 'gain_calibration/heldout_summary.json'),
           read(ROOT / 'gain_calibration/calibrated_coefficients.json'))
