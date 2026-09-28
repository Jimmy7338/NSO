#!/usr/bin/env python3
"""Plot the fixed planar sensor probe; seed ranges are not confidence intervals."""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


MODELS = [
    ('ideal', 'Ideal depth', '#67737C', 'o'),
    ('iid_010px', 'Random disparity: 0.10 px', '#3975B8', '^'),
    ('iid_025px', 'Random disparity: 0.25 px', '#24877F', 's'),
    ('bias_025px_iid_025px', 'Random 0.25 px + fixed bias 0.25 px', '#C7663E', 'D'),
]
PLANS = ['far_one', 'far_copied_24', 'far_fresh_24', 'near_one', 'near_fresh_24']
PLAN_LABELS = ['4 m\n1 frame', '4 m\n24 copies', '4 m\n24 fresh', '1 m\n1 frame', '1 m\n24 fresh']


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    summary = json.loads((args.source / 'summary.json').read_text())
    measurements = json.loads((args.source / 'measurements.json').read_text())
    if len(summary) != 20 or len(measurements) != 60:
        raise ValueError('The complete 4-model, 5-condition, 3-seed probe is required.')
    indexed = {(row['model'], row['plan']): row for row in summary}
    if len(indexed) != 20:
        raise ValueError('Duplicate summary conditions.')
    args.output.mkdir(parents=True, exist_ok=False)
    fig, axes = plt.subplots(1, 2, figsize=(12.8, 5.8))
    for j, (model, label, color, marker) in enumerate(MODELS):
        error_values, f1_values = [], []
        error_ranges, f1_ranges = [], []
        for plan in PLANS:
            selected = [row for row in measurements if row['model'] == model and row['plan'] == plan]
            if sorted(row['seed'] for row in selected) != [1901, 1902, 1903]:
                raise ValueError(f'Missing or duplicate seeds: {model}/{plan}')
            err = np.array([row['metrics']['mean_abs_error_m'] * 1000 for row in selected])
            f1 = np.array([row['metrics']['05cm']['f1'] for row in selected])
            entry = indexed[model, plan]
            if not np.isclose(err.mean(), entry['mean_abs_error_mm']) or not np.isclose(f1.mean(), entry['f1_05cm']):
                raise ValueError(f'Summary differs from measurements: {model}/{plan}')
            error_values.append(entry['mean_abs_error_mm'])
            f1_values.append(entry['f1_05cm'])
            error_ranges.append([err.mean() - err.min(), err.max() - err.mean()])
            f1_ranges.append([f1.mean() - f1.min(), f1.max() - f1.mean()])
        # A small horizontal offset keeps coincident F1=1 markers and ranges visible.
        x = np.arange(len(PLANS)) + (j - 1.5) * .045
        for ax, values, ranges in zip(axes, [error_values, f1_values], [error_ranges, f1_ranges]):
            ax.errorbar(x, values, yerr=np.asarray(ranges).T, label=label, color=color,
                        marker=marker, markersize=5, linewidth=1.6, capsize=3,
                        linestyle='--' if model == 'ideal' else '-')
    axes[0].set_yscale('symlog', linthresh=1, linscale=1)
    axes[0].set_yticks([0, 1, 10, 100], ['0', '1', '10', '100'])
    axes[0].set_ylim(-.08, 120)
    axes[0].set_ylabel('Mean absolute surface error (mm)')
    axes[0].set_title('Exterior error: linear below 1 mm, log above', fontsize=10)
    axes[1].set_ylabel('Exterior F1 @ 5 cm')
    axes[1].set_ylim(0, 1.07)
    axes[1].set_title('A 5 cm threshold can already be saturated', fontsize=10)
    for ax in axes:
        ax.set_xticks(range(len(PLANS)), PLAN_LABELS, fontsize=9)
        ax.set_xlim(-.3, 4.3)
        ax.grid(axis='y', alpha=.2)
        ax.spines[['top', 'right']].set_visible(False)
        ax.axvline(2.5, color='#B7BEC4', linewidth=.8, linestyle=':')
    handles, labels = axes[0].get_legend_handles_labels()
    fig.suptitle('TSDF observation response on a planar exterior', y=.985, fontsize=14)
    fig.text(.5, .935, 'Generic stereo surrogate; perfect pose; no ZED calibration or planning-policy result',
             ha='center', fontsize=10, color='#46515B')
    fig.legend(handles, labels, ncol=2, loc='upper center', bbox_to_anchor=(.5, .905),
               frameon=False, fontsize=9)
    fig.text(.5, .055, 'Points: means over 3 seeds. Error bars: seed min–max, not confidence intervals.',
             ha='center', fontsize=9)
    fig.text(.5, .023, 'Copies repeat the same depth frame; fresh frames use independent random noise. Travel cost is excluded.',
             ha='center', fontsize=9)
    fig.tight_layout(rect=[0, .09, 1, .80])
    fig.savefig(args.output / 'response.png', dpi=170, bbox_inches='tight')
    fig.savefig(args.output / 'response.svg', bbox_inches='tight')
    plt.close(fig)


if __name__ == '__main__':
    main()
