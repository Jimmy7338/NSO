#!/usr/bin/env python3
"""One paired B-to-B2 result figure; refuses incomplete evaluation summaries."""
from pathlib import Path
import argparse
import csv
import hashlib
import json

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
BASELINE = ROOT.parent / 'dual_template_exploration_20261119'
ARMS = ('C', 'J', 'G', 'S', 'S_no_fit', 'S_no_history')
LAYOUTS = ('shelf_left', 'shelf_right')


def read(p): return json.loads(p.read_text())
def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()


def final_rows(path):
    obj = read(path)
    if obj['status'] not in ('completed_independent_12_slot_evaluation', 'completed_with_failures'):
        raise ValueError('Final independent evaluation is not complete')
    if set(obj['rows']) != set(LAYOUTS) or any(set(obj['rows'][k]) != set(ARMS) for k in LAYOUTS):
        raise ValueError('All twelve declared slots, including failures, must be present')
    return obj


def build_rows(old, new):
    rows = []
    for arm in ARMS:
        for layout in LAYOUTS:
            a, b = old['rows'][layout][arm], new['rows'][layout][arm]
            complete = all(r.get('status') == 'completed_independently_evaluated' for r in (a, b))
            row = dict(layout=layout, arm=arm, B_status=a['status'], B2_status=b['status'],
                       B_F1_5cm=None, B2_F1_5cm=None, delta_F1_percentage_points=None,
                       B_full_coverage_action=None, B2_full_coverage_action=None,
                       delta_full_coverage_actions=None, B_paid_actions=None, B2_paid_actions=None)
            if complete:
                assert a['reference_sha256'] == b['reference_sha256'], (layout, arm, 'reference changed')
                for prefix, value in [('B', a), ('B2', b)]:
                    row[prefix + '_F1_5cm'] = value['surface_metrics']['f1_05cm']
                    row[prefix + '_full_coverage_action'] = value['first_coverage_action']['1.0']
                    row[prefix + '_paid_actions'] = value['paid_actions']
                    assert 0 <= row[prefix + '_F1_5cm'] <= 1
                row['delta_F1_percentage_points'] = 100 * (row['B2_F1_5cm'] - row['B_F1_5cm'])
                if row['B_full_coverage_action'] is not None and row['B2_full_coverage_action'] is not None:
                    row['delta_full_coverage_actions'] = row['B2_full_coverage_action'] - row['B_full_coverage_action']
            rows.append(row)
    return rows


def plot(rows, output):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from matplotlib.ticker import MaxNLocator
    import numpy as np
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 9, 'axes.labelsize': 9,
                         'pdf.fonttype': 42, 'ps.fonttype': 42, 'svg.fonttype': 'none',
                         'axes.spines.top': False, 'axes.spines.right': False})
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.35), sharey=True)
    names = ['C', 'J', 'G', 'S', r'$S_{\mathrm{no\ fit}}$', r'$S_{\mathrm{no\ history}}$']
    colors = ['#0072B2', '#D55E00']; markers = ['o', 's']; offsets = [-.17, .17]
    metrics = [('delta_F1_percentage_points', '$\\Delta F_1$ at 5 cm (percentage points)\nHigher is better', .1, 3),
               ('delta_full_coverage_actions', '$\\Delta$ first full-coverage actions\nLower is better', 1., 0)]
    for panel, (ax, (metric, xlabel, minimum, decimals)) in enumerate(zip(axes, metrics)):
        available = [r[metric] for r in rows if r[metric] is not None]
        limit = max([minimum] + [abs(x) for x in available]) * 1.5
        ax.axvline(0, color='#777777', lw=.7, zorder=1)
        for j, layout in enumerate(LAYOUTS):
            group = [next(r for r in rows if r['layout'] == layout and r['arm'] == arm) for arm in ARMS]
            for i, row in enumerate(group):
                value, y = row[metric], i + offsets[j]
                if value is None:
                    ax.text(0, y, 'not available', color=colors[j], va='center', ha='left', fontsize=7)
                    continue
                if abs(value) < 1e-12: value = 0.
                ax.plot([0, value], [y, y], color=colors[j], lw=1.1, alpha=.75, zorder=2)
                ax.scatter([value], [y], color=colors[j], marker=markers[j], s=20, zorder=3)
                text = f'{value:+.{decimals}f}' if value else ('0' if not decimals else f'{0:.{decimals}f}')
                dx = limit * .04 * (1 if value >= 0 else -1)
                ax.text(value + dx, y, text, ha='left' if value >= 0 else 'right', va='center', fontsize=7.5, color=colors[j])
        ax.set_xlim(-limit, limit); ax.set_ylim(len(ARMS)-.5, -.5)
        ax.set_yticks(np.arange(len(ARMS)), names)
        ax.set_xlabel(xlabel); ax.tick_params(axis='both', length=2.5)
        ax.xaxis.set_major_locator(MaxNLocator(nbins=5, integer=metric.endswith('actions')))
        ax.grid(axis='x', color='#E8E8E8', lw=.5, zorder=0)
        ax.text(.01, 1.02, f'({chr(97+panel)})', transform=ax.transAxes, fontsize=9)
    handles = [Line2D([0], [0], marker=m, color=c, lw=1, markersize=4,
                      label=label) for m,c,label in zip(markers,colors,['Shelf left', 'Shelf right'])]
    fig.legend(handles=handles, loc='upper center', bbox_to_anchor=(.54, 1.015), frameon=False, ncol=2)
    fig.tight_layout(rect=(0, 0, 1, .96), w_pad=2.0)
    output.mkdir(parents=True, exist_ok=True)
    for ext in ('pdf', 'svg', 'png'):
        fig.savefig(output / ('B_to_B2_delta.' + ext), dpi=300, bbox_inches='tight', facecolor='white')
    plt.close(fig)


def main():
    """Publication entry: redraw the fixed measured CSV without experiment assets."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--csv', type=Path, default=HERE / 'B_to_B2_delta.csv')
    parser.add_argument('--output', type=Path, default=HERE / 'redrawn')
    args = parser.parse_args()
    with args.csv.open(newline='') as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == 12
    assert {(r['layout'], r['arm']) for r in rows} == {(l, a) for l in LAYOUTS for a in ARMS}
    for r in rows:
        for key in ('delta_F1_percentage_points', 'delta_full_coverage_actions'):
            r[key] = None if r[key] == '' else float(r[key])
    plot(rows, args.output)
    print('Redrawn 12 fixed measured rows to', args.output)


if __name__ == '__main__': main()
