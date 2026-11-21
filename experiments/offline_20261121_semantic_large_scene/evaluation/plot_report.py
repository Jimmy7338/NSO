"""Render the frozen numerical contrasts, including explicit marks at exact zero."""

import argparse
import json
from pathlib import Path

import numpy as np
from matplotlib import pyplot as plt

METRICS = [('J', r'$\Delta J$'), ('C_free', r'$\Delta C_{\mathrm{free}}$'),
           ('F1_scene_05cm', r'$\Delta \mathrm{F1}_{5\,\mathrm{cm}}$')]
COLORS = ('#14627b', '#8a4e22')


def plot(rows, path, comparators):
    by_layout = {}
    for row in rows:
        by_layout.setdefault(row['layout'], {})[row['arm']] = row
    names = list(by_layout)
    locations = np.arange(len(names))
    series = [{key: [100 * (m['S'][key] - m[mode][key])
                    if 'S' in m and mode in m and m['S'].get(key) is not None and m[mode].get(key) is not None
                    else np.nan for m in by_layout.values()] for key, _ in METRICS} for mode in comparators]
    fig, axes = plt.subplots(1, 3, sharey=True, figsize=(9.1, max(2.6, 1.2 + len(names)*.55)),
                             layout='constrained')
    for ax, (metric, label) in zip(axes, METRICS):
        ax.axvline(0, color='#636871', linewidth=.8)
        ax.grid(axis='x', color='#dedfe0', linewidth=.65)
        ax.set_axisbelow(True)
        for index, mode in enumerate(comparators):
            positions = locations + .13 * (2*index-1)
            values = np.asarray(series[index][metric])
            ax.hlines(positions, 0, values, color=COLORS[index], linewidth=1.3)
            ax.scatter(values, positions, s=25, marker=('o', 's')[index],
                       color=COLORS[index], edgecolors='white', linewidth=.35, label=f'S − {mode}', zorder=3)
        if all(all(v == 0 for v in row[metric]) for row in series):
            ax.set_xlim(-.05, .05)
            ax.set_xticks([-.05, 0, .05])
        ax.set_title(label, fontsize=12)
        ax.set_xlabel('Percentage points')
        ax.spines[['top', 'right', 'left']].set_visible(False)
        ax.tick_params(axis='y', length=0)
        ax.set_yticks(locations, [n.replace('_', ' ').capitalize() for n in names])
    axes[0].invert_yaxis()
    axes[-1].legend(loc='best', frameon=False)
    for extension in ('pdf', 'svg'):
        fig.savefig(path.with_suffix('.'+extension), bbox_inches='tight')
    plt.close(fig)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('rows', type=Path)
    parser.add_argument('output', type=Path)
    parser.add_argument('--second', default='X')
    args = parser.parse_args()
    plot(json.loads(args.rows.read_text()), args.output, ('G', args.second))
