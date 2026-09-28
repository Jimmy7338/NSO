#!/usr/bin/env python3
"""Export the frozen information screens and the separate commitment effect."""
import argparse
import hashlib
import json
from pathlib import Path


def main(old, new, comparison, output):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import numpy as np

    old_data, new_data, change = [json.loads(p.read_text()) for p in (old, new, comparison)]
    if any(d['status'] != 'complete' for d in (old_data, new_data, change)):
        raise ValueError('Complete results required')
    output.mkdir(parents=True, exist_ok=False)
    old_seeds = old_data['reference_seed_results']
    new_seeds = new_data['reference_seed_results']
    seeds = [r['reference_seed'] for r in old_seeds]
    if seeds != [r['reference_seed'] for r in new_seeds]:
        raise ValueError('Reference seeds differ')
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.9), layout='constrained')
    x = np.arange(len(seeds))
    for offset, data, label, color in ((-.17, old_seeds, 'Full first roundtrip', '#7A8A99'),
                                      (.17, new_seeds, 'First outbound, then replan', '#277E9A')):
        values = [r['relative_information_value'] * 100 for r in data]
        axes[0].bar(x + offset, values, width=.32, label=label, color=color)
        for position, value in zip(x + offset, values):
            axes[0].text(position, value + .008, f'{value:.3f}', ha='center', fontsize=8)
    axes[0].axhline(.5, linestyle='--', color='#AA343B', label='Frozen gate: 0.5%')
    axes[0].set(xticks=x, xticklabels=seeds, ylabel='Relative information value (%)',
                title='Semantic information screen')
    axes[0].legend(frameon=False, fontsize=8, loc='upper left')
    axes[0].set_ylim(0, max(.5, *[r['relative_information_value'] * 100 for r in new_seeds]) * 1.45)
    parents = change['parents']
    axes[1].bar([p['parent'] for p in parents],
                [p['mean_candidate_joint_difference'] for p in parents], color='#848E79', width=.55)
    axes[1].axhline(0, color='black', linewidth=.7)
    axes[1].set(ylabel='Mean same-candidate joint-score difference',
                title='Commitment change: not semantic-policy gain')
    for axis in axes:
        axis.spines[['top', 'right']].set_visible(False)
        axis.grid(axis='y', alpha=.18)
        axis.set_axisbelow(True)
    fig.suptitle('Four development parents; no trained policy evaluated', fontsize=12)
    for extension in ('png', 'svg'):
        fig.savefig(output / ('commitment_comparison.' + extension), dpi=200)
    plt.close(fig)
    manifest = dict(input_sha256={str(p): hashlib.sha256(p.read_bytes()).hexdigest()
                                  for p in (old, new, comparison)},
                    plotting_source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                    note='Reference-sampling sensitivity is not a confidence interval.')
    (output / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('old', 'new', 'comparison', 'output'):
        parser.add_argument('--' + name, type=Path, required=True)
    args = parser.parse_args()
    main(args.old, args.new, args.comparison, args.output)
