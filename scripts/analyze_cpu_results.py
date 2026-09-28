#!/usr/bin/env python3
"""Analyze CPU grid runs; average starts within each map before comparing maps."""
from collections import defaultdict
import csv
import json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap


def analyze(folder):
    folder = Path(folder)
    metadata = json.loads((folder / 'run_metadata.json').read_text())
    if metadata.get('schema') != 'cpu_grid_v1':
        raise ValueError('not a CPU grid run')
    config = json.loads((folder / 'config.json').read_text())
    records = [json.loads(line) for line in (folder / 'episodes.jsonl').read_text().splitlines() if line]
    ids = [r['episode_key'] for r in records]
    if len(ids) != len(set(ids)):
        raise ValueError('duplicate episode keys')
    complete = [r for r in records if r['status'] == 'complete']
    expected_keys = {f"{m['layout']}_seed{m['seed']}_start{s}_{method}"
                     for m in config['maps'] for s in config['start_seeds'] for method in config['methods']}
    if set(ids) - expected_keys:
        raise ValueError('unexpected episode outside configured matrix')
    paired_assets = {}
    for record in complete:
        key = (record['map_id'], record['start_seed'])
        assets = (record['ground_truth_sha256'], tuple(record['initial_pose']), record.get('semantic_sha256'))
        if key in paired_assets and paired_assets[key] != assets:
            raise ValueError('paired methods have different maps, starts or semantic fields')
        paired_assets[key] = assets
    expected = metadata['expected_episodes']
    report = {'schema': 'cpu_grid_v1', 'expected_episodes': expected,
              'completed_episodes': len(complete), 'failed_episodes': len(records) - len(complete),
              'run_complete': metadata['status'] == 'complete' and len(complete) == expected and set(ids) == expected_keys,
              'statistics_unit': 'map; average repeated starts first',
              'methods': {}, 'paired_differences': {}}
    groups = defaultdict(list)
    curves = defaultdict(list)
    horizon = config['environment']['max_steps']
    for r in complete:
        groups[(r['method'], r['map_id'])].append(r)
        with (folder / r['artifact_dir'] / 'steps.csv').open() as stream:
            rows = list(csv.DictReader(stream))
        curves[(r['method'], r['map_id'])].append(np.interp(
            np.arange(horizon + 1), [int(x['steps']) for x in rows],
            [float(x['coverage_ratio']) for x in rows]))
    fields = ('coverage_ratio', 'coverage_auc', 'path_length_m', 'collisions', 'revisit_ratio',
              'episode_wall_time_s', 'decision_ms_median', 'decision_ms_p95')
    map_values = {}
    for method in config['methods']:
        values = {map_id: {field: float(np.mean([r[field] for r in rows])) for field in fields}
                  for (m, map_id), rows in groups.items() if m == method}
        map_values[method] = values
        summary = {'map_count': len(values),
                   'episode_count': sum(r['method'] == method for r in complete),
                   'unreached_80_count': sum(r['method'] == method and r['steps_to_80'] is None for r in complete),
                   'unreached_90_count': sum(r['method'] == method and r['steps_to_90'] is None for r in complete)}
        for field in fields:
            numbers = [v[field] for v in values.values()]
            summary[field] = {'mean': float(np.mean(numbers)) if numbers else None,
                              'std_across_maps': float(np.std(numbers)) if numbers else None}
        report['methods'][method] = summary
    if len(config['methods']) == 2:
        a, b = config['methods']
        common = sorted(set(map_values[a]) & set(map_values[b]))
        for field in ('coverage_ratio', 'coverage_auc'):
            diff = [map_values[b][m][field] - map_values[a][m][field] for m in common]
            report['paired_differences'][field] = {'direction': f'{b} minus {a}', 'map_count': len(common),
                'mean': float(np.mean(diff)) if diff else None, 'by_map': dict(zip(common, diff))}
    (folder / 'summary.json').write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
    fig, ax = plt.subplots(figsize=(8, 4.8))
    for method in config['methods']:
        map_curves = [np.mean(c, axis=0) for (m, _), c in curves.items() if m == method]
        if not map_curves:
            continue
        array = np.asarray(map_curves) * 100
        mean, std = array.mean(0), array.std(0)
        ax.plot(mean, label=method.replace('_', ' '))
        ax.fill_between(np.arange(horizon + 1), np.maximum(0, mean - std),
                        np.minimum(100, mean + std), alpha=.12)
    ax.set(xlabel='Atomic actions (turns included)', ylabel='Observed reachable area (%)',
           title='CPU exploration | mean ± SD across maps', ylim=(0, 100))
    ax.legend(loc='lower right'); ax.grid(alpha=.2)
    fig.tight_layout(); fig.savefig(folder / 'coverage_curves.png', dpi=170); plt.close(fig)
    # Compare both methods on the first map of each layout. Ground truth is
    # used for this post-experiment visualization only.
    layouts = list(dict.fromkeys(r['layout'] for r in complete))
    if layouts:
        fig, axes = plt.subplots(len(layouts), len(config['methods']),
                                 figsize=(5 * len(config['methods']), 4 * len(layouts)), squeeze=False)
        for row, layout in enumerate(layouts):
            first = next(r for r in complete if r['layout'] == layout)
            for col, method in enumerate(config['methods']):
                ax = axes[row, col]
                matches = [r for r in complete if r['map_id'] == first['map_id']
                           and r['start_seed'] == first['start_seed'] and r['method'] == method]
                if not matches:
                    ax.set_axis_off(); continue
                r = matches[0]
                with np.load(folder / r['artifact_dir'] / 'observations.npz') as data:
                    occupancy, belief, poses = data['occupancy'], data['final_belief'], data['poses']
                    display = np.ones(occupancy.shape)
                    display[belief == -1] = 0
                    display[occupancy] = 2
                    ax.imshow(display, cmap=ListedColormap(['#d5dbe3', '#fafbfc', '#26374a']), vmin=0, vmax=2)
                    ax.plot(poses[:, 1], poses[:, 0], color='#e36d3f', linewidth=1.1)
                    ax.scatter(poses[0, 1], poses[0, 0], marker='o', c='#16a085', s=35)
                    ax.scatter(poses[-1, 1], poses[-1, 0], marker='x', c='#d13f3f', s=35)
                ax.set_title(f"{layout} | {method}\ncoverage {r['coverage_ratio']:.1%}, actions {r['steps']}", fontsize=10)
                ax.set_xticks([]); ax.set_yticks([])
        fig.suptitle('Evaluation-only ground truth overlay | green: start, red: end, gray: unobserved', fontsize=12)
        fig.tight_layout(rect=(0, 0, 1, .985)); fig.savefig(folder / 'trajectories.png', dpi=120); plt.close(fig)
    return report


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('run_directory', type=Path)
    analyze(parser.parse_args().run_directory)
