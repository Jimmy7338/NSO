#!/usr/bin/env python3
"""Export reviewed development endpoints and descriptive scientific figures.

Reads one explicit summary snapshot; never runs policies, fusion or evaluation.
Missing results remain missing. Development parents are not a held-out sample.
"""
import argparse
import csv
import hashlib
import json
from pathlib import Path


METHODS = ['G', 'fixed_semantic', 'bayes_semantic', 'shared_semantic',
           'known_structure_reference']
LABELS = ['Geometry', 'Fixed semantic', 'Bayesian semantic', 'Shared reliability',
          'Known structure']
COLORS = ['#555555', '#E69F00', '#0072B2', '#009E73', '#CC79A7']
MARKERS = ['o', 's', '^', 'D', 'P']
PARENTS = [f'SEM_P{i:02d}' for i in range(6)]


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--summary', type=Path, required=True)
    parser.add_argument('--expected-summary-sha256', required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--allow-incomplete', action='store_true')
    args = parser.parse_args()
    summary_path = args.summary.resolve()
    if sha(summary_path) != args.expected_summary_sha256:
        raise ValueError('Summary differs from explicit hash')
    data = json.loads(summary_path.read_text())
    if data.get('schema') != 'semantic.scene_matrix.progress_summary.v1':
        raise ValueError('A verified scene-matrix summary is required')
    if data['states']['integrity_error']:
        raise ValueError('Resolve integrity errors before plotting results')
    if not data['all_declared_endpoints_reviewed'] and not args.allow_incomplete:
        raise ValueError('Incomplete results require explicit --allow-incomplete')
    rows = data['episodes']
    if len(rows) != 96 or len({r['run_id'] for r in rows}) != 96:
        raise ValueError('Expected all 96 unique slots, including missing outcomes')
    for row in rows:
        if row['metrics'] is not None:
            if abs(row['J_nav'] - row['C_nav'] * row['Q']) > 1e-12:
                raise ValueError('Inconsistent endpoint identity')
            if row['evaluation_execution'] not in ('recomputed', 'reused'):
                raise ValueError('Reviewed metric provenance required')
    output = args.output.resolve()
    if 'episodes' in output.parts or output.exists():
        raise ValueError('Use a new output directory outside episode archives')

    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import numpy as np
    from matplotlib.lines import Line2D

    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 9,
                         'axes.titlesize': 10, 'axes.labelsize': 9,
                         'axes.spines.top': False, 'axes.spines.right': False,
                         'pdf.fonttype': 42, 'ps.fonttype': 42,
                         'savefig.dpi': 240})
    output.mkdir(parents=True, exist_ok=False)
    fields = ['run_id', 'parent_id', 'condition', 'method', 'budget', 'tie_rule',
              'noise_seed', 'state', 'C_nav', 'Q', 'J_nav',
              'evaluation_execution', 'numerical_evaluation_recomputed']
    with (output / 'all_declared_endpoints.csv').open('x', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows({k: r.get(k) for k in fields} for r in rows)
    count = data['reviewed_metric_endpoints']
    snapshot = f'Development results: {count}/96 endpoints reviewed'
    if count < 96:
        snapshot += ' (incomplete; missing points are not zero)'
    saved = []

    def save(fig, name):
        for suffix in ('pdf', 'png'):
            path = output / f'{name}.{suffix}'
            fig.savefig(path, bbox_inches='tight')
            saved.append(path)
        plt.close(fig)

    conditions = [('nominal_relationship', 'Nominal class–structure relation'),
                  ('structure_relationship_shift', 'Class–structure relation shift')]
    fig, axes = plt.subplots(3, 2, figsize=(11.4, 8.2), sharex=True, sharey='row')
    labels = [('C_nav', 'Navigable-floor coverage'),
              ('Q', 'Equal-facility exterior F1 @ 5 cm'),
              ('J_nav', 'Joint endpoint: coverage × F1')]
    for col, (condition, title) in enumerate(conditions):
        for rr, (metric, ylabel) in enumerate(labels):
            ax = axes[rr, col]
            for method_i, method in enumerate(METHODS):
                for parent_i, parent in enumerate(PARENTS):
                    selected = [r for r in rows if r['run_id'].startswith('core_')
                                and r['condition'] == condition and r['parent_id'] == parent
                                and r['method'] == method]
                    if len(selected) != 1:
                        raise ValueError('Core matrix shape is not as declared')
                    value = selected[0].get(metric)
                    if value is not None:
                        ax.scatter(parent_i + (method_i - 2) * .14, value, s=28,
                                   color=COLORS[method_i], marker=MARKERS[method_i],
                                   edgecolors='white', linewidths=.35, zorder=3)
            ax.set_ylim(-.025, 1.055)
            ax.set_xlim(-.65, 5.65)
            ax.set_xticks(range(6), [p.removeprefix('SEM_') for p in PARENTS])
            ax.grid(axis='y', color='#E5E5E5', linewidth=.6)
            if col == 0:
                ax.set_ylabel(ylabel)
            if rr == 0:
                ax.set_title(title)
            if rr == 2:
                ax.set_xlabel('Development parent layout')
    handles = [Line2D([], [], marker=m, linestyle='', color=c, label=l)
               for m, c, l in zip(MARKERS, COLORS, LABELS)]
    fig.legend(handles=handles, loc='lower center', ncol=5, frameon=False,
               bbox_to_anchor=(.5, -.01))
    fig.suptitle(snapshot, fontsize=11)
    fig.tight_layout(rect=(0, .035, 1, .96))
    save(fig, 'core_endpoints')

    fig, axes = plt.subplots(1, 3, figsize=(11.4, 3.6))
    titles = ['Shared vs geometry · nominal', 'Shared vs Bayesian · shift',
              'Shared vs Bayesian · nominal']
    comparison_rows = []
    for ax, comp, title in zip(axes, data['core_comparisons'], titles):
        # Pair entries are read from the summary; differences are recomputed
        # below from the reviewed endpoints, never from unreviewed predictions.
        condition = 'structure_relationship_shift' if comp['name'] == 'candidate_increment' else 'nominal_relationship'
        baseline = 'G' if comp['name'] == 'semantic_information' else 'bayes_semantic'
        threshold = {'semantic_information': 5, 'candidate_increment': 2,
                     'reliable_noninferiority': -1}[comp['name']]
        ax.axhline(0, color='#777777', linewidth=.7)
        ax.axhline(threshold, color='#999999', linewidth=.8, linestyle='--')
        values = []
        for parent_i, parent in enumerate(PARENTS):
            matched = {r['method']: r for r in rows if r['run_id'].startswith('core_')
                       and r['parent_id'] == parent and r['condition'] == condition}
            base = matched[baseline]['J_nav']
            candidate = matched['shared_semantic']['J_nav']
            absolute = candidate - base if base is not None and candidate is not None else None
            relative = 100 * absolute / base if absolute is not None and base > 0 else None
            comparison_rows.append(dict(comparison=comp['name'], parent=parent,
                                        baseline_J=base, candidate_J=candidate,
                                        absolute_difference=absolute, relative_percent=relative))
            if relative is not None:
                values.append(relative)
                ax.scatter(parent_i, relative, color='#009E73', s=30, zorder=3)
        if not values:
            ax.text(.5, .45, 'Awaiting reviewed pairs', ha='center', transform=ax.transAxes,
                    color='#777777', fontsize=9)
        ax.set_xticks(range(6), [p.removeprefix('SEM_') for p in PARENTS])
        ax.set_xlim(-.5, 5.5)
        ax.set_title(title, fontsize=9)
        ax.set_xlabel('Development parent layout')
        ax.grid(axis='y', color='#E5E5E5', linewidth=.6)
    axes[0].set_ylabel('Paired relative joint-score difference (%)')
    fig.suptitle(snapshot, fontsize=10)
    fig.text(.5, .005, 'Dashed lines: investment thresholds, applied to ratio of group means; individual points are descriptive.',
             ha='center', fontsize=8)
    fig.tight_layout(rect=(0, .04, 1, .93))
    save(fig, 'paired_development_effects')
    with (output / 'paired_effects.csv').open('x', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(comparison_rows[0]))
        writer.writeheader()
        writer.writerows(comparison_rows)
    manifest = dict(schema='semantic.development_figures.v1', summary_path=str(summary_path),
                    summary_sha256=args.expected_summary_sha256, source_sha256=sha(__file__),
                    reviewed_endpoints=count, declared_endpoints=96,
                    all_endpoints_reviewed=data['all_declared_endpoints_reviewed'],
                    missing_values_imputed=False, numerical_evaluations=0, worlds=0,
                    uncertainty_intervals_computed=False,
                    limits=['Six development parents, not six independent geometry families or held-out confirmation.',
                            'Known-structure reference is not a proven optimum.',
                            'Reuse receipts remain labeled in CSV; points do not imply independent numerical recalculations.',
                            'Threshold lines do not establish full attribution, safety or parent-consistency gates.'],
                    files={p.name: sha(p) for p in saved + list(output.glob('*.csv'))})
    with (output / 'manifest.json').open('x') as stream:
        json.dump(manifest, stream, indent=2, sort_keys=True)
        stream.write('\n')
    if sha(summary_path) != args.expected_summary_sha256:
        raise ValueError('Input snapshot changed while plotting')
    print(json.dumps({'output': str(output), 'reviewed_endpoints': count, 'figure_files': len(saved)}))


if __name__ == '__main__':
    main()
