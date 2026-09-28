#!/usr/bin/env python3
"""Plot the frozen ablation and boundary comparisons from reviewed snapshots.

No policy, sensor, fusion, or surface evaluator is imported. Missing paired
outcomes stay missing, and repeated conditions never increase parent counts.
"""
import argparse
import csv
import hashlib
import json
import math
from pathlib import Path


METRICS = ('C_nav', 'Q', 'J_nav')
MATCH_FIELDS = ('asset_id', 'parent_id', 'condition', 'budget', 'noise_seed',
                'tie_rule', 'tie_seed')
CONDITIONS = [('nom', 'nominal_relationship', 'Nominal relation'),
              ('shift', 'structure_relationship_shift', 'Relation shift')]
CONTRASTS = [('NF', 'S', 'No planning-belief\nfeedback vs full'),
             ('NP', 'S', 'No future-information\nutility vs full'),
             ('NC', 'S', 'No cross-instance\nlookahead vs full'),
             ('NC', 'B', 'Shared belief\nvs independent Bayes')]


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def validate_snapshot(data, protocol):
    if data.get('schema') != 'semantic.scene_matrix.progress_summary.v1':
        raise ValueError('A verified scene-matrix summary is required')
    if data['states']['integrity_error']:
        raise ValueError('Resolve reported integrity errors before plotting')
    rows = data['episodes']
    ids = [r['run_id'] for r in rows]
    if len(ids) != 96 or len(set(ids)) != 96 or set(ids) != set(protocol['slots']):
        raise ValueError('All 96 unique frozen slots, including missing ones, are required')
    for row in rows:
        slot = protocol['slots'][row['run_id']]
        if any(row[k] != v for k, v in slot.items()):
            raise ValueError('Endpoint metadata differs from its declared slot')
        parent, condition = row['asset_id'].split('__', 1)
        if (row['parent_id'], row['condition']) != (parent, condition):
            raise ValueError('Parent/condition must follow the declared asset')
        if row['metrics'] is None:
            if any(row.get(k) is not None for k in METRICS):
                raise ValueError('Missing outcomes must not contain imputed metric values')
        else:
            if row['evaluation_execution'] not in ('recomputed', 'reused'):
                raise ValueError('Reviewed metric provenance is required')
            if any(not math.isfinite(row[k]) or not 0 <= row[k] <= 1 for k in METRICS):
                raise ValueError('Endpoint metrics must be finite fractions')
            if any(row[k] != row['metrics'].get(k) for k in METRICS):
                raise ValueError('Plotted metrics differ from the embedded reviewed evaluation')
            if abs(row['J_nav'] - row['C_nav'] * row['Q']) > 1e-12:
                raise ValueError('Joint endpoint identity does not hold')
    reviewed = sum(r['metrics'] is not None for r in rows)
    if data.get('reviewed_metric_endpoints', reviewed) != reviewed:
        raise ValueError('Reviewed count does not match the retained endpoint rows')
    if data.get('all_declared_endpoints_reviewed') is not (reviewed == len(rows)):
        raise ValueError('Completion flag does not match the retained endpoint rows')
    return {r['run_id']: r for r in rows}


def paired_row(candidate, baseline, contrast):
    if any(candidate[k] != baseline[k] for k in MATCH_FIELDS):
        raise ValueError('Ablation pairs must share all non-method conditions')
    complete = candidate['metrics'] is not None and baseline['metrics'] is not None
    result = dict(contrast=contrast, parent_id=candidate['parent_id'],
                  condition=candidate['condition'], candidate_run_id=candidate['run_id'],
                  baseline_run_id=baseline['run_id'], complete_reviewed_pair=complete,
                  candidate_state=candidate['state'], baseline_state=baseline['state'],
                  candidate_evaluation=candidate.get('evaluation_execution'),
                  baseline_evaluation=baseline.get('evaluation_execution'))
    for key in ('executed_paid_actions', 'execution_elapsed_s', 'collisions', 'returned_xy_and_yaw'):
        result['candidate_' + key] = candidate.get(key)
        result['baseline_' + key] = baseline.get(key)
    for metric in METRICS:
        result['candidate_' + metric] = candidate[metric]
        result['baseline_' + metric] = baseline[metric]
        result['delta_' + metric] = candidate[metric] - baseline[metric] if complete else None
    result['relative_delta_J_percent'] = (
        100 * result['delta_J_nav'] / baseline['J_nav']
        if complete and baseline['J_nav'] > 0 else None)
    return result


def build_tables(rows):
    ablations = []
    for tag, _, _ in CONDITIONS:
        for parent in ('P00', 'P04'):
            for candidate, baseline, _ in CONTRASTS:
                c = rows[f'ablation_{parent}_{tag}_{candidate}_b120_lexicographic']
                b = rows[f'core_{parent}_{tag}_{baseline}_b120_lexicographic']
                ablations.append(paired_row(c, b, f'{candidate}_vs_{baseline}'))
    boundary = []
    for method in ('G', 'B', 'S'):
        ids = [('budget', str(budget),
                f'{"core" if budget == 120 else "budget"}_P01_nom_{method}_b{budget}_lexicographic')
               for budget in (80, 120, 160)]
        ids += [('tie', tie,
                 f'{"core" if tie == "lexicographic" else "tie"}_P02_nom_{method}_b120_{tie}')
                for tie in ('lexicographic', 'reverse', 'seeded_random')]
        ids += [('noinfo', parent, f'noinfo_{parent}_none_{method}_b120_lexicographic')
                for parent in ('P01', 'P05')]
        ids += [('recognition', parent, f'recognition_{parent}_wrong_{method}_b120_lexicographic')
                for parent in ('P00', 'P04')]
        for block, level, run_id in ids:
            r = rows[run_id]
            boundary.append(dict(block=block, level=level, label=method,
                                 **{k: r.get(k) for k in ('run_id', 'parent_id', 'condition',
                                    'method', 'budget', 'tie_rule', 'tie_seed', 'noise_seed', 'state',
                                    'evaluation_execution', 'executed_paid_actions',
                                    'execution_elapsed_s', 'collisions', 'returned_xy_and_yaw', *METRICS)},
                                 reused_core_anchor=run_id.startswith('core_')))
    return ablations, boundary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--summary', type=Path, required=True)
    parser.add_argument('--expected-summary-sha256', required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--allow-incomplete', action='store_true')
    args = parser.parse_args()
    snapshot = args.summary.resolve()
    if sha(snapshot) != args.expected_summary_sha256:
        raise ValueError('Summary differs from the explicit SHA256 pin')
    data = json.loads(snapshot.read_text())
    protocol_path = Path(data['protocol_path'])
    if sha(protocol_path) != data['protocol_sha256']:
        raise ValueError('Frozen protocol binding differs')
    rows = validate_snapshot(data, json.loads(protocol_path.read_text()))
    if not data['all_declared_endpoints_reviewed'] and not args.allow_incomplete:
        raise ValueError('Incomplete snapshots require --allow-incomplete')
    if not any(r['metrics'] is not None and not r['run_id'].startswith('core_') for r in rows.values()):
        raise ValueError('At least one reviewed ablation/boundary outcome is required; no empty progress figures')
    output = args.output.resolve()
    if output.exists() or 'episodes' in output.parts:
        raise ValueError('Use a new output directory outside episode archives')
    ablations, boundary = build_tables(rows)

    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 9,
                         'axes.titlesize': 10, 'axes.spines.top': False,
                         'axes.spines.right': False, 'pdf.fonttype': 42,
                         'ps.fonttype': 42, 'savefig.dpi': 240})
    output.mkdir(parents=True, exist_ok=False)
    files = []
    for name, table in [('ablation_pairs.csv', ablations), ('boundary_endpoints.csv', boundary)]:
        path = output / name
        with path.open('x', newline='') as stream:
            writer = csv.DictWriter(stream, fieldnames=list(table[0]))
            writer.writeheader()
            writer.writerows(table)
        files.append(path)

    def save(fig, name):
        for suffix in ('pdf', 'png'):
            path = output / f'{name}.{suffix}'
            fig.savefig(path, bbox_inches='tight')
            files.append(path)
        plt.close(fig)

    stamp = f"Frozen development snapshot: {data['reviewed_metric_endpoints']}/96 endpoints reviewed"
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), sharey=True)
    colors = ['#0072B2', '#CC79A7']
    markers = ['o', '^']
    for ax, (_, condition, title) in zip(axes, CONDITIONS):
        ax.axhline(0, color='#777777', linewidth=.8)
        count = 0
        for pi, parent in enumerate(('SEM_P00', 'SEM_P04')):
            for ci, (candidate, baseline, _) in enumerate(CONTRASTS):
                row = next(r for r in ablations if r['condition'] == condition
                           and r['parent_id'] == parent and r['contrast'] == f'{candidate}_vs_{baseline}')
                value = row['relative_delta_J_percent']
                if value is not None:
                    ax.scatter(ci + (pi - .5) * .15, value, color=colors[pi],
                               marker=markers[pi], s=42, edgecolors='white', linewidths=.4, zorder=3)
                    count += 1
        ax.set_title(f'{title} · {count}/8 drawable paired effects')
        ax.set_xticks(range(4), [r[2] for r in CONTRASTS], fontsize=8)
        ax.set_xlim(-.5, 3.5)
        ax.grid(axis='y', color='#E5E5E5', linewidth=.6)
        if not count:
            ax.text(.5, .55, 'Relative effects unavailable\n(missing pair or zero baseline)',
                    ha='center', transform=ax.transAxes, color='#777777', fontsize=8)
    values = [r['relative_delta_J_percent'] for r in ablations if r['relative_delta_J_percent'] is not None]
    if values:
        span = max(1., max(values) - min(values))
        axes[0].set_ylim(min(-1., min(values) - .1 * span), max(1., max(values) + .1 * span))
    axes[0].set_ylabel('Relative joint-score difference (%)\n(first named policy minus comparison policy)')
    fig.legend(handles=[Line2D([], [], marker=m, linestyle='', color=c, label=p)
                        for m, c, p in zip(markers, colors, ('Parent P00', 'Parent P04'))],
               loc='lower center', bbox_to_anchor=(.5, .035), ncol=2, frameon=False)
    fig.suptitle(stamp, fontsize=11)
    fig.text(.5, .012, 'Two parents; conditions do not increase n. Missing/zero-denominator effects are blank. NC retains shared belief.',
             ha='center', fontsize=8)
    fig.tight_layout(rect=(0, .12, 1, .94))
    if any(r['candidate_J_nav'] is not None for r in ablations):
        save(fig, 'ablation_effects')
    else:
        plt.close(fig)

    groups = [('budget', ['80', '120', '160'], ['80', '120', '160'], 'Action budget · P01'),
              ('tie', ['lexicographic', 'reverse', 'seeded_random'], ['Lexical', 'Reverse', 'Seeded'], 'Tie rule · P02'),
              ('noinfo', ['P01', 'P05'], ['P01', 'P05'], 'Uninformative relation'),
              ('recognition', ['P00', 'P04'], ['P00', 'P04'], 'Persistent recognition error')]
    fig, axes = plt.subplots(3, 4, figsize=(12.4, 8), sharey='row')
    method_colors = ['#555555', '#0072B2', '#009E73']
    for col, (block, levels, labels, title) in enumerate(groups):
        for rr, metric in enumerate(METRICS):
            ax = axes[rr, col]
            count = 0
            for mi, method in enumerate(('G', 'B', 'S')):
                for li, level in enumerate(levels):
                    row = next(r for r in boundary if (r['block'], r['level'], r['label']) == (block, level, method))
                    value = row[metric]
                    if value is not None:
                        ax.scatter(li + (mi - 1) * .14, value, color=method_colors[mi],
                                   marker=['o', '^', 'D'][mi], s=29, edgecolors='white', linewidths=.3, zorder=3)
                        count += 1
            ax.set_xticks(range(len(levels)), labels, fontsize=8)
            ax.set_ylim(-.025, 1.055)
            ax.set_xlim(-.5, len(levels) - .5)
            ax.grid(axis='y', color='#E5E5E5', linewidth=.6)
            if rr == 0:
                ax.set_title(f'{title}\n{count}/{3 * len(levels)} endpoints', fontsize=9)
            if col == 0:
                ax.set_ylabel({'C_nav': 'Floor coverage C', 'Q': 'Exterior F1 Q', 'J_nav': 'Joint endpoint J = C × Q'}[metric])
            if not count:
                ax.text(.5, .45, 'Awaiting review', ha='center', transform=ax.transAxes, fontsize=8, color='#777777')
    fig.legend(handles=[Line2D([], [], marker=m, linestyle='', color=c, label=l)
                        for m, c, l in zip(['o', '^', 'D'], method_colors,
                                          ['Geometry G', 'Bayesian semantics B', 'Shared reliability S'])],
               loc='lower center', ncol=3, frameon=False, bbox_to_anchor=(.5, .02))
    fig.suptitle(stamp, fontsize=11)
    fig.text(.5, .008, 'Descriptive boundary checks: one parent for budget/tie; two for each error condition. Core anchors are reused, not new runs.',
             ha='center', fontsize=8)
    fig.tight_layout(rect=(0, .075, 1, .95))
    # Keep the CSV declaration, but do not export a new boundary figure whose
    # only measured points are old core anchors.
    if any(r['J_nav'] is not None and not r['reused_core_anchor'] for r in boundary):
        save(fig, 'boundary_endpoints')
    else:
        plt.close(fig)
    if sha(snapshot) != args.expected_summary_sha256 or sha(protocol_path) != data['protocol_sha256']:
        raise ValueError('Input bindings changed during export')
    manifest = dict(schema='semantic.attribution_figures.v1', summary_path=str(snapshot),
                    summary_sha256=args.expected_summary_sha256, protocol_sha256=data['protocol_sha256'],
                    source_sha256=sha(__file__), reviewed_endpoints=data['reviewed_metric_endpoints'],
                    ablation_pairs=16, original_ablation_variant_slots=12,
                    supplementary_mechanism_contrasts=['NC_vs_B'],
                    boundary_rows=30, boundary_core_anchors=6,
                    boundary_figure_has_new_reviewed_outcomes=any(
                        r['J_nav'] is not None and not r['reused_core_anchor'] for r in boundary),
                    missing_values_imputed=False, uncertainty_intervals_computed=False,
                    worlds=0, new_numeric_surface_evaluations=0,
                    limits=['Finite development comparisons, not held-out confirmation.',
                            'NC removes cross-instance future utility and retains actual shared reliability.',
                            'NC versus B contrasts shared belief with independent Bayesian belief; literal cross-future flags differ, but B has no shared belief update.',
                            'NC versus B is a supplementary mechanism contrast of saved runs, not a new primary gate or extra experiment.',
                            'No new independent parents from conditions, budgets, ties, or reuse receipts.',
                            'Paired differences do not prove that a disabled component was activated in the original trajectory.'],
                    files={p.name: sha(p) for p in files})
    with (output / 'manifest.json').open('x') as stream:
        json.dump(manifest, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write('\n')
    print(json.dumps(dict(output=str(output), reviewed_endpoints=data['reviewed_metric_endpoints'], worlds=0)))


if __name__ == '__main__':
    main()
