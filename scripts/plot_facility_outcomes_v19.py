"""Plot complete V19 endpoints and recorded coverage/quality checkpoints.

Requires all twelve physical cases and their replays. Reference ranges are
sampling sensitivity, not confidence intervals. Curves stop at recorded ends.
"""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import zipfile
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

OPTIONS = ('continue_coverage', 'observe_asset_A', 'observe_asset_B')
LABELS = ('Coverage first', 'Observe A first', 'Observe B first')
COLORS = ('#596E79', '#248E88', '#D48A45')
PARENTS = ('D19-P00', 'D19-P01')
ASSIGNMENTS = ('A_complex_B_simple', 'A_simple_B_complex')
ASSIGNMENT_LABELS = ('A complex / B simple', 'A simple / B complex')
REFERENCES = ('2026', '2027', '2028')


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(path.read_text())


def write(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False)+'\n')


def require_hash(path, expected):
    if not path.is_file() or sha(path) != expected:
        raise ValueError('Missing or changed input: '+str(path))


def load_inputs(source, runs):
    analysis = read(source / 'result.json')
    manifest = read(runs / 'manifest.json')
    if analysis.get('status') != 'complete' or analysis.get('unique_physical_outcomes') != 12:
        raise ValueError('Complete twelve-outcome V19 analysis is required')
    if manifest.get('status') != 'complete' or len(manifest.get('cases', [])) != 12:
        raise ValueError('All twelve physical runs and replays must complete before plotting')
    analysis_inventory = read(source / 'artifact_hashes.json')
    run_inventory = read(runs / 'artifact_hashes.json')
    for name, expected in analysis_inventory.items():
        require_hash(source / name, expected)
    if analysis['input_inventory_sha256'] != sha(runs / 'artifact_hashes.json'):
        raise ValueError('Analysis was not computed from this frozen run inventory')
    with (source / 'outcomes.csv').open(newline='') as stream:
        rows = list(csv.DictReader(stream))
    table = {(r['parent'], r['assignment'], r['option'], r['reference']): r for r in rows}
    expected_keys = {(p, a, o, s) for p in PARENTS for a in ASSIGNMENTS for o in OPTIONS for s in REFERENCES}
    if len(rows) != 36 or set(table) != expected_keys:
        raise ValueError('Every parent, assignment, option and reference must appear exactly once')
    records = {}
    for case in manifest['cases']:
        key = (case['parent'], case['assignment'], case['option'])
        if key in records:
            raise ValueError('Duplicate declared physical case')
        folder = f'case_{case["index"]:02d}'
        for name in (folder+'/result.json', folder+'/verification.json'):
            if name not in run_inventory:
                raise ValueError('Input is not sealed in the run inventory: '+name)
            require_hash(runs / name, run_inventory[name])
        if read(runs / folder / 'verification.json').get('status') != 'passed':
            raise ValueError('Physical replay must pass for every plotted condition')
        record = read(runs / folder / 'result.json')
        if record['case'] != case:
            raise ValueError('Stored case differs from the declared run manifest')
        for reference in REFERENCES:
            row, metric = table[(*key, reference)], record['after'][reference]
            comparisons = {'coverage': metric['coverage_2d'], 'asset_f1': metric['05cm']['asset_macro_f1'],
                           'joint': metric['05cm']['joint_external'], 'actions': record['paid_actions']}
            if any(float(row[name]) != value for name, value in comparisons.items()):
                raise ValueError('CSV and physical outcome disagree: '+str(key))
            if row['eligible'] not in ('True', 'False') or (row['eligible'] == 'True') != metric['eligible']:
                raise ValueError('CSV eligibility differs from the physical outcome')
        curve = record['quality_curve']
        actions = [point['action_id'] for point in curve]
        if (not actions or actions[0] != 0 or actions[-1] != record['paid_actions']
                or any(a >= b for a, b in zip(actions, actions[1:]))
                or record['paid_actions'] > case['budget']):
            raise ValueError('Missing, unordered or invalid physical checkpoints')
        if curve[0]['metrics'] != record['before']['2026'] or curve[-1]['metrics'] != record['after']['2026']:
            raise ValueError('Curve endpoints disagree with primary-reference results')
        for point in curve:
            for value in (point['metrics']['coverage_2d'], point['metrics']['05cm']['asset_macro_f1']):
                if not np.isfinite(value) or not 0 <= value <= 1:
                    raise ValueError('Coverage/quality checkpoint outside [0,1]')
        records[key] = record
    if set(records) != {key[:3] for key in expected_keys}:
        raise ValueError('Incomplete physical case table')
    return analysis, table, records


def style_axis(ax):
    ax.grid(axis='y', alpha=.18)
    ax.set_axisbelow(True)
    ax.spines[['top', 'right']].set_visible(False)


def plot_endpoints(table, output):
    fig, axes = plt.subplots(1, 2, figsize=(11.7, 5.1), sharey=True)
    for ax, parent in zip(axes, PARENTS):
        for assignment_index, assignment in enumerate(ASSIGNMENTS):
            for option_index, option in enumerate(OPTIONS):
                rows = [table[(parent, assignment, option, reference)] for reference in REFERENCES]
                value = float(rows[0]['joint']); values = [float(row['joint']) for row in rows]
                eligible = rows[0]['eligible'] == 'True'
                x = assignment_index+(option_index-1)*.24
                ax.bar(x, value, .22, color=COLORS[option_index],
                       hatch='' if eligible else '///', edgecolor='white' if eligible else '#293340', linewidth=.7)
                ax.errorbar(x, value, yerr=[[value-min(values)], [max(values)-value]],
                            fmt='none', ecolor='#293340', capsize=3, linewidth=1)
                ax.text(x, max(values)+.022, f'{value:.3f}', ha='center', va='bottom', fontsize=9)
        ax.set_title(parent)
        ax.set_xticks([0, 1], ASSIGNMENT_LABELS)
        ax.tick_params(axis='x', labelsize=9)
        ax.set_ylim(0, 1.10)
        style_axis(ax)
    axes[0].set_ylabel('J_external = 2D coverage x mean exterior F1 @ 5 cm')
    handles = [Patch(facecolor=color, label=label) for color, label in zip(COLORS, LABELS)]
    handles.append(Patch(facecolor='#DBDFE4', edgecolor='#293340', hatch='///', label='Ineligible endpoint'))
    fig.legend(handles=handles, loc='upper center', bbox_to_anchor=(.5, .925), ncol=4, frameon=False, fontsize=9)
    fig.suptitle('V19: final exterior-documentation outcomes', y=.99, fontsize=13)
    fig.text(.5, .028, 'All 12 outcomes retained. Bars: reference 2026; whiskers: min-max over 3 reference samples, not confidence intervals.\n'
             'Eligibility requires coverage >= 0.80, safe return, zero collisions and no recorded runtime failure.',
             ha='center', fontsize=8, linespacing=1.5)
    fig.tight_layout(rect=(0, .11, 1, .83))
    for extension in ('png', 'svg'):
        fig.savefig(output / f'final_outcomes.{extension}', dpi=160, bbox_inches='tight')
    plt.close(fig)


def plot_curves(records, output):
    fig, axes = plt.subplots(2, 2, figsize=(12, 8.4), sharex=True, sharey=True)
    maximum_budget = max(record['case']['budget'] for record in records.values())
    ends = []
    for row_index, parent in enumerate(PARENTS):
        for col_index, assignment in enumerate(ASSIGNMENTS):
            ax = axes[row_index, col_index]
            group = [records[(parent, assignment, option)] for option in OPTIONS]
            budgets = {record['case']['budget'] for record in group}
            if len(budgets) != 1:
                raise ValueError('Shared-option budget differs inside one paired state')
            budget = budgets.pop()
            ax.axhline(.8, color='#9B5962', linewidth=1.1, linestyle=':', zorder=0)
            ax.axvline(budget, color='#9B9FA5', linewidth=.9, linestyle=':', zorder=0)
            for record, color, option in zip(group, COLORS, OPTIONS):
                curve = record['quality_curve']
                actions = [point['action_id'] for point in curve]
                coverage = [point['metrics']['coverage_2d'] for point in curve]
                quality = [point['metrics']['05cm']['asset_macro_f1'] for point in curve]
                ax.plot(actions, coverage, color=color, linewidth=1.65, marker='.', markersize=3.5)
                ax.plot(actions, quality, color=color, linewidth=1.65, linestyle='--', marker='.', markersize=3.5)
                early = record['paid_actions'] < budget
                marker = 's' if early else 'o'
                ax.scatter([actions[-1], actions[-1]], [coverage[-1], quality[-1]],
                           marker=marker, s=37, facecolors='white', edgecolors=color, linewidths=1.6, zorder=6)
                ends.append(dict(parent=parent, assignment=assignment, option=option,
                    budget=budget, actual_paid_actions=record['paid_actions'], early_stop=early,
                    eligible=record['after']['2026']['eligible'], recorded_checkpoints=len(curve),
                    nominal_checkpoint_stride=record['curve_action_stride']))
            ax.text(.985, .035, 'End actions (coverage / A / B): '+
                    ' / '.join(str(record['paid_actions']) for record in group)+f'\nBudget: {budget}',
                    transform=ax.transAxes, ha='right', va='bottom', fontsize=8, color='#485462',
                    bbox=dict(facecolor='white', alpha=.8, edgecolor='none', pad=2))
            ax.set_title(parent+' | '+ASSIGNMENT_LABELS[col_index], fontsize=10)
            ax.set_xlim(0, maximum_budget+4); ax.set_ylim(0, 1.045)
            if col_index == 0:
                ax.set_ylabel('Coverage / mean exterior F1 @ 5 cm')
            if row_index == 1:
                ax.set_xlabel('Paid actions from the shared initial observation')
            style_axis(ax)
    handles = [Line2D([0], [0], color=color, lw=2, label=label) for color, label in zip(COLORS, LABELS)]
    handles.extend([Line2D([0], [0], color='#293340', lw=1.5, label='C_2D (solid)'),
                    Line2D([0], [0], color='#293340', lw=1.5, linestyle='--', label='Q_external (dashed)'),
                    Line2D([0], [0], color='#9B5962', lw=1.2, linestyle=':', label='Coverage gate 0.80'),
                    Line2D([0], [0], color='#293340', marker='s', markerfacecolor='white', linestyle='none', label='Early end'),
                    Line2D([0], [0], color='#293340', marker='o', markerfacecolor='white', linestyle='none', label='End at budget')])
    fig.legend(handles=handles, loc='upper center', bbox_to_anchor=(.5, .95), ncol=4, frameon=False, fontsize=9)
    fig.suptitle('V19: recorded coverage and exterior-quality trajectories', y=.995, fontsize=13)
    fig.text(.5, .023, 'Primary reference 2026; same action scale in all panels. Lines connect recorded 20-action checkpoints plus endpoints.\n'
             'No extrapolation after an early end; dotted vertical lines mark each parent budget. These are sampled trajectories, not per-action evaluations.',
             ha='center', fontsize=8, linespacing=1.5)
    fig.tight_layout(rect=(0, .09, 1, .85), h_pad=1.5, w_pad=1.8)
    for extension in ('png', 'svg'):
        fig.savefig(output / f'quality_trajectories.{extension}', dpi=160, bbox_inches='tight')
    plt.close(fig)
    return ends


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True, help='Complete V19 information analysis directory')
    parser.add_argument('--runs', type=Path, required=True, help='Complete twelve-case V19 pilot directory')
    parser.add_argument('--output', type=Path, required=True, help='New directory for final plots and metadata')
    args = parser.parse_args()
    source, runs, output = args.source.resolve(), args.runs.resolve(), args.output.resolve()
    if output in (source, runs) or source in output.parents or runs in output.parents:
        raise ValueError('Plot output must be separate from frozen input directories')
    script_hash = sha(__file__)
    analysis, table, records = load_inputs(source, runs)
    output.mkdir(parents=True, exist_ok=False)
    plot_endpoints(table, output)
    ends = plot_curves(records, output)
    if sha(__file__) != script_hash:
        raise ValueError('Plot script changed while producing figures')
    write(output / 'result.json', dict(status='complete', plotted_physical_cases=len(records),
        plotted_endpoint_rows=len(table), primary_reference=2026, curve_endpoints=ends,
        ineligible_cases=sum(not item['eligible'] for item in ends),
        early_stop_cases=sum(item['early_stop'] for item in ends),
        reference_range_is_confidence_interval=False, curves_extrapolated=False,
        analysis_inventory_sha256=sha(source / 'artifact_hashes.json'),
        run_inventory_sha256=sha(runs / 'artifact_hashes.json'),
        plot_script_sha256=script_hash, primary_screen_passed=analysis['primary_screen_passed']))
    with zipfile.ZipFile(output / 'sources.zip', 'w', zipfile.ZIP_DEFLATED) as archive:
        archive.write(__file__, Path(__file__).name)
    write(output / 'artifact_hashes.json', {p.name: sha(p) for p in output.iterdir() if p.name != 'artifact_hashes.json'})
    print(json.dumps({'status': 'complete', 'output': str(output), 'cases': len(records)}))


if __name__ == '__main__':
    main()
