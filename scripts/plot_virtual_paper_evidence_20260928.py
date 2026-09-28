#!/usr/bin/env python3
"""Render a paper figure from saved V36/V39 measurements; never run science code.

Only stdlib and matplotlib are imported. No World, planner, TSDF, ROI selection,
surface evaluation, new score definition, or statistical interval is involved.
Existing output directories are refused. Reproduce with:
    python3 scripts/plot_virtual_paper_evidence_20260928.py
"""
import argparse
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import shutil

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / 'docs/thesis/figures/v39_external'
V36 = ROOT / 'audit_results/v36_online_confirmation_20260918'
V39 = ROOT / 'audit_results/v39_external_cpu_20260920'
DEFAULT_OUTPUT = ROOT / 'docs/thesis/figures/virtual_paper_20260928'
CONDITIONS = tuple((p, h) for p in ('P00', 'P01') for h in (0, 1))
METHODS = ('G', 'S', 'SWAP', 'VISTA')
LABELS = {'G': 'G', 'S': 'S', 'SWAP': 'SWAP-I', 'VISTA': 'VISTA-I'}
COLORS = {'G': '#0072B2', 'S': '#D55E00', 'SWAP': '#009E73', 'VISTA': '#CC79A7'}
MARKERS = {'G': 'o', 'S': 's', 'SWAP': '^', 'VISTA': 'D'}
METRICS = ('C_map',) + tuple(f'{m}{t}' for t in (2, 5, 10) for m in ('P', 'R', 'F', 'J'))


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def location(path):
    try:
        return str(path.resolve().relative_to(ROOT))
    except ValueError:
        return str(path.resolve())


def require(condition, message):
    if not condition:
        raise ValueError(message)


def close(a, b, message):
    require(math.isclose(float(a), float(b), rel_tol=0, abs_tol=1e-15), message)


def load_inputs():
    tracked = {}

    def track(path):
        tracked[location(path)] = digest(path)
        return path

    def read(path):
        return json.loads(track(path).read_text(encoding='utf-8'))

    def read_csv(path):
        with track(path).open(newline='', encoding='utf-8') as stream:
            return list(csv.DictReader(stream))

    provenance = read(SOURCE / 'manifest.json')
    for name in ('measurements.csv', 'method_means.csv', 'v36_parity.json', 'captions.md'):
        path = track(SOURCE / name)
        require(digest(path) == provenance['outputs_sha256'][location(path)],
                f'V39 publication source hash mismatch: {path}')
    rows = read_csv(SOURCE / 'measurements.csv')
    means = read_csv(SOURCE / 'method_means.csv')
    parity = read(SOURCE / 'v36_parity.json')
    expected = {(*condition, method) for condition in CONDITIONS for method in METHODS}
    lookup = {(r['parent'], int(r['hypothesis']), r['method']): r for r in rows}
    require(len(rows) == len(lookup) == 16 and set(lookup) == expected,
            'Require the complete declared 4-condition x 4-method matrix')
    require(len(means) == 4 and {r['method'] for r in means} == set(METHODS), 'Bad method means')
    require(all(r['status'] == 'complete' and r['eligible'] == 'True' for r in rows),
            'Source eligibility/status changed; do not silently drop conditions')
    require(len(parity) == 8 and all(p['all_exact'] and all(p['checks'].values()) for p in parity),
            'Saved V36 parity audit did not pass all eight cases')

    old_seal = read(V36 / 'final_seal.json')
    new_seal = read(V39 / 'final_seal.json')
    old = read(V36 / 'result.json')
    new = read(V39 / 'result.json')
    require(digest(V36 / 'result.json') == old_seal['result.json'], 'V36 result seal mismatch')
    require(digest(V39 / 'result.json') == new_seal['result.json'], 'V39 result seal mismatch')
    require(old['full_matrix_collected'] and old['all_replays_passed'], 'Incomplete V36 result')
    require(new['status'] == 'complete' and new['all_completed_cases_replayed'], 'Incomplete V39 result')
    aggregate = {r['index']: r for r in new['rows']}

    def flatten(measurement):
        result = {'C_map': measurement['C_map']}
        for threshold in (2, 5, 10):
            for short, field in (('P', 'precision'), ('R', 'recall'), ('F', 'f1'), ('J', 'joint')):
                result[f'{short}{threshold}'] = measurement[f'{threshold:02d}cm'][field]
        return result

    for row in rows:
        main_path = ROOT / row['main_source']
        replay_path = ROOT / row['replay_source']
        main, replay = read(main_path), read(replay_path)
        for path in (main_path, replay_path):
            require(digest(path) == new_seal[str(path.relative_to(V39))], f'Seal mismatch: {path}')
            require(digest(path) == provenance['source_sha256'][location(path)], f'Provenance mismatch: {path}')
        require(replay['passed'] and replay['main_result_sha256'] == digest(main_path), 'Replay reference mismatch')
        require(int(row['paid_actions']) == main['paid_actions'] == 42, 'Action budget changed')
        require(row['returned'] == 'True' and main['returned'] and int(row['collisions']) == main['collisions'] == 0,
                'Return/collision contract changed')
        measured = flatten(main['stages']['final']['measurement'])
        for key in METRICS:
            close(row[key], measured[key], f'CSV/main mismatch: {row["index"]} {key}')
            agg_key = 'Q' + key[1:] if key.startswith('F') else key
            close(row[key], aggregate[int(row['index'])][agg_key], f'CSV/aggregate mismatch: {key}')
        for threshold in (2, 5, 10):
            close(row[f'J{threshold}'], float(row['C_map']) * float(row[f'F{threshold}']), 'J product changed')

    v36_rows = []
    for cell in old['table']:
        require(cell['measured'] and cell['independent_replay_passed'], 'V36 case incomplete')
        key = cell['parent'], cell['hypothesis'], cell['mode']
        measured = flatten(cell['stages']['final']['measurement'])
        for metric in METRICS:
            close(measured[metric], lookup[key][metric], f'V36/V39 metric parity mismatch: {key} {metric}')
        v36_rows.append(dict(parent=key[0], hypothesis=key[1], method=key[2],
                             **measured, paid_actions=cell['controller_summary']['actual_paid_actions'],
                             source=location(V36 / 'result.json'), physical_case=cell['physical_case']))
    require(len(v36_rows) == 8 and {(r['parent'], r['hypothesis'], r['method']) for r in v36_rows}
            == {(*c, m) for c in CONDITIONS for m in ('G', 'S')}, 'Incomplete old V36 matrix')

    mean_lookup = {r['method']: r for r in means}
    for method in METHODS:
        mean = mean_lookup[method]
        require(all(int(mean[k]) == 4 for k in ('completed', 'declared', 'eligible')), 'Mean row count changed')
        for metric in ('C_map', 'F2', 'F5', 'F10', 'J2', 'J5', 'J10', 'paid_actions',
                       'forward_m', 'turns', 'planning_s', 'total_s'):
            close(mean[metric], sum(float(lookup[(*c, method)][metric]) for c in CONDITIONS) / 4,
                  f'Four-condition mean mismatch: {method} {metric}')
    for condition in CONDITIONS:
        require(len({lookup[(*condition, method)]['C_map'] for method in METHODS}) == 1,
                'Coverage differs within condition; interpretation must be revised')

    paired = []
    for parent, hypothesis in CONDITIONS:
        g, s = (lookup[(parent, hypothesis, method)] for method in ('G', 'S'))
        paired.append(dict(parent=parent, hypothesis=hypothesis,
                           G_J5=float(g['J5']), S_J5=float(s['J5']),
                           delta_J5=float(s['J5'])-float(g['J5']),
                           delta_C_map=float(s['C_map'])-float(g['C_map']),
                           delta_F5=float(s['F5'])-float(g['F5'])))
    require(sum(r['delta_J5'] == 0 for r in paired) == 2 and sum(r['delta_J5'] > 0 for r in paired) == 2,
            'The four preserved paired effects changed')
    for name in ('V35_ONLINE_SEMANTIC_RESULT_20260918.md', 'V36_ONLINE_CONFIRMATION_RESULT_20260918.md'):
        track(ROOT / 'docs/research' / name)
    track(Path(__file__).resolve())
    return rows, means, v36_rows, paired, tracked


def draw(output, rows, v36_rows, means, paired):
    os.environ.setdefault('MPLBACKEND', 'Agg')
    os.environ.setdefault('MPLCONFIGDIR', '/tmp/nso_virtual_paper_20260928_mpl')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    plt.rcParams.update({'font.family': 'Liberation Sans', 'font.size': 10,
        'axes.labelsize': 10, 'axes.titlesize': 11, 'legend.fontsize': 10,
        'axes.linewidth': .7, 'xtick.major.width': .7, 'ytick.major.width': .7,
        'pdf.fonttype': 42, 'ps.fonttype': 42, 'svg.fonttype': 'none',
        'svg.hashsalt': 'nso-virtual-paper-20260928', 'savefig.facecolor': 'white',
        'mathtext.fontset': 'dejavusans'})
    lookup = {(r['parent'], int(r['hypothesis']), r['method']): r for r in rows}
    old_lookup = {(r['parent'], r['hypothesis'], r['method']): r for r in v36_rows}
    fig = plt.figure(figsize=(12, 5.4), facecolor='white')
    grid = fig.add_gridspec(1, 3, left=.06, right=.985, bottom=.21, top=.765,
                           width_ratios=(1.08, 1.08, 1), wspace=.29)
    ax_a, ax_b = fig.add_subplot(grid[0]), fig.add_subplot(grid[1])
    inner = grid[2].subgridspec(2, 1, hspace=.29)
    ax_cov, ax_f1 = fig.add_subplot(inner[0]), fig.add_subplot(inner[1])

    def style(ax, ylabel, show_x=True):
        ax.spines[['top', 'right']].set_visible(False)
        ax.set_axisbelow(True)
        ax.grid(axis='y', color='#E3E6E9', linewidth=.6)
        ax.set(xlim=(-.5, 3.5), ylim=(0, 1.025), ylabel=ylabel)
        ax.set_yticks((0, .25, .5, .75, 1) if ax in (ax_a, ax_b) else (0, .5, 1))
        ax.set_xticks(range(4), [f'{p}\nh{h}' if show_x else '' for p, h in CONDITIONS])
        ax.tick_params(axis='x', length=0, pad=6, labelsize=9)
        ax.tick_params(axis='y', labelsize=9)
        ax.axvline(1.5, color='#CED3D8', linestyle=(0, (3, 3)), lw=.7, zorder=0)

    style(ax_a, r'Joint score $J_5$')
    for index, (condition, difference) in enumerate(zip(CONDITIONS, paired)):
        for j, method in enumerate(('G', 'S')):
            x, value = index + (j-.5)*.31, old_lookup[(*condition, method)]['J5']
            ax_a.bar(x, value, width=.28, color=COLORS[method], edgecolor='white', linewidth=.4, zorder=2)
            label_offset = .063 if j == 1 and difference['delta_J5'] == 0 else .018
            ax_a.text(x, value+label_offset, f'{value:.3f}', ha='center', va='bottom', fontsize=7.9)
        label = '0' if difference['delta_J5'] == 0 else f'+{difference["delta_J5"]:.4f}'
        ax_a.text(index, .958, label, ha='center', va='center', fontsize=9.1,
                  color='#2F343A', fontweight='bold')
    ax_a.text(.02, .87, r'Labels: $S-G$', transform=ax_a.transAxes, fontsize=8.5)

    def paired_points(ax, metric, ylabel, show_x=True):
        style(ax, ylabel, show_x)
        offsets = (-.255, -.085, .085, .255)
        for i, condition in enumerate(CONDITIONS):
            ax.plot([i+o for o in offsets], [float(lookup[(*condition, m)][metric]) for m in METHODS],
                    color='#AFB6BD', linewidth=.85, zorder=1)
            for offset, method in zip(offsets, METHODS):
                ax.plot(i+offset, float(lookup[(*condition, method)][metric]),
                        marker=MARKERS[method], color=COLORS[method], markersize=5.8,
                        markeredgecolor='white', markeredgewidth=.35, linestyle='none', zorder=3)

    paired_points(ax_b, 'J5', r'Joint score $J_5$')
    paired_points(ax_cov, 'C_map', r'Coverage $C_{map}$', False)
    paired_points(ax_f1, 'F5', r'Surface $F_{1,5cm}$')
    ax_cov.text(.5, .19, 'Identical within every condition', transform=ax_cov.transAxes,
                ha='center', fontsize=8.5, color='#424951')

    for axis, title, subtitle in (
        (ax_a, 'A   Frozen G/S comparison', 'V36: all four paired conditions'),
        (ax_b, 'B   Four CPU mechanisms', 'V39: condition-matched measurements'),
        (ax_cov, 'C   Coverage and quality', 'V39: the factors of the joint score')):
        position = axis.get_position()
        fig.text(position.x0, .845, title, fontsize=11, fontweight='bold', va='bottom')
        fig.text(position.x0, .802, subtitle, fontsize=9, color='#4A5158', va='bottom')
    handles = [Line2D([], [], color=COLORS[m], marker=MARKERS[m], linestyle='none',
                      markersize=6, label=LABELS[m]) for m in METHODS]
    fig.legend(handles=handles, loc='upper center', bbox_to_anchor=(.5, .985),
               ncol=4, columnspacing=2.1, handletextpad=.6, frameon=False)
    mean_lookup = {r['method']: r for r in means}
    relative_gain = (float(mean_lookup['S']['J5'])/float(mean_lookup['G']['J5'])-1)*100
    fig.text(.06, .114,
             f'Mean G/S: {float(mean_lookup["G"]["J5"]):.4f} / {float(mean_lookup["S"]["J5"]):.4f}'
             f'  (+{relative_gain:.2f}%); 2 positive and 2 zero differences.  '
             r'$J_5=C_{map}\times F_{1,5cm}$; all score axes start at zero.', fontsize=9.1)
    fig.text(.06, .064,
             '2 seen development layouts; 4 paired conditions, not independent repeats. '
             '42 paid actions per run; replay does not add samples.', fontsize=9.1)
    fig.text(.06, .019,
             'SWAP-I / VISTA-I: CPU mechanism adaptations, not original-system rankings. '
             'Exact simulated poses; fixed public facility ROI.', fontsize=9.1)

    for extension in ('pdf', 'svg', 'png'):
        metadata = ({'Creator': 'NSO saved-evidence plotter 20260928', 'CreationDate': None, 'ModDate': None}
                    if extension == 'pdf' else {'Date': None} if extension == 'svg' else None)
        fig.savefig(output / f'main_evidence.{extension}', dpi=300, metadata=metadata)
    plt.close(fig)
    return matplotlib.__version__


def write_csv(path, rows):
    with path.open('w', newline='', encoding='utf-8') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator='\n')
        writer.writeheader()
        writer.writerows(rows)


def run(output):
    output = output.resolve()
    require(not output.exists(), f'Output exists; choose a new directory instead: {output}')
    rows, means, old, paired, tracked = load_inputs()
    output.mkdir(parents=True)
    for name in ('measurements.csv', 'method_means.csv'):
        shutil.copyfile(SOURCE / name, output / f'v39_{name}')
    write_csv(output / 'v36_measurements.csv', old)
    write_csv(output / 'paired_effects.csv', paired)
    version = draw(output, rows, old, means, paired)
    caption = """# Main paper figure: controlled virtual evidence

**English caption.** (A) All four saved V36 layout–configuration pairs for active
geometry G and semantic policy S; two zero effects and two positive effects are
retained. Labels above the bars are the signed within-condition S−G differences
in J5. (B) The same four conditions for G, S, SWAP-inspection-inspired/CPU
(SWAP-I), and VISTA-view-inspired/CPU (VISTA-I), under the shared 42-paid-action
budget (including an 18-action prefix). Grey segments join methods within each
condition; they are not interpolation or uncertainty intervals. These are CPU
mechanism adaptations, not reproductions or rankings of the original systems.
The eight G/S score rows in B match A and therefore provide no additional
independent evidence. (C) Measured map coverage and surface F1 at the fixed 5 cm
threshold are plotted on separate axes. Coverage is identical across methods
within each condition, so the observed within-condition J5 differences arise
from F1, not increased map coverage. J5 = C_map × F1@5cm is the existing saved
metric; no J_nav or new score was constructed. Every absolute score axis starts
at zero. The common sensor model, public safe graph, CPU TSDF, evaluator, exact
simulated poses and fixed public facility ROI constrain interpretation.

The evidence covers **2 seen development layouts and 4 paired conditions,
not independent repeats**. Deterministic replay verifies execution and does
not increase sample size. Condition means are descriptive with equal weight
per condition (equivalently equal weight per layout here). No confidence
intervals, significance claim, unseen-layout generalization or original-system
ranking is asserted. Public-ROI-external errors are excluded from precision.

**中文图注。** (A) V36已保存的四个布局—构型配对条件，完整保留两个零增益和
两个正增益，柱顶标注各条件的S−G联合指标差。(B) G、S、SWAP-I和VISTA-I在同样
四个条件下的终点J5，灰线只连接同条件的各方法结果；后二者是CPU机制适配，不代表
原作者完整系统排名。B中的G/S与A为相同数值，不增加独立证据。(C) 地图覆盖率与
5 cm表面F1分轴呈现；每个条件各方法覆盖相同，联合指标差来自表面F1。
全部结果均为既有42付费动作预算（含18动作共同前缀）下的保存评分。
仅涉及2个已见开发布局／4个配对条件，不是独立重复；回放不增加样本量。
人工类别、精确模拟位姿、固定公共ROI和公共安全图仍为适用条件，不作显著性或未知场景泛化宣称。

## Source and scope

- A: `audit_results/v36_online_confirmation_20260918/result.json`, complete 8-row G/S matrix.
- B/C: `docs/thesis/figures/v39_external/measurements.csv`, complete 16-row four-method matrix.
- Means: existing `method_means.csv`, checked against all four conditions.
- The saved V36 parity audit reports exact measurements/actions/arrays; this
  plotting run rechecks saved metric equality and the referenced JSON/CSV hashes,
  not raw sensor arrays or the entire original experiment inventory.
- V35 supplies separate development evidence about misleading-prior correction;
  it is not pooled into these V36/V39 scores. Its h1/2 cm X−Xnf negative result
  remains in `docs/research/V35_ONLINE_SEMANTIC_RESULT_20260918.md` and must remain
  visible wherever feedback robustness is discussed. This figure itself is not
  a feedback ablation.
- Saved 2/10 cm secondary-threshold P/R/F/J values are retained in the exported
  full source CSVs. This main figure uses the previously specified 5 cm metric.
- No new World, sensor packet, controller/planner call, TSDF integration,
  surface evaluation, physical run, ROI selection or case selection.

Reproduce from the repository root with
`python3 scripts/plot_virtual_paper_evidence_20260928.py --output NEW_DIRECTORY`.
Existing output directories are never overwritten. `manifest.json` records
SHA256 for the inputs actually checked and every output except itself.
"""
    (output / 'captions.md').write_text(caption, encoding='utf-8')
    for name, expected in tracked.items():
        require(digest(ROOT / name) == expected, f'Input changed while rendering: {name}')
    manifest = dict(
        scope='Saved V36/V39 controlled virtual evidence, presentation only',
        input_sha256=tracked,
        output_sha256={p.name: digest(p) for p in sorted(output.iterdir()) if p.is_file()},
        figure='main_evidence', matplotlib_version=version,
        source_rows=dict(V36=8, V39=16, V39_method_means=4),
        layouts=['P00', 'P01'], conditions=[dict(parent=p, hypothesis=h) for p, h in CONDITIONS],
        seen_development_layouts=2, condition_count=4,
        conditions_are_independent_repeats=False, replay_adds_samples=False,
        V39_G_S_duplicates_V36_metrics=True, all_conditions_retained=True,
        positive_G_S_conditions=2, zero_G_S_conditions=2,
        same_coverage_within_each_condition=True,
        saved_primary_metric='J5 = C_map * surface F1@5cm',
        confidence_intervals_computed=False, original_system_ranking=False,
        checked='Source CSV/JSON hashes, complete matrices, all saved threshold metrics, means and G/S metric parity; no raw-array reaudit',
        new_worlds=0, new_main_tasks=0, new_sensor_packets=0, new_controller_calls=0,
        new_planner_calls=0, new_TSDF_integrations=0, new_surface_evaluations=0,
        output_size_limit_bytes=15*1024*1024)
    (output / 'manifest.json').write_text(json.dumps(manifest, indent=2, ensure_ascii=False)+'\n', encoding='utf-8')
    size = sum(p.stat().st_size for p in output.iterdir() if p.is_file())
    require(size < 15*1024*1024, 'Figure package exceeds the 15 MiB limit')
    print(json.dumps(dict(output=str(output), bytes=size, files=len(list(output.iterdir())),
                          matrix_rows=len(rows), G_S_pairs=paired), indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=DEFAULT_OUTPUT)
    run(parser.parse_args().output)
