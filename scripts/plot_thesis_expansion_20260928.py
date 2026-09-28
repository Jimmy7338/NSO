#!/usr/bin/env python3
"""Plot the complete, sealed 128-case thesis expansion without new evaluation.

Formal figures require analysis_review/seal.json. Failed partial-map scores never
enter successful-performance means. Repeated noise seeds are plotted as repeats,
not treated as independently sampled scenes or as a basis for confidence bounds.
"""
from __future__ import annotations

import csv
import hashlib
import json
import os
from pathlib import Path
import shutil

os.environ.setdefault('MPLCONFIGDIR', '/tmp/nso-thesis-expansion-plots')
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / 'audit_results/thesis_expansion_20260928'
REVIEW = BASE / 'analysis_review'
OUT = ROOT / 'docs/thesis/figures/expansion_20260928'
CONFIG = ROOT / 'configs/virtual3d/thesis_expansion_20260928.json'
COLORS = {'G': '#0072B2', 'S': '#D55E00', 'X': '#009E73', 'Xnf': '#CC79A7'}
MARKERS = {'G': 'o', 'S': 's', 'X': '^', 'Xnf': 'v'}
MODES = ('G', 'S', 'X', 'Xnf')
CONTRASTS = (('S', 'G'), ('X', 'Xnf'), ('X', 'G'))
CONTRAST_COLORS = {('S', 'G'): COLORS['S'], ('X', 'Xnf'): COLORS['X'], ('X', 'G'): '#737373'}
CONTRAST_MARKERS = {('S', 'G'): 'o', ('X', 'Xnf'): 's', ('X', 'G'): '^'}
SCENES = ('P00', 'P01', 'P00_rib_shift', 'P01_lower_ribs')
THRESHOLDS = ('02cm', '05cm', '10cm')
BUDGETS = (30, 42, 54)
SOURCES, OUTPUTS, PLOTTED = {}, [], {}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def source(path):
    path = Path(path)
    SOURCES[str(path.relative_to(ROOT))] = {'sha256': sha(path), 'bytes': path.stat().st_size}
    return path


def read_json(path):
    return json.loads(source(path).read_text())


def boolean(value):
    if value not in ('True', 'False'):
        raise ValueError('Unexpected CSV boolean: '+repr(value))
    return value == 'True'


def numeric(value):
    return None if value in ('', 'None') else float(value)


def load_data():
    if not (REVIEW / 'seal.json').exists():
        raise RuntimeError('Final sealed analysis_review is not ready; no formal plots generated.')
    seal = read_json(REVIEW / 'seal.json')
    for relative, expected in seal.items():
        if sha(REVIEW / relative) != expected:
            raise ValueError('Review seal mismatch: '+relative)
    config = read_json(CONFIG)
    summary = read_json(REVIEW / 'summary.json')
    review_freeze = read_json(REVIEW / 'analysis_freeze.json')
    acquisition_freeze = read_json(BASE / 'source_freeze.json')
    assert acquisition_freeze['config_sha256'] == sha(CONFIG)
    assert review_freeze['source_freeze_sha256'] == sha(BASE / 'source_freeze.json')
    assert review_freeze['script_sha256'] == sha(ROOT / 'scripts/summarize_thesis_expansion_20260928.py')
    for case_id, expected_seal in review_freeze['acquisition_inputs'].items():
        assert sha(BASE / 'cases' / case_id / 'seal.json') == expected_seal
    source(ROOT / 'scripts/run_thesis_expansion_20260928.py')
    source(ROOT / 'scripts/summarize_thesis_expansion_20260928.py')
    rows = list(csv.DictReader(source(REVIEW / 'complete_endpoint_metrics.csv').open()))
    pairs = list(csv.DictReader(source(REVIEW / 'complete_paired_metrics.csv').open()))
    expected = {c['id']: c for c in config['cases']}
    assert len(rows) == len(expected) == config['declared_cases'] == 128
    assert len({r['id'] for r in rows}) == 128
    assert {r['id'] for r in rows} == set(expected)
    for row in rows:
        for key in ('budget', 'hypothesis', 'noise_seed', 'paid_actions'):
            row[key] = int(row[key])
        row['qualified'] = boolean(row['qualified'])
        for threshold in THRESHOLDS:
            for field in ('precision', 'recall', 'f1', 'joint'):
                name = f'{threshold}_{field}'; row[name] = numeric(row[name])
        for key in ('scene', 'hypothesis', 'budget', 'noise_seed', 'mode'):
            assert row[key] == expected[row['id']][key]
        if row['qualified']:
            assert row['status'] == 'completed' and row['measurement_scope'] == 'completed_online_endpoint'
            assert row['05cm_joint'] is not None
    keys = {(r['scene'], r['hypothesis'], r['budget'], r['noise_seed'], r['mode']): r for r in rows}
    assert len(pairs) == 96
    for pair in pairs:
        for key in ('budget', 'hypothesis', 'noise_seed'):
            pair[key] = int(pair[key])
        pair['both_qualified'] = boolean(pair['both_qualified'])
        key = (pair['scene'], pair['hypothesis'], pair['budget'], pair['noise_seed'])
        treatment, control = keys[key+(pair['treatment'],)], keys[key+(pair['control'],)]
        assert pair['both_qualified'] == (treatment['qualified'] and control['qualified'])
        for threshold in THRESHOLDS:
            name = threshold+'_delta'; pair[name] = numeric(pair[name])
            a, b = treatment[threshold+'_joint'], control[threshold+'_joint']
            if a is not None and b is not None:
                assert abs(pair[name] - (a-b)) < 1e-12
        for name in ('J5_treatment', 'J5_control'):
            pair[name] = numeric(pair[name])
    assert sum(r['qualified'] for r in rows) == summary['qualified']
    assert sum(r['status'] == 'completed' for r in rows) == summary['completed']
    return config, summary, rows, pairs


def save(fig, stem):
    for ext in ('pdf', 'png', 'svg'):
        path = OUT / f'{stem}.{ext}'
        metadata = {'CreationDate': None, 'ModDate': None} if ext == 'pdf' else {'Date': None} if ext == 'svg' else None
        fig.savefig(path, dpi=300, facecolor='white', metadata=metadata)
        OUTPUTS.append(path)
    plt.close(fig)


def select(rows, **kwargs):
    return [r for r in rows if all(r[k] == v for k, v in kwargs.items())]


def mean_if_complete(group, field, qualification):
    if not group or not all(r[qualification] and r[field] is not None for r in group):
        return None
    return float(np.mean([r[field] for r in group]))


def pair_summary(group, threshold='05cm'):
    values = [r[threshold+'_delta'] for r in group if r['both_qualified']]
    return {'declared': len(group), 'qualified': len(values),
            'mean': mean_if_complete(group, threshold+'_delta', 'both_qualified'),
            'qualified_values': values,
            'wins': sum(v > 1e-12 for v in values), 'ties': sum(abs(v) <= 1e-12 for v in values),
            'losses': sum(v < -1e-12 for v in values)}


def style_axis(ax):
    ax.spines[['top', 'right']].set_visible(False)
    ax.tick_params(length=3)
    ax.grid(axis='y', color='#E6E9EB', lw=.55)
    ax.set_axisbelow(True)


def contrast_label(pair):
    return f'{pair[0]} - {pair[1]}'


def budget_figure(rows, pairs):
    original = [r for r in rows if r['scene'] in ('P00', 'P01')]
    original_pairs = [r for r in pairs if r['scene'] in ('P00', 'P01')]
    fig = plt.figure(figsize=(7.4, 6.05))
    axes = [fig.add_axes([.09, .42, .385, .40]), fig.add_axes([.585, .42, .385, .40])]
    fig.text(.07, .963, 'Budget and category-feedback sensitivity', fontsize=12, weight='bold')
    fig.text(.07, .925, 'Original P00/P01 layouts • both configurations • two depth-noise repeats', fontsize=8, color='#4C575D')
    main_records = []
    for mode in MODES:
        means = []
        for budget in BUDGETS:
            group = select(original, budget=budget, mode=mode)
            mean = mean_if_complete(group, '05cm_joint', 'qualified')
            means.append(np.nan if mean is None else mean)
            main_records.append({'budget': budget, 'mode': mode, 'declared': len(group),
                                 'qualified': sum(r['qualified'] for r in group), 'mean_J5': mean})
        axes[0].plot(range(3), means, color=COLORS[mode], marker=MARKERS[mode], lw=1.5,
                     ms=5, ls='--' if mode in ('G', 'Xnf') else '-', label=mode)
    axes[0].set(xlim=(-.40, 2.35), ylim=(0, 1), xticks=range(3), xticklabels=BUDGETS,
                ylabel='Mean completed $J_5$', xlabel='Action budget $B$')
    axes[0].set_title('(a) Absolute endpoint quality', fontsize=9, loc='left', pad=8)
    axes[0].legend(frameon=False, ncols=2, fontsize=7, loc='lower right', columnspacing=1)
    pair_records = []
    all_deltas = []
    for ci, contrast in enumerate(CONTRASTS):
        color = CONTRAST_COLORS[contrast]
        for bi, budget in enumerate(BUDGETS):
            group = sorted(select(original_pairs, budget=budget, treatment=contrast[0], control=contrast[1]),
                           key=lambda r: (r['scene'], r['hypothesis'], r['noise_seed']))
            summary = pair_summary(group)
            pair_records.append(dict(budget=budget, treatment=contrast[0], control=contrast[1], **summary))
            x = bi + (ci-1)*.16
            qualified = [r for r in group if r['both_qualified']]
            jitter = np.linspace(-.035, .035, len(qualified))
            values = [r['05cm_delta'] for r in qualified]
            all_deltas.extend(values)
            axes[1].scatter(x+jitter, values, color=color, s=13, alpha=.55,
                            marker=CONTRAST_MARKERS[contrast], linewidth=0, zorder=4)
            if summary['mean'] is not None:
                axes[1].plot([x-.065, x+.065], [summary['mean']]*2, color=color, lw=2.5, zorder=5)
    axes[1].axhline(0, color='#47565D', lw=.8)
    lo, hi = min([0]+all_deltas), max([0]+all_deltas)
    margin = max(.018, (hi-lo)*.18)
    axes[1].set(xlim=(-.40, 2.35), ylim=(lo-margin, hi+margin), xticks=range(3), xticklabels=BUDGETS,
                ylabel=r'Paired $\Delta J_5$', xlabel='Action budget $B$')
    axes[1].set_title('(b) Paired endpoint differences', fontsize=9, loc='left', pad=8)
    handles = [Line2D([0], [0], color=CONTRAST_COLORS[c], marker=CONTRAST_MARKERS[c], lw=0,
                       markersize=4, label=contrast_label(c)) for c in CONTRASTS]
    axes[1].legend(handles=handles, frameon=False, fontsize=7, loc='upper right', handletextpad=.4)
    for ax in axes:
        style_axis(ax)
        for bi, budget in enumerate(BUDGETS):
            group = select(original, budget=budget)
            if not any(r['qualified'] for r in group):
                ax.axvspan(bi-.33, bi+.33, facecolor='#F0F1F2', hatch='///', edgecolor='#D6DBDE', lw=0, zorder=0)
                ax.text(bi, .49 if ax is axes[0] else .65, 'No qualified\nendpoints',
                        transform=ax.get_xaxis_transform(), ha='center', va='center', fontsize=7, color='#5A6268')
    table_ax = fig.add_axes([.09, .105, .88, .225]); table_ax.set_axis_off()
    table_ax.text(0, 1.04, '(c) Endpoint qualification (qualified / declared)', fontsize=9, weight='bold')
    cell_text = []
    for mode in MODES:
        cell_text.append([mode]+[f"{sum(r['qualified'] for r in select(original, budget=b, mode=mode))} / {len(select(original, budget=b, mode=mode))}" for b in BUDGETS])
    table = table_ax.table(cellText=cell_text, colLabels=['Method']+[f'B = {b}' for b in BUDGETS],
                          cellLoc='center', loc='center', bbox=[0, -.03, 1, .94])
    table.auto_set_font_size(False); table.set_fontsize(8)
    for (ri, ci), cell in table.get_celld().items():
        cell.set_edgecolor('#DDE2E5'); cell.set_linewidth(.5)
        cell.set_facecolor('#F1F4F5' if ri == 0 else 'white')
        if ci == 0 and ri > 0:
            cell.get_text().set_color(COLORS[MODES[ri-1]]); cell.get_text().set_weight('bold')
    fig.text(.09, .035, 'Points are paired noise repeats; short bars are means. Failed partial maps are excluded from completed quality.',
             fontsize=6.8, color='#4C575D')
    PLOTTED['budget_endpoint_means'] = main_records
    PLOTTED['budget_paired_differences'] = pair_records
    save(fig, 'budget_effects')


def variant_figure(rows, pairs):
    budget = 42
    fig, axes = plt.subplots(2, 1, figsize=(7.4, 6.4))
    fig.subplots_adjust(left=.105, right=.98, top=.82, bottom=.12, hspace=.56)
    fig.text(.075, .963, 'Geometry perturbations and surface-distance thresholds', fontsize=11.4, weight='bold')
    fig.text(.075, .928, 'Budget B = 42 • original layouts and two declared variants from the same scene family', fontsize=8, color='#4C575D')
    contrast_handles = [Line2D([0], [0], color=CONTRAST_COLORS[c], marker=CONTRAST_MARKERS[c], lw=1.5,
                               markersize=4, label=contrast_label(c)) for c in CONTRASTS]
    fig.legend(handles=contrast_handles, frameon=False, ncols=3, loc='upper center', bbox_to_anchor=(.52, .905),
               fontsize=8, columnspacing=2)
    records = []
    axes[0].axhline(0, color='#47565D', lw=.8)
    for si, scene in enumerate(SCENES):
        for ci, contrast in enumerate(CONTRASTS):
            group = sorted(select(pairs, scene=scene, budget=budget, treatment=contrast[0], control=contrast[1]),
                           key=lambda r: (r['hypothesis'], r['noise_seed']))
            summary = pair_summary(group)
            records.append(dict(scene=scene, budget=budget, treatment=contrast[0], control=contrast[1], **summary))
            x = si+(ci-1)*.19
            qualified = [r for r in group if r['both_qualified']]
            jitter = np.linspace(-.045, .045, len(qualified))
            for shift, row in zip(jitter, qualified):
                # Filled h1, open h0; different noise repeats remain individual points.
                axes[0].scatter(x+shift, row['05cm_delta'], s=25, marker=CONTRAST_MARKERS[contrast],
                    facecolors=CONTRAST_COLORS[contrast] if row['hypothesis'] else 'white',
                    edgecolors=CONTRAST_COLORS[contrast], linewidth=.8, zorder=4)
            if summary['mean'] is not None:
                axes[0].plot([x-.070, x+.070], [summary['mean']]*2, color=CONTRAST_COLORS[contrast], lw=2.3, zorder=5)
            if summary['qualified'] != summary['declared']:
                axes[0].text(x, .02, f"{summary['qualified']}/{summary['declared']}",
                             transform=axes[0].get_xaxis_transform(), ha='center', fontsize=6)
    axes[0].axvline(1.5, color='#A8B2B8', ls=':', lw=.8)
    axes[0].set(xlim=(-.5, 3.5), xticks=range(4),
                xticklabels=['P00\noriginal', 'P01\noriginal', 'P00\nrib shifted', 'P01\nlower ribs'],
                ylabel=r'Paired $\Delta J_5$')
    axes[0].set_title('(a) Every paired result at the 5 cm threshold', loc='left', fontsize=9, pad=8)
    axes[0].text(.99, .97, 'Open: h0   Filled: h1\n2 noise repeats per configuration',
                 transform=axes[0].transAxes, ha='right', va='top', fontsize=7,
                 bbox=dict(facecolor='white', edgecolor='none', alpha=.9, pad=2))
    axes[1].axhline(0, color='#47565D', lw=.8)
    thresholds = []
    families = [('originals', ('P00', 'P01'), '-'), ('same_family_variants', ('P00_rib_shift', 'P01_lower_ribs'), '--')]
    for family, scenes, linestyle in families:
        for contrast in CONTRASTS:
            group = [r for r in pairs if r['scene'] in scenes and r['budget'] == budget and
                     (r['treatment'], r['control']) == contrast]
            means = []
            for threshold in THRESHOLDS:
                info = pair_summary(group, threshold)
                thresholds.append(dict(family=family, budget=budget, threshold=threshold,
                                       treatment=contrast[0], control=contrast[1], **info))
                means.append(np.nan if info['mean'] is None else info['mean'])
            axes[1].plot([2, 5, 10], means, color=CONTRAST_COLORS[contrast], ls=linestyle,
                          marker=CONTRAST_MARKERS[contrast], markerfacecolor='white' if linestyle=='--' else CONTRAST_COLORS[contrast],
                          ms=4.5, lw=1.5)
    axes[1].set(xticks=[2, 5, 10], xlabel='Surface-distance threshold (cm)', ylabel=r'Mean paired $\Delta J_\tau$')
    axes[1].set_title('(b) Complete signed threshold summaries', fontsize=9, loc='left', pad=8)
    family_handles = [Line2D([0], [0], color='#33414A', ls=ls, lw=1.5,
                            label='Original layouts' if family=='originals' else 'Same-family variants') for family, _, ls in families]
    axes[1].legend(handles=family_handles, frameon=False, fontsize=7, loc='best')
    for ax in axes:
        style_axis(ax)
        lo, hi = ax.get_ylim(); pad = max(.018, (hi-lo)*.1)
        ax.set_ylim(min(lo, -pad), hi+pad)
    fig.text(.105, .027, 'Both configurations, both noise repeats and all three thresholds are retained. No independent-scene confidence intervals.',
             fontsize=6.8, color='#4C575D')
    PLOTTED['B42_scene_pair_differences'] = records
    PLOTTED['B42_threshold_means'] = thresholds
    save(fig, 'scene_and_threshold_effects')


def main():
    if shutil.disk_usage(ROOT).free < 128*1024**2:
        raise RuntimeError('128 MiB free-space reserve required')
    config, summary, rows, pairs = load_data()
    OUT.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({'font.family': 'Liberation Sans', 'font.size': 8,
        'axes.linewidth': .6, 'xtick.labelsize': 7, 'ytick.labelsize': 7,
        'pdf.fonttype': 42, 'svg.fonttype': 'none', 'svg.hashsalt': 'thesis-expansion-20260928'})
    budget_figure(rows, pairs)
    variant_figure(rows, pairs)
    captions = {
        'budget_effects': {
            'caption_zh': '原始P00/P01布局的预算与类别反馈扩展，共96个预先声明任务。(a) G、S、X和Xnf在不同动作预算下的终点联合质量。仅当对应8个声明任务全部合格时显示组均值；失败任务的部分地图得分不作为完成性能。(b) 相同布局、构型和噪声种子的配对差，保留零值及负值；小点是噪声重复，短横线为完整配对均值，不表示独立场景置信区间。(c) 每个方法与预算下的合格/声明任务数。G为共同几何规划，S为正确类别加几何反馈，X为交换类别加反馈，Xnf为交换类别并去除实际反馈与未来诊断预测的整体策略对照。',
            'caption_en': 'Budget and category-feedback expansion on the original P00/P01 layouts (96 declared trials). (a) Completed endpoint joint quality; a method-budget mean is shown only when all eight declared members qualify. Partial maps from failed episodes are not successful performance. (b) Paired contrasts matched by layout, configuration and noise seed, with zero and negative effects retained. Points are repeated noise conditions and short bars denote complete-pair means, not confidence intervals over independent scenes. (c) Qualified/declared counts. Xnf removes both observed geometric feedback and future diagnostic forecasting from the swapped-category policy.',
        },
        'scene_and_threshold_effects': {
            'caption_zh': '预算42步下的原始布局与同族几何扰动比较。(a) P00、P01及侧部位置移动和侧部高度降低两变体的全部配对J5差值；空心/实心分别表示h0/h1，每构型保留两个噪声重复，短横线为配对均值。(b) 原始布局与两变体分别等权汇总的2/5/10厘米完整阈值结果。所有持平及负向结果均保留。变体沿用已声明的同族结构和公开候选模板，不属于未知类别或独立场景泛化验证；重复种子不计为独立场景样本。',
            'caption_en': 'Original layouts and same-family geometry perturbations at B=42. (a) All paired J5 differences for P00, P01 and the rib-shift/lower-rib variants. Open/filled points denote h0/h1, with both noise repeats retained; short bars are paired means. (b) Signed, equally weighted contrasts at every declared 2/5/10 cm threshold, separately for originals and variants. Zero and negative effects are retained. Variants share the declared scene family and public candidate-template setting; neither unseen-category generalization nor independent-scene significance is inferred.'
        }
    }
    for name, payload in [('captions.json', captions), ('plotted_values.json', PLOTTED)]:
        path = OUT / name; path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False)+'\n'); OUTPUTS.append(path)
    source(Path(__file__))
    assert all(sha(ROOT/p) == entry['sha256'] for p, entry in SOURCES.items())
    provenance = {'version': 'thesis-expansion-figures-20260928-v1', 'sources': SOURCES,
        'outputs': {str(p.relative_to(ROOT)): {'sha256': sha(p), 'bytes': p.stat().st_size} for p in OUTPUTS},
        'declared_cases': len(rows), 'declared_pair_contrasts': len(pairs),
        'qualified': summary['qualified'], 'completed': summary['completed'], 'failed': summary['failed'],
        'method_colors': COLORS, 'raster_dpi': 300,
        'scope': {'formal_analysis_seal_verified': True, 'complete_declared_matrix_verified': True,
            'all_pair_delta_arithmetic_verified': True, 'failed_partial_scores_used_as_success': False,
            'complete_group_qualification_required_for_mean': True,
            'independent_scene_confidence_intervals': False, 'noise_seeds_counted_as_new_scenes': False,
            'analytic_category_mixture_plotted': False, 'new_quality_evaluations': 0, 'new_experiments': 0}}
    (OUT/'provenance.json').write_text(json.dumps(provenance, ensure_ascii=False, indent=2, allow_nan=False)+'\n')
    total = sum(p.stat().st_size for p in OUT.iterdir() if p.is_file())
    assert total < 10*1024**2
    print(json.dumps({'output': str(OUT), 'bytes': total, 'declared': len(rows),
                      'qualified': summary['qualified'], 'source_hashes': len(SOURCES)}, indent=2))


if __name__ == '__main__':
    main()
