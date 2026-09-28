#!/usr/bin/env python3
"""Static paper figures from an independently reviewed comparison snapshot.

No simulator, controller, TSDF integration or new quality measurements are run.
Missing/failed declared slots remain visible. Development is kept separate from
main confirmation and plotted at the original layout and budget resolution.
"""
import argparse
import csv
import hashlib
import io
import json
import os
from pathlib import Path
import sys

os.environ.setdefault('MPLCONFIGDIR', '/tmp/nso-article-online-results-mpl')
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
os.environ.setdefault('OMP_NUM_THREADS', '1')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import plot_article_scene_overview_20260928 as scene_plot

METHODS = ('NBV', 'G', 'B', 'S')
COLORS = {'NBV': '#7B8794', 'G': '#4477AA', 'B': '#CC9933', 'S': '#228833'}
INK = '#253641'
SOURCES = {}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def source(path):
    path = Path(path).resolve()
    key = str(path.relative_to(ROOT)) if path.is_relative_to(ROOT) else str(path)
    SOURCES[key] = dict(bytes=path.stat().st_size, sha256=sha(path))
    return path


def read(path):
    return json.loads(source(path).read_text())


def safe_child(root, name):
    p = Path(name)
    if p.is_absolute() or '..' in p.parts or not p.parts:
        raise ValueError('relative manifest member required')
    target = root / p
    if target.is_symlink() or not target.resolve().is_relative_to(root.resolve()):
        raise ValueError('manifest member escapes input root')
    return target


def verify(root):
    manifest = read(root / 'manifest.json')
    for name, record in manifest['files'].items():
        path = safe_child(root, name)
        if path.stat().st_size != record['bytes'] or sha(path) != record['sha256']:
            raise ValueError('input manifest mismatch: ' + str(path))
        source(path)
    return manifest


def trajectory(review_dir, run):
    folder = review_dir / run['run_id']
    verify(folder)
    review = read(folder / 'review.json')
    if not review['all_checks_passed'] or not review['qualified']:
        raise ValueError('figure input is not an independently qualified episode')
    if review['run_id'] != run['run_id'] or review['protocol_sha256'] != run['protocol_sha256']:
        raise ValueError('review is bound to a different episode')
    path = source(folder / 'trajectory.csv')
    with path.open(newline='') as f:
        rows = list(csv.DictReader(f))
    if len(rows) != run['paid_actions'] + 1:
        raise ValueError('trajectory count differs from reviewed paid actions')
    return rows, review


def save(fig, output, name, exports):
    for ext in ('pdf', 'svg', 'png'):
        path = output / (name + '.' + ext)
        metadata = ({'CreationDate': None, 'ModDate': None} if ext == 'pdf'
                    else {'Date': None} if ext == 'svg' else None)
        fig.savefig(path, dpi=230, bbox_inches='tight', facecolor='white', metadata=metadata)
        exports.append(path.name)
    plt.close(fig)


def route_panels(slots, records, review_dir, output, exports):
    groups = sorted({(r['phase'], r['scene_id'], r['budget'], r['noise_seed']) for r in slots})
    # One page per three matched conditions avoids shrinking a 96-run matrix.
    for page in range(0, len(groups), 3):
        subset = groups[page:page+3]
        fig, axes = plt.subplots(len(subset), 4, figsize=(13.6, 3.45*len(subset)), squeeze=False)
        for row_index, group in enumerate(subset):
            phase, scene_id, budget, noise = group
            for column, method in enumerate(METHODS):
                ax = axes[row_index, column]
                scene_plot.plan_map(ax, records[scene_id], show_axes=False)
                found = [r for r in slots if (r['phase'], r['scene_id'], r['budget'], r['noise_seed']) == group and r['method'] == method]
                if len(found) != 1:
                    raise ValueError('one declared slot per method and condition required')
                run = found[0]
                if run['status'] == 'reviewed_qualified':
                    rows, _ = trajectory(review_dir, run)
                    xy = np.array([[float(r['x_m']), float(r['y_m'])] for r in rows])
                    yaw = np.array([float(r['yaw_rad']) for r in rows])
                    color = COLORS[method]
                    ax.plot(xy[:, 0], xy[:, 1], color=color, lw=1.5, alpha=.95, zorder=10)
                    turns = [i for i, r in enumerate(rows) if r['sensor_action'] in ('left', 'right')]
                    observes = [i for i, r in enumerate(rows) if r['sensor_action'] == 'observe']
                    if turns:
                        ax.scatter(xy[turns, 0], xy[turns, 1], s=12, facecolors='none', edgecolors=color, lw=.65, zorder=11)
                    if observes:
                        ax.scatter(xy[observes, 0], xy[observes, 1], s=15, marker='s', c=color, edgecolors='white', lw=.3, zorder=12)
                    # Fixed temporal sampling of actual headings, no path smoothing.
                    sample = np.arange(0, len(rows), 16)
                    ax.quiver(xy[sample, 0], xy[sample, 1], .35*np.cos(yaw[sample]), .35*np.sin(yaw[sample]),
                              angles='xy', scale_units='xy', scale=1, width=.007, color=color, zorder=13)
                    ax.scatter(*xy[0], s=35, marker='D', c='white', edgecolors=INK, linewidth=.9, zorder=14)
                    detail = f"J = {run['J_nav']:.3f}  |  {run['path_length_m']:.1f} m  |  {run['turns']} turns"
                else:
                    ax.text(4., 4., run['status'].replace('_', '\n'), ha='center', va='center', color=INK,
                            fontsize=10, bbox=dict(facecolor='white', alpha=.93, edgecolor='#CBD1D5'), zorder=20)
                    detail = 'No qualified measured endpoint'
                if row_index == 0:
                    ax.set_title(method, color=COLORS[method], weight='bold', fontsize=13, pad=8)
                ax.text(.5, -.105, detail, transform=ax.transAxes, ha='center', fontsize=8, color=INK)
                if column == 0:
                    ax.set_ylabel(f"{scene_id.replace('ART1_', '')}\nB = {budget}, seed {noise}\n$y$ (m)", fontsize=9)
                if row_index == len(subset)-1:
                    ax.set_xlabel('$x$ (m)', fontsize=9)
        fig.legend(handles=[Line2D([], [], color=INK, lw=1.4, label='Executed route'),
                            Line2D([], [], color=INK, marker='o', markerfacecolor='none', lw=0, label='Turn position'),
                            Line2D([], [], color=INK, marker='s', lw=0, label='Extra paid observation'),
                            Line2D([], [], color=INK, marker='D', markerfacecolor='white', lw=0, label='Start / full-pose return')],
                   loc='upper center', bbox_to_anchor=(.5, 1.025), frameon=False, ncol=4, fontsize=9)
        fig.subplots_adjust(wspace=.15, hspace=.28)
        save(fig, output, f'online_routes_{page//3+1:02d}', exports)


def result_panels(slots, output, exports):
    groups = sorted({(r['phase'], r['scene_id'], r['budget'], r['noise_seed']) for r in slots})
    if len(groups) > 6:
        return  # Main matrix uses separate layout/seed paired-summary analysis.
    metrics = [('C_nav', 'Observed floor coverage'), ('F1', 'Macro surface F1'),
               ('J_nav', 'Joint documentation score'), ('path_length_m', 'Executed distance (m)'),
               ('turns', 'Paid turn actions'), ('planning_s', 'Planning wall time (s)')]
    fig, axes = plt.subplots(2, 3, figsize=(12., 6.8))
    x = np.arange(len(groups)); offsets = np.linspace(-.27, .27, 4)
    for ax, (key, label) in zip(axes.flat, metrics):
        for index, method in enumerate(METHODS):
            values = []
            for group in groups:
                found = [r for r in slots if (r['phase'], r['scene_id'], r['budget'], r['noise_seed']) == group and r['method'] == method]
                if len(found) != 1:
                    raise ValueError('incomplete declared method inventory')
                values.append(found[0].get(key) if found[0]['status'] == 'reviewed_qualified' else np.nan)
            values = np.asarray(values, dtype=float)
            ax.bar(x+offsets[index], values, width=.16, color=COLORS[method], label=method, zorder=3)
            for xx, value in zip(x+offsets[index], values):
                if not np.isfinite(value):
                    ax.plot(xx, 0, marker='x', color=COLORS[method], clip_on=False, zorder=4)
        ax.set_title(label, fontsize=10, loc='left', color=INK)
        ax.set_xticks(x, [g[1].replace('ART1_', '').replace('_DEV', '')+f'\nB{g[2]}/n{g[3]}' for g in groups], fontsize=8)
        ax.set_ylim(bottom=0)
        if key in ('C_nav', 'F1', 'J_nav'):
            ax.set_ylim(0, 1.04)
        ax.spines[['top', 'right']].set_visible(False)
        ax.spines[['bottom', 'left']].set_color('#B9C0C5')
        ax.tick_params(colors=INK, labelsize=8)
        ax.grid(axis='y', color='#E4E8EB', lw=.65, zorder=0)
    axes[0, 0].legend(ncol=4, loc='lower left', bbox_to_anchor=(0, 1.16), frameon=False, fontsize=9)
    fig.subplots_adjust(wspace=.27, hspace=.40)
    fig.text(.08, -.005, 'Each bar is one reviewed episode; crosses mark unavailable endpoints. No inferential error bars.\n'
             'Planning time is measured under the recorded concurrent workload; it excludes offline evaluation.', fontsize=8, color=INK)
    save(fig, output, 'online_quality_and_cost', exports)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--analysis', type=Path, required=True)
    parser.add_argument('--reviews', type=Path, default=ROOT/'audit_results/article_stage_20260928/episode_reviews_v1')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    analysis = args.analysis.resolve(); output = args.output.resolve()
    if output.exists():
        raise ValueError('exclusive new figure snapshot required')
    verify(analysis)
    slots = read(analysis/'slots.json')
    if len({r['phase'] for r in slots}) != 1:
        raise ValueError('plot development and main in separate figure packages')
    if any(r['method'] not in METHODS for r in slots):
        raise ValueError('targeted ablations require their own labeled figure')
    records, _ = scene_plot.load_scenes()
    SOURCES.update(scene_plot.SOURCES)
    source(Path(__file__)); source(Path(scene_plot.__file__))
    output.mkdir(parents=True)
    plt.rcParams.update({'font.family':'DejaVu Sans', 'pdf.fonttype':42, 'ps.fonttype':42,
                         'svg.fonttype':'none', 'axes.titleweight':'normal'})
    exports = []
    route_panels(slots, records, args.reviews.resolve(), output, exports)
    result_panels(slots, output, exports)
    captions = {
        'routes_en':'Matched condition and scale, actual saved trajectories. All four methods share sensors, the coarse graph, reconstruction and paid costs. Open circles identify turns, squares extra observations, and arrows fixed-step sampled headings. Coincident routes are shown in separate panels. Unstarted or unqualified slots are explicitly labeled. Ground-truth equipment silhouettes are for offline explanation only.',
        'routes_zh':'统一条件与尺度的实际保存轨迹。四方法共享传感、粗导航图、重建及成本。空心圆表示转向位置，方形表示额外观察，箭头按固定步间隔显示真实朝向。重合轨迹分栏显示；未运行与不合格槽位保留标识。设备真值轮廓仅用于离线解释。',
        'quality_en':'Measured coverage, all-instance macro surface F1, their product and actual costs for every declared development condition. Each bar is one episode; unavailable values are crosses, not zeros. Wall times include concurrent execution effects and exclude offline surface evaluation. These development episodes do not estimate performance on unseen layouts.',
        'quality_zh':'各预声明开发条件中的实测覆盖、全实例宏平均表面F1、乘积及真实成本。每根柱对应一条任务，缺失值用叉标示而不填零。耗时受并发负载影响，不含离线表面评价。开发任务不代表未见布局上的性能。'}
    (output/'captions.json').write_text(json.dumps(captions, ensure_ascii=False, indent=2)+'\n')
    exports.append('captions.json')
    manifest = dict(schema='article.online_figures.v1', analysis_snapshot=str(analysis),
                    phase=slots[0]['phase'], declared_slots=len(slots),
                    reviewed_qualified=sum(r['status']=='reviewed_qualified' for r in slots),
                    new_worlds=0, new_policy_runs=0, new_tsdf_integrations=0, new_surface_evaluations=0,
                    sources=SOURCES,
                    files={name:dict(bytes=(output/name).stat().st_size, sha256=sha(output/name)) for name in exports})
    (output/'manifest.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True)+'\n')
    print(json.dumps(dict(output=str(output), files=len(exports), declared_slots=len(slots),
                         reviewed_qualified=manifest['reviewed_qualified']), ensure_ascii=False))


if __name__ == '__main__':
    main()
