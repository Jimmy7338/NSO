#!/usr/bin/env python3
"""Publication figures from frozen static assets; no experiment or replay.

Private facility geometry is used for offline explanation only. Nine layouts
mean three development plus six predesigned test layouts, not nine completed
experiments. Costs are conditional geometry-only minima, not executed routes.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import sys

os.environ.setdefault('MPLCONFIGDIR', '/tmp/nso-article-scene-overview-mpl')
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
os.environ.setdefault('OMP_NUM_THREADS', '1')

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.colors import to_rgb
from matplotlib.lines import Line2D
from matplotlib.patches import FancyArrowPatch, Patch, Polygon, Rectangle
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from nso.article_scene_assets_v1 import load_private_article_scene, load_public_article_scene

ASSETS = ROOT/'audit_results/article_stage_20260928/scene_assets_v1'
ASSET_PIN = 'a0319f7e5683a91a56effdbbab28c90e226ce5b0ac7178c2b2408cb8954ae14f'
DEFAULT_OUTPUT = ROOT/'docs/thesis/figures/article_scene_layouts_20260928'
MAX_OUTPUT_BYTES = 15 * 1024 ** 2
COLORS = {'cabinet': '#4477AA', 'rack': '#CC9933', 'workstation': '#228833', 'machine': '#AA3377'}
FAMILIES = ('AISLE', 'CELL', 'LOOP')
TITLES = {'AISLE': 'Equipment aisles', 'CELL': 'Screened workcells', 'LOOP': 'Production-island loop'}
INK = '#22313C'
MUTED = '#5F6C76'
NAV = '#A3B4BF'
OCCLUDER = '#737B83'
HOME = '#183D47'
SOURCES = {}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def source(path):
    path = Path(path)
    SOURCES[str(path.relative_to(ROOT))] = dict(sha256=sha(path), bytes=path.stat().st_size)
    return path


def mix_white(color, fraction=.63):
    return tuple((1.-fraction)*np.array(to_rgb(color))+fraction)


def load_scenes():
    manifest = json.loads(source(ASSETS/'manifest.json').read_text())
    if sha(ASSETS/'manifest.json') != ASSET_PIN:
        raise ValueError('frozen article assets changed')
    records = {}
    for identity in sorted(manifest['assets']):
        folder = ASSETS/identity
        public = load_public_article_scene(folder, expected_manifest_sha256=ASSET_PIN)
        private = load_private_article_scene(folder, expected_manifest_sha256=ASSET_PIN)
        for name in manifest['assets'][identity]['files']:
            source(folder/name)
        audit = json.loads((folder/'evaluation_private/static_geometry_audit.json').read_text())
        records[identity] = dict(public=public, private=private, static=audit,
                                 family=manifest['assets'][identity]['family'],
                                 split=manifest['assets'][identity]['split'])
    if len(records) != 9 or sum(r['split'] == 'development' for r in records.values()) != 3:
        raise ValueError('fixed 3 development + 6 test layout inventory required')
    return records, manifest


def plan_map(ax, record, *, detailed=False, show_axes=True):
    """All maps retain world coordinates and an identical 0..8 metre extent."""
    workspace = record['public']['workspace']
    width, height = workspace['bounds_xy_m'][1]
    ax.set_facecolor('#FAFBFC')
    ax.add_patch(Rectangle((0., 0.), width, height, facecolor='white', edgecolor='none', zorder=0))
    ax.add_patch(Rectangle((.12, .12), width-.24, height-.24, facecolor='#FBFCFD', edgecolor='none', zorder=0))
    graph = record['public']['graph_spec']
    nodes = graph['nodes']
    lines = [[nodes[a], nodes[b]] for a, b in graph['edges']]
    ax.add_collection(LineCollection(lines, colors=NAV, linewidths=.80 if detailed else .65,
                                    alpha=.75, zorder=1))
    points = np.array(list(nodes.values()))
    ax.scatter(points[:, 0], points[:, 1], s=11 if detailed else 5,
               facecolors='white', edgecolors=NAV, linewidths=.65, zorder=2)
    metadata = record['private']['metadata']
    # The first box is the floor. Other background boxes include walls/screens.
    for i, box in enumerate(metadata['background_boxes'][1:]):
        x0, x1, y0, y1, _, _ = box
        wall = i < 4
        ax.add_patch(Rectangle((x0, y0), x1-x0, y1-y0,
            facecolor='#B7BDC2' if wall else OCCLUDER,
            edgecolor='#A0A8AE' if wall else '#5D656D', linewidth=.45 if wall else .6,
            hatch=None if wall else '////', zorder=3))
    for instance, marker in zip(metadata['private_instances'], record['private']['markers']):
        color = COLORS[instance['category']]
        x, y, _ = instance['position_world_m']
        w, d, _ = instance['dimensions_m']
        rotation = np.array(instance['world_from_local_rotation'])[:2, :2]
        outline = np.array([[-w/2, -d/2], [w/2, -d/2], [w/2, d/2], [-w/2, d/2]]) @ rotation.T + [x, y]
        ax.add_patch(Polygon(outline, facecolor=mix_white(color), edgecolor=color, lw=1.15, zorder=5))
        anchor, normal = np.asarray(marker['center_world_m'])[:2], np.asarray(marker['normal_world'])[:2]
        tangent = np.asarray(marker['u_world'])[:2]
        ends = np.stack((anchor-marker['width_m']*.5*tangent, anchor+marker['width_m']*.5*tangent))
        ax.plot(ends[:, 0], ends[:, 1], color=color, linewidth=2.8 if detailed else 2.1,
                solid_capstyle='butt', zorder=7)
        ax.add_patch(FancyArrowPatch(anchor+normal*.05, anchor+normal*(.49 if detailed else .39),
            arrowstyle='-|>', mutation_scale=10 if detailed else 7.5,
            linewidth=1.0 if detailed else .8, color=INK, zorder=7))
        ax.text(x, y, f"I{instance['instance_id']+1}", ha='center', va='center',
                fontsize=11 if detailed else 8.8, color=INK, weight='bold', zorder=8)
    start = workspace['start_position_world_m'][:2]
    ax.scatter(*start, s=110 if detailed else 57, marker='o', facecolors='white',
               edgecolors=HOME, linewidths=1.5, zorder=8)
    ax.text(*start, 'H', color=HOME, fontsize=9 if detailed else 6.7,
            weight='bold', ha='center', va='center', zorder=9)
    ax.add_patch(FancyArrowPatch((start[0]+.19, start[1]), (start[0]+.67, start[1]),
        arrowstyle='-|>', mutation_scale=10 if detailed else 7.5, lw=1.1, color=HOME, zorder=8))
    if detailed:
        ax.text(.25, .28, 'H: start + required return', color=HOME, fontsize=8.5,
                ha='left', va='center', zorder=9)
    ax.set(xlim=(-.10, 8.10), ylim=(-.10, 8.10), aspect='equal',
           xticks=(0, 2, 4, 6, 8), yticks=(0, 2, 4, 6, 8))
    ax.spines[['top', 'right']].set_visible(False)
    ax.spines[['bottom', 'left']].set_color('#B1B8BE')
    ax.spines[['bottom', 'left']].set_linewidth(.65)
    ax.tick_params(labelsize=8.3 if detailed else 7.7, colors=MUTED, length=2.8, width=.65)
    if show_axes:
        ax.set_xlabel('$x$ (m)', fontsize=9 if detailed else 8.5, labelpad=3)
        ax.set_ylabel('$y$ (m)', fontsize=9 if detailed else 8.5, labelpad=3)


def legend_handles():
    return ([Patch(facecolor=mix_white(c), edgecolor=c, label=name.capitalize()) for name, c in COLORS.items()]
            + [Patch(facecolor=OCCLUDER, edgecolor='#5D656D', hatch='////', label='Occluding structure'),
               Line2D([], [], color=NAV, marker='o', markersize=3, markerfacecolor='white', lw=.8,
                      label='Supplied coarse graph'),
               Line2D([], [], color=HOME, marker='$H$', markersize=8, lw=0,
                      label='Start / required return'),
               Line2D([], [], color=INK, marker='>', markersize=4, lw=.8,
                      label='Equipment front')])


def save(fig, output, name, exports):
    for ext in ('pdf', 'svg', 'png'):
        path = output/f'{name}.{ext}'
        metadata = {'CreationDate': None, 'ModDate': None} if ext == 'pdf' else {'Date': None} if ext == 'svg' else None
        fig.savefig(path, dpi=300, facecolor='white', metadata=metadata)
        exports.append(path)
    plt.close(fig)


def overview(records, output, exports):
    fig, axes = plt.subplots(3, 3, figsize=(10.6, 11.35))
    fig.subplots_adjust(left=.065, right=.985, top=.865, bottom=.108, hspace=.285, wspace=.225)
    fig.text(.065, .969, 'Industrial mapping layouts', ha='left', fontsize=17, weight='bold', color=INK)
    fig.text(.065, .938, '3 development + 6 predesigned test layouts  |  Static geometry, not experimental outcomes',
             ha='left', fontsize=9.2, color=MUTED)
    for col, (split, title) in enumerate([('DEV', 'DEVELOPMENT'), ('T0', 'TEST LAYOUT 1'), ('T1', 'TEST LAYOUT 2')]):
        box = axes[0, col].get_position()
        fig.text(box.x0, .907, title, color=INK if col == 0 else MUTED, fontsize=9.5, weight='bold')
    for row, family in enumerate(FAMILIES):
        for col, split in enumerate(('DEV', 'T0', 'T1')):
            ax = axes[row, col]
            identity = f'ART1_{family}_{split}'
            plan_map(ax, records[identity])
            ax.set_title(f'({chr(97+row*3+col)}) {TITLES[family]}', loc='left',
                         fontsize=9.4, weight='bold', color=INK, pad=10)
            ax.text(.995, .995, identity, transform=ax.transAxes, ha='right', va='top',
                    fontsize=6.8, color=MUTED,
                    bbox=dict(facecolor='white', edgecolor='none', pad=1.4, alpha=.9))
    fig.legend(handles=legend_handles(), loc='lower center', bbox_to_anchor=(.522, .037),
               ncol=4, fontsize=8.2, frameon=False, columnspacing=2.3, handlelength=1.65, labelspacing=.85)
    fig.text(.065, .017, 'Identical metre scale in all panels. Equipment labels and fronts are offline annotations; no executed path is shown.',
             fontsize=7.7, color=MUTED)
    save(fig, output, 'scene_overview_9layouts', exports)


QUESTIONS = {
    'AISLE': ('Which aisle next?', 'Choose approach directions and\nshare the finite travel budget\nbetween repeated installations.'),
    'CELL': ('When is a detour worthwhile?', 'Screens separate nearby devices\ninto different travel regions;\nclose observation can be costly.'),
    'LOOP': ('Which way around the island?', 'Two circulation directions couple\nobservation order, new views\nand the remaining return budget.'),
}


def details(records, output, exports):
    for index, family in enumerate(FAMILIES):
        identity = f'ART1_{family}_DEV'
        record = records[identity]
        fig = plt.figure(figsize=(10.4, 7.15))
        ax = fig.add_axes([.064, .17, .585, .735])
        plan_map(ax, record, detailed=True)
        fig.text(.064, .954, f'({chr(97+index)}) {TITLES[family]}', fontsize=16, color=INK, weight='bold')
        fig.text(.064, .916, f'{identity}  |  Development geometry and public navigation prior', fontsize=9.3, color=MUTED)
        side = fig.add_axes([.71, .22, .26, .64])
        side.axis('off')
        side.text(0, 1, 'EQUIPMENT / OFFLINE LABELS', fontsize=8.5, weight='bold', color=MUTED, transform=side.transAxes)
        for i, instance in enumerate(record['private']['metadata']['private_instances']):
            color = COLORS[instance['category']]
            y = .905-i*.092
            side.text(.01, y, f'I{i+1}', ha='left', va='center', fontsize=9.5, weight='bold', color=color,
                      transform=side.transAxes)
            side.text(.14, y, instance['category'].capitalize(), ha='left', va='center', fontsize=10,
                      color=INK, transform=side.transAxes)
        side.plot([0, 1], [.555, .555], color='#D9DFE3', lw=.8, transform=side.transAxes)
        side.text(0, .505, 'STATIC CLOSE-VIEW COST*', fontsize=8.5, weight='bold', color=MUTED,
                  transform=side.transAxes)
        costs = [r['minimum_individual_view_return_actions'] for r in record['static']['per_facility']]
        for i, cost in enumerate(costs):
            x = .035+i*.245
            side.text(x, .42, f'I{i+1}', color=MUTED, fontsize=9, transform=side.transAxes)
            side.text(x, .345, str(cost), color=INK, weight='bold', fontsize=15, transform=side.transAxes)
        side.text(0, .23, '*One close-front observation\n  + full-pose return, in paid actions.',
                  color=MUTED, fontsize=8, linespacing=1.4, transform=side.transAxes)
        question, body = QUESTIONS[family]
        side.text(0, .095, question, color=INK, fontsize=10, weight='bold', transform=side.transAxes)
        side.text(0, .045, body, color=MUTED, fontsize=9.3, va='top', linespacing=1.5, transform=side.transAxes)
        fig.legend(handles=legend_handles()[4:], loc='lower center', bbox_to_anchor=(.51, .073), ncol=4,
                   frameon=False, fontsize=8.3, handlelength=1.8, columnspacing=1.8)
        fig.text(.064, .041, 'Costs are conditional static minima over certified close-front coarse poses, not executed trajectories.',
                 color=MUTED, fontsize=8)
        fig.text(.064, .018, 'No classification, diagnostic revisit, sensor failure or online planning overhead is included in these static costs.',
                 color=MUTED, fontsize=8)
        save(fig, output, 'development_'+family.lower()+'_detail', exports)


def cost_figure(records, output, exports):
    identities = [f'ART1_{f}_{s}' for f in FAMILIES for s in ('DEV', 'T0', 'T1')]
    values = np.array([[x['minimum_individual_view_return_actions'] for x in records[i]['static']['per_facility']]
                       for i in identities], dtype=int)
    fig, ax = plt.subplots(figsize=(8.1, 5.8))
    fig.subplots_adjust(left=.245, right=.96, top=.77, bottom=.16)
    image = ax.imshow(values, cmap='Blues', vmin=0, vmax=160, aspect='auto', interpolation='nearest')
    ax.set(xticks=range(4), xticklabels=['Equipment I1', 'Equipment I2', 'Equipment I3', 'Equipment I4'],
           yticks=range(9), yticklabels=[i.replace('ART1_', '').replace('_DEV', ' / development').replace('_T0', ' / test 1').replace('_T1', ' / test 2') for i in identities])
    ax.tick_params(axis='both', length=0, labelsize=9, colors=INK)
    ax.xaxis.tick_top()
    for r in range(9):
        for c in range(4):
            value = values[r, c]
            ax.text(c, r, str(value), ha='center', va='center', fontsize=12, weight='bold' if r % 3 == 0 else 'normal',
                    color='white' if value > 95 else INK)
            if value > 120:
                ax.add_patch(Rectangle((c-.485, r-.46), .97, .92, fill=False, lw=1.6, edgecolor='#CC6633'))
    for y in (2.5, 5.5):
        ax.axhline(y, color='white', lw=3)
    for spine in ax.spines.values():
        spine.set_visible(False)
    fig.text(.055, .943, 'Static cost of a close-front observation and return', fontsize=14.2, weight='bold', color=INK)
    fig.text(.055, .896, '3 development + 6 predesigned test layouts  |  Four devices remain in every task denominator',
             fontsize=8.8, color=MUTED)
    fig.text(.055, .86, 'Paid actions: graph travel + one observation + full-pose return. Extra inspection is excluded.',
             fontsize=8.6, color=MUTED)
    fig.text(.055, .097, 'Orange outlines: cost exceeds 120 actions for that single close-front visit; this is not a task-failure label.',
             fontsize=8.1, color=MUTED)
    fig.text(.055, .052, 'Conditional minima within certified close-front coarse poses, not global continuous optima or measured policy cost.',
             fontsize=8.1, color=MUTED)
    save(fig, output, 'static_close_view_costs', exports)


def captions():
    return '''# New industrial-layout figures / 新工业布局图注

## Overview / 总览

**EN.** Nine predeclared industrial mapping layouts: one development layout and two test layouts for each of equipment aisles, screened workcells, and a production-island loop. All panels retain the same 0–8 m axes and world coordinate frame. Colored envelopes and identifiers locate the four facilities; arrows indicate their front sides. Gray segments and nodes are the shared coarse navigation prior, and H is the common start and required return pose. Classes, facility positions, and occluders are private asset information used here solely for offline explanation. These are 3 development + 6 predesigned test layouts, not nine completed runs; no executed trajectory or measured performance is shown.

**中。** 九个预先设计的工业主动建图布局：设备巷道、隔断工位与中央生产岛环路各包括一个开发布局和两个测试布局。全部子图保留统一的0–8 m坐标轴及世界坐标系。彩色包络与编号表示四台设备，箭头表示设备正面；灰色节点与线段为各方法共享的粗略导航先验，H为共同起点和规定返航位姿。类别、设施位置及遮挡信息仅用于离线解释，不作为控制器的隐藏真值输入。本图表示3个开发与6个预先设计的测试布局，不能解读为九次实验已经完成；图中不展示实际执行轨迹或实测性能。

## Development details / 开发布局详图

**EN.** Detailed views of the three development layouts under identical metre scaling. Equipment labels are offline annotations; marker-side arrows describe object orientation rather than robot motion. The equipment aisles require observation-direction and inter-device budget choices; workcell screens introduce detours; the central island couples circulation direction with observation order and return cost. The four costs are the minimum static primitive-action counts for a single close-front observation and full-pose return within the certified coarse candidate set. A valid pose is 0.35–1.30 m from the label in the horizontal plane, faces the label with cosine at least 0.5, and contains all nominal label corners in the field of view without intervening obstacle-envelope intersections. These conditional lower-bound costs exclude additional classification, diagnosis, revisits, and online sensing effects. They are neither executed trajectories nor proof of reconstruction quality.

**中。** 三个开发布局采用同一米制比例展示。设备类别为离线标注，标签面箭头表示物体朝向，不是机器人运动方向。巷道任务考查观察方向与跨设备预算分配；工位隔断引入绕行代价；中央生产岛把绕行方向、观察顺序与返航成本联系起来。每台设备旁的成本是经过认证的粗图候选集合内，单次近距离正面观察并按完整位姿返航所需原子动作数的静态最小值。候选到标签的水平距离为0.35–1.30 m、正面余弦不少于0.5，且标签四角处于名义相机视场、视线不与其他障碍包络相交。该条件下界不包含额外类别确认、诊断、重访及在线传感影响，不代表已执行轨迹或三维重建质量。

## Static cost matrix / 静态成本矩阵

**EN.** Static one-device close-view-and-return costs for all nine layouts, computed from the frozen geometry and common primitive navigation graph. Development rows use bold values. Orange outlines indicate costs above 120 actions for that single constrained visit; they do not mark an experiment failure. The task does not require visiting every device, and all four devices remain in the evaluation denominator. This table gives a geometric budget reference before outcome analysis; it is not a method ranking or a global optimum over unrestricted observation poses.

**中。** 九布局的单设备近观察与返航静态成本，由冻结几何和共享原子导航图计算。开发行使用粗体；橙框只标识相应受约束单次往返成本超过120动作，不表示实验失败。任务不以访问全部设备为资格要求，四台设备始终保留在评价分母中。该表用于结果分析前的几何预算说明，不构成方法排名，也不是任意观察位姿下的全局最优值。
'''


def prepare_output(output, replace):
    if output.exists():
        manifest_path = output/'manifest.json'
        if not replace or not manifest_path.is_file():
            raise FileExistsError('use a fresh output or --replace for this script-owned figure bundle')
        previous = json.loads(manifest_path.read_text())
        if previous.get('schema') != 'article.scene_figure_bundle.v1':
            raise ValueError('refuse replacing an unrelated directory')
        expected = set(previous['files']) | {'manifest.json'}
        if {p.name for p in output.iterdir()} != expected:
            raise ValueError('unexpected files in figure bundle')
        for name, row in previous['files'].items():
            if sha(output/name) != row['sha256']:
                raise ValueError('externally modified figure: '+name)
        for path in output.iterdir():
            path.unlink()
    else:
        output.mkdir(parents=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument('--replace', action='store_true')
    args = parser.parse_args()
    source(Path(__file__))
    records, asset_manifest = load_scenes()
    execution_source_hashes = dict(asset_manifest['source_sha256'])
    if any(sha(ROOT/name) != digest for name, digest in execution_source_hashes.items()):
        raise ValueError('frozen asset-source closure differs')
    prepare_output(args.output, args.replace)
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 9, 'text.color': INK,
        'axes.labelcolor': MUTED, 'pdf.fonttype': 42, 'ps.fonttype': 42,
        'svg.fonttype': 'none', 'svg.hashsalt': 'article_scene_layouts_20260928', 'savefig.pad_inches': .02})
    exports = []
    overview(records, args.output, exports)
    details(records, args.output, exports)
    cost_figure(records, args.output, exports)
    rows = []
    for identity, record in records.items():
        for instance, costs in zip(record['private']['metadata']['private_instances'], record['static']['per_facility']):
            rows.append(dict(scene_id=identity, split=record['split'], family=record['family'],
                device_id=f"I{instance['instance_id']+1}", offline_category=instance['category'],
                static_close_view_return_actions=costs['minimum_individual_view_return_actions'],
                certified_close_frontal_positions=costs['close_frontal_coarse_positions'],
                measured_policy_cost=False))
    csv_path = args.output/'scene_static_costs.csv'
    with csv_path.open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)
    exports.append(csv_path)
    caption_path = args.output/'figure_captions.md'
    caption_path.write_text(captions()); exports.append(caption_path)
    if any(sha(ROOT/name) != digest for name, digest in execution_source_hashes.items()):
        raise ValueError('execution dependencies changed during plotting')
    if any(sha(ROOT/name) != row['sha256'] for name, row in SOURCES.items()):
        raise ValueError('figure input changed during plotting')
    manifest = dict(schema='article.scene_figure_bundle.v1', asset_manifest_sha256=ASSET_PIN,
        sources=SOURCES, frozen_execution_dependencies_unchanged=True,
        layout_counts=dict(development=3, predesigned_test=6), plots_show_experiment_completion=False,
        new_worlds=0, new_sensor_packets=0, new_policy_executions=0, new_tsdf_integrations=0,
        new_quality_evaluations=0, coordinate_convention='shared world XY metres, identical [-0.1,8.1] plot extent',
        private_geometry_scope='offline explanatory annotation only; never loaded by policy',
        minimum_cost_scope='conditional static minimum over certified close-front coarse poses; not measured policy cost',
        dependencies=dict(numpy=np.__version__, matplotlib=matplotlib.__version__),
        files={p.name: dict(sha256=sha(p), bytes=p.stat().st_size) for p in exports})
    (args.output/'manifest.json').write_text(json.dumps(manifest, indent=2, sort_keys=True)+'\n')
    total = sum(p.stat().st_size for p in args.output.iterdir())
    if total > MAX_OUTPUT_BYTES:
        raise ValueError('figure bundle exceeds 15 MiB')
    print(json.dumps(dict(output=str(args.output), files=len(exports)+1, bytes=total,
                         manifest_sha256=sha(args.output/'manifest.json')), indent=2))


if __name__ == '__main__':
    main()
