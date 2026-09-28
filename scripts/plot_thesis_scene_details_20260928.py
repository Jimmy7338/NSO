#!/usr/bin/env python3
"""Draw saved V36 scenes, executed paths and TSDF meshes; no new experiment.

All eight conditions are retained. Meshes are projected without decimation,
completion, alignment fitting or re-scoring. Insets are 2-D windows of the same
projection; their boxes are determined from the declared geometry, not errors.
"""
from __future__ import annotations

import csv
import hashlib
import itertools
import json
import os
from pathlib import Path
import shutil

os.environ.setdefault('MPLCONFIGDIR', '/tmp/nso-scene-details-20260928-mpl')
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
os.environ.setdefault('OMP_NUM_THREADS', '1')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.collections import PolyCollection
from matplotlib.colors import to_rgb
from matplotlib.font_manager import FontProperties, findfont
from matplotlib.lines import Line2D
from matplotlib.patches import Patch, Rectangle
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'docs/thesis/figures/scene_details_20260928'
BASE = Path('audit_results/v36_online_confirmation_20260918')
SCENE = Path('configs/virtual3d/v33_direction_scene_r1_20260917.json')
LIMIT = 30 * 1024**2
RESERVE = 128 * 1024**2
COLORS = {'G': '#0072B2', 'S': '#D55E00'}
CASES = [('P00', 0), ('P00', 1), ('P01', 0), ('P01', 1)]
SOURCES: dict = {}
OUTPUTS: list[Path] = []


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def source(relative):
    path = ROOT / relative
    SOURCES[str(relative)] = {'sha256': sha(path), 'bytes': path.stat().st_size}
    return path


def read(relative):
    return json.loads(source(relative).read_text())


def save(fig, name):
    for ext in ('pdf', 'svg', 'png'):
        path = OUT / f'{name}.{ext}'
        metadata = {'CreationDate': None, 'ModDate': None} if ext == 'pdf' else {'Date': None} if ext == 'svg' else None
        fig.savefig(path, dpi=300, metadata=metadata, facecolor='white')
        OUTPUTS.append(path)
    plt.close(fig)


def device(points, parent):
    """Known rigid coordinate change; no registration against GT or a mesh."""
    points = np.asarray(points, dtype=float).copy()
    origin = np.asarray(parent['device_frame']['origin_xyz'])[:points.shape[-1]]
    points -= origin
    for _ in range(parent['device_frame']['quarter_turns_ccw']):
        x, y = points[..., 0].copy(), points[..., 1].copy()
        points[..., 0], points[..., 1] = y, -x
    return points


def bounds_local(box, parent):
    x0, x1, y0, y1, z0, z1 = box
    corners = device(list(itertools.product((x0, x1), (y0, y1), (z0, z1))), parent)
    return corners.min(axis=0), corners.max(axis=0)


def load_data():
    config = read(BASE / 'config.json')
    scene = read(SCENE)
    parents = {p['id']: p for p in scene['parents']}
    records = {}
    for cell in config['physical_cases']:
        p, h, mode = cell['parent'], cell['hypothesis'], cell['mode']
        folder = BASE / f"case{cell['index']:02d}"
        result = read(folder / 'result.json')
        trace = read(folder / 'trace.json')
        crop = read(folder / 'final_crop.json')
        with np.load(source(folder / 'final_extracted.npz'), allow_pickle=False) as arrays:
            vertices = arrays['vertices'].copy() - np.asarray(config['parents'][p]['translation'])
            triangles = arrays['triangles'].copy()
        vertices = device(vertices, parents[p])
        path = device([row['pose_v33'][:2] for row in trace], parents[p])
        assert result['physical_case'] == cell
        assert result['paid_actions'] == 42 and len(trace) == 43
        assert np.isfinite(vertices).all() and len(triangles) == crop['kept_triangles']
        assert np.array_equal(path[0], path[18]) and np.array_equal(path[0], path[42])
        records[p, h, mode] = dict(vertices=vertices, triangles=triangles, trace=trace,
            path=path, result=result, crop=crop, folder=str(folder))
    for p, h in CASES:
        np.testing.assert_array_equal(records[p, h, 'G']['path'][:19], records[p, h, 'S']['path'][:19])
        if h == 0:
            np.testing.assert_array_equal(records[p, h, 'G']['vertices'], records[p, h, 'S']['vertices'])
            np.testing.assert_array_equal(records[p, h, 'G']['triangles'], records[p, h, 'S']['triangles'])
    return config, parents, records


def overview(parents, records):
    fig, axes = plt.subplots(2, 2, figsize=(7.4, 7.65))
    fig.subplots_adjust(left=.08, right=.985, top=.90, bottom=.16, wspace=.19, hspace=.32)
    for index, ((p, h), ax) in enumerate(zip(CASES, axes.flat)):
        parent = parents[p]
        boxes = parent['hypotheses'][h]['assets'][0]['boxes']
        local_boxes = [bounds_local(b, parent) for b in boxes]
        nav = device(parent['nav_cells'], parent)
        ax.scatter(nav[:, 0], nav[:, 1], s=5, c='#D5D9DC', zorder=0)
        for box_index in [1, 2, 3, 4, 5, 0]:
            low, high = local_boxes[box_index]
            is_guard = box_index == 0
            color = '#E5D9C5' if is_guard else '#8BA8AF' if box_index >= 3 else '#E1E6E9'
            ax.add_patch(Rectangle(low[:2], *(high[:2] - low[:2]), facecolor=color,
                edgecolor='#77858B', lw=.6, hatch='////' if is_guard else None, zorder=2))
        prefix = records[p, h, 'G']['path'][:19]
        ax.plot(prefix[:, 0], prefix[:, 1], color='#D0D5D9', lw=6,
                solid_capstyle='round', zorder=3)
        for mode in ('S', 'G'):
            path = records[p, h, mode]['path'][18:]
            ax.plot(path[:, 0], path[:, 1], color=COLORS[mode],
                lw=2.0 if mode == 'S' else 1.5, ls='-' if mode == 'S' else (0, (3, 2)), zorder=4)
            # Outbound segment from actual successive saved positions.
            a, b = records[p, h, mode]['path'][24:26]
            if np.any(a != b):
                ax.annotate('', xy=b, xytext=a,
                    arrowprops=dict(arrowstyle='-|>', color=COLORS[mode], lw=1.6), zorder=6)
        cue = device([parent['hypotheses'][h]['assets'][0]['front_seed_xyz']], parent)[0]
        ax.scatter(cue[0], cue[1], marker='s', c='#B67625', s=24, linewidth=.7,
                   edgecolors='white', zorder=7)
        ax.scatter([0], [0], marker='D', c='#273339', s=27, zorder=7)
        ax.text(0, -.35, 'Start / return', ha='center', fontsize=7)
        for sign in (-1, 1):
            ax.annotate('', xy=(1.15*sign, -.53), xytext=(.48*sign, -.53),
                        arrowprops=dict(arrowstyle='-|>', color='#4D585F', lw=.7))
        ax.text(0, -.94, 'Left / right candidates at action 19', ha='center', fontsize=6.8, color='#4D585F')
        ax.text(0, 1.31, 'Front occluder', ha='center', va='center', fontsize=7, color='#5D513A',
                bbox=dict(facecolor='#EFE7D9', edgecolor='none', pad=.8))
        jg = records[p, h, 'G']['result']['stages']['final']['measurement']['05cm']['joint']
        js = records[p, h, 'S']['result']['stages']['final']['measurement']['05cm']['joint']
        step = 'G: left; S: left' if h == 0 else 'G: left; S: right'
        ax.set_title(f'({chr(97+index)}) {p} / h{h}  |  class {"A" if h == 0 else "B"}',
                     loc='left', fontsize=9.2, fontweight='bold', pad=8)
        ax.text(.02, .98, f'Action 19: {step}\nSaved $\\Delta J_5$ = {js-jg:+.4f}',
                transform=ax.transAxes, va='top', fontsize=7.5, linespacing=1.5,
                bbox=dict(facecolor='white', edgecolor='none', alpha=.93, pad=1.8))
        ax.set(xlim=(-3.55, 3.55), ylim=(-1.2, 5.95), aspect='equal', xlabel='$u$ (m)', ylabel='$v$ (m)',
               xticks=(-3, 0, 3), yticks=(0, 2, 4))
        ax.spines[['top', 'right']].set_visible(False)
        ax.tick_params(length=2.5)
    fig.suptitle('Designed equipment layouts and executed observation routes', x=.08, y=.97,
                 ha='left', fontsize=12, fontweight='bold')
    fig.text(.08, .935, 'Offline geometry context • equipment coordinates • 42 paid actions in every run', fontsize=8.2, color='#4D585F')
    handles = [Line2D([0], [0], color=COLORS['G'], lw=1.6, ls='--', label='G: geometry-only belief'),
               Line2D([0], [0], color=COLORS['S'], lw=2, label='S: class-conditioned belief'),
               Line2D([0], [0], color='#D0D5D9', lw=6, label='Shared 18-action prefix'),
               Patch(facecolor='#E5D9C5', edgecolor='#77858B', hatch='////', label='Front occluder'),
               Patch(facecolor='#8BA8AF', edgecolor='#77858B', label='Ribbed equipment surface'),
               Line2D([0], [0], color='#B67625', marker='s', lw=0, label='Artificial RGB class cue')]
    fig.legend(handles=handles, loc='lower center', bbox_to_anchor=(.525, .034), ncols=2,
               frameon=False, fontsize=7.4, handlelength=2.5, columnspacing=2.2, labelspacing=.85)
    fig.text(.08, .012, 'P01 is rotated into the equipment frame for display. Turns and retraced segments overlap.',
             fontsize=6.8, color='#4D585F')
    save(fig, 'scene_overview')


def projection(vertices, hypothesis):
    azimuth = np.deg2rad(55 if hypothesis == 1 else 125)
    elevation = np.deg2rad(24)
    eye = np.array([np.cos(azimuth)*np.cos(elevation), np.sin(azimuth)*np.cos(elevation), np.sin(elevation)])
    horizontal = np.array([-np.sin(azimuth), np.cos(azimuth), 0.])
    vertical = np.cross(eye, horizontal)
    v = np.asarray(vertices) - np.array([0., 2.6, .9])
    return np.column_stack([v @ horizontal, v @ vertical, v @ eye])


def prepare_facets(record, hypothesis):
    vertices, triangles = record['vertices'], record['triangles']
    projected = projection(vertices, hypothesis)[triangles]
    order = np.argsort(projected[:, :, 2].mean(axis=1), kind='stable')
    facets = vertices[triangles]
    normals = np.cross(facets[:, 1] - facets[:, 0], facets[:, 2] - facets[:, 0])
    lengths = np.linalg.norm(normals, axis=1)
    normals = np.divide(normals, lengths[:, None], out=np.zeros_like(normals), where=lengths[:, None] > 0)
    light = np.array([.45 if hypothesis else -.45, .6, 1.])
    light /= np.linalg.norm(light)
    shade = .60 + .40*np.abs(normals @ light)
    # One shared neutral surface material for all methods; color cannot suggest quality.
    rgb = np.array(to_rgb('#67818E'))
    colors = np.clip(rgb[None, :] * shade[:, None] + .10, 0, 1)
    return projected[order, :, :2], colors[order]


def detail_window(parent, hypothesis):
    boxes = parent['hypotheses'][hypothesis]['assets'][0]['boxes'][3:]
    points = []
    for box in boxes:
        low, high = bounds_local(box, parent)
        points.extend(itertools.product(*zip(low, high)))
    projected = projection(points, hypothesis)[:, :2]
    low, high = projected.min(axis=0)-.10, projected.max(axis=0)+.10
    center = (low + high) / 2
    # All conditions use the same 3.40 m x 2.55 m projected window size.
    half = np.array([1.70, 1.275])
    return center-half, center+half


def render(ax, prepared, limits):
    facets, colors = prepared
    collection = PolyCollection(facets, closed=True, facecolors=colors, edgecolors='none',
                                linewidth=0, antialiased=False, rasterized=True)
    ax.add_collection(collection)
    ax.set(xlim=limits[0], ylim=limits[1], aspect='equal')
    ax.set_axis_off()
    return collection


def mesh_figures(parents, records):
    prepared = {key: prepare_facets(record, key[1]) for key, record in records.items()}
    windows = {(p, h): detail_window(parents[p], h) for p, h in CASES}
    full_limits = ((-3.9, 3.9), (-2.1, 2.15))
    display = {}
    for kind in ('comparison', 'details'):
        fig, axes = plt.subplots(4, 2, figsize=(7.4, 9.0))
        fig.subplots_adjust(left=.045, right=.985, top=.915, bottom=.06, hspace=.26, wspace=.055)
        for row, (p, h) in enumerate(CASES):
            low, high = windows[p, h]
            for column, mode in enumerate(('G', 'S')):
                ax = axes[row, column]
                record = records[p, h, mode]
                limits = full_limits if kind == 'comparison' else ((low[0], high[0]), (low[1], high[1]))
                render(ax, prepared[p, h, mode], limits)
                if kind == 'comparison':
                    ax.add_patch(Rectangle(low, *(high-low), fill=False, lw=.85,
                        linestyle=(0, (3, 2)), edgecolor='#B67625', zorder=5))
                metric = record['result']['stages']['final']['measurement']['05cm']
                if kind == 'comparison':
                    title = f'{p} / h{h}   {mode}     $F_1^{{5cm}}$ = {metric["f1"]:.3f};  $J_5$ = {metric["joint"]:.3f}'
                else:
                    title = f'{p} / h{h}   {mode}    '+('same geometry as paired method' if h == 0 else 'ribbed-side detail')
                ax.set_title(title, loc='left', fontsize=7.8, fontweight='bold', color=COLORS[mode], pad=5)
                # Common metric scale bar, expressed in projected metres.
                x, y = limits[0][0]+.14, limits[1][0]+.10
                size = 1.0 if kind == 'comparison' else .5
                ax.plot([x, x+size], [y, y], color='#394A52', lw=1.5)
                ax.text(x+size/2, y+.08, f'{size:g} m', ha='center', fontsize=7, color='#394A52')
                if kind == 'comparison':
                    display[f'{p}/h{h}/{mode}'] = {
                        'source_folder': record['folder'], 'triangles_projected': len(record['triangles']),
                        'saved_metrics_copied': metric,
                        'camera': {'azimuth_degrees': 55 if h else 125, 'elevation_degrees': 24,
                                   'center_device_m': [0, 2.6, .9], 'projection': 'orthographic'},
                        'full_projection_limits_m': full_limits,
                        'detail_projection_limits_m': [low.tolist(), high.tolist()],
                        'detail_selection': 'projection of all three declared ribs; fixed 3.40 x 2.55 m display window',
                    }
        fig.text(.045, .975, 'Saved TSDF surfaces: all four paired conditions' if kind == 'comparison'
                 else 'Magnified views of the same saved surfaces', fontsize=12, fontweight='bold')
        fig.text(.045, .945, 'Identical view and metric scale within each pair • all saved triangles projected' if kind == 'comparison'
                 else 'Dashed windows in the full views • no filling, smoothing or new reconstruction', fontsize=8.2, color='#4D585F')
        fig.text(.045, .018, 'A shared neutral material shows geometry; shading is illumination, not error. h0 pairs have identical meshes.',
                 fontsize=6.8, color='#4D585F')
        save(fig, f'mesh_{kind}')
    return display


def write_tables(records):
    path = OUT / 'saved_measurements.csv'
    with path.open('w', newline='') as handle:
        writer = csv.writer(handle)
        writer.writerow(['parent', 'hypothesis', 'mode', 'paid_actions', 'precision_5cm', 'recall_5cm', 'f1_5cm', 'joint_5cm', 'saved_triangles', 'source_folder'])
        for (p, h, m), record in sorted(records.items()):
            r = record['result']; q = r['stages']['final']['measurement']['05cm']
            writer.writerow([p, h, m, r['paid_actions'], *[q[k] for k in ('precision', 'recall', 'f1', 'joint')], len(record['triangles']), record['folder']])
    OUTPUTS.append(path)
    path = OUT / 'saved_trajectories.csv'
    with path.open('w', newline='') as handle:
        writer = csv.writer(handle)
        writer.writerow(['parent', 'hypothesis', 'mode', 'paid', 'u_m', 'v_m', 'source_x_m', 'source_y_m', 'source_heading', 'action_just_completed', 'next_action', 'source_trace'])
        for (p, h, m), record in sorted(records.items()):
            for point, row in zip(record['path'], record['trace']):
                writer.writerow([p, h, m, row['paid'], *point, *row['pose_v33'], row['action'], row['next_action'], record['folder']+'/trace.json'])
    OUTPUTS.append(path)


def main():
    if shutil.disk_usage(ROOT).free < RESERVE + LIMIT:
        raise RuntimeError('Insufficient space for bounded figure output plus reserve')
    OUT.mkdir(parents=True, exist_ok=True)
    font = findfont(FontProperties(family='Liberation Sans'), fallback_to_default=False)
    plt.rcParams.update({'font.family': 'Liberation Sans', 'font.size': 8,
        'axes.linewidth': .6, 'xtick.labelsize': 7, 'ytick.labelsize': 7,
        'pdf.fonttype': 42, 'ps.fonttype': 42, 'svg.fonttype': 'none',
        'svg.hashsalt': 'nso-scene-details-20260928', 'savefig.facecolor': 'white'})
    config, parents, records = load_data()
    overview(parents, records)
    display = mesh_figures(parents, records)
    write_tables(records)
    captions = {
        'scene_overview': {
            'caption_zh': '两种设施布局及其双构型的离线几何示意与真实执行路径。P00和P01分别为短体与长体布局，h0/h1的侧部结构及人工RGB类别提示对应不同。斜线区域为共同前遮挡板，深色矩形为侧部结构。浅灰粗线表示共同18动作前缀，蓝虚线G与橙实线S为之后保存的实际轨迹；黑色菱形为共同起终点，箭头标示候选方向和真实行进方向，原地转向及重复经过路段叠加绘制。两个h0条件动作与最终评分一致，两个h1条件在第19个付费动作选择不同方向。P01仅按已声明设备坐标系旋转显示；完整设施几何仅用于本图离线解释，不作为运行时已知真值输入。每条轨迹均42动作返航。',
            'caption_en': 'Offline equipment geometry and saved executed routes for two layouts and both configurations. Hatched regions are front occluders; darker blocks are lateral ribs. G (dashed blue) and S (solid orange) share an 18-action prefix and the start/return anchor. Both h0 pairs coincide, whereas both h1 pairs choose different directions at paid action 19. Turns and retraced segments overlap. P01 is displayed in its declared equipment frame. Ground-truth geometry is explanatory context, not online planner input.',
            'suggested_position': '实验设置之后、定量主结果之前；用于贯穿双构型方法例子。'},
        'mesh_comparison': {
            'caption_zh': '四个条件的保存终点TSDF公共ROI网格。每对G/S采用完全相同的正交视角、尺度、材质与光照，并绘制全部保存三角面片；h0从左后侧、h1从右后侧观察，视角在每对方法之间保持不变。标注的F1和J5直接取自冻结评分，不重新测量。虚框对应下一图的同投影局部窗口。h1两对显示侧部观测引起的重建差别；h0两对网格逐数组完全一致。该图不进行补洞、平滑或真值表面叠加，明暗不表示重建误差。',
            'caption_en': 'Saved final public-ROI TSDF meshes for all four conditions. Each G/S pair shares the same orthographic camera, scale, neutral material and lighting; all saved triangles are projected. h0 and h1 are viewed from the rear-left and rear-right respectively. F1 and J5 are copied from frozen measurements. Dashed windows identify the subsequent detail views. Both h0 pairs are array-identical. Shading is illumination, not error; no completion, smoothing or ground-truth surface is added.',
            'suggested_position': '主结果和机制时间线之后的三维重建定性分析。'},
        'mesh_details': {
            'caption_zh': '前图虚框的同投影放大，放大窗口由已声明的侧部结构范围确定，所有条件均使用3.40m×2.55m的投影窗口。显示仍来自同一批保存网格，不重新裁剪重建表面或生成几何。h1的G侧部网格仍较稀疏，S通过侧向观测呈现更多可见表面；两个h0条件提供持平对照。局部观察仅用于解释全局冻结评分，不额外定义或报告局部质量指标。',
            'caption_en': 'Magnified windows of the identical mesh projections. Windows are determined from the declared lateral-rib geometry and use a fixed 3.40 m by 2.55 m projected extent across conditions. These are display crops of saved surfaces, not new reconstructions. The h1 comparisons show the additional observed side surfaces; both h0 pairs retain identical geometry. No local quality score is introduced.',
            'suggested_position': '紧接整体网格图，或双栏文章的补充材料。'},
    }
    path = OUT / 'captions.json'
    path.write_text(json.dumps(captions, ensure_ascii=False, indent=2)+'\n')
    OUTPUTS.append(path)
    source(Path(__file__).relative_to(ROOT))
    for relative, value in SOURCES.items():
        assert sha(ROOT/relative) == value['sha256'], relative
    receipt = {
        'version': 'saved-v36-scene-details-20260928-v1', 'date': '2026-09-28',
        'command': 'python3 scripts/plot_thesis_scene_details_20260928.py',
        'sources': SOURCES, 'display': display, 'font_path': font, 'dpi': 300,
        'outputs': {str(p.relative_to(ROOT)): {'sha256': sha(p), 'bytes': p.stat().st_size} for p in OUTPUTS},
        'scope': {'all_four_conditions': True, 'methods_per_condition': ['G', 'S'],
                  'new_worlds': 0, 'new_sensor_packets': 0, 'new_planner_calls': 0,
                  'new_tsdf_integrations': 0, 'new_quality_evaluations': 0,
                  'mesh_completion': False, 'mesh_decimation': False, 'geometry_smoothing': False,
                  'display_only_rigid_transform': True, 'alignment_fitting': False,
                  'source_arrays_unchanged': True, 'same_camera_and_scale_within_pairs': True,
                  'all_triangles_projected_before_viewport_clipping': True,
                  'local_crop_is_projection_viewport_only': True,
                  'h0_pair_vertices_and_triangles_array_equal': True,
                  'mesh_shading_encodes_quality': False},
    }
    (OUT / 'provenance.json').write_text(json.dumps(receipt, ensure_ascii=False, indent=2)+'\n')
    total = sum(p.stat().st_size for p in OUT.iterdir() if p.is_file())
    assert total <= LIMIT, total
    print(json.dumps({'output': str(OUT), 'total_bytes': total, 'sources_verified': len(SOURCES),
                      'figures': ['scene_overview', 'mesh_comparison', 'mesh_details'],
                      'conditions': len(CASES), 'saved_meshes': len(records)}, indent=2))


if __name__ == '__main__':
    main()
