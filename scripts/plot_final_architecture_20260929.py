#!/usr/bin/env python3
"""Draw the executed V35 interfaces as vector artwork with an unchanged user image.

This presentation-only script does not import or execute the simulator,
controller, reconstruction backend or evaluator. The user-supplied PNG is
shown in full, without crop, pixel edits, annotations on the robot or synthesis.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = ROOT / 'docs/thesis/figures/final_architecture_20260929'
PHOTO = ROOT / 'docs/thesis/assets/robot_20260928/robot_cutout_user.png'
SOURCES = (
    'nso/cpu_four_modules_v35.py',
    'nso/observation_belief_v35.py',
    'nso/online_planner_v35.py',
    'nso/surface_measurement_v34.py',
    'scripts/run_online_routes_v36.py',
)
os.environ.setdefault('MPLCONFIGDIR', '/tmp/nso-final-architecture-mpl-20260929')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
from matplotlib.path import Path as PlotPath
from PIL import Image


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def draw(output):
    output = Path(output).resolve()
    if output.exists():
        raise ValueError('Use a fresh output directory; original figures are never overwritten.')
    pins = {str(PHOTO.relative_to(ROOT)): sha(PHOTO), **{p: sha(ROOT / p) for p in SOURCES},
            str(Path(__file__).resolve().relative_to(ROOT)): sha(__file__)}
    output.mkdir(parents=True)
    plt.rcParams.update({
        'font.family': 'DejaVu Sans', 'font.size': 9,
        'pdf.fonttype': 42, 'ps.fonttype': 42,
        'svg.fonttype': 'none', 'svg.hashsalt': 'nso-final-architecture-20260929',
        'axes.unicode_minus': False,
    })
    width, height = 7.0, 7.15
    fig = plt.figure(figsize=(width, height), facecolor='white')
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set(xlim=(0, width), ylim=(0, height), aspect='equal')
    ax.axis('off')
    ink = '#233640'; muted = '#526875'; blue = '#276B93'
    teal = '#187D78'; orange = '#A96525'; edge = '#CBD6DC'
    texts, arrows, boxes = [], [], []

    def text(x, y, value, *, size=9, weight='normal', color=ink, ha='left', va='center', **kw):
        if size < 8:
            raise ValueError('All declared text sizes must be at least 8 pt.')
        item = ax.text(x, y, value, fontsize=size, fontweight=weight, color=color,
                       ha=ha, va=va, zorder=5, **kw)
        texts.append(item)
        return item

    def box(name, x, y, w, h, *, face='white', border=edge, lw=.85, dashed=False, radius=.07):
        item = FancyBboxPatch((x, y), w, h,
            boxstyle=f'round,pad=0,rounding_size={radius}', facecolor=face,
            edgecolor=border, linewidth=lw, linestyle=(0, (4, 3)) if dashed else '-', zorder=1)
        ax.add_patch(item)
        boxes.append(dict(name=name, bounds_inches=[x, y, w, h]))
        return item

    def arrow(name, points, *, color=blue, lw=1.35, head=True, dashed=False):
        path = PlotPath(points, [PlotPath.MOVETO] + [PlotPath.LINETO] * (len(points) - 1))
        ax.add_patch(FancyArrowPatch(path=path, arrowstyle='-|>' if head else '-',
            mutation_scale=9, linewidth=lw, color=color,
            linestyle=(0, (3, 3)) if dashed else '-', zorder=3,
            joinstyle='round', capstyle='round'))
        arrows.append(dict(name=name, points_inches=points, color=color, dashed=dashed))

    text(.31, 6.90, 'Category-conditioned active observation', size=13.3, weight='bold')
    text(.31, 6.63, 'Executed CPU architecture · local binary task', size=10.0, color=muted)

    box('shared_public_inputs', .31, 6.18, 6.37, .29, face='#F1F5F7', radius=.035)
    text(3.495, 6.325, 'Shared task priors: two configurations · safe pose graph · action budget',
         size=8.8, ha='center')

    # The left simulator provides the frame at the top and receives the legal
    # action at the bottom. Both channels are visible without a crossed loop.
    box('cpu_environment', .32, 3.08, 1.42, 2.96, face='#F7FAFC')
    text(.48, 5.85, 'CPU environment', size=9.1, weight='bold')
    box('acquired_observation', .48, 4.32, 1.10, 1.28, border='#7A9CB2')
    text(1.03, 5.38, 'Sensor frame', size=8.6, weight='bold', ha='center')
    text(1.03, 5.08, 'RGB-D + scan', size=8.5, ha='center')
    text(1.03, 4.86, 'pose + paid step', size=8.2, ha='center')
    text(1.03, 4.54, 'map digest only', size=8.1, color=muted, ha='center')
    box('primitive_execution', .48, 3.25, 1.10, .75, border='#7A9CB2')
    text(1.03, 3.78, 'Execute action', size=8.6, weight='bold', ha='center')
    text(1.03, 3.52, '1 m / 90°', size=8.6, ha='center')
    arrow('execution_acquires_next_frame', [(1.03, 4.0), (1.03, 4.32)])

    box('decision_layer', 1.96, 4.06, 4.72, 1.98, face='#F6FAFB', border='#ABC2CE')
    text(2.12, 5.90, 'DECISION LAYER', size=8.2, weight='bold', color=muted)
    box('OV-SDF', 2.12, 5.12, 1.97, .62, border=teal, lw=1.10)
    text(2.27, 5.56, 'OV-SDF', size=10.0, weight='bold', color=teal)
    text(2.27, 5.35, 'Category support → Ls', size=8.8)
    box('IGCR', 2.12, 4.22, 1.97, .70, border=teal, lw=1.10)
    text(2.27, 4.75, 'IGCR', size=10.0, weight='bold', color=teal)
    text(2.27, 4.52, 'Depth/scan vs. templates', size=8.5)
    text(2.27, 4.34, 'p(h0) = σ(Ls + Lg)', size=8.4)
    arrow('rgb_to_category', [(1.58, 5.37), (2.12, 5.37)], color=teal)
    arrow('measured_geometry_to_feedback', [(1.58, 4.58), (2.12, 4.58)], color=teal)
    arrow('finite_category_prior_to_belief', [(3.12, 5.12), (3.12, 4.92)], color=teal)

    box('STGHP', 4.65, 4.33, 1.86, 1.41, border=blue, lw=1.10)
    text(4.80, 5.53, 'STGHP', size=10.0, weight='bold', color=blue)
    text(4.80, 5.27, 'Belief + exposure state', size=8.3)
    text(4.80, 5.06, 'Graph + remaining budget', size=8.0)
    text(4.80, 4.76, 'One-event lookahead', size=8.5)
    text(4.80, 4.54, 'Select next atomic action', size=8.0)
    arrow('corrected_belief_to_planning', [(4.09, 4.59), (4.65, 4.59)], color=teal)
    text(4.37, 4.80, 'p(h)', size=8.4, color=teal, ha='center')
    arrow('shared_graph_and_predictions', [(5.58, 6.18), (5.58, 5.74)], color=muted, lw=.95)

    box('execution_layer', 1.96, 3.08, 4.72, .90, face='#FCF8F3', border='#DCC6AE')
    text(2.12, 3.83, 'EXECUTION LAYER', size=8.2, weight='bold', color=muted)
    box('RPN-UQ', 4.65, 3.20, 1.86, .62, border=orange, lw=1.10)
    text(4.80, 3.63, 'RPN-UQ', size=10.0, weight='bold', color=orange)
    text(4.80, 3.40, '1 + d(return) ≤ budget', size=8.3)
    arrow('proposed_action_to_guard', [(5.58, 4.33), (5.58, 3.82)], color=blue)
    arrow('approved_action_to_environment', [(4.65, 3.51), (1.58, 3.51)], color=orange)
    text(3.11, 3.27, 'legal action + return reserve', size=8.3, ha='center', color=orange)

    # Acquired frames branch to the real mapper; evaluation has no feedback
    # arrow to the controller. The map digest is an opaque recorded summary.
    arrow('acquired_frame_to_mapper', [(.48, 4.81), (.17, 4.81), (.17, 2.45), (.32, 2.45)], color=blue)
    box('measured_mapping', .32, 1.93, 2.35, .98, face='#F3F8FB', border='#94B5C8')
    text(.49, 2.70, 'Measured mapping', size=9.7, weight='bold', color=blue)
    text(.49, 2.45, '2D occupancy + 3D TSDF', size=8.8)
    text(.49, 2.19, 'Each acquired frame fused once', size=8.1)
    arrow('sealed_predictions_to_evaluation', [(2.67, 2.45), (3.03, 2.45)], color=blue)
    box('offline_evaluation', 3.03, 1.93, 3.65, .98, face='#F7F8FA', border='#BBC6CE')
    text(3.20, 2.70, 'Offline evaluation after prediction sealing', size=9.0, weight='bold')
    text(3.20, 2.44, 'Fixed floor / surface references', size=8.6)
    text(3.20, 2.20, 'Coverage · precision / recall / F1 · J5', size=8.6)
    text(.32, 1.72, 'Planning uses public forecasts; reconstructed surfaces use acquired depth.',
         size=8.4, color=muted)

    # The full RGBA image is placed without processing its pixels or aspect.
    box('optional_platform_interfaces', .32, .20, 6.36, 1.28,
        face='white', border='#AEBBC2', dashed=True)
    with Image.open(PHOTO) as im:
        if im.mode != 'RGBA' or im.width != im.height:
            raise ValueError('Expected the unmodified square user PNG.')
        ax.imshow(im, extent=(.43, 1.61, .25, 1.43), interpolation='none', zorder=2)
        photo_info = dict(width_px=im.width, height_px=im.height, mode=im.mode,
                          rendered_bounds_inches=[.43, .25, 1.18, 1.18])
    text(1.83, 1.23, 'Optional ROS 1 platform interfaces', size=10.0, weight='bold')
    text(1.83, .94, 'ZED RGB-D / pose + scan → observation input', size=8.7)
    text(1.83, .69, 'Observation target → navigation goal', size=8.7)
    text(1.83, .40, 'User-supplied image · future deployment, no physical results',
         size=8.0, color=muted)

    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    text_records = []
    for item in texts:
        bbox = item.get_window_extent(renderer).transformed(fig.dpi_scale_trans.inverted())
        if bbox.x0 < 0 or bbox.y0 < 0 or bbox.x1 > width or bbox.y1 > height:
            raise ValueError(f'Text outside figure: {item.get_text()}')
        text_records.append(dict(text=item.get_text(), font_pt=item.get_fontsize(),
                                  bounds_inches=[bbox.x0, bbox.y0, bbox.width, bbox.height]))
    for ext in ('pdf', 'svg', 'png'):
        metadata = {'CreationDate': None, 'ModDate': None} if ext == 'pdf' else (
            {'Date': None} if ext == 'svg' else {'Software': 'NSO vector architecture plotter'})
        fig.savefig(output / f'final_architecture.{ext}', dpi=320, metadata=metadata)
    plt.close(fig)
    captions = '''# Final system architecture / 最终系统架构

**English.** Executed CPU architecture for the local V35/V36 binary-configuration task. OV-SDF registers category evidence as a finite configuration prior, and IGCR compares acquired depth and scan observations with the public templates to form bounded geometric support. Their combined belief enters STGHP together with public exposure state, the common graph and remaining budget. STGHP considers at most one future diagnostic event per planning call and proposes the next atomic action; RPN-UQ checks the legal edge and the cost of returning to the starting position and orientation. Execution acquires a new sensor frame. These are four implemented interfaces within a decision/execution hierarchy; OV-SDF and IGCR share one actual belief update. Every method receives the same two public templates and navigation graph; G suppresses the category prior while retaining active geometric diagnosis and revisiting. The common 18-action prefix precedes autonomous choices. A zero-action initial frame and each subsequent paid frame enter the same measured mapper once. Mapping receives RGB-D, scan and pose data: the occupancy map uses the sensor streams, while surface TSDF integration uses acquired depth; the controller receives only an opaque measured-map digest, not measured quality. Predicted exposure is not inserted into the reconstructed surface. Final predictions are sealed before offline ground-truth evaluation; the evaluation has no feedback path to the policy. The bottom panel uses the complete, unaltered user-supplied transparent robot PNG. ROS 1, ZED output and goal-based navigation are user-reported prospective integration interfaces; the pictured hardware provides no physical performance evidence. Diagram geometry, text and arrows are program-drawn vectors, with the robot image as the only embedded raster image.

**中文。** 局部 V35/V36 二构型任务实际执行的 CPU 架构。OV-SDF将类别证据登记为有限构型先验，IGCR比较实测深度/扫描与公开模板，形成受限几何证据；二者在同一次实际信念更新中组合。STGHP依据该信念、公开曝光状态、共同安全图和剩余预算，每次规划至多预测一次未来诊断，只下发下一原子动作；RPN-UQ检查合法边及回到起始位置和朝向的代价，随后执行并取得新观测。G与S共享双模板、导航图、几何诊断和补看能力，区别为是否启用类别先验。共同18动作前缀之后才自主选择。初始零动作帧及每个付费帧均只融合一次；二维占据图使用传感流，三维TSDF仅由实际深度生成，控制器仅记录不透明的实测地图摘要，不取得真实质量。预测曝光不补入重建表面，终点预测封存后才使用真值离线评价，评价没有通往规划器的反馈。下方保留用户供给的完整透明小车PNG，未裁剪、补绘、修图或改变硬件。ROS 1、ZED和目标点导航为用户说明的后续接口，照片不表示已经完成实车性能验证。示意图的框、文字和箭头均由程序生成矢量，只有实拍机器人为嵌入位图。
'''
    (output / 'captions.md').write_text(captions)
    (output / 'layout.json').write_text(json.dumps(dict(width_inches=width, height_inches=height,
        boxes=boxes, arrows=arrows, texts=text_records, photo=photo_info), ensure_ascii=False, indent=2) + '\n')
    shutil.copyfile(__file__, output / 'source.py')
    if any(sha(ROOT / path) != pin for path, pin in pins.items()):
        raise ValueError('A source input changed during rendering.')
    manifest = dict(schema='nso.publication_architecture.v1',
        source_sha256=pins, width_inches=width, height_inches=height,
        minimum_font_pt=min(t['font_pt'] for t in text_records),
        source_photo=str(PHOTO.relative_to(ROOT)), source_photo_sha256=sha(PHOTO),
        photo_pixel_editing=False, photo_crop=False, photo_aspect_preserved=True,
        photo_display='Full original RGBA array, scaled for figure placement only.',
        scientific_results_in_figure=False, new_worlds=0, new_policy_runs=0,
        new_tsdf_integrations=0, new_surface_evaluations=0, physical_trials=0,
        generative_graphics=False, publication_scope='Executed local V35/V36 interfaces; optional hardware panel',
        font='DejaVu Sans; embedded TrueType PDF / editable SVG text',
        rendering=dict(matplotlib=matplotlib.__version__, png_dpi=320),
        files={p.name: dict(bytes=p.stat().st_size, sha256=sha(p))
               for p in sorted(output.iterdir()) if p.is_file()})
    (output / 'manifest.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps(dict(output=str(output), width_inches=width,
        minimum_font_pt=manifest['minimum_font_pt'], photo_unchanged=True,
        bytes=sum(p.stat().st_size for p in output.iterdir() if p.is_file())), ensure_ascii=False))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=DEFAULT_OUTPUT)
    draw(parser.parse_args().output)
