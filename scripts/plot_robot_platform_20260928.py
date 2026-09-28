#!/usr/bin/env python3
"""Place an unaltered user photo and vector labels in a scientific platform figure.

The original JPEGs remain byte-identical. This script lays out a photograph and
an interface diagram; it performs no retouching or synthesized reconstruction.
"""
import hashlib
import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ASSETS = ROOT / 'docs/thesis/assets/robot_20260928'
OUT = ROOT / 'docs/thesis/figures/robot_platform_20260928'
os.environ.setdefault('MPLCONFIGDIR', str(ROOT / 'tmp/mpl-robot-platform'))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
from PIL import Image


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    photos = sorted(ASSETS.glob('*.jpg'))
    selected = ASSETS / '20260928-142915.jpg'
    assert len(photos) == 2 and selected in photos
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 8.5,
                         'pdf.fonttype': 42, 'ps.fonttype': 42,
                         'svg.fonttype': 'none', 'svg.hashsalt': 'nso-robot-20260928'})
    fig = plt.figure(figsize=(7.05, 5.10), facecolor='white')
    ax = fig.add_axes([.025, .09, .485, .86])
    with Image.open(selected) as photo:
        ax.imshow(photo)
        width, height = photo.size
    ax.set_axis_off()
    ax.set_title('(a) Available mobile platform', loc='left', pad=8,
                 fontsize=9, fontweight='bold')
    locations = [(1, .44, .333), (2, .51, .137),
                 (3, .38, .44), (4, .49, .69)]
    for number, x, y in locations:
        ax.text(x*width, y*height, str(number), ha='center', va='center',
                fontsize=9, fontweight='bold', color='#15324B',
                bbox={'boxstyle': 'circle,pad=.28', 'fc': 'white',
                      'ec': '#0072B2', 'lw': 1.25, 'alpha': .96})
    right = fig.add_axes([.54, .09, .44, .86])
    right.set(xlim=(0, 1), ylim=(0, 1))
    right.set_axis_off()
    right.set_title('(b) Sensors and interface mapping', loc='left', pad=8,
                    fontsize=9, fontweight='bold')
    entries = [
        ('1', 'ZED stereo camera', 'RGB-D observations and camera pose'),
        ('2', 'Laser scanner', 'Geometric scan observations'),
        ('3', 'Positioning equipment', 'GNSS / IMU inputs'),
        ('4', 'Mobile base', 'ROS 1 navigation goals supported'),
    ]
    for index, (number, title, detail) in enumerate(entries):
        y = .955-index*.117
        right.text(.025, y, number, color='#0072B2', fontweight='bold',
                   va='top', fontsize=10)
        right.text(.10, y, title, fontweight='bold', va='top', fontsize=8.7)
        right.text(.10, y-.043, detail, va='top', fontsize=7.0, color='#52616B')
    right.plot([0, 1], [.49, .49], color='#C9D2D8', lw=.7)
    right.text(0, .455, 'Observation-to-goal connection', fontsize=8.4,
               fontweight='bold', va='top')
    for index, label in enumerate([
        'RGB-D / scan / pose observations',
        'Category prior + geometric feedback',
        'Budgeted observation goal',
        'ROS 1 navigation interface',
    ]):
        y = .355-index*.085
        right.add_patch(FancyBboxPatch((.025, y), .94, .060,
                        boxstyle='round,pad=.008,rounding_size=.015',
                        facecolor='#F1F6F9', edgecolor='#91A7B5', lw=.75))
        right.text(.495, y+.030, label, ha='center', va='center', fontsize=7.4)
        if index < 3:
            right.annotate('', xy=(.495, y-.020), xytext=(.495, y-.006),
                           arrowprops={'arrowstyle': '-|>', 'color': '#52616B',
                                       'lw': .7, 'mutation_scale': 7})
    fig.text(.50, .035,
             'Laboratory platform and planned interfaces. Quantitative results use the virtual environment.',
             ha='center', fontsize=7.0, color='#52616B')
    for ext in ('pdf', 'svg', 'png'):
        metadata = {'CreationDate': None, 'ModDate': None} if ext == 'pdf' else (
            {'Date': None} if ext == 'svg' else None)
        fig.savefig(OUT/f'robot_platform.{ext}', dpi=260, metadata=metadata)
    plt.close(fig)
    captions = '''# Robot platform and interfaces

**中文。** 用户提供的小车平台及观测—导航接口对应关系。照片按原始构图展示，数字标注为矢量叠加；未裁除、补绘或替换硬件。可见部件包括双目相机、激光雷达、定位设备和移动底盘。ROS 1、ZED深度/位姿输出、GNSS/IMU配置及目标点自主导航能力依据用户说明；未通过本图测定传感器型号、标定或实车性能。右侧为本方法与已有接口的连接示意，论文性能结论来自虚拟实验。

**English.** User-supplied mobile platform and observation-to-navigation interfaces. The original photograph is shown with vector number labels; hardware is neither retouched nor synthesized. ROS 1, ZED depth/pose output, GNSS/IMU configuration and goal-based navigation are user-reported. The interface diagram illustrates a connection for future deployment; the photograph is not evidence of physical performance validation.
'''
    (OUT/'captions.md').write_text(captions)
    record = {'selected_photo': str(selected.relative_to(ROOT)),
              'source_photos': {p.name: {'sha256': sha(p), 'bytes': p.stat().st_size}
                                for p in photos},
              'photo_pixel_editing': False,
              'layout': 'full original photo with vector numbered annotations',
              'physical_performance_trials': 0,
              'experimental_scores_from_photos': False,
              'hardware_details_source': 'user-provided photographs and prior ROS 1/sensor/navigation replies',
              'source_script_sha256': sha(Path(__file__)),
              'output_sha256': {p.name: sha(p) for p in sorted(OUT.iterdir())
                                 if p.name != 'manifest.json'}}
    (OUT/'manifest.json').write_text(json.dumps(record, ensure_ascii=False, indent=2)+'\n')
    print(json.dumps({'figure': str(OUT/'robot_platform.pdf'), 'source_photos': len(photos)}))


if __name__ == '__main__':
    main()
