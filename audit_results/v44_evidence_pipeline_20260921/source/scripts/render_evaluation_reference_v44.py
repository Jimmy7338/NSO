#!/usr/bin/env python3
"""Publication-sized static reference illustration, never a reconstruction plot."""
import argparse
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def run(source, output):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection
    import numpy as np
    from nso.offline_evaluation_v44 import load_reference_v44, sha256
    source, output = Path(source).resolve(), Path(output).resolve()
    declaration = json.loads((source/'reference_predeclaration.json').read_text())
    assert declaration['actual_development_episode_count'] == 0
    output.mkdir(parents=True, exist_ok=False)
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 9,
        'axes.titlesize': 11, 'axes.labelsize': 8, 'xtick.labelsize': 7, 'ytick.labelsize': 7,
        'pdf.fonttype': 42, 'ps.fonttype': 42, 'svg.hashsalt': 'nso-v44-static-reference',
        'savefig.facecolor': 'white'})
    fig = plt.figure(figsize=(9.2, 7.3), layout='constrained')
    grid = fig.add_gridspec(2, 2, width_ratios=(1., 1.08))
    inputs = {'reference_predeclaration.json': sha256(source/'reference_predeclaration.json')}
    records = []
    palette = ['#0072B2', '#D55E00', '#009E73', '#CC79A7', '#E69F00', '#56B4E9']
    for row, item in enumerate(declaration['references']):
        asset = item['asset_id']; folder = source/'references'/asset
        reference, domain, record = load_reference_v44(folder,
            manifest_sha256=item['manifest_sha256'], asset_id=asset)
        for path in folder.iterdir():
            if path.is_file(): inputs[str(path.relative_to(source))] = sha256(path)
        descriptor = record['coverage']; origin = descriptor['origin_xy_m']; resolution = descriptor['resolution_m']
        extent = [origin[0], origin[0]+domain.shape[1]*resolution,
                  origin[1], origin[1]+domain.shape[0]*resolution]
        ax = fig.add_subplot(grid[row, 0])
        ax.imshow(domain, origin='lower', extent=extent, interpolation='nearest',
                  cmap=ListedColormap(['#EBEDF0', '#A8CFDF']), vmin=0, vmax=1)
        candidates = json.loads((folder/'candidate_views.json').read_text())
        start = candidates['start_xy_m']
        ax.scatter(*start, marker='*', s=90, color='#D55E00', edgecolors='white', linewidths=.55, zorder=5)
        ax.set(xlabel='World x (m)', ylabel='World y (m)', aspect='equal')
        ax.set_title(f"{'(a)' if row == 0 else '(c)'}  {asset}: fixed navigation domain", loc='left', fontweight='bold')
        ax.text(.02, .98, f"{descriptor['denominator_area_m2']:.2f} m² · {descriptor['denominator_cells']:,} cells",
                transform=ax.transAxes, va='top', fontsize=8,
                bbox=dict(facecolor='white', edgecolor='none', alpha=.92, pad=3))
        for spine in ax.spines.values(): spine.set_color('#9CA3AF'); spine.set_linewidth(.6)
        ax3 = fig.add_subplot(grid[row, 1], projection='3d')
        targets = reference.triangle_instance_id >= 0
        triangles = reference.vertices[reference.triangles[targets]]
        ax3.add_collection3d(Poly3DCollection(triangles, facecolors='#D9DEE4', edgecolors='none', alpha=.2))
        ids = record['target_instance_inventory']
        for index, instance in enumerate(ids):
            points = reference.points[reference.point_instance_id == instance]
            ax3.scatter(points[:, 0], points[:, 1], points[:, 2], s=2.4,
                        c=palette[index % len(palette)], depthshade=False, label=f'F{index+1}')
        ax3.set(xlim=(extent[0], extent[1]), ylim=(extent[2], extent[3]),
                zlim=(0, max(2., float(reference.points[:, 2].max())+.1)),
                xlabel='x (m)', ylabel='y (m)', zlabel='z (m)')
        ax3.set_box_aspect((extent[1]-extent[0], extent[3]-extent[2], 3.5))
        ax3.view_init(elev=28, azim=-60)
        ax3.tick_params(labelsize=6, pad=0)
        ax3.set_title(f"{'(b)' if row == 0 else '(d)'}  Fixed observable facility surface", loc='left', fontweight='bold')
        ax3.text2D(.02, .98, f'{len(ids)} facilities · {len(reference.points):,} reference samples',
                   transform=ax3.transAxes, va='top', fontsize=8)
        ax3.legend(loc='lower left', ncol=len(ids), frameon=False, fontsize=7,
                   handletextpad=.2, columnspacing=.7, markerscale=2)
        for axis in (ax3.xaxis, ax3.yaxis, ax3.zaxis):
            axis.pane.fill = False
            axis._axinfo['grid']['color'] = (.85, .87, .89, .55)
        records.append(dict(asset_id=asset, reference_manifest_sha256=item['manifest_sha256'],
            domain_cells=descriptor['denominator_cells'], samples=len(reference.points)))
    fig.suptitle('Static evaluation references — no trajectory or reconstructed prediction',
                 fontsize=12, fontweight='bold')
    for suffix in ('pdf', 'svg', 'png'):
        fig.savefig(output/f'evaluation_reference.{suffix}', dpi=240)
    plt.close(fig)
    provenance = dict(scope='static development reference illustration; no performance results',
        input_sha256=inputs, generator_source_sha256=sha256(Path(__file__)), references=records,
        physical_actions=0, new_worlds=0, reconstruction_visualization=False,
        output_sha256={p.name: sha256(p) for p in sorted(output.iterdir()) if p.is_file()})
    (output/'provenance.json').write_text(json.dumps(provenance, indent=2)+'\n')
    print(json.dumps(dict(output=str(output), references=len(records), performance_result=False)))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    run(args.source, args.output)
