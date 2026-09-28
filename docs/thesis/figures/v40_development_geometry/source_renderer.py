#!/usr/bin/env python3
"""Plot materialized development geometry, explicitly not reconstructed maps."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

ROOT = Path(__file__).resolve().parents[1]
TITLES = {'A': 'Cabinet aisles', 'B': 'Shelf occlusion', 'C': 'Mixed facilities',
          'D': 'Geometry-sufficient control', 'E': 'Coverage pressure',
          'F': 'Class / structure shift'}
COLORS = ['#0072B2', '#E69F00', '#009E73', '#CC79A7']


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def run(source, output):
    meshes = sorted(source.glob('DEV_*/renderer_private/geometry.npz'))
    if len(meshes) != 6:
        raise ValueError('expected exactly six materialized development meshes')
    output.mkdir(parents=True, exist_ok=False)
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 9,
                         'pdf.fonttype': 42, 'ps.fonttype': 42,
                         'svg.fonttype': 'none', 'axes.linewidth': .6})
    fig = plt.figure(figsize=(13.2, 8.6), facecolor='white')
    rows = []
    for index, mesh in enumerate(meshes):
        name = mesh.parents[1].name
        family = name.split('_')[1]
        with np.load(mesh, allow_pickle=False) as data:
            vertices = data['vertices'].astype(float)
            triangles = data['triangles'].astype(int)
            owner = data['triangle_instance_id'].astype(int)
        assert np.isfinite(vertices).all() and owner.shape == (len(triangles),)
        faces = vertices[triangles]
        ax = fig.add_subplot(2, 3, index + 1, projection='3d')
        # The unchanged source mesh is shown without adding hypothetical surfaces.
        for instance in sorted(set(owner.tolist())):
            selected = faces[owner == instance]
            color = '#DBE1E6' if instance == -1 else COLORS[instance % len(COLORS)]
            ax.add_collection3d(Poly3DCollection(selected, facecolor=color,
                edgecolor='#B3BEC7' if instance == -1 else color,
                linewidth=.15, alpha=.13 if instance == -1 else .9,
                rasterized=False))
        lower, upper = vertices.min(axis=0), vertices.max(axis=0)
        spans = np.maximum(upper - lower, .1)
        ax.set(xlim=(lower[0], upper[0]), ylim=(lower[1], upper[1]),
               zlim=(0, max(upper[2], 1.8)))
        ax.set_box_aspect((spans[0], spans[1], max(upper[2], 1.8)))
        ax.view_init(elev=34, azim=-58)
        ax.set_xlabel('x (m)', labelpad=-1)
        ax.set_ylabel('y (m)', labelpad=-1)
        ax.set_zlabel('z (m)', labelpad=-3)
        ax.tick_params(labelsize=7, pad=0)
        ax.set_title(f'{family}  {TITLES[family]}', loc='left', fontweight='semibold', pad=4)
        for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
            axis.pane.fill = False
            axis._axinfo['grid']['linewidth'] = .35
            axis._axinfo['grid']['color'] = (.83, .86, .89, .7)
        rows.append({'parent': name, 'mesh_file': str(mesh.relative_to(ROOT)),
                     'mesh_sha256': sha(mesh), 'vertices': len(vertices),
                     'triangles': len(triangles), 'instances': len(set(owner.tolist()) - {-1}),
                     'bounds_m': [lower.tolist(), upper.tolist()]})
    fig.suptitle('Development scene geometry', x=.05, y=.985, ha='left', fontsize=18, fontweight='semibold')
    fig.text(.05, .947, 'Ground-truth assets for development only | No sensor trajectory or reconstruction result',
             fontsize=10, color='#5B6770')
    fig.legend(handles=[Patch(facecolor=c, label=f'Instance {i + 1}') for i, c in enumerate(COLORS)],
               loc='lower center', bbox_to_anchor=(.5, .03), ncol=4, frameon=False)
    fig.text(.05, .016, 'Color identifies facilities, not semantic confidence or error. All axes are metric; panel extents differ.',
             fontsize=9, color='#5B6770')
    fig.subplots_adjust(left=.025, right=.98, bottom=.1, top=.89, wspace=.08, hspace=.07)
    for extension in ('png', 'pdf', 'svg'):
        fig.savefig(output / ('development_scenes.' + extension), dpi=200, facecolor='white')
    plt.close(fig)
    result = {'status': 'rendered', 'scope': 'six GT development assets; not six experiments',
              'source_rows': rows, 'new_worlds': 0, 'new_sensor_trajectories': 0,
              'new_reconstructions': 0, 'performance_claim': False,
              'camera': {'elevation': 34, 'azimuth': -58}, 'source_sha256': sha(Path(__file__)),
              'outputs': {p.name: {'sha256': sha(p), 'bytes': p.stat().st_size}
                          for p in sorted(output.iterdir())}}
    if sum(p.stat().st_size for p in output.iterdir()) > 16 * 1024**2:
        raise RuntimeError('figure output exceeds declared 16 MiB limit')
    (output / 'manifest.json').write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps({'status': result['status'], 'scenes': len(rows), 'output': str(output)}))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, default=ROOT / 'audit_results/v40_p1_development_geometry_20260920')
    parser.add_argument('--output', type=Path, default=ROOT / 'docs/thesis/figures/v40_development_geometry')
    args = parser.parse_args()
    run(args.source.resolve(), args.output.resolve())
