#!/usr/bin/env python3
"""Plot saved V42 analytic receipts; never rerun a sensor, planner or mapper."""
import argparse
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def render(source, output):
    output.mkdir(parents=True, exist_ok=False)
    inputs = [source/name for name in ('result.json', 'analytic_packet_001.npz',
        'analytic_tsdf_mesh.npz', 'mapper_snapshot.json')]
    receipt = json.loads(inputs[0].read_text())
    with np.load(inputs[1], allow_pickle=False) as packet:
        rgb = packet['rgb'].copy()
    with np.load(inputs[2], allow_pickle=False) as mesh:
        vertices, triangles, colors = [mesh[name].copy() for name in ('vertices', 'triangles', 'vertex_colors')]
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 8, 'axes.titlesize': 9,
        'axes.labelsize': 8, 'legend.fontsize': 7, 'xtick.labelsize': 7, 'ytick.labelsize': 7,
        'axes.spines.top': False, 'axes.spines.right': False, 'pdf.fonttype': 42,
        'svg.fonttype': 'none', 'axes.linewidth': .7})
    fig, axs = plt.subplots(2, 2, figsize=(7.2, 5.4), constrained_layout=False)
    fig.subplots_adjust(left=.08, right=.97, top=.9, bottom=.16, wspace=.33, hspace=.7)
    fig.suptitle('Observed evidence to view forecast and measured TSDF', y=.975, fontsize=11)
    fig.text(.5, .935, 'Analytic interface validation · two scripted noiseless RGB-D packets',
             ha='center', color='#555555', fontsize=8)
    ax = axs[0, 0]
    ax.imshow(rgb, interpolation='nearest')
    ax.set_title('(a) Second measured RGB packet', loc='left')
    ax.set_xlabel('Image column'); ax.set_ylabel('Image row')
    ax.set_xticks([0, 47, 95]); ax.set_yticks([0, 35, 71])

    palette = {'G': '#777777', 'S': '#0072B2', 'flat': '#CC79A7', 'permuted': '#D55E00'}
    labels = {'G': 'Geometry', 'S': 'Semantic', 'flat': 'Flat prior', 'permuted': 'Permuted prior'}
    x = np.arange(4)
    ax = axs[0, 1]
    for index, name in enumerate(palette):
        ax.bar(x+(index-1.5)*.19, receipt['final_posteriors'][name], width=.18,
               label=labels[name], color=palette[name])
    ax.set_xticks(x, ['Planar', 'Recessed', 'Louvered', 'Open frame'], rotation=18)
    ax.set_ylim(0, 1.05); ax.set_ylabel('Structure weight')
    ax.set_title('(b) Posterior after identical residuals', loc='left')
    ax.legend(frameon=False, ncol=2, loc='upper right')

    ax = axs[1, 0]
    markers = {'G': 'o', 'S': 's', 'flat': 'x', 'permuted': '^'}
    for name in palette:
        ax.plot(np.arange(1, 9), receipt['candidate_expected_new_surface_area_m2'][name],
                marker=markers[name], markersize=4, color=palette[name], linewidth=1,
                linestyle='--' if name == 'flat' else '-', label=labels[name])
    ax.set_xticks(range(1, 9)); ax.set_xlabel('Predeclared candidate view index')
    ax.set_ylabel('Predicted novel surface area (m²)')
    ax.set_title('(c) Uncalibrated planning proxy', loc='left')
    ax.grid(axis='y', color='#e8e8e8', linewidth=.5)

    ax = axs[1, 1]
    if len(vertices):
        ax.scatter(vertices[:, 1], vertices[:, 2], s=2., c=np.clip(colors, 0, 1),
                   linewidths=0, rasterized=True)
    ax.set_aspect('equal'); ax.set_xlabel('World y (m)'); ax.set_ylabel('World z (m)')
    ax.set_title('(d) TSDF vertices, front projection', loc='left')
    ax.text(.02, .02, f'{len(vertices):,} vertices / {len(triangles):,} triangles\nMeasured depth only; no prototype fusion',
            transform=ax.transAxes, fontsize=6.5, va='bottom',
            bbox=dict(facecolor='white', edgecolor='none', alpha=.85, pad=2.))
    fig.text(.08, .065, 'Geometry and flat-prior curves coincide. Candidate actions were not executed.\n'
        'This figure does not measure semantic reconstruction gains or multi-scene generalization.',
        fontsize=7, color='#444444', linespacing=1.6)
    for extension in ('pdf', 'svg', 'png'):
        fig.savefig(output/f'observed_planning.{extension}', dpi=220, facecolor='white')
    plt.close(fig)
    provenance = dict(scope='saved analytic interface receipts; no new computation of research performance',
        inputs={str(path.relative_to(ROOT)): sha(path) for path in inputs},
        source_script_sha256=sha(Path(__file__)),
        rendered_vertex_count=len(vertices), rendered_triangle_count=len(triangles),
        mesh_postprocessed=False, displayed_geometry='all saved TSDF vertices projected to world y/z',
        actual_development_worlds_created=0, semantic_performance_claim_added=False,
        outputs={path.name: dict(sha256=sha(path), bytes=path.stat().st_size) for path in sorted(output.iterdir())})
    (output/'provenance.json').write_text(json.dumps(provenance, indent=2)+'\n')
    print(json.dumps(dict(status='rendered_saved_receipts', outputs=str(output),
        total_bytes=sum(p.stat().st_size for p in output.iterdir()))))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, default=ROOT/'audit_results/v42_observed_planning_20260921')
    parser.add_argument('--output', type=Path, default=ROOT/'docs/thesis/figures/v42_observed_planning')
    args = parser.parse_args()
    render(args.source.resolve(), args.output.resolve())
