#!/usr/bin/env python3
"""Render the pinned saved-distance supplement; never recompute mesh distances."""
import argparse
from pathlib import Path
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from nso.episode_driver_v43 import canonical_bytes, file_sha256
from nso.offline_evaluation_v44 import _array_sha
from nso.semantic_experiment import read_json


COLOR_MAX_M = 1.0


def ecdf(distance, weight):
    if (distance.ndim != 1 or weight.shape != distance.shape or not len(distance)
            or not np.isfinite(distance).all() or np.any(distance < 0)
            or not np.isfinite(weight).all() or np.any(weight <= 0)):
        raise ValueError('this finite-distance figure requires every saved point and positive weight')
    order = np.argsort(distance, kind='stable')
    support, first = np.unique(distance[order], return_index=True)
    probability = np.cumsum(np.add.reduceat(weight[order], first))/float(weight.sum())
    probability[-1] = 1.
    return np.r_[0., support], np.r_[0., probability]


def plot(*, receipt, receipt_sha256, output_dir):
    receipt, output = Path(receipt).resolve(), Path(output_dir).resolve()
    if output.exists() or output.is_relative_to(receipt.parent):
        raise ValueError('new figure directory outside the immutable distance supplement required')
    if file_sha256(receipt) != receipt_sha256:
        raise ValueError('explicit receipt SHA256 mismatch')
    record = read_json(receipt)
    if (record.get('schema') != 'semantic.continuous_distance.supplement.v1'
            or record.get('status') != 'continuous_distance_completed'
            or record.get('run_id') != 'core_P00_nom_G_b120_lexicographic'
            or record.get('primary_numerical_evaluation_recomputed') is not False
            or record['summary'].get('all_task_instances_in_macro_denominator') is not True):
        raise ValueError('complete whole-facility continuous-distance receipt required')
    npz = receipt.parent/record['distance_archive']['path']
    if file_sha256(npz) != record['distance_archive']['sha256']:
        raise ValueError('saved distance archive differs from pinned receipt')
    with np.load(npz, allow_pickle=False) as saved:
        if set(saved.files) != set(record['distance_archive']['arrays']):
            raise ValueError('complete saved distance array inventory required')
        data = {key:saved[key] for key in saved.files}
    for key, value in data.items():
        row = record['distance_archive']['arrays'][key]
        if (value.dtype.str != row['dtype'] or list(value.shape) != row['shape']
                or _array_sha(value) != row['sha256']):
            raise ValueError('saved array pin mismatch: '+key)
    summary = record['summary']; ids = data['reference_instance_id']
    d = data['reference_distance_m']; p = data['reference_points_m']
    macro = data['reference_equal_facility_macro_weight']
    pd, pw = data['prediction_distance_to_full_scene_m'], data['prediction_area_weight_m2']
    if (len(d) != summary['reference_samples'] or p.shape != (len(d), 3)
            or len(pd) != summary['sampling']['prediction_samples']
            or np.unique(ids).tolist() != summary['all_task_instance_ids']):
        raise ValueError('full saved sample/facility counts differ')
    for key in np.unique(ids):
        if not np.isclose(macro[ids == key].sum(), 1./len(np.unique(ids)), rtol=1e-12, atol=1e-14):
            raise ValueError('equal-facility macro mass is inconsistent')
    rx, ry = ecdf(d, macro); px, py = ecdf(pd, pw)
    bg = summary['prediction_to_full_scene']['nearest_gt_background_area_m2']/float(pw.sum())
    threshold = record['unchanged_primary_metrics_from_review']['threshold_m']
    source_sha = file_sha256(__file__)
    plt.rcParams.update({'font.family':'DejaVu Sans', 'font.size':10.5, 'axes.titlesize':11,
        'axes.labelsize':10.5, 'legend.fontsize':9, 'xtick.labelsize':9, 'ytick.labelsize':9,
        'pdf.fonttype':42, 'ps.fonttype':42, 'savefig.facecolor':'white', 'path.simplify':False})
    fig = plt.figure(figsize=(12.7, 5.7), facecolor='white')
    grid = fig.add_gridspec(1, 2, left=.067, right=.955, bottom=.17, top=.84, wspace=.18)
    ax = fig.add_subplot(grid[0,0]); cloud = fig.add_subplot(grid[0,1], projection='3d')
    blue, orange = '#0072B2', '#D55E00'
    ax.step(rx, ry, where='post', color=blue, lw=1.8,
        label='Facility reference → prediction\nEqual facility weights')
    ax.step(px, py, where='post', color=orange, lw=1.6,
        label='Prediction → full scene GT\nArea weights; includes background')
    xmax = np.ceil(max(float(d.max()), float(pd.max()))/.1)*.1
    ax.axvline(threshold, color='#606060', ls=(0,(3,3)), lw=1.)
    ax.text(threshold+.01, .45, '5 cm reference', rotation=90, color='#555555', fontsize=8.5)
    ax.set(xlim=(0.,xmax), ylim=(0.,1.025), xlabel='Surface distance (m)', ylabel='Cumulative weighted fraction')
    ax.set_title('(a) Different error domains', loc='left', pad=12, fontweight='semibold')
    ax.grid(axis='y', color='#dddddd', lw=.6)
    for spine in ('top','right'):
        ax.spines[spine].set_visible(False)
    ax.legend(loc='lower left', frameon=False, bbox_to_anchor=(.015,.015))
    zoom = ax.inset_axes([.55,.15,.41,.37])
    zoom.step(rx, ry, where='post', color=blue, lw=1.25)
    zoom.step(px, py, where='post', color=orange, lw=1.25)
    zoom.axvline(threshold, color='#606060', ls=(0,(3,3)), lw=.8)
    zoom.set(xlim=(0.,.06), ylim=(0.,1.025), xticks=[0.,.03,.06], yticks=[0.,.5,1.])
    zoom.tick_params(labelsize=7.5)
    zoom.set_title('Near-surface detail (m)', fontsize=8, pad=4)
    zoom.grid(axis='y', color='#e4e4e4', lw=.5)
    colors = cloud.scatter(p[:,0], p[:,1], p[:,2], c=d, s=10, cmap='viridis',
        norm=Normalize(vmin=0., vmax=COLOR_MAX_M, clip=True), edgecolors='none',
        depthshade=False, rasterized=True)
    for key in np.unique(ids):
        center = np.mean(p[ids == key], axis=0)
        cloud.text(center[0], center[1], float(p[ids == key,2].max())+.15, f'Facility {key}',
            fontsize=8, ha='center', bbox=dict(facecolor='white', edgecolor='none', alpha=.8, pad=1.))
    span = np.ptp(p, axis=0)
    cloud.set_box_aspect(span)  # True metric aspect; height is not exaggerated.
    cloud.view_init(elev=27, azim=-57)
    cloud.set(xlabel='x (m)', ylabel='y (m)', zlabel='z (m)')
    cloud.set_zlim(0., max(1.8, float(p[:,2].max())+.3))
    cloud.set_zticks([0., .8, 1.6])
    cloud.tick_params(labelsize=8, pad=0)
    cloud.set_title('(b) All fixed facility reference samples', loc='left', pad=12, fontweight='semibold')
    for axis in (cloud.xaxis, cloud.yaxis, cloud.zaxis):
        axis.pane.fill = False
        axis._axinfo['grid']['color'] = '#e8e8e8'
    cloud.set_position([.565, .19, .35, .63])
    color_axis = fig.add_axes([.746, .754, .19, .018])
    bar = fig.colorbar(colors, cax=color_axis, orientation='horizontal',
        extend='max' if np.any(d > COLOR_MAX_M) else 'neither')
    bar.set_label('Reference → prediction distance (m)', fontsize=8)
    bar.ax.tick_params(labelsize=8)
    cloud.text2D(.05,-.06, f'{len(d):,} / {len(d):,} samples · color cap {COLOR_MAX_M:.1f} m '
        f'({np.count_nonzero(d > COLOR_MAX_M)} above cap)', transform=cloud.transAxes, fontsize=8.5)
    fig.suptitle('P00 · Geometry baseline · Saved endpoint', x=.067, ha='left', y=.97,
        fontsize=15, fontweight='semibold')
    fig.text(.067,.897, f'4 facilities  |  {len(d):,} reference samples  |  {len(pd):,} complete prediction samples',
        fontsize=10, color='#444444')
    fig.text(.067,.067, f'{bg:.2%} of predicted area has background as its nearest GT surface. '
        'Small whole-scene error does not imply facility completeness.', fontsize=9.5, color='#333333')
    output.mkdir(parents=True)
    (output/'plot.source.py').write_bytes(Path(__file__).read_bytes())
    png, pdf = output/'continuous_distance.png', output/'continuous_distance.pdf'
    fig.savefig(png, dpi=300)
    fig.savefig(pdf, metadata={'Title':'P00 saved endpoint: distinct distance domains',
        'Subject':'Complete fixed reference samples and whole-prediction background-inclusive geometry; descriptive supplement'})
    plt.close(fig)
    if (file_sha256(receipt) != receipt_sha256 or file_sha256(npz) != record['distance_archive']['sha256']
            or file_sha256(__file__) != source_sha):
        raise ValueError('input/source changed while rendering')
    artifact = dict(schema='semantic.continuous_distance.figure.v1',
        run_id=record['run_id'], receipt_path=str(receipt), receipt_sha256=receipt_sha256,
        distances_path=str(npz), distances_sha256=record['distance_archive']['sha256'],
        source_sha256=source_sha, plotted_reference_samples=len(d), plotted_prediction_samples=len(pd),
        plotted_facility_ids=np.unique(ids).tolist(), all_reference_points_in_scatter=True,
        all_saved_samples_in_ecdf=True, reference_ecdf_weights='equal_facility_macro',
        prediction_ecdf_weights='full_prediction_area', domains_identical=False,
        background_nearest_surface_area_fraction=bg, color_range_m=[0.,COLOR_MAX_M],
        color_only_clipped_samples=int(np.count_nonzero(d > COLOR_MAX_M)),
        geometric_aspect='metric range ratios; no vertical exaggeration',
        threshold_line_m=threshold, threshold_line_changes_no_metric=True,
        distances_recomputed=False, new_worlds=0, primary_scores_recomputed=False,
        outputs={p.name:dict(sha256=file_sha256(p),bytes=p.stat().st_size) for p in (png,pdf)},
        matplotlib_version=matplotlib.__version__, visual_review='pending')
    (output/'figure_manifest.json').write_bytes(canonical_bytes(artifact))
    return artifact


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--receipt', type=Path, required=True)
    parser.add_argument('--receipt-sha256', required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    print(canonical_bytes(plot(**vars(parser.parse_args()))).decode(), end='')


if __name__ == '__main__':
    main()
