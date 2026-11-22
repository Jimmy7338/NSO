#!/usr/bin/env python3
"""Render actual trajectories and matched source surfaces from sealed results."""

import argparse
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.lines import Line2D
from matplotlib.ticker import MaxNLocator
import numpy as np

from study import PRIOR, PROTOCOL, ROOT, RUNS, ARMS, read


COLORS = {'category': '#0072B2', 'geometry': '#B34A00', 'both': '#16836E'}
LAYOUTS = {'paired_chain_17': 'Chain 17', 'paired_cycle_20': 'Cycle 20'}
FACILITIES = {'open31': 'Open shelf', 'closed118': 'Closed cabinet'}


def world_xy(positions, rows, resolution):
    cells = np.asarray(positions)
    return (cells[:, [1, 0]] * np.array([1, -1]) + [0.5, rows - 0.5]) * resolution


def save(fig, stem):
    fig.savefig(stem.with_suffix('.pdf'), bbox_inches='tight')
    fig.savefig(stem.with_suffix('.png'), bbox_inches='tight', dpi=240)
    plt.close(fig)


def actual_trajectory(out, cases):
    fig, axes = plt.subplots(1, 2, figsize=(10.2, 5.1), layout='constrained')
    for ax, case in zip(axes, cases, strict=True):
        layout, fork = case['layout'], case['paid_before']
        scene = PRIOR / 'scenes'
        initial = read(scene/'prepared'/layout/'initial_state.json')
        with np.load(scene/'evaluator_references'/layout/'evaluator_truth.npz', allow_pickle=False) as truth:
            occupied = truth['occupancy']
        resolution = initial['grid_resolution_m']
        height, width = occupied.shape
        extent = (0, width * resolution, 0, height * resolution)
        ax.imshow(occupied, cmap=ListedColormap(['#FCFCFD', '#CBD2D9']), origin='upper',
                  extent=extent, interpolation='nearest', zorder=0)
        actual, manifests = {}, {}
        for arm in ARMS:
            directory = RUNS / layout / arm
            manifests[arm] = read(directory/'manifest.json')
            xy = [initial['anchor'][:2], *(row['position'] for row in read(directory/'steps.json'))]
            actual[arm] = world_xy(xy, height, resolution)
        np.testing.assert_array_equal(actual['category'][:fork+1], actual['geometry'][:fork+1])
        goal_pose = read(RUNS/layout/'category/intervention.json')['target_pose']
        assert goal_pose == read(RUNS/layout/'geometry/intervention.json')['target_pose']
        goal = world_xy([goal_pose[:2]], height, resolution)[0]
        prefix = actual['category'][:fork+1]
        ax.plot(*prefix.T, color='#5C6574', lw=1.5, zorder=2)
        for arm, pattern in (('category', '-'), ('geometry', (0, (4, 2)))):
            path = actual[arm]
            event = manifests[arm]['route_end']
            end = event['actual_paid']
            ax.plot(*path[end:].T, color=COLORS[arm], lw=1.2, alpha=.28, ls=pattern, zorder=1)
            ax.plot(*path[fork:end+1].T, color=COLORS[arm], lw=2.4, ls=pattern, zorder=3)
            marker = 'o' if event['status'] == 'arrived' else 'x'
            ax.scatter(*path[end], s=46, marker=marker, c=COLORS[arm], zorder=5)
        ax.scatter(*prefix[-1], s=56, marker='D', facecolor='white', edgecolor='#263649', linewidth=1.2, zorder=7)
        ax.scatter(*goal, s=74, marker='+', color='#19283A', linewidth=1.4, zorder=8)
        ax.set(xlabel='World x (m)', ylabel='World y (m)', xlim=extent[:2], ylim=extent[2:], aspect='equal')
        ax.set_title(f"{LAYOUTS[layout]}  ·  fork at action {fork}", loc='left', weight='semibold')
        actual_event = [f"{arm.capitalize()} {manifests[arm]['route_end']['status']} at {manifests[arm]['route_end']['actual_paid']}"
                        for arm in ARMS]
        ax.text(.98, .03, '\n'.join(actual_event), transform=ax.transAxes, ha='right', va='bottom',
                fontsize=8.1, bbox=dict(facecolor='white', alpha=.90, edgecolor='none', pad=3))
        ax.spines[['top','right']].set_visible(False)
    legend = [Line2D([], [], color='#5C6574', lw=1.7, label='Shared executed prefix'),
              Line2D([], [], color=COLORS['category'], lw=2.3, label='Category route'),
              Line2D([], [], color=COLORS['geometry'], ls='--', lw=2.3, label='Geometry route'),
              Line2D([], [], color='#263649', marker='D', markerfacecolor='white', ls='', label='Fork pose'),
              Line2D([], [], color='#263649', marker='+', ls='', label='Shared goal')]
    fig.legend(handles=legend, loc='outside lower center', ncol=5, frameon=False, columnspacing=1.2)
    save(fig, out/'executed_routes')


def show_surface(ax, reference, category, geometry, before, title):
    cat_new = np.logical_and(category, ~before)
    geo_new = np.logical_and(geometry, ~before)
    memberships = (('both', cat_new & geo_new), ('category', cat_new & ~geo_new),
                   ('geometry', geo_new & ~cat_new))
    scene_center = (reference.max(axis=0) + reference.min(axis=0)) / 2
    xyz = reference - [*scene_center[:2], 0]
    ax.scatter(*xyz.T, c='#9EA5AF', s=1.5, alpha=.18, linewidths=0, depthshade=False, rasterized=True)
    for label, masked in memberships:
        ax.scatter(*xyz[masked].T, c=COLORS[label], s=2.5, alpha=.90, linewidths=0,
                   depthshade=False, rasterized=True)
    ranges = np.maximum(np.ptp(xyz, axis=0), .15)
    ax.set_box_aspect(ranges, zoom=.86)
    ax.view_init(elev=24, azim=-55)
    ax.set_proj_type('ortho')
    ax.set(xlabel='x (m)', ylabel='y (m)', zlabel='z (m)')
    for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
        axis.set_major_locator(MaxNLocator(nbins=4))
    ax.tick_params(labelsize=7, pad=1)
    ax.set_title(title, fontsize=9, pad=8)


def measured_surfaces(out, cases, measurements, result_rows):
    fig, axes = plt.subplots(2, 2, figsize=(9.4, 8.3), subplot_kw={'projection':'3d'})
    fig.subplots_adjust(left=.02, right=.98, bottom=.10, top=.94, hspace=.19, wspace=.07)
    for row_axes, case, measured in zip(axes, cases, result_rows, strict=True):
        layout = case['layout']
        target = measured['primary_target']
        for ax, stage in zip(row_axes, ('equal_clock','terminal'), strict=True):
            if stage == 'equal_clock':
                clock = measured['shared_route_evaluation_paid']
                caption = f'Paired action {clock}' if clock is not None else 'Paired action'
            else:
                ending = [measured['arms'][arm]['terminal'].get('receipt', {}).get('actual_paid') for arm in ARMS]
                caption = f'Terminal {ending[0]}' if ending[0] == ending[1] else f'Terminals {ending[0]}/{ending[1]}'
            title = f'{LAYOUTS[layout]} · {FACILITIES[target]} · {caption}'
            path = measurements/'surface_masks'/f'{layout}_{stage}.npz'
            if not path.exists():
                ax.set_axis_off()
                ax.set_title(f'{title} unavailable')
                continue
            with np.load(path, allow_pickle=False) as masks:
                keys = ('points_xyz_m','category_represented','geometry_represented','fork_represented')
                show_surface(ax, *(masks[f'{target}_{key}'] for key in keys), title)
    handles = [Line2D([], [], marker='o', ls='', ms=5.4, color=COLORS[label], label=description)
               for label, description in (('both','New in both'),('category','New only on category route'),
                                          ('geometry','New only on geometry route'))]
    handles.append(Line2D([], [], marker='o', ls='', ms=5, color='#9EA5AF', alpha=.55, label='Evaluable reference'))
    fig.legend(handles=handles, loc='lower center', bbox_to_anchor=(.5, .045), ncol=4,
               frameon=False, columnspacing=1.1)
    save(fig, out/'measured_facility_surfaces')


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument('--measurement', type=Path, default=ROOT/'evaluation/results')
    cli.add_argument('--out', type=Path, default=ROOT/'results')
    args = cli.parse_args()
    args.out.mkdir(exist_ok=True, parents=True)
    plt.rcParams.update({'font.size':9, 'font.family':'DejaVu Sans', 'pdf.fonttype':42, 'axes.labelpad':3})
    cases = read(PROTOCOL)['cases']
    by_layout = read(args.measurement/'measured.json')['cases']
    rows = [by_layout[case['layout']] for case in cases]
    actual_trajectory(args.out, cases)
    measured_surfaces(args.out, cases, args.measurement, rows)


if __name__ == '__main__':
    main()
