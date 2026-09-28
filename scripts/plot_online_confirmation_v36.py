#!/usr/bin/env python3
"""Static V36 scientific figures from the terminal sealed batch, without reruns."""
import argparse
import io
import os
from pathlib import Path
import sys

os.environ.setdefault('MPLBACKEND', 'Agg')
os.environ.setdefault('MPLCONFIGDIR', '/tmp/nso_v36_matplotlib')
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from nso.research_evidence_v31 import (ROOT, freeze, read, seal, sha,
    verify_inventory, verify_sources, write, write_bytes)
from scripts.run_online_routes_v36 import OUTPUT as BATCH, frozen, verify_seal


OUTPUT = ROOT/'audit_results/v36_confirmation_figures_20260918'
PARENTS = ('P00', 'P01')
MODES = ('G', 'S')
THRESHOLDS = ('02cm', '05cm', '10cm')


def run():
    # No score is read until the terminal batch seal has verified successfully.
    frozen()
    verify_seal(BATCH, 'final_seal.json')
    result = read(BATCH/'result.json')
    if OUTPUT.exists():
        raise FileExistsError('V36 figure pack cannot be overwritten')
    OUTPUT.mkdir()
    inputs = {str(p.relative_to(ROOT)): sha(p)
              for p in sorted(BATCH.rglob('*')) if p.is_file()}
    freeze(OUTPUT, [Path(__file__)], input_sha256=inputs,
        scope='Terminal saved V36 scores only; no new World, observation, controller, DP, TSDF or quality evaluation')
    rows = [dict(parent=cell['parent'], hypothesis=cell['hypothesis'],
                 mode=cell['mode'], physical_case=cell['physical_case'],
                 measurement=cell['stages']['final']['measurement'],
                 independent_replay_passed=cell['independent_replay_passed'])
            for cell in result['table'] if cell['measured']]
    scores = {(r['parent'], r['hypothesis'], r['mode']): r['measurement'] for r in rows}
    complete = (result['full_matrix_collected'] is True and len(rows) == 8
                and set(scores) == {(p, h, m) for p in PARENTS for h in (0, 1) for m in MODES})
    write(OUTPUT, OUTPUT/'plot_data.json', dict(rows=rows,
        source_result_sha256=sha(BATCH/'result.json'),
        source_final_seal_sha256=sha(BATCH/'final_seal.json'),
        full_matrix_collected=complete,
        axis_policy='All absolute joint-score axes start at zero; zero and negative S-G retained',
        weighting='Two configurations equally weighted within each parent; deterministic replay is not a new sample'))
    figures = {}
    if complete:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        plt.rcParams.update({'font.size': 10, 'svg.fonttype': 'none',
                             'svg.hashsalt': 'nso-v36-confirmation'})

        def save(fig, name):
            for extension in ('png', 'svg'):
                path = ROOT/'docs/research/figures'/f'v36_{name}_20260918.{extension}'
                buffer = io.BytesIO()
                fig.savefig(buffer, format=extension, dpi=160)
                write_bytes(OUTPUT, path, buffer.getvalue())
                figures[str(path.relative_to(ROOT))] = sha(path)
            plt.close(fig)

        colors = ('#3274a1', '#2d9755')
        labels = ('Active geometry (G)', 'Semantic (S)')
        width = .34
        figure, axes = plt.subplots(1, 2, figsize=(11.2, 5.2), sharey=True,
                                    constrained_layout=True)
        for axis, parent in zip(axes, PARENTS):
            x = np.arange(2)
            for index, (mode, label, color) in enumerate(zip(MODES, labels, colors)):
                values = [scores[parent, h, mode]['05cm']['joint'] for h in (0, 1)]
                bars = axis.bar(x+(index-.5)*width, values, width, color=color, label=label)
                for bar, value in zip(bars, values):
                    axis.text(bar.get_x()+bar.get_width()/2, value+.014,
                              f'{value:.4f}', ha='center', fontsize=9)
            for h in (0, 1):
                delta = scores[parent, h, 'S']['05cm']['joint']-scores[parent, h, 'G']['05cm']['joint']
                axis.text(h, .94, f'S - G = {delta:+.4f}', ha='center', fontsize=9)
            axis.set_xticks(x, ('Configuration h0', 'Configuration h1'))
            axis.set_ylim(0, 1.05)
            axis.set_ylabel('Measured J5 = C_map x surface F1 at 5 cm')
            axis.set_title(parent+' | both paired configurations retained')
            axis.grid(axis='y', alpha=.25)
            axis.set_axisbelow(True)
        handles, names = axes[0].get_legend_handles_labels()
        figure.legend(handles, names, loc='outside upper center', ncols=2, frameon=False)
        figure.suptitle('V36 frozen online CPU policy | depth-noise base seed 350918', fontsize=13)
        figure.supxlabel('One designed scene family, two known layouts; artificial RGB classes and exact synthetic pose. Replays are not new samples.', fontsize=8)
        save(figure, 'paired_joint')

        figure, axes = plt.subplots(1, 2, figsize=(11.2, 5.2), sharey=True,
                                    constrained_layout=True)
        for axis, parent in zip(axes, PARENTS):
            x = np.arange(len(THRESHOLDS))
            for index, (mode, label, color) in enumerate(zip(MODES, labels, colors)):
                values = [sum(scores[parent, h, mode][threshold]['joint'] for h in (0, 1))/2
                          for threshold in THRESHOLDS]
                expected = [result['prescribed_paired_comparison'][parent][threshold]['mean_joint'][mode]
                            for threshold in THRESHOLDS]
                np.testing.assert_allclose(values, expected, rtol=0., atol=1e-15)
                bars = axis.bar(x+(index-.5)*width, values, width, color=color, label=label)
                for bar, value in zip(bars, values):
                    axis.text(bar.get_x()+bar.get_width()/2, value+.014,
                              f'{value:.4f}', ha='center', fontsize=8)
            for index, threshold in enumerate(THRESHOLDS):
                delta = result['prescribed_paired_comparison'][parent][threshold]['S_vs_G']['paired_mean_joint_difference']
                axis.text(index, .94, f'{delta:+.4f}', ha='center', fontsize=9)
            axis.set_xticks(x, ('2 cm\nsecondary', '5 cm\nprimary', '10 cm\nsecondary'))
            axis.set_ylim(0, 1.05)
            axis.set_ylabel('Equal-weight mean C_map x surface F1')
            axis.set_title(parent+' | signed labels show mean S - G')
            axis.grid(axis='y', alpha=.25)
            axis.set_axisbelow(True)
        handles, names = axes[0].get_legend_handles_labels()
        figure.legend(handles, names, loc='outside upper center', ncols=2, frameon=False)
        figure.suptitle('V36 prescribed surface-distance thresholds | both configurations equally weighted', fontsize=13)
        figure.supxlabel('All prescribed thresholds are displayed; no scene-family or natural-image generalization claim is inferred from these means.', fontsize=8)
        save(figure, 'threshold_means')

    verify_seal(BATCH, 'final_seal.json')
    verify_sources(OUTPUT)
    for relative, expected in inputs.items():
        if sha(ROOT/relative) != expected:
            raise ValueError('frozen source evidence changed: '+relative)
    for relative, expected in figures.items():
        if sha(ROOT/relative) != expected:
            raise ValueError('figure bytes changed: '+relative)
    write(OUTPUT, OUTPUT/'result.json', dict(
        status='complete' if complete else 'incomplete_matrix_no_figures',
        measured_cells=len(rows), full_matrix_collected=complete, figures=figures,
        source_result_sha256=sha(BATCH/'result.json'),
        source_final_seal_sha256=sha(BATCH/'final_seal.json'),
        new_worlds=0, new_sensor_packets=0, new_controller_calls=0, new_DP_calls=0,
        new_TSDF_integrations=0, new_quality_evaluations=0, new_main_tasks=0))
    seal(OUTPUT)
    verify_inventory(OUTPUT)
    print(read(OUTPUT/'result.json'))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', action='store_true')
    arguments = parser.parse_args()
    if arguments.run:
        run()
    else:
        parser.print_help()
