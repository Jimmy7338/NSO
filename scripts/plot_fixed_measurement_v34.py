#!/usr/bin/env python3
"""Plot the sealed V34 fixed-route measurements; never simulate or rescore.

Only --run writes artifacts. Figure values come from saved measurement/result
JSON. The h0 alias is one physical trajectory, not an independent replicate.
"""
import argparse
import hashlib
import io
import json
import os
from pathlib import Path
import shutil
import sys
import time
import traceback
import zipfile

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / 'audit_results/v34_fixed_measurement_20260918'
FIGURES = ROOT / 'docs/research/figures/v34_fixed_measurement_20260918'
PACK = ROOT / 'audit_results/v34_fixed_measurement_figures_20260918'
CAP = 2 * 1024 ** 2
RESERVE = 64 * 1024 ** 2
THRESHOLDS = ('02cm', '05cm', '10cm')
METRICS = ('C_map', 'f1', 'joint')
COLORS = ('#69727c', '#2478a5', '#d07d28', '#438c65')
ZERO_CALLS = dict(worlds=0, sensor_packets=0, mapper_updates=0,
    TSDF_integrations=0, new_surface_scores=0, new_main_tasks=0)


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def files(folder):
    return sorted(p for p in folder.rglob('*') if p.is_file()) if folder.exists() else []


def guard(size=0):
    used = sum(p.stat().st_size for directory in (PACK, FIGURES) for p in files(directory))
    if used + size > CAP or shutil.disk_usage(ROOT).free - size < RESERVE:
        raise RuntimeError('figure evidence 2 MiB cap or 64 MiB reserve would be exceeded')


def write_bytes(path, payload):
    guard(len(payload))
    with Path(path).open('xb') as stream:
        stream.write(payload)


def write(path, value):
    write_bytes(path, (json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n').encode())


def verify_input_seal(expected_inventory_sha=None):
    inventory_path = SOURCE / 'final_seal.json'
    if expected_inventory_sha is not None and sha(inventory_path) != expected_inventory_sha:
        raise ValueError('final measurement inventory changed during plotting')
    inventory = read(inventory_path)
    actual = {str(p.relative_to(SOURCE)) for p in files(SOURCE) if p.name != 'final_seal.json'}
    if actual != set(inventory):
        raise ValueError('measurement artifact set differs from the final seal')
    for relative, expected in inventory.items():
        path = (SOURCE / relative).resolve()
        if not path.is_relative_to(SOURCE.resolve()) or sha(path) != expected:
            raise ValueError('sealed measurement bytes differ: ' + relative)
    return inventory


def saved_plot_data(result, config):
    """Select saved scalar fields only; no metric or confidence computation."""
    table = result['table']
    unique = []
    for physical in config['physical_cases']:
        rows = [row for row in table if row['physical_case'] == physical['index']]
        measured = [row for row in rows if row['measured']]
        if not measured:
            continue
        if any(row['stages'] != measured[0]['stages'] for row in measured):
            raise ValueError('alias cells disagree on their saved measurement')
        policies = [row['policy'] for row in rows]
        alias = len(rows) > 1
        label = f"h{physical['hypothesis']}: " + ('G = class oracle' if alias else policies[0].replace('_', ' '))
        stages = {}
        for stage in ('prefix', 'final'):
            measurement = measured[0]['stages'][stage]['measurement']
            stages[stage] = dict(paid_actions=measurement['paid_actions'],
                C_map=measurement['C_map'], f1=measurement['05cm']['f1'],
                joint=measurement['05cm']['joint'], eligible=measurement['eligible'])
        unique.append(dict(physical_case=physical['index'], hypothesis=physical['hypothesis'],
            label=label, comparison_cells=[row['cell_index'] for row in rows],
            alias=alias, alias_is_independent_sample=False, policies=policies, stages=stages,
            all_saved_replays_passed=all(row['independent_replay_passed'] for row in rows)))
    comparison = result['prescribed_paired_comparison']
    if comparison and set(comparison) != set(THRESHOLDS):
        raise ValueError('saved three-threshold comparison is incomplete')
    return dict(source_status=result['status'], unique_physical_trajectories=unique,
        declared_physical_cases=len(config['physical_cases']), declared_comparison_cells=len(table),
        measured_comparison_cells=sum(row['measured'] for row in table),
        uniform_hypothesis_prior=[.5, .5], threshold_comparison=comparison,
        all_final_eligible=result['all_final_eligible'], all_replays_passed=result['all_replays_passed'],
        full_matrix_collected=result['full_matrix_collected'],
        source_measured_fixed_route_gate_passed=result['measured_fixed_route_gate_passed'],
        metric='C_map times public-ROI external-vertical surface F1',
        precision_limitation='Prediction outside the common padded facility ROI is excluded from precision.',
        scope='P00 fixed offline G and class-oracle routes; exact synthetic poses; not full architecture or autonomous ANS.',
        scalar_provenance='source result.json table[].stages[].measurement and prescribed_paired_comparison only')


def axes_style(axis):
    axis.set_ylim(0, 1.08)
    axis.set_yticks([0, .2, .4, .6, .8, 1.])
    axis.grid(axis='y', color='#d8dde2', linewidth=.55, zorder=0)
    axis.spines[['top', 'right']].set_visible(False)
    axis.tick_params(axis='x', length=0)


def export(figure, stem, pyplot):
    paths = []
    for extension in ('png', 'svg'):
        buffer = io.BytesIO()
        figure.savefig(buffer, format=extension, dpi=180, facecolor='white',
            metadata={'Date': None} if extension == 'svg' else None)
        path = FIGURES / f'{stem}.{extension}'
        write_bytes(path, buffer.getvalue())
        paths.append(path)
    pyplot.close(figure)
    return paths


def draw_endpoint(data, pyplot):
    rows = data['unique_physical_trajectories']
    figure, axes = pyplot.subplots(1, 3, figsize=(12.8, 4.25), layout='constrained')
    titles = ('Full reachable-floor coverage', 'Vertical surface F1 at 5 cm', 'Joint score at 5 cm')
    labels = ('$C_{map}$', '$F1_{5cm}$', '$J_5 = C_{map} \\times F1_{5cm}$')
    for axis, metric, title, ylabel in zip(axes, METRICS, titles, labels):
        axes_style(axis)
        values = [row['stages']['final'][metric] for row in rows]
        bars = axis.bar(range(len(rows)), values, width=.6,
            color=[COLORS[i % len(COLORS)] for i in range(len(rows))], zorder=3)
        for bar, value, row in zip(bars, values, rows):
            axis.text(bar.get_x()+bar.get_width()/2, value+.023, f'{value:.3f}', ha='center', fontsize=9)
            if not row['stages']['final']['eligible']:
                bar.set_hatch('//')
        axis.set(title=title, ylabel=ylabel, xticks=range(len(rows)),
            xticklabels=[row['label'].replace(': ', ':\n') + ('\n(shared alias)' if row['alias'] else '') for row in rows])
        axis.tick_params(axis='x', labelsize=8)
        if metric == 'C_map':
            axis.axhline(.8, color='#a54b4b', linestyle='--', linewidth=1.0, zorder=4)
            axis.text(.99, .8, 'C80 qualification', transform=axis.get_yaxis_transform(),
                va='bottom', ha='right', color='#9b4444', fontsize=8)
    figure.suptitle(f"V34 P00: {len(rows)} unique fixed trajectories, {data['measured_comparison_cells']} measured comparison cells", fontsize=12)
    figure.supxlabel('h0 shared trajectory counts once. Hatched bars: final qualification failed. Surface precision excludes predictions outside the common ROI.', fontsize=8)
    return export(figure, '01_unique_endpoints', pyplot)


def draw_thresholds(data, pyplot):
    comparison = data['threshold_comparison']
    figure, axis = pyplot.subplots(figsize=(8.5, 4.35), layout='constrained')
    if comparison:
        axes_style(axis)
        x = [0, 1, 2]
        left = [comparison[key]['G_mean_joint'] for key in THRESHOLDS]
        right = [comparison[key]['class_oracle_mean_joint'] for key in THRESHOLDS]
        axis.bar([value-.18 for value in x], left, width=.32, color=COLORS[1], label='Fixed G witness', zorder=3)
        axis.bar([value+.18 for value in x], right, width=.32, color=COLORS[2], label='Fixed class-oracle witness', zorder=3)
        for i, key in enumerate(THRESHOLDS):
            delta = comparison[key]['paired_mean_joint_difference']
            axis.text(i, max(left[i], right[i])+.025, f'Saved paired difference: {delta:+.4f}', ha='center', fontsize=8)
        axis.set(xticks=x, xticklabels=['2 cm (secondary)', '5 cm (primary)', '10 cm (secondary)'],
            ylabel='Mean $C_{map} \\times F1$', title='P00 prescribed paired comparison: equal weight for h0 and h1')
        axis.legend(loc='upper left', frameon=False, fontsize=9)
    else:
        axis.axis('off')
        axis.text(.5, .55, 'Paired comparison unavailable', ha='center', va='center', fontsize=14)
        axis.text(.5, .4, 'Collection stopped before the complete declared matrix.\nNo missing branch is imputed.',
            ha='center', va='center', fontsize=10)
    figure.supxlabel('Four declared cells include the h0 alias. Fixed routes, exact synthetic poses; no autonomous or full-architecture advantage claimed.', fontsize=8)
    return export(figure, '02_prescribed_threshold_comparison', pyplot)


def draw_response(data, pyplot):
    rows = data['unique_physical_trajectories']
    figure, axes = pyplot.subplots(1, 3, figsize=(12.0, 4.1), layout='constrained')
    labels = ('$C_{map}$', '$F1_{5cm}$', '$J_5$')
    for axis, metric, ylabel in zip(axes, METRICS, labels):
        axes_style(axis)
        for i, row in enumerate(rows):
            values = [row['stages'][stage][metric] for stage in ('prefix', 'final')]
            axis.plot([0, 1], values, marker='o', markersize=5,
                linewidth=1.6, color=COLORS[i % len(COLORS)], label=row['label'], zorder=3)
        axis.set(xticks=[0, 1], xticklabels=['Paid prefix: 18 actions', 'Frozen route endpoint'],
            xlim=(-.08, 1.08), ylabel=ylabel)
        axis.tick_params(axis='x', labelsize=8)
        if metric == 'C_map':
            axis.axhline(.8, color='#a54b4b', linestyle='--', linewidth=.9)
    axes[1].legend(loc='lower right', frameon=False, fontsize=8)
    paid = ', '.join(f"case{row['physical_case']:02d}: {row['stages']['final']['paid_actions']}" for row in rows)
    figure.suptitle('P00 prefix-to-final response of the same observed TSDF history', fontsize=12)
    figure.supxlabel('Endpoint paid actions — ' + paid + '. Lines connect two saved snapshots; they are not learning curves.', fontsize=8)
    return export(figure, '03_prefix_to_final', pyplot)


def run():
    if not sys.dont_write_bytecode:
        raise ValueError('invoke with Python -B')
    for name in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
        if os.environ.get(name) != '1':
            raise ValueError(name + '=1 is required')
    if PACK.exists() or FIGURES.exists():
        raise FileExistsError('figure output already exists; no overwrite or automatic retry')
    verify_input_seal()
    result = read(SOURCE / 'result.json')
    config = read(SOURCE / 'config.json')
    data = saved_plot_data(result, config)
    if not data['unique_physical_trajectories']:
        raise ValueError('no saved complete trajectory measurements to plot')
    inputs = {str(path.relative_to(ROOT)): sha(path)
        for path in (SOURCE / 'final_seal.json', SOURCE / 'result.json', SOURCE / 'config.json')}
    guard()
    PACK.mkdir()
    FIGURES.mkdir()
    started = time.monotonic()
    try:
        script = Path(__file__).resolve()
        archive_bytes = io.BytesIO()
        with zipfile.ZipFile(archive_bytes, 'w', zipfile.ZIP_DEFLATED, compresslevel=9) as archive:
            archive.write(script, str(script.relative_to(ROOT)))
        write_bytes(PACK / 'sources.zip', archive_bytes.getvalue())
        write(PACK / 'manifest.json', dict(status='frozen_before_plotting',
            source_sha256={str(script.relative_to(ROOT)): sha(script)},
            source_archive_sha256=sha(PACK / 'sources.zip'), input_sha256=inputs,
            total_figure_and_pack_cap_bytes=CAP, reserve_bytes=RESERVE, calls=ZERO_CALLS,
            figure_values_from_saved_results_only=True, new_measurements_or_scores_computed=False))
        write(PACK / 'plot_data.json', data)
        os.environ['MPLBACKEND'] = 'Agg'
        os.environ['MPLCONFIGDIR'] = str(PACK / 'matplotlib_cache')
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as pyplot
        pyplot.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 9,
            'axes.titlesize': 10, 'axes.labelsize': 10, 'svg.fonttype': 'none',
            'svg.hashsalt': 'v34-fixed-measurement-20260918', 'savefig.transparent': False})
        paths = draw_endpoint(data, pyplot) + draw_thresholds(data, pyplot) + draw_response(data, pyplot)
        captions = (
            '# V34 P00 fixed-route measurement figures\n\n'
            'All values are read from the final sealed acquisition analysis. No sensors, fusion, '
            'surface sampling or quality evaluation run during plotting.\n\n'
            '1. **Unique endpoints:** each physical trajectory appears once. The h0 G/class-oracle '
            'cells share a trajectory and are not independent observations. C_map uses the full '
            'reachable raster; F1 and J use the declared 5 cm threshold.\n'
            '2. **Prescribed threshold comparison:** saved G and class-oracle mean joint scores '
            'use the public uniform h0/h1 prior. 5 cm is primary; 2/10 cm are secondary. '
            'Differences are descriptive, without confidence intervals or a significance claim.\n'
            '3. **Prefix-to-final response:** two frozen snapshots from the same paid history; '
            'the endpoint action count is shown. The lines are not a temporal quality reconstruction.\n\n'
            'These are offline fixed witness routes with exact synthetic poses, not an autonomous '
            'ANS or full-architecture comparison. Prediction outside the common padded facility '
            'ROI is excluded from surface precision.\n')
        write_bytes(FIGURES / 'README.md', captions.encode())
        paths.append(FIGURES / 'README.md')
        for relative, expected in inputs.items():
            if sha(ROOT / relative) != expected:
                raise ValueError('plot input changed: ' + relative)
        verify_input_seal(inputs[str((SOURCE / 'final_seal.json').relative_to(ROOT))])
        if sha(script) != read(PACK / 'manifest.json')['source_sha256'][str(script.relative_to(ROOT))]:
            raise ValueError('plot source changed during execution')
        write(PACK / 'result.json', dict(status='complete', source_status=result['status'],
            figures={str(path.relative_to(ROOT)): sha(path) for path in paths},
            unique_physical_trajectories=len(data['unique_physical_trajectories']),
            measured_comparison_cells=data['measured_comparison_cells'], calls=ZERO_CALLS,
            matplotlib_version=matplotlib.__version__, elapsed_seconds=time.monotonic()-started,
            source_final_seal_sha256=sha(SOURCE / 'final_seal.json')))
        write(PACK / 'artifact_hashes.json', {str(path.relative_to(ROOT)): sha(path)
            for directory in (PACK, FIGURES) for path in files(directory)})
        guard()
        print(json.dumps(dict(status='complete', png_files=3, svg_files=3,
            figure_directory=str(FIGURES), calls=ZERO_CALLS)), flush=True)
    except BaseException:
        if not (PACK / 'failure.json').exists():
            write(PACK / 'failure.json', dict(status='failed', traceback=traceback.format_exc(), calls=ZERO_CALLS,
                partial_artifacts_preserved=True))
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', action='store_true', help='plot only after final measurement result and seal exist')
    args = parser.parse_args()
    if args.run:
        run()
    else:
        parser.print_help()


if __name__ == '__main__':
    main()
