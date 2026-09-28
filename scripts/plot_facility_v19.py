"""Show actual V19 layouts and initial RGB/depth, with no visibility wedges.

Optional world checks record the existing four tests in a separate small audit.
Both outputs are new directories; neither the world nor old evidence is edited.
"""
import os
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['OPENBLAS_NUM_THREADS'] = '1'
import argparse
import hashlib
import io
import json
from pathlib import Path
import sys
import time
import unittest
import zipfile
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch, Rectangle
from matplotlib.lines import Line2D

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from env.facility_documentation_v19 import FacilityWorldV19, PARENTS_V19


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path, data):
    path.write_text(json.dumps(data, indent=2, allow_nan=False)+'\n')


def loaded_project_sources(initial):
    paths = {Path(__file__).resolve()}
    for module in list(sys.modules.values()):
        name = getattr(module, '__file__', None)
        if not name:
            continue
        path = Path(name).resolve()
        if path.is_relative_to(ROOT) and path.suffix == '.py' and not path.is_relative_to(ROOT / '.venv-3d'):
            paths.add(path)
    result = {}
    for path in sorted(paths):
        name = str(path.relative_to(ROOT))
        current = sha(path)
        if name not in initial or initial[name] != current:
            raise ValueError('Project source changed during the audit: '+name)
        result[name] = current
    return result


def run_checks(output, initial):
    output.mkdir(parents=True, exist_ok=False)
    stream = io.StringIO()
    suite = unittest.defaultTestLoader.loadTestsFromName('tests.virtual3d.test_facility_v19')
    count = suite.countTestCases()
    started = time.perf_counter()
    result = unittest.TextTestRunner(stream=stream, verbosity=2).run(suite)
    sources = loaded_project_sources(initial)
    (output / 'test_output.txt').write_text(stream.getvalue())
    write(output / 'source_sha256.json', sources)
    report = dict(status='complete', passed=result.wasSuccessful() and count == 4,
        tests_run=result.testsRun, expected_tests=4, failures=len(result.failures), errors=len(result.errors),
        skipped=len(result.skipped), elapsed_s=time.perf_counter()-started,
        test_module='tests.virtual3d.test_facility_v19',
        scope='Initial sensor pairing, connected safe floor, actual mapper two marked assets, semantic intervention invariance and disparity-noise response; no full policy outcome',
        python_version=sys.version.split()[0], numpy_version=np.__version__,
        source_sha256_file='source_sha256.json')
    write(output / 'result.json', report)
    write(output / 'artifact_sha256.json', {p.name: sha(p) for p in output.iterdir() if p.name != 'artifact_sha256.json'})
    print(json.dumps(report), flush=True)
    if not report['passed']:
        raise RuntimeError('V19 world checks failed; see independent test_output.txt')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--checks-output', type=Path)
    args = parser.parse_args()
    initial = {str(p.relative_to(ROOT)): sha(p) for directory in ('env', 'nso', 'utils', 'scripts', 'tests')
               for p in (ROOT / directory).rglob('*.py')}
    if args.checks_output:
        run_checks(args.checks_output, initial)
    args.output.mkdir(parents=True, exist_ok=False)
    fig, axes = plt.subplots(3, 2, figsize=(12, 13), gridspec_kw={'height_ratios': [1.25, 1, 1]})
    colors = {2: '#3A8B9A', 3: '#D38A42'}
    records = []
    for col, parent in enumerate(PARENTS_V19):
        world = FacilityWorldV19(parent, 'A_complex_B_simple', sensor_model='iid_025px', noise_seed=1901)
        frame = world.sense()
        top = axes[0, col]
        top.imshow(world.reachable.astype(np.uint8), origin='upper', extent=(0, world.width, 0, world.height),
                   cmap=ListedColormap(['#D8DEE5', '#F5F8FB']), interpolation='nearest', vmin=0, vmax=1)
        for x, y, z, sx, sy, sz, owner in world._solid_primitives:
            if z+sz <= 0:
                continue
            color = '#485462' if owner == 1 else colors[world.objects[int(owner)-100]['category']]
            top.add_patch(Rectangle((x, y), sx, sy, facecolor=color, edgecolor='white' if owner != 1 else color, linewidth=.35))
        for item in world.objects:
            cx, front, _ = item['front_center']
            top.text(cx, front-.30, item['name'], ha='center', va='top', fontsize=10, weight='bold', color='#263440')
        start = frame.world_from_camera[:2, 3]
        top.scatter([start[0]], [start[1]], marker='*', color='#7655A3', s=110, zorder=10)
        top.annotate('', xy=(start[0], start[1]+.75), xytext=start,
                     arrowprops={'arrowstyle': '->', 'color': '#7655A3', 'lw': 1.5})
        top.text(start[0]+.3, start[1]-.20, 'Start', fontsize=8, color='#604584')
        top.set(xlim=(0, world.width), ylim=(0, world.height), xlabel='x / m', ylabel='y / m',
                title=f'{parent}: {world.width:g} x {world.height:g} m')
        top.set_aspect('equal')
        top.set_xticks(np.arange(0, world.width+.01, 2)); top.set_yticks(np.arange(0, world.height+.01, 2))
        top.tick_params(labelsize=8)
        axes[1, col].imshow(frame.color_rgb, interpolation='nearest')
        axes[1, col].set_title('Initial RGB: physical color markers', fontsize=11)
        axes[1, col].axis('off')
        cmap = matplotlib.colormaps['viridis'].copy(); cmap.set_bad('#111827')
        depth = np.ma.masked_equal(frame.depth_m, 0.)
        rendered = axes[2, col].imshow(depth, cmap=cmap, vmin=0, vmax=5, interpolation='nearest')
        axes[2, col].set_title('Initial measured depth: disparity sigma 0.25 px', fontsize=11)
        axes[2, col].axis('off')
        fig.colorbar(rendered, ax=axes[2, col], fraction=.045, pad=.025, label='Axial depth / m')
        records.append(dict(parent=parent, assignment=world.assignment,
            sensor_model=world.config.stereo_model, noise_seed=world.noise_seed, action_step=world.step_count,
            shape=list(world.shape), start_world_xy_m=start.tolist(),
            safe_cells=int(world.reachable.sum()), mission_assets=len(world.objects),
            marker_pixels={str(k): int(np.count_nonzero(frame.semantic == k)) for k in (2, 3)},
            valid_depth_pixels=int(np.count_nonzero(frame.depth_m)),
            depth_sha256=hashlib.sha256(frame.depth_m.tobytes()).hexdigest(),
            rgb_sha256=hashlib.sha256(frame.color_rgb.tobytes()).hexdigest(),
            labels='A-F are evaluator ground-truth map labels; no A/B overlay or instance identity is added to sensor images'))
    legend = [Patch(facecolor='#F5F8FB', edgecolor='#CAD3DD', label='Safe floor'),
              Patch(facecolor='#D8DEE5', label='Obstacle / safety buffer'),
              Patch(facecolor=colors[2], label='Closed enclosure'),
              Patch(facecolor=colors[3], label='Enclosure + exterior attachments'),
              Line2D([0], [0], marker='*', linestyle='none', color='#7655A3', markersize=10, label='Start + heading')]
    fig.suptitle('V19 exterior-documentation fixtures and actual initial sensors', fontsize=14, y=.994)
    fig.legend(handles=legend, loc='upper center', bbox_to_anchor=(.5, .970), ncol=3, frameon=False, fontsize=9)
    fig.text(.5, .022, 'Top: simulator truth; A-F identify all six assets. Safe floor is not an observed-area mask.\n'
             'Middle/bottom: actual 96 x 72 RGB and noisy depth at step 0; black depth pixels are invalid.\n'
             'A complex / B simple shown. Generic stereo assumptions, exact pose; not ZED calibration or policy results.',
             ha='center', va='bottom', fontsize=8, linespacing=1.5)
    fig.tight_layout(rect=(0, .085, 1, .925), h_pad=1.7, w_pad=2.)
    fig.savefig(args.output / 'layouts.png', dpi=150, bbox_inches='tight')
    fig.savefig(args.output / 'layouts.svg', bbox_inches='tight')
    plt.close(fig)
    sources = loaded_project_sources(initial)
    write(args.output / 'result.json', dict(status='complete', rows=['ground_truth_map', 'actual_rgb', 'actual_measured_depth'],
        layouts=records, source_sha256=sources, visibility_wedge_drawn=False,
        scope='Visual scene/sensor audit only; no measured policy outcome'))
    with zipfile.ZipFile(args.output / 'sources.zip', 'w', zipfile.ZIP_DEFLATED) as archive:
        archive.write(__file__, Path(__file__).name)
    write(args.output / 'artifact_sha256.json', {p.name: sha(p) for p in args.output.iterdir() if p.name != 'artifact_sha256.json'})
    print(str((args.output / 'layouts.png').resolve()), flush=True)


if __name__ == '__main__':
    main()
