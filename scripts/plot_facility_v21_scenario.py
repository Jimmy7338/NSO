#!/usr/bin/env python3
"""Static V21 scene explanation with fixed historical V20 N trajectories.

A-complex/B-simple is fixed for both parents before plotting, not selected by
outcome. Static world construction is permitted; sensing, scanning and stepping
are disabled. No planner/evaluator runs and no reference caches are generated.
All ground-truth geometry is used only to explain the procedural fixtures.
"""
import os
os.environ.setdefault('OMP_NUM_THREADS', '1')
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
os.environ.setdefault('PYTHONDONTWRITEBYTECODE', '1')
import argparse
from dataclasses import asdict
import gzip
import hashlib
import json
from pathlib import Path
import sys
import zipfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
SOURCE = ROOT / 'eval_results/facility_v20_coverage_probe_20260915'
OUTPUT = ROOT / 'audit_results/facility_v21_scenario_20260915'
ASSIGNMENT = 'A_complex_B_simple'


def require(ok, reason):
    if not ok:
        raise ValueError(reason)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for data in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(data)
    return h.hexdigest()


def read(path):
    if Path(path).suffix == '.gz':
        with gzip.open(path, 'rt', encoding='utf-8') as stream:
            return json.load(stream)
    return json.loads(Path(path).read_text(encoding='utf-8'))


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def dump(path, value):
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + '\n', encoding='utf-8')


def geometry_hash(*arrays):
    # Same declared array-byte format as frozen FacilityEvaluatorV19 metadata.
    import numpy as np
    h = hashlib.sha256()
    for array in arrays:
        a = np.ascontiguousarray(array)
        h.update(str(a.dtype).encode()); h.update(str(a.shape).encode()); h.update(a.tobytes())
    return h.hexdigest()


def load_scene(source, manifest, index):
    import numpy as np
    from env.facility_documentation_v19 import FacilityWorldV19

    class StaticWorld(FacilityWorldV19):
        def sense(self):
            raise RuntimeError('Sensing is disabled for this static figure')
        def scan(self):
            raise RuntimeError('Scanning is disabled for this static figure')
        def step(self, *args, **kwargs):
            raise RuntimeError('Paid actions are disabled for this static figure')

    case = manifest['cases'][index]
    require(case['index'] == index and case['assignment'] == ASSIGNMENT and case['mode'] == 'N', 'Fixed plotting case changed')
    folder = source / f'case_{index:02d}'; inventory = read(folder / 'artifact_hashes.json')
    wanted_files = ('result.json', 'runtime_audit.json.gz', 'packets/0000.npz', 'verification.json')
    for name in wanted_files:
        require(sha(folder / name) == inventory[name], 'Historical plotting input changed: ' + name)
    result = read(folder / 'result.json'); audit = read(folder / 'runtime_audit.json.gz')
    require(result['case'] == case and read(folder / 'verification.json')['status'] == 'passed', 'Historical case/replay differs')
    require(digest(audit) == result['runtime_audit_sha256'], 'Historical runtime audit digest differs')
    reference_name = f'references/{case["parent"]}_{ASSIGNMENT}_2026.npz'
    reference = Path(manifest['reference_source_root']) / reference_name
    require(sha(reference) == manifest['reference_sha256'][reference_name], 'Frozen reference changed')
    with np.load(reference, allow_pickle=False) as cache:
        metadata = json.loads(str(cache['metadata'].item()))
    world = StaticWorld(case['parent'], ASSIGNMENT, sensor_model=case['sensor_model'], noise_seed=case['noise_seed'])
    public = asdict(world.config)
    require(public == metadata['signature_payload']['config'], 'Static world configuration differs from frozen reference')
    actual_world_hash = geometry_hash(np.asarray(world.mesh.vertices), np.asarray(world.mesh.triangles),
        np.asarray(world.triangle_classes), world.reachable, world.intrinsic)
    require(actual_world_hash == metadata['signature_payload']['world_hash'], 'Static geometry differs from frozen reference')
    require(world.step_count == world.moves == world.collisions == 0, 'Static constructor executed actions')
    with np.load(folder / 'packets/0000.npz', allow_pickle=False) as packet:
        initial = json.loads(str(packet['metadata'].item()))
    require(initial['position'] == list(world.start) and initial['heading'] == 0, 'Recorded initial pose differs')
    poses = [initial['position'] + [initial['heading']]] + [a['pose'] for a in result['actions']]
    resolution = world.config.resolution_m
    xy = np.asarray([((p[1] + .5) * resolution, world.height - (p[0] + .5) * resolution) for p in poses])
    authorizations = [r for r in audit if r['event'] == 'paid_action_authorized']
    require([r['next_action_id'] for r in authorizations] == list(range(1, result['paid_actions'] + 1)), 'Trajectory authorization sequence differs')
    require(len(poses) == result['paid_actions'] + 1, 'Trajectory action count differs')
    for a, recorded in zip(authorizations, result['actions']):
        require(a['assessment']['action'] == recorded['action'], 'Trajectory action is not authorized')
    record = dict(parent=case['parent'], assignment=ASSIGNMENT, fixed_case_index=index,
        width_m=world.width, height_m=world.height, sensor_max_depth_m=world.config.max_depth_m,
        budget=case['budget'], paid_actions=result['paid_actions'], asset_count=len(world.objects),
        objects=world.objects, nominal_primitive_boxes=[list(p) for p in world._solid_primitives],
        start_recorded_xy_m=xy[0].tolist(), final_recorded_xy_m=xy[-1].tolist(),
        original_trajectory_sha256=digest(result['actions']), plotted_xy_sha256=geometry_hash(xy),
        trajectory_phases=dict(__import__('collections').Counter(r['phase'] for r in authorizations)),
        termination=result['termination'], static_world_hash=actual_world_hash,
        frozen_reference_world_hash=metadata['signature_payload']['world_hash'],
        reference_path=str(reference), reference_sha256=sha(reference), reference_arrays_loaded=False,
        input_case_inventory_sha256=sha(folder / 'artifact_hashes.json'),
        input_artifact_sha256={name: inventory[name] for name in wanted_files},
        sensor_calls=0, scan_calls=0, paid_actions_executed_by_plotter=0)
    return dict(world=world, case=case, result=result, xy=xy,
                phases=[r['phase'] for r in authorizations], record=record)


def topdown(output, scenes):
    import numpy as np
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap
    from matplotlib.collections import LineCollection
    from matplotlib.patches import Patch, Rectangle
    from matplotlib.lines import Line2D
    colors = {2: '#4f9994', 3: '#d7994c'}; travel = '#225da9'; returning = '#a52a75'
    fig, axes = plt.subplots(1, 2, figsize=(13.5, 6.9))
    for ax, scene in zip(axes, scenes):
        w = scene['world']; case = scene['case']; xy = scene['xy']
        ax.imshow(w.reachable.astype(int), origin='upper', extent=(0, w.width, 0, w.height),
                  cmap=ListedColormap(['#dde2e7', '#f8fafb']), interpolation='nearest', vmin=0, vmax=1)
        # Orthographic projections of actual solid primitives, not fabricated
        # room polygons, evaluated visibility masks or predicted sensor wedges.
        for x, y, z, sx, sy, sz, owner in w._solid_primitives:
            if z + sz <= 0:
                continue
            color = '#46505a' if owner == 1 else colors[w.objects[int(owner) - 100]['category']]
            ax.add_patch(Rectangle((x, y), sx, sy, facecolor=color, edgecolor='#ffffff' if owner != 1 else color,
                                   linewidth=.55, zorder=2))
        segments = np.stack([xy[:-1], xy[1:]], axis=1)
        for phase, color, style in [('outbound', travel, '-'), ('return', returning, '--')]:
            subset = np.asarray([s for s, p in zip(segments, scene['phases']) if p == phase])
            if len(subset):
                ax.add_collection(LineCollection(subset, colors=color, linewidths=2, linestyles=style, alpha=.85, zorder=4))
        moved = [i for i in range(len(segments)) if np.linalg.norm(segments[i, 1] - segments[i, 0]) > 0]
        for i in moved[::14]:
            before, after = segments[i]; midpoint = (before + after) / 2; delta = (after - before) * 1.7
            color = returning if scene['phases'][i] == 'return' else travel
            ax.annotate('', xy=midpoint + delta / 2, xytext=midpoint - delta / 2,
                        arrowprops=dict(arrowstyle='-|>', lw=1.2, color=color, mutation_scale=9), zorder=5)
        for item in w.objects:
            x, y, _ = item['front_center']
            ax.text(x, y + 1.48, item['name'], ha='center', va='bottom', weight='bold', fontsize=12, color='#27313b', zorder=6,
                    bbox=dict(boxstyle='round,pad=.1', fc='white', ec='none', alpha=.8))
        ax.scatter(*xy[0], marker='*', s=190, color='#603c86', edgecolor='white', linewidth=.8, zorder=8)
        ax.annotate('Start / returned endpoint', xy=xy[0], xytext=(xy[0, 0] + .5, xy[0, 1] - .65),
                    fontsize=8, color='#603c86', ha='left', va='top')
        ax.annotate('', xy=(.9, w.height - .8), xytext=(.9, w.height - 1.7),
                    arrowprops=dict(arrowstyle='->', lw=1.3, color='#46505a'))
        ax.text(.9, w.height - .6, 'N', ha='center', fontsize=9)
        ax.set_title(f'{case["parent"]}: {w.width:g} x {w.height:g} m\nRecorded N path: {scene["result"]["paid_actions"]} / {case["budget"]} paid actions', fontsize=12, pad=12)
        ax.set(xlim=(0, w.width), ylim=(0, w.height), xlabel='x / m', ylabel='y / m', aspect='equal')
        ax.set_xticks(np.arange(0, w.width + .1, 2)); ax.set_yticks(np.arange(0, w.height + .1, 2))
        ax.tick_params(labelsize=9); ax.grid(alpha=.12, zorder=0)
    handles = [Patch(facecolor=colors[2], label='Closed body'), Patch(facecolor=colors[3], label='Body + exposed attachments'),
        Patch(facecolor='#46505a', label='Walls / occluding panels'),
        Line2D([0], [0], color=travel, lw=2, label='Recorded N outbound'),
        Line2D([0], [0], color=returning, lw=2, linestyle='--', label='Recorded N return'),
        Patch(facecolor='#dde2e7', label='Obstacle / safety margin')]
    fig.suptitle('V21 procedural exterior-documentation task and fixed historical N trajectories', fontsize=14, y=.995)
    fig.legend(handles=handles, ncol=3, loc='upper center', bbox_to_anchor=(.5, .95), frameon=False, fontsize=9)
    fig.text(.5, .037, 'Six exterior assets per layout; A complex / B simple is fixed in both panels. RGB-D and lidar maximum range: 5 m.\n'
             'Maps and A-F labels are simulator truth for explanation only. Pale floor is safe ground truth, not observed coverage.\n'
             'Paths are saved V20 N case 00 / 02; turns remain paid even where the path does not move. These are procedural fixtures, not field trials.',
             ha='center', va='bottom', fontsize=8.5, linespacing=1.4)
    fig.tight_layout(rect=(0, .14, 1, .86), w_pad=2)
    fig.savefig(output / 'scenario_topdown.png', dpi=170, bbox_inches='tight')
    fig.savefig(output / 'scenario_topdown.svg', bbox_inches='tight'); plt.close(fig)


def exterior_models(output, world):
    import numpy as np
    import matplotlib.pyplot as plt
    from matplotlib.colors import to_rgb
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection
    fig = plt.figure(figsize=(10.7, 5.4))
    colors = ['#d7994c', '#4f9994']
    for panel, object_id in enumerate([0, 1], 1):
        ax = fig.add_subplot(1, 2, panel, projection='3d')
        item = world.objects[object_id]; mesh = world.instance_mesh(object_id)
        vertices = np.asarray(mesh.vertices).copy(); vertices[:, :2] -= np.asarray(item['front_center'][:2])
        faces = vertices[np.asarray(mesh.triangles)]
        normals = np.cross(faces[:, 1] - faces[:, 0], faces[:, 2] - faces[:, 0])
        normals /= np.maximum(np.linalg.norm(normals, axis=1, keepdims=True), 1e-12)
        light = np.asarray([-.5, .6, .7]); light /= np.linalg.norm(light)
        shade = .60 + .35 * np.maximum(0, normals @ light)
        face_colors = np.asarray(to_rgb(colors[panel - 1]))[None, :] * shade[:, None]
        ax.add_collection3d(Poly3DCollection(faces, facecolors=face_colors, edgecolors='none'))
        ax.set(xlim=(-1.23, 1.23), ylim=(-.12, 1.38), zlim=(0, 1.75),
               xlabel='x from centre / m', ylabel='Behind front / m', zlabel='z / m')
        ax.set_box_aspect((2.46, 1.50, 1.75)); ax.view_init(elev=24, azim=60)
        ax.set_title('A: closed body + exposed attachments' if object_id == 0 else 'B: closed body', fontsize=11, pad=12)
        ax.tick_params(labelsize=8)
        ax.xaxis.pane.fill = ax.yaxis.pane.fill = ax.zaxis.pane.fill = False
    fig.suptitle('Task exterior surfaces: actual fixture mesh, rear oblique view', fontsize=14, y=.995)
    fig.text(.5, .025, 'Fixed P00 A-complex/B-simple geometry. Both include a common broad front face.\n'
             'Solid external attachments are physical geometry; no internal shelves are added.\n'
             'Illustration viewpoint and colors explain geometry only; they are not sensor observations or semantic predictions.',
             ha='center', va='bottom', fontsize=9, linespacing=1.45)
    fig.tight_layout(rect=(.01, .16, .99, .94))
    fig.savefig(output / 'facility_exteriors.png', dpi=170, bbox_inches='tight')
    fig.savefig(output / 'facility_exteriors.svg', bbox_inches='tight'); plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, default=SOURCE)
    parser.add_argument('--output', type=Path, default=OUTPUT)
    args = parser.parse_args(); source = args.source.resolve(); output = args.output.resolve()
    require(not output.exists(), 'Output must be new; preserve existing figures/evidence')
    require(not output.is_relative_to(source) and not source.is_relative_to(output), 'Output/source must be separate')
    manifest = read(source / 'manifest.json'); inventory = read(source / 'artifact_hashes.json')
    require(manifest['status'] == 'complete' and manifest['mode'] == 'N', 'Complete N evidence required')
    require(sha(source / 'manifest.json') == inventory['manifest.json'], 'Source manifest hash differs')
    source_hashes = {**manifest['source_sha256'], str(Path(__file__).resolve().relative_to(ROOT)): sha(Path(__file__).resolve())}
    for name, wanted in source_hashes.items():
        require(sha(ROOT / name) == wanted, 'Static figure source differs: ' + name)
    scenes = [load_scene(source, manifest, index) for index in [0, 2]]
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.size': 10, 'axes.spines.top': False, 'axes.spines.right': False})
    output.mkdir(parents=True, exist_ok=False)
    try:
        topdown(output, scenes); exterior_models(output, scenes[0]['world'])
        for name, wanted in source_hashes.items():
            require(sha(ROOT / name) == wanted, 'Source changed while plotting: ' + name)
        for scene in scenes:
            require(scene['world'].step_count == scene['world'].moves == scene['world'].collisions == 0, 'Static plotter executed actions')
        report = dict(status='complete', purpose='Static task explanation with fixed historical N paths',
            figures=['scenario_topdown.png', 'scenario_topdown.svg', 'facility_exteriors.png', 'facility_exteriors.svg'],
            cases=[scene['record'] for scene in scenes], fixed_case_indices=[0, 2], fixed_assignment=ASSIGNMENT,
            no_outcome_based_case_selection=True, procedural_fixture_not_large_scale_field_trial=True,
            all_ground_truth_for_explanation_only=True, planner_invoked=False, evaluator_invoked=False,
            sensor_calls=0, scan_calls=0, new_physical_actions=0, reference_regenerated=False,
            visibility_wedge_drawn=False, sources_sha256=source_hashes,
            source_root=str(source), source_manifest_sha256=sha(source / 'manifest.json'),
            source_inventory_sha256=sha(source / 'artifact_hashes.json'))
        dump(output / 'result.json', report); dump(output / 'source_sha256.json', source_hashes)
        with zipfile.ZipFile(output / 'sources.zip', 'x', zipfile.ZIP_DEFLATED) as archive:
            for name in source_hashes:
                archive.write(ROOT / name, name)
        dump(output / 'artifact_hashes.json', {p.name: sha(p) for p in sorted(output.iterdir()) if p.is_file() and p.name != 'artifact_hashes.json'})
        print(json.dumps(dict(status='complete', output=str(output), world_hashes_match_frozen_references=True,
            new_physical_actions=0, figures=report['figures']), indent=2))
    except Exception as error:
        dump(output / 'failure.json', dict(status='failed', error=repr(error))); raise


if __name__ == '__main__':
    main()
