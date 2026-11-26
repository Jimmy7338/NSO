"""Remeasure fixed branches; fit exclusively on calibration layout outcomes."""
from experiment import B11, BASE, HERE, LAYOUTS, PROTOCOL, sha, verify
import argparse
import copy
import json
import sys
from pathlib import Path

import numpy as np
import open3d as o3d
import trimesh

sys.path[:0] = [str(BASE / 'dual_template_exploration_20261119/audit'), str(B11 / 'evaluation')]
from surface_metrics import fixed_camera_poses, ray_scene, subset_mesh, surface_samples, unique_sampling_faces, visible_reference
from paired_metrics import surface_increment
from world import make_world


def read(path):
    return json.loads(Path(path).read_text())


def save(path, data):
    with Path(path).open('x') as output:
        json.dump(data, output, indent=2, allow_nan=False)
        output.write('\n')


def checked(resource):
    p = Path(resource['path'])
    if sha(p) != resource['sha256']:
        raise ValueError(f'Resource changed: {p}')
    return p


def reference(world, seed):
    scene, _, labels = world.source_scene_for_evaluator()
    triangles, vertices, owners = [], [], []
    offset = 0
    for node in scene.graph.nodes_geometry:
        transform, label = scene.graph[node]
        mesh = scene.geometry[label]
        vertices.append(trimesh.transform_points(mesh.vertices, transform))
        triangles.append(mesh.faces + offset)
        owners.extend([labels[node]['owner']] * len(mesh.faces))
        offset += len(mesh.vertices)
    source = o3d.geometry.TriangleMesh(o3d.utility.Vector3dVector(np.concatenate(vertices)),
                                       o3d.utility.Vector3iVector(np.concatenate(triangles)))
    deduplicated, filtered, duplicates = unique_sampling_faces(source, np.asarray(owners))
    object_mesh = subset_mesh(deduplicated, filtered == 1)
    poses, _ = fixed_camera_poses(world.reachable, world.config, world.start)
    count = PROTOCOL['evaluation']['samples_per_layout']
    sampled, _ = surface_samples(object_mesh, count, seed)
    visible, _, _ = visible_reference(sampled, ray_scene(source), poses, world.config, tolerance_m=.001)
    if not visible.any():
        raise ValueError('No source-visible object')
    return sampled[visible].astype(np.float32), object_mesh.get_surface_area() / count, {
        'seed': seed, 'drawn': count, 'retained': int(visible.sum()), 'camera_poses': len(poses),
        'fixed_sampling_pose_contract': 'source-reachable camera lattice, separate from routes and actor',
        'duplicates': duplicates,
    }


def distances(mesh_path, points):
    mesh = o3d.io.read_triangle_mesh(str(mesh_path))
    if len(mesh.triangles) == 0:
        return np.full(len(points), np.inf)
    scene = ray_scene(mesh)
    return scene.compute_distance(o3d.core.Tensor(points), nthreads=1).numpy()


def evaluate_layout(job_root, layout):
    folder = job_root / layout
    if not (folder / 'run_receipt.json').exists():
        failure = read(folder / 'failure.json') if (folder / 'failure.json').exists() else None
        return dict(layout=layout, split=LAYOUTS[layout]['split'], status='missing_or_failed_run', failure=failure, rows=[])
    ledger = read(folder / 'run_receipt.json')
    assert ledger['freeze_sha256'] == sha(HERE / 'freeze.json')
    forecasts = read(checked(ledger['predictions']))
    before = checked(ledger['before_mesh'])
    assert forecasts['recorded_before_actual_branches']
    fit_hash = sha(HERE / 'calibrated_coefficients.json') if ledger['split'] == 'heldout' else None
    assert forecasts['correction_coefficients_sha256'] == fit_hash
    world = make_world(layout)
    seed = PROTOCOL['seed'] + list(LAYOUTS).index(layout) + 1
    points, unit, sampling = reference(world, seed)
    before_dist = distances(before, points)
    weights = np.full(len(points), unit)
    predictions = {row['id']: row for row in forecasts['predictions']}
    outcomes = []
    for branch in ledger['branches']:
        entry = read(checked(branch['execution']))
        actual = surface_increment(before_dist, distances(checked(branch['after_mesh']), points), weights)
        outcomes.append(dict(layout=layout, family=LAYOUTS[layout]['family'],
                             endpoint_reached=entry['endpoint_reached'],
                             actions=entry['completed_actions'], collisions=entry['simulated_collisions'],
                             path_m=entry['path_length_m'], execution_wall_s=entry['elapsed_s'],
                             **predictions[branch['candidate_id']], **actual))
    return dict(layout=layout, split=LAYOUTS[layout]['split'], status='evaluated', rows=outcomes,
                sampling=sampling, modeled_support_tracks=forecasts['modeled_support_tracks'],
                before_sha256=sha(before), source_receipt_sha256=sha(folder / 'run_receipt.json'))


ACTUAL = 'gross_new_area_m2'
RAW = 'raw_acquisition_m2'
NEAR = 'support_near_m2'
FAR = 'support_far_m2'


def complete(rows):
    return [row for row in rows if row['endpoint_reached']]


def fit(calibration, path, binding_path):
    groups = [complete(layout['rows']) for layout in calibration]
    if any(not rows for rows in groups):
        raise ValueError('Each declared calibration layout must produce at least one measurable planned endpoint')
    near = np.array([row[NEAR] for rows in groups for row in rows])
    far = np.array([row[FAR] for rows in groups for row in rows])
    target = np.array([row[ACTUAL] for rows in groups for row in rows])
    weight = np.array([1.0/len(rows)/len(groups) for rows in groups for row in rows])

    def rank(n, f):
        pn, pf = n/100., f/100.
        loss = float(np.dot(weight, np.abs(pn*near + pf*far - target)))
        return (round(loss, 14), (100-n)**2+(100-f)**2, n, f)

    allowed = ((n, f) for n in range(101) for f in range(n+1))
    beta_near, beta_far = min(allowed, key=lambda pair: rank(*pair))
    beta_global = min(range(101), key=lambda s: rank(s, s))
    save(path, dict(status='calibration_coefficients_locked_before_heldout',
                    freeze_sha256=sha(HERE / 'freeze.json'),
                    calibration_sources_sha256={str(binding_path): sha(binding_path)},
                    layout_ids=[l['layout'] for l in calibration], endpoints=sum(map(len, groups)),
                    beta_near=beta_near/100., beta_far=beta_far/100., beta_global=beta_global/100.,
                    layout_balanced_calibration_MAE_m2=rank(beta_near, beta_far)[0],
                    global_layout_balanced_calibration_MAE_m2=rank(beta_global, beta_global)[0]))


def metrics(rows, key):
    measured = complete(rows)
    if not measured:
        return dict(measured=0, mae_m2=None, bias_m2=None,
                    concordant_pairs=0, comparable_pairs=0, forecast_ties=0, selected_route=None)
    errors = [row[key] - row[ACTUAL] for row in measured]
    comparable = concordant = tied = 0
    for i, row in enumerate(measured):
        for other in measured[i+1:]:
            actual_diff = row[ACTUAL]-other[ACTUAL]
            if abs(actual_diff) <= 1.e-12:
                continue
            comparable += 1
            predicted_diff = row[key]-other[key]
            if abs(predicted_diff) <= 1.e-12:
                tied += 1
            elif predicted_diff*actual_diff > 0.:
                concordant += 1
    chosen = min(rows, key=lambda row: (-row[key], row['id']))
    best_gross = max(row[ACTUAL] for row in measured)
    chosen_actual = chosen[ACTUAL] if chosen['endpoint_reached'] else None
    return dict(measured=len(measured), mae_m2=float(np.abs(errors).mean()), bias_m2=float(np.mean(errors)),
                comparable_pairs=comparable, concordant_pairs=concordant, forecast_ties=tied,
                selected_route=chosen['route'], selected_endpoint_reached=chosen['endpoint_reached'],
                actual_selected_gross_m2=chosen_actual,
                actual_selected_lost_m2=chosen['lost_area_m2'] if chosen_actual is not None else None,
                actual_selected_net_m2=chosen['net_new_area_m2'] if chosen_actual is not None else None,
                best_measured_gross_m2=best_gross,
                selection_regret_m2=best_gross-chosen_actual if chosen_actual is not None else None)


def comparison(layouts, fit_params):
    layouts = copy.deepcopy(layouts)
    names = ['raw_visibility_m2', RAW, 'scaled_acquisition_m2', 'corrected_acquisition_m2']
    for layout in layouts:
        for row in layout['rows']:
            row['corrected_acquisition_m2'] = fit_params['beta_near']*row[NEAR] + fit_params['beta_far']*row[FAR]
            row['scaled_acquisition_m2'] = fit_params['beta_global']*row[RAW]
        layout['metrics'] = {name: metrics(layout['rows'], name) for name in names}
    summary = {}
    for name in names:
        evaluated = [l['metrics'][name] for l in layouts if l['metrics'][name]['measured']]
        summary[name] = {}
        for field in ('mae_m2','bias_m2','actual_selected_gross_m2','actual_selected_net_m2','selection_regret_m2'):
            values = [group[field] for group in evaluated if group.get(field) is not None]
            summary[name][f'mean_layout_{field}'] = float(np.mean(values)) if values else None
            summary[name][f'mean_layout_{field}_n'] = len(values)
        for field in ('measured','comparable_pairs','concordant_pairs','forecast_ties'):
            summary[name][field] = sum(g[field] for g in evaluated)
    changed = [l['layout'] for l in layouts
               if l['metrics'][RAW]['selected_route'] != l['metrics']['corrected_acquisition_m2']['selected_route']]
    return dict(independent_layout_count=len(layouts), layouts=layouts, metrics=summary,
                changed_route_layouts=changed, failures=[l['layout'] for l in layouts if l['status']!='evaluated'],
                partial_routes=[dict(layout=l['layout'], route=row['route']) for l in layouts
                                for row in l['rows'] if not row['endpoint_reached']])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('split', choices=('calibration','heldout'))
    parser.add_argument('--run-root', type=Path, required=True)
    args = parser.parse_args()
    verify()
    out = HERE / f'{args.split}_labels.json'
    layouts = [evaluate_layout(args.run_root, key) for key, row in LAYOUTS.items() if row['split']==args.split]
    save(out, layouts)
    fit_file = HERE / 'calibrated_coefficients.json'
    if args.split == 'calibration':
        fit(layouts, fit_file, out)
        save(HERE / 'calibration_summary.json', comparison(layouts, read(fit_file)))
    else:
        fit_params = read(fit_file)
        for path, expected in fit_params['calibration_sources_sha256'].items():
            assert sha(path)==expected
        results = comparison(layouts, fit_params)
        results['fit_sha256'] = sha(fit_file)
        results['freeze_sha256'] = sha(HERE / 'freeze.json')
        save(HERE / 'heldout_summary.json', results)
    print(json.dumps(dict(status='evaluated', split=args.split, available=sum(bool(l['rows']) for l in layouts))))


if __name__ == '__main__':
    main()
