#!/usr/bin/env python3
"""Read-only legacy GT area/reference audit; never rerun a navigation policy."""
import argparse
from collections import Counter
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import sys
import time
import zipfile

import numpy as np
import open3d as o3d

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from env.virtual3d_v2 import VirtualConfigV2, VirtualWorldV2
from env.virtual3d_inspection_v4 import union_surface_from_boxes
from utils.reconstruction_metrics import surface_samples, ray_scene


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def recover_box_primitives(mesh, triangle_classes):
    """Reject arbitrary meshes: require each original 8-vertex/12-face box."""
    vertices = np.asarray(mesh.vertices)
    triangles = np.asarray(mesh.triangles)
    labels = np.asarray(triangle_classes)
    if len(vertices) % 8 or len(triangles) != len(vertices) // 8 * 12:
        raise ValueError('not a sequence of 8-vertex/12-triangle boxes')
    if labels.shape != (len(triangles),):
        raise ValueError('triangle classes must come from verified scene construction')
    primitives = []
    max_error = 0.
    for index in range(len(vertices) // 8):
        v = vertices[8 * index:8 * (index + 1)]
        t = triangles[12 * index:12 * (index + 1)] - 8 * index
        low, high = v.min(axis=0), v.max(axis=0)
        if np.any(high <= low):
            raise ValueError('degenerate primitive')
        canonical = o3d.geometry.TriangleMesh.create_box(*(high - low)).translate(low)
        if not np.array_equal(t, np.asarray(canonical.triangles)):
            raise ValueError('triangle block is not the canonical closed box topology')
        error = float(np.max(np.abs(v - np.asarray(canonical.vertices))))
        if error > 1e-12 or len(np.unique(v, axis=0)) != 8:
            raise ValueError('vertices do not define the canonical axis-aligned box')
        block_labels = labels[12 * index:12 * (index + 1)]
        if not np.all(block_labels == block_labels[0]):
            raise ValueError('nonuniform physical class within a box')
        max_error = max(max_error, error)
        primitives.append([*low, *(high - low), int(block_labels[0])])
    return np.asarray(primitives), dict(
        all_box_blocks_validated=True, primitive_count=len(primitives),
        maximum_reconstruction_coordinate_error_m=max_error,
        topology='canonical Open3D closed outward 8-vertex/12-triangle box')


def observable_mask(points, mesh, world):
    """Frozen evaluator visibility rule, independently matched to saved points."""
    visible = np.zeros(len(points), bool)
    truth = ray_scene(mesh)
    c = world.config
    vertical_tan = c.height_px / c.width_px * np.tan(np.deg2rad(c.fov_deg / 2))
    for r, col in np.argwhere(world.reachable):
        if r % 4 or col % 4:
            continue
        origin = np.array([(col + .5) * c.resolution_m,
                           (world.shape[0] - r - .5) * c.resolution_m,
                           c.camera_height_m])
        delta = points - origin
        horizontal = np.linalg.norm(delta[:, :2], axis=1)
        possible = ((~visible) & (horizontal > .15) & (horizontal < c.max_depth_m)
                    & (np.abs(delta[:, 2]) <= horizontal * vertical_tan))
        ids = np.flatnonzero(possible)
        if not len(ids):
            continue
        rays = np.column_stack([np.tile(origin, (len(ids), 1)), delta[ids]]).astype(np.float32)
        hit = truth.cast_rays(o3d.core.Tensor(rays), nthreads=1)['t_hit'].numpy()
        visible[ids[np.abs(hit - 1.) < 1e-4]] = True
    return visible


def surface_membership(points, normals, primitives):
    """Classify generic samples by solid occupancy and exterior multiplicity."""
    inward = points - normals * 1e-7
    outward = points + normals * 1e-7
    inside_inward = np.zeros(len(points), bool)
    inside_outward = np.zeros(len(points), bool)
    multiplicity = np.zeros(len(points), np.int32)
    axis = np.argmax(np.abs(normals), axis=1)
    sign = normals[np.arange(len(points)), axis]
    tolerance = 1e-9
    for box in primitives:
        low, high = box[:3], box[:3] + box[3:6]
        inside_inward |= ((inward > low) & (inward < high)).all(axis=1)
        inside_outward |= ((outward > low) & (outward < high)).all(axis=1)
        within = ((points >= low - tolerance) & (points <= high + tolerance)).all(axis=1)
        plane = np.where(sign > 0, high[axis], low[axis])
        on_face = np.abs(points[np.arange(len(points)), axis] - plane) <= tolerance
        multiplicity += within & on_face
    if not inside_inward.all() or not (multiplicity >= 1).all():
        raise AssertionError('invalid primitive sample orientation or boundary matching')
    return inside_outward, multiplicity


def triangle_normals(mesh, ids):
    v = np.asarray(mesh.vertices)
    t = np.asarray(mesh.triangles)[ids]
    normals = np.cross(v[t[:, 1]] - v[t[:, 0]], v[t[:, 2]] - v[t[:, 0]])
    return normals / np.linalg.norm(normals, axis=1)[:, None]


def class_shares(classes, weights=None):
    weights = np.ones(len(classes)) if weights is None else np.asarray(weights)
    total = weights.sum()
    return {str(category): float(weights[classes == category].sum() / total)
            for category in (1, 2, 3)}


def duplicate_triangles(mesh):
    vertices = np.asarray(mesh.vertices)
    triangles = np.asarray(mesh.triangles)
    seen = Counter(tuple(sorted(tuple(point) for point in vertices[triangle]))
                   for triangle in triangles)
    return dict(exact_duplicate_triangle_groups=sum(n > 1 for n in seen.values()),
                exact_duplicate_triangle_excess=sum(n - 1 for n in seen.values()),
                limitation='Exact full triangles only; partial coplanar overlap is diagnosed by sampled multiplicity.')


def audit_scene(run, config, entry):
    started = time.monotonic()
    c = VirtualConfigV2(**(config['environment'] | entry.get('environment', {})))
    args = {key: value for key, value in entry.items() if key != 'environment'}
    world = VirtualWorldV2(c, **args)
    name = (f"{args['layout']}_{args['seed']}_{args.get('semantic_condition', 'aligned')}"
            f'_d{c.depth_sigma_m}_p{c.pose_noise_m}_opaque{int(c.occluded_objects)}')
    path = run / (name + '_reference.npz')
    with np.load(path, allow_pickle=False) as frozen:
        for key, current in (('vertices', np.asarray(world.mesh.vertices)),
                             ('triangles', np.asarray(world.mesh.triangles)),
                             ('occupancy', world.occupancy), ('reachable', world.reachable)):
            if not np.array_equal(current, frozen[key]):
                raise AssertionError(f'{name}: frozen {key} mismatch')
        frozen_points = frozen['points'].copy()
        frozen_classes = frozen['classes'].copy()
    primitives, recovery = recover_box_primitives(world.mesh, world.triangle_classes)
    union, union_classes, union_audit = union_surface_from_boxes(primitives)
    original_area = float(world.mesh.get_surface_area())
    union_area = float(union.get_surface_area())
    count = config['reference_samples']
    points, ids = surface_samples(world.mesh, count, 2026)
    visible = observable_mask(points, world.mesh, world)
    if not np.array_equal(points[visible].astype(np.float32), frozen_points):
        raise AssertionError(f'{name}: reference points not reproduced exactly')
    classes = world.triangle_classes[ids]
    if not np.array_equal(classes[visible], frozen_classes):
        raise AssertionError(f'{name}: reference classes mismatch')
    internal, multiplicity = surface_membership(points, triangle_normals(world.mesh, ids), primitives)
    # Inverse multiplicity estimates uniform union area using existing samples.
    # Internal samples have zero weight; no metric or prediction is evaluated.
    weights = np.where(internal, 0., 1. / multiplicity)
    union_points, union_ids = surface_samples(union, count, 2026)
    union_visible = observable_mask(union_points, union, world)
    old_points_union_visibility = observable_mask(points, union, world)
    visible_multiplicity = multiplicity[visible & ~internal]
    visible_duplicate = visible & ~internal & (multiplicity > 1)
    leaked_points = points[visible & internal]
    leaked_distances = (ray_scene(union).compute_distance(
        o3d.core.Tensor(leaked_points.astype(np.float32)), nthreads=1).numpy()
        if len(leaked_points) else np.empty(0))
    exterior_estimate = original_area * float(weights.mean())
    visible_estimate = original_area * float(np.mean(weights * visible))
    effective_count = float(weights[visible].sum())
    return dict(
        scene=name, layout=args['layout'], seed=args['seed'], occluded_objects=c.occluded_objects,
        configuration=asdict(c), elapsed_s=time.monotonic() - started,
        frozen_reference_path=str(path.relative_to(ROOT)) if path.is_relative_to(ROOT) else str(path),
        frozen_reference_sha256=sha256(path), frozen_geometry_and_reachable_match=True,
        frozen_reference_points_and_classes_reproduced_exactly=True,
        primitive_recovery=recovery, union_surface=union_audit,
        geometry=dict(original_triangle_count=len(world.mesh.triangles), union_triangle_count=len(union.triangles),
                      original_sum_triangle_area_m2=original_area, unique_union_exterior_area_m2=union_area,
                      removed_area_m2=original_area - union_area,
                      removed_fraction_of_original=(original_area - union_area) / original_area,
                      original_overcount_relative_to_union=original_area / union_area - 1.,
                      **duplicate_triangles(world.mesh)),
        sampling=dict(candidate_count=count, seed=2026,
                      original_internal_sample_count=int(internal.sum()),
                      original_exterior_duplicate_sample_count=int(np.count_nonzero(~internal & (multiplicity > 1))),
                      estimated_unique_area_from_legacy_samples_m2=exterior_estimate,
                      estimated_unique_area_mc_standard_error_m2=original_area * float(weights.std(ddof=1)) / np.sqrt(count),
                      original_visible_reference_count=int(visible.sum()),
                      visible_internal_sample_count=int(np.count_nonzero(visible & internal)),
                      visible_internal_distance_to_union_m=leaked_distances.tolist(),
                      visible_exterior_duplicate_sample_count=int(visible_duplicate.sum()),
                      visible_exterior_duplicate_class_counts={str(k): int(v) for k, v in sorted(Counter(classes[visible_duplicate].tolist()).items())},
                      visible_exterior_duplicate_fraction=float(visible_duplicate.sum() / visible.sum()),
                      visible_exterior_multiplicity_histogram={str(k): int(v) for k, v in sorted(Counter(visible_multiplicity.tolist()).items())},
                      visible_inverse_multiplicity_weight_sum=effective_count,
                      visible_redundant_or_internal_weight_fraction=1. - effective_count / visible.sum(),
                      original_visible_measure_estimate_m2=original_area * float(visible.mean()),
                      unique_visible_measure_from_legacy_samples_estimate_m2=visible_estimate,
                      union_visible_reference_count=int(union_visible.sum()),
                      unique_visible_measure_from_new_union_samples_estimate_m2=union_area * float(union_visible.mean()),
                      original_samples_visibility_change_with_union_ray_truth=int(np.count_nonzero(old_points_union_visibility != visible)),
                      original_visible_class_shares=class_shares(classes[visible]),
                      deweighted_original_visible_class_shares=class_shares(classes[visible], weights[visible]),
                      union_visible_class_shares=class_shares(union_classes[union_ids[union_visible]])),
        limitations=['Areas include all exterior surfaces, while metrics use reachable-view visibility filtering.',
                     'Reference diagnostics are finite-sample estimates, not reconstructed-mesh metric changes.',
                     'Multiplicity is physical face coverage, not identical sampled coordinate duplication.',
                     'Unique-reference class shares also include independent Monte Carlo resampling variation.'])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, default=ROOT / 'eval_results/joint_v4_feedback_development_20260910')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--compare-run', type=Path, action='append', default=[],
                        help='Read-only check that another completed run used identical references.')
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError('audit output must be new; frozen results are never overwritten')
    run = args.run.resolve()
    if args.output.resolve().is_relative_to(run):
        raise ValueError('write this audit outside the frozen experiment directory')
    metadata = json.loads((run / 'run_metadata.json').read_text())
    config = json.loads((run / 'config.json').read_text())
    if metadata['status'] != 'complete':
        raise ValueError('audit requires a completed source run')
    frozen_dependencies = ('env/__init__.py', 'env/virtual3d.py', 'env/virtual3d_v2.py',
                           'utils/grid_geometry.py',
                           'utils/rgbd_contract.py', 'utils/reconstruction_metrics.py')
    sources = (*frozen_dependencies, 'env/virtual3d_inspection_v4.py',
               'scripts/audit_legacy_surface_reference.py')
    before = {name: sha256(ROOT / name) for name in sources}
    with zipfile.ZipFile(run / 'sources.zip') as archive:
        for name in frozen_dependencies:
            recorded = metadata['files_sha256'][name]
            if before[name] != recorded or hashlib.sha256(archive.read(name)).hexdigest() != recorded:
                raise AssertionError(f'legacy dependency differs from frozen source: {name}')
    rows = []
    for entry in config['scenes']:
        row = audit_scene(run, config, entry)
        rows.append(row)
        print(row['scene'], 'area reduction', round(100 * row['geometry']['removed_fraction_of_original'], 3),
              '%', 'visible duplicate', row['sampling']['visible_exterior_duplicate_sample_count'],
              '/', row['sampling']['original_visible_reference_count'], flush=True)
    related_runs = []
    for other in args.compare_run:
        other = other.resolve()
        other_metadata = json.loads((other / 'run_metadata.json').read_text())
        if other_metadata['status'] != 'complete':
            raise ValueError(f'comparison run is not completed: {other}')
        matches = []
        for row in rows:
            name = row['scene'] + '_reference.npz'
            with np.load(run / name, allow_pickle=False) as first, np.load(other / name, allow_pickle=False) as second:
                if set(first.files) != set(second.files) or any(
                        not np.array_equal(first[key], second[key]) for key in first.files):
                    raise AssertionError(f'comparison reference mismatch: {other / name}')
            matches.append(dict(scene=row['scene'], sha256=sha256(other / name), all_arrays_equal=True))
        related_runs.append(dict(run=str(other), metadata_sha256=sha256(other / 'run_metadata.json'),
                                 scene_count=len(matches), references=matches))
    if any(sha256(ROOT / name) != value for name, value in before.items()):
        raise AssertionError('audit dependencies changed during execution')
    output = dict(
        scope='Posthoc geometry/reference sensitivity audit only; no navigation or reconstruction metric reevaluation.',
        source_run=str(run), source_run_metadata_sha256=sha256(run / 'run_metadata.json'),
        source_archive_sha256=sha256(run / 'sources.zip'), source_config_sha256=sha256(run / 'config.json'),
        source_sha256=before, legacy_dependencies_match_frozen_archive=True,
        related_reference_runs=related_runs,
        reference_sampling_count=config['reference_samples'], reference_sampling_seed=2026,
        scene_count=len(rows), frozen_results_modified=False, navigation_episodes_run=0,
        prediction_mesh_metrics_recomputed=False, scenes=rows,
        interpretation='Total triangle area reduction is not F1, recall, precision, or joint-metric bias. '
                       'Visibility removes most internal surfaces, but exposed coplanar duplicates can retain extra weight. '
                       'The direction of any method contrast change requires reevaluating identical stored reconstructions.')
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('x') as stream:
        json.dump(output, stream, indent=2, allow_nan=False)
        stream.write('\n')


if __name__ == '__main__':
    main()
