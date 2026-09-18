"""Evaluator-only external-geometry contract for the six-facility V19 task.

References, attribution windows and diagnostic cameras are fixed from the
world before trajectories are scored. No category enters metric computation.
The V19 world contains solid exteriors and exposed attachments only, so its
legacy all-facility surface target equals the external target by construction.
"""
from dataclasses import asdict
import hashlib
import json
from pathlib import Path

import numpy as np
import open3d as o3d
from scipy.ndimage import binary_erosion, distance_transform_edt

from env.virtual3d import camera_pose
from utils.facility_metrics_v18 import observable_reference
from utils.reconstruction_metrics import ReconstructionEvaluator, ray_scene, surface_samples


CONTRACT = 'facility-external-v19-1'


def _array_hash(*arrays):
    h = hashlib.sha256()
    for array in arrays:
        a = np.ascontiguousarray(array)
        h.update(str(a.dtype).encode()); h.update(str(a.shape).encode()); h.update(a.tobytes())
    return h.hexdigest()


def clip_mesh_to_bounds(mesh, bounds):
    """Clip triangles, including crossing triangles, against a fixed AABB.

    There are no synthetic cap faces. Unlike centroid/vertex selection, this
    includes the in-window portion of a large false triangle crossing a window.
    """
    bounds = np.asarray(bounds, float)
    vertices = np.asarray(mesh.vertices); triangles = np.asarray(mesh.triangles)
    output = o3d.geometry.TriangleMesh()
    if not len(triangles):
        return output
    faces = vertices[triangles]
    intersects = ((faces.max(axis=1) >= bounds[0]) & (faces.min(axis=1) <= bounds[1])).all(axis=1)
    faces = faces[intersects]
    contained = ((faces >= bounds[0]) & (faces <= bounds[1])).all(axis=(1, 2))
    fragments = list(faces[contained])
    for face in faces[~contained]:
        polygon = list(face)
        for axis in range(3):
            for side in range(2):
                if not polygon:
                    break
                limit = bounds[side, axis]
                clipped = []
                previous = polygon[-1]
                previous_inside = previous[axis] >= limit if side == 0 else previous[axis] <= limit
                for point in polygon:
                    inside = point[axis] >= limit if side == 0 else point[axis] <= limit
                    if inside != previous_inside:
                        fraction = (limit - previous[axis]) / (point[axis] - previous[axis])
                        clipped.append(previous + fraction * (point - previous))
                    if inside:
                        clipped.append(point)
                    previous, previous_inside = point, inside
                polygon = clipped
        for j in range(1, len(polygon) - 1):
            triangle = np.asarray([polygon[0], polygon[j], polygon[j + 1]])
            if np.linalg.norm(np.cross(triangle[1] - triangle[0], triangle[2] - triangle[0])) > 1e-12:
                fragments.append(triangle)
    if fragments:
        vertices = np.asarray(fragments).reshape(-1, 3)
        output.vertices = o3d.utility.Vector3dVector(vertices)
        output.triangles = o3d.utility.Vector3iVector(np.arange(len(vertices)).reshape(-1, 3))
    return output


def _error_stats(distance):
    if not len(distance) or not np.isfinite(distance).all():
        return dict(mean_m=None, rmse_m=None, p95_m=None)
    return dict(mean_m=float(np.mean(distance)), rmse_m=float(np.sqrt(np.mean(np.square(distance)))),
                p95_m=float(np.percentile(distance, 95)))


def _camera_rays(world, pose):
    c = world.config
    v, u = np.mgrid[:c.height_px, :c.width_px]
    local = np.stack([(u - world.intrinsic[0, 2]) / world.intrinsic[0, 0],
                      (v - world.intrinsic[1, 2]) / world.intrinsic[1, 1], np.ones_like(u)], axis=-1)
    directions = local.reshape(-1, 3) @ pose[:3, :3].T
    return np.column_stack([np.tile(pose[:3, 3], (len(directions), 1)), directions]).astype(np.float32)


def _fixed_views(world, instance_truth, world_truth, bounds, stride):
    """Up to four full-object cardinal views, chosen without a reconstruction."""
    center = bounds.mean(axis=0)
    corners = np.asarray([[x, y, z] for x in bounds[:, 0] for y in bounds[:, 1] for z in bounds[:, 2]])
    c = world.config; views = []
    cells = [(int(r), int(col)) for r, col in np.argwhere(world.reachable) if r % stride == 0 and col % stride == 0]
    for heading in range(4):
        choices = []
        for cell in cells:
            pose = camera_pose(cell, heading, c, world.shape[0])
            local = (corners - pose[:3, 3]) @ pose[:3, :3]
            if np.any(local[:, 2] <= .15) or np.any(local[:, 2] > c.max_depth_m):
                continue
            uv = local[:, :2] / local[:, 2, None]
            uv = uv * [world.intrinsic[0, 0], world.intrinsic[1, 1]] + world.intrinsic[:2, 2]
            if np.any(uv < [1, 1]) or np.any(uv > [c.width_px - 2, c.height_px - 2]):
                continue
            target_local = (center - pose[:3, 3]) @ pose[:3, :3]
            cost = (target_local[2] - 2.5) ** 2 + 2. * target_local[0] ** 2
            choices.append((float(cost), cell, pose))
        for _, cell, pose in sorted(choices, key=lambda x: (x[0], x[1])):
            rays = _camera_rays(world, pose)
            tensor = o3d.core.Tensor(rays)
            hit = instance_truth.cast_rays(tensor, nthreads=1)['t_hit'].numpy()
            world_hit = world_truth.cast_rays(tensor, nthreads=1)['t_hit'].numpy()
            in_range = np.isfinite(hit) & (hit > .15) & (hit <= c.max_depth_m)
            occluded = in_range & (world_hit < hit - 1e-4)
            visible = in_range & ~occluded
            if visible.sum() < 8:
                continue
            views.append(dict(cell=cell, heading=heading, pose=pose, rays=rays,
                              truth_mask=visible.reshape(c.height_px, c.width_px),
                              valid_mask=(~occluded).reshape(c.height_px, c.width_px)))
            break
    return views


class FacilityEvaluatorV19:
    def __init__(self, world, count_per_asset=3000, seed=2026, reference_cache=None,
                 predicted_per_asset=4000, reference_stride=4, global_count=12000):
        if min(count_per_asset, predicted_per_asset, reference_stride, global_count) < 1:
            raise ValueError('positive sample counts and reference stride required')
        if not world.objects:
            raise ValueError('mission requires fixed facilities')
        self.config = world.config; self.seed = int(seed)
        self.predicted_per_asset = int(predicted_per_asset)
        self.truth = ray_scene(world.mesh)
        specs = [dict(id=int(item['id']), bounds=np.asarray(item['evaluation_bounds'], float).tolist())
                 for item in world.objects]
        if len({item['id'] for item in specs}) != len(specs):
            raise ValueError('duplicate mission instance id')
        for i, item in enumerate(specs):
            bounds = np.asarray(item['bounds'])
            if bounds.shape != (2, 3) or not np.isfinite(bounds).all() or np.any(bounds[0] >= bounds[1]):
                raise ValueError('invalid evaluation window')
            for previous in specs[:i]:
                other = np.asarray(previous['bounds'])
                if np.all(np.minimum(bounds[1], other[1]) > np.maximum(bounds[0], other[0])):
                    raise ValueError('mission attribution windows overlap')
        config = asdict(world.config)
        signature_payload = dict(contract=CONTRACT, count_per_asset=int(count_per_asset), seed=self.seed,
            reference_stride=int(reference_stride), global_count=int(global_count), config=config, specs=specs,
            world_hash=_array_hash(np.asarray(world.mesh.vertices), np.asarray(world.mesh.triangles),
                                   np.asarray(world.triangle_classes), world.reachable, world.intrinsic),
            implementation_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
        self.reference_signature = hashlib.sha256(json.dumps(signature_payload, sort_keys=True).encode()).hexdigest()
        path = Path(reference_cache) if reference_cache is not None else None
        cached = None
        if path is not None and path.exists():
            with np.load(path, allow_pickle=False) as archive:
                cached = {name: archive[name].copy() for name in archive.files}
            metadata = json.loads(str(cached['metadata']))
            if metadata['reference_signature'] != self.reference_signature:
                raise ValueError('reference cache does not match frozen world or evaluator contract')
        if cached is None:
            self.global_evaluator = ReconstructionEvaluator(world, count=global_count, seed=seed)
        else:
            self.global_evaluator = ReconstructionEvaluator.__new__(ReconstructionEvaluator)
            self.global_evaluator.truth = self.truth; self.global_evaluator.config = world.config
            self.global_evaluator.reference = cached['global_reference']
            self.global_evaluator.classes = cached['global_classes']
        self.instances = []; storage = {}
        for spec in specs:
            identifier = spec['id']; bounds = np.asarray(spec['bounds'])
            mesh = world.instance_mesh(identifier); truth = ray_scene(mesh)
            prefix = f'asset_{identifier}'
            if cached is None:
                points, _ = surface_samples(mesh, count_per_asset, seed + 1009 * identifier)
                visible = observable_reference(world, points, self.truth, stride=reference_stride)
                if not visible.any():
                    raise ValueError('mission instance has no observable external surface')
                reference = points[visible].astype(np.float32)
                area = float(mesh.get_surface_area()) * float(visible.mean())
                # Apply the same fixed window to the diagnostic target so a
                # floor-exclusion margin cannot lower the perfect-mesh IoU.
                silhouette_truth = ray_scene(clip_mesh_to_bounds(mesh, bounds))
                views = _fixed_views(world, silhouette_truth, self.truth, bounds, reference_stride)
                sampled = len(points)
            else:
                reference = cached[f'{prefix}_reference']; area = float(cached[f'{prefix}_area'])
                sampled = int(cached[f'{prefix}_sampled']); views = []
                for j in range(int(cached[f'{prefix}_views'])):
                    stem = f'{prefix}_view_{j}'
                    views.append(dict(cell=tuple(int(x) for x in cached[f'{stem}_cell']),
                        heading=int(cached[f'{stem}_heading']), pose=cached[f'{stem}_pose'],
                        rays=cached[f'{stem}_rays'], truth_mask=cached[f'{stem}_truth_mask'],
                        valid_mask=cached[f'{stem}_valid_mask']))
            self.instances.append(dict(id=identifier, bounds=bounds, truth=truth, reference=reference,
                sampled=sampled, observable_samples=len(reference), observable_area_m2=area, views=views))
            storage[f'{prefix}_reference'] = reference; storage[f'{prefix}_area'] = np.asarray(area)
            storage[f'{prefix}_sampled'] = np.asarray(sampled); storage[f'{prefix}_views'] = np.asarray(len(views))
            for j, view in enumerate(views):
                for key, value in view.items():
                    storage[f'{prefix}_view_{j}_{key}'] = np.asarray(value)
        if path is not None and cached is None:
            metadata = dict(reference_signature=self.reference_signature, signature_payload=signature_payload)
            storage.update(metadata=np.asarray(json.dumps(metadata, sort_keys=True)),
                           global_reference=self.global_evaluator.reference, global_classes=self.global_evaluator.classes)
            path.parent.mkdir(parents=True, exist_ok=True)
            # Cache creation is exclusive: never overwrite an existing frozen reference.
            with path.open('xb') as handle:
                np.savez_compressed(handle, **storage)

    def _silhouette(self, reconstruction, views):
        rows = []
        for view in views:
            truth = view['truth_mask']; valid = view['valid_mask']
            if reconstruction is None:
                predicted = np.zeros_like(truth)
            else:
                hit = reconstruction.cast_rays(o3d.core.Tensor(view['rays']), nthreads=1)['t_hit'].numpy()
                predicted = (np.isfinite(hit) & (hit > .15) & (hit <= self.config.max_depth_m)).reshape(truth.shape) & valid
            intersection = int(np.sum(predicted & truth)); union = int(np.sum(predicted | truth))
            truth_boundary = truth & ~binary_erosion(truth)
            predicted_boundary = predicted & ~binary_erosion(predicted)
            boundary_distance = None
            if predicted_boundary.any() and truth_boundary.any():
                distances = np.concatenate([distance_transform_edt(~truth_boundary)[predicted_boundary],
                                            distance_transform_edt(~predicted_boundary)[truth_boundary]])
                boundary_distance = float(np.mean(distances))
            rows.append(dict(cell=list(view['cell']), heading=int(view['heading']),
                truth_pixels=int(truth.sum()), predicted_pixels=int(predicted.sum()), iou=intersection / union if union else 1.,
                boundary_symmetric_mean_px=boundary_distance))
        return dict(views=rows, mean_iou=float(np.mean([x['iou'] for x in rows])) if rows else None,
                    scope='fixed reachable cardinal cameras; truth-occluded target pixels excluded; diagnostic only')

    def evaluate(self, mesh, coverage, *, returned=False, collisions=0, failed=False, thresholds=(.02, .05, .10)):
        if not np.isfinite(coverage) or not 0 <= coverage <= 1:
            raise ValueError('coverage outside [0,1]')
        if collisions < 0 or int(collisions) != collisions:
            raise ValueError('collision count must be a nonnegative integer')
        thresholds = tuple(float(x) for x in thresholds)
        if not thresholds or any(not np.isfinite(x) or x <= 0 for x in thresholds):
            raise ValueError('positive finite thresholds required')
        rows = []
        for item in self.instances:
            attributed = clip_mesh_to_bounds(mesh, item['bounds'])
            predicted, _ = surface_samples(attributed, self.predicted_per_asset, 812 + 1009 * item['id'])
            reconstruction = ray_scene(attributed) if len(attributed.triangles) else None
            accuracy = item['truth'].compute_distance(o3d.core.Tensor(predicted.astype(np.float32)), nthreads=1).numpy() if len(predicted) else np.empty(0)
            recall_distance = reconstruction.compute_distance(o3d.core.Tensor(item['reference']), nthreads=1).numpy() if reconstruction is not None else np.full(len(item['reference']), np.inf)
            accuracy_stats = _error_stats(accuracy); completeness_stats = _error_stats(recall_distance)
            reference_bounds = np.asarray([item['reference'].min(axis=0), item['reference'].max(axis=0)])
            reference_size = reference_bounds[1] - reference_bounds[0]
            predicted_bounds = None; size_error = None; center_error = None
            if len(attributed.vertices):
                vertices = np.asarray(attributed.vertices)
                predicted_bounds = np.asarray([vertices.min(axis=0), vertices.max(axis=0)])
                size_error = np.abs((predicted_bounds[1] - predicted_bounds[0]) - reference_size)
                center_error = float(np.linalg.norm(predicted_bounds.mean(axis=0) - reference_bounds.mean(axis=0)))
            row = dict(id=item['id'], predicted_samples=len(predicted), reference_samples=item['observable_samples'],
                observable_area_m2=item['observable_area_m2'], missing=not len(predicted),
                surface_error_mean_m=accuracy_stats['mean_m'], surface_error_rmse_m=accuracy_stats['rmse_m'],
                surface_error_p95_m=accuracy_stats['p95_m'], accuracy=accuracy_stats, completeness=completeness_stats,
                dimensions=dict(reference_size_xyz_m=reference_size.tolist(),
                    predicted_size_xyz_m=(predicted_bounds[1] - predicted_bounds[0]).tolist() if predicted_bounds is not None else None,
                    absolute_size_error_xyz_m=size_error.tolist() if size_error is not None else None,
                    mean_absolute_size_error_m=float(size_error.mean()) if size_error is not None else None,
                    reference_center_error_m=center_error,
                    scope='AABB of fixed observable reference versus clipped prediction; missing is null, not zero'),
                silhouette=self._silhouette(reconstruction, item['views']))
            for threshold in thresholds:
                p = float(np.mean(accuracy <= threshold)) if len(accuracy) else 0.
                r = float(np.mean(recall_distance <= threshold))
                row[f'{round(threshold * 100):02d}cm'] = dict(precision=p, recall=r, f1=2*p*r/(p+r) if p+r else 0.)
            rows.append(row)
        result = dict(contract=CONTRACT, reference_signature=self.reference_signature, instances=rows,
            mission_asset_count=len(rows), missing_asset_count=sum(x['missing'] for x in rows), coverage_2d=float(coverage),
            returned=bool(returned), collisions=int(collisions), failed=bool(failed),
            eligible=bool(coverage >= .8 and returned and collisions == 0 and not failed),
            reference_source='fixed reachable lattice and four executable cardinal headings; independent of trajectory',
            attribution='exact triangle clipping to fixed disjoint instance windows; outside-window errors retained by global metric',
            legacy_facility_relation='same target surface: V19 solid exteriors and exposed attachments contain no task-excluded interiors; V19 uses per-window sampling rather than V18 global prediction sampling')
        for threshold in thresholds:
            tag = f'{round(threshold * 100):02d}cm'; values = [row[tag]['f1'] for row in rows]
            mean = float(np.mean(values))
            result[tag] = dict(external_macro_f1=mean, joint_external=float(coverage * mean), asset_macro_f1=mean,
                joint_asset=float(coverage * mean), macro_precision=float(np.mean([x[tag]['precision'] for x in rows])),
                macro_recall=float(np.mean([x[tag]['recall'] for x in rows])),
                completion_fraction=float(np.mean([x[tag]['precision'] >= .95 and x[tag]['recall'] >= .8 for x in rows])),
                worst_quartile_mean_f1=float(np.mean(sorted(values)[:max(1, int(np.ceil(len(values) / 4)))])))
        # Missing geometry has no finite geometric distance; null summaries must
        # always be read alongside missing counts and the zero-scored main F1.
        result['continuous_external'] = dict(missing_asset_count=result['missing_asset_count'])
        for direction in ('accuracy', 'completeness'):
            result['continuous_external'][direction] = {}
            for statistic in ('mean_m', 'rmse_m', 'p95_m'):
                values = [row[direction][statistic] for row in rows]
                result['continuous_external'][direction][f'complete_mission_macro_{statistic}'] = float(np.mean(values)) if all(x is not None for x in values) else None
        result['global_legacy'] = self.global_evaluator.evaluate(mesh, coverage, thresholds=thresholds)
        result['facility_surface_legacy'] = dict(target_identical_to_external=True, estimator='V19 fixed-window estimator; not a replay of V18 sampling',
            scores={f'{round(t * 100):02d}cm':dict(result[f'{round(t * 100):02d}cm']) for t in thresholds})
        return result
