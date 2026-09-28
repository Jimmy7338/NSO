"""Evaluation-only three-projection exterior-outline task, not 3D surface F1.

Triangle projections are united in fixed world XY/XZ/YZ coordinates. Bounded
holes are excluded from this envelope task; disconnected exterior components
remain. No 3D volume filling, category, visit count or prediction-side alignment.
"""
import hashlib
import json

import numpy as np
import shapely
from shapely.geometry import GeometryCollection, Polygon

from utils.facility_metrics_v19 import clip_mesh_to_bounds


CONTRACT = 'facility-three-projection-outline-v23-1'
PROJECTIONS = {'xy': (0, 1), 'xz': (0, 2), 'yz': (1, 2)}


def mesh_arrays(mesh):
    vertices = np.asarray(mesh.vertices, dtype=float)
    triangles = np.asarray(mesh.triangles)
    if vertices.ndim != 2 or vertices.shape[1] != 3 or not np.isfinite(vertices).all():
        raise ValueError('finite Nx3 mesh vertices required')
    if triangles.ndim != 2 or triangles.shape[1] != 3 or triangles.dtype.kind not in 'iu':
        raise ValueError('integer Nx3 mesh triangles required')
    if len(triangles) and (triangles.min() < 0 or triangles.max() >= len(vertices)):
        raise ValueError('triangle index outside mesh')
    return vertices, triangles


def polygon_parts(geometry):
    if geometry.is_empty:
        return []
    if geometry.geom_type == 'Polygon':
        return [geometry]
    if geometry.geom_type in ('MultiPolygon', 'GeometryCollection'):
        return [p for g in geometry.geoms for p in polygon_parts(g)]
    return []


def projected_envelope(mesh, axes):
    vertices, triangles = mesh_arrays(mesh)
    if not len(triangles):
        return GeometryCollection()
    points = vertices[triangles][:, :, axes]
    a = points[:, 1] - points[:, 0]
    b = points[:, 2] - points[:, 0]
    area2 = np.abs(a[:, 0] * b[:, 1] - a[:, 1] * b[:, 0])
    # Truly degenerate projections are not thickened into artificial surfaces.
    points = points[area2 > 1e-14]
    if not len(points):
        return GeometryCollection()
    geometry = shapely.union_all(shapely.polygons(points), grid_size=0.)
    shells = [Polygon(p.exterior) for p in polygon_parts(geometry)]
    result = shapely.union_all(shells, grid_size=0.).simplify(0.)
    if not result.is_valid:
        raise ValueError('invalid projected polygon union')
    return result


def boundary_samples(geometry, spacing):
    """Midpoint quadrature with actual segment-length weights, plus vertices.

    Every boundary point is at most max_spacing/2 from a midpoint sample. This
    provides a conservative Hausdorff upper bound via 1-Lipschitz distance.
    """
    points, weights, endpoints, max_spacing = [], [], [], 0.
    for polygon in polygon_parts(geometry):
        coords = np.asarray(polygon.exterior.coords)
        start, delta = coords[:-1], np.diff(coords, axis=0)
        for p, vector in zip(start, delta):
            length = float(np.linalg.norm(vector))
            if length <= 0:
                continue
            count = max(1, int(np.ceil(length / spacing)))
            step = length / count
            points.append(p + ((np.arange(count) + .5) / count)[:, None] * vector)
            weights.append(np.full(count, step))
            endpoints.append(p)
            max_spacing = max(max_spacing, step)
    return (np.concatenate(points) if points else np.empty((0, 2)),
            np.concatenate(weights) if weights else np.empty(0),
            np.asarray(endpoints).reshape(-1, 2), max_spacing)


def _directed(source, target, spacing):
    points, weights, endpoints, max_spacing = boundary_samples(source, spacing)
    distances = shapely.distance(shapely.points(points), target.boundary)
    end_distances = shapely.distance(shapely.points(endpoints), target.boundary)
    maximum = float(max(np.max(distances), np.max(end_distances)))
    return dict(distances=distances, weights=weights,
                mean=float(np.average(distances, weights=weights)),
                lower=maximum, upper=maximum + max_spacing / 2,
                samples=len(points), length=float(weights.sum()))


def compare_projection(predicted, reference, spacing, thresholds):
    output = dict(predicted_area_m2=float(predicted.area), reference_area_m2=float(reference.area),
                  predicted_components=len(polygon_parts(predicted)),
                  reference_components=len(polygon_parts(reference)))
    if predicted.is_empty:
        output.update(iou=0., boundary_symmetric_mean_m=None, hausdorff_lower_m=None,
                      hausdorff_upper_m=None, missing_projection=True,
                      predicted_boundary_samples=0, reference_boundary_samples=0)
        for t in thresholds:
            output[f'{round(t * 100):02d}cm'] = dict(precision=0., recall=0., f1=0.)
        return output
    forward = _directed(predicted, reference, spacing)
    reverse = _directed(reference, predicted, spacing)
    equal = predicted.boundary.equals(reference.boundary)
    output.update(iou=float(predicted.intersection(reference).area / predicted.union(reference).area),
                  boundary_symmetric_mean_m=(forward['mean'] + reverse['mean']) / 2,
                  hausdorff_lower_m=0. if equal else max(forward['lower'], reverse['lower']),
                  hausdorff_upper_m=0. if equal else max(forward['upper'], reverse['upper']),
                  missing_projection=False, predicted_boundary_samples=forward['samples'],
                  reference_boundary_samples=reverse['samples'],
                  predicted_boundary_length_m=forward['length'], reference_boundary_length_m=reverse['length'])
    for t in thresholds:
        p = float(np.average(forward['distances'] <= t + 1e-12, weights=forward['weights']))
        r = float(np.average(reverse['distances'] <= t + 1e-12, weights=reverse['weights']))
        output[f'{round(t * 100):02d}cm'] = dict(precision=p, recall=r, f1=2*p*r/(p+r) if p+r else 0.)
    return output


class OutlineEvaluatorV23:
    """Freeze evaluation-side instance attribution and reference projections.

    `assets` contains id, mesh and disjoint bounds; runtime must never receive
    this evaluator or its references. All three projections are required, even
    when the current trajectory did not observe the corresponding surfaces.
    """
    def __init__(self, assets, boundary_spacing_m=.01, thresholds=(.02, .05, .10),
                 completion_tolerance_m=.05, completion_minimum_iou=.90):
        if not assets:
            raise ValueError('at least one task asset required')
        values = [boundary_spacing_m, completion_tolerance_m, *thresholds]
        if not thresholds or not np.isfinite(values).all() or min(values) <= 0:
            raise ValueError('positive finite spacing and tolerances required')
        if .05 not in thresholds or len({round(t*100) for t in thresholds}) != len(thresholds):
            raise ValueError('unique centimetre threshold tags including 5cm required')
        if not np.isfinite(completion_minimum_iou) or not 0 < completion_minimum_iou <= 1:
            raise ValueError('completion IoU must be in (0,1]')
        self.spacing = float(boundary_spacing_m)
        self.thresholds = tuple(float(t) for t in thresholds)
        self.completion_tolerance = float(completion_tolerance_m)
        self.completion_iou = float(completion_minimum_iou)
        self.assets, signatures = [], []
        for item in assets:
            identifier = item['id']
            if isinstance(identifier, bool) or not isinstance(identifier, int):
                raise ValueError('integer asset id required')
            bounds = np.array(item['bounds'], dtype=float, copy=True)
            if bounds.shape != (2, 3) or not np.isfinite(bounds).all() or np.any(bounds[0] >= bounds[1]):
                raise ValueError('finite nondegenerate attribution bounds required')
            for previous in self.assets:
                if identifier == previous['id']:
                    raise ValueError('duplicate task asset id')
                if np.all(np.minimum(bounds[1], previous['bounds'][1]) > np.maximum(bounds[0], previous['bounds'][0])):
                    raise ValueError('overlapping task attribution windows')
            raw_vertices, raw_triangles = mesh_arrays(item['mesh'])
            if len(raw_triangles):
                used_raw = raw_vertices[np.unique(raw_triangles)]
                lower, upper = used_raw.min(axis=0), used_raw.max(axis=0)
                if (np.any(lower[:2] < bounds[0, :2] - 1e-9)
                        or np.any(upper > bounds[1] + 1e-9)):
                    raise ValueError('task attribution window truncates reference geometry')
                if lower[2] < bounds[0, 2] - 1e-9 and not (
                        lower[2] >= -1e-9 and 0 <= bounds[0, 2] <= .01 + 1e-9):
                    raise ValueError('only the declared ground-contact band [0,0.01m] may be excluded')
            truth = clip_mesh_to_bounds(item['mesh'], bounds)
            vertices, triangles = mesh_arrays(truth)
            if not len(triangles):
                raise ValueError('empty clipped task reference')
            projections = {name: projected_envelope(truth, axes) for name, axes in PROJECTIONS.items()}
            if any(g.is_empty for g in projections.values()):
                raise ValueError('task requires nondegenerate reference in all three projections')
            used = vertices[np.unique(triangles)]
            shape_bounds = np.array([used.min(axis=0), used.max(axis=0)])
            self.assets.append(dict(id=identifier, bounds=bounds, projections=projections, shape_bounds=shape_bounds))
            signatures.append(dict(id=identifier, bounds=bounds.tolist(),
                mesh_sha256=hashlib.sha256(vertices.tobytes()+triangles.tobytes()).hexdigest()))
        self.contract_parameters = dict(contract=CONTRACT, projections=PROJECTIONS,
            boundary_spacing_m=self.spacing, thresholds=self.thresholds,
            completion_tolerance_m=self.completion_tolerance,
            completion_minimum_iou=self.completion_iou, assets=signatures,
            quality='mean_projections(min(boundary_length_F1,projection_IoU))',
            reference_crop='containment required except an optional ground-contact band within [0,0.01m]',
            shapely_version=shapely.__version__, geos_version=shapely.geos_version_string)
        self.reference_signature = hashlib.sha256(json.dumps(self.contract_parameters, sort_keys=True).encode()).hexdigest()

    @classmethod
    def from_world(cls, world, **kwargs):
        return cls([dict(id=int(item['id']), mesh=world.instance_mesh(item['id']),
                         bounds=item['evaluation_bounds']) for item in world.objects], **kwargs)

    def evaluate(self, mesh, coverage, *, returned=False, collisions=0, failed=False,
                 paid_actions=None, budget=None):
        mesh_arrays(mesh)
        if isinstance(coverage, bool) or not np.isfinite(coverage) or not 0 <= coverage <= 1:
            raise ValueError('coverage must be in [0,1]')
        if isinstance(collisions, bool) or not isinstance(collisions, int) or collisions < 0:
            raise ValueError('nonnegative integer collision count required')
        if (paid_actions is None) != (budget is None):
            raise ValueError('paid actions and budget must be provided together')
        if budget is not None and any(isinstance(x, bool) or not isinstance(x, int) or x < 0 for x in (paid_actions, budget)):
            raise ValueError('nonnegative integer paid actions and budget required')
        rows = []
        for item in self.assets:
            prediction = clip_mesh_to_bounds(mesh, item['bounds'])
            vertices, triangles = mesh_arrays(prediction)
            projections = {name: compare_projection(projected_envelope(prediction, axes),
                item['projections'][name], self.spacing, self.thresholds) for name, axes in PROJECTIONS.items()}
            dimensions = dict(absolute_size_error_xyz_m=None, center_error_inf_m=None,
                              reference_size_xyz_m=np.diff(item['shape_bounds'], axis=0)[0].tolist(),
                              predicted_size_xyz_m=None)
            if len(triangles):
                used = vertices[np.unique(triangles)]
                box = np.array([used.min(axis=0), used.max(axis=0)])
                size = np.diff(box, axis=0)[0]
                dimensions.update(predicted_size_xyz_m=size.tolist(),
                    absolute_size_error_xyz_m=np.abs(size-np.diff(item['shape_bounds'], axis=0)[0]).tolist(),
                    center_error_inf_m=float(np.max(np.abs(box.mean(axis=0)-item['shape_bounds'].mean(axis=0)))))
            tol = self.completion_tolerance
            completed = bool(len(triangles) and all(not p['missing_projection'] and p['hausdorff_upper_m'] <= tol
                and p['iou'] >= self.completion_iou
                for p in projections.values()) and max(dimensions['absolute_size_error_xyz_m']) <= tol
                and dimensions['center_error_inf_m'] <= tol)
            row = dict(id=item['id'], missing=not len(triangles), projections=projections,
                       dimensions=dimensions, completed=completed,
                       measurement_provenance='raw submitted mesh; this evaluator does not certify which surfaces were observed or inferred')
            for t in self.thresholds:
                tag = f'{round(t * 100):02d}cm'
                row[tag] = dict(outline_f1=float(np.mean([p[tag]['f1'] for p in projections.values()])),
                    outline_quality=float(np.mean([min(p[tag]['f1'], p['iou']) for p in projections.values()])))
            rows.append(row)
        compliant = None if budget is None else paid_actions <= budget
        result = dict(contract=CONTRACT, reference_signature=self.reference_signature, instances=rows,
            coverage_2d=float(coverage), mission_asset_count=len(rows), missing_asset_count=sum(r['missing'] for r in rows),
            completion_fraction=float(np.mean([r['completed'] for r in rows])), returned=bool(returned),
            collisions=collisions, failed=bool(failed), primitive_budget_compliant=compliant,
            eligible=bool(coverage >= .8 and returned and not collisions and not failed and compliant is True),
            budget_verified=budget is not None,
            qualification_incomplete=[] if budget is not None else ['primitive_action_budget'],
            scope='three orthogonal exterior outlines, not unique 3D geometry, observed surface completeness or volumetric occupancy',
            attribution='fixed disjoint evaluation-only windows; outside-window geometry requires companion global metrics',
            completion_is_primary_reward=False, reference_is_trajectory_independent=True,
            category_or_visit_count_used=False, three_dimensional_hole_filling=False)
        for t in self.thresholds:
            tag = f'{round(t * 100):02d}cm'
            quality = float(np.mean([row[tag]['outline_quality'] for row in rows]))
            result[tag] = dict(outline_macro_f1=float(np.mean([row[tag]['outline_f1'] for row in rows])),
                               outline_macro_quality=quality, joint_outline=float(coverage*quality))
        return result
