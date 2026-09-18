"""Evaluation-only exterior contours with exact degenerate-line support.

No reference or class is accepted by the projection function. Polygonization
can enclose a submitted closed contour; it cannot add a missing connection or
prove that any horizontal 3D face was measured.
"""
import hashlib
import json
import numpy as np
import shapely
from shapely.geometry import GeometryCollection, LineString, Polygon

from utils.facility_outline_v23 import mesh_arrays, polygon_parts, PROJECTIONS
from nso.facility_outline_v30 import (FullInstanceOutlineEvaluatorV30,
    _geometry_receipt, _outside_window)

CONTRACT = 'supported-exterior-outline-v31-1'
AREA2_EPS = 1e-14


def _lines(geometry):
    if geometry.is_empty:
        return []
    if geometry.geom_type in ('LineString', 'LinearRing'):
        return [geometry] if geometry.length > 0 else []
    if geometry.geom_type in ('MultiLineString', 'GeometryCollection'):
        return [line for part in geometry.geoms for line in _lines(part)]
    raise ValueError('line support expected')


def _shell_union(geometry):
    shells = [Polygon(p.exterior) for p in polygon_parts(geometry)]
    return shapely.union_all(shells) if shells else GeometryCollection()


def validate_surface_v31(mesh):
    vertices,triangles=mesh_arrays(mesh)
    if len(triangles):
        faces=vertices[triangles]
        area3=np.linalg.norm(np.cross(faces[:,1]-faces[:,0],faces[:,2]-faces[:,0]),axis=1)
        if (area3<=AREA2_EPS).any():
            raise ValueError('zero-area 3D triangle is not measured surface evidence')
    return vertices,triangles


def supported_projection_v31(mesh, axes):
    """Construct predicted area/line support without reference-dependent repair."""
    if tuple(axes) not in tuple(PROJECTIONS.values()):
        raise ValueError('only the frozen three orthogonal projections are supported')
    vertices, triangles = validate_surface_v31(mesh)
    empty = GeometryCollection()
    if not len(triangles):
        return dict(area=empty, support=empty, residual_lines=empty,
            receipt=dict(input_triangles=0, degenerate_projection_triangles=0,
                polygonized_cells=0, area_m2=0., residual_line_length_m=0.,
                support_length_m=0., nonempty_area=False, added_connections=0))
    faces = vertices[triangles]
    points = faces[:, :, axes]
    a, b = points[:, 1]-points[:, 0], points[:, 2]-points[:, 0]
    area2 = np.abs(a[:, 0]*b[:, 1]-a[:, 1]*b[:, 0])
    flat = area2 <= AREA2_EPS
    raw_area = shapely.union_all(shapely.polygons(points[~flat])) if (~flat).any() else empty
    segments = []
    for triangle in points[flat]:
        pairs = ((0,1), (1,2), (2,0))
        i,j = max(pairs, key=lambda pair: float(np.linalg.norm(triangle[pair[0]]-triangle[pair[1]])))
        if np.linalg.norm(triangle[i]-triangle[j]) <= 0:
            raise ValueError('positive 3D surface projected to a zero-length numerical support')
        segments.append(LineString([triangle[i], triangle[j]]))
    degenerate_lines = shapely.union_all(segments) if segments else empty
    # Internal triangulation edges are absent. There is no snapping, buffering,
    # morphological closing, convex hull, or inserted bridging segment.
    inputs = ([raw_area.boundary] if not raw_area.is_empty else []) + segments
    noded = shapely.union_all(inputs) if inputs else empty
    cells = shapely.polygonize(shapely.get_parts(noded)) if not noded.is_empty else empty
    area = _shell_union(shapely.union_all([raw_area, cells]))
    residual = degenerate_lines.difference(area) if not degenerate_lines.is_empty else empty
    support = shapely.union_all(([area.boundary] if not area.is_empty else []) + _lines(residual))
    if not area.is_valid or not support.is_valid:
        raise ValueError('invalid exact projection union')
    return dict(area=area, support=support, residual_lines=residual,
        receipt=dict(input_triangles=len(triangles), degenerate_projection_triangles=int(flat.sum()),
            polygonized_cells=len(polygon_parts(cells)), area_m2=float(area.area),
            residual_line_length_m=float(residual.length), support_length_m=float(support.length),
            nonempty_area=not area.is_empty, added_connections=0))


def _line_samples(geometry, spacing):
    samples=[];weights=[];ends=[];maximum_step=0.
    for line in _lines(geometry):
        coordinates=np.asarray(line.coords)
        for start, stop in zip(coordinates[:-1], coordinates[1:]):
            delta=stop-start;length=float(np.linalg.norm(delta))
            if length<=0: continue
            count=max(1,int(np.ceil(length/spacing)));step=length/count
            samples.append(start+((np.arange(count)+.5)/count)[:,None]*delta)
            weights.append(np.full(count,step));ends.extend((start,stop))
            maximum_step=max(maximum_step,step)
    return (np.concatenate(samples) if samples else np.empty((0,2)),
        np.concatenate(weights) if weights else np.empty(0),
        np.asarray(ends).reshape(-1,2),maximum_step)


def _directed_lines(source,target,spacing):
    points,weights,endpoints,step=_line_samples(source,spacing)
    distances=shapely.distance(shapely.points(points),target)
    end_distances=shapely.distance(shapely.points(endpoints),target)
    maximum=float(max(np.max(distances),np.max(end_distances)))
    return dict(distances=distances,weights=weights,mean=float(np.average(distances,weights=weights)),
        lower=maximum,upper=maximum+step/2,samples=len(points),length=float(weights.sum()))


def compare_supported_projection_v31(prediction, reference, spacing=.01, thresholds=(.02,.05,.10)):
    if reference.is_empty or reference.area<=0:
        raise ValueError('nonempty reference exterior area required')
    area,support=prediction['area'],prediction['support']
    result=dict(predicted_area_m2=float(area.area),reference_area_m2=float(reference.area),
        predicted_components=len(polygon_parts(area)),reference_components=len(polygon_parts(reference)),
        iou=float(area.intersection(reference).area/area.union(reference).area),
        missing_projection=area.is_empty,missing_support=support.is_empty,
        representation=prediction['receipt'],prediction_crop=False)
    if support.is_empty:
        result.update(boundary_symmetric_mean_m=None,hausdorff_lower_m=None,hausdorff_upper_m=None,
            predicted_boundary_samples=0,reference_boundary_samples=0,
            predicted_boundary_length_m=0.,reference_boundary_length_m=float(reference.boundary.length))
        for t in thresholds: result[f'{round(t*100):02d}cm']=dict(precision=0.,recall=0.,f1=0.)
        return result
    forward=_directed_lines(support,reference.boundary,spacing)
    reverse=_directed_lines(reference.boundary,support,spacing)
    equal=support.equals(reference.boundary)
    result.update(boundary_symmetric_mean_m=(forward['mean']+reverse['mean'])/2,
        hausdorff_lower_m=0. if equal else max(forward['lower'],reverse['lower']),
        hausdorff_upper_m=0. if equal else max(forward['upper'],reverse['upper']),
        predicted_boundary_samples=forward['samples'],reference_boundary_samples=reverse['samples'],
        predicted_boundary_length_m=forward['length'],reference_boundary_length_m=reverse['length'])
    for t in thresholds:
        p=float(np.average(forward['distances']<=t+1e-12,weights=forward['weights']))
        r=float(np.average(reverse['distances']<=t+1e-12,weights=reverse['weights']))
        result[f'{round(t*100):02d}cm']=dict(precision=p,recall=r,f1=2*p*r/(p+r) if p+r else 0.)
    return result


class FullSupportedOutlineEvaluatorV31(FullInstanceOutlineEvaluatorV30):
    """New candidate score; old V23/V30 outputs are never rewritten."""
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs)
        self.contract_parameters.update(contract=CONTRACT,projection_area2_epsilon=AREA2_EPS,
            missing_connection_inference=False,reference_dependent_prediction_processing=False,
            line_support='exact residual degenerate segments plus filled exterior boundary',
            output_precision='unique attributed nonempty output slots / all nonempty output slots; empty=0',
            empty_seed_ignored=True,role='development candidate, not a replacement for frozen historical scores')
        self.reference_signature=hashlib.sha256(json.dumps(self.contract_parameters,sort_keys=True).encode()).hexdigest()

    def _row(self,mesh,reference,slot,reason):
        vertices,triangles=mesh_arrays(mesh)
        projections={name:compare_supported_projection_v31(supported_projection_v31(mesh,axes),
            reference['projections'][name],self.reference.spacing,self.reference.thresholds)
            for name,axes in PROJECTIONS.items()}
        shape_bounds=reference['shape_bounds']
        dimensions=dict(absolute_size_error_xyz_m=None,center_error_inf_m=None,
            reference_size_xyz_m=(shape_bounds[1]-shape_bounds[0]).tolist(),predicted_size_xyz_m=None)
        if len(triangles):
            used=vertices[np.unique(triangles)];box=np.array([used.min(axis=0),used.max(axis=0)])
            size=box[1]-box[0]
            dimensions.update(predicted_size_xyz_m=size.tolist(),
                absolute_size_error_xyz_m=np.abs(size-(shape_bounds[1]-shape_bounds[0])).tolist(),
                center_error_inf_m=float(np.max(np.abs(box.mean(axis=0)-shape_bounds.mean(axis=0)))))
        tol=self.reference.completion_tolerance
        completed=bool(len(triangles) and all(not p['missing_projection'] and p['hausdorff_upper_m']<=tol
            and p['iou']>=self.reference.completion_iou for p in projections.values())
            and max(dimensions['absolute_size_error_xyz_m'])<=tol and dimensions['center_error_inf_m']<=tol)
        row=dict(id=reference['id'],assigned_observed_slot=slot,association_reason=reason,
            missing=not len(triangles),projections=projections,dimensions=dimensions,completed=completed,
            full_predicted_mesh=_geometry_receipt(mesh),outside_window=_outside_window(mesh,reference['bounds']),
            measurement_provenance='complete submitted surfaces; closed 2D contour is not an observed 3D cap')
        for t in self.reference.thresholds:
            tag=f'{round(t*100):02d}cm'
            row[tag]=dict(outline_f1=float(np.mean([p[tag]['f1'] for p in projections.values()])),
                outline_quality=float(np.mean([min(p[tag]['f1'],p['iou']) for p in projections.values()])))
        return row

    def evaluate(self,observed_meshes,seeds,observed_track_count,coverage,**kwargs):
        meshes,seeds=tuple(observed_meshes),tuple(seeds)
        if len(meshes)!=len(seeds): raise ValueError('one seed per submitted output required')
        # Validate all outputs before attribution: unassigned or duplicated
        # slots cannot hide malformed surface geometry from the contract.
        validated=[validate_surface_v31(mesh) for mesh in meshes]
        nonempty=[i for i,(_,triangles) in enumerate(validated) if len(triangles)]
        # An empty extra output cannot invalidate a real match by duplicating
        # its seed. Nonempty duplicates remain missing, with no best-of choice.
        effective=[seed if i in nonempty else None for i,seed in enumerate(seeds)]
        result=super().evaluate(meshes,effective,observed_track_count,coverage,**kwargs)
        credited={row['assigned_observed_slot'] for row in result['instances'] if not row['missing']}
        unmatched=sorted(set(nonempty)-credited)
        precision=len(credited)/len(nonempty) if nonempty else 0.
        result.update(contract=CONTRACT,historical_task_eligible=result['eligible'],
            candidate_output_gate_passed=not unmatched,eligible=bool(result['eligible'] and not unmatched),
            output_instance_precision=precision,nonempty_output_slots=nonempty,
            uniquely_matched_nonempty_slots=sorted(credited),unmatched_nonempty_slots=unmatched,
            empty_seed_slots_ignored=[i for i in range(len(meshes)) if i not in nonempty and seeds[i] is not None],
            unique_complete_seed_association=bool(len(credited)==len(self.assets) and not unmatched),
            original_observed_track_count=observed_track_count,
            role='new preregistered development candidate; historical scores retained separately',
            limitations=['not full 3D surface accuracy','interior projection-equivalent errors are unidentifiable',
                'output precision depends on submitted instance fragmentation; not natural segmentation accuracy'])
        for t in self.reference.thresholds:
            tag=f'{round(t*100):02d}cm';row=result[tag]
            row.update(base_outline_macro_quality=row['outline_macro_quality'],
                base_joint_outline=row['joint_outline'])
            row['outline_macro_quality']*=precision;row['joint_outline']*=precision
        return result
