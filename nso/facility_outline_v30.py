"""Evaluation-only full observed instances; no prediction crop or alignment.

This companion retains V23's projection quality formula, but uses fixed seed
association instead of clipping a global mesh into truth windows. It does not
replace the original V23 headline metric and must never enter a planner.
"""
from collections import Counter
import hashlib
import json
import numpy as np
import open3d as o3d

from utils.facility_outline_v23 import (OutlineEvaluatorV23, PROJECTIONS,
    mesh_arrays, projected_envelope, compare_projection)


CONTRACT = 'facility-full-instance-outline-v30-1'


def _geometry_receipt(mesh):
    vertices, triangles = mesh_arrays(mesh)
    return dict(vertices=len(vertices), triangles=len(triangles),
        sha256=hashlib.sha256(vertices.tobytes()+triangles.tobytes()).hexdigest())


def _outside_window(mesh, bounds):
    """Diagnostics only; never return a cropped/filtered prediction."""
    vertices, triangles = mesh_arrays(mesh)
    if not len(triangles):
        return dict(used_vertices=0,outside_vertices=0,triangles_any_vertex_outside=0,
            triangles_all_vertices_outside=0,outside_triangle_area_lower_m2=0.,
            outside_triangle_area_upper_m2=0.,maximum_coordinate_excess_m=0.)
    faces=vertices[triangles];used=vertices[np.unique(triangles)]
    outside=((used < bounds[0]-1e-9)|(used > bounds[1]+1e-9)).any(axis=1)
    per_vertex=((faces < bounds[0]-1e-9)|(faces > bounds[1]+1e-9)).any(axis=2)
    any_out=per_vertex.any(axis=1)
    # All vertices beyond the same separating AABB halfspace certify that the
    # complete triangle is outside. "All vertices outside" alone is insufficient.
    all_out=(((faces < bounds[0]-1e-9).all(axis=1))|
             ((faces > bounds[1]+1e-9).all(axis=1))).any(axis=1)
    area=np.linalg.norm(np.cross(faces[:,1]-faces[:,0],faces[:,2]-faces[:,0]),axis=1)/2
    excess=np.maximum(np.maximum(bounds[0]-used,used-bounds[1]),0.)
    return dict(used_vertices=len(used),outside_vertices=int(outside.sum()),
        triangles_any_vertex_outside=int(any_out.sum()),
        triangles_all_vertices_outside=int(per_vertex.all(axis=1).sum()),
        outside_triangle_area_lower_m2=float(area[all_out].sum()),
        outside_triangle_area_upper_m2=float(area[any_out].sum()),
        maximum_coordinate_excess_m=float(excess.max()))


class FullInstanceOutlineEvaluatorV30:
    """Reference-only construction, then score complete predicted instances.

    assets: V23-compatible {id, mesh, bounds} dictionaries. The disjoint bounds
    associate actual first observed seeds; they NEVER clip predicted meshes.
    Missing/ambiguous/duplicate associations remain zero in the fixed macro.
    Caller must finish and freeze the observed snapshot before constructing us.
    """
    def __init__(self, assets, boundary_spacing_m=.01, thresholds=(.02,.05,.10),
                 completion_tolerance_m=.05, completion_minimum_iou=.9):
        self.reference=OutlineEvaluatorV23(assets,boundary_spacing_m=boundary_spacing_m,
            thresholds=thresholds,completion_tolerance_m=completion_tolerance_m,
            completion_minimum_iou=completion_minimum_iou)
        self.assets=self.reference.assets
        parameters=dict(contract=CONTRACT,reference_contract=self.reference.contract_parameters,
            prediction_crop=False,prediction_alignment=False,
            assignment='unique first observed seed in fixed reference-only task window; duplicates missing',
            quality='fixed_asset_macro(mean_projections(min(boundary_length_F1,projection_IoU)))',
            role='companion; original V23 fixed-window metric remains separately reported')
        self.contract_parameters=parameters
        self.reference_signature=hashlib.sha256(json.dumps(parameters,sort_keys=True).encode()).hexdigest()

    @classmethod
    def from_world(cls, world, **kwargs):
        # This is explicitly evaluation-side, called only after output freeze.
        return cls([dict(id=int(item['id']),mesh=world.instance_mesh(item['id']),
            bounds=item['evaluation_bounds']) for item in world.objects],**kwargs)

    def _row(self, mesh, reference, slot, reason):
        vertices,triangles=mesh_arrays(mesh)
        projections={name:compare_projection(projected_envelope(mesh,axes),
            reference['projections'][name],self.reference.spacing,self.reference.thresholds)
            for name,axes in PROJECTIONS.items()}
        shape_bounds=reference['shape_bounds']
        dimensions=dict(absolute_size_error_xyz_m=None,center_error_inf_m=None,
            reference_size_xyz_m=np.diff(shape_bounds,axis=0)[0].tolist(),predicted_size_xyz_m=None)
        if len(triangles):
            used=vertices[np.unique(triangles)];box=np.array([used.min(axis=0),used.max(axis=0)])
            size=box[1]-box[0]
            dimensions.update(predicted_size_xyz_m=size.tolist(),
                absolute_size_error_xyz_m=np.abs(size-(shape_bounds[1]-shape_bounds[0])).tolist(),
                center_error_inf_m=float(np.max(np.abs(box.mean(axis=0)-shape_bounds.mean(axis=0)))))
        tol=self.reference.completion_tolerance
        completed=bool(len(triangles) and all(not p['missing_projection']
            and p['hausdorff_upper_m']<=tol and p['iou']>=self.reference.completion_iou
            for p in projections.values()) and max(dimensions['absolute_size_error_xyz_m'])<=tol
            and dimensions['center_error_inf_m']<=tol)
        row=dict(id=reference['id'],assigned_observed_slot=slot,association_reason=reason,
            missing=not len(triangles),projections=projections,dimensions=dimensions,completed=completed,
            full_predicted_mesh=_geometry_receipt(mesh),outside_window=_outside_window(mesh,reference['bounds']),
            measurement_provenance='entire submitted instance mesh; no prediction crop or alignment',
            evaluator_does_not_certify_observed_vs_inferred=True)
        for threshold in self.reference.thresholds:
            tag=f'{round(threshold*100):02d}cm'
            row[tag]=dict(outline_f1=float(np.mean([p[tag]['f1'] for p in projections.values()])),
                outline_quality=float(np.mean([min(p[tag]['f1'],p['iou']) for p in projections.values()])))
        return row

    def evaluate(self, observed_meshes, seeds, observed_track_count, coverage, *,
                 returned=False, collisions=0, failed=False, paid_actions=None, budget=None):
        meshes=tuple(observed_meshes);seeds=tuple(seeds)
        if len(meshes)!=len(seeds):
            raise ValueError('one seed receipt per submitted observed mesh required')
        if (type(observed_track_count) is not int or observed_track_count<0
                or isinstance(coverage,bool) or not np.isfinite(coverage) or not 0<=coverage<=1
                or type(collisions) is not int or collisions<0
                or type(returned) is not bool or type(failed) is not bool):
            raise ValueError('valid observed count, coverage and actual qualification required')
        if (paid_actions is None)!=(budget is None):
            raise ValueError('paid action and budget must be provided together')
        if budget is not None and any(type(v) is not int or v<0 for v in (paid_actions,budget)):
            raise ValueError('nonnegative integer action budget required')
        before=[_geometry_receipt(mesh) for mesh in meshes]
        association=[]
        for slot,seed in enumerate(seeds):
            matches=[]
            if seed is not None:
                point=np.asarray(seed['observed_seed_xyz'],float)
                if point.shape!=(3,) or not np.isfinite(point).all():
                    raise ValueError('finite observed first seed point required')
                matches=[item['id'] for item in self.assets if np.all(point>=item['bounds'][0])
                    and np.all(point<=item['bounds'][1])]
            association.append(dict(observed_slot=slot,matches=matches,
                reference_id=matches[0] if len(matches)==1 else None,
                status='unique' if len(matches)==1 else 'ambiguous' if matches else 'unassigned'))
        counts=Counter(row['reference_id'] for row in association if row['reference_id'] is not None)
        rows=[];credited=set()
        for reference in self.assets:
            matched=[r['observed_slot'] for r in association if r['reference_id']==reference['id']]
            if len(matched)==1:
                slot=matched[0];mesh=meshes[slot];credited.add(slot);reason='unique_fixed_seed'
            else:
                slot=None;mesh=o3d.geometry.TriangleMesh()
                reason='missing_seed_association' if not matched else 'duplicate_seed_association'
            rows.append(self._row(mesh,reference,slot,reason))
        support=[]
        for slot,mesh in enumerate(meshes):
            support.append(dict(observed_slot=slot,submitted_mesh=before[slot],
                credited_to_task=slot in credited,association=association[slot],
                reference_window_diagnostics=[dict(id=ref['id'],**_outside_window(mesh,ref['bounds']))
                    for ref in self.assets]))
        association_ok=bool(len(meshes)==len(self.assets)==observed_track_count
            and all(row['status']=='unique' for row in association)
            and all(counts[ref['id']]==1 for ref in self.assets))
        compliant=None if budget is None else paid_actions<=budget
        result=dict(contract=CONTRACT,reference_signature=self.reference_signature,
            instances=rows,submitted_instance_support=support,fixed_seed_association=association,
            unique_complete_seed_association=association_ok,
            unassigned_or_duplicate_observed_slots=sorted(set(range(len(meshes)))-credited),
            mission_asset_count=len(rows),missing_asset_count=sum(r['missing'] for r in rows),
            coverage_2d=float(coverage),returned=returned,collisions=collisions,failed=failed,
            primitive_budget_compliant=compliant,budget_verified=budget is not None,
            eligible=bool(coverage>=.8 and returned and not collisions and not failed and compliant is True),
            qualification_incomplete=[] if budget is not None else ['primitive_action_budget'],
            completion_fraction=float(np.mean([row['completed'] for row in rows])),
            prediction_crop=False,prediction_alignment=False,category_or_visit_count_used=False,
            role='fixed companion to original V23 window scores; not a post-result replacement',
            scope='three exterior projections of full attributed instance; not unique 3D or calibrated surface accuracy')
        for threshold in self.reference.thresholds:
            tag=f'{round(threshold*100):02d}cm'
            quality=float(np.mean([row[tag]['outline_quality'] for row in rows]))
            result[tag]=dict(outline_macro_f1=float(np.mean([row[tag]['outline_f1'] for row in rows])),
                outline_macro_quality=quality,joint_outline=float(coverage*quality))
        if before!=[_geometry_receipt(mesh) for mesh in meshes]:
            raise RuntimeError('evaluation mutated prediction geometry')
        return result
