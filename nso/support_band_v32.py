"""Offline exterior-support agreement; no inferred surface or occupancy repair.

The tolerance band is a scoring region around each independently constructed
boundary support. It is never returned as a mesh, occupied volume, or completion.
"""
import hashlib
import json
import numpy as np
import shapely

from utils.facility_outline_v23 import PROJECTIONS, mesh_arrays
from nso.facility_outline_v30 import _geometry_receipt, _outside_window
from nso.supported_outline_v31 import (
    FullSupportedOutlineEvaluatorV31, supported_projection_v31,
    compare_supported_projection_v31)

CONTRACT = 'exterior-support-band-v32-r0'
QUAD_SEGS = 16


def support_band_v32(support, tolerance):
    """Reference-free distance band. Overlap does not certify connectivity."""
    if isinstance(tolerance, bool) or not np.isfinite(tolerance) or tolerance <= 0:
        raise ValueError('positive finite support-band tolerance required')
    return shapely.buffer(support, float(tolerance), quad_segs=QUAD_SEGS,
                          cap_style='round', join_style='round')


def component_count(geometry):
    if geometry.is_empty:
        return 0
    if geometry.geom_type == 'Polygon':
        return 1
    return sum(component_count(g) for g in geometry.geoms)


def compare_support_band_v32(prediction, reference, spacing=.01,
                             thresholds=(.02, .05, .10)):
    result = compare_supported_projection_v31(prediction, reference, spacing, thresholds)
    support = prediction['support']
    before = support.wkb
    result.update(legacy_region_iou=result['iou'],
                  connectivity_certified=False, observed_surface_completion_certified=False,
                  band_is_occupied_region=False)
    for tolerance in thresholds:
        tag = f'{round(tolerance*100):02d}cm'
        predicted_band = support_band_v32(support, tolerance)
        reference_band = support_band_v32(reference.boundary, tolerance)
        intersection = predicted_band.intersection(reference_band).area
        union = predicted_band.union(reference_band).area
        band_iou = float(intersection / union)
        row = result[tag]
        row.update(band_iou=band_iou, quality=min(row['f1'], band_iou),
            band_intersection_m2=float(intersection), band_union_m2=float(union),
            predicted_band_area_m2=float(predicted_band.area),
            reference_band_area_m2=float(reference_band.area),
            predicted_band_components=component_count(predicted_band),
            reference_band_components=component_count(reference_band),
            missed_reference_support_m=float(result['reference_boundary_length_m']*(1-row['recall'])),
            excess_predicted_support_m=float(result['predicted_boundary_length_m']*(1-row['precision'])))
    if support.wkb != before:
        raise AssertionError('distance-band evaluation mutated support')
    return result


class FullSupportBandEvaluatorV32(FullSupportedOutlineEvaluatorV31):
    """A separate support diagnostic, not the frozen region-quality score."""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.contract_parameters.update(contract=CONTRACT,
            quality='output_precision * fixed_asset_macro(mean_projections(min(F1_tau,band_IoU_tau)))',
            band_radius='each frozen distance tolerance,2/5/10cm', band_quad_segs=QUAD_SEGS,
            band_cap='round', band_join='round', geometry_repair=False,
            band_is_occupied_region=False, connectivity_certified=False,
            complete_3d_accuracy=False, completion_certification='not defined for this diagnostic',
            role='new exterior-support task diagnostic; old region Q/J retained')
        self.reference_signature = hashlib.sha256(
            json.dumps(self.contract_parameters, sort_keys=True).encode()).hexdigest()

    def _row(self, mesh, reference, slot, reason):
        vertices, triangles = mesh_arrays(mesh)
        projections = {name: compare_support_band_v32(
            supported_projection_v31(mesh, axes), reference['projections'][name],
            self.reference.spacing, self.reference.thresholds)
            for name, axes in PROJECTIONS.items()}
        bounds = reference['shape_bounds']
        dimensions = dict(absolute_size_error_xyz_m=None, center_error_inf_m=None,
            reference_size_xyz_m=(bounds[1]-bounds[0]).tolist(), predicted_size_xyz_m=None)
        if len(triangles):
            used = vertices[np.unique(triangles)]
            box = np.array([used.min(axis=0), used.max(axis=0)])
            size = box[1]-box[0]
            dimensions.update(predicted_size_xyz_m=size.tolist(),
                absolute_size_error_xyz_m=np.abs(size-(bounds[1]-bounds[0])).tolist(),
                center_error_inf_m=float(np.max(np.abs(box.mean(axis=0)-bounds.mean(axis=0)))))
        row = dict(id=reference['id'], assigned_observed_slot=slot, association_reason=reason,
            missing=not len(triangles), projections=projections, dimensions=dimensions,
            completed=False, completion_status='not_certified_by_support_band',
            connectivity_certified=False, full_predicted_mesh=_geometry_receipt(mesh),
            outside_window=_outside_window(mesh, reference['bounds']),
            measurement_provenance='unchanged complete submitted mesh; bands are evaluation regions only')
        for tolerance in self.reference.thresholds:
            tag = f'{round(tolerance*100):02d}cm'
            row[tag] = dict(outline_f1=float(np.mean([p[tag]['f1'] for p in projections.values()])),
                outline_quality=float(np.mean([p[tag]['quality'] for p in projections.values()])),
                v31_region_outline_quality=float(np.mean([
                    min(p[tag]['f1'], p['legacy_region_iou']) for p in projections.values()])))
        return row

    def evaluate(self, *args, **kwargs):
        result = super().evaluate(*args, **kwargs)
        result.update(contract=CONTRACT, completion_fraction=None,
            completion_status='not_certified_by_support_band', connectivity_certified=False,
            observed_surface_completion_certified=False, full_3d_accuracy_certified=False,
            semantic_efficacy_proven=False, candidate_ready_as_sole_training_target=False,
            role='posthoc exterior-support diagnostic; not a replacement for original region Q/J',
            limitations=['bands do not establish actual occupancy, topology, or full3D surface accuracy',
                'hard F1 distance thresholds remain; global position-noise continuity is not claimed',
                'V31 support extraction may change at projection topology transitions',
                'sub-tolerance real gaps may have high support agreement',
                'output precision depends on submitted instance fragmentation'])
        for tolerance in self.reference.thresholds:
            tag = f'{round(tolerance*100):02d}cm'
            base = float(np.mean([r[tag]['v31_region_outline_quality'] for r in result['instances']]))
            result[tag].update(v31_region_outline_quality=base*result['output_instance_precision'])
        return result
