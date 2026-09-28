"""Draft, evaluation-only removal of faces rejected by the frozen v40 gate.

This module does not import an evaluator, mapper, controller, scene, reference,
semantic label or ROI. It does not authorize a new measurement. An external
predeclaration must govern whether and how the returned prediction is scored.
The original scientific artifacts and failed-attempt state remain unchanged.
"""
import hashlib
import json

import numpy as np


SCHEMA = 'article.prediction_numeric_face_adapter.v1'
# Exactly equivalent to the frozen evaluator's twice_area <= 1e-12 rejection.
MAX_REMOVED_FACE_AREA_M2 = 5e-13
MAX_PREDICTION_FACES = 1000000
MAX_PREDICTION_VERTICES = 2000000


def array_sha256(value):
    """SHA of dtype, shape and C-order values; independent of NPZ timestamps."""
    value = np.asarray(value)
    digest = hashlib.sha256(json.dumps(dict(dtype=value.dtype.str,shape=list(value.shape)),
        sort_keys=True,separators=(',',':')).encode()+b'\n')
    digest.update(np.ascontiguousarray(value).tobytes())
    return digest.hexdigest()


def prepare_prediction_mesh_v1(vertices, triangles):
    """Return original vertices, ordered retained faces, and an auditable record.

    Only finite, structurally valid prediction arrays are accepted. The sole
    selection rule is area <= 5e-13 m^2. Vertex values/indices and surviving
    face order are unchanged. With no removed faces, the input arrays themselves
    are returned. No geometry is merged, repaired, cropped or resampled.
    """
    vertices = np.asarray(vertices); triangles = np.asarray(triangles)
    if (vertices.ndim != 2 or vertices.shape[1] != 3 or vertices.dtype.kind not in 'fiu'
            or len(vertices) > MAX_PREDICTION_VERTICES or not np.isfinite(vertices).all()):
        raise ValueError('bounded finite numeric vertices[N,3] required')
    if (triangles.ndim != 2 or triangles.shape[1] != 3 or triangles.dtype.kind not in 'iu'
            or len(triangles) > MAX_PREDICTION_FACES):
        raise ValueError('bounded integer triangles[M,3] required')
    if triangles.size and (triangles.min() < 0 or triangles.max() >= len(vertices)):
        raise ValueError('triangle index outside original vertices')
    xyz = np.asarray(vertices,dtype=np.float64)[triangles]
    with np.errstate(over='ignore',invalid='ignore'):
        twice = np.linalg.norm(np.cross(xyz[:,1]-xyz[:,0],xyz[:,2]-xyz[:,0]),axis=1)
    if not np.isfinite(twice).all():
        raise ValueError('nonfinite triangle area is rejected, never filtered')
    area = twice * .5
    removed = np.flatnonzero(area <= MAX_REMOVED_FACE_AREA_M2)
    keep = np.flatnonzero(area > MAX_REMOVED_FACE_AREA_M2)
    retained = triangles if not len(removed) else triangles[keep]
    removed_xyz = xyz[removed]
    edge_lengths = np.linalg.norm(removed_xyz-np.roll(removed_xyz,1,axis=1),axis=2)
    report = dict(schema=SCHEMA,status='geometry_prepared_no_quality_evaluation',
        fixed_maximum_removed_face_area_m2=MAX_REMOVED_FACE_AREA_M2,
        equivalent_frozen_v40_rule='reject twice_area <= 1e-12',
        original_vertices=len(vertices),original_faces=len(triangles),retained_faces=len(retained),
        removed_faces=len(removed),removed_original_face_indices=removed.tolist(),
        removed_total_area_m2=float(area[removed].sum()),
        removed_maximum_face_area_m2=float(area[removed].max()) if len(removed) else 0.,
        removed_maximum_edge_m=float(edge_lengths.max()) if edge_lengths.size else 0.,
        original_total_area_m2=float(area.sum()),retained_total_area_m2=float(area[keep].sum()),
        original_vertices_array_sha256=array_sha256(vertices),
        original_triangles_array_sha256=array_sha256(triangles),
        adapted_vertices_array_sha256=array_sha256(vertices),
        adapted_triangles_array_sha256=array_sha256(retained),
        retained_original_face_indices_sha256=array_sha256(keep),
        original_vertices_values_order_and_dtype_unchanged=True,
        retained_faces_values_order_and_dtype_unchanged=True,
        prediction_arrays_exactly_unchanged=not len(removed),
        gt_reference_read=False,semantic_labels_read=False,roi_applied=False,
        new_surface_evaluations=0,new_tsdf_integrations=0,new_worlds=0,
        quality_reuse_eligibility='requires independent equality of prediction, reference and all evaluator settings' if not len(removed)
            else 'new derived measurement required; cannot reuse original score',
        sampling_warning='Removing a face may shift later random samples in the frozen evaluator; no metric-equivalence claim is made.')
    return vertices,retained,report
