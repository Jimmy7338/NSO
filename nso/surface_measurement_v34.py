"""V34 evaluation-only ROI surface P/R/F1; no world, class or policy input."""
import hashlib
import math
import numpy as np
import open3d as o3d

CONTRACT = 'v34-area-surface-1'
SAMPLES = 32768
PREDICTION_SEED = 340918
REFERENCE_SEED = 340919
THRESHOLDS = (.02, .05, .10)
ROI_PADDING_M = .20
GROUND_BAND_M = .02
GROUND_NORMAL_COSINE = math.cos(math.radians(10.))
MINIMUM_TRIANGLE_AREA_M2 = 1e-12


def _facets(mesh):
    vertices = np.asarray(mesh.vertices, dtype=np.float64)
    triangles = np.asarray(mesh.triangles)
    if (vertices.ndim != 2 or vertices.shape[1] != 3 or not np.isfinite(vertices).all()
            or triangles.ndim != 2 or triangles.shape[1] != 3
            or not np.issubdtype(triangles.dtype, np.integer)
            or (len(triangles) and (triangles.min() < 0 or triangles.max() >= len(vertices)))):
        raise ValueError('finite vertices and valid integer triangle indices required')
    faces = vertices[triangles].copy()
    areas = _areas(faces)
    valid = faces[areas > MINIMUM_TRIANGLE_AREA_M2]
    # Sort geometric coordinates, not IDs; identical triangles count once.
    canonical = sorted({tuple(sorted(map(tuple, face))) for face in valid})
    output = np.asarray(canonical, dtype=np.float64).reshape(-1, 3, 3)
    return output, dict(raw_triangles=len(faces), raw_area_m2=float(areas.sum()),
        degenerate_triangles=int((areas <= MINIMUM_TRIANGLE_AREA_M2).sum()),
        exact_duplicate_triangles=len(valid)-len(output),
        canonical_triangles=len(output), canonical_area_m2=float(_areas(output).sum()))


def _areas(faces):
    return np.linalg.norm(np.cross(faces[:, 1]-faces[:, 0], faces[:, 2]-faces[:, 0]), axis=1)/2


def _mesh(faces):
    mesh = o3d.geometry.TriangleMesh()
    mesh.vertices = o3d.utility.Vector3dVector(np.asarray(faces, float).reshape(-1, 3))
    mesh.triangles = o3d.utility.Vector3iVector(np.arange(len(faces)*3).reshape(-1, 3))
    return mesh


def _hash(faces):
    return hashlib.sha256(np.ascontiguousarray(faces, dtype=np.float64).tobytes()).hexdigest()


def _sample(faces, seed):
    area = _areas(faces)
    if not len(faces): return np.empty((0, 3), dtype=np.float32)
    rng = np.random.Generator(np.random.PCG64(seed))
    chosen = faces[rng.choice(len(faces), SAMPLES, p=area/area.sum())]
    u, v = np.sqrt(rng.random(SAMPLES)), rng.random(SAMPLES)
    return ((1-u)[:, None]*chosen[:, 0]+(u*(1-v))[:, None]*chosen[:, 1]
            +(u*v)[:, None]*chosen[:, 2]).astype(np.float32)


def _scene(faces):
    scene = o3d.t.geometry.RaycastingScene(nthreads=1)
    scene.add_triangles(o3d.t.geometry.TriangleMesh.from_legacy(_mesh(faces)))
    return scene


def _distance(scene, points):
    result = scene.compute_distance(o3d.core.Tensor(points), nthreads=1).numpy().astype(float)
    if not np.isfinite(result).all(): raise ValueError('nonfinite triangle surface distance')
    return result


def _stats(distances):
    if not len(distances): return dict(mean_m=None, rmse_m=None, p95_m=None)
    return dict(mean_m=float(np.mean(distances)),
        rmse_m=float(np.sqrt(np.mean(distances**2))),
        p95_m=float(np.percentile(distances, 95)))


def extract_observed_asset_mesh(raw_mesh, public_bounds):
    """Exact padded public-ROI clipping; known-plane removal, no GT argument.

    public_bounds must be frozen from the UNION of both configuration envelopes,
    not the actual hypothesis. Caller provenance establishes that common input.
    Returned mesh is a new, uncapped geometry; raw_mesh is never mutated.
    """
    bounds = np.asarray(public_bounds, dtype=float)
    if bounds.shape != (2, 3) or not np.isfinite(bounds).all() or np.any(bounds[0] >= bounds[1]):
        raise ValueError('positive finite public 2x3 bounds required')
    roi = bounds+np.array([[-ROI_PADDING_M]*3, [ROI_PADDING_M]*3])
    faces, normalization = _facets(raw_mesh)
    fragments, touched, fully_outside = [], 0, 0
    for face in faces:
        outside = ((face < roi[0]) | (face > roi[1])).any()
        touched += int(outside)
        polygon = list(face)
        if outside:
            for axis in range(3):
                for side in (0, 1):
                    if not polygon: break
                    limit = roi[side, axis]
                    clipped = []
                    previous = polygon[-1]
                    previous_inside = previous[axis] >= limit if side == 0 else previous[axis] <= limit
                    for point in polygon:
                        inside = point[axis] >= limit if side == 0 else point[axis] <= limit
                        if inside != previous_inside:
                            fraction = (limit-previous[axis])/(point[axis]-previous[axis])
                            clipped.append(previous+fraction*(point-previous))
                        if inside: clipped.append(point)
                        previous, previous_inside = point, inside
                    polygon = clipped
        pieces = [np.asarray([polygon[0], polygon[j], polygon[j+1]])
                  for j in range(1, len(polygon)-1)]
        pieces = [p for p in pieces if _areas(p[None])[0] > MINIMUM_TRIANGLE_AREA_M2]
        if not pieces: fully_outside += 1
        fragments.extend(pieces)
    clipped = np.asarray(fragments, float).reshape(-1, 3, 3)
    normals = np.cross(clipped[:, 1]-clipped[:, 0], clipped[:, 2]-clipped[:, 0])
    areas = _areas(clipped)
    ground = ((np.abs(clipped[:, :, 2]) <= GROUND_BAND_M).all(axis=1)
              & (np.abs(normals[:, 2])/(2*areas) >= GROUND_NORMAL_COSINE))
    kept, kept_normalization = _facets(_mesh(clipped[~ground]))
    clipped_area = float(areas.sum())
    audit = dict(contract=CONTRACT, public_bounds=bounds.tolist(), padded_roi=roi.tolist(),
        padding_m=ROI_PADDING_M, ground_z_m=0., ground_band_m=GROUND_BAND_M,
        ground_maximum_normal_tilt_degrees=10., raw=normalization,
        clipped_triangles=len(clipped), clipped_area_m2=clipped_area,
        outside_area_m2=max(0., normalization['canonical_area_m2']-clipped_area),
        source_triangles_with_outside_portion=touched, fully_removed_by_roi_triangles=fully_outside,
        ground_removed_triangles=int(ground.sum()), ground_removed_area_m2=float(areas[ground].sum()),
        kept_triangles=len(kept), kept_area_m2=float(_areas(kept).sum()),
        kept_geometry_sha256=_hash(kept), output_normalization=kept_normalization,
        caps_added=False, truth_input=False, alignment_or_snapping=False,
        outside_roi_geometry_penalized_by_precision=False,
        precision_scope='public ROI facility surface; not uncropped full-instance or global-map precision')
    return _mesh(kept), audit


class SurfaceMeasurementV34:
    """Fixed-reference standard surface F1; GT is evaluation-only."""
    def __init__(self, full_reference, vertical_reference):
        full, full_audit = _facets(full_reference)
        vertical, vertical_audit = _facets(vertical_reference)
        if not len(full) or not len(vertical): raise ValueError('both complete references must be nonempty')
        normals = np.cross(vertical[:, 1]-vertical[:, 0], vertical[:, 2]-vertical[:, 0])
        if np.any(np.abs(normals[:, 2]) > 1e-7*np.linalg.norm(normals, axis=1)):
            raise ValueError('recall reference must contain vertical surfaces only')
        self._truth = _scene(full)
        self._reference_points = _sample(vertical, REFERENCE_SEED)
        if np.max(_distance(self._truth, self._reference_points)) > 1e-5:
            raise ValueError('vertical reference is not on full reference surface')
        self.reference = dict(full_geometry_sha256=_hash(full), vertical_geometry_sha256=_hash(vertical),
            full_area_m2=full_audit['canonical_area_m2'], vertical_area_m2=vertical_audit['canonical_area_m2'],
            sample_count=SAMPLES, reference_seed=REFERENCE_SEED, prediction_seed=PREDICTION_SEED)

    def evaluate(self, predicted_mesh, C_map, *, returned=False, collisions=0, failed=False,
                 paid_actions=None, budget=None):
        if (isinstance(C_map, bool) or not np.isfinite(C_map) or not 0 <= C_map <= 1
                or type(returned) is not bool or type(failed) is not bool
                or type(collisions) is not int or collisions < 0):
            raise ValueError('finite coverage and actual qualification fields required')
        if (paid_actions is None) != (budget is None): raise ValueError('paid actions and budget must be paired')
        if budget is not None and any(type(v) is not int or v < 0 for v in (paid_actions, budget)):
            raise ValueError('nonnegative integer action accounting required')
        faces, normalization = _facets(predicted_mesh)
        if len(faces):
            accuracy = _distance(self._truth, _sample(faces, PREDICTION_SEED))
            completeness = _distance(_scene(faces), self._reference_points)
        else:
            accuracy = completeness = np.empty(0)
        compliant = None if budget is None else paid_actions <= budget
        result = dict(contract=CONTRACT, reference=dict(self.reference), prediction=normalization,
            prediction_geometry_sha256=_hash(faces), missing=not len(faces),
            precision_error=_stats(accuracy), recall_error=_stats(completeness),
            C_map=float(C_map), returned=returned, collisions=collisions, failed=failed,
            paid_actions=paid_actions, budget=budget, budget_compliant=compliant,
            eligible=bool(C_map >= .8 and returned and not collisions and not failed and compliant is True),
            qualification_incomplete=['action_budget'] if budget is None else [],
            main_threshold_m=.05, inference_performed=False, alignment_performed=False,
            prediction_scope='submitted public-ROI facility surface; outside-ROI errors excluded',
            recall_scope='fixed complete external vertical reference; not observed subset')
        for threshold in THRESHOLDS:
            p = float(np.mean(accuracy <= threshold)) if len(accuracy) else 0.
            r = float(np.mean(completeness <= threshold)) if len(completeness) else 0.
            f1 = 2*p*r/(p+r) if p+r else 0.
            result[f'{round(threshold*100):02d}cm'] = dict(precision=p, recall=r, f1=f1,
                joint=float(C_map*f1))
        result['main_joint'] = result['05cm']['joint']
        return result
