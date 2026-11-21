"""Measured-only partial oriented envelope fit, no source mesh or hidden bounds.

The fit preserves a prior point estimate on an unobserved extent, and records
that endpoint as unresolved. It does not call a point estimate an observed size.
"""
import numpy as np

NORMAL_COMPONENT_MIN = .7
MIN_FACE_POINTS = 8
MIN_ORIENTATION_CONCENTRATION = .25


def fit_envelope(cloud, normals, prior_dimensions):
    cloud = np.asarray(cloud, float)
    normals = np.asarray(normals, float)
    if cloud.ndim != 2 or cloud.shape[1] != 3 or normals.shape != cloud.shape:
        raise ValueError('paired measured points/normals required')
    if not len(cloud) or not np.isfinite(cloud).all():
        raise ValueError('finite measured points required')
    good = np.isfinite(normals).all(1)
    horizontal = np.linalg.norm(normals[:, :2], axis=1)
    usable = good & (horizontal > NORMAL_COMPONENT_MIN) & (np.abs(normals[:, 2]) < .35)
    yaw = 0.; concentration = 0.
    if usable.sum() >= MIN_FACE_POINTS:
        angles = np.arctan2(normals[usable, 1], normals[usable, 0])
        moment = np.mean(np.exp(4j * angles))
        concentration = float(abs(moment))
        if concentration >= MIN_ORIENTATION_CONCENTRATION:
            yaw = float(np.angle(moment) / 4.)
    rotation = np.array([[np.cos(yaw), -np.sin(yaw), 0.],
                         [np.sin(yaw), np.cos(yaw), 0.], [0., 0., 1.]])
    local = cloud @ rotation
    local_normals = normals @ rotation
    low, high = np.quantile(local, [.05, .95], axis=0)
    dims = np.array(prior_dimensions, float).copy()
    center = (low + high) / 2.
    extent_audit = []
    for axis in (0, 1):
        plus = usable & (local_normals[:, axis] > NORMAL_COMPONENT_MIN)
        minus = usable & (local_normals[:, axis] < -NORMAL_COMPONENT_MIN)
        have_low = int(plus.sum()) >= MIN_FACE_POINTS
        have_high = int(minus.sum()) >= MIN_FACE_POINTS
        near = float(np.median(local[plus, axis])) if have_low else None
        far = float(np.median(local[minus, axis])) if have_high else None
        observed_both = have_low and have_high and far > near + .08
        span = float(high[axis] - low[axis])
        if observed_both:
            lower = min(near, float(low[axis])); upper = max(far, float(high[axis]))
            dims[axis] = upper - lower
            center[axis] = (lower + upper) / 2.
        else:
            # Keep the inherited extent as a prior when the other face has not
            # been seen. Observed tangential span is a lower bound, not proof of
            # the hidden object's endpoint.
            dims[axis] = max(float(dims[axis]), span)
            if have_low and not have_high:
                center[axis] = near + dims[axis] / 2.
            elif have_high and not have_low:
                center[axis] = far - dims[axis] / 2.
        extent_audit.append(dict(axis=axis, both_faces_observed=bool(observed_both),
            low_face_points=int(plus.sum()), high_face_points=int(minus.sum()),
            observed_span_lower_bound_m=span,
            extent_interval_m=[span, float(dims[axis]) if observed_both else None],
            unresolved_extent_uses_prior=not bool(observed_both),
            point_estimate_m=float(dims[axis])))
    # Grounded-object assumption is retained. A measured upper horizontal face
    # constrains a top candidate, not certainty that no unseen layer exists.
    # Without this face, observed max height is only a lower bound; retain the
    # inherited prior and .08 m extension, without clipping higher observations.
    top_points = good & (normals[:, 2] < -NORMAL_COMPONENT_MIN)
    top_candidate = None
    if int(top_points.sum()) >= MIN_FACE_POINTS:
        upper = float(np.quantile(cloud[top_points, 2], .95))
        band = top_points & (np.abs(cloud[:, 2] - upper) <= .04)
        if int(band.sum()) >= MIN_FACE_POINTS and upper >= float(high[2]) - .08:
            top_candidate = float(np.median(cloud[band, 2]))
    if top_candidate is not None:
        dims[2] = max(top_candidate, float(high[2]))
    else:
        dims[2] = max(float(dims[2]), float(high[2]) + .08)
    center[2] = dims[2] / 2.
    return dict(center=center @ rotation.T, dims=dims, rotation=rotation,
                audit=dict(yaw_rad=yaw, orientation_concentration=concentration,
                           orientation_observed=bool(concentration >= MIN_ORIENTATION_CONCENTRATION),
                           dimensions_source='measured opposite-face bounds; unresolved extents retain prior',
                           grounded_vertical_assumption=True, top_quantile_m=float(high[2]),
                           vertical_extent=dict(observed_height_lower_bound_m=float(high[2]),
                               upper_bound_m=None,complete_extent_observed=False,
                               upper_horizontal_face_candidate_m=top_candidate,
                               point_estimate_m=float(dims[2])),
                           extent_uncertainty=extent_audit))


def contradiction_fraction(points, areas, keyframes, classify_history):
    """Unique-sample area contradiction, with B2 valid-depth/surface veto.

    Only a free or surface classification is resolved evidence. A sample behind
    a measured obstacle or out of view supplies neither support nor conflict.
    Repeated frames never multiply the same sample's evidence.
    """
    areas = np.asarray(areas, float)
    state = classify_history(points, keyframes)
    resolved = state['strict_free'] | state['measured_surface']
    resolved_area = float(areas[resolved].sum())
    free_area = float(areas[state['strict_free']].sum())
    return (free_area / resolved_area if resolved_area else 0.), dict(
        strict_free_area_m2=free_area, resolved_area_m2=resolved_area,
        unresolved_area_m2=float(areas[~resolved].sum()),
        measured_surface_area_m2=float(areas[state['measured_surface']].sum()))
