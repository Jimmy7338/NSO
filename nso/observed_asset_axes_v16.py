"""Observed asset axes for shared candidate geometry; no semantic scoring.

The V8.1 task-start camera axis was valid only for its aligned fixture. Use
the already measured, viewing-history-oriented surface normal instead.
This still estimates a visible face, not the object's hidden full extent.
"""
from copy import deepcopy
import numpy as np

CARDINAL_AXES = np.asarray(((0., 1.), (1., 0.), (0., -1.), (-1., 0.)))


def reorient_observed_asset(asset):
    """Return a copy. Unavailable orientation is an explicit caller error."""
    low = np.asarray(asset['observed_low'], dtype=float)
    high = np.asarray(asset['observed_high'], dtype=float)
    normal = np.asarray(asset['normal_out_xy'], dtype=float)
    if low.shape != (3,) or high.shape != (3,) or normal.shape != (2,):
        raise ValueError('invalid measured geometry shape')
    if not all(np.isfinite(x).all() for x in (low, high, normal)) or np.any(high < low):
        raise ValueError('invalid measured bounds or normal')
    if not asset.get('orientation_available') or asset['normal_sign_depth_support'] <= 0:
        raise ValueError('depth-supported observing history required')
    norm = np.linalg.norm(normal)
    if norm < 1e-10:
        raise ValueError('undefined measured direction')
    unit = normal / norm
    front = CARDINAL_AXES[int(np.argmax(CARDINAL_AXES @ unit))].copy()
    back = -front
    side = np.asarray((-back[1], back[0]))
    center, span = (low + high) / 2., high - low
    depth = float(span[:2] @ np.abs(back))
    width = float(span[:2] @ np.abs(side))
    result = deepcopy(asset)
    result.update(aabb_center=center, front_axis=front, back_axis=back, side_axis=side,
        measured_width_m=width, measured_depth_m=depth, measured_height_m=float(span[2]),
        rear_boundary_xy=center[:2] + back * depth / 2.,
        axis_evidence=dict(schema='observed_asset_axes_v16/1',
            source='observed_camera_direction_fallback' if asset['normal_fallback'] else 'depth_surface_normal',
            normal_out_xy=unit.tolist(), cardinal_alignment=float(front @ unit),
            discrete_axis_tie=bool(np.count_nonzero(CARDINAL_AXES @ unit == max(CARDINAL_AXES @ unit)) > 1),
            hidden_extent_inferred=False, semantic_labels_used=False,
            task_start_camera_used=False, calibrated=False))
    return result


def measured_assets_v16(mapper):
    """Explicit future integration entrypoint; original callers stay frozen."""
    from nso.competition_candidates_v8_1 import measured_assets
    # Preserve the original measured clusters and ordering. Geometry is the
    # only change; do not silently discard an unsupported cluster.
    return [reorient_observed_asset(asset) for asset in measured_assets(mapper)]
