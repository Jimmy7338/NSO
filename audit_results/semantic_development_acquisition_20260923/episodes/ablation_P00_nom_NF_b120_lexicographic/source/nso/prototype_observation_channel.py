"""Uncalibrated diagnostic channels from shared public structure prototypes.

Only an already observed label plane and a public candidate camera are used.
The output is a finite forecast for planning, NOT a rendered future sensor
observation. Prototype self-occlusion is modeled; external occlusion, missing
sensor returns, pose-error distributions and real sensor calibration are not.
No class, posterior, selected actual structure or evaluation geometry enters.
"""
from collections.abc import Mapping
from functools import lru_cache
import hashlib
import json

import numpy as np

from nso.development_geometry_v40 import STRUCTURES
from nso.observed_residual_v41 import build_prototype_bank_v41, _ray_box_depths
from nso.surface_evaluation_v40 import CandidateViewV40


CONFIGURATION = {
    'ray_grid_columns': 8,
    'ray_grid_rows': 6,
    'depth_difference_threshold_m': 1e-6,
    'truncation_m': .25,
    'evidence_scale_m': .05,
    'scale_weights': [1./3., 1./3., 1./3.],
    'scale_handling': 'uniform likelihood-kernel marginalization over comparison scale; uniform source-scale mixture',
    'missing_hit_handling': 'ignore rays unless both public predictions hit; no common ray gives neutral kernel 1',
    'distinct_ray_rule': 'at least two structures predict finite different depths at the same nominal scale',
}
_RECEIPT_KEYS = {'plane_fit', 'conflict', 'source_frames'}
_SOURCE_KEYS = {'frame_id', 'paid_step', 'observation_sha256', 'support_sha256', 'mask_sha256'}
_PLANE_KEYS = {
    'accepted', 'reason', 'frame_id', 'paid_step', 'observation_sha256',
    'marker_mask_sha256', 'marker_pixels', 'depth_convention', 'valid_depth_pixels',
    'plane_rms_m', 'observed_marker_extent_m', 'pixel_footprint_bound_m',
    'marker_completeness_tolerance_m', 'anchor_world_m', 'outward_normal_world',
    'horizontal_world', 'up_world', 'anchor_uncertainty_m', 'yaw_uncertainty_deg',
    'estimated_from_current_pixels', 'object_centre_known', 'completeness_guaranteed',
}


def _digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'),
                                    allow_nan=False).encode()).hexdigest()


@lru_cache(maxsize=1)
def _bank():
    return build_prototype_bank_v41()


def _validated_plane(receipt):
    if receipt is None:
        return None, 'no_observed_label_plane'
    if not isinstance(receipt, Mapping) or set(receipt) != _RECEIPT_KEYS:
        raise ValueError('exact V42 per-instance observed-plane receipt required')
    if type(receipt['conflict']) is not bool:
        raise ValueError('observed plane conflict flag must be bool')
    if not isinstance(receipt['source_frames'], (list, tuple)):
        raise ValueError('observed source-frame receipt list required')
    for source in receipt['source_frames']:
        if not isinstance(source, Mapping) or set(source) != _SOURCE_KEYS:
            raise ValueError('observed source-frame field whitelist violation')
    plane = receipt['plane_fit']
    if plane is not None and (not isinstance(plane, Mapping) or set(plane)-_PLANE_KEYS):
        raise ValueError('observed label-plane field whitelist violation')
    if receipt['conflict']:
        return None, 'conflicting_observed_label_planes'
    if plane is None or plane.get('accepted') is not True:
        return None, 'no_reliable_observed_label_plane'
    anchor = np.asarray(plane.get('anchor_world_m'), dtype=float)
    outward = np.asarray(plane.get('outward_normal_world'), dtype=float)
    if (anchor.shape != (3,) or outward.shape != (3,) or
            not np.isfinite(anchor).all() or not np.isfinite(outward).all() or
            not np.isclose(np.linalg.norm(outward), 1., rtol=0., atol=1e-6) or
            abs(outward[2]) > 1e-6):
        raise ValueError('finite measured anchor and upright unit plane normal required')
    # A caller cannot supply the original scene pose here; these are precisely
    # the two public geometric estimates used by the V42 nominal-area forecast.
    return (tuple(anchor.tolist()), tuple(outward.tolist())), None


@lru_cache(maxsize=512)
def _predictions(anchor_tuple, outward_tuple, intrinsic_tuple, transform_tuple, width, height):
    anchor, outward = np.array(anchor_tuple), np.array(outward_tuple)
    intrinsic = np.array(intrinsic_tuple).reshape(3, 3)
    transform = np.array(transform_tuple).reshape(4, 4)
    rotation = np.column_stack((np.cross([0., 0., 1.], outward), -outward, [0., 0., 1.]))
    x = (np.arange(8)+.5)*width/8.-.5
    y = (np.arange(6)+.5)*height/6.-.5
    xx, yy = np.meshgrid(x, y)
    rays = np.column_stack((xx.ravel(), yy.ravel(), np.ones(48))) @ np.linalg.inv(intrinsic).T
    rays /= rays[:, 2, None]
    directions = (rays @ transform[:3, :3].T) @ rotation
    predictions = np.empty((4, 3, 48))
    for index, (name, scale, boxes, local_anchor) in enumerate(_bank().candidates):
        translation = anchor-rotation@local_anchor
        origin = (transform[:3, 3]-translation) @ rotation
        predictions[index//3, index % 3] = _ray_box_depths(origin, directions, boxes)
    predictions.flags.writeable = False
    return predictions


def build_channel(plane_receipt, candidate, *, repeated_view=False):
    """Return an H x H public-prototype diagnostic forecast (H=4).

    ``plane_receipt`` is one value from
    ``ViewQualityPredictorV42.observed_plane_receipts()``; it contains
    ``plane_fit``, ``conflict`` and ``source_frames``. No instance snapshot or
    class is accepted. ``candidate`` is a strict ``CandidateViewV40``.

    For each possible generating structure and scale, candidate depths are
    compared with each nominal alternative on jointly finite distinguishing
    rays. The kernel is exp(-mean(min(abs(z1-z2), .25))/.05). Kernel likelihoods
    are uniformly marginalized over alternative scales, normalized over four
    pseudo-class outcomes, and averaged over generating scales. This defines
    a stochastic planning proxy, not an empirically calibrated confusion table.
    A missed prototype ray provides no empty-space evidence.
    """
    if type(candidate) is not CandidateViewV40:
        raise TypeError('strict public CandidateViewV40 required')
    if (candidate.near_m != .1 or candidate.far_m != 4. or
            candidate.width > 640 or candidate.height > 480):
        raise ValueError('bounded camera with axial clipping [0.1,4.0] required')
    if type(repeated_view) is not bool:
        raise ValueError('repeated_view must be bool')
    plane, reason = _validated_plane(plane_receipt)
    if repeated_view:
        reason = 'previously_paid_identical_candidate_view'
    candidate_receipt = dict(view_id=candidate.view_id,
        intrinsic=candidate.intrinsic.tolist(), world_from_camera=candidate.world_from_camera.tolist(),
        width=candidate.width, height=candidate.height, near_m=candidate.near_m, far_m=candidate.far_m)
    receipt = dict(schema='prototype_diagnostic_channel.v1',
        structure_names=list(STRUCTURES), outcome_names=list(STRUCTURES),
        candidate=candidate_receipt, candidate_sha256=_digest(candidate_receipt),
        observed_plane_receipt_sha256=None if plane_receipt is None else _digest(dict(plane_receipt)),
        prototype_bank_sha256=_bank().sha256,
        configuration=json.loads(json.dumps(CONFIGURATION)),
        configuration_sha256=_digest(CONFIGURATION),
        score_is_calibrated=False, observation_channel_is_calibrated=False,
        actual_future_sensor_rendered=False, ground_truth_used=False,
        class_or_structure_posterior_used=False, external_occlusion_modeled=False,
        missing_return_is_empty_space_evidence=False,
        pose_uncertainty_marginalized=False, prototype_self_occlusion_modeled=True,
        distinguishing_ray_count=0, pair_common_ray_counts=[[0.]*4 for _ in range(4)],
        prototype_hit_counts=[[0]*3 for _ in range(4)],
        fallback=reason is not None, fallback_reason=reason,
        limitation='uncalibrated public nominal-depth diagnostic proxy; no external visibility or real sensor forecast')
    uniform = np.full((4, 4), .25)
    if reason is not None:
        receipt['channel'] = uniform.tolist()
        return receipt
    anchor, outward = plane
    depths = _predictions(anchor, outward, tuple(candidate.intrinsic.ravel()),
                         tuple(candidate.world_from_camera.ravel()), candidate.width, candidate.height)
    finite = np.isfinite(depths)
    receipt['prototype_hit_counts'] = finite.sum(axis=2).tolist()
    receipt['prototype_depth_predictions_sha256'] = hashlib.sha256(depths.tobytes()).hexdigest()
    distinguishing = np.zeros(48, dtype=bool)
    for scale in range(3):
        valid = finite[:, scale]
        lower = np.min(np.where(valid, depths[:, scale], np.inf), axis=0)
        upper = np.max(np.where(valid, depths[:, scale], -np.inf), axis=0)
        distinguishing |= (valid.sum(axis=0) >= 2) & (upper-lower > 1e-6)
    receipt['distinguishing_ray_count'] = int(distinguishing.sum())
    if not finite.any() or not distinguishing.any():
        receipt.update(channel=uniform.tolist(), fallback=True,
                       fallback_reason='no_visible_public_prototype' if not finite.any()
                       else 'no_jointly_hit_distinguishing_rays')
        return receipt
    kernels = np.ones((4, 3, 4, 3))
    common_counts = np.zeros((4, 3, 4, 3), dtype=int)
    for actual in range(4):
        for source_scale in range(3):
            for alternative in range(4):
                for comparison_scale in range(3):
                    common = distinguishing & finite[actual, source_scale] & finite[alternative, comparison_scale]
                    count = int(common.sum())
                    common_counts[actual, source_scale, alternative, comparison_scale] = count
                    if count:
                        errors = np.minimum(np.abs(depths[actual, source_scale, common]
                                                   - depths[alternative, comparison_scale, common]), .25)
                        kernels[actual, source_scale, alternative, comparison_scale] = np.exp(-errors.mean()/.05)
    marginal = kernels.mean(axis=3)
    conditional_outcomes = marginal/marginal.sum(axis=2, keepdims=True)
    channel = conditional_outcomes.mean(axis=1)
    channel /= channel.sum(axis=1, keepdims=True)
    receipt['pair_common_ray_counts'] = common_counts.mean(axis=(1, 3)).tolist()
    receipt['maximum_channel_column_range'] = float(np.max(np.ptp(channel, axis=0)))
    if receipt['maximum_channel_column_range'] <= 1e-12:
        receipt.update(channel=uniform.tolist(), fallback=True,
                       fallback_reason='indistinguishable_channel_rows')
    else:
        receipt['channel'] = channel.tolist()
    return receipt
