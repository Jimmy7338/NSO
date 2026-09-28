"""Observed ground removal for object association only, under a public flat-floor model.

A single revision to ArticleV1: exclude a measured, calibrated ground band
before constructing depth components. Full RGB-D remains unchanged for fusion.
All identity ambiguity, semantic qualification and feedback rules are inherited.
This is a new offline research version, not the frozen article-v1 controller.
"""
from copy import deepcopy
import hashlib
import json
from types import MappingProxyType

import numpy as np

from nso.instance_belief_v40 import PaidRGBDObservationV40
from nso.observed_instances_article_v1 import ObservedInstancesArticleV1

# Declared before saved-path outcomes. These are design constants, not a fit.
GROUND_POLICY = MappingProxyType(dict(
    camera_height_m=.9, depth_relative_sigma=.01, minimum_support_pixels=64,
    minimum_xy_span_m=.5, calibration_tolerance_m=.01,
    maximum_ground_shift_m=.03, maximum_band_m=.06,
    sigma_multiplier=3., maximum_standardized_rms=3.,
    near_m=.1, far_m=4., minimum_variance_m2=1e-12,
    assumption='public known camera height, world vertical, horizontal static floor; exact poses',
    scope='association proposals only; unchanged complete RGB-D fusion and evaluation',
))


def _mask_sha(mask):
    h = hashlib.sha256(json.dumps(list(mask.shape), separators=(',', ':')).encode())
    h.update(np.ascontiguousarray(mask, dtype=np.uint8).tobytes())
    return h.hexdigest()


def _weighted_median(values, weights):
    order = np.argsort(values, kind='stable')
    cumulative = np.cumsum(weights[order])
    return float(values[order[np.searchsorted(cumulative, .5*cumulative[-1], side='left')]])


def ground_mask_from_observation(observation):
    """Accept only the original paid-packet whitelist; never read class values.

    The ground height hypothesis uses T_camera,z - public mounting height.
    It must also be supported by a broad, low-noise measured horizontal patch.
    Insufficient or inconsistent observations return an entirely empty mask.
    """
    if type(observation) is not PaidRGBDObservationV40:
        raise TypeError('strict PaidRGBDObservationV40 required; no scene metadata accepted')
    # Revalidate the frozen packet arrays without mutating or replacing them.
    observation = PaidRGBDObservationV40.from_mapping({key: getattr(observation, key)
        for key in ('frame_id','paid_step','rgb','depth_m','intrinsic','world_from_camera')})
    depth = observation.depth_m
    height, width = depth.shape
    yy, xx = np.indices(depth.shape)
    rays = np.column_stack((xx.ravel(), yy.ravel(), np.ones(height*width))) @ np.linalg.inv(observation.intrinsic).T
    transform = observation.world_from_camera
    directions = rays @ transform[:3, :3].T
    points = depth.ravel()[:, None]*directions + transform[:3, 3]
    expected_height = float(transform[2, 3]-GROUND_POLICY['camera_height_m'])
    sigma_z = GROUND_POLICY['depth_relative_sigma']*depth.ravel()*np.abs(directions[:, 2])
    valid = ((depth.ravel() >= GROUND_POLICY['near_m']) & (depth.ravel() <= GROUND_POLICY['far_m'])
             & (directions[:, 2] < 0))
    coarse = valid & (np.abs(points[:, 2]-expected_height)
                     <= GROUND_POLICY['maximum_ground_shift_m']+GROUND_POLICY['maximum_band_m'])
    mask = np.zeros(depth.shape, dtype=bool)
    receipt = dict(schema='article.observed_ground_association.v2', policy=dict(GROUND_POLICY),
        frame_id=observation.frame_id, paid_step=observation.paid_step,
        observation_sha256=observation.sha256(), source='current_paid_depth_intrinsic_pose_and_public_mounting_height',
        class_values_read=False, private_geometry_read=False, future_observation_used=False,
        original_depth_modified=False, original_rgb_modified=False, full_frame_fusion_preserved=True,
        expected_ground_z_m=expected_height, coarse_support_pixels=int(coarse.sum()),
        accepted=False, reason=None, excluded_pixels=0)
    def finish(reason):
        receipt.update(reason=reason, excluded_pixels=int(mask.sum()), mask_sha256=_mask_sha(mask))
        mask.flags.writeable = False
        return mask, receipt
    if coarse.sum() < GROUND_POLICY['minimum_support_pixels']:
        return finish('insufficient_near_ground_support')
    z = points[coarse, 2]
    measured_height = _weighted_median(z, 1./np.maximum(sigma_z[coarse]**2, GROUND_POLICY['minimum_variance_m2']))
    receipt['measured_ground_z_m'] = measured_height
    if abs(measured_height-expected_height) > GROUND_POLICY['maximum_ground_shift_m']:
        return finish('ground_height_inconsistent_with_public_mount')
    normalization = sigma_z[coarse]+GROUND_POLICY['calibration_tolerance_m']/GROUND_POLICY['sigma_multiplier']
    rms = float(np.sqrt(np.mean(((z-measured_height)/normalization)**2)))
    receipt['standardized_residual_rms'] = rms
    if rms > GROUND_POLICY['maximum_standardized_rms']:
        return finish('measured_ground_residual_exceeds_noise_contract')
    band = GROUND_POLICY['sigma_multiplier']*sigma_z+GROUND_POLICY['calibration_tolerance_m']
    if np.any(band[coarse] > GROUND_POLICY['maximum_band_m']):
        return finish('ground_noise_band_exceeds_declared_maximum')
    inliers = coarse & (np.abs(points[:, 2]-measured_height) <= band)
    receipt['validated_support_pixels'] = int(inliers.sum())
    if inliers.sum() < GROUND_POLICY['minimum_support_pixels']:
        return finish('insufficient_validated_ground_support')
    span = np.ptp(points[inliers, :2], axis=0)
    receipt['validated_xy_span_m'] = span.tolist()
    if np.any(span < GROUND_POLICY['minimum_xy_span_m']):
        return finish('insufficient_ground_spatial_extent')
    mask.ravel()[inliers] = True
    receipt['accepted'] = True
    return finish('measured_ground_band_excluded_from_association_only')


class ObservedInstancesArticleGroundV2(ObservedInstancesArticleV1):
    """One common geometric proposal change; all ArticleV1 rules inherited."""
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._current_ground_mask = None
        self.last_ground_receipt = None

    def _proposal_components(self, valid, world, marker, rejected):
        if self._current_ground_mask is None or self._current_ground_mask.shape != valid.shape:
            raise RuntimeError('ground mask must come from this current paid packet')
        yield from super()._proposal_components(valid & ~self._current_ground_mask, world, marker, rejected)

    def observe(self, observation):
        mask, ground = ground_mask_from_observation(observation)
        self._current_ground_mask = mask
        result = super().observe(observation)
        self.last_ground_receipt = deepcopy(ground)
        result['article_ground_association'] = deepcopy(ground)
        return result
