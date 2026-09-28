"""Bounded paid-marker plane estimation under the public upright mount model.

The legacy successful single-frame fit is retained. The fallback estimates a
vertical plane from several measured RGB-D frames with declared 1% axial depth
noise. Linearized three-sigma and leave-one-view envelopes are diagnostics, not
calibrated confidence guarantees. No asset, object pose or future sensor enters.
"""
from copy import deepcopy
import math

import numpy as np
from scipy.optimize import least_squares

from nso.instance_belief_v40 import PaidRGBDObservationV40
from nso.observed_instances_v41 import _payload_sha, support_digest
from nso.observed_residual_v41 import (
    DEPTH_CONVENTION, _array_hash, _digest, _indices,
    fit_marker_plane_v41, observed_points_v41, pixel_mask_sha256_v41,
)


_ASSOCIATION_KEYS = {
    'component_first_pixel', 'component_pixels', 'instance_id', 'pixel_indices',
    'points_world_m', 'marker_pixel_indices', 'frame_id', 'paid_step',
    'observation_sha256', 'support_sha256', 'novel_support_voxels',
    'duplicate_measurement', 'no_new_support', 'geometry_feedback_eligible',
    'association', 'marker_anchor_world_m', 'marker_observation_sha256',
    'marker_frame_id', 'marker_paid_step', 'association_uncertain',
    'article_feedback_eligibility',
}


def _angle(a, b):
    return float(np.degrees(np.arccos(np.clip(np.dot(a, b), -1., 1.))))


def _fit(frames, relative_noise):
    """Two-parameter axial-noise WLS; camera rays, not bearing as normal."""
    points = np.concatenate([f['points'] for f in frames])
    rays = np.concatenate([f['rays'] for f in frames])
    origins = np.concatenate([np.tile(f['origin'], (len(f['depth']), 1)) for f in frames])
    depth = np.concatenate([f['depth'] for f in frames])
    center = points.mean(axis=0)
    _, axes = np.linalg.eigh(np.cov(points[:, :2], rowvar=False, bias=True))
    normal = np.r_[axes[:, 0], 0.]
    if np.dot(normal, origins.mean(axis=0)-center) < 0:
        normal = -normal
    theta = math.atan2(normal[1], normal[0])
    sigma = np.maximum(.001, relative_noise*depth)

    def residual(parameters):
        n = np.array([math.cos(parameters[0]), math.sin(parameters[0]), 0.])
        denominator = rays @ n
        if np.any(np.abs(denominator) < 1e-5):
            return np.full(len(depth), 1e6)
        predicted = ((center-origins) @ n + parameters[1])/denominator
        return (depth-predicted)/sigma

    fitted = least_squares(residual, [theta, 0.],
        bounds=([theta-math.pi/3., -.20], [theta+math.pi/3., .20]),
        max_nfev=80, ftol=1e-10, xtol=1e-10, gtol=1e-10)
    if not fitted.success or not np.isfinite(fitted.fun).all():
        return None
    information = fitted.jac.T @ fitted.jac
    if np.linalg.cond(information) > 1e8:
        return None
    reduced = float(fitted.fun @ fitted.fun/max(1, len(depth)-2))
    covariance = np.linalg.inv(information)*max(1., reduced)
    normal = np.array([math.cos(fitted.x[0]), math.sin(fitted.x[0]), 0.])
    return dict(center=center, parameters=fitted.x, covariance=covariance,
                normal=normal, reduced_chi_square=reduced,
                outlier_fraction=float(np.mean(np.abs(fitted.fun) > 4.)),
                rms=float(np.sqrt(np.mean(((points-center)@normal-fitted.x[1])**2))))


def _anchor(frames, fit, parameters):
    """Bound label-centre ambiguity using public size and pixel footprints.

    Ray/plane intersections remove axial noise from the *extent estimate*, not
    from the measured inputs or reconstruction. The centre interval includes
    unobserved edge extent and half-pixel support. It is not proof of no occlusion.
    """
    normal = np.array([math.cos(parameters[0]), math.sin(parameters[0]), 0.])
    horizontal = np.cross([0., 0., 1.], normal)
    offset = float(fit['center'] @ normal + parameters[1])
    coordinates, footprint = [], np.zeros(2)
    for frame in frames:
        denom = frame['rays'] @ normal
        distance = offset-frame['origin'] @ normal
        if np.any(np.abs(denom) < 1e-5):
            return None
        depths = distance/denom
        if np.any(depths <= .05) or np.any(depths > 4.5):
            return None
        xyz = frame['origin']+frame['rays']*depths[:, None]
        uv = np.column_stack((xyz@horizontal, xyz[:, 2]))
        coordinates.append(uv)
        for delta in ((-.5, -.5), (-.5, .5), (.5, -.5), (.5, .5)):
            directions = frame['rays'] + delta[0]*frame['pixel_dx'] + delta[1]*frame['pixel_dy']
            denominator = directions@normal
            if np.any(np.abs(denominator) < 1e-5):
                return None
            corners = frame['origin']+directions*(distance/denominator)[:, None]
            change = np.column_stack((corners@horizontal, corners[:, 2]))-uv
            footprint = np.maximum(footprint, np.max(np.abs(change), axis=0))
    uv = np.concatenate(coordinates)
    low, high = uv.min(axis=0), uv.max(axis=0)
    half = np.array([.16, .15])
    lower, upper = high-half-footprint, low+half+footprint
    if np.any(lower > upper):
        return None
    middle = (lower+upper)*.5
    anchor = normal*offset+horizontal*middle[0]+np.array([0., 0., middle[1]])
    return dict(anchor=anchor, extent=high-low, footprint=footprint,
                centre_ambiguity=float(np.linalg.norm((upper-lower)*.5)))


class ObservedMarkerPlanesArticleV1:
    """Observe monotonic paid frames; plane(id) returns a reliable fit or None."""

    def __init__(self, *, maximum_instances=8, maximum_paid_frames=512,
                 maximum_views=8, maximum_points=1024, minimum_views=3,
                 minimum_points=24, minimum_translation_m=.15,
                 minimum_rotation_deg=8., relative_depth_noise=.01):
        for name, value in dict(maximum_instances=maximum_instances,
                maximum_paid_frames=maximum_paid_frames, maximum_views=maximum_views,
                maximum_points=maximum_points, minimum_views=minimum_views,
                minimum_points=minimum_points).items():
            if type(value) is not int or value < 1:
                raise ValueError(name+' must be a positive integer')
        if not 3 <= minimum_views <= maximum_views <= 32 or not 8 <= minimum_points <= maximum_points <= 8192:
            raise ValueError('bounded multi-view capacities required')
        if (not .1 <= minimum_translation_m <= 1. or not 5. <= minimum_rotation_deg <= 30.
                or not .001 <= relative_depth_noise <= .05):
            raise ValueError('supported view separation and declared noise required')
        self.configuration = dict(schema='paid_marker_planes.article.v1',
            maximum_instances=maximum_instances, maximum_paid_frames=maximum_paid_frames,
            maximum_views=maximum_views, maximum_points=maximum_points,
            minimum_views=minimum_views, minimum_points=minimum_points,
            minimum_translation_m=float(minimum_translation_m),
            minimum_rotation_deg=float(minimum_rotation_deg),
            relative_depth_noise=float(relative_depth_noise),
            public_marker_dimensions_m=[.32, .30], anchor_limit_m=.06,
            yaw_limit_deg=12., parameter_envelope='linearized 3-sigma plus leave-one-view sensitivity',
            confidence_calibrated=False, gravity_alignment_assumed=True)
        self.configuration_sha256 = _digest(self.configuration)
        self._instances, self._frames, self._payloads = {}, set(), set()
        self._last_step = -1

    def plane(self, instance_id):
        state = self._instances.get(instance_id)
        return None if state is None or state['conflict'] else deepcopy(state['plane'])

    def receipt(self, instance_id):
        state = self._instances.get(instance_id)
        return dict(plane_fit=self.plane(instance_id),
                    conflict=False if state is None else state['conflict'],
                    source_frames=[] if state is None else deepcopy(state['sources']))

    def _multiview(self, frames):
        result = dict(accepted=False, reason='insufficient_distinct_marker_views',
            plane_source='paid_multiview_marker', estimated_from_current_pixels=False,
            frame_count=len(frames), valid_depth_pixels=sum(len(f['depth']) for f in frames),
            source_frames=[deepcopy(f['source']) for f in frames],
            configuration_sha256=self.configuration_sha256, confidence_calibrated=False,
            object_centre_known=False, completeness_guaranteed=False)
        if (len(frames) < self.configuration['minimum_views']
                or result['valid_depth_pixels'] < self.configuration['minimum_points']):
            return result
        fit = _fit(frames, self.configuration['relative_depth_noise'])
        if fit is None:
            result['reason'] = 'ill_conditioned_marker_parameters'
            return result
        result.update(plane_rms_m=fit['rms'], reduced_chi_square=fit['reduced_chi_square'],
                      noise_outlier_fraction=fit['outlier_fraction'],
                      linearized_parameter_covariance=fit['covariance'].tolist())
        if fit['reduced_chi_square'] > 4. or fit['outlier_fraction'] > .10:
            result['reason'] = 'marker_inconsistent_with_declared_depth_noise'
            return result
        anchor = _anchor(frames, fit, fit['parameters'])
        if anchor is None:
            result['reason'] = 'marker_extent_inconsistent_with_public_mount'
            return result
        normal = fit['normal']
        for frame in frames:
            toward = frame['origin']-anchor['anchor']
            if np.dot(normal, toward)/np.linalg.norm(toward) < .35:
                result['reason'] = 'marker_views_disagree_on_front_or_are_oblique'
                return result
        yaw = max(1., float(np.degrees(3.*math.sqrt(fit['covariance'][0, 0]))))
        uncertainty = max(.005, anchor['centre_ambiguity'])
        eigenvalues, axes = np.linalg.eigh(fit['covariance'])
        for axis, value in zip(axes.T, eigenvalues):
            for sign in (-1., 1.):
                altered = _anchor(frames, fit, fit['parameters']+sign*3.*math.sqrt(max(0., value))*axis)
                if altered is None:
                    result['reason'] = 'uncertainty_envelope_has_ambiguous_marker_extent'
                    return result
                uncertainty = max(uncertainty, np.linalg.norm(altered['anchor']-anchor['anchor'])
                                  + altered['centre_ambiguity'])
        leave_one = []
        for index in range(len(frames)):
            subset = frames[:index]+frames[index+1:]
            check = _fit(subset, self.configuration['relative_depth_noise'])
            projected = None if check is None else _anchor(subset, check, check['parameters'])
            if projected is None:
                result['reason'] = 'leave_one_view_fit_unidentifiable'
                return result
            deviation = _angle(normal, check['normal'])
            displacement = float(np.linalg.norm(projected['anchor']-anchor['anchor']))
            yaw = max(yaw, deviation)
            uncertainty = max(uncertainty, displacement+projected['centre_ambiguity'])
            leave_one.append(dict(excluded_frame_id=frames[index]['source']['frame_id'],
                                  yaw_change_deg=deviation, anchor_change_m=displacement))
        result.update(anchor_world_m=anchor['anchor'].tolist(), outward_normal_world=normal.tolist(),
            horizontal_world=np.cross([0., 0., 1.], normal).tolist(), up_world=[0., 0., 1.],
            observed_marker_extent_m=anchor['extent'].tolist(),
            pixel_footprint_bound_m=float(np.max(anchor['footprint'])),
            anchor_uncertainty_m=float(uncertainty), yaw_uncertainty_deg=float(yaw),
            leave_one_view=leave_one, depth_convention=DEPTH_CONVENTION)
        if uncertainty > .06 or yaw > 12.:
            result['reason'] = 'marker_parameter_uncertainty_too_large'
            return result
        result.update(accepted=True, reason='observed_paid_multiview_upright_label_plane')
        return result

    def observe(self, observation, accepted_associations):
        if type(observation) is not PaidRGBDObservationV40:
            raise TypeError('strict paid RGBD observation required')
        if observation.depth_m.shape[0] > 480 or observation.depth_m.shape[1] > 640:
            raise ValueError('bounded camera image up to 640 by 480 required')
        if observation.frame_id in self._frames or observation.paid_step <= self._last_step:
            raise ValueError('unique frames and increasing paid steps required')
        if len(self._frames) >= self.configuration['maximum_paid_frames']:
            raise ValueError('paid-frame capacity exceeded')
        if (not isinstance(accepted_associations, (list, tuple))
                or len(accepted_associations) > self.configuration['maximum_instances']):
            raise ValueError('bounded accepted association sequence required')
        packet, payload = observation.sha256(), _payload_sha(observation)
        validated, seen, used_pixels = [], set(), set()
        for item in accepted_associations:
            if not isinstance(item, dict) or set(item)-_ASSOCIATION_KEYS:
                raise ValueError('observed association whitelist violation')
            key = item['instance_id']
            if not isinstance(key, str) or not key or key in seen:
                raise ValueError('unique opaque instance ID required')
            if (item['observation_sha256'] != packet or item['frame_id'] != observation.frame_id
                    or item['paid_step'] != observation.paid_step):
                raise ValueError('paid frame/hash binding mismatch')
            pixels = _indices(item['pixel_indices'], observation.depth_m.shape, 'support pixels')
            marker = _indices(item.get('marker_pixel_indices', []), observation.depth_m.shape, 'marker pixels')
            if not np.isin(marker, pixels).all() or used_pixels.intersection(pixels.tolist()):
                raise ValueError('unique nonoverlapping association support required')
            supplied = np.asarray(item['points_world_m'], dtype=float)
            measured = observed_points_v41(observation, pixels)
            if (supplied.shape != measured.shape or not np.allclose(supplied, measured, atol=1e-7, rtol=0)
                    or support_digest(supplied) != item['support_sha256']):
                raise ValueError('association support differs from paid depth/hash')
            seen.add(key); used_pixels.update(pixels.tolist())
            validated.append((key, item, marker))
        if len(set(self._instances) | seen) > self.configuration['maximum_instances']:
            raise ValueError('instance capacity exceeded')
        duplicate = payload in self._payloads
        results = []
        for key, item, marker in validated:
            state = self._instances.setdefault(key, dict(plane=None, conflict=False, frames=[], sources=[]))
            source = dict(frame_id=observation.frame_id, paid_step=observation.paid_step,
                observation_sha256=packet, payload_sha256=payload,
                support_sha256=item['support_sha256'],
                marker_mask_sha256=pixel_mask_sha256_v41(marker, observation.depth_m.shape),
                intrinsic_sha256=_array_hash(observation.intrinsic),
                world_from_camera_sha256=_array_hash(observation.world_from_camera),
                depth_sha256=_array_hash(observation.depth_m))
            row = dict(instance_id=key, accepted=False, cache_added=False,
                       duplicate_measurement=duplicate, source=source)
            results.append(row)
            if item.get('association_uncertain', False) or duplicate or item.get('duplicate_measurement', False):
                row['reason'] = 'uncertain_association_or_duplicate_payload'
                continue
            single = fit_marker_plane_v41(observation, marker)
            row['single_frame_fit'] = single
            if single['accepted']:
                proposed = deepcopy(single)
                proposed.update(plane_source='paid_single_frame_legacy', source_frames=[source])
            else:
                valid = (observation.depth_m.ravel()[marker] >= .1) & (observation.depth_m.ravel()[marker] <= 4.)
                if len(marker) < 4 or valid.sum() < 4 or valid.mean() < .9:
                    row['reason'] = 'insufficient_current_marker_depth'
                    continue
                marker = marker[valid]
                origin = observation.world_from_camera[:3, 3]
                rotation = observation.world_from_camera[:3, :3]
                near = any(np.linalg.norm(origin-f['origin']) < self.configuration['minimum_translation_m']
                    and np.degrees(np.arccos(np.clip((np.trace(rotation.T@f['rotation'])-1.)/2., -1., 1.)))
                    < self.configuration['minimum_rotation_deg'] for f in state['frames'])
                if near:
                    row['reason'] = 'near_repeat_view_excluded'
                    continue
                if (len(state['frames']) >= self.configuration['maximum_views']
                        or len(marker)+sum(len(f['depth']) for f in state['frames']) > self.configuration['maximum_points']):
                    row['reason'] = 'marker_cache_capacity_reached'
                    continue
                depth = observation.depth_m.ravel()[marker].astype(float)
                points = observed_points_v41(observation, marker)
                inverse = np.linalg.inv(observation.intrinsic)
                frame = dict(points=points, depth=depth, origin=origin.copy(), rotation=rotation.copy(),
                    rays=(points-origin)/depth[:, None], pixel_dx=rotation@inverse[:, 0],
                    pixel_dy=rotation@inverse[:, 1], source=source)
                state['frames'].append(frame); state['sources'].append(source)
                row['cache_added'] = True
                proposed = self._multiview(state['frames'])
            row['current_plane_fit'] = deepcopy(proposed)
            row['reason'] = proposed['reason']
            if proposed['accepted']:
                if state['plane'] is None:
                    state['plane'] = deepcopy(proposed)
                    if not state['sources']:
                        state['sources'] = [source]
                elif (np.linalg.norm(np.asarray(proposed['anchor_world_m'])-state['plane']['anchor_world_m']) > .15
                        or _angle(proposed['outward_normal_world'], state['plane']['outward_normal_world']) > 20.):
                    state['conflict'] = True
                    row['reason'] = 'conflicting_paid_marker_planes'
                row['accepted'] = not state['conflict']
            row['conflict'] = state['conflict']
            row['reliable_cached_plane'] = self.plane(key) is not None
        self._frames.add(observation.frame_id); self._payloads.add(payload)
        self._last_step = observation.paid_step
        return dict(schema='paid_marker_planes.observe.article.v1', results=results,
                    frame_id=observation.frame_id, paid_step=observation.paid_step,
                    observation_sha256=packet, configuration_sha256=self.configuration_sha256,
                    ground_truth_used=False, future_observation_used=False)
