"""Observed-only nominal-surface forecasting for STGHP; no future sensor input.

This is a finite-quadrature *forecast*, not a reconstruction or performance
measurement. Shared structure prototypes are located using a measured label
plane. A structure posterior mixes their expected newly visible physical area.
Neither class-specific importance weights nor actual scene geometry enter.
"""
from copy import deepcopy
import hashlib
import json
import math
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree

from nso.development_geometry_v40 import STRUCTURES, union_box_mesh
from nso.instance_belief_v40 import PaidRGBDObservationV40
from nso.observed_instances_v41 import support_digest
from nso.observed_residual_v41 import (
    _ray_box_depths, build_prototype_bank_v41, fit_marker_plane_v41,
    observed_points_v41, pixel_mask_sha256_v41,
)
from nso.surface_evaluation_v40 import CandidateViewV40


_ASSOCIATION_KEYS = {
    'component_first_pixel', 'component_pixels', 'instance_id', 'pixel_indices',
    'points_world_m', 'marker_pixel_indices', 'frame_id', 'paid_step',
    'observation_sha256', 'support_sha256', 'novel_support_voxels',
    'duplicate_measurement', 'no_new_support', 'geometry_feedback_eligible',
    'association', 'marker_anchor_world_m', 'marker_observation_sha256',
    'marker_frame_id', 'marker_paid_step',
}
_SNAPSHOT_KEYS = {
    'instance_id', 'anchor_world_m', 'support_points_world_m', 'support_sha256',
    'structure_names', 'structure_probabilities', 'geometry_log_scores',
    'geometry_prior', 'active_structure_prior', 'observed_class',
    'semantic_conditioning_used', 'class_conflict', 'association_uncertain',
    'distinct_class_supports', 'geometric_feedback_frames', 'novel_support_frames',
    'accepted_observations', 'probability_calibrated', 'marker_anchor_world_m',
    'marker_observation_sha256', 'marker_frame_id', 'marker_paid_step',
}


def _digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'),
                                    allow_nan=False).encode()).hexdigest()


def _measurement_digest(observation):
    # Match V41's identity exactly, including array layout metadata. Equal raw
    # bytes with a different image shape describe different measured rays.
    fields = []
    for name in ('rgb', 'depth_m', 'intrinsic', 'world_from_camera'):
        array = np.ascontiguousarray(getattr(observation, name))
        fields.append(hashlib.sha256(f'{array.dtype.str}:{array.shape}:'.encode()+array.tobytes()).hexdigest())
    return hashlib.sha256(''.join(fields).encode()).hexdigest()


def _finite_positive(value, name, minimum, maximum):
    if (isinstance(value, bool) or not isinstance(value, (float, int))
            or not math.isfinite(value) or not minimum <= value <= maximum):
        raise ValueError(name + ' outside finite supported interval')
    return float(value)


def _pixels(value, shape):
    values = np.asarray(value)
    if values.size == 0:
        return np.empty(0, np.int64)
    if (values.ndim != 1 or values.dtype.kind not in 'iu'
            or np.any(values < 0) or np.any(values >= np.prod(shape))
            or np.any(np.diff(values.astype(np.int64)) <= 0)):
        raise ValueError('sorted unique current-packet pixels required')
    return values.astype(np.int64)


def _sample_union(boxes, spacing):
    """Deterministic area quadrature on the exterior of a union, no internal faces."""
    vertices, triangles = union_box_mesh(boxes)
    xyz = vertices[triangles]
    cross = np.cross(xyz[:, 1]-xyz[:, 0], xyz[:, 2]-xyz[:, 0])
    norms = np.linalg.norm(cross, axis=1)
    areas = norms / 2.
    counts = np.maximum(1, np.ceil(areas / spacing**2).astype(np.int64))
    if counts.sum() > 20000:
        raise ValueError('prototype surface sample cap exceeded')
    points, weights, normals = [], [], []
    golden = (math.sqrt(5.)-1.)/2.
    for triangle, area, normal, count in zip(xyz, areas, cross/norms[:, None], counts):
        u = np.sqrt((np.arange(count)+.5)/count)
        v = (np.arange(count)*golden+.5) % 1.
        points.append(np.column_stack((1.-u, u*(1.-v), u*v)) @ triangle)
        weights.append(np.full(count, area/count))
        normals.append(np.tile(normal, (count, 1)))
    return np.concatenate(points), np.concatenate(weights), np.concatenate(normals)


def _candidate_receipt(candidate):
    if type(candidate) is not CandidateViewV40:
        raise TypeError('strict public CandidateViewV40 required')
    if (candidate.width > 640 or candidate.height > 480
            or candidate.near_m != .1 or candidate.far_m != 4.):
        raise ValueError('bounded V41 camera with axial clipping [0.1,4.0] required')
    return dict(view_id=candidate.view_id, intrinsic=candidate.intrinsic.tolist(),
                world_from_camera=candidate.world_from_camera.tolist(),
                width=candidate.width, height=candidate.height,
                near_m=candidate.near_m, far_m=candidate.far_m)


class ViewQualityPredictorV42:
    """Public observed-plane adapter and posterior expected novel-area forecast.

    Call ``observe(packet, association_receipt['accepted'])`` once per paid
    packet, then ``forecast(ledger.snapshot()['instances'][i], candidates)``.
    The same predictor can score S and G snapshots: its observation state is
    semantic-free. Its support settings must match the V41 association ledger.
    """

    def __init__(self, *, support_voxel_m=.05, maximum_support_points=4096,
                 surface_spacing_m=.08, observed_distance_m=.08,
                 maximum_instances=64):
        self.support_voxel_m = _finite_positive(support_voxel_m, 'support_voxel_m', .01, .2)
        self.surface_spacing_m = _finite_positive(surface_spacing_m, 'surface_spacing_m', .05, .25)
        self.observed_distance_m = _finite_positive(observed_distance_m, 'observed_distance_m', .01, .3)
        if type(maximum_support_points) is not int or not 16 <= maximum_support_points <= 4096:
            raise ValueError('maximum_support_points must be an integer in [16,4096]')
        if type(maximum_instances) is not int or not 1 <= maximum_instances <= 64:
            raise ValueError('maximum_instances must be an integer in [1,64]')
        self.maximum_support_points = maximum_support_points
        self.maximum_instances = maximum_instances
        self.bank = build_prototype_bank_v41()
        self._samples = [_sample_union(boxes, self.surface_spacing_m)
                         for _, _, boxes, _ in self.bank.candidates]
        self._instances = {}
        self._frames = set()
        self._payloads = set()
        self._paid_cameras = []
        self._last_step = -1
        self._forecast_cache = {}
        self.source_sha256 = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
        self.configuration = dict(support_voxel_m=self.support_voxel_m,
            maximum_support_points=maximum_support_points, maximum_instances=maximum_instances,
            maximum_paid_frames=512, maximum_candidate_views_per_call=256,
            repeated_view_absolute_tolerance=1e-9,
            repeated_view_rule='exclude any previously paid same K/T/image shape; static-scene new-view-only eligibility, independent of association',
            surface_spacing_m=self.surface_spacing_m, observed_distance_m=self.observed_distance_m,
            scale_probability='uniform over shared 0.85,1.0,1.15 nominal dimensions',
            sampling='deterministic triangle-area quadrature; golden-sequence barycentric samples',
            prototype_bank_sha256=self.bank.sha256, source_sha256=self.source_sha256)
        self.configuration_sha256 = _digest(self.configuration)

    def observe(self, observation, accepted_associations):
        """Validate current measured support and independently fit/cache its label plane.

        Geometry is reconstructed from paid pixels and compared with supplied
        association points. Unknown input fields are rejected, rather than
        silently accepting a private scene extension at this boundary.
        """
        if type(observation) is not PaidRGBDObservationV40:
            raise TypeError('strict paid RGBD packet required')
        if observation.frame_id in self._frames or observation.paid_step != self._last_step+1:
            raise ValueError('unique frame and consecutive paid steps starting at zero required')
        if len(self._frames) >= 512:
            raise ValueError('bounded 512-frame analytic/runtime evidence history exceeded')
        if not isinstance(accepted_associations, (list, tuple)) or len(accepted_associations) > self.maximum_instances:
            raise ValueError('bounded accepted association sequence required')
        packet_sha = observation.sha256()
        validated, seen = [], set()
        for item in accepted_associations:
            if not isinstance(item, dict) or set(item)-_ASSOCIATION_KEYS:
                raise ValueError('observed association field whitelist violation')
            instance_id = item['instance_id']
            if not isinstance(instance_id, str) or not instance_id or instance_id in seen:
                raise ValueError('unique nonempty observed instance identifier required')
            if (item['observation_sha256'] != packet_sha or item['frame_id'] != observation.frame_id
                    or item['paid_step'] != observation.paid_step):
                raise ValueError('current packet identity/hash mismatch')
            pixels = _pixels(item['pixel_indices'], observation.depth_m.shape)
            marker = _pixels(item.get('marker_pixel_indices', []), observation.depth_m.shape)
            if not np.isin(marker, pixels).all():
                raise ValueError('marker pixels must be inside uniquely associated support')
            depth = observation.depth_m.ravel()[pixels]
            if len(pixels) < 16 or np.any((depth < .1) | (depth > 4.)):
                raise ValueError('at least 16 valid current axial-depth pixels required')
            points = observed_points_v41(observation, pixels)
            supplied = np.asarray(item['points_world_m'], dtype=np.float64)
            if (supplied.shape != points.shape or not np.array_equal(supplied, points)
                    or item['support_sha256'] != support_digest(points)):
                raise ValueError('association support does not equal current measured pixels')
            plane = fit_marker_plane_v41(observation, marker)
            validated.append((instance_id, points, pixels, plane))
            seen.add(instance_id)
        if len(set(self._instances) | seen) > self.maximum_instances:
            raise ValueError('instance observation capacity exceeded')
        # This excludes the caller's ID and step. Repaid unchanged observations
        # are valid actions but cannot manufacture independent support.
        payload = _measurement_digest(observation)
        duplicate = payload in self._payloads
        results = []
        for instance_id, points, pixels, plane in validated:
            state = self._instances.setdefault(instance_id, dict(support={}, plane=None,
                plane_conflict=False, source_frames=[]))
            if not duplicate:
                for point in points:
                    voxel = tuple(np.floor(point/self.support_voxel_m).astype(np.int64))
                    if voxel not in state['support'] and len(state['support']) < self.maximum_support_points:
                        state['support'][voxel] = point.copy()
            if plane['accepted']:
                if state['plane'] is None:
                    state['plane'] = deepcopy(plane)
                elif (np.linalg.norm(np.asarray(plane['anchor_world_m'])-state['plane']['anchor_world_m']) > .15
                        or np.dot(plane['outward_normal_world'], state['plane']['outward_normal_world']) < math.cos(math.radians(20.))):
                    state['plane_conflict'] = True
            source = dict(frame_id=observation.frame_id, paid_step=observation.paid_step,
                observation_sha256=packet_sha, support_sha256=support_digest(points),
                mask_sha256=pixel_mask_sha256_v41(pixels, observation.depth_m.shape))
            state['source_frames'].append(source)
            results.append(dict(instance_id=instance_id, current_plane_fit=plane,
                reliable_cached_plane=state['plane'] is not None and not state['plane_conflict'],
                duplicate_measurement=duplicate, **source))
        self._last_step = observation.paid_step
        self._frames.add(observation.frame_id)
        self._payloads.add(payload)
        camera = dict(intrinsic=observation.intrinsic.tolist(),
            world_from_camera=observation.world_from_camera.tolist(),
            width=int(observation.depth_m.shape[1]), height=int(observation.depth_m.shape[0]))
        self._paid_cameras.append(dict(**camera, camera_geometry_sha256=_digest(camera),
            frame_id=observation.frame_id, paid_step=observation.paid_step,
            observation_sha256=packet_sha))
        self._forecast_cache.clear()
        return dict(schema='v42.observed_view_evidence.v1', paid_step=self._last_step,
                    observation_sha256=packet_sha, results=results,
                    duplicate_measurement=duplicate,
                    ground_truth_used=False, future_observation_used=False)

    def observed_plane_receipts(self):
        return deepcopy({key: dict(plane_fit=value['plane'], conflict=value['plane_conflict'],
                                   source_frames=value['source_frames'])
                         for key, value in self._instances.items()})

    def _snapshot(self, instance):
        if not isinstance(instance, dict) or set(instance) != _SNAPSHOT_KEYS:
            raise ValueError('exact V41 public instance snapshot whitelist required')
        key = instance['instance_id']
        if key not in self._instances:
            raise ValueError('instance has no current observed provenance')
        if instance['structure_names'] != list(STRUCTURES):
            raise ValueError('shared structure bank order mismatch')
        state = self._instances[key]
        points = np.asarray(list(state['support'].values()), dtype=np.float64).reshape(-1, 3)
        if (not np.array_equal(np.asarray(instance['support_points_world_m']), points)
                or instance['support_sha256'] != support_digest(points)):
            raise ValueError('ledger support must equal accumulated paid measured support')
        p, prior, scores = (np.asarray(instance[field], dtype=float) for field in
                           ('structure_probabilities', 'active_structure_prior', 'geometry_log_scores'))
        if any(a.shape != (4,) or not np.isfinite(a).all() for a in (p, prior, scores)):
            raise ValueError('finite shared four-structure belief required')
        if np.any(prior <= 0) or np.any(p <= 0) or not np.isclose(p.sum(), 1., atol=1e-10, rtol=0):
            raise ValueError('positive normalized structure probabilities required')
        if not np.isclose(prior.sum(), 1., atol=1e-10, rtol=0):
            raise ValueError('positive normalized active prior required')
        logits = np.log(prior)+scores
        expected = np.exp(logits-logits.max()); expected /= expected.sum()
        if not np.allclose(p, expected, atol=1e-10, rtol=0):
            raise ValueError('posterior differs from declared observed evidence and active prior')
        return state, points, p

    def _areas(self, state, support, candidate):
        plane = state['plane']
        outward = np.asarray(plane['outward_normal_world'])
        rotation = np.column_stack((np.cross([0., 0., 1.], outward), -outward, [0., 0., 1.]))
        anchor = np.asarray(plane['anchor_world_m'])
        cam_rotation = candidate.world_from_camera[:3, :3]
        origin = candidate.world_from_camera[:3, 3]
        known_tree = cKDTree(support)
        results = []
        for (name, scale, boxes, local_anchor), (local_points, weights, local_normals) in zip(self.bank.candidates, self._samples):
            translation = anchor-rotation@local_anchor
            world_points = local_points@rotation.T+translation
            cam = (world_points-origin)@cam_rotation
            z = cam[:, 2]
            allowed = (z >= candidate.near_m) & (z <= candidate.far_m)
            projection = cam@candidate.intrinsic.T
            with np.errstate(divide='ignore', invalid='ignore'):
                uv = projection[:, :2]/projection[:, 2, None]
            allowed &= ((uv[:, 0] >= -.5) & (uv[:, 0] < candidate.width-.5)
                        & (uv[:, 1] >= -.5) & (uv[:, 1] < candidate.height-.5))
            toward_camera = origin-world_points
            allowed &= np.einsum('ij,ij->i', local_normals@rotation.T, toward_camera) > 1e-10
            selected = np.flatnonzero(allowed)
            visible = np.zeros(len(local_points), dtype=bool)
            if len(selected):
                local_origin = (origin-translation)@rotation
                local_rays = (local_points[selected]-local_origin)/z[selected, None]
                first_depth = _ray_box_depths(local_origin, local_rays, boxes)
                visible[selected] = np.abs(first_depth-z[selected]) <= 1e-6
            distance = known_tree.query(world_points, k=1, workers=1)[0]
            unseen = distance > self.observed_distance_m
            results.append(dict(structure=name, scale=scale,
                visible_area_m2=float(weights[visible].sum()),
                new_surface_area_m2=float(weights[visible & unseen].sum()),
                already_observed_area_m2=float(weights[visible & ~unseen].sum()),
                total_prototype_area_m2=float(weights.sum()),
                sampled_surface_points=len(weights), visible_surface_points=int(visible.sum())))
        return results

    def forecast(self, instance_snapshot, candidates):
        state, support, probabilities = self._snapshot(instance_snapshot)
        if not isinstance(candidates, (list, tuple)) or not 1 <= len(candidates) <= 256:
            raise ValueError('1..256 public candidate views required')
        receipts = [_candidate_receipt(candidate) for candidate in candidates]
        if len({row['view_id'] for row in receipts}) != len(receipts):
            raise ValueError('unique public candidate view identifiers required')
        plane = state['plane']
        reason = ('association_uncertain' if instance_snapshot['association_uncertain'] else
                  'conflicting_observed_label_planes' if state['plane_conflict'] else
                  'no_reliable_observed_label_plane' if plane is None else None)
        geometry_receipt = dict(instance_id=instance_snapshot['instance_id'],
            support_sha256=support_digest(support), plane_fit=plane,
            source_frames=state['source_frames'], configuration_sha256=self.configuration_sha256,
            paid_camera_history_sha256=_digest(self._paid_cameras))
        geometry_sha = _digest(geometry_receipt)
        rows = []
        for candidate, receipt in zip(candidates, receipts):
            candidate_sha = _digest(receipt)
            if reason is None:
                cache_key = (geometry_sha, candidate_sha)
                if cache_key not in self._forecast_cache:
                    if len(self._forecast_cache) >= 1024:
                        self._forecast_cache.clear()
                    self._forecast_cache[cache_key] = self._areas(state, support, candidate)
                components = deepcopy(self._forecast_cache[cache_key])
                new_areas = [float(np.mean([r['new_surface_area_m2'] for r in components if r['structure'] == name]))
                             for name in STRUCTURES]
                visible_areas = [float(np.mean([r['visible_area_m2'] for r in components if r['structure'] == name]))
                                 for name in STRUCTURES]
            else:
                components, new_areas, visible_areas = [], [0.]*4, [0.]*4
            # This is an eligibility mask, not evidence that every predicted
            # surface was actually measured. Missing pixels remain missing.
            # The forecast has no stochastic/repeated-view precision model,
            # so nominal mismatch cannot justify paying the identical view
            # indefinitely. Turns and translations remain eligible.
            matched = [dict(frame_id=paid['frame_id'], paid_step=paid['paid_step'],
                observation_sha256=paid['observation_sha256'], camera_geometry_sha256=paid['camera_geometry_sha256'])
                for paid in self._paid_cameras
                if paid['width'] == candidate.width and paid['height'] == candidate.height
                and np.allclose(paid['intrinsic'], candidate.intrinsic, atol=1e-9, rtol=0)
                and np.allclose(paid['world_from_camera'], candidate.world_from_camera, atol=1e-9, rtol=0)]
            effective_areas = [0.]*4 if matched else new_areas
            rows.append(dict(view_id=candidate.view_id, candidate=receipt, candidate_sha256=candidate_sha,
                expected_new_surface_area_m2=float(probabilities@np.asarray(effective_areas)),
                structure_new_surface_area_m2=effective_areas, structure_visible_area_m2=visible_areas,
                unexcluded_structure_new_surface_area_m2=new_areas,
                unexcluded_expected_new_surface_area_m2=float(probabilities@np.asarray(new_areas)),
                repeated_view_excluded=bool(matched), matching_paid_views=matched,
                component_areas_before_view_exclusion=True,
                components=components, fallback=reason is not None, fallback_reason=reason))
        return dict(schema='v42.observed_posterior_view_quality.v1', ans_module='STGHP',
            instance_id=instance_snapshot['instance_id'], paid_step=self._last_step,
            structure_names=list(STRUCTURES), structure_probabilities=probabilities.tolist(),
            semantic_conditioning_used=bool(instance_snapshot['semantic_conditioning_used']),
            instance_snapshot_sha256=_digest(instance_snapshot),
            geometry_evidence_sha256=geometry_sha, observed_geometry=geometry_receipt,
            prototype_bank_sha256=self.bank.sha256, configuration=deepcopy(self.configuration),
            configuration_sha256=self.configuration_sha256, candidates=rows,
            metric='posterior expected newly visible nominal exterior surface area in square metres',
            known_surface_rule='distance to accumulated measured support <= observed_distance_m',
            repeated_view_exclusion_is_not_measured_surface_evidence=True,
            scale_posterior_calibrated=False, pose_uncertainty_marginalized=False,
            external_occlusion_modeled=False, candidate_navigation_checked=False,
            observation_resolution_or_noise_gain_modeled=False,
            ground_truth_used=False, future_observation_used=False,
            class_importance_weights_used=False, tsdf_quality_measured=False,
            limitation='common upright label mount and nominal shape family; external occlusion, pose uncertainty, and repeated-view accuracy gain are not modeled')
