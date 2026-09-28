"""Recover novel paid geometry evidence after association-cache saturation.

This version leaves the frozen local association and category-registration
pipeline unchanged. A full point cache is a memory bound, not a reason to
discard all later geometric measurements. The narrow fallback below admits a
novel, uniquely associated current measurement only from a sufficiently new
view, and only within a documented bound on successful-feedback count.
This reuses the numeric frame cap but counts successful updates, whereas the
legacy cap counts observations that enlarge the association support cache.

The view separation is a conservative evidence-accounting heuristic, not a
claim of statistically independent or calibrated likelihoods. G/B/S must use
the same frontend. No simulator, template truth, or evaluator is accessed.
"""
from copy import deepcopy
import math

import numpy as np

from nso.observed_instances_local import ObservedInstancesLocal


class ObservedInstancesArticleV1(ObservedInstancesLocal):
    """Only rescue cache-blocked evidence; preserve all earlier behavior."""

    def __init__(self, *, rescue_translation_m=.25, rescue_rotation_deg=30., **kwargs):
        for name, value in [('rescue_translation_m', rescue_translation_m),
                            ('rescue_rotation_deg', rescue_rotation_deg)]:
            if isinstance(value, bool) or not math.isfinite(value) or value <= 0:
                raise ValueError(name + ' must be positive finite')
        if rescue_rotation_deg > 180:
            raise ValueError('rescue_rotation_deg must not exceed 180')
        super().__init__(**kwargs)
        self.rescue_translation_m = float(rescue_translation_m)
        self.rescue_rotation_deg = float(rescue_rotation_deg)
        self._applied_views = {}
        self._rescued_counts = {}
        self._current_evidence_pose = None

    def _new_evidence_view(self, key, origin, axis):
        for row in self._applied_views.get(key, []):
            distance = float(np.linalg.norm(origin - row['origin']))
            angle = math.degrees(math.acos(float(np.clip(np.dot(axis, row['axis']), -1., 1.))))
            if (distance < self.rescue_translation_m - 1e-9
                    and angle < self.rescue_rotation_deg - 1e-9):
                return False
        return True

    def observe(self, observation):
        receipt = super().observe(observation)
        origin = observation.world_from_camera[:3, 3].copy()
        axis = observation.world_from_camera[:3, 2].copy()
        self._current_evidence_pose = (observation.frame_id, origin, axis)
        for row in receipt['accepted']:
            key = row['instance_id']
            state = self._instances[key]
            association = self._frames[observation.frame_id]['instances'][key]
            legacy = bool(row['geometry_feedback_eligible'])
            saturated = len(state['support']) >= self.maximum_support_points
            completed = len(state['geometry_views'])
            new_view = self._new_evidence_view(key, origin, axis)
            fraction = None
            reason = 'legacy_eligibility_preserved'
            rescued = False
            if not legacy:
                if row['duplicate_measurement']:
                    reason = 'duplicate_payload'
                elif not saturated or not row['no_new_support']:
                    reason = 'not_blocked_by_full_association_cache'
                elif completed >= self.maximum_evidence_frames:
                    reason = 'successful_feedback_capacity'
                elif not new_view:
                    reason = 'previously_used_or_nearby_feedback_view'
                else:
                    points = np.asarray(row['points_world_m'], dtype=float)
                    novel = self._distances(points, self._support(state)) > self.voxel_size_m
                    fraction = float(novel.mean())
                    if (row['novel_support_voxels'] < self.minimum_novel_points
                            or fraction < self.minimum_novel_fraction):
                        reason = 'insufficient_novel_measured_support'
                    else:
                        rescued = True
                        reason = 'novel_paid_view_after_association_cache_full'
            audit = dict(policy='article_cache_independent_geometry_v1',
                         legacy_eligible=legacy, association_cache_full=saturated,
                         legacy_novel_support_frame_count=state['evidence_frames'],
                         legacy_support_frame_capacity_exhausted=(
                             state['evidence_frames'] >= self.maximum_evidence_frames),
                         successful_feedback_count_before=completed,
                         successful_feedback_capacity=self.maximum_evidence_frames,
                         new_feedback_view=bool(new_view), novel_fraction=fraction,
                         rescue_translation_m=self.rescue_translation_m,
                         rescue_rotation_deg=self.rescue_rotation_deg,
                         cache_rescue_eligible=rescued, reason=reason)
            association['geometry_feedback_eligible'] = legacy or rescued
            association['article_feedback_eligibility'] = deepcopy(audit)
            row['geometry_feedback_eligible'] = legacy or rescued
            row['article_feedback_eligibility'] = audit
        receipt['evidence_policy'] = 'article_cache_independent_geometry_v1'
        return receipt

    def apply_geometry_feedback(self, instance_id, *, frame_id, observation_sha256,
                                log_likelihoods):
        # Keep validation ahead of any eligibility mutation.
        if instance_id not in self._instances or frame_id not in self._frames:
            raise ValueError('known instance and current paid frame required')
        frame = self._frames[frame_id]
        if frame['step'] != self._last_step or frame['observation_sha256'] != observation_sha256:
            raise ValueError('current paid frame and exact observation SHA required')
        if instance_id not in frame['instances']:
            raise ValueError('uniquely associated current measurement required')
        scores = np.asarray(log_likelihoods, dtype=float)
        if scores.shape != (len(self.names),) or not np.isfinite(scores).all():
            raise ValueError('finite per-structure paid geometry scores required')
        association = frame['instances'][instance_id]
        audit = association['article_feedback_eligibility']
        uninformative_rescue = audit['cache_rescue_eligible'] and np.ptp(scores) <= 1e-6
        if uninformative_rescue:
            association['geometry_feedback_eligible'] = False
        result = super().apply_geometry_feedback(instance_id, frame_id=frame_id,
            observation_sha256=observation_sha256, log_likelihoods=scores)
        if result['applied']:
            stored_frame, origin, axis = self._current_evidence_pose
            if stored_frame != frame_id:
                raise RuntimeError('feedback pose does not belong to current paid frame')
            self._applied_views.setdefault(instance_id, []).append(
                dict(origin=origin.copy(), axis=axis.copy(), frame_id=frame_id))
            if audit['cache_rescue_eligible']:
                self._rescued_counts[instance_id] = self._rescued_counts.get(instance_id, 0) + 1
                result['reason'] = 'novel_paid_geometry_applied_after_cache_full'
        elif uninformative_rescue:
            result['reason'] = 'uninformative_cache_rescue_not_accumulated'
        result['article_feedback_eligibility'] = deepcopy(audit)
        return result

    def snapshot(self):
        result = super().snapshot()
        result['evidence_policy'] = dict(
            name='article_cache_independent_geometry_v1',
            association_and_category_registration_unchanged=True,
            rescue_requires_novel_paid_support=True,
            rescue_translation_m=self.rescue_translation_m,
            rescue_rotation_deg=self.rescue_rotation_deg,
            successful_feedback_capacity=self.maximum_evidence_frames,
            rescued_feedback_counts=dict(self._rescued_counts),
            probability_calibrated=False)
        return result
