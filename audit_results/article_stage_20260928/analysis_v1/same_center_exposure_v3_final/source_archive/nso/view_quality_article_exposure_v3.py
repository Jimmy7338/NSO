"""Same-optical-center nominal exposure de-duplication for the article planner.

This changes a planning eligibility proxy only. Previously attempted nominal
surface samples are not asserted to have been measured or reconstructed. All
association, support, planes, shape priors, scales and physical mapping remain
owned by the unchanged ArticleV1 implementation.
"""
import hashlib
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree

from nso.controller_article_v1 import ArticleViewPredictorV1
from nso.observed_residual_v41 import _ray_box_depths
from nso.view_quality_v42 import _digest


OPTICAL_CENTER_ATOL_M = 1e-9


def nominal_visible_mask(local_points, local_normals, boxes, rotation,
                         translation, intrinsic, world_from_camera,
                         width, height, near_m=.1, far_m=4.):
    """Use the frozen V42 frustum, facing and prototype self-occlusion rules.

    No actual or future scene is rendered. The mask indexes the same fixed
    quadrature samples as the original nominal-area predictor.
    """
    world_points = local_points @ rotation.T + translation
    origin = world_from_camera[:3, 3]
    camera_points = (world_points - origin) @ world_from_camera[:3, :3]
    z = camera_points[:, 2]
    allowed = (z >= near_m) & (z <= far_m)
    projection = camera_points @ intrinsic.T
    with np.errstate(divide='ignore', invalid='ignore'):
        uv = projection[:, :2] / projection[:, 2, None]
    allowed &= ((uv[:, 0] >= -.5) & (uv[:, 0] < width-.5)
                & (uv[:, 1] >= -.5) & (uv[:, 1] < height-.5))
    allowed &= np.einsum('ij,ij->i', local_normals @ rotation.T,
                         origin-world_points) > 1e-10
    selected = np.flatnonzero(allowed)
    visible = np.zeros(len(local_points), dtype=bool)
    if len(selected):
        local_origin = (origin-translation) @ rotation
        local_rays = (local_points[selected]-local_origin) / z[selected, None]
        first_depth = _ray_box_depths(local_origin, local_rays, boxes)
        visible[selected] = np.abs(first_depth-z[selected]) <= 1e-6
    return visible


class ArticleExposureViewPredictorV3(ArticleViewPredictorV1):
    """Drop already paid angular exposure at the same optical center only.

    Construction has the same ``residual=ArticleObservedResidualV1(...)`` and
    remaining keyword arguments as ArticleViewPredictorV1. All methods must
    use this common predictor. A translated camera remains eligible, as do
    samples outside every previous same-center frustum/depth/self-visible set.
    """

    def __init__(self, *, residual, **kwargs):
        super().__init__(residual=residual, **kwargs)
        self._exposure_mask_cache = {}
        base_source = self.source_sha256
        self.source_sha256 = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
        self.configuration.update(
            article_exposure_version='article.same_center_paid_exposure.v3',
            source_sha256=self.source_sha256,
            frozen_v42_source_sha256=base_source,
            same_optical_center_atol_m=OPTICAL_CENTER_ATOL_M,
            exposure_camera_compatibility='same intrinsic and image shape; current fixed axial range',
            exposure_rule='candidate self-visible samples minus union of previous same-optical-center self-visible samples',
            exposure_scope='paid nominal opportunity eligibility, not measured reconstruction',
            exposure_history_includes_initial_frame=True,
            translated_view_exclusion=False,
            missing_depth_recovery_or_repeated_precision_gain_modeled=False)
        self.configuration_sha256 = _digest(self.configuration)

    def observe(self, observation, accepted_associations):
        result = super().observe(observation, accepted_associations)
        self._exposure_mask_cache.clear()
        return result

    def _same_center_history(self, candidate):
        compatible = []
        unique = set()
        for paid in self._paid_cameras:
            transform = np.asarray(paid['world_from_camera'])
            if (paid['width'] != candidate.width or paid['height'] != candidate.height
                    or not np.allclose(paid['intrinsic'], candidate.intrinsic,
                                       atol=1e-9, rtol=0)
                    or not np.allclose(transform[:3, 3],
                                       candidate.world_from_camera[:3, 3],
                                       atol=OPTICAL_CENTER_ATOL_M, rtol=0)):
                continue
            # Repeated frames do not increase a set union or its computation.
            identity = paid['camera_geometry_sha256']
            if identity not in unique:
                compatible.append(paid)
                unique.add(identity)
        return compatible

    def _areas(self, state, support, candidate):
        rows = super()._areas(state, support, candidate)
        history = self._same_center_history(candidate)
        plane = state['plane']
        outward = np.asarray(plane['outward_normal_world'])
        rotation = np.column_stack((np.cross([0., 0., 1.], outward),
                                    -outward, [0., 0., 1.]))
        anchor = np.asarray(plane['anchor_world_m'])
        known_tree = cKDTree(support)
        geometry = _digest(dict(plane=plane, prototype_bank=self.bank.sha256,
                                history=[p['camera_geometry_sha256'] for p in history]))
        for index, (row, prototype, samples) in enumerate(
                zip(rows, self.bank.candidates, self._samples)):
            _, _, boxes, local_anchor = prototype
            local_points, weights, local_normals = samples
            before = row['new_surface_area_m2']
            removed = 0.
            if history:
                translation = anchor-rotation @ local_anchor
                cache_key = (geometry, index)
                if cache_key not in self._exposure_mask_cache:
                    union = np.zeros(len(local_points), dtype=bool)
                    for paid in history:
                        union |= nominal_visible_mask(local_points, local_normals,
                            boxes, rotation, translation, np.asarray(paid['intrinsic']),
                            np.asarray(paid['world_from_camera']), paid['width'],
                            paid['height'], candidate.near_m, candidate.far_m)
                    if len(self._exposure_mask_cache) >= 1024:
                        self._exposure_mask_cache.clear()
                    self._exposure_mask_cache[cache_key] = union
                union = self._exposure_mask_cache[cache_key]
                visible = nominal_visible_mask(local_points, local_normals, boxes,
                    rotation, translation, candidate.intrinsic,
                    candidate.world_from_camera, candidate.width, candidate.height,
                    candidate.near_m, candidate.far_m)
                world_points = local_points @ rotation.T + translation
                unseen = known_tree.query(world_points, k=1, workers=1)[0] > self.observed_distance_m
                removed = float(weights[visible & unseen & union].sum())
                row['new_surface_area_m2'] = float(weights[visible & unseen & ~union].sum())
            row.update(new_surface_area_before_exposure_m2=before,
                excluded_previously_exposed_unseen_area_m2=removed,
                same_center_distinct_paid_views=len(history),
                exposure_exclusion_is_not_measured_surface=True)
        return rows

    def forecast(self, instance_snapshot, candidates):
        result = super().forecast(instance_snapshot, candidates)
        result.update(
            exposure_version='article.same_center_paid_exposure.v3',
            metric='posterior expected nominal exterior area eligible after same-center paid exposure de-duplication',
            measured_support_rule_unchanged=True,
            exposure_exclusion_is_not_measured_surface=True,
            actual_depth_and_tsdf_modified=False,
            limitation=result['limitation'] + '; same-center attempted exposure is only an eligibility proxy; missing-depth recovery is not predicted')
        return result
