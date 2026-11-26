"""Observed same-category inspection baseline for the native same-goal pool."""

import numpy as np
import open3d as o3d


LABELS = frozenset((2, 3))
VOXEL_M = .15
OCCLUSION_M = .12
QUALITY_SCALE_M_INV2 = .25
AZIMUTHS = 8


class Inspection:
    """Score opportunity to revisit already measured fixture surfels.

    Uses observed classes only to identify fixtures belonging to the comparison.
    Both classes receive the same rule. No depth, occupancy or CAD from the
    simulator enters this predictor. Current measured TSDF may veto a view.
    """

    def __init__(self, mapper, packet, ray):
        from env.virtual3d import camera_pose

        self.camera_pose = camera_pose
        self.config = mapper.config
        self.shape = mapper.shape
        self.intrinsic = np.asarray(packet.frame.intrinsic, dtype=float)
        self.ray = ray
        self.view_cache = {}
        evidence = [(key, row) for key, row in sorted(mapper.quality.items()) if row['label'] in LABELS]
        tracks = mapper.instance_support_index.memberships_for_keys(np.asarray([key for key, _ in evidence], dtype=np.int64).reshape(-1, 3)) if evidence else ()
        paired = [(row, int(track)) for (_, row), track in zip(evidence, tracks, strict=True) if track > 0]
        self.points = np.asarray([row['point'] for row, _ in paired], dtype=float).reshape(-1, 3)
        self.normals = np.asarray([row['normal'] for row, _ in paired], dtype=float).reshape(-1, 3)
        self.bits = np.asarray([row['bits'] for row, _ in paired], dtype=np.int64)
        self.opportunity = np.asarray([1. - int(bits).bit_count() / AZIMUTHS for bits in self.bits])
        information = np.asarray([row['information'] for row, _ in paired])
        self.old_quality = information / (information + QUALITY_SCALE_M_INV2)
        self.categories = {int(label): sum(int(row['label'] == label) for row, _ in paired) for label in sorted(LABELS)}
        self.track_count = len({track for _, track in paired})

    def _pose(self, raw_pose):
        pose = tuple(map(int, raw_pose))
        if pose in self.view_cache:
            return self.view_cache[pose]
        out = np.zeros(len(self.points))
        transform = self.camera_pose(pose[:2], pose[2], self.config, self.shape[0])
        center = transform[:3, 3]
        delta = self.points - center
        local = delta @ transform[:3, :3]
        z = local[:, 2]
        intr = self.intrinsic
        u = local[:, 0] / np.maximum(z, .001) * intr[0, 0] + intr[0, 2]
        v = local[:, 1] / np.maximum(z, .001) * intr[1, 1] + intr[1, 2]
        ids = np.flatnonzero((z > .15) & (z <= self.config.max_depth_m)
                             & (u >= -.5) & (u < self.config.width_px - .5)
                             & (v >= -.5) & (v < self.config.height_px - .5))
        if len(ids) and self.ray is not None:
            rays = np.column_stack((np.broadcast_to(center, (len(ids), 3)), delta[ids])).astype(np.float32)
            hit = self.ray.cast_rays(o3d.core.Tensor(rays), nthreads=1)['t_hit'].numpy()
            ids = ids[~np.isfinite(hit) | ((1. - hit) * z[ids] <= OCCLUSION_M)]
        if len(ids):
            direction = -delta[ids]
            squared = np.einsum('ij,ij->i', direction, direction)
            incidence = np.abs(np.einsum('ij,ij->i', self.normals[ids], direction)) / np.sqrt(squared)
            new_info = incidence ** 2 / np.maximum(squared, .25)
            quality = new_info / (new_info + QUALITY_SCALE_M_INV2)
            angles = np.arctan2(direction[:, 1], direction[:, 0])
            azimuth = np.floor((angles + np.pi) * AZIMUTHS / (2 * np.pi)).astype(int) % AZIMUTHS
            unseen = ((self.bits[ids] >> azimuth) & 1) == 0
            angular = unseen * self.opportunity[ids] * quality
            quality_increment = np.maximum(0., quality - self.old_quality[ids])
            out[ids] = np.maximum(angular, quality_increment)
        self.view_cache[pose] = out
        return out

    def score(self, states):
        best = np.zeros(len(self.points))
        for pose in states[1:]:
            np.maximum(best, self._pose(pose), out=best)
        return float(best.sum() * VOXEL_M ** 2)


def rank_direct(state, rows, weights, scale, tolerance, rank_admitted):
    """Replace only the forecast area. Delegate the shared ranking and ties."""
    mapper = state['mapper']
    ray = mapper._dual_template_state['cache']['ray']
    inspection = Inspection(mapper, state['packet'], ray)
    shares = []
    for row in rows:
        if row['admitted']:
            value = inspection.score(row['outbound_states']) if len(inspection.points) else 0.
            shares.append(dict(row, scores={'D': float(scale) * value}, direct_unscaled_m2=value))
    native = next(row for row in shares if row['route_id'] == 'native')
    winner, ranking = rank_admitted(shares, native, 'D', weights, float(mapper.config.resolution_m) ** 2, tolerance)
    return winner, dict(scale=scale, observed_support_voxels=len(inspection.points), support_track_count=inspection.track_count,
                        observed_category_counts=inspection.categories, values=[row['direct_unscaled_m2'] for row in shares],
                        scored_pool=ranking, selected_route_id=winner['route_id'])
