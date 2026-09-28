"""Observed-surface precision proxy for common V19 geometric planning.

This reads measured quality keys and the reconstructed TSDF mesh only. It is
independent of the hypothetical hidden-plane aperture used for completeness.
It predicts independent scalar-depth information over actual future route
frames, not calibrated TSDF F1 or guaranteed visibility behind unknown space.
"""
import numpy as np
import open3d as o3d
from scipy.special import erf

from env.virtual3d import camera_pose


class ObservedPrecisionResponseV19:
    def __init__(self, mapper, assets, *, intrinsic=None, max_points_per_asset=128,
                 occlusion_tolerance_m=.12):
        if isinstance(max_points_per_asset, bool) or int(max_points_per_asset) != max_points_per_asset or max_points_per_asset < 1:
            raise ValueError('positive integer support cap required')
        if not np.isfinite(occlusion_tolerance_m) or occlusion_tolerance_m < 0:
            raise ValueError('nonnegative finite observed-mesh tolerance required')
        self.config = mapper.config; self.shape = mapper.shape
        self.occlusion_tolerance_m = float(occlusion_tolerance_m)
        sigma = {'ideal': 0., 'iid_010px': .10, 'iid_025px': .25,
                 'bias_025px_iid_025px': .25}
        if self.config.stereo_model not in sigma:
            raise ValueError('undeclared stereo model for precision proxy')
        self.sigma_px = sigma[self.config.stereo_model]
        self.fb = float(self.config.stereo_reference_fx_px * self.config.stereo_baseline_m)
        if not np.isfinite(self.fb) or self.fb <= 0:
            raise ValueError('positive finite stereo focal-baseline product required')
        if intrinsic is None:
            c = self.config; focal = c.width_px / (2 * np.tan(np.deg2rad(c.fov_deg / 2)))
            intrinsic = [[focal, 0., (c.width_px - 1) / 2], [0., focal, (c.height_px - 1) / 2], [0., 0., 1.]]
        self.intrinsic = np.asarray(intrinsic, float)
        if self.intrinsic.shape != (3, 3) or not np.isfinite(self.intrinsic).all() or min(self.intrinsic[0, 0], self.intrinsic[1, 1]) <= 0:
            raise ValueError('valid finite camera intrinsics required')
        quality = list(mapper.quality.values())
        points = np.asarray([q['point'] for q in quality]) if quality else np.empty((0, 3))
        if points.shape != (len(quality), 3) or not np.isfinite(points).all():
            raise ValueError('finite three-dimensional measured quality points required')
        selected = []; self.slices = []
        for asset in assets:
            start = len(selected)
            if asset['marked_points'] > 0:
                low, high = np.asarray(asset['observed_low']), np.asarray(asset['observed_high'])
                ids = np.flatnonzero(((points >= low - .15) & (points <= high + .15)).all(axis=1))
                if len(ids) > max_points_per_asset:
                    ids = ids[np.linspace(0, len(ids) - 1, int(max_points_per_asset), dtype=int)]
                selected.extend(quality[i] for i in ids)
            self.slices.append(slice(start, len(selected)))
        self.points = np.asarray([q['point'] for q in selected], float).reshape(-1, 3)
        ranges = np.asarray([q['best_range'] for q in selected], float)
        counts = np.asarray([q['n'] for q in selected], float)
        if not np.isfinite(ranges).all() or not np.isfinite(counts).all() or np.any(ranges <= 0) or np.any(counts < 1):
            raise ValueError('measured support requires positive finite ranges and counts')
        prior_sigma = np.maximum(.002, self.sigma_px * ranges ** 2 / self.fb)
        self.prior_information = counts / prior_sigma ** 2
        self.prior_precision = erf(.05 * np.sqrt(self.prior_information / 2.))
        self.ray = None
        mesh = mapper.mesh()
        if len(mesh.triangles):
            self.ray = o3d.t.geometry.RaycastingScene(nthreads=1)
            self.ray.add_triangles(o3d.t.geometry.TriangleMesh.from_legacy(mesh))
        self.pose_cache = {}

    def _pose_response(self, raw_pose):
        pose_key = tuple(int(x) for x in raw_pose)
        if len(pose_key) != 3 or any(float(x) != y for x, y in zip(raw_pose, pose_key)) or not 0 <= pose_key[2] < 4:
            raise ValueError('route poses require discrete row, column, heading')
        if not 0 <= pose_key[0] < self.shape[0] or not 0 <= pose_key[1] < self.shape[1]:
            raise ValueError('route pose outside mapper grid')
        if pose_key in self.pose_cache:
            return self.pose_cache[pose_key]
        pose = camera_pose(pose_key[:2], pose_key[2], self.config, self.shape[0])
        delta = self.points - pose[:3, 3]
        local = delta @ pose[:3, :3]; z = local[:, 2]
        u = local[:, 0] / np.maximum(z, .001) * self.intrinsic[0, 0] + self.intrinsic[0, 2]
        v = local[:, 1] / np.maximum(z, .001) * self.intrinsic[1, 1] + self.intrinsic[1, 2]
        possible = ((z > .15) & (z <= self.config.max_depth_m) & (u >= -.5)
                    & (u < self.config.width_px - .5) & (v >= -.5) & (v < self.config.height_px - .5))
        ids = np.flatnonzero(possible)
        blocked = np.zeros(len(self.points), bool); no_hit = np.zeros(len(self.points), bool)
        if len(ids) and self.ray is not None:
            rays = np.column_stack([np.tile(pose[:3, 3], (len(ids), 1)), delta[ids]]).astype(np.float32)
            hit = self.ray.cast_rays(o3d.core.Tensor(rays), nthreads=1)['t_hit'].numpy()
            finite = np.isfinite(hit)
            blocked[ids] = finite & ((1. - hit) * z[ids] > self.occlusion_tolerance_m)
            no_hit[ids] = ~finite
        elif len(ids):
            no_hit[ids] = True
        visible = possible & ~blocked
        information = np.zeros(len(self.points))
        information[visible] = 1. / np.maximum(.002, self.sigma_px * z[visible] ** 2 / self.fb) ** 2
        result = dict(information=information, visible=visible, blocked=blocked, no_hit=no_hit)
        self.pose_cache[pose_key] = result
        return result

    def score_route(self, route):
        """Accumulate one new acquisition per paid outbound state after start."""
        states = route['outbound_states']
        if not len(states):
            raise ValueError('route must contain its current start state')
        information = self.prior_information.copy()
        measurements = np.zeros(len(self.points), int)
        blocked_count = unknown_count = 0
        for pose in states[1:]:
            response = self._pose_response(pose)
            information += response['information']
            measurements += response['visible']
            blocked_count += int(response['blocked'].sum())
            unknown_count += int(response['no_hit'].sum())
        improvement = np.maximum(0., erf(.05 * np.sqrt(information / 2.)) - self.prior_precision)
        gains = [float(improvement[part].mean()) if part.stop > part.start else 0. for part in self.slices]
        return dict(per_asset_expected_precision_change=gains,
            per_asset_support_counts=[part.stop - part.start for part in self.slices],
            per_asset_predicted_measurements=[int(measurements[part].sum()) for part in self.slices],
            paid_future_observations=len(states) - 1, known_occluded_point_frames=blocked_count,
            no_known_hit_point_frames=unknown_count, occlusion_tolerance_m=self.occlusion_tolerance_m,
            threshold_m=.05, depth_sigma_floor_m=.002, class_used=False, evaluation_truth_used=False,
            hidden_aperture_used=False, calibrated_future_tsdf_quality=False,
            covariance_assumption='independent scalar depth approximation; historical best Euclidean range/count proxy, future axial depth',
            systematic_disparity_bias_modelled=False,
            systematic_bias_sensitivity_active=self.config.stereo_model.startswith('bias_'),
            unknown_rays_are_visibility_certificates=False)
