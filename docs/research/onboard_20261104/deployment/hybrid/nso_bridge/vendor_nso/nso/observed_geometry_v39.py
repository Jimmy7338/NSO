"""Paid-depth surfel ledger and sparse observed-only view forecasts.

This module imports no renderer, mapper, mesh evaluator or public CAD surfaces.
Visibility is a sparse measured-point z-buffer approximation, not knowledge of
unobserved free space. Boundaries include sensor/FOV discontinuities, not holes.
"""
from types import MappingProxyType
import math
import numpy as np
from nso.cpu_four_modules_v35 import ObservationV35


def _readonly(value):
    out = np.array(value, copy=True)
    out.flags.writeable = False
    return out


def _calibration(intrinsic, world_from_camera):
    k, t = np.asarray(intrinsic, float), np.asarray(world_from_camera, float)
    if k.shape != (3, 3) or t.shape != (4, 4) or not np.isfinite(k).all() or not np.isfinite(t).all():
        raise ValueError('finite 3x3 intrinsics and 4x4 observed camera pose required')
    if k[0, 0] <= 0 or k[1, 1] <= 0 or not np.allclose(k[2], [0, 0, 1]) or abs(k[0, 1])+abs(k[1, 0]) > 1e-12:
        raise ValueError('positive zero-skew pinhole calibration required')
    r = t[:3, :3]
    if not np.allclose(r.T @ r, np.eye(3), atol=1e-7) or not np.isclose(np.linalg.det(r), 1., atol=1e-7) or not np.allclose(t[3], [0, 0, 0, 1]):
        raise ValueError('rigid observed camera pose required')
    return k, t


def _origin_key(origin):
    return tuple(np.round(origin, 9))


class ObservedGeometryV39:
    """One representative per observed voxel; no unobserved surfaces invented.

    Eight directions are retained in first-distinct-observation order. All
    observed camera centers remain in a set, so revisiting a discarded center
    cannot earn angular novelty. Counts do not represent calibrated certainty.
    """
    def __init__(self, public_bounds=None, *, voxel_m=.1, stride=4,
                 max_directions=8, max_range_m=4., normal_absolute_jump_m=.08,
                 normal_relative_jump=.05):
        numeric = (voxel_m, max_range_m, normal_absolute_jump_m, normal_relative_jump)
        if not np.isfinite(numeric).all() or min(numeric) <= 0:
            raise ValueError('positive finite geometry configuration required')
        if type(stride) is not int or stride < 1 or type(max_directions) is not int or max_directions < 1:
            raise ValueError('positive integer stride and direction cap required')
        self.voxel_m, self.stride = float(voxel_m), stride
        self.max_directions, self.max_range_m = max_directions, float(max_range_m)
        self.normal_absolute_jump_m = float(normal_absolute_jump_m)
        self.normal_relative_jump = float(normal_relative_jump)
        self.public_bounds = None
        if public_bounds is not None:
            bounds = np.asarray(public_bounds, float)
            if bounds.shape != (2, 3) or not np.isfinite(bounds).all() or np.any(bounds[1] <= bounds[0]):
                raise ValueError('finite public min/max bounds required')
            self.public_bounds = _readonly(bounds)
        self._cells = {}
        self._frames = set()
        self._last_step = -1
        self._shape = None
        self._snapshot = None

    def observe(self, observation, intrinsic, world_from_camera):
        if type(observation) is not ObservationV35:
            raise TypeError('sanitized paid ObservationV35 required')
        k, t = _calibration(intrinsic, world_from_camera)
        if observation.frame_id in self._frames or observation.step <= self._last_step:
            raise ValueError('new frame and strictly increasing paid step required')
        d = np.asarray(observation.depth, float)
        if self._shape is not None and d.shape != self._shape:
            raise ValueError('image shape must remain fixed')
        height, width = d.shape
        yy, xx = np.mgrid[:height, :width]
        local = np.stack(((xx-k[0, 2])*d/k[0, 0], (yy-k[1, 2])*d/k[1, 1], d), axis=-1)
        measured = (d > 0) & (np.linalg.norm(local, axis=-1) <= self.max_range_m+1e-7)
        normal_valid = np.zeros(d.shape, bool)
        normals = np.zeros(local.shape, float)
        boundary = measured.copy()
        if height >= 3 and width >= 3:
            center = d[1:-1, 1:-1]
            neighbors = (d[:-2, 1:-1], d[2:, 1:-1], d[1:-1, :-2], d[1:-1, 2:])
            support = measured[1:-1, 1:-1].copy()
            limit = np.maximum(self.normal_absolute_jump_m, self.normal_relative_jump*center)
            for offset, neighbor in zip((measured[:-2, 1:-1], measured[2:, 1:-1], measured[1:-1, :-2], measured[1:-1, 2:]), neighbors):
                support &= offset & (np.abs(neighbor-center) <= limit)
            raw = np.cross(local[1:-1, 2:]-local[1:-1, :-2], local[2:, 1:-1]-local[:-2, 1:-1])
            length = np.linalg.norm(raw, axis=-1)
            support &= length > 1e-10
            normals[1:-1, 1:-1] = np.divide(raw, length[..., None], out=np.zeros_like(raw), where=length[..., None] > 1e-10)
            normal_valid[1:-1, 1:-1] = support
            boundary[1:-1, 1:-1] = measured[1:-1, 1:-1] & ~support
        sampled = measured & ((yy % self.stride) == 0) & ((xx % self.stride) == 0)
        rows, cols = np.nonzero(sampled)
        points = local[rows, cols] @ t[:3, :3].T + t[:3, 3]
        ns = normals[rows, cols] @ t[:3, :3].T
        nv, bd = normal_valid[rows, cols], boundary[rows, cols]
        if self.public_bounds is not None:
            keep = np.all((points >= self.public_bounds[0]) & (points <= self.public_bounds[1]), axis=1)
            points, ns, nv, bd = points[keep], ns[keep], nv[keep], bd[keep]
        keys = np.floor(points/self.voxel_m).astype(np.int64)
        _, indices = np.unique(keys, axis=0, return_index=True)
        origin = t[:3, 3]
        center_key = _origin_key(origin)
        added = 0
        for i in indices:
            key, point = tuple(keys[i]), points[i]
            delta = origin-point
            distance = float(np.linalg.norm(delta))
            direction = delta/distance
            normal = ns[i].copy()
            if nv[i] and normal @ direction < 0:
                normal *= -1
            old = self._cells.get(key)
            if old is None:
                old = dict(point=point.copy(), point_count=0, normal=np.zeros(3), normal_count=0,
                    boundary=False, poses=set(), centers=set(), directions=[], ranges=[], focals=[],
                    inspection_records=[], best_range=distance)
                self._cells[key] = old
                added += 1
            old['point_count'] += 1
            old['point'] += (point-old['point'])/old['point_count']
            if nv[i]:
                quality = (distance, abs(float(normal @ direction)), float(k[0, 0]*k[1, 1]))
                # Preserve nondominated actually measured inspection quality,
                # independently of the capped angular-history approximation.
                # (shorter range, larger incidence cosine, larger focal product)
                dominates = lambda a, b: a[0] <= b[0] and a[1] >= b[1] and a[2] >= b[2]
                if not any(dominates(q, quality) for q in old['inspection_records']):
                    old['inspection_records'] = [q for q in old['inspection_records'] if not dominates(quality, q)] + [quality]
                if old['normal_count'] and normal @ old['normal'] < 0:
                    normal *= -1
                old['normal_count'] += 1
                old['normal'] += (normal-old['normal'])/old['normal_count']
                old['normal'] /= max(float(np.linalg.norm(old['normal'])), 1e-12)
            old['boundary'] |= bool(bd[i])
            old['poses'].add(tuple(observation.pose))
            old['best_range'] = min(old['best_range'], distance)
            if center_key not in old['centers']:
                old['centers'].add(center_key)
                distinct = not old['directions'] or max(float(direction @ v) for v in old['directions']) < 1-1e-6
                if distinct and len(old['directions']) < self.max_directions:
                    old['directions'].append(direction.copy())
                    old['ranges'].append(distance)
                    old['focals'].append(float(k[0, 0]*k[1, 1]))
        self._frames.add(observation.frame_id)
        self._last_step = observation.step
        self._shape = d.shape
        self._snapshot = None
        return dict(step=observation.step, valid_depth_pixels=int(measured.sum()), sampled_points=len(points),
            distinct_cells_updated=len(indices), cells_added=added, total_cells=len(self._cells),
            valid_sampled_normals=int(nv.sum()), sampled_boundary_points=int(bd.sum()),
            empty_observation=not bool(measured.any()), boundary_is_true_hole=False,
            depth_only_geometry=True, calibrated_uncertainty=False)

    def snapshot(self):
        if self._snapshot is None:
            cells = [self._cells[key] for key in sorted(self._cells)]
            n, cap = len(cells), self.max_directions
            directions = np.zeros((n, cap, 3))
            ranges, focals = np.zeros((n, cap)), np.zeros((n, cap))
            quality_cap = max((len(c['inspection_records']) for c in cells), default=0)
            quality = np.zeros((n, quality_cap, 3))
            for i, cell in enumerate(cells):
                count = len(cell['directions'])
                directions[i, :count], ranges[i, :count], focals[i, :count] = cell['directions'], cell['ranges'], cell['focals']
                quality[i, :len(cell['inspection_records'])] = np.asarray(cell['inspection_records']).reshape(-1, 3)
            self._snapshot = MappingProxyType({key: _readonly(value) for key, value in dict(
                cell_keys=np.asarray(sorted(self._cells), dtype=np.int64).reshape(-1, 3),
                points=np.asarray([c['point'] for c in cells]).reshape(-1, 3),
                normals=np.asarray([c['normal'] for c in cells]).reshape(-1, 3),
                normal_valid=np.asarray([c['normal_count'] > 0 for c in cells], bool),
                past_view_directions=directions, past_view_ranges_m=ranges, past_view_focal_products=focals,
                history_direction_count=np.asarray([len(c['directions']) for c in cells], int),
                inspection_records_range_cosine_focal=quality,
                inspection_record_count=np.asarray([len(c['inspection_records']) for c in cells], int),
                distinct_pose_count=np.asarray([len(c['poses']) for c in cells], int),
                best_range_m=np.asarray([c['best_range'] for c in cells]),
                boundary=np.asarray([c['boundary'] for c in cells], bool)).items()})
        return self._snapshot

    def score_view(self, intrinsic, world_from_camera, *, mode='inspection', semantic_relevance=1.,
                   max_range_m=3., inspection_angle_deg=45., minimum_resolution_px_per_cm2=None):
        k, t = _calibration(intrinsic, world_from_camera)
        if mode not in ('inspection', 'vista'):
            raise ValueError('declared inspection or vista view mode required')
        if not np.isfinite([max_range_m, inspection_angle_deg]).all() or max_range_m <= 0 or not 0 < inspection_angle_deg < 90:
            raise ValueError('positive finite range and inspection angle within (0,90) required')
        # 2.56 px/cm2 at a 480 px reference focal length; f=48 gives .0256.
        minimum = 2.56 * (k[0, 0]*k[1, 1])/(480.*480.) if minimum_resolution_px_per_cm2 is None else float(minimum_resolution_px_per_cm2)
        if not math.isfinite(minimum) or minimum <= 0:
            raise ValueError('positive finite minimum projected pixel resolution required')
        s = self.snapshot()
        n = len(s['points'])
        relevance = np.asarray(semantic_relevance, float)
        if not np.isfinite(relevance).all() or np.any((relevance < 0) | (relevance > 1)):
            raise ValueError('finite semantic relevance in [0,1] required')
        if relevance.ndim == 0:
            relevance = np.full(n, float(relevance))
        if relevance.shape != (n,) or not np.isfinite(relevance).all() or np.any((relevance < 0) | (relevance > 1)):
            raise ValueError('one finite semantic relevance in [0,1] per surfel, or a scalar, required')
        output = dict(mode=mode, observed_surfels=n, visible_surfels=0, visible_pixels=0,
            geometry_gain=0., semantic_gain=0., inspection_gain=0., novelty_sum=0.,
            new_inspection_surfels=0, previously_qualified_surfels=0,
            new_inspection_indices=[], visible_indices=[], visible_novelty=[],
            potential_inspection_area_proxy_m2=0., minimum_resolution_px_per_cm2=minimum,
            visibility_scope='sparse observed-point z-buffer forecast; unknown space not certified free',
            boundary_is_true_hole=False, geometry_gain_definition='sum of pixelwise maximum angular novelty / all image pixels',
            semantic_gain_definition='(maximum visible relevance + mean positive visible-pixel relevance) / 2',
            inspection_gain_definition='newly qualifying observed surfels / all observed surfels',
            formula_combination_left_to_planner=True)
        if not n or self._shape is None:
            return output
        height, width = self._shape
        camera = (s['points']-t[:3, 3]) @ t[:3, :3]
        distance = np.linalg.norm(camera, axis=1)
        front = (camera[:, 2] > 1e-9) & (distance > 1e-9)
        u, v = np.zeros(n), np.zeros(n)
        u[front] = k[0, 0]*camera[front, 0]/camera[front, 2]+k[0, 2]
        v[front] = k[1, 1]*camera[front, 1]/camera[front, 2]+k[1, 2]
        projected = front & (u >= -.5) & (u < width-.5) & (v >= -.5) & (v < height-.5)
        ids = np.flatnonzero(projected)
        pixels = np.floor(v[ids]+.5).astype(int)*width + np.floor(u[ids]+.5).astype(int)
        zbuffer = np.full(height*width, np.inf)
        np.minimum.at(zbuffer, pixels, camera[ids, 2])
        visible = (distance[ids] <= max_range_m) & (camera[ids, 2] <= zbuffer[pixels]+math.sqrt(3)*self.voxel_m)
        ids, pixels = ids[visible], pixels[visible]
        if not len(ids):
            return output
        direction = np.divide(t[:3, 3]-s['points'], distance[:, None], out=np.zeros_like(camera), where=distance[:, None] > 0)
        history = np.arange(self.max_directions)[None, :] < s['history_direction_count'][:, None]
        dots = np.einsum('nij,nj->ni', s['past_view_directions'], direction)
        maximum_dot = np.where(history, dots, -1.).max(axis=1)
        novelty = np.clip((1-maximum_dot)/2, 0., 1.)
        center_key = _origin_key(t[:3, 3])
        seen_center = np.zeros(n, bool)
        for i, key in enumerate(sorted(self._cells)):
            if center_key in self._cells[key]['centers']:
                seen_center[i] = True
                novelty[i] = 0.
        geometry_pixels = np.zeros(height*width)
        semantic_pixels = np.zeros(height*width)
        np.maximum.at(geometry_pixels, pixels, novelty[ids])
        np.maximum.at(semantic_pixels, pixels, relevance[ids])
        positive = semantic_pixels[semantic_pixels > 0]
        semantic = .5*(float(positive.max())+float(positive.mean())) if len(positive) else 0.
        cosine = np.einsum('ij,ij->i', s['normals'], direction)
        cutoff = math.cos(math.radians(inspection_angle_deg))
        resolution = (k[0, 0]*k[1, 1]/10000.)*np.maximum(cosine, 0)/np.maximum(distance**2, 1e-12)
        current_good = s['normal_valid'] & (cosine >= cutoff) & (resolution >= minimum) & (distance <= max_range_m)
        quality = s['inspection_records_range_cosine_focal']
        quality_mask = np.arange(quality.shape[1])[None, :] < s['inspection_record_count'][:, None]
        old_range, old_cosine, old_focal = quality[:, :, 0], quality[:, :, 1], quality[:, :, 2]
        old_resolution = old_focal/10000.*old_cosine/np.maximum(old_range**2, 1e-12)
        # Adapt the same reference-resolution criterion to each past calibration.
        old_minimum = (2.56*old_focal/(480.*480.) if minimum_resolution_px_per_cm2 is None else minimum)
        qualified = s['normal_valid'] & np.any(quality_mask & (old_cosine >= cutoff)
            & (old_range <= max_range_m) & (old_resolution >= old_minimum), axis=1)
        new_indices = ids[current_good[ids] & ~qualified[ids] & ~seen_center[ids] & (relevance[ids] > 0)]
        new_count = len(new_indices)
        output.update(visible_surfels=len(ids), visible_pixels=len(np.unique(pixels)),
            geometry_gain=float(geometry_pixels.mean()), semantic_gain=semantic,
            inspection_gain=new_count/n, novelty_sum=float(novelty[ids].sum()),
            new_inspection_indices=new_indices.tolist(), visible_indices=ids.tolist(), visible_novelty=novelty[ids].tolist(),
            new_inspection_surfels=new_count, previously_qualified_surfels=int(qualified.sum()),
            potential_inspection_area_proxy_m2=new_count*self.voxel_m**2)
        return output
