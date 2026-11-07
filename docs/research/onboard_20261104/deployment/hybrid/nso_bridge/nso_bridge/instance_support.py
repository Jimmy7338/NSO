"""Optional measured-voxel instance support; track IDs are not object identity.

Uses the same pixel and voxel selection as QualityMapperV2.update. Cross-frame
association uses measured voxel overlap, without class or inferred geometry.
Within one image, confidence resolves mask overlap exactly as in the frontend.
"""
import numpy as np


def canonical_owner_map(instances, valid, stride):
    """Assign full-resolution masks globally by detector score, then subsample.

    Semantic classes do not affect support assignment; indices name each mask.
    Optional assigned_mask must match the exact valid mask used for assignment.
    """
    valid = np.asarray(valid)
    if valid.ndim != 2 or valid.dtype != np.bool_:
        raise ValueError('valid must be a two-dimensional boolean mask')
    if isinstance(stride, (bool, np.bool_)) or not isinstance(stride, (int, np.integer)) or stride <= 0:
        raise ValueError('stride must be a positive integer')
    checked = []
    used = set()
    for i, item in enumerate(instances):
        idx = item.get('instance_index', i)
        if isinstance(idx, (bool, np.bool_)) or not isinstance(idx, (int, np.integer)) or idx < 0 or int(idx) >= np.iinfo(np.int64).max:
            raise ValueError('instance indices must be unique nonnegative integers')
        idx = int(idx)
        if idx in used:
            raise ValueError('duplicate instance index')
        used.add(idx)
        mask = np.asarray(item['mask'])
        if mask.shape != valid.shape or mask.dtype != np.bool_:
            raise ValueError('instance mask must be boolean and match valid shape')
        score = float(item['detector_score'])
        if not np.isfinite(score):
            raise ValueError('detector score must be finite')
        assigned = item.get('assigned_mask')
        if assigned is not None:
            assigned = np.asarray(assigned)
            if assigned.shape != valid.shape or assigned.dtype != np.bool_:
                raise ValueError('assigned mask must be boolean and match valid shape')
        checked.append((score, i, idx, mask, assigned))
    out = np.zeros(valid.shape, dtype=np.int64)
    owned = np.zeros(valid.shape, dtype=bool)
    for _, _, idx, mask, assigned in sorted(checked, key=lambda x: (-x[0], x[1])):
        measured = mask & valid
        support = measured & ~owned
        if assigned is not None and not np.array_equal(assigned, support):
            raise ValueError('assigned_mask does not match score-ordered ownership')
        out[support] = idx + 1
        owned |= measured
    return np.ascontiguousarray(out[::stride, ::stride])


class InstanceSupportIndex:
    voxel_m = .15

    def __init__(self):
        self._owners = {}
        self._frames = set()
        self._next_track = 1
        self._events = []

    @staticmethod
    def _sample(frame, owner_map):
        d = np.asarray(frame.depth_m)
        owners = np.asarray(owner_map)
        if d.ndim != 2 or owners.shape != d.shape:
            raise ValueError('owner map must match the two-dimensional depth image')
        if owners.dtype.kind not in 'iu' or np.any(owners < 0):
            raise ValueError('owner map must contain nonnegative integers')
        h, w = d.shape
        v, u = np.mgrid[:h, :w]
        local = np.stack([(u-frame.intrinsic[0, 2])*d/frame.intrinsic[0, 0],
                          (v-frame.intrinsic[1, 2])*d/frame.intrinsic[1, 1], d], axis=-1)
        dx = np.roll(local, -1, axis=1)-np.roll(local, 1, axis=1)
        dy = np.roll(local, -1, axis=0)-np.roll(local, 1, axis=0)
        length = np.linalg.norm(np.cross(dx, dy), axis=-1)
        valid = (d > .15) & (length > 1e-8)
        valid[[0, -1], :] = False
        valid[:, [0, -1]] = False
        valid &= (np.roll(d, 1, 0)>0) & (np.roll(d, -1, 0)>0) & (np.roll(d, 1, 1)>0) & (np.roll(d, -1, 1)>0)
        rows, cols = np.nonzero(valid & (v % 3 == 0) & (u % 3 == 0))
        pts = local[rows, cols] @ frame.world_from_camera[:3, :3].T + frame.world_from_camera[:3, 3]
        keys = np.floor(pts / .15).astype(int)
        _, indices = np.unique(keys, axis=0, return_index=True)
        return keys[indices], owners[rows[indices], cols[indices]]

    def observe(self, frame, owner_map, frame_id, instance_ids=None):
        """Associate first sampled measured voxels. instance_ids are informational only."""
        if frame_id in self._frames:
            return {'frame_id': frame_id, 'duplicate': True, 'new_voxels': 0}
        keys, labels = self._sample(frame, owner_map)
        # Match all groups against ownership as it stood before this frame.
        prior = self._owners.copy()
        events = []
        for label in np.unique(labels):
            label = int(label)
            if not label:
                continue
            group = [tuple(int(x) for x in k) for k in keys[labels == label]]
            found = sorted({prior[k] for k in group if k in prior})
            if len(found) > 1:
                events.append({'instance_index': label, 'match': 'ambiguous',
                               'existing_tracks': found, 'assigned_track': 0,
                               'new_voxels': 0, 'unassigned_voxels': sum(k not in prior for k in group)})
                continue
            if found:
                track, match = found[0], 'matched'
            else:
                track, match = self._next_track, 'new'
                self._next_track += 1
            new = 0
            for key in group:
                if key not in self._owners:
                    self._owners[key] = track
                    new += 1
            events.append({'instance_index': label, 'match': match, 'existing_tracks': found,
                           'assigned_track': track, 'new_voxels': new, 'unassigned_voxels': 0})
        self._frames.add(frame_id)
        result = {'frame_id': frame_id, 'duplicate': False, 'sampled_voxels': len(keys),
                  'unlabeled_sampled_voxels': int(np.count_nonzero(labels == 0)),
                  'new_voxels': sum(e['new_voxels'] for e in events), 'instances': events}
        self._events.append(result)
        return result

    def memberships(self, points):
        """Return one owner track per world point; 0 means unassigned."""
        points = np.asarray(points)
        if points.ndim != 2 or points.shape[1] != 3 or not np.isfinite(points).all():
            raise ValueError('points must be finite Nx3 world coordinates')
        keys = np.floor(points / self.voxel_m).astype(int)
        return np.asarray([self._owners.get(tuple(k), 0) for k in keys], dtype=np.int64)

    def memberships_for_keys(self, keys):
        """Look up original QualityMapperV2 dictionary keys without re-quantizing averaged points."""
        keys = np.asarray(keys)
        if keys.ndim != 2 or keys.shape[1] != 3 or keys.dtype.kind not in 'iu':
            raise ValueError('keys must be an Nx3 integer array')
        return np.asarray([self._owners.get(tuple(k), 0) for k in keys], dtype=np.int64)

    def summary(self):
        return {'frames': len(self._frames), 'tracks': self._next_track-1,
                'owned_voxels': len(self._owners), 'observations': list(self._events),
                'interpretation': 'measured voxel support tracks; not object identity or ground truth'}
