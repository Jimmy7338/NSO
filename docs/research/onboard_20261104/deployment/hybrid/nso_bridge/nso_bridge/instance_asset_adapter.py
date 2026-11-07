"""Optional measured-instance partition of the existing quality evidence.

Tracks are measured support associations, not certified object identities. The
original patch extractor retains ownership of all geometry and eligibility rules.
"""
import numpy as np


class _QualityPartition:
    """Delegate mapper state while exposing only one exclusive evidence subset."""
    def __init__(self, mapper, evidence):
        self._mapper = mapper
        self._evidence = evidence

    def __getattr__(self, name):
        return getattr(self._mapper, name)

    def quality_evidence(self, max_points=1600):
        # The caller already sampled the original evidence once at 10000.
        # The original extractor requests that same limit; do not resample.
        if max_points != 10000:
            raise ValueError('instance partition requires original 10000 point contract')
        return self._evidence


def observed_instance_patches(mapper, config, original_observed_patches):
    """Extract disjoint instance and unknown patches using the original extractor."""
    q = mapper.quality_evidence(max_points=10000)
    if q is None:
        return []
    n = len(q['point'])
    if hasattr(mapper, 'quality'):
        rows = [(key, row) for key, row in mapper.quality.items()
                if .12 < row['point'][2] < 1.8]
        if len(rows) > 10000:
            rows = [rows[i] for i in np.linspace(0, len(rows)-1, 10000, dtype=int)]
        if len(rows) != n or not np.array_equal(
                np.asarray([row['point'] for _, row in rows]), q['point']):
            raise ValueError('quality voxel keys do not align with sampled evidence')
        keys = np.asarray([key for key, _ in rows], dtype=np.int64).reshape(-1, 3)
        tracks = np.asarray(mapper.instance_support_index.memberships_for_keys(keys))
        original_keys_preserved = True
    else:
        # Test doubles without the real quality ledger cannot expose its keys.
        tracks = np.asarray(mapper.instance_support_index.memberships(q['point']))
        original_keys_preserved = False
    if tracks.shape != (n,) or tracks.dtype.kind not in 'iu' or np.any(tracks < 0):
        raise ValueError('instance memberships must be nonnegative integer track IDs')
    if any(len(values) != n for values in q.values()):
        raise ValueError('quality evidence arrays must align')
    stride = int(mapper.shape[0]) * int(mapper.shape[1]) + 1
    patches = []
    for track in np.unique(tracks):
        ids = np.flatnonzero(tracks == track)
        evidence = {key: values[ids] for key, values in q.items()}
        partition = _QualityPartition(mapper, evidence)
        for patch in original_observed_patches(partition, config):
            row = dict(patch)
            group = int(row['group'])
            if not 0 < group < stride:
                raise ValueError('original patch group outside grid component range')
            row['source_group'] = group
            row['instance_track_id'] = int(track)
            row['original_quality_voxel_keys_preserved'] = original_keys_preserved
            row['group'] = int(track) * stride + group
            patches.append(row)
    return patches
