"""Offline measured_instance_association_v1; never installs a production hook.

Track IDs name inferred measured-support associations. They are not physical
object identities. No category controls pixel ownership or geometric matching.
"""
from collections import Counter, defaultdict
import numpy as np


MIN_SHARED = 3
MIN_JACCARD = 0.20


def quality_samples(frame, owner_map):
    """Return the exact first sample per native 15 cm quality voxel."""
    depth = frame.depth_m
    height, width = depth.shape
    v, u = np.mgrid[:height, :width]
    local = np.stack([
        (u - frame.intrinsic[0, 2]) * depth / frame.intrinsic[0, 0],
        (v - frame.intrinsic[1, 2]) * depth / frame.intrinsic[1, 1], depth], axis=-1)
    dx = np.roll(local, -1, axis=1) - np.roll(local, 1, axis=1)
    dy = np.roll(local, -1, axis=0) - np.roll(local, 1, axis=0)
    length = np.linalg.norm(np.cross(dx, dy), axis=-1)
    valid = (depth > .15) & (length > 1e-8)
    valid[[0, -1], :] = False
    valid[:, [0, -1]] = False
    for axis in (0, 1):
        valid &= (np.roll(depth, 1, axis) > 0) & (np.roll(depth, -1, axis) > 0)
    rows, cols = np.nonzero(valid & (v % 3 == 0) & (u % 3 == 0))
    points = local[rows, cols] @ frame.world_from_camera[:3, :3].T + frame.world_from_camera[:3, 3]
    keys = np.floor(points / .15).astype(int)
    _, first = np.unique(keys, axis=0, return_index=True)
    return keys[first], owner_map[rows[first], cols[first]], frame.semantic[rows[first], cols[first]]


class MeasuredAssociation:
    def __init__(self):
        self.tracks = {}
        self.votes = defaultdict(Counter)
        self.latest_label = {}
        self.events = []
        self.frames = set()
        self.next_track = 1

    def observe(self, frame_id, keys, owners, labels):
        if frame_id in self.frames:
            raise ValueError("Duplicate frame is not new evidence")
        self.frames.add(frame_id)
        keys = [tuple(map(int, key)) for key in keys]
        current = {
            int(owner): {key for key, value in zip(keys, owners) if value == owner}
            for owner in np.unique(owners) if owner != 0}
        edges = []
        for owner, support in current.items():
            for track, previous in self.tracks.items():
                shared = len(support & previous)
                if shared < MIN_SHARED:
                    continue
                jaccard = shared / len(support | previous)
                if jaccard >= MIN_JACCARD:
                    edges.append((-jaccard, -shared, track, owner))
        assigned, used_tracks = {}, set()
        for negative_iou, negative_shared, track, owner in sorted(edges):
            if owner not in assigned and track not in used_tracks:
                assigned[owner] = (track, "matched", -negative_iou, -negative_shared)
                used_tracks.add(track)
        events = []
        for owner in sorted(current):
            if owner not in assigned:
                track = self.next_track
                self.next_track += 1
                self.tracks[track] = set()
                assigned[owner] = (track, "new", None, 0)
            track, match, iou, shared = assigned[owner]
            self.tracks[track].update(current[owner])
            events.append(dict(local_owner=owner, track=track, match=match,
                               sampled_keys=len(current[owner]), jaccard=iou, shared=shared))
        # Annotation is deliberately downstream of geometry-only assignment.
        for key, owner, category in zip(keys, owners, labels):
            if not owner:
                continue
            track = assigned[int(owner)][0]
            self.votes[key][track] += 1
            if category in (2, 3):
                self.latest_label[(key, track)] = int(category)
        self.events.append(dict(frame_id=frame_id, instances=events,
                                sampled_voxels=len(keys), unowned=int(np.count_nonzero(owners == 0))))

    def membership(self, keys):
        result = []
        for key in keys:
            counts = self.votes.get(tuple(map(int, key)), {})
            if not counts:
                result.append(0)
                continue
            track, count = max(counts.items(), key=lambda item: (item[1], -item[0]))
            result.append(track if 2 * count > sum(counts.values()) else 0)
        return np.asarray(result, dtype=np.int64)

    def labels_for(self, keys, membership):
        return np.asarray([
            self.latest_label.get((tuple(map(int, key)), int(track)), 0) if track else 0
            for key, track in zip(keys, membership)], dtype=np.int16)


def quality_keys(mapper, evidence):
    rows = [(key, row) for key, row in mapper.quality.items() if .12 < row['point'][2] < 1.8]
    if len(rows) > 10000:
        rows = [rows[i] for i in np.linspace(0, len(rows) - 1, 10000, dtype=int)]
    assert np.array_equal(np.asarray([row['point'] for _, row in rows]), evidence['point'])
    return np.asarray([key for key, _ in rows], dtype=np.int64)


class QualityView:
    def __init__(self, mapper, evidence):
        self.mapper, self.evidence = mapper, evidence

    def __getattr__(self, name):
        return getattr(self.mapper, name)

    def quality_evidence(self, max_points=1600):
        if max_points != 10000:
            raise ValueError("Original patch extraction sampling contract required")
        return self.evidence


def associated_patches(mapper, evidence, keys, association, extractor):
    memberships = association.membership(keys)
    labels = association.labels_for(keys, memberships)
    stride = int(np.prod(mapper.shape)) + 1
    patches = []
    for track in np.unique(memberships):
        selected = np.flatnonzero(memberships == track)
        subset = {name: values[selected].copy() for name, values in evidence.items()}
        subset['label'] = labels[selected]
        for patch in extractor(QualityView(mapper, subset)):
            patch = dict(patch)
            patch.update(source_group=int(patch['group']), association_track=int(track))
            patch['group'] = int(track) * stride + patch['source_group']
            patches.append(patch)
    return patches, memberships, labels
