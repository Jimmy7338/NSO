"""Directed navigation graph with a sensor-only swept-footprint exception."""

from collections import deque
from dataclasses import dataclass
from functools import cached_property

import numpy as np
from scipy.ndimage import distance_transform_edt
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components

from sensor_probe.evidence import AccumulatedEvidence


DIRECTIONS = ((-1, 0), (0, 1), (1, 0), (0, -1))


def inflated(obstacles, radius_cells):
    distance = distance_transform_edt(np.pad(~obstacles, 1))[1:-1, 1:-1]
    return distance <= radius_cells + np.sqrt(2.) / 2


@dataclass
class CertifiedGraph:
    safe: np.ndarray
    edges: np.ndarray
    added: dict

    @cached_property
    def index(self):
        cells = np.argwhere(self.safe)
        ids = np.full(self.safe.shape, -1, dtype=np.int32)
        ids[tuple(cells.T)] = np.arange(len(cells))
        return cells, ids

    @cached_property
    def oriented(self):
        cells, ids = self.index
        base = np.arange(len(cells)) * 4
        sources, targets = [], []
        for h, (dr, dc) in enumerate(DIRECTIONS):
            sources.extend((base+h, base+h))
            targets.extend((base+(h+1)%4, base+(h-1)%4))
            move = self.edges[cells[:, 0], cells[:, 1], h]
            r, c = cells[move].T
            sources.append(base[move]+h)
            targets.append(ids[r+dr, c+dc]*4+h)
        row, col = np.concatenate(sources), np.concatenate(targets)
        return csr_matrix((np.ones(len(row)), (row, col)), shape=(len(base)*4,)*2), cells, ids

    def orientation_graph(self):
        return self.oriented

    @cached_property
    def labels(self):
        cells, ids = self.index
        origins, targets = [], []
        for h, (dr, dc) in enumerate(DIRECTIONS):
            r, c = np.nonzero(self.edges[..., h])
            origins.extend(ids[r, c])
            targets.extend(ids[r+dr, c+dc])
        graph = csr_matrix((np.ones(len(origins), dtype=np.int8), (origins, targets)),
                           shape=(len(cells), len(cells)))
        count, components = connected_components(graph, directed=False)
        labels = np.zeros(self.safe.shape, dtype=np.int32)
        labels[tuple(cells.T)] = components + 1
        return labels, count

    def regions(self):
        return self.labels

    def shortest_paths(self, allowed, start):
        """Preserve the frontier policy's plain unit-cell BFS and tie order."""
        distance = np.full(self.safe.shape, -1, dtype=np.int32)
        parent = np.full((*self.safe.shape, 2), -1, dtype=np.int32)
        distance[start] = 0
        queue = deque([start])
        while queue:
            r, c = queue.popleft()
            for h, (dr, dc) in enumerate(DIRECTIONS):
                if not self.edges[r, c, h]:
                    continue
                next_cell = r+dr, c+dc
                if allowed[next_cell] and distance[next_cell] < 0:
                    distance[next_cell] = distance[r, c] + 1
                    parent[next_cell] = (r, c)
                    queue.append(next_cell)
        return distance, parent


class SweepEvidence:
    """Keep the original known-free graph and certify corrections from raw observations."""

    def __init__(self, shape, resolution_m=.20, radius_m=.20):
        self.shape = tuple(shape)
        self.resolution = float(resolution_m)
        self.radius = float(radius_m)
        self.raw = AccumulatedEvidence()
        self._belief_key = None
        self._version = 0
        self._graph = None
        self._frontier = None

    def observe(self, frame, scan):
        self.raw.observe(frame, scan)
        self._version += 1

    def xy(self, cell):
        return np.asarray(((cell[1] + .5)*self.resolution,
                           (self.shape[0] - cell[0] - .5)*self.resolution))

    def build(self, belief):
        belief = np.asarray(belief)
        if belief.shape != self.shape or not np.isin(belief, (-1, 0, 1)).all():
            raise ValueError("swept navigation requires the unchanged measured tri-state belief")
        key = self._version, belief.tobytes()
        if self._belief_key == key:
            return self._graph
        known = belief == 0
        scale = self.radius / self.resolution
        legacy = ~inflated(~known, scale)
        occupied_margin = inflated(belief == 1, scale)
        safe = legacy.copy()
        edges = np.zeros((*self.shape, 4), dtype=bool)
        added = {}
        for h, (dr, dc) in enumerate(DIRECTIONS[1:3], 1):
            a = (slice(None, -1 if dr else None), slice(None, -1 if dc else None))
            b = (slice(dr, None), slice(dc, None))
            old_edges = legacy[a] & legacy[b]
            edges[a + (h,)] = old_edges
            edges[b + ((h+2) % 4,)] = old_edges
            # This exception fixes occupied-cell inflation around known-free
            # centers. Unknown-only boundaries have no admission shortcut.
            investigate = known[a] & known[b] & ~old_edges & (occupied_margin[a] | occupied_margin[b])
            for r, c in np.argwhere(investigate):
                start, end = (int(r), int(c)), (int(r+dr), int(c+dc))
                p, q = self.xy(start), self.xy(end)
                dimensions = np.asarray(self.shape[::-1]) * self.resolution
                if np.any(np.minimum(p, q) < self.radius) or np.any(np.maximum(p, q) > dimensions-self.radius):
                    continue
                proof = self.raw.certify(p, q, self.radius)
                if not proof["allowed"]:
                    continue
                safe[start] = safe[end] = True
                edges[start + (h,)] = edges[end + ((h+2) % 4,)] = True
                added[start + (h,)] = added[end + ((h+2) % 4,)] = proof
        self._belief_key, self._graph = key, CertifiedGraph(safe, edges, added)
        self._frontier = None
        return self._graph

    def frontier(self, belief):
        """Keep known cells beside unknown space eligible for frontier proposals."""
        guarded = self.build(belief)
        if self._frontier is None:
            legacy = (belief == 0) & ~inflated(belief == 1, self.radius / self.resolution)
            candidates = legacy | guarded.safe
            edges = guarded.edges.copy()
            for h, (dr, dc) in enumerate(DIRECTIONS[1:3], 1):
                a = (slice(None, -1 if dr else None), slice(None, -1 if dc else None))
                b = (slice(dr, None), slice(dc, None))
                native = legacy[a] & legacy[b]
                edges[a + (h,)] |= native
                edges[b + ((h+2) % 4,)] |= native
            self._frontier = CertifiedGraph(candidates, edges, guarded.added)
        return self._frontier
