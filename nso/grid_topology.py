"""Observed-space doorway cuts and persistent overlap-matched region graph."""
from collections import deque
from itertools import combinations
import numpy as np
from scipy.ndimage import label
from utils.grid_geometry import DIRECTIONS


class ObservedTopology:
    def __init__(self, max_door_width=7, min_region_cells=12):
        self.max_door_width = max_door_width
        self.min_region_cells = min_region_cells
        self.reset()

    def reset(self):
        self.labels = None
        self.nodes = {}
        self.edges = []
        self.next_id = 1
        self.updates = 0

    def update(self, belief):
        free = belief == 0
        cuts = np.zeros(free.shape, bool)
        # Both ends must be OBSERVED obstacles. Unknown boundaries are never walls.
        for axis in (0, 1):
            grid = belief if axis == 0 else belief.T
            target = cuts if axis == 0 else cuts.T
            for r, row in enumerate(grid):
                walls = np.flatnonzero(row == 1)
                for a, b in zip(walls[:-1], walls[1:]):
                    if 1 < b - a <= self.max_door_width + 1 and np.all(row[a+1:b] == 0):
                        target[r, a+1:b] = True
        cores, count = label(free & ~cuts)
        sizes = np.bincount(cores.ravel())
        local = np.zeros(free.shape, np.int32)
        next_local = 1
        for k in range(1, count + 1):
            if sizes[k] >= self.min_region_cells:
                local[cores == k] = next_local
                next_local += 1
        # Multi-source expansion assigns narrow passages to adjacent region cores.
        queue = deque(map(tuple, np.argwhere(local > 0)))
        h, w = free.shape
        while queue:
            r, c = queue.popleft()
            for dr, dc in DIRECTIONS:
                nr, nc = r + dr, c + dc
                if 0 <= nr < h and 0 <= nc < w and free[nr, nc] and local[nr, nc] == 0:
                    local[nr, nc] = local[r, c]
                    queue.append((nr, nc))
        remaining, count = label(free & (local == 0))
        for k in range(1, count + 1):
            local[remaining == k] = next_local
            next_local += 1
        # Greedy maximum-overlap one-to-one association: no duplicate region IDs.
        matches = []
        if self.labels is not None:
            for k in range(1, next_local):
                old, n = np.unique(self.labels[local == k], return_counts=True)
                matches.extend((-int(size), k, int(o)) for o, size in zip(old, n) if o)
        mapping, used = {}, set()
        for _, k, old in sorted(matches):
            if k not in mapping and old not in used:
                mapping[k] = old
                used.add(old)
        for k in range(1, next_local):
            if k not in mapping:
                mapping[k] = self.next_id
                self.next_id += 1
        labels = np.zeros_like(local)
        for k, node in mapping.items():
            labels[local == k] = node
        edges = set()
        for a, b in ((labels[:-1], labels[1:]), (labels[:, :-1], labels[:, 1:])):
            mask = (a > 0) & (b > 0) & (a != b)
            edges.update(tuple(sorted((int(x), int(y)))) for x, y in zip(a[mask], b[mask]))
        self.labels, self.edges = labels, sorted(edges)
        self.nodes = {n: {'cells': int(np.count_nonzero(labels == n))} for n in mapping.values()}
        self.updates += 1
        return self

    def hops(self, start_node):
        distance = {int(start_node): 0}
        queue = deque([int(start_node)])
        while queue:
            node = queue.popleft()
            for a, b in self.edges:
                neighbor = b if a == node else a if b == node else None
                if neighbor is not None and neighbor not in distance:
                    distance[neighbor] = distance[node] + 1
                    queue.append(neighbor)
        return distance
