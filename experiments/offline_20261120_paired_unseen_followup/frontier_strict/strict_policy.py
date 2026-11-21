"""Declared one-line repair: only actual measured-grid frontiers are candidates.

The inherited act(), gain, heading, BFS, representative, cap and tie rules are
unchanged. An actual frontier current pose can still be generated normally.
"""
import numpy as np
from scipy.ndimage import binary_dilation, label
from nso.frontier_policy import FrontierPolicy as OriginalFrontierPolicy
from nso.frontier_policy import CROSS, shortest_paths, trace_path


class FrontierPolicy(OriginalFrontierPolicy):
    def _select(self, obs, traversable):
        distance, parent = shortest_paths(traversable, obs.position)
        frontier = traversable & binary_dilation(obs.belief == -1, structure=CROSS) & (distance >= 0)
        components, count = label(frontier, structure=CROSS)
        candidates = set()  # Sole algorithm change: do not inject non-frontier current pose.
        for component in range(1, count + 1):
            cells = np.argwhere(components == component)
            # Multiple representatives of a large boundary; deterministic tie order.
            order = np.lexsort((cells[:, 1], cells[:, 0], distance[tuple(cells.T)]))
            for i in np.linspace(0, len(order) - 1, min(4, len(order)), dtype=int):
                candidates.add(tuple(map(int, cells[order[i]])))
        candidates = sorted(candidates, key=lambda cell: (distance[cell], cell))[:self.max_candidates]
        scored = []
        for cell in candidates:
            gains = [self._gain_at(obs.belief, cell, h) for h in range(4)]
            heading = min(range(4), key=lambda h: (-gains[h], min((h - obs.heading) % 4,
                                                                 (obs.heading - h) % 4), h))
            gain = gains[heading]
            if gain == 0:
                continue
            d = int(distance[cell])
            turns = min((heading - obs.heading) % 4, (obs.heading - heading) % 4)
            score = gain / (d + turns + 1)
            score = self._candidate_score(obs, cell, heading, gain, d, parent, score)
            if not np.isfinite(score):
                continue
            priority = (d, -gain, cell) if self.method == 'nearest_frontier' else (-score, d, cell)
            scored.append((priority, cell, heading, gain, d, score))
        if not scored:
            return False
        _, self.goal, self.goal_heading, self._gain, self._distance, self._score = min(scored)
        self._path = trace_path(parent, obs.position, self.goal)
        self.goal_count += 1
        self._selected(obs)
        return True
