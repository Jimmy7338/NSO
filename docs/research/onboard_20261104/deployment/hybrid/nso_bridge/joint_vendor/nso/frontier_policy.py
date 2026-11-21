"""Two deterministic geometry baselines using ONLY GridObservation."""
from collections import deque
from dataclasses import dataclass

import numpy as np
from scipy.ndimage import binary_dilation, label

from utils.grid_geometry import DIRECTIONS, inflated_obstacles, visible_mask

CROSS = np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]], dtype=bool)
METHODS = ('nearest_frontier', 'gain_per_cost')


def shortest_paths(traversable, start):
    """Unit-cost four-neighbor BFS (equivalent to Dijkstra for unit edges)."""
    distance = np.full(traversable.shape, -1, dtype=np.int32)
    parent = np.full((*traversable.shape, 2), -1, dtype=np.int32)
    distance[start] = 0
    queue = deque([start])
    height, width = traversable.shape
    while queue:
        r, c = queue.popleft()
        for dr, dc in DIRECTIONS:
            nr, nc = r + dr, c + dc
            if (0 <= nr < height and 0 <= nc < width and traversable[nr, nc]
                    and distance[nr, nc] < 0):
                distance[nr, nc] = distance[r, c] + 1
                parent[nr, nc] = (r, c)
                queue.append((nr, nc))
    return distance, parent


def trace_path(parent, start, goal):
    path = []
    while goal != start:
        path.append(goal)
        goal = tuple(map(int, parent[goal]))
        if goal[0] < 0:
            raise ValueError('goal has no observed-free path')
    return path[::-1]


@dataclass(frozen=True)
class Decision:
    action: str
    goal: tuple | None
    goal_heading: int | None
    predicted_gain_cells: int
    distance_cells: int
    score: float
    new_goal: bool


class FrontierPolicy:
    def __init__(self, config, method='nearest_frontier', max_candidates=24):
        if method not in METHODS or max_candidates < 1:
            raise ValueError('invalid policy configuration')
        self.config, self.method, self.max_candidates = config, method, max_candidates
        self.reset()

    def reset(self):
        self.goal = None
        self.goal_heading = None
        self._path = []
        self._failed_cells = set()
        self._gain = self._distance = 0
        self._score = 0.
        self.goal_count = 0
        self._scans = 0
        self._known_count = -1

    def _gain_at(self, belief, cell, heading):
        visible = visible_mask(belief == 1, cell, heading,
                               int(self.config.sensor_range_m / self.config.resolution_m),
                               self.config.sensor_fov_deg)
        return int(np.count_nonzero(visible & (belief == -1)))

    def _candidate_score(self, obs, cell, heading, gain, distance, parent, base_score):
        return base_score

    def _selected(self, obs):
        pass

    def _select(self, obs, traversable):
        distance, parent = shortest_paths(traversable, obs.position)
        frontier = traversable & binary_dilation(obs.belief == -1, structure=CROSS) & (distance >= 0)
        components, count = label(frontier, structure=CROSS)
        candidates = {obs.position}  # turning in place may reveal unseen space
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

    def act(self, obs):
        known_count = int(np.count_nonzero(obs.belief != -1))
        if known_count != self._known_count:
            self._scans = 0
            self._known_count = known_count
        if obs.collision:
            dr, dc = DIRECTIONS[obs.heading]
            self._failed_cells.add((obs.position[0] + dr, obs.position[1] + dc))
            self.goal = None
        traversable = ((obs.belief == 0) & ~inflated_obstacles(
            obs.belief == 1, self.config.robot_radius_m / self.config.resolution_m))
        for cell in self._failed_cells:
            if 0 <= cell[0] < traversable.shape[0] and 0 <= cell[1] < traversable.shape[1]:
                traversable[cell] = False
        traversable[obs.position] = True
        while self._path and self._path[0] == obs.position:
            self._path.pop(0)
        if self.goal is not None:
            if any(not traversable[cell] for cell in self._path):
                self.goal = None
            elif obs.position == self.goal and obs.heading == self.goal_heading:
                self.goal = None
        new_goal = False
        if self.goal is None:
            new_goal = self._select(obs, traversable)
            if not new_goal:
                self._scans += 1
                return Decision('right' if self._scans <= 4 else 'stop', None, None, 0, 0, 0., False)
        if self._path:
            next_cell = self._path[0]
            delta = (next_cell[0] - obs.position[0], next_cell[1] - obs.position[1])
            heading = DIRECTIONS.index(delta)
        else:
            heading = self.goal_heading
        turn = (heading - obs.heading) % 4
        action = 'right' if turn in (1, 2) else 'left' if turn == 3 else 'forward'
        # At an aligned viewing target, resample on the next observation; normally
        # _select excludes it since its currently visible unknown gain is zero.
        if not self._path and turn == 0:
            action = 'right'
        return Decision(action, self.goal, self.goal_heading, self._gain,
                        self._distance, self._score, new_goal)
