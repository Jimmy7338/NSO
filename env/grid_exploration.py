"""Partially observed CPU grid exploration with private evaluation ground truth."""
from dataclasses import dataclass

import numpy as np
from scipy.ndimage import label

from utils.grid_geometry import DIRECTIONS, inflated_obstacles, visible_mask
from utils.paper_eval import coverage_metrics

UNKNOWN, FREE, OCCUPIED = -1, 0, 1
ACTIONS = ('forward', 'left', 'right', 'stop')


@dataclass(frozen=True)
class GridConfig:
    resolution_m: float = .1
    robot_radius_m: float = .1
    sensor_range_m: float = 1.5
    sensor_fov_deg: float = 90
    max_steps: int = 500

    def __post_init__(self):
        if self.resolution_m <= 0 or self.robot_radius_m < 0:
            raise ValueError('invalid spatial configuration')
        if self.sensor_range_m < self.resolution_m or not 0 < self.sensor_fov_deg <= 360:
            raise ValueError('invalid sensor configuration')
        if self.max_steps < 1:
            raise ValueError('max_steps must be positive')


@dataclass(frozen=True)
class GridObservation:
    """The entire policy interface: observations and perfect odometry only."""
    belief: np.ndarray
    visible: np.ndarray
    position: tuple
    heading: int
    step: int
    collision: bool


def readonly_copy(array):
    result = array.copy()
    result.setflags(write=False)
    return result


class GridExplorationEnv:
    def __init__(self, occupancy, config=None, semantic_field=None):
        occupancy = np.asarray(occupancy)
        if occupancy.ndim != 2 or min(occupancy.shape) < 5 or not np.isin(occupancy, [0, 1]).all():
            raise ValueError('occupancy must be a binary 2D grid, at least 5x5')
        self.config = config or GridConfig()
        self._world = occupancy.astype(bool, copy=True)
        self._semantic_world = None
        if semantic_field is not None:
            field = np.asarray(semantic_field, dtype=np.float32)
            if field.shape != occupancy.shape or not np.isfinite(field).all() or np.any((field < 0) | (field > 1)):
                raise ValueError('semantic field must match occupancy and lie in [0, 1]')
            self._semantic_world = field.copy()
        self._blocked = inflated_obstacles(self._world, self.config.robot_radius_m / self.config.resolution_m)
        self._component_labels, count = label(~self._blocked)
        if count == 0:
            raise ValueError('no valid robot-center positions')
        self._visibility_cache = {}
        self._done = True

    def reset(self, seed=0, start=None, heading=None):
        rng = np.random.default_rng(seed)
        if start is None:
            counts = np.bincount(self._component_labels.ravel()); counts[0] = 0
            cells = np.argwhere(self._component_labels == counts.argmax())
            start = tuple(map(int, cells[rng.integers(len(cells))]))
        if (len(start) != 2 or any(int(v) != v for v in start)
                or not (0 <= start[0] < self._world.shape[0] and 0 <= start[1] < self._world.shape[1])
                or self._blocked[tuple(start)]):
            raise ValueError('start is outside traversable robot-center space')
        self.position = tuple(map(int, start))
        self.heading = int(rng.integers(4)) if heading is None else int(heading)
        if self.heading not in range(4):
            raise ValueError('heading must be 0..3')
        self._reachable = self._component_labels == self._component_labels[self.position]
        self._belief = np.full(self._world.shape, UNKNOWN, dtype=np.int8)
        self._visited = {self.position}
        self.steps = self.moves = self.revisits = self.collisions = 0
        self.path_length_m = 0.
        self._done = False
        self.termination_reason = None
        self._sense()
        return self._observation(False)

    def _sense(self):
        key = (self.position, self.heading)
        if key not in self._visibility_cache:
            self._visibility_cache[key] = visible_mask(
                self._world, self.position, self.heading,
                int(self.config.sensor_range_m / self.config.resolution_m), self.config.sensor_fov_deg)
        self._visible = self._visibility_cache[key]
        self._belief[self._visible] = self._world[self._visible]

    def _observation(self, collision):
        if self._semantic_world is not None:
            from env.grid_semantics import SemanticObservation
            semantic = np.full(self._world.shape, np.nan, dtype=np.float32)
            semantic[self._visible] = self._semantic_world[self._visible]
            return SemanticObservation(readonly_copy(self._belief), readonly_copy(self._visible),
                                       self.position, self.heading, self.steps, collision,
                                       readonly_copy(semantic))
        return GridObservation(readonly_copy(self._belief), readonly_copy(self._visible),
                               self.position, self.heading, self.steps, collision)

    def step(self, action):
        if self._done:
            raise RuntimeError('call reset before stepping a terminated episode')
        if action not in ACTIONS:
            raise ValueError(f'unknown action: {action}')
        collision = False
        self.steps += 1
        if action == 'forward':
            dr, dc = DIRECTIONS[self.heading]
            target = (self.position[0] + dr, self.position[1] + dc)
            if (not (0 <= target[0] < self._world.shape[0] and 0 <= target[1] < self._world.shape[1])
                    or self._blocked[target]):
                collision = True
                self.collisions += 1
            else:
                self.position = target
                self.moves += 1
                self.path_length_m += self.config.resolution_m
                self.revisits += int(target in self._visited)
                self._visited.add(target)
        elif action in ('left', 'right'):
            self.heading = (self.heading + (1 if action == 'right' else -1)) % 4
        else:
            self._done = True
            self.termination_reason = 'policy_stop'
        if self.steps >= self.config.max_steps and not self._done:
            self._done = True
            self.termination_reason = 'budget_exhausted'
        self._sense()
        return self._observation(collision), self._done

    def evaluation_metrics(self):
        """Evaluator only. Never passed to the policy or used as a stop oracle."""
        result = coverage_metrics(self._belief != UNKNOWN, self._reachable,
                                  self.config.resolution_m * 100)
        result.update(coverage_reference='start_connected_robot_center_grid',
                      steps=self.steps, path_length_m=self.path_length_m,
                      collisions=self.collisions, successful_moves=self.moves,
                      revisited_moves=self.revisits,
                      revisit_ratio=self.revisits / self.moves if self.moves else 0.,
                      termination_reason=self.termination_reason)
        return result

    def evaluation_assets(self):
        """Copies for archives/plots, never part of GridObservation."""
        return {'occupancy': self._world.copy(), 'reachable': self._reachable.copy()}
