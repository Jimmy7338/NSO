"""Physical exploration metrics, independent of policy rewards (schema v2)."""
from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import List, Optional

import numpy as np

METRICS_SCHEMA_VERSION = 2


def coverage_metrics(explored_map, explorable_map, map_resolution_cm):
    """Count observed cells inside the aligned GT explorable map.

    The denominator is the supplied map, not an inferred navmesh area.
    Empty references are missing data, never fabricated zero coverage.
    """
    if map_resolution_cm <= 0:
        raise ValueError('map_resolution_cm must be positive')
    explored, reference = np.asarray(explored_map), np.asarray(explorable_map)
    if explored.shape != reference.shape:
        raise ValueError('explored/reference map shapes differ')
    if not np.isfinite(explored).all() or not np.isfinite(reference).all():
        raise ValueError('non-finite exploration map')
    reference = reference > 0
    observed = int(np.count_nonzero((explored > 0) & reference))
    total = int(np.count_nonzero(reference))
    return {
        'observed_free_cells': observed,
        'explorable_free_cells': total,
        'explored_area_m2': observed * (map_resolution_cm / 100.0) ** 2,
        'coverage_ratio': observed / total if total else None,
        'coverage_reference': 'aligned_gt_explorable_map',
    }


class GoalMetrics:
    """One attempt per global target, success at any step within its budget.

    Coordinates are global map (row, col), with GT agent positions used only
    by this evaluator. Repeated local replans do not create new attempts.
    """
    def __init__(self, budget, radius_cells):
        self.budget, self.radius_cells = budget, radius_cells
        self.attempts = self.completed = self.successes = self.unreachable = 0
        self.goal = None
        self.age = 0
        self.reached = self.invalid = False

    def start(self, goal):
        self.goal = np.asarray(goal, dtype=float)
        self.age = 0
        self.reached = self.invalid = False
        self.attempts += 1

    def mark_unreachable(self, invalid):
        if self.goal is not None and invalid and not self.invalid:
            self.unreachable += 1
            self.invalid = True

    def advance(self, agent_cell, terminal=False):
        if self.goal is None:
            return
        self.age += 1
        self.reached |= bool(np.linalg.norm(np.asarray(agent_cell) - self.goal)
                             <= self.radius_cells)
        if self.age >= self.budget or terminal:
            self.completed += 1
            self.successes += int(self.reached)
            self.goal = None

    def snapshot(self):
        return {'goal_attempt_count': self.attempts,
                'completed_goal_count': self.completed,
                'embodied_goal_success_count': self.successes,
                'unreachable_goal_count': self.unreachable}


@dataclass
class EpisodePaperMetrics:
    explored_area_m2: Optional[float] = None
    coverage_ratio: Optional[float] = None
    # No GT/estimated trajectory pairs are currently recorded. Do not label
    # sensor-pose increments as trajectory ATE.
    trajectory_drift_rmse_cm: Optional[float] = None
    odometry_step_translation_rmse_cm: Optional[float] = None
    odometry_step_rotation_rmse_deg: Optional[float] = None
    pose_sample_count: int = 0
    observed_free_cells: Optional[int] = None
    explorable_free_cells: Optional[int] = None
    unreachable_goal_count: int = 0
    goal_attempt_count: int = 0
    completed_goal_count: int = 0
    loop_closure_count: int = 0
    semantic_reward_mean: Optional[float] = None
    embodied_goal_success_rate: Optional[float] = None
    step_count: int = 0

    def to_dict(self):
        return asdict(self)


class PaperMetricsTracker:
    def __init__(self, map_resolution_cm=5.0):
        self.map_resolution_cm = map_resolution_cm
        self.reset_episode()

    def reset_episode(self):
        self._metrics = EpisodePaperMetrics()
        self._translation_sq, self._rotation_sq, self._sem_rewards = [], [], []

    def update_step(self, *, info=None, pose_err=None, sem_reward=None,
                    coverage_ratio=None, explored_area_m2=None):
        """Call exactly once for each environment action, using terminal info.

        exp_reward/exp_ratio are intentionally not accepted: they are reward
        and coverage-increment aliases in old environments.
        """
        info = info or {}
        self._metrics.step_count += 1
        pose_err = info.get('pose_err', pose_err)
        sem_reward = info.get('semantic_reward_sample', sem_reward)
        if pose_err is not None:
            dx, dy, angle_rad = pose_err
            if np.isfinite(pose_err).all():
                self._translation_sq.append((dx * 100) ** 2 + (dy * 100) ** 2)
                self._rotation_sq.append(float(np.rad2deg(angle_rad)) ** 2)
        if sem_reward is not None:
            self._sem_rewards.append(float(sem_reward))
        self._metrics.coverage_ratio = info.get('coverage_ratio', coverage_ratio)
        self._metrics.explored_area_m2 = info.get('explored_area_m2', explored_area_m2)
        for key in ('observed_free_cells', 'explorable_free_cells',
                    'unreachable_goal_count', 'goal_attempt_count', 'completed_goal_count'):
            if key in info:
                setattr(self._metrics, key, info[key])
        completed = self._metrics.completed_goal_count
        self._metrics.embodied_goal_success_rate = (
            info.get('embodied_goal_success_count', 0) / completed if completed else None)

    def record_loop(self):
        """Separate late detector event; does not increment environment steps."""
        self._metrics.loop_closure_count += 1

    def snapshot(self):
        result = EpisodePaperMetrics(**self._metrics.to_dict())
        result.pose_sample_count = len(self._translation_sq)
        if self._translation_sq:
            result.odometry_step_translation_rmse_cm = float(np.sqrt(np.mean(self._translation_sq)))
            result.odometry_step_rotation_rmse_deg = float(np.sqrt(np.mean(self._rotation_sq)))
        if self._sem_rewards:
            result.semantic_reward_mean = float(np.mean(self._sem_rewards))
        return result


def aggregate_episode_metrics(episodes: List[EpisodePaperMetrics]):
    if not episodes:
        return {}
    out = {}
    for key in episodes[0].to_dict():
        values = [ep.to_dict()[key] for ep in episodes if ep.to_dict()[key] is not None]
        out[key] = float(np.mean(values)) if values else None
        out[f'{key}_std'] = float(np.std(values)) if values else None
        out[f'{key}_count'] = len(values)
    return out
