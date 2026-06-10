"""论文对齐评估指标：覆盖率、轨迹漂移、无效目标频次等。"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np


@dataclass
class EpisodePaperMetrics:
    """单回合论文指标快照。"""

    explored_area_m2: float = 0.0
    coverage_ratio: float = 0.0
    trajectory_drift_rmse_cm: float = 0.0
    unreachable_goal_count: int = 0
    loop_closure_count: int = 0
    semantic_reward_mean: float = 0.0
    embodied_goal_success_rate: float = 0.0
    step_count: int = 0

    def to_dict(self) -> Dict[str, float]:
        return {
            "explored_area_m2": self.explored_area_m2,
            "coverage_ratio": self.coverage_ratio,
            "trajectory_drift_rmse_cm": self.trajectory_drift_rmse_cm,
            "unreachable_goal_count": self.unreachable_goal_count,
            "loop_closure_count": self.loop_closure_count,
            "semantic_reward_mean": self.semantic_reward_mean,
            "embodied_goal_success_rate": self.embodied_goal_success_rate,
            "step_count": self.step_count,
        }


class PaperMetricsTracker:
    """
    在线累积论文实验指标，供训练/评估日志与消融实验汇总。
    """

    def __init__(self, map_resolution_cm: float = 5.0):
        self.map_resolution_cm = map_resolution_cm
        self._pose_err_sq: List[float] = []
        self._sem_rewards: List[float] = []
        self._unreachable: int = 0
        self._loop_closures: int = 0
        self._embodied_attempts: int = 0
        self._embodied_successes: int = 0
        self._last_coverage: float = 0.0
        self._last_area_m2: float = 0.0
        self._steps: int = 0

    def reset_episode(self):
        self._pose_err_sq = []
        self._sem_rewards = []
        self._unreachable = 0
        self._loop_closures = 0
        self._embodied_attempts = 0
        self._embodied_successes = 0
        self._last_coverage = 0.0
        self._last_area_m2 = 0.0
        self._steps = 0

    def update_step(
        self,
        pose_err: Optional[List[float]] = None,
        sem_reward: Optional[float] = None,
        exp_ratio: Optional[float] = None,
        exp_reward: Optional[float] = None,
        unreachable: bool = False,
        loop_detected: bool = False,
        embodied_success: Optional[bool] = None,
    ):
        self._steps += 1
        if pose_err is not None:
            dx, dy, do = pose_err
            # 平移误差 (cm) + 旋转误差折算为等效位移 (cm)
            drift_cm = np.sqrt((dx * 100) ** 2 + (dy * 100) ** 2)
            rot_equiv_cm = abs(np.rad2deg(do)) * 0.5
            self._pose_err_sq.append(drift_cm ** 2 + rot_equiv_cm ** 2)
        if sem_reward is not None:
            self._sem_rewards.append(float(sem_reward))
        if exp_ratio is not None:
            self._last_coverage = float(exp_ratio)
        if exp_reward is not None and exp_ratio is not None:
            # eval 模式下 exp_reward 已按面积缩放
            self._last_area_m2 = float(exp_reward)
        if unreachable:
            self._unreachable += 1
        if loop_detected:
            self._loop_closures += 1
        if embodied_success is not None:
            self._embodied_attempts += 1
            if embodied_success:
                self._embodied_successes += 1

    def snapshot(self) -> EpisodePaperMetrics:
        drift_rmse = 0.0
        if self._pose_err_sq:
            drift_rmse = float(np.sqrt(np.mean(self._pose_err_sq)))
        success_rate = 0.0
        if self._embodied_attempts > 0:
            success_rate = self._embodied_successes / self._embodied_attempts
        sem_mean = float(np.mean(self._sem_rewards)) if self._sem_rewards else 0.0
        return EpisodePaperMetrics(
            explored_area_m2=self._last_area_m2,
            coverage_ratio=self._last_coverage,
            trajectory_drift_rmse_cm=drift_rmse,
            unreachable_goal_count=self._unreachable,
            loop_closure_count=self._loop_closures,
            semantic_reward_mean=sem_mean,
            embodied_goal_success_rate=success_rate,
            step_count=self._steps,
        )


def aggregate_episode_metrics(
    episodes: List[EpisodePaperMetrics],
) -> Dict[str, float]:
    """多回合均值汇总，用于论文表格生成。"""
    if not episodes:
        return {}
    keys = episodes[0].to_dict().keys()
    out: Dict[str, float] = {}
    for k in keys:
        vals = [ep.to_dict()[k] for ep in episodes]
        out[k] = float(np.mean(vals))
        out[f"{k}_std"] = float(np.std(vals))
    return out
