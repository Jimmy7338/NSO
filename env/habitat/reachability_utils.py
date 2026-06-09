"""FMM 可达性标签与论文式 (2) 全局目标掩码。"""
from __future__ import annotations

import numpy as np
import skimage.morphology

from env.utils.fmm_planner import FMMPlanner


def build_traversible_local(grid: np.ndarray, explored: np.ndarray,
                            visited: np.ndarray, collision: np.ndarray,
                            selem) -> np.ndarray:
    """局部规划窗口内的可通行栅格（grid/visited/collision 同尺寸）。"""
    patch = np.asarray(grid, dtype=np.float32)
    traversible = skimage.morphology.binary_dilation(patch, selem) != True
    traversible = traversible.astype(np.float32)
    traversible[np.asarray(collision) > 0.5] = 0.0
    traversible[np.asarray(visited) > 0.5] = 1.0
    return traversible


def build_traversible(grid: np.ndarray, explored: np.ndarray, visited: np.ndarray,
                      collision: np.ndarray, gx1: int, gx2: int, gy1: int, gy2: int,
                      selem) -> np.ndarray:
    """全图坐标下的局部窗口（保留兼容）。"""
    return build_traversible_local(
        grid[gx1:gx2, gy1:gy2],
        explored[gx1:gx2, gy1:gy2] if explored.shape != grid.shape else explored,
        visited[gx1:gx2, gy1:gy2],
        collision[gx1:gx2, gy1:gy2],
        selem,
    )


def fmm_reachability_map(traversible: np.ndarray, start: tuple, dt: int = 10) -> np.ndarray:
    """
    由当前位姿出发的 FMM 距离场，转为 [0,1] 可达性（越近越大）。
  start: (row, col) 与 _get_stg 一致。
    """
    h, w = traversible.shape
    planner = FMMPlanner(traversible.astype(np.double), 360 // dt)
    goal = [int(np.clip(start[1], 0, w - 1)), int(np.clip(start[0], 0, h - 1))]
    mask = planner.set_goal(goal)
    if not np.any(mask):
        return np.zeros((h, w), dtype=np.float32)
    dist = np.array(planner.fmm_dist, dtype=np.float32)
    dist = np.where(np.isfinite(dist), dist, dist[~np.isnan(dist)].max() + 1.0
                    if np.any(np.isfinite(dist)) else 1.0)
    dmax = float(np.max(dist)) + 1e-6
    reach = 1.0 - np.clip(dist / dmax, 0.0, 1.0)
    return reach.astype(np.float32)


def goal_reachable_label(traversible: np.ndarray, start: tuple, goal: tuple,
                         dt: int = 10) -> float:
    """单点标签：FMM 能否规划到 goal（无 replan）。"""
    h, w = traversible.shape
    gx, gy = int(np.clip(goal[0], 0, h - 1)), int(np.clip(goal[1], 0, w - 1))
    planner = FMMPlanner(traversible.astype(np.double), 360 // dt)
    if not np.any(planner.set_goal([gy, gx])):
        return 0.0
    sx, sy = int(np.clip(start[0], 0, h - 1)), int(np.clip(start[1], 0, w - 1))
    _, _, replan = planner.get_short_term_goal([sx, sy])
    return 0.0 if replan else 1.0


def mask_global_goals(
    raw_goals,
    reach_maps,
    local_w: int,
    local_h: int,
    alpha: float = 2.0,
    eps: float = 1e-4,
    num_candidates: int = 16,
    free_maps=None,
):
    """
    论文式 (2) 的离散近似：在候选栅格上 Softmax(log π + α log M_reach)。
    raw_goals: list of [gx, gy]；reach_maps: (N,H,W) numpy。
    """
    masked = []
    for e, (gx, gy) in enumerate(raw_goals):
        m = reach_maps[e]
        free = free_maps[e] if free_maps is not None else (m > 0.05)
        ys, xs = np.where(free)
        if len(xs) == 0:
            masked.append([int(gx), int(gy)])
            continue
        if num_candidates > 0 and len(xs) > num_candidates:
            idx = np.random.choice(len(xs), num_candidates, replace=False)
            xs, ys = xs[idx], ys[idx]
        log_reach = np.log(m[ys, xs] + eps) * alpha
        dist_pen = -((xs - gy) ** 2 + (ys - gx) ** 2).astype(np.float32) / max(local_w, 1)
        logits = log_reach + 0.01 * dist_pen
        logits -= logits.max()
        prob = np.exp(logits)
        prob /= prob.sum() + eps
        pick = int(np.argmax(prob))
        masked.append([int(ys[pick]), int(xs[pick])])
    return masked
