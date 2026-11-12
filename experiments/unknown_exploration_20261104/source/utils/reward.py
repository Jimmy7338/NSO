"""
NSO 融合奖励函数

对应论文第 4.5 节，公式 (10)–(14)。

R_total = R_ig + λ_sem · R_sem + λ_struct · R_struct + λ_front · R_front + R_intrinsic

各分项：
  R_ig       : 信息增益覆盖奖励（互信息近似，替代简单面积增量）
  R_sem      : 开放词汇语义密度奖励（fresh mask 归一化）
  R_struct   : 拓扑结构感知奖励（门框/走廊形态学）
  R_front    : 前沿引导奖励（门框邻域 Boost）
  R_intrinsic: 重复访问惩罚
"""
from __future__ import annotations

import math
from typing import List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# 1. 信息增益覆盖奖励（IGCR）
# ---------------------------------------------------------------------------

def compute_igcr(
    explored_prev: np.ndarray,
    explored_curr: np.ndarray,
    obstacle_prob: Optional[np.ndarray] = None,
    resolution_m: float = 0.05,
) -> float:
    """
    信息增益覆盖奖励（论文公式 10）。

    R_ig = Σ_{(i,j) ∈ Δexp} H(M_obs(i,j))

    其中 Δexp 为本步新增探索格点，H(p) = -p log p - (1-p) log(1-p)。

    若 obstacle_prob 未提供，等价于面积增量（每格贡献 1）。

    Parameters
    ----------
    explored_prev   : (H, W) 上一步探索地图（0/1）
    explored_curr   : (H, W) 当前探索地图（0/1）
    obstacle_prob   : (H, W) 占据概率估计 ∈ [0,1]，None 则降级为面积增量
    resolution_m    : 地图分辨率（m/格点），用于面积归一化

    Returns
    -------
    reward : float
    """
    delta = (explored_curr > explored_prev).astype(np.float32)
    n_new = delta.sum()

    if n_new == 0:
        return 0.0

    if obstacle_prob is None:
        # 降级：简单面积增量（与 ANS 一致）
        return float(n_new) * (resolution_m ** 2)

    # 信息增益近似：对新增格点处的占据概率计算二元熵
    p = obstacle_prob[delta > 0].astype(np.float64)
    p = np.clip(p, 1e-6, 1.0 - 1e-6)
    entropy = -(p * np.log2(p) + (1 - p) * np.log2(1 - p))
    return float((entropy * delta[delta > 0]).sum())


# ---------------------------------------------------------------------------
# 2. 语义密度奖励（R_sem）
# ---------------------------------------------------------------------------

def compute_semantic_reward(
    sem_density: np.ndarray,
    fresh_mask: np.ndarray,
    sigma: float = 1e-6,
) -> float:
    """
    基于 fresh mask 的归一化开放词汇语义密度奖励（论文公式 11）。

    R_sem = (1/N_active) · Σ_{(i,j)} [ S^fused_{i,j} · M_fresh(i,j) ] / (max S + σ)

    Parameters
    ----------
    sem_density : (H, W) float32 OV-SDF 融合语义密度场
    fresh_mask  : (H, W) binary，已观测-已访问的 fresh 区域
    sigma       : 平滑项

    Returns
    -------
    reward : float
    """
    active = float(fresh_mask.sum())
    if active < 1.0:
        return 0.0

    sem_max = float(sem_density.max())
    weighted = sem_density * fresh_mask
    reward = float(weighted.sum()) / (active * (sem_max + sigma))
    return reward


# ---------------------------------------------------------------------------
# 3. 结构感知奖励（R_struct）
# ---------------------------------------------------------------------------

def compute_structural_reward(
    obstacle_map: np.ndarray,
    explored_map: np.ndarray,
    fresh_mask: np.ndarray,
    doorway_cells: Optional[List[np.ndarray]] = None,
    w_door: float = 2.0,
    w_narrow: float = 1.0,
    w_open: float = 0.5,
    narrow_width: int = 4,
    open_kernel: int = 9,
    resolution_m: float = 0.05,
) -> float:
    """
    形态学拓扑结构感知奖励（论文第 4.5.3 节）。

    在 fresh_mask 区域内，对三种结构类型加权：
      - 门框区域：w_door（由拓扑图门框格点或形态学窄通道判定）
      - 狭窄通道：w_narrow
      - 开阔区域：w_open

    Parameters
    ----------
    obstacle_map  : (H, W) 二值障碍地图
    explored_map  : (H, W) 二值探索地图
    fresh_mask    : (H, W) binary fresh 区域
    doorway_cells : 拓扑图门框格点列表（STGHP 输出）
    """
    H, W = obstacle_map.shape
    reward = 0.0

    free_space = ((obstacle_map == 0) & (explored_map > 0)).astype(np.float32)

    # 门框标记图（来自 STGHP 拓扑边）
    door_map = np.zeros((H, W), dtype=np.float32)
    if doorway_cells:
        for cell in doorway_cells:
            r, c = int(cell[0]), int(cell[1])
            if 0 <= r < H and 0 <= c < W:
                for dr in range(-2, 3):
                    for dc in range(-2, 3):
                        nr, nc = r + dr, c + dc
                        if 0 <= nr < H and 0 <= nc < W:
                            door_map[nr, nc] = 1.0
    else:
        # 形态学估算：骨架局部宽度 < narrow_width 的格点
        try:
            from skimage.morphology import skeletonize
            from scipy.ndimage import binary_dilation
            skel = skeletonize(free_space > 0)
            skel_rows, skel_cols = np.where(skel)
            for r, c in zip(skel_rows.tolist(), skel_cols.tolist()):
                # 局部宽度估算
                r1, r2 = max(0, r - narrow_width), min(H, r + narrow_width)
                c1, c2 = max(0, c - narrow_width), min(W, c + narrow_width)
                local_free = (obstacle_map[r1:r2, c1:c2] == 0).sum()
                width_est = math.sqrt(float(local_free))
                if width_est < narrow_width:
                    door_map[r, c] = 1.0
        except ImportError:
            pass

    # 开阔区域标记：形态学膨胀判定
    try:
        from scipy.ndimage import uniform_filter
        local_free_density = uniform_filter(free_space, size=open_kernel)
        open_map = (local_free_density > 0.7).astype(np.float32)
    except ImportError:
        open_map = np.zeros((H, W), dtype=np.float32)

    # 狭窄通道 = 骨架上非门框、非开阔区域
    narrow_map = np.zeros((H, W), dtype=np.float32)
    try:
        from skimage.morphology import skeletonize
        skel = skeletonize(free_space > 0).astype(np.float32)
        narrow_map = skel * (1 - door_map) * (1 - open_map)
    except ImportError:
        pass

    # 加权求和（仅计 fresh_mask 内）
    reward += w_door * float((door_map * fresh_mask).sum())
    reward += w_narrow * float((narrow_map * fresh_mask).sum())
    reward += w_open * float((open_map * fresh_mask).sum())

    # 归一化
    total_fresh = float(fresh_mask.sum())
    if total_fresh > 0:
        reward /= total_fresh

    return reward


# ---------------------------------------------------------------------------
# 4. 前沿引导奖励（R_front）
# ---------------------------------------------------------------------------

def compute_frontier_reward(
    explored_map: np.ndarray,
    obstacle_map: np.ndarray,
    fresh_mask: np.ndarray,
    doorway_cells: Optional[List[np.ndarray]] = None,
    room_boost: float = 1.5,
    door_boost_radius: int = 5,
) -> float:
    """
    前沿区域奖励，对门框邻域内的前沿施加 Room Exploration Boost（论文第 4.5.4 节）。

    Parameters
    ----------
    explored_map   : (H, W)
    obstacle_map   : (H, W)
    fresh_mask     : (H, W)
    doorway_cells  : 拓扑图门框格点
    room_boost     : 门框邻域前沿的额外奖励系数
    door_boost_radius: 门框影响半径（格点）
    """
    H, W = explored_map.shape

    # 前沿：已探索自由格点邻域内有未探索格点
    try:
        from scipy.ndimage import binary_dilation
        free = (explored_map > 0) & (obstacle_map == 0)
        dilated = binary_dilation(free, iterations=1)
        frontier = dilated & (explored_map == 0) & (obstacle_map == 0)
    except ImportError:
        frontier = np.zeros((H, W), dtype=bool)

    frontier_fresh = frontier.astype(np.float32) * fresh_mask

    # 基础前沿奖励
    reward = float(frontier_fresh.sum())

    # 门框邻域 Boost
    if doorway_cells and reward > 0:
        door_boost_map = np.zeros((H, W), dtype=np.float32)
        for cell in doorway_cells:
            r, c = int(cell[0]), int(cell[1])
            r1 = max(0, r - door_boost_radius)
            r2 = min(H, r + door_boost_radius + 1)
            c1 = max(0, c - door_boost_radius)
            c2 = min(W, c + door_boost_radius + 1)
            door_boost_map[r1:r2, c1:c2] = 1.0

        boosted = frontier_fresh * door_boost_map
        reward += (room_boost - 1.0) * float(boosted.sum())

    # 归一化
    n_frontier = float(frontier.sum())
    if n_frontier > 0:
        reward /= n_frontier

    return reward


# ---------------------------------------------------------------------------
# 5. 内在惩罚（R_intrinsic）
# ---------------------------------------------------------------------------

def compute_intrinsic_penalty(
    goal_cell: Tuple[int, int],
    visited_map: np.ndarray,
    penalty: float = 0.1,
) -> float:
    """
    对选择已探索格点作为长期目标施加惩罚（抑制重复访问）。

    Parameters
    ----------
    goal_cell   : (row, col) 长期目标格点
    visited_map : (H, W) 已访问格点图（0/1）
    penalty     : 惩罚幅度

    Returns
    -------
    reward : float（负数）
    """
    r, c = goal_cell
    H, W = visited_map.shape
    if 0 <= r < H and 0 <= c < W and visited_map[r, c] > 0:
        return -penalty
    return 0.0


# ---------------------------------------------------------------------------
# 组合：完整融合奖励
# ---------------------------------------------------------------------------

class NSO_RewardComputer:
    """
    NSO 完整融合奖励计算器（论文公式 10）。

    R_total = R_ig + λ_sem·R_sem + λ_struct·R_struct + λ_front·R_front + R_intrinsic

    使用方式：
        reward_computer = NSO_RewardComputer(args)
        r = reward_computer.compute(
            explored_prev, explored_curr, obstacle_map, visited_map,
            sem_density, goal_cell, doorway_cells, obstacle_prob
        )
    """

    def __init__(
        self,
        lambda_sem: float = 0.12,
        lambda_struct: float = 0.12,
        lambda_front: float = 0.15,
        intrinsic_penalty: float = 0.1,
        w_door: float = 2.0,
        w_narrow: float = 1.0,
        w_open: float = 0.5,
        room_boost: float = 1.5,
        door_boost_radius: int = 5,
        narrow_width: int = 4,
        open_kernel: int = 9,
        resolution_m: float = 0.05,
        use_igcr: bool = True,
    ):
        self.lambda_sem = lambda_sem
        self.lambda_struct = lambda_struct
        self.lambda_front = lambda_front
        self.intrinsic_penalty = intrinsic_penalty
        self.w_door = w_door
        self.w_narrow = w_narrow
        self.w_open = w_open
        self.room_boost = room_boost
        self.door_boost_radius = door_boost_radius
        self.narrow_width = narrow_width
        self.open_kernel = open_kernel
        self.resolution_m = resolution_m
        self.use_igcr = use_igcr

    def compute_fresh_mask(
        self,
        obstacle_map: np.ndarray,
        visited_map: np.ndarray,
    ) -> np.ndarray:
        """
        fresh mask = clip(已观测障碍图 - 已访问图, 0, 1)
        即：已观测但尚未被仔细访问的区域。
        """
        obs_observed = (obstacle_map >= 0).astype(np.float32)
        visited = (visited_map > 0).astype(np.float32)
        return np.clip(obs_observed - visited, 0.0, 1.0)

    def compute(
        self,
        explored_prev: np.ndarray,
        explored_curr: np.ndarray,
        obstacle_map: np.ndarray,
        visited_map: np.ndarray,
        sem_density: Optional[np.ndarray] = None,
        goal_cell: Optional[Tuple[int, int]] = None,
        doorway_cells: Optional[List[np.ndarray]] = None,
        obstacle_prob: Optional[np.ndarray] = None,
    ) -> Tuple[float, dict]:
        """
        计算完整融合奖励。

        Returns
        -------
        total_reward : float
        breakdown    : dict，各分项奖励值（用于日志记录与消融分析）
        """
        fresh_mask = self.compute_fresh_mask(obstacle_map, visited_map)

        # R_ig：信息增益覆盖奖励
        r_ig = compute_igcr(
            explored_prev, explored_curr,
            obstacle_prob, self.resolution_m,
        )
        if not self.use_igcr:
            # 降级为面积增量（消融对比用）
            n_new = float((explored_curr > explored_prev).sum())
            r_ig = n_new * (self.resolution_m ** 2) * 0.02

        # R_sem：语义密度奖励
        r_sem = 0.0
        if sem_density is not None:
            r_sem = compute_semantic_reward(sem_density, fresh_mask)

        # R_struct：结构感知奖励
        r_struct = compute_structural_reward(
            obstacle_map, explored_curr, fresh_mask,
            doorway_cells=doorway_cells,
            w_door=self.w_door,
            w_narrow=self.w_narrow,
            w_open=self.w_open,
            narrow_width=self.narrow_width,
            open_kernel=self.open_kernel,
            resolution_m=self.resolution_m,
        )

        # R_front：前沿引导奖励
        r_front = compute_frontier_reward(
            explored_curr, obstacle_map, fresh_mask,
            doorway_cells=doorway_cells,
            room_boost=self.room_boost,
            door_boost_radius=self.door_boost_radius,
        )

        # R_intrinsic：重复访问惩罚
        r_intrinsic = 0.0
        if goal_cell is not None:
            r_intrinsic = compute_intrinsic_penalty(
                goal_cell, visited_map, self.intrinsic_penalty
            )

        total = (
            r_ig
            + self.lambda_sem * r_sem
            + self.lambda_struct * r_struct
            + self.lambda_front * r_front
            + r_intrinsic
        )

        breakdown = {
            "r_ig": r_ig,
            "r_sem": r_sem,
            "r_struct": r_struct,
            "r_front": r_front,
            "r_intrinsic": r_intrinsic,
            "r_total": total,
        }

        return total, breakdown

    @classmethod
    def from_args(cls, args) -> "NSO_RewardComputer":
        """从 argparse Namespace 构建奖励计算器。"""
        use_igcr = getattr(args, "use_igcr", True)
        return cls(
            lambda_sem=getattr(args, "semantic_reward_coeff", 0.12),
            lambda_struct=getattr(args, "structural_reward_coeff", 0.12),
            lambda_front=getattr(args, "frontier_reward_coeff", 0.15),
            intrinsic_penalty=getattr(args, "intrinsic_penalty", 0.1),
            w_door=getattr(args, "w_struct_door", 2.0),
            w_narrow=getattr(args, "w_struct_narrow", 1.0),
            w_open=getattr(args, "w_struct_open", 0.5),
            room_boost=getattr(args, "room_exploration_boost", 1.5),
            door_boost_radius=getattr(args, "door_boost_distance", 5),
            narrow_width=getattr(args, "narrow_width_cells", 4),
            open_kernel=getattr(args, "open_kernel", 9),
            resolution_m=getattr(args, "map_resolution", 5) / 100.0,
            use_igcr=use_igcr,
        )
