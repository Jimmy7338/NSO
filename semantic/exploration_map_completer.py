"""
探索地图补全：在已探索区域基础上，推断未观测但很可能可通行的区域。
基于几何传播、语义密度与门框结构先验，规则实现（无需额外模型）。
"""

from typing import Optional, Tuple

import numpy as np

try:
    import cv2
except ImportError:
    cv2 = None


class ExplorationMapCompleter:
    def __init__(
        self,
        confidence_thresh: float = 0.5,
        max_completion_distance: int = 15,
    ):
        self.confidence_thresh = confidence_thresh
        self.max_completion_distance = max(1, int(max_completion_distance))

    def complete_exploration_map(
        self,
        explored_map: np.ndarray,
        obstacle_map: np.ndarray,
        semantic_map: Optional[np.ndarray] = None,
        door_map: Optional[np.ndarray] = None,
        visited_map: Optional[np.ndarray] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        explored = np.asarray(explored_map, dtype=np.float32)
        obstacle = np.asarray(obstacle_map, dtype=np.float32)
        h, w = explored.shape

        free = (obstacle < 0.5).astype(np.uint8)
        known_explored = ((explored > 0.25) & free).astype(np.uint8)

        if cv2 is None:
            return explored.copy(), np.where(known_explored > 0, 1.0, 0.0).astype(np.float32)

        if not np.any(known_explored):
            return explored.copy(), np.zeros((h, w), dtype=np.float32)

        # 到最近已探索自由格的距离
        inv = (1 - known_explored).astype(np.uint8)
        dist = cv2.distanceTransform(inv, cv2.DIST_L2, 5)

        # 仅在可通行、尚未标为已探索的区域内补全
        candidate = (free > 0) & (explored <= 0.25) & (dist > 0) & (dist <= self.max_completion_distance)
        if not np.any(candidate):
            return explored.copy(), np.where(known_explored > 0, 1.0, 0.3).astype(np.float32)

        # 局部开阔度：已探索邻域内的自由比例
        k = min(9, max(3, self.max_completion_distance // 2) | 1)
        free_float = (free * known_explored).astype(np.float32)
        openness = cv2.blur(free_float, (k, k))
        if openness.max() > 0:
            openness = openness / (openness.max() + 1e-6)

        # 距离越近、开阔度越高，置信度越高
        dist_score = np.clip(1.0 - dist / float(self.max_completion_distance), 0.0, 1.0)
        confidence = dist_score * (0.5 + 0.5 * openness)
        confidence = confidence * candidate.astype(np.float32)

        # 语义引导：高语义密度区域更可能延伸探索
        if semantic_map is not None:
            sem = np.asarray(semantic_map, dtype=np.float32)
            if sem.ndim == 3:
                sem = np.sum(sem, axis=0)
            if sem.shape == (h, w):
                smax = float(sem.max()) + 1e-6
                sem_norm = sem / smax
                sem_blur = cv2.GaussianBlur(sem_norm, (k, k), 0)
                confidence = np.clip(confidence + 0.25 * sem_blur * candidate, 0.0, 1.0)

        # 门框先验：门洞附近更可能连通未探索房间
        if door_map is not None:
            door = np.asarray(door_map, dtype=np.float32)
            if door.shape == (h, w) and np.any(door > 0):
                door_u8 = (door > 0.1).astype(np.uint8)
                door_prox = cv2.distanceTransform((1 - door_u8).astype(np.uint8), cv2.DIST_L2, 3)
                door_boost = np.clip(1.0 - door_prox / max(self.max_completion_distance, 1), 0.0, 1.0)
                confidence = np.clip(confidence + 0.35 * door_boost * candidate, 0.0, 1.0)

        # 形态学闭运算：连接窄缝内的可探索区域
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        morph_mask = cv2.morphologyEx(
            (confidence >= self.confidence_thresh).astype(np.uint8),
            cv2.MORPH_CLOSE,
            kernel,
        )
        morph_mask = morph_mask & candidate.astype(np.uint8)

        completed = explored.copy()
        apply_mask = morph_mask.astype(bool)
        completed[apply_mask] = np.maximum(completed[apply_mask], confidence[apply_mask])

        # 已探索区域置信度为 1
        out_conf = confidence.copy()
        out_conf[known_explored > 0] = 1.0
        out_conf[~free.astype(bool)] = 0.0

        return completed, out_conf.astype(np.float32)
