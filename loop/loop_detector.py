from dataclasses import dataclass
from typing import List, Optional, Dict

import numpy as np
import faiss


@dataclass
class LoopMatch:
    matched_step: int
    current_step: int
    distance: float
    semantic_sim: float
    matched_pose: np.ndarray
    current_pose: np.ndarray
    entry_index: int


class LoopDetector:
    """
    使用 FAISS 基于语义增强 NetVLAD 描述子进行回环检索。
    """

    def __init__(self,
                 descriptor_dim: int,
                 semantic_dim: int,
                 sim_thresh: float = 0.75,
                 semantic_thresh: float = 0.6,
                 min_step_gap: int = 200,
                 top_k: int = 5,
                 min_db_size: int = 10,
                 max_db_size: int = 5000):
        self.descriptor_dim = descriptor_dim
        self.semantic_dim = semantic_dim
        self.sim_thresh = sim_thresh
        self.semantic_thresh = semantic_thresh
        self.min_step_gap = min_step_gap
        self.top_k = top_k
        self.min_db_size = min_db_size
        self.max_db_size = max_db_size

        self.index = faiss.IndexFlatIP(descriptor_dim)
        self.entries: List[Dict] = []

    @staticmethod
    def _to_numpy(vec) -> np.ndarray:
        if isinstance(vec, np.ndarray):
            return vec
        return np.asarray(vec, dtype=np.float32)

    def _rebuild_index(self):
        self.index.reset()
        if not self.entries:
            return
        data = np.stack([entry["descriptor"] for entry in self.entries]).astype(np.float32)
        self.index.add(data)

    def add_keyframe(self,
                     descriptor: np.ndarray,
                     pose: np.ndarray,
                     step: int):
        descriptor = self._to_numpy(descriptor).astype(np.float32)
        if descriptor.shape[0] != self.descriptor_dim:
            raise ValueError(f"描述子维度不匹配: {descriptor.shape[0]} vs {self.descriptor_dim}")

        pose = self._to_numpy(pose).astype(np.float32)
        entry = {
            "descriptor": descriptor,
            "semantic": descriptor[-self.semantic_dim:],
            "pose": pose,
            "step": int(step),
        }

        # 优化：避免频繁重建索引，使用增量添加
        if len(self.entries) >= self.max_db_size:
            # 移除最旧的条目
            self.entries.pop(0)
            # 只有在索引很大时才重建，否则直接添加新条目
            if len(self.entries) > 1000:
                self._rebuild_index()
            else:
                # 对于小索引，直接添加更高效
                self.index.add(descriptor.reshape(1, -1))
                self.entries.append(entry)
                return

        self.entries.append(entry)
        self.index.add(descriptor.reshape(1, -1))

    def detect_loop(self,
                    descriptor: np.ndarray,
                    pose: np.ndarray,
                    step: int) -> Optional[LoopMatch]:
        if len(self.entries) < self.min_db_size:
            return None

        descriptor = self._to_numpy(descriptor).astype(np.float32)
        pose = self._to_numpy(pose).astype(np.float32)

        distances, indices = self.index.search(descriptor.reshape(1, -1), self.top_k)
        distances = distances.flatten()
        indices = indices.flatten()

        semantic_vec = descriptor[-self.semantic_dim:]
        semantic_norm = np.linalg.norm(semantic_vec) + 1e-6

        best_match = None

        for dist, idx in zip(distances, indices):
            if idx < 0 or idx >= len(self.entries):
                continue
            if dist < self.sim_thresh:
                continue

            entry = self.entries[idx]
            step_gap = abs(step - entry["step"])
            if step_gap < self.min_step_gap:
                continue

            sem_entry = entry["semantic"]
            sem_sim = float(np.dot(semantic_vec, sem_entry) /
                            ((np.linalg.norm(sem_entry) + 1e-6) * semantic_norm))
            if sem_sim < self.semantic_thresh:
                continue

            best_match = LoopMatch(
                matched_step=entry["step"],
                current_step=int(step),
                distance=float(dist),
                semantic_sim=sem_sim,
                matched_pose=entry["pose"],
                current_pose=pose,
                entry_index=int(idx),
            )
            break

        return best_match

    def size(self) -> int:
        return len(self.entries)


