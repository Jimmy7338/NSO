"""
语义地图更新器：封装 SemanticMap2D 的检测更新与可选 SSC/体素融合流程。
"""

from typing import Any, Dict, Optional

import numpy as np
import torch

from semantic.semantic_map import SemanticMap2D


class SemanticMapUpdater:
    """对 SemanticMap2D 的轻量封装，便于在训练循环中统一调用。"""

    def __init__(self, semantic_map: SemanticMap2D):
        self.semantic_map = semantic_map

    def update_from_detections(
        self,
        scene_idx: int,
        detections: Dict[str, Any],
        local_map: torch.Tensor,
        local_pose: torch.Tensor,
        lmb_e: np.ndarray,
        full_pose: Optional[torch.Tensor] = None,
    ) -> None:
        self.semantic_map.update_with_detections(
            scene_idx=scene_idx,
            detections=detections,
            local_map=local_map,
            local_pose=local_pose,
            lmb_e=lmb_e,
            full_pose=full_pose,
        )

    def apply_ssc(
        self,
        scene_idx: int,
        ssc_completer,
        explored_map: np.ndarray,
        obstacle_map: np.ndarray,
        visited_map: Optional[np.ndarray] = None,
        door_map: Optional[np.ndarray] = None,
        save_comparison: bool = False,
        step_global: int = 0,
    ) -> None:
        self.semantic_map.apply_ssc_completion(
            scene_idx=scene_idx,
            ssc_completer=ssc_completer,
            explored_map=explored_map,
            obstacle_map=obstacle_map,
            visited_map=visited_map,
            door_map=door_map,
            save_comparison=save_comparison,
            step_global=step_global,
        )

    def apply_voxel(
        self,
        scene_idx: int,
        topdown_occupancy: np.ndarray,
        topdown_semantic: np.ndarray,
        map_origin: np.ndarray,
        map_resolution_cm: int,
    ) -> None:
        self.semantic_map.update_with_voxel_completion(
            scene_idx=scene_idx,
            topdown_occupancy=topdown_occupancy,
            topdown_semantic=topdown_semantic,
            map_origin=map_origin,
            map_resolution_cm=map_resolution_cm,
        )

    def get_density_window(
        self, scene_idx: int, gx1: int, gx2: int, gy1: int, gy2: int
    ) -> np.ndarray:
        return self.semantic_map.get_full_density_window(scene_idx, gx1, gx2, gy1, gy2)
