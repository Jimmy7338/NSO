"""
基于 RGB-D 的体素/俯视投影补全，将单帧深度投影到俯视栅格并与全局地图融合。
"""

from typing import Optional, Tuple

import numpy as np

try:
    import cv2
except ImportError:
    cv2 = None

from env.utils import depth_utils as du


class VoxelBasedCompletion:
    def __init__(
        self,
        voxel_size: float = 0.05,
        voxel_grid_size: Tuple[int, int, int] = (200, 200, 50),
        map_resolution_cm: int = 5,
        use_semantic_segmentation: bool = True,
        device=None,
        agent_height: float = 1.25,
        agent_view_angle: float = 0.0,
        vision_range: int = 64,
        obs_threshold: float = 1.0,
        du_scale: int = 1,
    ):
        self.voxel_size = voxel_size
        self.voxel_grid_size = voxel_grid_size
        self.map_resolution_cm = map_resolution_cm
        self.use_semantic_segmentation = use_semantic_segmentation
        self.device = device
        self.agent_height = agent_height
        self.agent_view_angle = agent_view_angle
        self.vision_range = vision_range
        self.obs_threshold = obs_threshold
        self.du_scale = max(1, int(du_scale))
        self.z_bins = [0.0, 2.0]

    def _camera_matrix(self, frame_width: int, frame_height: int, hfov: float):
        return du.get_camera_matrix(frame_width, frame_height, hfov)

    def process_rgbd_frame(
        self,
        rgb_image: np.ndarray,
        depth_image: np.ndarray,
        camera_pose: np.ndarray,
        frame_width: int,
        frame_height: int,
        hfov: float,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        将 RGB-D 投影为以智能体为中心的局部俯视占据图与语义标签图。

        Returns:
            topdown_occupancy: (vision_range, vision_range) float
            topdown_semantic: (vision_range, vision_range) int, 0=无, 1..C=类别占位
        """
        grid = self.vision_range
        topdown_occ = np.zeros((grid, grid), dtype=np.float32)
        topdown_sem = np.zeros((grid, grid), dtype=np.int32)

        if cv2 is None:
            return topdown_occ, topdown_sem

        depth = np.asarray(depth_image, dtype=np.float32)
        if depth.ndim == 3:
            depth = depth[:, :, 0]
        depth = np.nan_to_num(depth, nan=0.0, posinf=0.0, neginf=0.0)
        max_range_m = self.vision_range * self.map_resolution_cm / 100.0
        depth[depth <= 0] = np.nan
        depth[depth > max_range_m] = np.nan

        cam = self._camera_matrix(frame_width, frame_height, hfov)
        pc = du.get_point_cloud_from_z(depth, cam, scale=self.du_scale)
        agent_view = du.transform_camera_view(pc, self.agent_height, self.agent_view_angle)

        shift = [grid * self.map_resolution_cm / 100.0 / 2.0, 0.0, np.pi / 2.0]
        centered = du.transform_pose(agent_view, shift)

        pose = np.asarray(camera_pose, dtype=np.float64).reshape(-1)
        if pose.size >= 3 and abs(pose[2]) > 10.0:
            pose = pose.copy()
            pose[2] = np.deg2rad(pose[2])

        binned = du.bin_points(
            centered,
            grid,
            self.z_bins,
            self.map_resolution_cm,
        )
        obstacle = binned[:, :, 1] / max(self.obs_threshold, 1e-6)
        obstacle = (obstacle >= 0.5).astype(np.float32)
        explored = (binned.sum(axis=2) > 0).astype(np.float32)

        topdown_occ = np.maximum(obstacle, explored * 0.5)

        if self.use_semantic_segmentation and rgb_image is not None:
            rgb = np.asarray(rgb_image)
            if rgb.ndim == 3 and rgb.shape[0] == 3:
                rgb = np.transpose(rgb, (1, 2, 0))
            if rgb.ndim == 3:
                small = cv2.resize(rgb, (grid, grid), interpolation=cv2.INTER_AREA)
                gray = cv2.cvtColor(small.astype(np.uint8), cv2.COLOR_RGB2GRAY)
                # 粗粒度“语义”占位：亮/暗/中 → 1/2/3
                labels = np.zeros((grid, grid), dtype=np.int32)
                labels[gray > 170] = 1
                labels[(gray > 85) & (gray <= 170)] = 2
                labels[gray <= 85] = 3
                topdown_sem = labels * (topdown_occ > 0.1).astype(np.int32)

        return topdown_occ, topdown_sem

    def fuse_with_existing_map(
        self,
        new_topdown_occupancy: np.ndarray,
        new_topdown_semantic: np.ndarray,
        existing_semantic_map: np.ndarray,
        existing_explored_map: np.ndarray,
        map_origin: np.ndarray,
        map_resolution_cm: int,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """将局部俯视结果融合进全局语义计数图与探索图。"""
        existing_sem = np.asarray(existing_semantic_map, dtype=np.float32)
        existing_exp = np.asarray(existing_explored_map, dtype=np.float32)

        if existing_sem.ndim == 2:
            existing_sem = existing_sem[np.newaxis, ...]

        num_classes, h, w = existing_sem.shape
        occ = np.asarray(new_topdown_occupancy, dtype=np.float32)
        sem = np.asarray(new_topdown_semantic, dtype=np.int32)

        if occ.shape != (h, w):
            occ = cv2.resize(occ, (w, h), interpolation=cv2.INTER_LINEAR) if cv2 else occ
        if sem.shape != (h, w):
            sem = (
                cv2.resize(sem.astype(np.float32), (w, h), interpolation=cv2.INTER_NEAREST).astype(np.int32)
                if cv2
                else sem
            )

        fused_exp = np.maximum(existing_exp, (occ > 0.2).astype(np.float32))
        fused_sem = existing_sem.copy()

        mask = occ > 0.2
        for label in np.unique(sem[mask]):
            li = int(label)
            if li <= 0 or li > num_classes:
                continue
            cell_mask = mask & (sem == li)
            fused_sem[li - 1][cell_mask] += occ[cell_mask]

        return fused_sem, fused_exp
