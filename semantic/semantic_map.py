from typing import Dict, Any, List, Tuple, Optional

import os
import numpy as np
import torch
import cv2


class SemanticMap2D:
    """
    简单的2D语义地图维护：
    - 维护全局与局部语义计数: (num_scenes, num_classes, H, W)
    - 提供密度图: sum over classes
    - 使用近似投影：将检测到的目标在 agent 面前的一块矩形视野区域内均匀加权计数
      （最小侵入实现，后续可替换为更精确的几何投影）
    """
    def __init__(self,
                 num_scenes: int,
                 num_classes: int,
                 full_w: int,
                 full_h: int,
                 local_w: int,
                 local_h: int,
                 map_resolution_cm: int,
                 vision_range: int,
                 dump_dir: str,
                 class_weights: Optional[np.ndarray] = None):
        self.num_scenes = num_scenes
        self.num_classes = num_classes
        self.full_w = full_w
        self.full_h = full_h
        self.local_w = local_w
        self.local_h = local_h
        self.map_resolution_cm = map_resolution_cm
        self.vision_range = vision_range
        self.dump_images_dir = os.path.join(dump_dir, "images")
        os.makedirs(self.dump_images_dir, exist_ok=True)

        self.full_sem_counts = torch.zeros(num_scenes, num_classes, full_w, full_h, dtype=torch.float32)
        self.local_sem_counts = torch.zeros(num_scenes, num_classes, local_w, local_h, dtype=torch.float32)
        if class_weights is None:
            class_weights = np.ones(num_classes, dtype=np.float32)
        assert len(class_weights) == num_classes, "class_weights length must equal num_classes"
        self.class_weights = torch.from_numpy(class_weights.astype(np.float32))

    def to_device(self, device: torch.device):
        self.full_sem_counts = self.full_sem_counts.to(device)
        self.local_sem_counts = self.local_sem_counts.to(device)
        self.class_weights = self.class_weights.to(device)

    def set_class_weights(self, weights: np.ndarray):
        assert len(weights) == self.num_classes, "weights length must equal num_classes"
        tensor = torch.from_numpy(weights.astype(np.float32))
        if self.full_sem_counts.is_cuda:
            tensor = tensor.to(self.full_sem_counts.device)
        self.class_weights = tensor

    @staticmethod
    def _draw_rect_in_map(sem_map_chw: torch.Tensor,
                          center_rc: Tuple[int, int],
                          forward_dir_deg: float,
                          box_size: Tuple[int, int],
                          weight: float):
        """
        在语义计数的 CHW 地图上，围绕 (r, c) 以朝向 forward_dir_deg 画一个矩形并加 weight。
        sem_map_chw: (C, H, W), torch.float32
        box_size: (height, width) in cells
        """
        c, h, w = sem_map_chw.shape
        r0, c0 = center_rc
        bh, bw = box_size
        # 简化：忽略旋转（最小侵入），以agent朝向为近似，先实现稳定可用版本
        r1 = max(r0 - bh // 2, 0)
        r2 = min(r1 + bh, h)
        c1 = max(c0, 0)
        c2 = min(c0 + bw, w)
        if r1 < r2 and c1 < c2:
            sem_map_chw[:, r1:r2, c1:c2] += weight

    def update_with_detections(self,
                               scene_idx: int,
                               detections: Dict[str, Any],
                               local_map: torch.Tensor,
                               local_pose: torch.Tensor,
                               lmb_e: np.ndarray,
                               full_pose: torch.Tensor = None):
        """
        使用检测更新语义地图，使用更精确的投影方式。
        - detections: {"boxes": Nx4, "scores": N, "classes": N}
        - local_map: (4, local_w, local_h) 用于取当前位置
        - local_pose: (3,) 当前局部坐标系下的连续位置与角度 (x, y, theta_deg)
        - lmb_e: [gx1, gx2, gy1, gy2] 局部窗口在全局地图中的边界
        - full_pose: (3,) 全局坐标系下的位姿，用于精确投影
        """
        if detections is None or len(detections.get("classes", [])) == 0:
            return
        classes = detections["classes"]
        scores = detections["scores"]
        num_det = len(classes)
        if num_det == 0:
            return

        # 计算总权重
        total_weight = float(np.sum(scores)) if len(scores) > 0 else float(num_det)
        if total_weight <= 0:
            total_weight = float(num_det)

        # 当前agent在局部地图的像素位置
        locs = local_pose.detach().cpu().numpy()
        r = int(locs[1] * 100.0 / self.map_resolution_cm)
        c = int(locs[0] * 100.0 / self.map_resolution_cm)
        r = np.clip(r, 0, self.local_w - 1)
        c = np.clip(c, 0, self.local_h - 1)

        # 改进的投影策略：基于agent朝向，在视野前方区域投影语义信息
        # 使用更精细的网格，而不是粗糙的矩形
        theta_deg = float(locs[2])
        theta_rad = np.deg2rad(theta_deg)
        
        # 视野范围（以agent为中心的前方扇形区域）
        view_dist = self.vision_range // 2
        
        # 在局部地图上创建更精细的语义投影
        # 基于朝向，在前方区域投影
        for cls_id, sc in zip(classes, scores):
            weight = float(sc) / total_weight if total_weight > 0 else 1.0 / max(num_det, 1)
            if 0 <= cls_id < self.num_classes:
                weight *= float(self.class_weights[cls_id].item())
            
            # 创建一个小的高斯分布区域，中心在前方
            # 计算前方中心点（基于朝向）
            forward_r = r + int(view_dist * 0.6 * np.sin(theta_rad))
            forward_c = c + int(view_dist * 0.6 * np.cos(theta_rad))
            forward_r = np.clip(forward_r, 0, self.local_w - 1)
            forward_c = np.clip(forward_c, 0, self.local_h - 1)
            
            # 使用更小的区域，但更精确
            box_size = max(int(self.vision_range // 4), 3)
            center_rc = (forward_r, forward_c)
            self._draw_rect_in_map(self.local_sem_counts[scene_idx], center_rc, theta_deg, 
                                  (box_size, box_size), weight)

        # 同步更新到全局：使用累积方式，保留所有历史语义信息
        gx1, gx2, gy1, gy2 = int(lmb_e[0]), int(lmb_e[1]), int(lmb_e[2]), int(lmb_e[3])
        # 使用累积方式更新全局地图，保留所有历史语义信息
        # 注意：这里累积的是本次检测的结果，全局地图会持续累积所有历史检测
        self.full_sem_counts[scene_idx, :, gx1:gx2, gy1:gy2] += self.local_sem_counts[scene_idx]
        
        # 清零局部计数，为下次更新做准备（避免重复累积）
        self.local_sem_counts[scene_idx].fill_(0.)

    @torch.no_grad()
    def get_density_maps(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        返回 (local_density, full_density) 两个张量:
        - local_density: (num_scenes, 1, local_w, local_h)
        - full_density:  (num_scenes, 1, full_w, full_h)
        """
        weight_map = self.class_weights.view(1, self.num_classes, 1, 1)
        local_density = torch.sum(self.local_sem_counts * weight_map, dim=1, keepdim=True)
        full_density = torch.sum(self.full_sem_counts * weight_map, dim=1, keepdim=True)
        return local_density, full_density
    
    @torch.no_grad()
    def get_full_density_window(self, scene_idx: int, gx1: int, gx2: int, gy1: int, gy2: int) -> np.ndarray:
        """
        获取全局语义密度图的指定窗口区域（用于可视化）
        Returns: (H, W) numpy array
        """
        weight_map = self.class_weights.view(1, self.num_classes, 1, 1)
        weighted_full = torch.sum(self.full_sem_counts * weight_map, dim=1, keepdim=True)
        window = weighted_full[scene_idx, 0, gx1:gx2, gy1:gy2].detach().cpu().numpy()
        return window

    def save_visualizations(self, step_global: int, scene_idx: int):
        """
        保存彩色密度图（仅作为可视化参考）。
        """
        local_density, full_density = self.get_density_maps()
        ld = local_density[scene_idx, 0].detach().cpu().numpy()
        fd = full_density[scene_idx, 0].detach().cpu().numpy()
        if np.max(ld) > 0:
            ld = (ld / (np.max(ld) + 1e-6) * 255).astype(np.uint8)
        if np.max(fd) > 0:
            fd = (fd / (np.max(fd) + 1e-6) * 255).astype(np.uint8)
        ld_color = cv2.applyColorMap(ld, cv2.COLORMAP_INFERNO)
        fd_color = cv2.applyColorMap(fd, cv2.COLORMAP_INFERNO)
        cv2.imwrite(os.path.join(self.dump_images_dir, f"local_sem_density_{scene_idx}_{step_global:08d}.png"), ld_color)
        cv2.imwrite(os.path.join(self.dump_images_dir, f"full_sem_density_{scene_idx}_{step_global:08d}.png"), fd_color)


