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

    def apply_ssc_completion(self,
                            scene_idx: int,
                            ssc_completer,
                            explored_map: np.ndarray,
                            obstacle_map: np.ndarray,
                            visited_map: Optional[np.ndarray] = None,
                            door_map: Optional[np.ndarray] = None,
                            rgb_image: Optional[np.ndarray] = None,
                            depth_image: Optional[np.ndarray] = None,
                            save_comparison: bool = False,
                            step_global: int = 0):
        """
        应用语义场景补全（SSC）来补全未观测区域的语义信息
        
        Args:
            scene_idx: 场景索引
            ssc_completer: SSCCompleter 实例
            explored_map: (H, W) 已探索区域
            obstacle_map: (H, W) 障碍物地图
            visited_map: (H, W) 已访问区域（可选）
            door_map: (H, W) 门框地图（可选）
            rgb_image: (H, W, 3) RGB 图像（深度学习模式需要）
            depth_image: (H, W) 深度图像（深度学习模式需要）
            save_comparison: 是否保存补全前后对比图
            step_global: 全局步数（用于保存对比图）
        """
        # 获取当前语义地图（numpy 格式）
        semantic_map_np = self.full_sem_counts[scene_idx].detach().cpu().numpy()  # (C, H, W)
        original_semantic_map = semantic_map_np.copy()  # 保存原始地图用于对比
        
        # 调用 SSC 补全器
        completed_semantic_map, confidence_map = ssc_completer.complete(
            semantic_map=semantic_map_np,
            explored_map=explored_map,
            obstacle_map=obstacle_map,
            visited_map=visited_map,
            door_map=door_map,
            rgb_image=rgb_image,
            depth_image=depth_image
        )
        
        # 保存对比可视化（如果启用）
        if save_comparison:
            try:
                self.save_ssc_comparison(
                    scene_idx=scene_idx,
                    step_global=step_global,
                    original_semantic_map=original_semantic_map,
                    completed_semantic_map=completed_semantic_map,
                    confidence_map=confidence_map,
                    explored_map=explored_map
                )
            except Exception as e:
                print(f"[SSC] 保存对比图失败: {e}")
        
        # 只更新置信度足够高的补全区域
        high_confidence_mask = confidence_map > ssc_completer.completer.confidence_thresh
        if np.any(high_confidence_mask):
            # 将补全后的语义信息更新到地图中
            # 使用加权平均：原有值 * (1 - confidence) + 补全值 * confidence
            for c in range(self.num_classes):
                original = self.full_sem_counts[scene_idx, c].detach().cpu().numpy()
                completed = completed_semantic_map[c]
                # 只在未探索区域应用补全
                unobserved_mask = (explored_map == 0) & high_confidence_mask
                if np.any(unobserved_mask):
                    updated = original.copy()
                    updated[unobserved_mask] = (
                        original[unobserved_mask] * (1 - confidence_map[unobserved_mask]) +
                        completed[unobserved_mask] * confidence_map[unobserved_mask]
                    )
                    self.full_sem_counts[scene_idx, c] = torch.from_numpy(updated).to(
                        self.full_sem_counts.device
                    )

    def update_with_voxel_completion(self,
                                     scene_idx: int,
                                     topdown_occupancy: np.ndarray,
                                     topdown_semantic: np.ndarray,
                                     map_origin: np.ndarray,
                                     map_resolution_cm: int):
        """
        使用体素补全结果更新语义地图
        
        Args:
            scene_idx: 场景索引
            topdown_occupancy: (H, W) 体素补全的占据地图
            topdown_semantic: (H, W) 体素补全的语义地图（类别ID）
            map_origin: (2,) 地图原点
            map_resolution_cm: 地图分辨率（厘米）
        """
        # 体素网格尺寸
        voxel_h, voxel_w = topdown_occupancy.shape
        
        # 地图尺寸
        map_h, map_w = self.full_sem_counts.shape[2], self.full_sem_counts.shape[3]
        
        # 计算缩放比例（体素网格到地图网格）
        scale_x = map_w / voxel_w if voxel_w > 0 else 1.0
        scale_y = map_h / voxel_h if voxel_h > 0 else 1.0
        
        # 更新语义计数
        # 将体素语义标签映射到地图网格
        for vy in range(voxel_h):
            for vx in range(voxel_w):
                if topdown_occupancy[vy, vx] > 0:
                    sem_label = int(topdown_semantic[vy, vx])
                    if 0 < sem_label <= self.num_classes:
                        # 计算对应的地图坐标
                        map_y = int(vy * scale_y)
                        map_x = int(vx * scale_x)
                        
                        # 确保坐标在范围内
                        if 0 <= map_y < map_h and 0 <= map_x < map_w:
                            # 累加语义计数（使用占据值作为权重）
                            occupancy_value = float(topdown_occupancy[vy, vx])
                            self.full_sem_counts[scene_idx, sem_label - 1, map_y, map_x] += occupancy_value
    
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
    
    def save_ssc_comparison(self,
                           scene_idx: int,
                           step_global: int,
                           original_semantic_map: np.ndarray,
                           completed_semantic_map: np.ndarray,
                           confidence_map: np.ndarray,
                           explored_map: np.ndarray):
        """
        保存 SSC 补全前后的对比可视化
        
        Args:
            scene_idx: 场景索引
            step_global: 全局步数
            original_semantic_map: (num_classes, H, W) 补全前的语义地图
            completed_semantic_map: (num_classes, H, W) 补全后的语义地图
            confidence_map: (H, W) 置信度地图
            explored_map: (H, W) 已探索区域
        """
        try:
            import matplotlib
            matplotlib.use('Agg')  # 非交互式后端
            import matplotlib.pyplot as plt
        except ImportError:
            print("[SSC] 警告: matplotlib 未安装，无法保存对比图")
            return None
        
        # 计算密度图
        weight_map = self.class_weights.view(1, self.num_classes, 1, 1).detach().cpu().numpy()
        
        original_density = np.sum(original_semantic_map * weight_map[0], axis=0)
        completed_density = np.sum(completed_semantic_map * weight_map[0], axis=0)
        
        # 归一化
        max_val = max(np.max(original_density), np.max(completed_density), 1e-6)
        original_density_norm = original_density / max_val
        completed_density_norm = completed_density / max_val
        
        # 创建对比图
        fig, axes = plt.subplots(2, 2, figsize=(12, 12))
        
        # 补全前的语义地图
        axes[0, 0].imshow(original_density_norm, cmap='inferno', vmin=0, vmax=1)
        axes[0, 0].set_title('Before SSC Completion', fontsize=12)
        axes[0, 0].axis('off')
        
        # 补全后的语义地图
        axes[0, 1].imshow(completed_density_norm, cmap='inferno', vmin=0, vmax=1)
        axes[0, 1].set_title('After SSC Completion', fontsize=12)
        axes[0, 1].axis('off')
        
        # 差异图（补全增加的部分）
        diff_map = completed_density_norm - original_density_norm
        diff_map = np.clip(diff_map, 0, 1)  # 只显示增加的部分
        axes[1, 0].imshow(diff_map, cmap='hot', vmin=0, vmax=1)
        axes[1, 0].set_title('Completion Difference (Added)', fontsize=12)
        axes[1, 0].axis('off')
        
        # 置信度地图
        axes[1, 1].imshow(confidence_map, cmap='viridis', vmin=0, vmax=1)
        axes[1, 1].set_title('Completion Confidence', fontsize=12)
        axes[1, 1].axis('off')
        
        # 添加探索区域轮廓
        try:
            from scipy import ndimage
            for ax in axes.flat:
                explored_contour = explored_map > 0.5
                if np.any(explored_contour):
                    contour = ndimage.binary_erosion(explored_contour) ^ explored_contour
                    y, x = np.where(contour)
                    if len(x) > 0 and len(y) > 0:
                        ax.plot(x, y, 'c-', linewidth=0.5, alpha=0.5)
        except ImportError:
            pass  # scipy 未安装，跳过轮廓绘制
        
        plt.tight_layout()
        
        # 保存图像
        output_path = os.path.join(self.dump_images_dir, f"ssc_comparison_{scene_idx}_{step_global:08d}.png")
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        return output_path


