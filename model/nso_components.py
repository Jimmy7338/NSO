"""
NSO 核心组件集成模块

统一初始化与管理 NSO 三大核心创新模块：
  1. OVSemanticDensityField  —— 开放词汇语义密度场
  2. SemanticTopologicalGraph —— 语义拓扑图层次规划
  3. ReachabilityHeadUQ + RPNTrainer —— 不确定性感知 RPN

使用方式（在 main.py 中）：
    from model.nso_components import NSO_Components
    nso = NSO_Components(args)
    nso.initialize(device, num_scenes, full_w, full_h, local_w, local_h)
    # 每步更新
    nso.update_semantic(scene_idx, rgb_frame, ...)
    nso.update_topo(step, obs_map, exp_map, sem_density, agent_cell)
    target = nso.select_topo_target(step, mu_reach, sigma2_reach)
    doorway_cells = nso.get_doorway_cells()
"""
from __future__ import annotations

import os
from typing import List, Optional, Tuple

import numpy as np
import torch


class NSO_Components:
    """
    NSO 核心创新模块的统一管理器。

    所有模块均支持懒初始化（initialize 调用后才分配显存），
    以便在 main.py 的参数解析阶段即可实例化而不占用 GPU。
    """

    def __init__(self, args):
        self.args = args
        self.use_ov_sem = getattr(args, "use_open_vocab_semantic", False)
        self.use_topo = getattr(args, "use_topo_graph", False)
        self.use_rpn_uq = getattr(args, "use_rpn_uq", False)
        self.use_igcr = getattr(args, "use_igcr", False)

        # 将在 initialize() 中赋值
        self._ovsdf = None
        self._topo: List = []          # 每个场景一个 TopoGraph
        self._rpn_uq = None
        self._rpn_trainer = None
        self._reward_computers: List = []

        self._initialized = False

    # ------------------------------------------------------------------
    # 初始化
    # ------------------------------------------------------------------

    def initialize(
        self,
        device,
        num_scenes: int,
        full_w: int,
        full_h: int,
        local_w: int,
        local_h: int,
        rpn_in_channels: int = 4,
    ):
        """
        延迟初始化所有 NSO 子模块。

        Parameters
        ----------
        device       : torch.device 或 str
        num_scenes   : 并行场景数
        full_w/h     : 全局地图尺寸（格点）
        local_w/h    : 局部地图尺寸（格点）
        rpn_in_channels: RPN 输入通道数（通常 4）
        """
        args = self.args
        dev_str = str(device)

        # 1. OV-SDF
        if self.use_ov_sem:
            from model.clip_semantic_map import OVSemanticDensityField
            queries = [q.strip() for q in getattr(
                args, "ov_queries",
                "indoor furniture and appliances,doorway and passage"
            ).split(",")]
            raw_weights = getattr(args, "ov_query_weights", "0.7,0.3")
            weights = [float(w) for w in raw_weights.split(",")]
            # 补齐权重
            if len(weights) < len(queries):
                weights += [1.0 / len(queries)] * (len(queries) - len(weights))
            weights = weights[:len(queries)]
            total_w = sum(weights)
            weights = [w / total_w for w in weights]

            self._ovsdf = OVSemanticDensityField(
                num_scenes=num_scenes,
                full_w=full_w,
                full_h=full_h,
                local_w=local_w,
                local_h=local_h,
                map_resolution_cm=getattr(args, "map_resolution", 5),
                vision_range=getattr(args, "vision_range", 64),
                queries=queries,
                query_weights=weights,
                clip_model=getattr(args, "clip_model", "ViT-B/32"),
                gdino_config=getattr(args, "gdino_config", None),
                gdino_weights=getattr(args, "gdino_weights", None),
                yolo_model=getattr(args, "load_semantic", "yolov8n.pt")
                           if getattr(args, "load_semantic", "0") != "0"
                           else "yolov8n.pt",
                device=dev_str,
            )
            print(f"[NSO] OV-SDF 已初始化，查询: {queries}")

        # 2. STGHP 拓扑图（每个场景独立维护）
        if self.use_topo:
            from model.topo_graph import SemanticTopologicalGraph
            self._topo = [
                SemanticTopologicalGraph(
                    map_resolution_cm=getattr(args, "map_resolution", 5),
                    narrow_width_cells=getattr(args, "narrow_width_cells", 4),
                    topo_update_period=getattr(args, "topo_update_period", 100),
                    lambda_F=getattr(args, "topo_lambda_F", 1.0),
                    lambda_S=getattr(args, "topo_lambda_S", 0.5),
                    lambda_D=getattr(args, "topo_lambda_D", 0.3),
                )
                for _ in range(num_scenes)
            ]
            print(f"[NSO] STGHP 拓扑图已初始化，{num_scenes} 个场景")

        # 3. RPN-UQ
        if self.use_rpn_uq:
            from model.reachability import ReachabilityHeadUQ, RPNTrainer
            self._rpn_uq = ReachabilityHeadUQ(
                in_channels=rpn_in_channels,
                hidden=64,
                dropout_p=getattr(args, "rpn_dropout", 0.1),
                t_mc=getattr(args, "rpn_mc_samples", 10),
            ).to(device)

            self._rpn_trainer = RPNTrainer(
                rpn=self._rpn_uq,
                lr=getattr(args, "goal_reachability_lr", 1e-4),
                batch_size=64,
                update_interval=getattr(args, "num_local_steps", 25),
                lambda_ece=getattr(args, "rpn_lambda_ece", 0.1),
                device=str(device),
            )
            print(f"[NSO] RPN-UQ 已初始化（MC={self._rpn_uq.t_mc}, dropout={self._rpn_uq.dropout_p}）")

        # 4. 奖励计算器（每个场景独立，但共享参数）
        if getattr(args, "paper_rewards", 0):
            from utils.reward import NSO_RewardComputer
            rc = NSO_RewardComputer.from_args(args)
            self._reward_computers = [rc] * num_scenes  # 共享同一实例
            print("[NSO] 融合奖励计算器已初始化（IGCR + OV-SDF + 结构 + 前沿）")

        self._initialized = True

    # ------------------------------------------------------------------
    # 语义更新接口
    # ------------------------------------------------------------------

    def update_semantic(
        self,
        scene_idx: int,
        rgb_frame: np.ndarray,
        agent_x_cm: float,
        agent_y_cm: float,
        agent_yaw_deg: float,
        map_origin_x_cm: float,
        map_origin_y_cm: float,
        local_map_origin_x: Optional[int] = None,
        local_map_origin_y: Optional[int] = None,
    ):
        """更新 OV-SDF 语义密度场（每步调用）。"""
        if self._ovsdf is None:
            return
        self._ovsdf.update(
            scene_idx=scene_idx,
            rgb_frame=rgb_frame,
            agent_x_cm=agent_x_cm,
            agent_y_cm=agent_y_cm,
            agent_yaw_deg=agent_yaw_deg,
            map_origin_x_cm=map_origin_x_cm,
            map_origin_y_cm=map_origin_y_cm,
            use_local=True,
            local_map_origin_x=local_map_origin_x,
            local_map_origin_y=local_map_origin_y,
        )

    def get_sem_density(
        self, scene_idx: int, normalized: bool = True
    ) -> Optional[np.ndarray]:
        """获取语义密度图 numpy 数组。"""
        if self._ovsdf is None:
            return None
        if normalized:
            t = self._ovsdf.get_normalized_sem(scene_idx)
        else:
            t = self._ovsdf.get_full_sem(scene_idx)
        return t.cpu().numpy()

    def get_local_sem_channel(self, scene_idx: int) -> Optional[torch.Tensor]:
        """获取局部语义密度 Tensor，供全局策略输入用。"""
        if self._ovsdf is None:
            return None
        return self._ovsdf.get_local_sem(scene_idx)

    # ------------------------------------------------------------------
    # 拓扑图接口
    # ------------------------------------------------------------------

    def update_topo(
        self,
        step: int,
        scene_idx: int,
        obstacle_map: np.ndarray,
        explored_map: np.ndarray,
        agent_cell: Optional[Tuple[int, int]] = None,
        sem_density: Optional[np.ndarray] = None,
        force: bool = False,
    ):
        """更新指定场景的拓扑图（满足周期时触发）。"""
        if not self._topo:
            return
        self._topo[scene_idx].update(
            step=step,
            obstacle_map=obstacle_map,
            explored_map=explored_map,
            sem_density=sem_density,
            agent_cell=agent_cell,
            force=force,
        )

    def select_topo_target(
        self,
        scene_idx: int,
        step: int,
        mu_reach: Optional[np.ndarray] = None,
        sigma2_reach: Optional[np.ndarray] = None,
    ) -> Optional[np.ndarray]:
        """
        在拓扑图上选择目标节点，返回目标格点坐标 (row, col) 或 None。
        """
        if not self._topo:
            return None
        result = self._topo[scene_idx].select_target_node(
            step, mu_reach, sigma2_reach
        )
        if result is None:
            return None
        _, target_cell = result
        return target_cell

    def get_doorway_cells(self, scene_idx: int) -> List[np.ndarray]:
        """获取当前场景拓扑图中所有门框格点坐标。"""
        if not self._topo:
            return []
        return self._topo[scene_idx].get_doorway_cells()

    def needs_topo_replan(
        self,
        scene_idx: int,
        target_cell: Tuple[int, int],
        mu_reach: Optional[np.ndarray],
        sigma2_reach: Optional[np.ndarray],
    ) -> bool:
        """检查当前目标是否需要拓扑层重规划。"""
        if not self._topo:
            return False
        return self._topo[scene_idx].needs_replan(target_cell, mu_reach, sigma2_reach)

    def get_topo_stats(self, scene_idx: int) -> dict:
        if not self._topo:
            return {}
        return self._topo[scene_idx].get_stats()

    def draw_topo_on_map(
        self, scene_idx: int, base_map: np.ndarray
    ) -> Optional[np.ndarray]:
        if not self._topo:
            return None
        return self._topo[scene_idx].draw_on_map(base_map)

    # ------------------------------------------------------------------
    # RPN-UQ 接口
    # ------------------------------------------------------------------

    def predict_reachability_uq(
        self,
        map_input: torch.Tensor,
        t_mc: Optional[int] = None,
    ) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
        """
        MC-Dropout 可达性预测。

        Parameters
        ----------
        map_input : (B, C, H, W) 或 (C, H, W) 局部地图

        Returns
        -------
        (mu_reach, sigma2_reach) ：(B,H,W) 或 None
        """
        if self._rpn_uq is None:
            return None, None
        if map_input.dim() == 3:
            map_input = map_input.unsqueeze(0)
        mu, sigma2 = self._rpn_uq.predict_with_uncertainty(map_input, t_mc)
        return mu, sigma2

    def apply_uq_mask_to_logits(
        self,
        log_policy: torch.Tensor,
        mu_reach: torch.Tensor,
        sigma2_reach: torch.Tensor,
    ) -> torch.Tensor:
        """应用 RPN-UQ 掩码调制（论文公式6）。"""
        from model.reachability import apply_uq_mask
        alpha = getattr(self.args, "reachability_mask_alpha", 2.0)
        beta = getattr(self.args, "reachability_mask_beta", 1.0)
        return apply_uq_mask(log_policy, mu_reach, sigma2_reach, alpha, beta)

    def add_rpn_experience(self, map_input: torch.Tensor, label: torch.Tensor):
        """添加一条可达性训练样本。"""
        if self._rpn_trainer is None:
            return
        self._rpn_trainer.add_experience(map_input, label)

    def train_rpn_step(self) -> Optional[float]:
        """执行一步 RPN 自监督训练，返回 loss 或 None。"""
        if self._rpn_trainer is None:
            return None
        return self._rpn_trainer.step()

    @property
    def rpn_uq(self):
        return self._rpn_uq

    # ------------------------------------------------------------------
    # 奖励计算接口
    # ------------------------------------------------------------------

    def compute_reward(
        self,
        scene_idx: int,
        explored_prev: np.ndarray,
        explored_curr: np.ndarray,
        obstacle_map: np.ndarray,
        visited_map: np.ndarray,
        goal_cell: Optional[Tuple[int, int]] = None,
        obstacle_prob: Optional[np.ndarray] = None,
    ) -> Tuple[float, dict]:
        """
        计算 NSO 完整融合奖励。

        Returns
        -------
        (total_reward, breakdown_dict)
        """
        if not self._reward_computers:
            # 降级：返回简单面积奖励
            n_new = float((explored_curr > explored_prev).sum())
            r = n_new * 0.02 * ((getattr(self.args, "map_resolution", 5) / 100.0) ** 2)
            return r, {"r_total": r}

        rc = self._reward_computers[scene_idx]
        sem_density = self.get_sem_density(scene_idx, normalized=True)
        doorway_cells = self.get_doorway_cells(scene_idx)

        return rc.compute(
            explored_prev=explored_prev,
            explored_curr=explored_curr,
            obstacle_map=obstacle_map,
            visited_map=visited_map,
            sem_density=sem_density,
            goal_cell=goal_cell,
            doorway_cells=doorway_cells,
            obstacle_prob=obstacle_prob,
        )

    # ------------------------------------------------------------------
    # 场景重置
    # ------------------------------------------------------------------

    def reset_scene(self, scene_idx: int):
        """重置单个场景的所有 NSO 状态。"""
        if self._ovsdf is not None:
            self._ovsdf.reset_scene(scene_idx)
        if self._topo and scene_idx < len(self._topo):
            self._topo[scene_idx].reset()

    def reset_all(self):
        """重置所有场景。"""
        if self._ovsdf is not None:
            self._ovsdf.reset_all()
        for topo in self._topo:
            topo.reset()

    # ------------------------------------------------------------------
    # 状态摘要（日志用）
    # ------------------------------------------------------------------

    def get_summary(self, scene_idx: int = 0) -> dict:
        summary = {
            "use_ov_sem": self.use_ov_sem,
            "use_topo": self.use_topo,
            "use_rpn_uq": self.use_rpn_uq,
            "use_igcr": self.use_igcr,
        }
        if self._topo and scene_idx < len(self._topo):
            summary["topo"] = self.get_topo_stats(scene_idx)
        if self._rpn_trainer is not None:
            summary["rpn_loss"] = self._rpn_trainer.get_mean_loss()
        return summary
