"""
语义拓扑图层次规划（STGHP）

对应论文第 4.3 节。

在线从占据地图提取室内拓扑图 G=(V,E)：
  - 节点 V：已探索连通分量（房间/区域）
  - 有向边 E：门框/狭窄通道连接，携带通过置信度与未探索邻域信息

两级层次规划：
  1. 拓扑层（低频，每 N_topo 步）：图上信息增益最大化选择目标房间
  2. 几何层（高频，每 N_global 步）：在目标房间内 FMM 前沿探索

同时提供 RPN-UQ 置信度门控接口，与 model/reachability.py 集成。
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch


# ---------------------------------------------------------------------------
# 数据结构
# ---------------------------------------------------------------------------

@dataclass
class RoomNode:
    """拓扑图节点：代表一个已探索连通区域（房间/走廊）。"""
    node_id: int
    centroid: np.ndarray              # 格点坐标 (row, col)
    cells: List[Tuple[int, int]]      # 所属格点列表
    explored_area: float = 0.0        # 已探索面积（m²）
    frontier_length: float = 0.0      # 未探索边界长度（格点数）
    mean_sem_density: float = 0.0     # 平均语义密度
    last_visited_step: int = -1       # 上次访问步数


@dataclass
class DoorwayEdge:
    """拓扑图有向边：代表两个房间之间的门框或通道。"""
    edge_id: int
    from_node: int                    # 源节点 ID
    to_node: int                      # 目标节点 ID
    doorway_cell: np.ndarray          # 门框中心格点坐标 (row, col)
    passage_confidence: float = 0.5   # 通过置信度（来自 RPN-UQ 均值）
    unexplored_area: float = 0.0      # 目标节点未探索邻域面积
    detection_method: str = "morphology"  # "groundingdino" | "morphology"


# ---------------------------------------------------------------------------
# 在线拓扑图
# ---------------------------------------------------------------------------

class SemanticTopologicalGraph:
    """
    在线语义拓扑图，对应论文公式 (4)–(5)。

    Parameters
    ----------
    map_resolution_cm : 地图分辨率（cm/格点）
    narrow_width_cells: 狭窄通道宽度阈值（格点），小于此值判为门框
    min_room_area_cells: 最小房间面积（格点数），过滤噪声连通分量
    topo_update_period : 拓扑图更新周期（步数）
    lambda_F, lambda_S, lambda_D : 目标节点选择权重（前沿/语义/距离）
    """

    def __init__(
        self,
        map_resolution_cm: int = 5,
        narrow_width_cells: int = 4,
        min_room_area_cells: int = 50,
        topo_update_period: int = 100,
        lambda_F: float = 1.0,
        lambda_S: float = 0.5,
        lambda_D: float = 0.3,
        doorway_conf_thresh: float = 0.3,
        replan_mu_thresh: float = 0.3,
        replan_var_thresh: float = 0.15,
    ):
        self.map_resolution_cm = map_resolution_cm
        self.narrow_width_cells = narrow_width_cells
        self.min_room_area_cells = min_room_area_cells
        self.topo_update_period = topo_update_period
        self.lambda_F = lambda_F
        self.lambda_S = lambda_S
        self.lambda_D = lambda_D
        self.doorway_conf_thresh = doorway_conf_thresh
        self.replan_mu_thresh = replan_mu_thresh
        self.replan_var_thresh = replan_var_thresh

        # 图数据
        self.nodes: Dict[int, RoomNode] = {}
        self.edges: List[DoorwayEdge] = []
        self._next_node_id: int = 0
        self._next_edge_id: int = 0

        # 当前目标节点
        self.current_target_node: Optional[int] = None
        self.current_agent_node: Optional[int] = None

        # 格点→节点的快查表（稀疏字典，按需更新）
        self._cell_to_node: Dict[Tuple[int, int], int] = {}

        # 上次更新步数
        self._last_update_step: int = -1

    def reset(self):
        self.nodes.clear()
        self.edges.clear()
        self._next_node_id = 0
        self._next_edge_id = 0
        self.current_target_node = None
        self.current_agent_node = None
        self._cell_to_node.clear()
        self._last_update_step = -1

    # ------------------------------------------------------------------
    # 拓扑图更新
    # ------------------------------------------------------------------

    def update(
        self,
        step: int,
        obstacle_map: np.ndarray,       # (H, W) 二值：1=障碍
        explored_map: np.ndarray,        # (H, W) 二值：1=已探索
        sem_density: Optional[np.ndarray] = None,  # (H, W) 语义密度
        agent_cell: Optional[Tuple[int, int]] = None,
        force: bool = False,
    ):
        """
        更新拓扑图（如果满足更新周期或 force=True）。

        Parameters
        ----------
        obstacle_map  : 当前全局障碍地图
        explored_map  : 当前全局探索地图
        sem_density   : 可选语义密度图（来自 OV-SDF）
        agent_cell    : 智能体当前格点坐标 (row, col)
        """
        if not force and (step - self._last_update_step) < self.topo_update_period:
            return
        self._last_update_step = step

        # 1. 连通分量分析提取房间节点
        self._extract_room_nodes(obstacle_map, explored_map, sem_density)

        # 2. 门框/通道检测提取边
        self._extract_doorway_edges(obstacle_map, explored_map)

        # 3. 更新当前智能体所在节点
        if agent_cell is not None:
            self.current_agent_node = self._cell_to_node.get(agent_cell)

    def _extract_room_nodes(
        self,
        obstacle_map: np.ndarray,
        explored_map: np.ndarray,
        sem_density: Optional[np.ndarray],
    ):
        """从自由空间连通分量提取房间节点。"""
        try:
            from scipy.ndimage import label
        except ImportError:
            return

        # 自由空间 = 已探索 且 无障碍
        free_space = (explored_map > 0) & (obstacle_map == 0)
        labeled, num_features = label(free_space)

        new_nodes: Dict[int, RoomNode] = {}
        new_cell_to_node: Dict[Tuple[int, int], int] = {}

        res_m = self.map_resolution_cm / 100.0

        for comp_id in range(1, num_features + 1):
            mask = labeled == comp_id
            cell_count = int(mask.sum())
            if cell_count < self.min_room_area_cells:
                continue

            rows, cols = np.where(mask)
            centroid = np.array([rows.mean(), cols.mean()])
            explored_area = cell_count * (res_m ** 2)

            # 前沿边界：已探索自由空间的边缘（8-邻域中有未探索格点）
            frontier_length = self._compute_frontier_length(mask, explored_map)

            # 平均语义密度
            mean_sem = 0.0
            if sem_density is not None:
                mean_sem = float(sem_density[mask].mean())

            # 查找是否已有对应节点（通过质心距离匹配）
            node_id = self._find_or_create_node(centroid, cell_count)

            node = RoomNode(
                node_id=node_id,
                centroid=centroid,
                cells=list(zip(rows.tolist(), cols.tolist())),
                explored_area=explored_area,
                frontier_length=frontier_length,
                mean_sem_density=mean_sem,
            )
            new_nodes[node_id] = node
            for r, c in zip(rows.tolist(), cols.tolist()):
                new_cell_to_node[(r, c)] = node_id

        self.nodes = new_nodes
        self._cell_to_node = new_cell_to_node

    def _find_or_create_node(self, centroid: np.ndarray, area: int) -> int:
        """按质心距离匹配已有节点，否则新建。"""
        best_id = None
        best_dist = float("inf")
        for nid, node in self.nodes.items():
            d = float(np.linalg.norm(node.centroid - centroid))
            if d < best_dist:
                best_dist = d
                best_id = nid

        # 距离阈值：如果最近节点超过 30 格点则视为新区域
        if best_id is None or best_dist > 30:
            nid = self._next_node_id
            self._next_node_id += 1
            return nid
        return best_id

    @staticmethod
    def _compute_frontier_length(
        free_mask: np.ndarray,
        explored_map: np.ndarray,
    ) -> float:
        """计算前沿边界格点数量。"""
        try:
            from scipy.ndimage import binary_dilation
        except ImportError:
            return 0.0
        dilated = binary_dilation(free_mask, iterations=1)
        frontier = dilated & (explored_map == 0)
        return float(frontier.sum())

    def _extract_doorway_edges(
        self,
        obstacle_map: np.ndarray,
        explored_map: np.ndarray,
    ):
        """用形态学方法检测门框/狭窄通道，构建拓扑边。"""
        try:
            from scipy.ndimage import label, binary_erosion
            from skimage.morphology import skeletonize
        except ImportError:
            return

        if len(self.nodes) < 2:
            return

        free_space = (explored_map > 0) & (obstacle_map == 0)
        if not free_space.any():
            return

        # 骨架化自由空间，在骨架上找窄通道
        skeleton = skeletonize(free_space)

        # 在骨架格点处检测局部宽度 < narrow_width_cells 的位置
        new_edges: List[DoorwayEdge] = []
        edge_cells_seen = set()

        skel_rows, skel_cols = np.where(skeleton)
        for r, c in zip(skel_rows.tolist(), skel_cols.tolist()):
            width = self._local_width(obstacle_map, r, c)
            if width >= self.narrow_width_cells:
                continue

            # 找此格点属于哪个节点
            src_node = self._cell_to_node.get((r, c))
            if src_node is None:
                continue

            # 在门框两侧寻找目标节点
            for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1),
                           (-1, -1), (-1, 1), (1, -1), (1, 1)]:
                nr, nc = r + dr, c + dc
                tgt_node = self._cell_to_node.get((nr, nc))
                if tgt_node is None or tgt_node == src_node:
                    continue

                key = (min(src_node, tgt_node), max(src_node, tgt_node))
                if key in edge_cells_seen:
                    continue
                edge_cells_seen.add(key)

                # 未探索邻域面积
                unexplored = float(
                    (~free_space[
                        max(0, r - 10):r + 10,
                        max(0, c - 10):c + 10
                    ]).sum()
                ) * ((self.map_resolution_cm / 100.0) ** 2)

                edge = DoorwayEdge(
                    edge_id=self._next_edge_id,
                    from_node=src_node,
                    to_node=tgt_node,
                    doorway_cell=np.array([r, c]),
                    unexplored_area=unexplored,
                    detection_method="morphology",
                )
                new_edges.append(edge)
                self._next_edge_id += 1

        self.edges = new_edges

    @staticmethod
    def _local_width(obstacle_map: np.ndarray, r: int, c: int, radius: int = 6) -> float:
        """在 (r, c) 处用局部形态估算自由空间宽度（格点数）。"""
        H, W = obstacle_map.shape
        r1, r2 = max(0, r - radius), min(H, r + radius)
        c1, c2 = max(0, c - radius), min(W, c + radius)
        patch = obstacle_map[r1:r2, c1:c2]
        free_count = (patch == 0).sum()
        return float(free_count ** 0.5)

    # ------------------------------------------------------------------
    # 拓扑层目标选择（论文公式 5）
    # ------------------------------------------------------------------

    def select_target_node(
        self,
        current_step: int,
        mu_reach: Optional[np.ndarray] = None,
        sigma2_reach: Optional[np.ndarray] = None,
    ) -> Optional[Tuple[int, np.ndarray]]:
        """
        在拓扑图上选择信息增益最大的目标节点。

        论文公式 (5):
            v* = argmax_v [ λ_F · F_v + λ_S · S_v - λ_D · d_G(v_cur, v) ]

        Returns
        -------
        (node_id, target_cell) 或 None（图为空时）
        """
        if not self.nodes:
            return None

        cur_node = self.current_agent_node
        scores: List[Tuple[float, int]] = []

        for nid, node in self.nodes.items():
            if nid == cur_node:
                continue

            # 图上距离（启发式：无路径时用欧氏距离）
            graph_dist = self._graph_distance(cur_node, nid)

            # 可达性约束（RPN-UQ 门控）
            tgt_cell = (int(node.centroid[0]), int(node.centroid[1]))
            if mu_reach is not None:
                r, c = tgt_cell
                H, W = mu_reach.shape
                if 0 <= r < H and 0 <= c < W:
                    mu = float(mu_reach[r, c])
                    var = float(sigma2_reach[r, c]) if sigma2_reach is not None else 0.0
                    if mu < self.replan_mu_thresh or var > self.replan_var_thresh:
                        continue  # 跳过不可达/高不确定目标节点

            score = (
                self.lambda_F * node.frontier_length
                + self.lambda_S * node.mean_sem_density
                - self.lambda_D * graph_dist
            )
            scores.append((score, nid))

        if not scores:
            return None

        scores.sort(reverse=True)
        best_nid = scores[0][1]
        best_node = self.nodes[best_nid]
        self.current_target_node = best_nid

        # 目标格点：取节点质心
        target_cell = (int(best_node.centroid[0]), int(best_node.centroid[1]))
        return best_nid, np.array(target_cell)

    def _graph_distance(self, src: Optional[int], dst: int) -> float:
        """
        BFS 图上最短路径（以边数计）。
        如果 src 为 None 或不连通，返回欧氏距离的近似值。
        """
        if src is None or src not in self.nodes or dst not in self.nodes:
            # 欧氏距离近似
            if dst in self.nodes and src in self.nodes:
                return float(np.linalg.norm(
                    self.nodes[dst].centroid - self.nodes[src].centroid
                ))
            return 1e6

        if src == dst:
            return 0.0

        # BFS
        from collections import deque
        adj: Dict[int, List[int]] = {nid: [] for nid in self.nodes}
        for e in self.edges:
            adj[e.from_node].append(e.to_node)
            adj[e.to_node].append(e.from_node)  # 无向

        visited = {src}
        queue = deque([(src, 0)])
        while queue:
            node, dist = queue.popleft()
            if node == dst:
                return float(dist)
            for nb in adj.get(node, []):
                if nb not in visited:
                    visited.add(nb)
                    queue.append((nb, dist + 1))
        # 不连通
        return float(np.linalg.norm(
            self.nodes[dst].centroid - self.nodes[src].centroid
        ))

    # ------------------------------------------------------------------
    # 几何层接口：检查是否需要重规划
    # ------------------------------------------------------------------

    def needs_replan(
        self,
        target_cell: Tuple[int, int],
        mu_reach: Optional[np.ndarray],
        sigma2_reach: Optional[np.ndarray],
    ) -> bool:
        """
        检查当前目标是否需要拓扑层重规划。
        条件：μ_reach < thresh 或 σ²_reach > thresh
        """
        if mu_reach is None:
            return False
        r, c = target_cell
        H, W = mu_reach.shape
        if not (0 <= r < H and 0 <= c < W):
            return True
        mu = float(mu_reach[r, c])
        var = float(sigma2_reach[r, c]) if sigma2_reach is not None else 0.0
        return mu < self.replan_mu_thresh or var > self.replan_var_thresh

    def get_doorway_cells(self) -> List[np.ndarray]:
        """返回所有门框格点坐标（用于奖励函数中的结构感知项）。"""
        return [e.doorway_cell for e in self.edges]

    def get_frontier_cells_in_target(
        self,
        target_node_id: int,
        obstacle_map: np.ndarray,
        explored_map: np.ndarray,
        max_frontiers: int = 20,
    ) -> List[np.ndarray]:
        """
        在目标房间节点内枚举前沿格点（供几何层 FMM 使用）。
        """
        if target_node_id not in self.nodes:
            return []

        node = self.nodes[target_node_id]
        cells = set(node.cells)

        frontiers = []
        for r, c in node.cells:
            # 前沿：已探索自由格点，8-邻域内有未探索格点
            for dr in [-1, 0, 1]:
                for dc in [-1, 0, 1]:
                    nr, nc = r + dr, c + dc
                    if (nr, nc) not in cells and 0 <= nr < explored_map.shape[0]:
                        if explored_map[nr, nc] == 0 and obstacle_map[nr, nc] == 0:
                            frontiers.append(np.array([r, c]))
                            break
                else:
                    continue
                break
            if len(frontiers) >= max_frontiers:
                break

        return frontiers

    # ------------------------------------------------------------------
    # 调试与可视化
    # ------------------------------------------------------------------

    def get_stats(self) -> dict:
        return {
            "num_nodes": len(self.nodes),
            "num_edges": len(self.edges),
            "current_agent_node": self.current_agent_node,
            "current_target_node": self.current_target_node,
            "total_frontier_length": sum(
                n.frontier_length for n in self.nodes.values()
            ),
        }

    def draw_on_map(self, base_map: np.ndarray) -> np.ndarray:
        """
        在灰度地图上绘制拓扑图（节点+边），返回 RGB。

        Parameters
        ----------
        base_map : (H, W) uint8 灰度图

        Returns
        -------
        vis : (H, W, 3) uint8 RGB
        """
        if base_map.ndim == 2:
            vis = np.stack([base_map] * 3, axis=-1)
        else:
            vis = base_map.copy()

        # 绘制边（蓝色）
        for e in self.edges:
            if e.from_node in self.nodes and e.to_node in self.nodes:
                n1 = self.nodes[e.from_node].centroid
                n2 = self.nodes[e.to_node].centroid
                r1, c1 = int(n1[0]), int(n1[1])
                r2, c2 = int(n2[0]), int(n2[1])
                try:
                    import cv2
                    cv2.line(vis, (c1, r1), (c2, r2), (0, 0, 200), 1)
                    # 门框中心标记
                    dc = e.doorway_cell
                    cv2.circle(vis, (int(dc[1]), int(dc[0])), 3, (0, 200, 200), -1)
                except ImportError:
                    pass

        # 绘制节点（绿色=普通，红色=目标，黄色=当前）
        for nid, node in self.nodes.items():
            r, c = int(node.centroid[0]), int(node.centroid[1])
            if nid == self.current_target_node:
                color = (0, 0, 255)  # 红=目标
            elif nid == self.current_agent_node:
                color = (0, 255, 255)  # 黄=当前
            else:
                color = (0, 255, 0)  # 绿
            try:
                import cv2
                cv2.circle(vis, (c, r), 5, color, -1)
            except ImportError:
                H, W = vis.shape[:2]
                for dr in range(-3, 4):
                    for dc_off in range(-3, 4):
                        nr, nc = r + dr, c + dc_off
                        if 0 <= nr < H and 0 <= nc < W:
                            vis[nr, nc] = color
        return vis
