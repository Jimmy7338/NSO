"""
开放词汇语义密度场（OV-SDF）

对应论文第 4.2 节。使用 CLIP 嵌入空间中的余弦相似度核替代固定类别
YOLOv8 检测器，构建开放词汇语义密度通道 M_sem。

支持：
  - GroundingDINO 开放词汇检测前端（优先）
  - 降级模式：YOLOv8 + CLIP 特征嵌入（当 GroundingDINO 不可用时）
  - 离线模式：仅 CLIP 图像块嵌入（当检测器均不可用时）
"""
from __future__ import annotations

import math
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# 可选依赖的惰性导入
# ---------------------------------------------------------------------------

def _try_import_clip():
    try:
        import clip  # openai/CLIP
        return clip
    except ImportError:
        return None


def _try_import_groundingdino():
    try:
        from groundingdino.util.inference import load_model as gd_load_model
        from groundingdino.util.inference import predict as gd_predict
        return gd_load_model, gd_predict
    except ImportError:
        return None, None


def _try_import_yolo():
    try:
        from ultralytics import YOLO
        return YOLO
    except ImportError:
        return None


# ---------------------------------------------------------------------------
# CLIP 编码器包装
# ---------------------------------------------------------------------------

class CLIPEncoder:
    """轻量 CLIP 封装，缓存文本嵌入。"""

    def __init__(self, model_name: str = "ViT-B/32", device: str = "cpu"):
        clip = _try_import_clip()
        if clip is None:
            raise ImportError(
                "openai-clip 未安装，请运行: pip install git+https://github.com/openai/CLIP.git"
            )
        self.clip = clip
        self.device = device
        self.model, self.preprocess = clip.load(model_name, device=device)
        self.model.eval()
        self._text_cache: Dict[str, torch.Tensor] = {}

    @torch.no_grad()
    def encode_text(self, query: str) -> torch.Tensor:
        """返回归一化文本嵌入，形状 (D,)。"""
        if query not in self._text_cache:
            tokens = self.clip.tokenize([query]).to(self.device)
            feat = self.model.encode_text(tokens)
            feat = F.normalize(feat.float(), dim=-1).squeeze(0)
            self._text_cache[query] = feat
        return self._text_cache[query]

    @torch.no_grad()
    def encode_image_crop(self, img_np: np.ndarray) -> torch.Tensor:
        """
        对 HxWx3 uint8 numpy 图像进行 CLIP 视觉编码。
        返回归一化嵌入，形状 (D,)。
        """
        from PIL import Image
        pil = Image.fromarray(img_np)
        tensor = self.preprocess(pil).unsqueeze(0).to(self.device)
        feat = self.model.encode_image(tensor)
        feat = F.normalize(feat.float(), dim=-1).squeeze(0)
        return feat

    @torch.no_grad()
    def cosine_sim(self, img_np: np.ndarray, query: str) -> float:
        """计算图像裁剪与查询文本之间的余弦相似度 ∈ [-1,1]。"""
        v = self.encode_image_crop(img_np)
        t = self.encode_text(query)
        return float((v * t).sum().item())


# ---------------------------------------------------------------------------
# 开放词汇检测前端
# ---------------------------------------------------------------------------

class OpenVocabDetector:
    """
    开放词汇检测前端，优先使用 GroundingDINO，降级到 YOLOv8+CLIP。

    detect() 返回列表：每条为 (bbox_xyxy: np.ndarray, conf: float, text_query: str)
    """

    def __init__(
        self,
        device: str = "cpu",
        gdino_config: Optional[str] = None,
        gdino_weights: Optional[str] = None,
        yolo_model: str = "yolov8n.pt",
        box_thresh: float = 0.35,
        text_thresh: float = 0.25,
    ):
        self.device = device
        self.box_thresh = box_thresh
        self.text_thresh = text_thresh
        self._backend = "none"

        # 尝试 GroundingDINO
        gd_load, self._gd_predict = _try_import_groundingdino()
        if gd_load is not None and gdino_config and gdino_weights:
            try:
                self._gdino = gd_load(gdino_config, gdino_weights)
                self._gdino = self._gdino.to(device)
                self._gdino.eval()
                self._backend = "groundingdino"
            except Exception as e:
                print(f"[OVDetector] GroundingDINO 加载失败: {e}，降级到 YOLO")

        # 降级到 YOLOv8
        if self._backend == "none":
            YOLO = _try_import_yolo()
            if YOLO is not None:
                try:
                    self._yolo = YOLO(yolo_model)
                    self._backend = "yolo"
                except Exception as e:
                    print(f"[OVDetector] YOLOv8 加载失败: {e}，使用离线模式")

        print(f"[OVDetector] 后端: {self._backend}")

    @property
    def available(self) -> bool:
        return self._backend != "none"

    def detect(
        self, img_np: np.ndarray, queries: List[str]
    ) -> List[Tuple[np.ndarray, float, str]]:
        """
        检测图像中匹配查询的物体。
        返回：[(bbox_xyxy [4], confidence, matched_query), ...]
        """
        if self._backend == "groundingdino":
            return self._detect_gdino(img_np, queries)
        elif self._backend == "yolo":
            return self._detect_yolo(img_np)
        else:
            return []

    def _detect_gdino(
        self, img_np: np.ndarray, queries: List[str]
    ) -> List[Tuple[np.ndarray, float, str]]:
        """GroundingDINO 检测。"""
        import torch
        from PIL import Image

        caption = " . ".join(queries)
        image_pil = Image.fromarray(img_np)

        with torch.no_grad():
            boxes, logits, phrases = self._gd_predict(
                model=self._gdino,
                image=image_pil,
                caption=caption,
                box_threshold=self.box_thresh,
                text_threshold=self.text_thresh,
            )

        H, W = img_np.shape[:2]
        results = []
        for box, logit, phrase in zip(boxes, logits.tolist(), phrases):
            # box 为 cxcywh 归一化
            cx, cy, bw, bh = box.tolist()
            x1 = (cx - bw / 2) * W
            y1 = (cy - bh / 2) * H
            x2 = (cx + bw / 2) * W
            y2 = (cy + bh / 2) * H
            bbox = np.array([x1, y1, x2, y2], dtype=np.float32)
            # 找最匹配的查询
            matched_q = min(queries, key=lambda q: abs(phrase.find(q.split()[0])))
            results.append((bbox, float(logit), matched_q))
        return results

    def _detect_yolo(
        self, img_np: np.ndarray
    ) -> List[Tuple[np.ndarray, float, str]]:
        """YOLOv8 检测（降级模式，query 设为类名）。"""
        results = self._yolo(img_np, verbose=False)
        out = []
        for r in results:
            if r.boxes is None:
                continue
            for box, conf, cls in zip(
                r.boxes.xyxy.cpu().numpy(),
                r.boxes.conf.cpu().numpy(),
                r.boxes.cls.cpu().numpy(),
            ):
                cls_name = self._yolo.names[int(cls)]
                out.append((box.astype(np.float32), float(conf), cls_name))
        return out


# ---------------------------------------------------------------------------
# 核心：开放词汇语义密度场
# ---------------------------------------------------------------------------

class OVSemanticDensityField:
    """
    开放词汇语义密度场（OV-SDF）。

    论文公式 (2):
        S_{i,j}(q) = Σ_t Σ_{k∈O_{i,j}^t} cos(φ_v(I_t[b_k]), φ_t(q)) · conf_k

    支持多查询加权融合：
        S_fused = Σ_m w_m · S(q_m)

    Parameters
    ----------
    num_scenes      : 并行场景数量
    full_w, full_h  : 全局地图尺寸（格点）
    local_w, local_h: 局部地图尺寸（格点）
    map_resolution_cm: 地图分辨率（cm/格点）
    vision_range    : 视野范围（格点）
    queries         : 查询词列表，如 ["indoor furniture", "doorway"]
    query_weights   : 对应查询权重，默认均匀
    clip_model      : CLIP 模型名称
    device          : PyTorch 设备
    """

    def __init__(
        self,
        num_scenes: int,
        full_w: int,
        full_h: int,
        local_w: int,
        local_h: int,
        map_resolution_cm: int = 5,
        vision_range: int = 64,
        queries: Optional[List[str]] = None,
        query_weights: Optional[List[float]] = None,
        clip_model: str = "ViT-B/32",
        detector_backend: str = "auto",
        gdino_config: Optional[str] = None,
        gdino_weights: Optional[str] = None,
        yolo_model: str = "yolov8n.pt",
        device: str = "cpu",
        dump_dir: str = "/tmp/nso_ovsdf",
    ):
        self.num_scenes = num_scenes
        self.full_w = full_w
        self.full_h = full_h
        self.local_w = local_w
        self.local_h = local_h
        self.map_resolution_cm = map_resolution_cm
        self.vision_range = vision_range
        self.device = device

        # 默认查询配置（论文 §4.2.2）
        if queries is None:
            queries = ["indoor furniture and appliances", "doorway and passage"]
        if query_weights is None:
            n = len(queries)
            query_weights = [1.0 / n] * n
        assert len(queries) == len(query_weights)
        self.queries = queries
        self.query_weights = query_weights

        # CLIP 编码器（惰性初始化，避免在无 GPU 环境下启动时崩溃）
        self._clip: Optional[CLIPEncoder] = None
        self._clip_model_name = clip_model

        # 开放词汇检测前端
        self._detector = OpenVocabDetector(
            device=device,
            gdino_config=gdino_config,
            gdino_weights=gdino_weights,
            yolo_model=yolo_model,
        )

        # 语义密度张量（全局 + 局部）
        # 维度：(num_scenes, H, W) — 已融合的密度场
        self.full_sem = torch.zeros(num_scenes, full_w, full_h, dtype=torch.float32)
        self.local_sem = torch.zeros(num_scenes, local_w, local_h, dtype=torch.float32)

        # 每个查询独立的密度张量（用于消融）
        self.full_sem_per_query = [
            torch.zeros(num_scenes, full_w, full_h, dtype=torch.float32)
            for _ in queries
        ]

    def _get_clip(self) -> CLIPEncoder:
        if self._clip is None:
            self._clip = CLIPEncoder(self._clip_model_name, device=self.device)
        return self._clip

    def to_device(self, device: str):
        self.device = device
        self.full_sem = self.full_sem.to(device)
        self.local_sem = self.local_sem.to(device)
        self.full_sem_per_query = [t.to(device) for t in self.full_sem_per_query]

    def reset_scene(self, scene_idx: int):
        """重置单个场景的密度场。"""
        self.full_sem[scene_idx].zero_()
        self.local_sem[scene_idx].zero_()
        for t in self.full_sem_per_query:
            t[scene_idx].zero_()

    def reset_all(self):
        self.full_sem.zero_()
        self.local_sem.zero_()
        for t in self.full_sem_per_query:
            t.zero_()

    # ------------------------------------------------------------------
    # 主更新接口
    # ------------------------------------------------------------------

    def update(
        self,
        scene_idx: int,
        rgb_frame: np.ndarray,
        agent_x_cm: float,
        agent_y_cm: float,
        agent_yaw_deg: float,
        map_origin_x_cm: float,
        map_origin_y_cm: float,
        use_local: bool = True,
        local_map_origin_x: Optional[int] = None,
        local_map_origin_y: Optional[int] = None,
    ):
        """
        用当前帧更新语义密度场。

        Parameters
        ----------
        rgb_frame       : HxWx3 uint8 RGB 图像
        agent_x_cm/y_cm : Agent 在全局地图中的位置（cm）
        agent_yaw_deg   : Agent 朝向（度，0=正北）
        map_origin_*    : 全局地图原点坐标（cm）
        """
        if not self._detector.available and self._clip is None:
            # 离线模式：仅用整帧 CLIP 嵌入更新智能体前方区域
            self._update_dense_clip(
                scene_idx, rgb_frame,
                agent_x_cm, agent_y_cm, agent_yaw_deg,
                map_origin_x_cm, map_origin_y_cm,
                use_local, local_map_origin_x, local_map_origin_y,
            )
            return

        # 1. 开放词汇检测
        detections = self._detector.detect(rgb_frame, self.queries)

        # 2. 对每个检测框：CLIP 相似度 × conf → 投影到地图
        clip_enc = self._get_clip()
        H_img, W_img = rgb_frame.shape[:2]

        for bbox_xyxy, conf, matched_q in detections:
            x1, y1, x2, y2 = bbox_xyxy.astype(int)
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(W_img - 1, x2), min(H_img - 1, y2)
            if x2 <= x1 or y2 <= y1:
                continue
            crop = rgb_frame[y1:y2, x1:x2]

            # 对每个查询计算余弦相似度
            scores = {}
            for q in self.queries:
                try:
                    sim = clip_enc.cosine_sim(crop, q)
                    sim = max(0.0, sim)  # 只保留正相关
                except Exception:
                    sim = 0.0
                scores[q] = sim

            # 物体中心像素 → 全局地图格点（近似投影：忽略深度，取视野中心射线）
            cx_pix = (x1 + x2) / 2.0
            cy_pix = (y1 + y2) / 2.0

            # 简化投影：将检测物体假设在 vision_range/2 处
            dist_cells = self.vision_range * 0.5
            yaw_rad = math.radians(agent_yaw_deg)
            dx_cm = math.cos(yaw_rad) * dist_cells * self.map_resolution_cm
            dy_cm = math.sin(yaw_rad) * dist_cells * self.map_resolution_cm

            cell_x = int((agent_x_cm + dx_cm - map_origin_x_cm) / self.map_resolution_cm)
            cell_y = int((agent_y_cm + dy_cm - map_origin_y_cm) / self.map_resolution_cm)

            for q_idx, q in enumerate(self.queries):
                sim = scores.get(q, 0.0)
                w = self.query_weights[q_idx]
                contrib = float(sim * conf * w)
                self._splat_to_map(
                    self.full_sem_per_query[q_idx][scene_idx],
                    cell_x, cell_y, contrib,
                )

        # 3. 合并为融合密度
        fused = sum(
            self.query_weights[i] * self.full_sem_per_query[i][scene_idx]
            for i in range(len(self.queries))
        )
        self.full_sem[scene_idx] = fused

        # 4. 裁剪局部地图
        if use_local and local_map_origin_x is not None:
            lox, loy = local_map_origin_x, local_map_origin_y
            lox = max(0, min(lox, self.full_w - self.local_w))
            loy = max(0, min(loy, self.full_h - self.local_h))
            self.local_sem[scene_idx] = self.full_sem[
                scene_idx, lox:lox + self.local_w, loy:loy + self.local_h
            ]

    def _update_dense_clip(
        self,
        scene_idx: int,
        rgb_frame: np.ndarray,
        agent_x_cm: float,
        agent_y_cm: float,
        agent_yaw_deg: float,
        map_origin_x_cm: float,
        map_origin_y_cm: float,
        use_local: bool,
        local_map_origin_x: Optional[int],
        local_map_origin_y: Optional[int],
    ):
        """离线模式：用整帧 CLIP 嵌入均匀更新视野扇形区域。"""
        clip_enc = self._get_clip()
        scores = {}
        for q in self.queries:
            try:
                sim = clip_enc.cosine_sim(rgb_frame, q)
                scores[q] = max(0.0, sim)
            except Exception:
                scores[q] = 0.0

        yaw_rad = math.radians(agent_yaw_deg)
        for r in range(1, self.vision_range + 1):
            for angle_offset in np.linspace(-math.pi / 3, math.pi / 3, 15):
                a = yaw_rad + angle_offset
                dx = math.cos(a) * r * self.map_resolution_cm
                dy = math.sin(a) * r * self.map_resolution_cm
                cx = int((agent_x_cm + dx - map_origin_x_cm) / self.map_resolution_cm)
                cy = int((agent_y_cm + dy - map_origin_y_cm) / self.map_resolution_cm)
                decay = 1.0 / (1.0 + r * 0.05)
                for q_idx, q in enumerate(self.queries):
                    contrib = scores[q] * decay * self.query_weights[q_idx]
                    self._splat_to_map(
                        self.full_sem_per_query[q_idx][scene_idx],
                        cx, cy, contrib * 0.1,
                    )

        fused = sum(
            self.query_weights[i] * self.full_sem_per_query[i][scene_idx]
            for i in range(len(self.queries))
        )
        self.full_sem[scene_idx] = fused

    @staticmethod
    def _splat_to_map(
        sem_map: torch.Tensor,  # (H, W)
        cx: int,
        cy: int,
        value: float,
        radius: int = 2,
    ):
        """将一个语义贡献值以高斯 splat 方式写入地图。"""
        H, W = sem_map.shape
        for dr in range(-radius, radius + 1):
            for dc in range(-radius, radius + 1):
                r, c = cy + dr, cx + dc
                if 0 <= r < H and 0 <= c < W:
                    dist2 = dr * dr + dc * dc
                    weight = math.exp(-dist2 / (radius ** 2 + 1e-6))
                    sem_map[r, c] += value * weight

    # ------------------------------------------------------------------
    # 访问接口
    # ------------------------------------------------------------------

    def get_full_sem(self, scene_idx: Optional[int] = None) -> torch.Tensor:
        """返回全局语义密度图。"""
        if scene_idx is None:
            return self.full_sem
        return self.full_sem[scene_idx]

    def get_local_sem(self, scene_idx: Optional[int] = None) -> torch.Tensor:
        if scene_idx is None:
            return self.local_sem
        return self.local_sem[scene_idx]

    def get_normalized_sem(self, scene_idx: int) -> torch.Tensor:
        """返回 [0,1] 归一化的全局密度图。"""
        s = self.full_sem[scene_idx]
        s_max = s.max()
        if s_max < 1e-6:
            return s
        return s / (s_max + 1e-6)

    def get_query_sem(self, query_idx: int, scene_idx: int) -> torch.Tensor:
        """返回单个查询对应的密度图（用于消融实验）。"""
        return self.full_sem_per_query[query_idx][scene_idx]
