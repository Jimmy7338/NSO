import torch
import torch.nn.functional as F
from torchvision import transforms
import numpy as np


class SemanticVLADExtractor:
    """
    使用预训练 NetVLAD 模型提取全局描述子，并融合语义统计信息。
    """

    def __init__(self,
                 num_semantic_classes: int,
                 device: torch.device,
                 resize: int = 224,
                 cache_model: bool = True,
                 lazy_load: bool = True):
        self.device = device
        self.num_semantic_classes = num_semantic_classes
        self.resize = resize
        self.cache_model = cache_model
        self.lazy_load = lazy_load
        self._netvlad = None  # 延迟加载
        self._preprocess = None  # 延迟初始化preprocess
        # 可选缓存最近一次结果，加速重复图像调用
        self._last_frame_hash = None
        self._last_feature = None

    @property
    def netvlad(self):
        """延迟加载NetVLAD模型，只在第一次使用时加载"""
        if self._netvlad is None:
            import torch.hub
            print("[Loop] 正在加载 NetVLAD 模型（首次使用，可能需要一些时间）...")
            self._netvlad = torch.hub.load(
                "lyakaap/NetVLAD-pytorch", "netvlad",
                pretrained=True, trust_repo=True
            )
            self._netvlad = self._netvlad.to(self.device)
            self._netvlad.eval()
            # 使用torch.compile加速（如果支持）
            try:
                if hasattr(torch, 'compile'):
                    self._netvlad = torch.compile(self._netvlad, mode='reduce-overhead')
                    print("[Loop] NetVLAD 模型已编译加速")
            except:
                pass
            print("[Loop] NetVLAD 模型加载完成")
        return self._netvlad
    
    @property
    def preprocess(self):
        """延迟初始化预处理"""
        if self._preprocess is None:
            self._preprocess = transforms.Compose([
                transforms.Resize((self.resize, self.resize)),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
            ])
        return self._preprocess

    @torch.no_grad()
    def _encode_vlad(self, rgb_tensor: torch.Tensor) -> torch.Tensor:
        """
        输入 RGB Tensor (3,H,W)，值范围 [0,255] 或 [0,1]
        输出 NetVLAD 描述子 (D,)
        优化：减少不必要的CPU/GPU传输
        """
        # 如果已经在CPU上，直接处理；否则先移到CPU
        if rgb_tensor.is_cuda:
            rgb_tensor = rgb_tensor.detach().cpu()
        else:
            rgb_tensor = rgb_tensor.detach()
        
        # 统一转换为float32并归一化到[0,1]
        if rgb_tensor.dtype != torch.float32:
            rgb_tensor = rgb_tensor.float()
        if rgb_tensor.max() > 1.0:
            rgb_tensor = rgb_tensor / 255.0

        # 预处理（resize + normalize）
        # 优化：使用更小的resize尺寸可以加速（但可能降低精度）
        x = self.preprocess(rgb_tensor)
        x = x.unsqueeze(0).to(self.device, non_blocking=True)  # 使用non_blocking加速
        
        # 获取模型（延迟加载）
        model = self.netvlad
        vlad_feat = model(x)  # (1,D)
        vlad_feat = F.normalize(vlad_feat, p=2, dim=1)
        return vlad_feat.squeeze(0)

    @staticmethod
    def _hash_tensor(tensor: torch.Tensor) -> int:
        return hash(tensor.detach().cpu().numpy().tobytes())

    @staticmethod
    def _semantic_hist(classes: np.ndarray,
                       scores: np.ndarray,
                       num_classes: int) -> np.ndarray:
        hist = np.zeros((num_classes,), dtype=np.float32)
        if classes.size == 0:
            return hist
        if scores is None or scores.size == 0:
            for c in classes:
                if 0 <= c < num_classes:
                    hist[c] += 1.0
        else:
            for c, s in zip(classes, scores):
                if 0 <= c < num_classes:
                    hist[c] += float(s)
        norm = np.linalg.norm(hist)
        if norm > 0:
            hist /= norm
        return hist

    def encode(self,
               rgb_tensor: torch.Tensor,
               detections: dict,
               cache: bool = True,
               lightweight: bool = False) -> np.ndarray:
        """
        返回语义增强描述子 (D + num_semantic_classes,) 或仅语义直方图 (num_semantic_classes,)
        
        Args:
            lightweight: 如果True，仅使用语义直方图，跳过NetVLAD（更快但精度较低）
        """
        classes = detections.get("classes", np.array([], dtype=np.int32))
        scores = detections.get("scores", np.array([], dtype=np.float32))
        if isinstance(classes, torch.Tensor):
            classes = classes.cpu().numpy()
        if isinstance(scores, torch.Tensor):
            scores = scores.cpu().numpy()

        semantic_hist = self._semantic_hist(classes, scores, self.num_semantic_classes)
        
        # 轻量级模式：仅使用语义直方图
        if lightweight:
            return semantic_hist.astype(np.float32)
        
        # 完整模式：NetVLAD + 语义直方图
        frame_hash = None
        if cache and self.cache_model:
            frame_hash = self._hash_tensor(rgb_tensor)
            if frame_hash == self._last_frame_hash and self._last_feature is not None:
                base_feat = self._last_feature
            else:
                base_feat = self._encode_vlad(rgb_tensor)
                self._last_frame_hash = frame_hash
                self._last_feature = base_feat
        else:
            base_feat = self._encode_vlad(rgb_tensor)

        semantic_hist_tensor = torch.from_numpy(semantic_hist).to(base_feat.device)
        combined = torch.cat([base_feat, semantic_hist_tensor], dim=0)
        combined = F.normalize(combined, p=2, dim=0)
        return combined.detach().cpu().numpy()

