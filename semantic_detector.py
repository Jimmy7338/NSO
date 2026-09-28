import os
from typing import List, Tuple, Dict, Any, Optional

import numpy as np
import torch

try:
    from ultralytics import YOLO
except Exception as e:
    YOLO = None

try:
    from semantic.class_mapping import (
        map_yolo_to_custom, get_num_custom_classes, get_custom_class_name,
        filter_indoor_classes, is_indoor_class
    )
    USE_CLASS_MAPPING = True
    HAS_INDOOR_FILTER = True
except ImportError:
    USE_CLASS_MAPPING = False
    HAS_INDOOR_FILTER = False
    print("警告: 无法导入类别映射模块，将使用原始YOLO类别")


class SemanticDetector:
    """
    轻量封装 YOLOv8n 做语义检测，最小侵入式接入。
    支持类别映射到自定义语义标签集。
    """
    def __init__(self, 
                 model_name: str = "yolov8n.pt", 
                 device: torch.device = torch.device("cpu"),
                 use_custom_mapping: bool = True,
                 indoor_only: bool = True):
        assert YOLO is not None, "ultralytics 未安装，请先安装 ultralytics"
        self.device = device
        print(f"[SemanticDetector] 正在加载模型: {model_name}")
        print(f"[SemanticDetector] 如果模型不存在，将自动从网上下载（可能需要几分钟）...")
        self.model = YOLO(model_name)
        print(f"[SemanticDetector] 模型加载完成，正在移动到设备: {device}")
        # ultralytics 会自行选择设备，这里确保与项目一致
        if "cuda" in str(device) and torch.cuda.is_available():
            self.model.to(int(str(device).split(":")[-1]))
        else:
            self.model.to("cpu")
        print(f"[SemanticDetector] 模型已移动到设备: {device}")
        self.class_names = self.model.names
        self.use_custom_mapping = use_custom_mapping and USE_CLASS_MAPPING
        self.indoor_only = indoor_only and HAS_INDOOR_FILTER
        
        if self.use_custom_mapping:
            self.num_classes = get_num_custom_classes()
            self.custom_class_names = [get_custom_class_name(i) for i in range(self.num_classes)]
            print(f"使用自定义语义标签集，共 {self.num_classes} 个类别")
        else:
            self.num_classes = len(self.class_names)
            self.custom_class_names = None
            if self.indoor_only:
                print(f"使用原始YOLO类别（室内场景过滤），共 {self.num_classes} 个类别")
            else:
                print(f"使用原始YOLO类别，共 {self.num_classes} 个类别")

    @torch.no_grad()
    def detect_batch(self, images_chw_uint8: torch.Tensor, conf: float = 0.2) -> List[Dict[str, Any]]:
        """
        输入: BxCxHxW uint8 tensor，范围[0,255]
        输出: 每个batch元素的检测结果字典:
            {
                "boxes": Nx4 (x1,y1,x2,y2, 像素坐标, 相对原图尺寸),
                "scores": N,
                "classes": N (int) - 如果使用自定义映射，这里是自定义类别ID；否则是YOLO类别ID
            }
        """
        if isinstance(images_chw_uint8, torch.Tensor):
            b, c, h, w = images_chw_uint8.shape
            imgs_list = []
            for i in range(b):
                img = images_chw_uint8[i].detach().cpu().numpy()
                img = np.transpose(img, (1, 2, 0))  # HWC
                imgs_list.append(img)
        else:
            raise ValueError("images_chw_uint8 必须为 torch.Tensor")

        results = self.model.predict(imgs_list, conf=conf, verbose=False)
        outputs: List[Dict[str, Any]] = []
        for r in results:
            if r.boxes is None or len(r.boxes) == 0:
                outputs.append({"boxes": np.zeros((0, 4), dtype=np.float32),
                                "scores": np.zeros((0,), dtype=np.float32),
                                "classes": np.zeros((0,), dtype=np.int32)})
                continue
            boxes_xyxy = r.boxes.xyxy.detach().cpu().numpy().astype(np.float32)
            scores = r.boxes.conf.detach().cpu().numpy().astype(np.float32)
            yolo_classes = r.boxes.cls.detach().cpu().numpy().astype(np.int32)
            
            # 如果启用室内场景过滤且不使用自定义映射，先过滤掉不合理的类别
            if not self.use_custom_mapping and self.indoor_only:
                # 创建mask来过滤不合理的类别
                valid_mask = np.array([is_indoor_class(int(cls_id)) for cls_id in yolo_classes], dtype=bool)
                if np.any(valid_mask):
                    yolo_classes = yolo_classes[valid_mask]
                    scores = scores[valid_mask]
                    boxes_xyxy = boxes_xyxy[valid_mask]
                else:
                    # 所有类别都被过滤掉了
                    classes = np.zeros((0,), dtype=np.int32)
                    boxes_xyxy = np.zeros((0, 4), dtype=np.float32)
                    scores = np.zeros((0,), dtype=np.float32)
                    outputs.append({"boxes": boxes_xyxy, "scores": scores, "classes": classes})
                    continue
            
            # 如果使用自定义映射，将YOLO类别ID映射到自定义类别ID
            if self.use_custom_mapping:
                mapped_classes = map_yolo_to_custom(yolo_classes.tolist())
                # 过滤掉被忽略的类别（映射为-1的）
                valid_mask = np.array([cid >= 0 for cid in mapped_classes])
                if np.any(valid_mask):
                    classes = np.array(mapped_classes)[valid_mask].astype(np.int32)
                    boxes_xyxy = boxes_xyxy[valid_mask]
                    scores = scores[valid_mask]
                else:
                    # 所有类别都被过滤掉了
                    classes = np.zeros((0,), dtype=np.int32)
                    boxes_xyxy = np.zeros((0, 4), dtype=np.float32)
                    scores = np.zeros((0,), dtype=np.float32)
            else:
                classes = yolo_classes
            
            outputs.append({"boxes": boxes_xyxy, "scores": scores, "classes": classes})
        return outputs

    def get_class_names(self) -> List[str]:
        if self.use_custom_mapping and self.custom_class_names is not None:
            return self.custom_class_names
        return self.class_names


