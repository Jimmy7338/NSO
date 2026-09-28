# semantic/semantic_identifier.py
import torch
import cv2
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg

class SemanticIdentifier:
    def __init__(self, cfg_file="configs/Base-RCNN-FPN.yaml", weights="model_final.pth"):
        self.cfg = get_cfg()
        self.cfg.merge_from_file(cfg_file)
        self.cfg.MODEL.WEIGHTS = weights
        self.cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5  # 过滤低置信度结果
        self.predictor = DefaultPredictor(self.cfg)

    def predict(self, rgb_img):
        """
        输入：RGB图像（H, W, 3，uint8）
        输出：语义标签图（H, W，int），每个像素对应类别ID（0为背景）
        """
        # 转换为Detectron2所需格式（BGR -> RGB，原项目可能用RGB，需确认）
        if rgb_img.shape[-1] == 3:
            rgb_img = cv2.cvtColor(rgb_img, cv2.COLOR_BGR2RGB)  # 若原项目是BGR则转换
        outputs = self.predictor(rgb_img)
        instances = outputs["instances"].to("cpu")
        
        # 初始化语义标签图（背景为0）
        sem_labels = torch.zeros(rgb_img.shape[:2], dtype=torch.int32)
        for i in range(len(instances)):
            mask = instances.pred_masks[i]  # 实例掩码（H, W）
            class_id = instances.pred_classes[i].item()  # 类别ID
            sem_labels[mask] = class_id  # 覆盖掩码区域的类别
        return sem_labels.numpy()