"""
NSO：面向大尺度未知场景的开放词汇语义增强层次化主动覆盖探索

核心创新模块包：
  - nso.clip_semantic_map : OV-SDF（开放词汇语义密度场）
  - nso.topo_graph        : STGHP（语义拓扑图层次规划）
  - nso.reachability_uq   : RPN-UQ（不确定性感知可达性预测）
  - nso.components        : NSO_Components（统一管理器）
"""

from nso.clip_semantic_map import OVSemanticDensityField, CLIPEncoder, OpenVocabDetector
from nso.topo_graph import SemanticTopologicalGraph, RoomNode, DoorwayEdge
from nso.reachability_uq import (
    ReachabilityHeadUQ,
    RPNTrainer,
    apply_uq_mask,
    apply_mask_point_estimate,
    calibration_ece,
)
from nso.components import NSO_Components

__all__ = [
    "OVSemanticDensityField",
    "CLIPEncoder",
    "OpenVocabDetector",
    "SemanticTopologicalGraph",
    "RoomNode",
    "DoorwayEdge",
    "ReachabilityHeadUQ",
    "RPNTrainer",
    "apply_uq_mask",
    "apply_mask_point_estimate",
    "calibration_ece",
    "NSO_Components",
]
