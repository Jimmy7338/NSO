"""
NSO：面向大尺度未知场景的开放词汇语义增强层次化主动覆盖探索

核心创新模块包：
  - nso.clip_semantic_map : OV-SDF（开放词汇语义密度场）
  - nso.topo_graph        : STGHP（语义拓扑图层次规划）
  - nso.reachability_uq   : RPN-UQ（不确定性感知可达性预测）
  - nso.components        : NSO_Components（统一管理器）
"""

# Keep geometric planning usable without importing neural/vision dependencies.
from importlib import import_module

_EXPORTS = {
    'OVSemanticDensityField': 'clip_semantic_map',
    'CLIPEncoder': 'clip_semantic_map',
    'OpenVocabDetector': 'clip_semantic_map',
    'SemanticTopologicalGraph': 'topo_graph',
    'RoomNode': 'topo_graph', 'DoorwayEdge': 'topo_graph',
    'ReachabilityHeadUQ': 'reachability_uq', 'RPNTrainer': 'reachability_uq',
    'apply_uq_mask': 'reachability_uq',
    'apply_mask_point_estimate': 'reachability_uq',
    'calibration_ece': 'reachability_uq', 'NSO_Components': 'components',
}
__all__ = list(_EXPORTS)


def __getattr__(name):
    if name not in _EXPORTS:
        raise AttributeError(name)
    value = getattr(import_module(f'nso.{_EXPORTS[name]}'), name)
    globals()[name] = value
    return value
