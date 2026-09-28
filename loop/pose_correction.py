"""回环检测后的轻量位姿图校正（论文 3.5.1）。"""
from __future__ import annotations

import numpy as np


def se2_blend(pose_a: np.ndarray, pose_b: np.ndarray, weight: float = 0.35) -> np.ndarray:
    """
    将当前位姿向历史匹配位姿靠拢。
    pose: [x, y, theta_deg]
    """
    w = float(np.clip(weight, 0.0, 1.0))
    out = pose_a.astype(np.float64).copy()
    out[0] = (1.0 - w) * pose_a[0] + w * pose_b[0]
    out[1] = (1.0 - w) * pose_a[1] + w * pose_b[1]
    da = ((pose_b[2] - pose_a[2] + 180.0) % 360.0) - 180.0
    out[2] = pose_a[2] + w * da
    return out.astype(np.float32)


def apply_loop_pose_correction(
    full_pose,
    origins,
    lmb,
    env_idx: int,
    current_pose: np.ndarray,
    matched_pose: np.ndarray,
    weight: float = 0.35,
    device=None,
):
    """修正 full_pose / origins 链上的平移（保持与 main 中张量一致）。"""
    import torch

    corrected = se2_blend(current_pose, matched_pose, weight=weight)
    if device is None:
        device = full_pose.device
    full_pose[env_idx, 0] = corrected[0]
    full_pose[env_idx, 1] = corrected[1]
    full_pose[env_idx, 2] = corrected[2]
    return corrected
