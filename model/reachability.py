"""
可达性预测网络（向后兼容入口）

旧接口（ReachabilityHead 点估计）保留在此文件。
新 RPN-UQ 实现（MC-Dropout 不确定性感知）在 nso/reachability_uq.py。

main.py 和 env/habitat/reachability_utils.py 中引用的旧接口不受影响。
"""
from __future__ import annotations

import os

import torch
import torch.nn as nn


def infer_rpn_in_channels(checkpoint_path: str) -> int:
    """从 checkpoint 第一层卷积权重推断 RPN 输入通道数（2 或 4）。"""
    if not checkpoint_path or not os.path.isfile(checkpoint_path):
        raise FileNotFoundError(checkpoint_path)
    state = torch.load(checkpoint_path, map_location="cpu")
    # 兼容新旧两种 checkpoint 格式
    weight = (
        state.get("encoder.0.weight")
        or state.get("net.0.weight")
    )
    if weight is None:
        raise KeyError(f"无效的 RPN checkpoint: {checkpoint_path}")
    return int(weight.shape[1])


def default_rpn_in_channels(use_semantic: bool) -> int:
    return 4 if use_semantic else 2


class ReachabilityHead(nn.Module):
    """
    原始点估计 RPN（向后兼容）。
    新代码请使用 nso.reachability_uq.ReachabilityHeadUQ。
    """

    def __init__(self, in_channels: int = 4, hidden: int = 32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, hidden, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, hidden, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, 1, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """返回 logits，形状 (B, H, W)。"""
        return self.net(x).squeeze(1)

    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self.forward(x))
