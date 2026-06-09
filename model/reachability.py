"""可达性预测网络 (RPN)，对应论文式 (2)(3)。"""
from __future__ import annotations

import torch
import torch.nn as nn


class ReachabilityHead(nn.Module):
    """
    从局部地图张量预测 M_reach ∈ [0,1]^{H×W}。
    输入通道默认 4：障碍、探索、当前位置、语义密度（无则零填充）。
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
