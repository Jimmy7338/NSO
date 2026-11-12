"""
不确定性感知可达性预测网络（RPN-UQ）

对应论文第 4.4 节。

在原 ReachabilityHead 基础上引入 MC-Dropout 不确定性估计：
  - 训练时：Dropout 正常开启
  - 推理时：执行 T_MC 次前向传播，计算均值 μ_reach 与方差 σ²_reach
  - 掩码调制（论文公式 6）：
      P_final(g) ∝ π_act · μ_reach^α · exp(−β σ²_reach)

损失函数（论文公式 9）：
  L_reach = BCE(μ, y) + λ_ece · ECE(μ, y)

同时保留原 ReachabilityHead 接口以向后兼容。
"""
from __future__ import annotations

import os
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# 辅助函数
# ---------------------------------------------------------------------------

def infer_rpn_in_channels(checkpoint_path: str) -> int:
    """从 checkpoint 第一层卷积权重推断 RPN 输入通道数（2 或 4）。"""
    if not checkpoint_path or not os.path.isfile(checkpoint_path):
        raise FileNotFoundError(checkpoint_path)
    state = torch.load(checkpoint_path, map_location="cpu")
    # 兼容 RPN-UQ 新旧两种 checkpoint 格式
    weight = state.get("encoder.0.weight") or state.get("net.0.weight")
    if weight is None:
        raise KeyError(f"无效的 RPN checkpoint: {checkpoint_path}")
    return int(weight.shape[1])


def default_rpn_in_channels(use_semantic: bool) -> int:
    return 4 if use_semantic else 2


def calibration_ece(
    probs: torch.Tensor,
    labels: torch.Tensor,
    n_bins: int = 10,
) -> torch.Tensor:
    """
    期望校准误差（ECE）近似。

    Parameters
    ----------
    probs  : (N,) 预测概率
    labels : (N,) 二值标签
    """
    bins = torch.linspace(0.0, 1.0, n_bins + 1, device=probs.device)
    ece = probs.new_zeros(1)
    N = probs.numel()
    if N == 0:
        return ece

    for i in range(n_bins):
        lo, hi = float(bins[i]), float(bins[i + 1])
        mask = (probs >= lo) & (probs < hi)
        if not mask.any():
            continue
        acc = labels[mask].float().mean()
        conf = probs[mask].mean()
        ece += mask.float().sum() / N * (acc - conf).abs()
    return ece


# ---------------------------------------------------------------------------
# 原始 ReachabilityHead（向后兼容）
# ---------------------------------------------------------------------------

class ReachabilityHead(nn.Module):
    """
    从局部地图张量预测 M_reach ∈ [0,1]^{H×W}。
    输入通道默认 4：障碍、探索、当前位置、语义密度（无则零填充）。
    （保留此类以向后兼容现有 checkpoint 和调用代码）
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


# ---------------------------------------------------------------------------
# 新增：MC-Dropout 不确定性感知 RPN
# ---------------------------------------------------------------------------

class ReachabilityHeadUQ(nn.Module):
    """
    带 MC-Dropout 的不确定性感知可达性预测头（RPN-UQ）。

    论文第 4.4 节公式 (7)(8)。

    Architecture
    ------------
    输入 (B, C_in, H, W) →
      Encoder（3 层卷积+Dropout）→
      均值头：logit μ (B, H, W)
      方差头：log σ² (B, H, W)

    推理时（MC 模式）：
      - 执行 T_MC 次 forward（Dropout 保持 train 模式）
      - 输出样本均值 μ 与样本方差 σ²
    """

    def __init__(
        self,
        in_channels: int = 4,
        hidden: int = 64,
        dropout_p: float = 0.1,
        t_mc: int = 10,
    ):
        super().__init__()
        self.t_mc = t_mc
        self.dropout_p = dropout_p

        # 共享编码器
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels, hidden, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Dropout2d(dropout_p),
            nn.Conv2d(hidden, hidden, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Dropout2d(dropout_p),
            nn.Conv2d(hidden, hidden // 2, 3, padding=1),
            nn.ReLU(inplace=True),
        )

        # 均值头
        self.mean_head = nn.Conv2d(hidden // 2, 1, 1)

        # 对数方差头（预测 log σ²，确保方差为正）
        self.logvar_head = nn.Conv2d(hidden // 2, 1, 1)

    def forward(
        self, x: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        单次前向传播，返回 (logit_mean, log_var)。
        形状均为 (B, H, W)。
        """
        feat = self.encoder(x)
        logit = self.mean_head(feat).squeeze(1)
        logvar = self.logvar_head(feat).squeeze(1)
        return logit, logvar

    # ------------------------------------------------------------------
    # 推理接口
    # ------------------------------------------------------------------

    @torch.no_grad()
    def predict_with_uncertainty(
        self,
        x: torch.Tensor,
        t_mc: Optional[int] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        MC-Dropout 推理，返回 (μ_reach, σ²_reach)。

        Parameters
        ----------
        x    : (B, C, H, W) 输入地图
        t_mc : MC 采样次数，默认使用 self.t_mc

        Returns
        -------
        mu    : (B, H, W) 可达性均值概率
        sigma2: (B, H, W) 可达性方差
        """
        T = t_mc if t_mc is not None else self.t_mc
        # 启用 Dropout（保持 train 模式的 Dropout 行为）
        self.train()

        samples = []
        for _ in range(T):
            logit, _ = self.forward(x)
            prob = torch.sigmoid(logit)
            samples.append(prob)

        stacked = torch.stack(samples, dim=0)  # (T, B, H, W)
        mu = stacked.mean(dim=0)               # (B, H, W)
        sigma2 = stacked.var(dim=0)            # (B, H, W)

        self.eval()
        return mu, sigma2

    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        """向后兼容接口，返回点估计概率（单次推理）。"""
        logit, _ = self.forward(x)
        return torch.sigmoid(logit)

    # ------------------------------------------------------------------
    # 损失函数（论文公式 9）
    # ------------------------------------------------------------------

    def compute_loss(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        lambda_ece: float = 0.1,
    ) -> torch.Tensor:
        """
        ECE-aware BCE 损失。

        Parameters
        ----------
        x    : (B, C, H, W) 输入地图
        y    : (B, H, W) 二值可达性标签 [0,1]
        lambda_ece : ECE 正则化系数

        Returns
        -------
        loss : 标量
        """
        logit, logvar = self.forward(x)
        mu_prob = torch.sigmoid(logit)

        # BCE
        bce = F.binary_cross_entropy_with_logits(logit, y.float(), reduction="mean")

        # ECE 正则化（计算批次内的 ECE）
        probs_flat = mu_prob.detach().reshape(-1)
        labels_flat = y.float().reshape(-1)
        ece = calibration_ece(probs_flat, labels_flat)

        # 可选：对数方差作为 aleatoric 不确定性的辅助损失
        # (鼓励网络在高错误区域输出高方差)
        sigma2 = torch.exp(logvar.clamp(-10, 4))
        aleatoric = (
            0.5 * ((y.float() - mu_prob).detach() ** 2) / (sigma2 + 1e-6)
            + 0.5 * logvar
        ).mean()

        loss = bce + lambda_ece * ece + 0.01 * aleatoric
        return loss


# ---------------------------------------------------------------------------
# 掩码调制函数（论文公式 6）
# ---------------------------------------------------------------------------

def apply_uq_mask(
    log_policy: torch.Tensor,
    mu_reach: torch.Tensor,
    sigma2_reach: torch.Tensor,
    alpha: float = 2.0,
    beta: float = 1.0,
    epsilon: float = 1e-6,
) -> torch.Tensor:
    """
    不确定性感知目标采样掩码调制。

    论文公式 (6):
        P_final(g) = Softmax(
            log π_act(g) + α · log(μ_reach(g) + ε) − β · σ²_reach(g)
        )

    Parameters
    ----------
    log_policy   : (B, H*W) 或 (B, H, W) log 策略分布
    mu_reach     : (B, H, W) 或 (B, H*W) 可达性均值
    sigma2_reach : (B, H, W) 或 (B, H*W) 可达性方差
    alpha        : 均值调制强度（论文 α=2.0）
    beta         : 方差惩罚强度（论文 β=1.0）
    epsilon      : 平滑项

    Returns
    -------
    adjusted_log_prob : 与 log_policy 同形状
    """
    orig_shape = log_policy.shape
    log_policy_flat = log_policy.reshape(orig_shape[0], -1)
    mu_flat = mu_reach.reshape(orig_shape[0], -1)
    var_flat = sigma2_reach.reshape(orig_shape[0], -1)

    adjusted = (
        log_policy_flat
        + alpha * torch.log(mu_flat + epsilon)
        - beta * var_flat
    )
    return adjusted.reshape(orig_shape)


def apply_mask_point_estimate(
    log_policy: torch.Tensor,
    m_reach: torch.Tensor,
    alpha: float = 2.0,
    epsilon: float = 1e-6,
) -> torch.Tensor:
    """
    原始点估计掩码（向后兼容，对应旧版 RPN 接口）。

    论文原公式 (5):
        P_final(g) = Softmax(log π_act(g) + α · log(M_reach(g) + ε))
    """
    orig_shape = log_policy.shape
    log_policy_flat = log_policy.reshape(orig_shape[0], -1)
    m_flat = m_reach.reshape(orig_shape[0], -1)
    adjusted = log_policy_flat + alpha * torch.log(m_flat + epsilon)
    return adjusted.reshape(orig_shape)


# ---------------------------------------------------------------------------
# RPN 训练器：在线自监督更新
# ---------------------------------------------------------------------------

class RPNTrainer:
    """
    RPN-UQ 在线自监督训练器。

    维护一个固定大小的经验 Buffer，每 update_interval 步执行一次梯度更新。
    标签由 FMM 几何可达场与具身回溯结果联合确定（论文公式 7）。
    """

    def __init__(
        self,
        rpn: ReachabilityHeadUQ,
        lr: float = 1e-4,
        buffer_size: int = 2000,
        batch_size: int = 64,
        update_interval: int = 25,
        lambda_ece: float = 0.1,
        device: str = "cpu",
    ):
        self.rpn = rpn
        self.batch_size = batch_size
        self.update_interval = update_interval
        self.lambda_ece = lambda_ece
        self.device = device

        self.optimizer = torch.optim.Adam(rpn.parameters(), lr=lr)

        # 经验 Buffer
        self._inputs: list = []
        self._labels: list = []
        self._buffer_size = buffer_size
        self._step = 0
        self._loss_history: list = []

    def add_experience(
        self,
        map_input: torch.Tensor,  # (C, H, W) 单场景局部地图
        label: torch.Tensor,       # (H, W) 二值可达性标签
    ):
        """添加一条训练样本到 Buffer。"""
        self._inputs.append(map_input.detach().cpu())
        self._labels.append(label.detach().cpu())
        if len(self._inputs) > self._buffer_size:
            self._inputs.pop(0)
            self._labels.pop(0)

    def step(self) -> Optional[float]:
        """
        执行一步训练（如果满足 update_interval 条件）。

        Returns
        -------
        loss_val : float 或 None（未触发训练时）
        """
        self._step += 1
        if self._step % self.update_interval != 0:
            return None
        if len(self._inputs) < self.batch_size:
            return None

        # 随机采样 mini-batch
        import random
        indices = random.sample(range(len(self._inputs)), self.batch_size)
        x_batch = torch.stack([self._inputs[i] for i in indices]).to(self.device)
        y_batch = torch.stack([self._labels[i] for i in indices]).to(self.device)

        self.rpn.train()
        self.optimizer.zero_grad()
        loss = self.rpn.compute_loss(x_batch, y_batch, self.lambda_ece)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.rpn.parameters(), max_norm=1.0)
        self.optimizer.step()

        loss_val = float(loss.item())
        self._loss_history.append(loss_val)
        return loss_val

    def get_mean_loss(self, window: int = 100) -> float:
        if not self._loss_history:
            return 0.0
        return float(sum(self._loss_history[-window:]) / len(self._loss_history[-window:]))
