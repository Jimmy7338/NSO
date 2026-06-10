#!/usr/bin/env python3
"""验证论文对齐模块（不启动 Habitat 仿真）。"""
from __future__ import annotations

import os
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)
os.environ.setdefault("MPLBACKEND", "Agg")

import numpy as np
import torch


def test_reachability_head():
    from model import ReachabilityHead

    m = ReachabilityHead(in_channels=2)
    x = torch.randn(2, 2, 48, 48)
    p = m.predict_proba(x)
    assert p.shape == (2, 48, 48), p.shape
    assert p.min() >= 0 and p.max() <= 1


def test_fmm_utils():
    import importlib.util
    path = os.path.join(ROOT, "env/habitat/reachability_utils.py")
    spec = importlib.util.spec_from_file_location("reachability_utils", path)
    ru = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(ru)
    fmm_reachability_map = ru.fmm_reachability_map
    goal_reachable_label = ru.goal_reachable_label
    mask_global_goals = ru.mask_global_goals

    trav = np.ones((20, 20), dtype=np.float32)
    trav[5:15, 5:15] = 1.0
    trav[0, :] = 0
    trav[-1, :] = 0
    start = (10, 10)
    reach = fmm_reachability_map(trav, start)
    assert reach.shape == (20, 20)
    assert reach[start] > 0.5
    label = goal_reachable_label(trav, start, (12, 12))
    assert label in (0.0, 1.0)
    goals = mask_global_goals(
        [[10, 10]], [reach], 20, 20, alpha=2.0, num_candidates=8,
        free_maps=[trav > 0.5])
    assert len(goals) == 1


def test_paper_eval():
    from utils.paper_eval import PaperMetricsTracker, aggregate_episode_metrics

    tracker = PaperMetricsTracker()
    tracker.update_step(pose_err=[0.01, 0.02, 0.0], exp_ratio=0.5)
    snap = tracker.snapshot()
    assert snap.coverage_ratio == 0.5
    agg = aggregate_episode_metrics([snap, snap])
    assert "coverage_ratio" in agg


def test_embodied_reach():
    import importlib.util
    path = os.path.join(ROOT, "env/habitat/reachability_utils.py")
    spec = importlib.util.spec_from_file_location("reachability_utils", path)
    ru = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(ru)
    assert ru.embodied_goal_reached((10, 10), (12, 12), 10) == 1.0
    assert ru.embodied_goal_reached((10, 10), (25, 25), 10) == 0.0


def test_global_reward_flags():
    text = open(os.path.join(ROOT, "arguments.py"), encoding="utf-8").read()
    for name in [
        "paper_rewards", "paper_mode", "use_structural_reward",
        "use_intrinsic_goal_penalty", "reachability_mask_alpha",
        "loop_pose_correction",
    ]:
        assert name in text, name


def test_exploration_env_methods():
    path = os.path.join(ROOT, "env/habitat/exploration_env.py")
    text = open(path, encoding="utf-8").read()
    assert "def get_reachability_supervision" in text
    assert "paper_rewards" in text
    assert "intrinsic_penalty" in text


def main():
    tests = [
        test_reachability_head,
        test_fmm_utils,
        test_paper_eval,
        test_embodied_reach,
        test_global_reward_flags,
        test_exploration_env_methods,
    ]
    for t in tests:
        name = t.__name__
        print(f"[RUN] {name}")
        t()
        print(f"[OK]  {name}")
    print("\n全部检查通过。")


if __name__ == "__main__":
    main()
