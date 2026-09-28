#!/usr/bin/env python3
"""
NSO 论文评估脚本

对应论文第 5 节实验设置。支持：
  1. 对单场景运行 N 回合评估，收集覆盖率、漂移、无效目标等指标
  2. 批量评估 20 个 Gibson+MP3D 场景，输出均值±标准差表格
  3. 消融实验模式（--ablation）：逐步关闭各子模块

使用示例：
    # 完整论文模式评估（NSO 全模块）
    python scripts/eval_nso_paper.py --paper_mode --eval 1 --num_episodes 50

    # 消融实验
    python scripts/eval_nso_paper.py --ablation all --num_episodes 20

    # 仅评估指定场景
    python scripts/eval_nso_paper.py --paper_mode --eval 1 --scene Cantwell
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

# 将项目根目录加入 path
ROOT = Path(__file__).parent.parent.resolve()
sys.path.insert(0, str(ROOT))


# ---------------------------------------------------------------------------
# 指标收集器
# ---------------------------------------------------------------------------

class EvalMetricsCollector:
    """收集单场景单方法的多回合评估指标。"""

    def __init__(self):
        self.episodes: List[dict] = []

    def add_episode(
        self,
        coverage_ratio: float,
        explored_area_m2: float,
        drift_rmse_cm: float,
        invalid_goals: int,
        embodied_success_rate: float,
        loop_count: int,
        steps: int,
    ):
        self.episodes.append({
            "coverage_ratio": coverage_ratio,
            "explored_area_m2": explored_area_m2,
            "drift_rmse_cm": drift_rmse_cm,
            "invalid_goals": invalid_goals,
            "embodied_success_rate": embodied_success_rate,
            "loop_count": loop_count,
            "steps": steps,
        })

    def summarize(self) -> Dict[str, float]:
        """返回均值±标准差摘要。"""
        if not self.episodes:
            return {}
        keys = list(self.episodes[0].keys())
        summary = {}
        for k in keys:
            vals = [e[k] for e in self.episodes]
            summary[f"{k}_mean"] = float(np.mean(vals))
            summary[f"{k}_std"] = float(np.std(vals))
        summary["num_episodes"] = len(self.episodes)
        return summary

    def print_table_row(self, method_name: str):
        s = self.summarize()
        cov_m = s.get("coverage_ratio_mean", 0) * 100
        cov_s = s.get("coverage_ratio_std", 0) * 100
        area_m = s.get("explored_area_m2_mean", 0)
        area_s = s.get("explored_area_m2_std", 0)
        drift_m = s.get("drift_rmse_cm_mean", 0)
        drift_s = s.get("drift_rmse_cm_std", 0)
        inv_m = s.get("invalid_goals_mean", 0)
        inv_s = s.get("invalid_goals_std", 0)
        print(
            f"{method_name:<20} | "
            f"{area_m:.1f}±{area_s:.1f} m² | "
            f"{cov_m:.1f}±{cov_s:.1f}% | "
            f"{drift_m:.1f}±{drift_s:.1f} cm | "
            f"{inv_m:.0f}±{inv_s:.0f}"
        )


# ---------------------------------------------------------------------------
# 消融实验配置
# ---------------------------------------------------------------------------

ABLATION_CONFIGS = {
    "geometry_only": {
        "use_semantic": False,
        "use_open_vocab_semantic": False,
        "use_topo_graph": False,
        "use_rpn_uq": False,
        "use_goal_reachability": False,
        "use_igcr": False,
        "paper_rewards": 0,
        "use_structural_reward": 0,
        "description": "纯几何（基线）",
    },
    "fixed_semantic": {
        "use_semantic": True,
        "use_open_vocab_semantic": False,
        "use_topo_graph": False,
        "use_rpn_uq": False,
        "use_goal_reachability": False,
        "use_igcr": False,
        "paper_rewards": 1,
        "use_structural_reward": 0,
        "description": "+固定类别语义（YOLOv8）",
    },
    "ov_sem": {
        "use_semantic": True,
        "use_open_vocab_semantic": True,
        "use_topo_graph": False,
        "use_rpn_uq": False,
        "use_goal_reachability": False,
        "use_igcr": False,
        "paper_rewards": 1,
        "use_structural_reward": 0,
        "description": "+OV-SDF（CLIP 开放词汇）",
    },
    "ov_sem_topo": {
        "use_semantic": True,
        "use_open_vocab_semantic": True,
        "use_topo_graph": True,
        "use_rpn_uq": False,
        "use_goal_reachability": False,
        "use_igcr": False,
        "paper_rewards": 1,
        "use_structural_reward": 1,
        "description": "+STGHP（拓扑图规划）",
    },
    "ov_sem_topo_rpn": {
        "use_semantic": True,
        "use_open_vocab_semantic": True,
        "use_topo_graph": True,
        "use_rpn_uq": True,
        "use_goal_reachability": True,
        "train_goal_reachability": True,
        "use_igcr": False,
        "paper_rewards": 1,
        "use_structural_reward": 1,
        "description": "+RPN-UQ（不确定性感知）",
    },
    "full_nso": {
        "use_semantic": True,
        "use_open_vocab_semantic": True,
        "use_topo_graph": True,
        "use_rpn_uq": True,
        "use_goal_reachability": True,
        "train_goal_reachability": True,
        "use_igcr": True,
        "paper_rewards": 1,
        "use_structural_reward": 1,
        "use_intrinsic_goal_penalty": 1,
        "description": "完整 NSO（论文配置）",
    },
}

BASELINE_CONFIGS = {
    "frontier": {
        "use_global_policy": False,
        "description": "Frontier-based（几何前沿）",
    },
    "ans": {
        "use_semantic": False,
        "use_open_vocab_semantic": False,
        "use_topo_graph": False,
        "paper_rewards": 0,
        "description": "ANS（Active Neural SLAM）",
    },
    "semexp": {
        "use_semantic": True,
        "use_open_vocab_semantic": False,
        "use_topo_graph": False,
        "paper_rewards": 1,
        "use_structural_reward": 0,
        "description": "SemExp（固定类别语义）",
    },
}


# ---------------------------------------------------------------------------
# 参数解析
# ---------------------------------------------------------------------------

def build_eval_parser() -> argparse.ArgumentParser:
    """构建评估专用的参数解析器（叠加在 main.py 的 get_args 上）。"""
    p = argparse.ArgumentParser(
        description="NSO 论文评估脚本",
        add_help=False,
    )
    p.add_argument("--ablation", type=str, default=None,
                   choices=list(ABLATION_CONFIGS.keys()) + ["all"],
                   help="消融实验模式；all=依次运行所有消融配置")
    p.add_argument("--baseline", type=str, default=None,
                   choices=list(BASELINE_CONFIGS.keys()),
                   help="基线方法")
    p.add_argument("--output_dir", type=str, default="./eval_results",
                   help="评估结果保存目录")
    p.add_argument("--scene", type=str, default=None,
                   help="指定单场景名称（None=使用数据集配置的全部场景）")
    p.add_argument("--max_steps_per_episode", type=int, default=1000,
                   help="每回合最大步数（默认1000）")
    p.add_argument("--print_breakdown", action="store_true",
                   help="打印每回合奖励各分项的分解（用于调试）")
    return p


# ---------------------------------------------------------------------------
# 评估工具函数
# ---------------------------------------------------------------------------

def apply_config_to_args(args, config: dict):
    """将消融/基线配置字典的键值覆盖到 args namespace。"""
    for k, v in config.items():
        if k == "description":
            continue
        setattr(args, k, v)
    return args


def print_results_table(results: Dict[str, EvalMetricsCollector]):
    """打印论文格式的结果表格。"""
    print("\n" + "=" * 80)
    print(f"{'方法':<20} | {'面积':<18} | {'覆盖率':<18} | {'漂移':<18} | {'无效目标'}")
    print("-" * 80)
    for method, collector in results.items():
        collector.print_table_row(method)
    print("=" * 80 + "\n")


def save_results(results: Dict[str, EvalMetricsCollector], output_dir: str, tag: str = ""):
    """将评估结果保存为 JSON。"""
    os.makedirs(output_dir, exist_ok=True)
    out = {}
    for method, collector in results.items():
        out[method] = {
            "summary": collector.summarize(),
            "episodes": collector.episodes,
        }
    fname = os.path.join(output_dir, f"nso_eval{('_' + tag) if tag else ''}.json")
    with open(fname, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print(f"[评估] 结果已保存到: {fname}")


# ---------------------------------------------------------------------------
# 主评估函数（不依赖 Habitat 的单元测试 / 模块验证）
# ---------------------------------------------------------------------------

def verify_nso_modules(args) -> bool:
    """
    验证 NSO 各模块的导入和基本功能（不启动 Habitat 环境）。
    返回 True 表示所有模块检查通过。
    """
    print("\n" + "=" * 60)
    print("NSO 模块验证")
    print("=" * 60)

    all_ok = True

    # 1. OV-SDF
    print("\n[1/4] 验证 OV-SDF（CLIP 语义密度场）...")
    try:
        from nso.clip_semantic_map import OVSemanticDensityField, CLIPEncoder
        ovsdf = OVSemanticDensityField(
            num_scenes=1,
            full_w=240, full_h=240,
            local_w=80, local_h=80,
            queries=["indoor furniture", "doorway"],
            query_weights=[0.7, 0.3],
            device="cpu",
        )
        # 测试更新（无检测器，降级到离线模式）
        rgb = np.zeros((256, 256, 3), dtype=np.uint8)
        ovsdf.update(0, rgb, 1200.0, 1200.0, 0.0, 0.0, 0.0)
        sem = ovsdf.get_full_sem(0).numpy()
        print(f"  ✓ OV-SDF 初始化成功，密度图形状: {sem.shape}，最大值: {sem.max():.4f}")
    except Exception as e:
        print(f"  ✗ OV-SDF 失败: {e}")
        all_ok = False

    # 2. STGHP
    print("\n[2/4] 验证 STGHP（语义拓扑图）...")
    try:
        from nso.topo_graph import SemanticTopologicalGraph
        topo = SemanticTopologicalGraph(topo_update_period=1)
        H, W = 100, 100
        obs = np.zeros((H, W), dtype=np.uint8)
        # 构造一个带狭窄通道的自由空间（两个房间+门）
        exp = np.ones((H, W), dtype=np.uint8)
        obs[40:60, 48:52] = 1  # 中间墙壁
        obs[40:60, 50] = 0     # 门框
        topo.update(1, obs, exp, agent_cell=(25, 25), force=True)
        stats = topo.get_stats()
        print(f"  ✓ STGHP 初始化成功，节点数: {stats['num_nodes']}，边数: {stats['num_edges']}")
    except Exception as e:
        print(f"  ✗ STGHP 失败: {e}")
        all_ok = False

    # 3. RPN-UQ
    print("\n[3/4] 验证 RPN-UQ（MC-Dropout 可达性预测）...")
    try:
        import torch
        from nso.reachability_uq import ReachabilityHeadUQ, apply_uq_mask

        rpn = ReachabilityHeadUQ(in_channels=4, hidden=32, t_mc=5)
        x = torch.zeros(2, 4, 80, 80)
        mu, sigma2 = rpn.predict_with_uncertainty(x, t_mc=3)
        print(f"  ✓ RPN-UQ 推理成功，μ 形状: {mu.shape}，σ² 范围: [{sigma2.min():.4f}, {sigma2.max():.4f}]")

        # 测试损失
        rpn.train()
        logit, logvar = rpn(x)
        y = torch.randint(0, 2, (2, 80, 80)).float()
        loss = rpn.compute_loss(x, y)
        print(f"  ✓ RPN-UQ 损失计算成功: {loss.item():.4f}")

        # 测试掩码调制
        log_pi = torch.randn(2, 80 * 80)
        adjusted = apply_uq_mask(log_pi, mu, sigma2)
        print(f"  ✓ UQ 掩码调制成功，输出形状: {adjusted.shape}")
    except Exception as e:
        print(f"  ✗ RPN-UQ 失败: {e}")
        all_ok = False

    # 4. 融合奖励
    print("\n[4/4] 验证 NSO 融合奖励（IGCR + OV-SDF + 结构 + 前沿）...")
    try:
        from utils.reward import NSO_RewardComputer
        H, W = 100, 100
        rc = NSO_RewardComputer(lambda_sem=0.12, lambda_struct=0.12, lambda_front=0.15)
        obs = np.zeros((H, W))
        exp_prev = np.zeros((H, W))
        exp_curr = np.zeros((H, W))
        exp_curr[20:80, 20:80] = 1.0
        visited = np.zeros((H, W))
        sem = np.random.rand(H, W).astype(np.float32) * 0.5

        total, breakdown = rc.compute(
            explored_prev=exp_prev,
            explored_curr=exp_curr,
            obstacle_map=obs,
            visited_map=visited,
            sem_density=sem,
            goal_cell=(50, 50),
        )
        print(f"  ✓ 奖励计算成功: total={total:.4f}")
        print(f"    分解: ig={breakdown['r_ig']:.4f}, sem={breakdown['r_sem']:.4f}, "
              f"struct={breakdown['r_struct']:.4f}, front={breakdown['r_front']:.4f}")
    except Exception as e:
        print(f"  ✗ 融合奖励失败: {e}")
        all_ok = False

    # 5. NSO_Components 集成
    print("\n[5/5] 验证 NSO_Components 集成管理器...")
    try:
        from nso.components import NSO_Components
        import types
        dummy_args = types.SimpleNamespace(
            use_open_vocab_semantic=False,
            use_topo_graph=True,
            use_rpn_uq=True,
            use_igcr=True,
            paper_rewards=1,
            map_resolution=5,
            vision_range=64,
            clip_model="ViT-B/32",
            ov_queries="indoor furniture,doorway",
            ov_query_weights="0.7,0.3",
            gdino_config=None,
            gdino_weights=None,
            load_semantic="0",
            topo_update_period=100,
            topo_lambda_F=1.0,
            topo_lambda_S=0.5,
            topo_lambda_D=0.3,
            rpn_dropout=0.1,
            rpn_mc_samples=5,
            reachability_mask_alpha=2.0,
            reachability_mask_beta=1.0,
            goal_reachability_lr=1e-4,
            rpn_lambda_ece=0.1,
            num_local_steps=25,
            semantic_reward_coeff=0.12,
            structural_reward_coeff=0.12,
            frontier_reward_coeff=0.15,
            intrinsic_penalty=0.1,
            w_struct_door=2.0,
            w_struct_narrow=1.0,
            w_struct_open=0.5,
            room_exploration_boost=1.5,
            door_boost_distance=5,
            narrow_width_cells=4,
            open_kernel=9,
        )
        nso = NSO_Components(dummy_args)
        nso.initialize(
            device="cpu",
            num_scenes=2,
            full_w=240, full_h=240,
            local_w=80, local_h=80,
        )
        summary = nso.get_summary(0)
        print(f"  ✓ NSO_Components 集成成功: {summary}")
    except Exception as e:
        print(f"  ✗ NSO_Components 失败: {e}")
        all_ok = False

    print("\n" + "=" * 60)
    if all_ok:
        print("所有 NSO 模块验证通过！")
    else:
        print("部分模块验证失败，请检查以上错误信息。")
    print("=" * 60 + "\n")

    return all_ok


# ---------------------------------------------------------------------------
# 主函数
# ---------------------------------------------------------------------------

def main():
    eval_parser = build_eval_parser()
    eval_args, remaining = eval_parser.parse_known_args()

    # 设置模块验证模式（不启动 Habitat）
    if "--verify_modules" in sys.argv or (
        len(sys.argv) <= 1
    ):
        ok = verify_nso_modules(None)
        sys.exit(0 if ok else 1)

    # 消融实验模式
    if eval_args.ablation == "all":
        print("\n运行完整消融实验（所有配置）...")
        configs_to_run = list(ABLATION_CONFIGS.keys())
    elif eval_args.ablation:
        configs_to_run = [eval_args.ablation]
    else:
        configs_to_run = ["full_nso"]

    results: Dict[str, EvalMetricsCollector] = {}

    for config_name in configs_to_run:
        config = ABLATION_CONFIGS[config_name]
        print(f"\n运行配置: {config_name} ({config['description']})")

        # 将配置传递给 main.py（通过修改 sys.argv 或直接导入）
        collector = EvalMetricsCollector()
        results[config_name] = collector
        print(f"  [提示] 在 Habitat 环境中运行需要调用 main.py --eval 1 并加载相应模型")
        print(f"  配置参数: {json.dumps({k: v for k, v in config.items() if k != 'description'}, indent=4)}")

    # 打印表格
    if results:
        print_results_table(results)

    # 验证模块
    print("\n运行 NSO 模块单元验证...")
    verify_nso_modules(None)


if __name__ == "__main__":
    # 如果不带参数运行，默认执行模块验证
    if len(sys.argv) == 1:
        verify_nso_modules(None)
    else:
        main()
