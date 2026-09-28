# 自训练权重目录

> **维护：** 李兆宇  
> 本目录存放 NSO 分阶段训练流水线的**最终 checkpoint**（`model_best.*`），供后续阶段加载与论文复现。

## 当前纳入版本库的权重

| 目录 | 阶段 | 内容 | 用途 |
|------|------|------|------|
| `stage1_slam_local/` | 1 | `model_best.slam`, `model_best.local` | SLAM + 局部策略微调 |
| `stage2_paper_global/` | 2 | `model_best.slam`, `model_best.local` | 论文融合奖励 + 全局 PPO |
| `paper_fast/` | 3 | `model_best.global`, `model_best.reach` | 快速论文模式（冻结 SLAM/Local，4 进程） |

## 云盘上的最新权重（未纳入 git）

以下在 `/mnt/nso_data/nso_runs/models/`，包含 RPN-UQ 与回环联调的最新结果，需手动备份：

| 目录 | 说明 |
|------|------|
| `stage3_rpn/` | RPN-UQ 训练后：`global`, `local`, `slam`, **`reach`** |
| `stage4_ssc_loop/` | 回环后端联调：`global`, **`reach`** |
| `paper_fast/` | 已归档至 `trained_models/paper_fast/`（2025-06 最新） |

## 加载示例

```bash
# 阶段 2：从 stage1 启动
STAGE2_LOAD_DIR=trained_models/stage1_slam_local \
  bash scripts/train_stage2_paper_global.sh

# 论文评估：加载 stage3 RPN 权重
python main.py --paper_mode --eval 1 \
  --load_global /mnt/nso_data/nso_runs/models/stage3_rpn/model_best.global \
  --load_slam /mnt/nso_data/nso_runs/models/stage3_rpn/model_best.slam \
  --load_local /mnt/nso_data/nso_runs/models/stage3_rpn/model_best.local \
  --goal_reachability_model_path /mnt/nso_data/nso_runs/models/stage3_rpn/model_best.reach
```

## 与预训练权重的关系

- `pretrained_models/`：ANS 官方权重，作为阶段 1 起点
- `trained_models/`：我在 NSO 流水线上的自训练产出
- 大文件通过 **Git LFS** 托管（见根目录 `.gitattributes`）

中间产物（`periodic_*`、可视化帧、`train.log`）不纳入仓库。
