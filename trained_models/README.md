# 自训练最终权重

各阶段流水线收敛后的 `model_best.*`，供后续阶段加载或复现实验。

| 目录 | 说明 |
|------|------|
| `stage1_slam_local/` | 阶段 1：SLAM + Local 微调 |
| `stage2_paper_global/` | 阶段 2：论文奖励 + Global PPO + 语义 |

运行时加载示例：

```bash
STAGE2_LOAD_DIR=trained_models/stage1_slam_local bash scripts/train_stage2_paper_global.sh
```

中间过程（`periodic_*`、可视化图片、train.log）不纳入仓库，见根目录 `.gitignore`。
