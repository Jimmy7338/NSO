# NSO 使用说明

> 作者：李兆宇  
> 更完整的技术说明见 [完整项目指南](./COMPLETE_PROJECT_GUIDE.md)，论文复现流程见 [训练与实验方案](./TRAINING_AND_EVALUATION_PLAN.md)。

---

## 推荐：论文完整配置

我设计 `--paper_mode` 作为一键开关，启用 OV-SDF、STGHP、RPN-UQ、IGCR 四项创新及回环后端：

```bash
conda activate nso   # Habitat 1.x；Habitat 2.x 用 nso_h2，见 HABITAT2_MIGRATION.md
cd NSO

python main.py --paper_mode --eval 0 \
  --exp_name paper_nso \
  -d /path/to/output/
```

或使用脚本：

```bash
bash scripts/run_nso_paper_train.sh
```

---

## 分阶段训练（推荐流水线）

我按模块依赖关系设计了分阶段训练，便于调试与消融：

| 阶段 | 脚本 | 训练内容 | 产出 |
|------|------|---------|------|
| 0 | `train_stage0_smoke.sh` | 冒烟测试 | — |
| 1 | `train_stage1_slam_local.sh` | SLAM + Local | `trained_models/stage1_slam_local/` |
| 2 | `train_stage2_paper_global.sh` | 全局 PPO + 融合奖励 | `trained_models/stage2_paper_global/` |
| 3 | `train_stage3_rpn.sh` | RPN-UQ 自监督 | 云盘 `stage3_rpn/` |
| 4 | `train_stage4_ssc_loop.sh` | 回环后端联调（旧流水线） | 云盘 `stage4_ssc_loop/` |

阶段 1 可直接复用 `pretrained_models/model_best.*`（ANS 官方权重）跳过。

```bash
# 示例：从 stage1 权重启动 stage2
STAGE2_LOAD_DIR=trained_models/stage1_slam_local \
  bash scripts/train_stage2_paper_global.sh
```

---

## 评估

```bash
python main.py --paper_mode --eval 1 \
  --load_global path/to/model_best.global \
  --load_slam path/to/model_best.slam \
  --load_local path/to/model_best.local \
  --goal_reachability_model_path path/to/model_best.reach \
  --num_episodes 50 \
  --max_episode_length 1000
```

论文指标（覆盖率、漂移、无效目标等）由 `utils/paper_eval.py` 自动记录。

消融实验：

```bash
python scripts/eval_nso_paper.py --ablation all
```

---

## 指定场景

```bash
python main.py --paper_mode --priority_scene Cantwell
```

论文主实验场景：Cantwell（161.2 m²）、Adrian（94.7 m²）、Beechwood（203.5 m²）。

---

## 模块开关（消融用）

| 参数 | 创新点 |
|------|--------|
| `--use_open_vocab_semantic` | OV-SDF |
| `--use_topo_graph` | STGHP |
| `--use_rpn_uq` | RPN-UQ |
| `--use_igcr` | IGCR |
| `--use_loop_detection` | 回环后端（系统组件） |

关闭 `--use_open_vocab_semantic` 并启用 `--use_semantic` 可回退到 YOLOv8 固定类别（论文 NSO-fixed 消融）。

---

## 并行线程与 GPU

```bash
python main.py --auto_gpu_config 0 -n 4 --num_processes_per_gpu 4
```

单卡 RTX 3090（24 GB）建议 `-n 1`，避免 Adrian 等场景的 navmesh 构建内存峰值。

---

## 日志与模型保存

```bash
python main.py --paper_mode -d /mnt/nso_data/nso_runs/ --exp_name my_exp --save_periodic 50000
```

产出路径：`{dump_location}/models/{exp_name}/model_best.{global,local,slam,reach}`

---

## 相关文档

- [训练与实验方案](./TRAINING_AND_EVALUATION_PLAN.md)
- [语义奖励说明](./SEMANTIC_REWARD_EXPLANATION.md)（历史文档，OV-SDF 见论文 §4.2）
- [故障排除](./TROUBLESHOOTING.md)
