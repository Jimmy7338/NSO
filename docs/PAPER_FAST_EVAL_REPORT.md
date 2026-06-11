# paper_fast 训练与评估报告

> 日期：2026-06-11  
> 权重：`trained_models/paper_fast/model_best.{global,reach}`  
> 评估原始数据：`eval_results/paper_fast/`

---

## 1. 训练概况（已暂停）

| 项目 | 配置 / 结果 |
|------|-------------|
| 脚本 | `scripts/train_fast_paper.sh` |
| 模式 | `--paper_mode`，冻结 SLAM/Local，仅训 Global + RPN |
| 并行 | 4 进程，`--auto_gpu_config` |
| 回合长度 | 1000 步 |
| 语义检测 | 每 5 步，`conf_thresh=0.15` |
| 回环检测 | 关闭（评估时可再开） |
| 运行时长 | **约 6 小时 4 分** |
| 总步数 | **664,520** timesteps |
| 训练速度 | **~30 FPS** |
| 回合奖励（停止时） | 均值 501.3 / 中位 511.4（范围 416–579） |
| Reach Loss（停止时） | 1.42 |
| 语义奖励（停止时） | 0.79 |

训练日志摘录见 `eval_results/paper_fast/training_final_excerpt.txt`（完整日志在服务器 `/mnt/nso_data/nso_runs/models/paper_fast/train.log`）。

---

## 2. 评估设置

| 项目 | 值 |
|------|-----|
| 脚本 | `scripts/eval_paper_fast.sh` |
| 数据集 | Gibson **`val`** split |
| 评估回合 | **20** |
| 每回合步数 | **500** |
| 进程数 | 1（单卡串行评估） |
| 加载权重 | `paper_fast` global + reach；`pretrained_models` slam + local |
| 耗时 | **约 24 分钟** |

复现：

```bash
bash scripts/eval_paper_fast.sh 20
```

---

## 3. 评估结果（核心指标）

### 3.1 汇总表

| 指标 | 均值 | 标准差 | 中位数 | 最小 | 最大 |
|------|------|--------|--------|------|------|
| **单回合最大覆盖率** | **79.55%** | 25.74% | 96.24% | 0.71% | 100% |
| **单回合最大探索面积 (m²)** | **12,123** | 581 | 12,230 | 10,925 | 13,712 |
| 在线语义覆盖率（逐步） | 4.38% | 4.71% | 2.80% | 0.10% | 20.9% |
| 无效目标计数（逐步均值） | 0.84 | 2.74 | 0 | 0 | 11 |

> 共 **80** 条场景-回合记录（20 评估回合 × 多场景轮换）。

### 3.2 跨回合平均覆盖率曲线（按评估步数）

| 步数区间（约） | 平均覆盖率 |
|----------------|------------|
| 起始 | 26.3% |
| 100 步 | 39.0% |
| 200 步 | 48.8% |
| 300 步 | 56.3% |
| 400 步 | 61.4% |
| **500 步（终点）** | **79.5%** |

对应探索面积曲线末点：**12,123 m²**。

### 3.3 与预训练基线对比（定性）

| 对比项 | 说明 |
|--------|------|
| 预训练 `pretrained_models` | ANS 官方权重，未针对论文奖励微调 |
| `paper_fast` | 6h 论文模式快速训练后，val 上平均最大覆盖率约 **80%** |
| 方差较大 | 部分场景覆盖率 <10%，部分接近 100%，与场景难度和初始位姿相关 |

---

## 4. 结论与后续

**已验证：**

- 论文四模块（OV-SDF / STGHP / RPN-UQ / IGCR）在 `--paper_mode` 下可端到端训练与评估；
- 快速训练配置（冻结 SLAM/Local、4 进程）在 6h 内收敛到可用全局策略；
- val 上平均探索覆盖率约 **80%**，探索面积约 **12k m²**。

**待改进：**

- 低覆盖率场景（<20%）需排查：初始位姿、语义检测置信度、目标可达性；
- 轨迹漂移在线指标为 0（Habitat2 评估路径未回传 GT 位姿误差），需补 GT 对齐；
- 训练可继续至更多步数或开启回环/SSC 做 stage4 对比。

---

## 5. 归档文件索引

| 路径 | 说明 |
|------|------|
| `docs/PAPER_FAST_EVAL_REPORT.md` | 本报告 |
| `eval_results/paper_fast/nso_eval_paper_fast.json` | 结构化指标（机器可读） |
| `eval_results/paper_fast/explored_ratio.txt` | 逐步覆盖率原始矩阵 |
| `eval_results/paper_fast/explored_area.txt` | 逐步探索面积原始矩阵 |
| `eval_results/paper_fast/train.log` | main.py 评估日志 |
| `eval_results/paper_fast/eval_console.log` | 评估终端完整输出 |
| `eval_results/paper_fast/training_final_excerpt.txt` | 训练日志首尾摘录 |
| `scripts/eval_paper_fast.sh` | 评估入口脚本 |
| `scripts/summarize_eval_results.py` | JSON 汇总脚本 |

---

## 6. 代码修复记录

评估过程中修复 `main.py`：`exp_ratio` 为 `None` 时 `np.asarray` 产生 `object`  dtype 导致崩溃，已改为 `float(infos[env_idx].get('exp_ratio') or 0.0)`。
