# 对比实验指南

> **作者：** 李兆宇  
> **论文：** [`Semantic_Enhanced_Active_SLAM_Paper.tex`](../Semantic_Enhanced_Active_SLAM_Paper.tex) §5  
> **评估脚本：** `scripts/eval_nso_paper.py`、`utils/paper_eval.py`

本文档说明如何复现论文中的基线对比与消融实验。

---

## 一、对比方法

| 方法 | 说明 | 配置要点 |
|------|------|---------|
| **Frontier** | 纯几何前沿 | 关闭语义、拓扑、RPN、IGCR |
| **ANS** | Active Neural SLAM 复现 | 仅几何地图 + 全局 PPO |
| **SemExp** | 封闭集语义 ObjectNav 思路迁移 | 固定类别语义图 |
| **PONI** | 势场语义导航基线 | 论文表 2 数值 |
| **OVRL** | 开放词汇预训练表征 | 无拓扑 / RPN |
| **NSO-fixed** | 固定 YOLO 类别（消融） | `--use_open_vocab_semantic` 关闭 |
| **NSO (Ours)** | 完整系统 | `--paper_mode` |

论文报告 **20 场景**（Gibson 10 + MP3D 10），每场景 **50 回合**，1000 步/回合。

---

## 二、快速运行

### 2.1 模块与消融自检（无需 Habitat）

```bash
conda activate nso
cd ~/NSO
python scripts/eval_nso_paper.py
python scripts/eval_nso_paper.py --ablation all --scene Cantwell
```

### 2.2 完整 NSO 评估

```bash
python main.py --paper_mode --eval 1 \
  --load_global <ckpt>/model_best.global \
  --load_slam <ckpt>/model_best.slam \
  --load_local <ckpt>/model_best.local \
  --goal_reachability_model_path <ckpt>/model_best.reach \
  --num_episodes 50 \
  --max_episode_length 1000
```

### 2.3 ANS 基线（关闭 NSO 创新）

```bash
python main.py --eval 1 \
  --load_global pretrained_models/model_best.global \
  --load_slam pretrained_models/model_best.slam \
  --load_local pretrained_models/model_best.local \
  --num_episodes 50
```

确保未传入 `--paper_mode` 及 `--use_open_vocab_semantic` 等 NSO 开关。

---

## 三、消融配置矩阵

`scripts/eval_nso_paper.py` 内 `ABLATION_CONFIGS` 定义六档配置：

| 配置 | OV-SDF | STGHP | RPN-UQ | IGCR |
|------|--------|-------|--------|------|
| 纯几何 | × | × | × | × |
| +固定语义 | fixed | × | × | × |
| +OV-SDF | ✓ | × | × | × |
| +STGHP | ✓ | ✓ | × | × |
| +RPN-UQ | ✓ | ✓ | ✓ | × |
| 完整 NSO | ✓ | ✓ | ✓ | ✓ |

各档需分别训练全局策略（或从 full NSO checkpoint 继训 5k 回合以节省算力）。

---

## 四、记录指标

由 `utils/paper_eval.py` 自动写入 TensorBoard / JSON：

| 指标 | 含义 |
|------|------|
| 覆盖率 | 已探索可通行区域 / 全图可通行 |
| 探索面积 (m²) | 累计新探索栅格面积 |
| 漂移 RMSE (cm) | 相对真值位姿误差 |
| 无效目标频次 | FMM 不可达的全局目标次数 |
| 具身成功率 | 25 步内到达全局目标比例 |
| 回环次数 | NetVLAD 触发并成功 PGO 的次数 |

---

## 五、论文目标结果（20 场景均值）

| 方法 | 覆盖率 (%) | 漂移 (cm) | 无效目标 |
|------|-----------|----------|---------|
| ANS | 74.2±3.1 | 48.7±6.2 | 43±11 |
| PONI | 76.1±3.6 | 47.2±6.0 | 31±8 |
| **NSO** | **92.4±1.8** | **11.9±1.7** | **2±1** |

典型场景 Cantwell（161.2 m²）：NSO 覆盖率 92.4%，无效目标 2 次。

---

## 六、与旧版对比脚本的关系

早期 `scripts/run_comparison.sh` 对比的是「ANS 基础版 vs YOLO 语义增强版」，路径指向 `/home/ubuntu/lzy/...`。  
**论文复现请以本指南与 `--paper_mode` 为准**；旧脚本仅作历史参考。

---

## 七、结果导出

```bash
# 评估结果默认目录
ls eval_results/

# TensorBoard
tensorboard --logdir $NSO_RUN_ROOT/models/paper_nso/
```

将 `eval_results/*.json` 汇总后回填论文表 2–4。
