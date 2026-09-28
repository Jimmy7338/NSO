# 对比实验指南

> **作者：** 李兆宇
> **当前入口：** [Goal](research/GOAL_PROMPT.md) · [V46 论文稿](thesis/NSO_Manuscript_V46_20260922.tex) · [证据复核](thesis/V46_EVIDENCE_REVIEW_20260922.md)
> **历史评估脚本：** `scripts/eval_nso_paper.py`、`utils/paper_eval.py`
> **证据说明更新：** 2026-09-23

**本文保留早期 ANS 神经网络路径的拟议比较方案与操作示例，未证明这些方法的适配、训练和完整实验已经执行。** 不能据此复现当前 CPU 论文结果，也不能把配置开关当作原方法复现。原有无实测依据的成绩表已删除。

当前 S/G 受控比较、SWAP-I/VISTA-I 共同 CPU 机制适配和 TARE 原生核心迁移分别见 [README](../README.md#当前可复核的受控结果)。两个已见模板布局上的结果不支持独立泛化或完整原系统领先；外部适配比较也不替代内部语义归因。后续有限实验矩阵与验收规则以 [Goal](research/GOAL_PROMPT.md) 为准。

---

## 一、历史拟议对比方法（执行状态未验证）

| 方法 | 说明 | 配置要点 |
|------|------|---------|
| **Frontier** | 纯几何前沿 | 关闭语义、拓扑、RPN、IGCR |
| **ANS** | 拟采用 Active Neural SLAM | 仅几何地图 + 全局 PPO，需核验实现与权重 |
| **SemExp** | 封闭集语义 ObjectNav 思路迁移 | 固定类别语义图 |
| **PONI** | 势场语义导航候选基线 | 任务适配与实际测量待完成，无可引用成绩 |
| **OVRL** | 开放词汇预训练表征 | 无拓扑 / RPN |
| **NSO-fixed** | 固定 YOLO 类别（消融） | `--use_open_vocab_semantic` 关闭 |
| **NSO (Ours)** | 历史神经模块组合 | `--paper_mode` 仅为配置入口 |

历史计划采用 **20 场景**（Gibson 10 + MP3D 10），每场景 **50 回合**，1000 步/回合；尚无已核实的完整执行结果。

---

## 二、历史操作示例

以下命令依赖完整神经环境、真实权重和场景数据，应先核查其可用性。它们不是当前 CPU 结果的复现命令。

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

## 三、历史消融配置矩阵

`scripts/eval_nso_paper.py` 内 `ABLATION_CONFIGS` 定义六档配置：

| 配置 | OV-SDF | STGHP | RPN-UQ | IGCR |
|------|--------|-------|--------|------|
| 纯几何 | × | × | × | × |
| +固定语义 | fixed | × | × | × |
| +OV-SDF | ✓ | × | × | × |
| +STGHP | ✓ | ✓ | × | × |
| +RPN-UQ | ✓ | ✓ | ✓ | × |
| 完整 NSO | ✓ | ✓ | ✓ | ✓ |

这些开关仅表示拟议消融组合，不能据此宣布模块有独立收益。实际比较需固定训练数据、预算、权重选择和评价口径，并记录各组实际执行情况。

---

## 四、历史神经路径拟记录指标

`utils/paper_eval.py` 提供相关记录接口；运行前需核查原始观测、分母及有效样本是否完整，不能仅凭字段存在认定指标有效：

| 指标 | 含义 |
|------|------|
| 覆盖率 | 已探索可通行区域 / 全图可通行 |
| 探索面积 (m²) | 累计新探索栅格面积 |
| 漂移 RMSE (cm) | 相对真值位姿误差 |
| 无效目标频次 | FMM 不可达的全局目标次数 |
| 具身成功率 | 25 步内到达全局目标比例 |
| 回环次数 | NetVLAD 触发并成功 PGO 的次数 |

---

## 五、结果状态

此前列出的 ANS、PONI、NSO 覆盖率、漂移与无效目标均值及“±”数值，以及 Cantwell 单场景成绩，没有可核实的实测依据，均已删除。不得作为论文结果、标准差或预期成功阈值使用。

当前已复核的 V36 比较中，G/S 的平均 `J5 = C_map × F1@5cm` 分别为 **0.702576 / 0.745881**，相对差 **6.1638%**。该结果来自两个已见、相近模板布局及两个隐藏构型，四个配对条件两胜两平；覆盖相同，差异来自表面 F1。它不代表上表神经方法在 Gibson/MP3D 上的成绩。详细边界见 [V46 证据复核](thesis/V46_EVIDENCE_REVIEW_20260922.md)及 [严格审查](thesis/V46_STRICT_REVIEW_20260923.md)。

---

## 六、与旧版对比脚本的关系

早期 `scripts/run_comparison.sh` 对比的是「ANS 基础版 vs YOLO 语义增强版」，路径指向 `/home/ubuntu/lzy/...`。
该脚本与本指南的神经路径操作均属历史参考；当前研究入口以 [Goal](research/GOAL_PROMPT.md) 和 [README](../README.md) 为准。

---

## 七、历史结果目录查看示例

```bash
# 评估结果默认目录
ls eval_results/

# TensorBoard
tensorboard --logdir $NSO_RUN_ROOT/models/paper_nso/
```

目录或 JSON 文件存在不表示完成了有效比较。论文仅采用任务、方法、原始轨迹和评价口径均可追溯的实际结果；不同版本指标、机制适配和原生系统迁移分别报告。
