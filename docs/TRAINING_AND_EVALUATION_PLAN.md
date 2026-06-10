# NSO 训练与实验检验方案

> **作者：** 李兆宇  
> **配套论文：** [`Semantic_Enhanced_Active_SLAM_Paper.tex`](../Semantic_Enhanced_Active_SLAM_Paper.tex)  
> **对应代码：** `nso/`（OV-SDF / STGHP / RPN-UQ / IGCR）+ `main.py --paper_mode`  
> **硬件基准：** 单卡 RTX 3090（24 GB），Habitat-Lab + Gibson/MP3D

本文档记录我为复现论文实验而设计的分阶段训练与评估流程。

---

## 一、总体路线图

```
阶段0  环境与模块自检（0.5 天）
  ↓
阶段1  SLAM + 局部策略（可复用 ANS 预训练，跳过）
  ↓
阶段2  全局策略 PPO + 融合奖励（3~5 天）
  ↓
阶段3  RPN-UQ 在线自监督（与阶段2 并行，1~2 天）
  ↓
阶段4  主实验：20 场景 × 50 回合（2~3 天）
  ↓
阶段5  消融 + 基线对比（3~4 天）
  ↓
阶段6  结果汇总 → 回填论文表格
```

总计约 2~3 周（单卡）。Checkpoint 默认存于 `$NSO_RUN_ROOT/models/<exp_name>/`（我使用 `/mnt/nso_data/nso_runs/`）。

---

## 二、阶段 0：环境与数据自检

### 0.1 模块单元验证（无需 Habitat）

```bash
conda activate nso
cd ~/NSO
python scripts/eval_nso_paper.py
```

应全部通过：OV-SDF、STGHP、RPN-UQ、融合奖励、NSO_Components。

### 0.2 数据集

```bash
python verify_mp3d_setup.py
ls data/scene_datasets/gibson   # 多为软链接，指向云盘场景
```

论文实验场景（§5.1）：
- **Gibson（10）：** Cantwell、Adrian、Denmark、Eastville、Edgemere、Elmira、Eudora、Greigsville、Pablo、Sands
- **MP3D（10）：** val split 中可行进面积 > 80 m² 的场景
- **典型分析：** Cantwell（161.2 m²）、Adrian（94.7 m²）、Beechwood（203.5 m²）

### 0.3 冒烟

```bash
bash scripts/train_stage0_smoke.sh
```

---

## 三、阶段 1：SLAM + 局部策略

`pretrained_models/model_best.{slam,local,global}` 为 ANS 官方权重，**建议直接复用，跳过本阶段**。

若需从头训练：

```bash
bash scripts/train_stage1_slam_local.sh
```

收敛标准：占据 loss < 0.1，位姿误差 < 5 cm/步。

---

## 四、阶段 2：全局策略（论文核心训练）

### 4.1 推荐命令

```bash
bash scripts/run_nso_paper_train.sh
# 等价于 python main.py --paper_mode ...
```

`--paper_mode` 启用：OV-SDF、STGHP、RPN-UQ、IGCR、融合奖励、回环后端。

### 4.2 超参数（论文表 1）

| 参数 | 值 |
|------|-----|
| `num_local_steps` | 25 |
| `global_lr` | 2.5e-5 |
| `γ` | 0.99 |
| `map_resolution` | 5 cm |
| `λ_sem`, `λ_struct` | 0.12 |
| `λ_front` | 0.15 |
| `α` (RPN μ) | 2.0 |
| `β` (RPN σ²) | 1.0 |
| `T_MC` | 10 |
| `N_topo` | 100 |

### 4.3 监控（TensorBoard）

```bash
tensorboard --logdir $NSO_RUN_ROOT/models/paper_nso/
```

| 指标 | 期望 |
|------|------|
| `g_episode_rewards` | 上升后平台 |
| `reach_losses` | 降至 < 0.3 |
| Paper 覆盖率快照 | 400 步 > 80% |
| 无效目标频次 | 趋近个位数 |

---

## 五、阶段 3：RPN-UQ 校准

RPN-UQ 在阶段 2 中已在线更新（每 25 步）。本阶段做校准检验：

- 收集 ≥ 5000 条 (预测概率, 实际可达) 样本
- **ECE < 0.05** → 进入主实验
- 否则加大 `rpn_lambda_ece` 离线精调

---

## 六、阶段 4：主实验评估

### 6.1 协议

- 20 场景，50 回合/场景，固定种子
- 1000 步/回合，冻结所有网络（`--eval 1`）
- RPN-UQ 推理保持 MC-Dropout 开启

```bash
python main.py --paper_mode --eval 1 \
  --load_global <best_ckpt> \
  --load_slam pretrained_models/model_best.slam \
  --load_local pretrained_models/model_best.local \
  --goal_reachability_model_path <best_ckpt>/model_best.reach \
  --num_episodes 50 --max_episode_length 1000
```

### 6.2 指标

覆盖率、探索面积 (m²)、漂移 RMSE (cm)、无效目标频次、具身成功率、回环次数。

### 6.3 对比基线（论文表 2）

Frontier、ANS、SemExp、PONI、OVRL、NSO-fixed、**NSO (Ours)**

### 6.4 论文目标结果（20 场景均值）

| 方法 | 覆盖率 | 漂移 | 无效目标 |
|------|--------|------|---------|
| ANS | 74.2±3.1% | 48.7 cm | 43 |
| PONI | 76.1±3.6% | 47.2 cm | 31 |
| **NSO** | **92.4±1.8%** | **11.9 cm** | **2** |

---

## 七、阶段 5：消融实验

配置矩阵（`scripts/eval_nso_paper.py` 内 `ABLATION_CONFIGS`）：

| 配置 | OV-SDF | STGHP | RPN-UQ | IGCR | 覆盖率 (400步) |
|------|--------|-------|--------|------|----------------|
| 纯几何 | × | × | × | × | 42.1% |
| +固定语义 | fixed | × | × | × | 61.4% |
| +OV-SDF | ✓ | × | × | × | 67.2% |
| +STGHP | ✓ | ✓ | × | × | 80.3% |
| +RPN-UQ | ✓ | ✓ | ✓ | × | 89.7% |
| **完整 NSO** | ✓ | ✓ | ✓ | ✓ | **92.4%** |

```bash
python scripts/eval_nso_paper.py --ablation all --scene Cantwell
```

消融各配置需分别训练全局策略；为节省算力，可从 full NSO checkpoint 继训 5k 回合。

---

## 八、阶段 6：结果回填

1. `eval_results/*.json` → 论文表 2/3/4
2. 导出可视化：拓扑图叠加轨迹、语义密度热力图、RPN 可靠性曲线
3. 核对论文叙述与实测数字一致

---

## 九、风险与应对

| 风险 | 应对 |
|------|------|
| CLIP 推理拖慢训练 | `--semantic_interval 5` 降频 |
| RPN 早期标签噪声 | 前 50k 步 α 线性升温 |
| Adrian 场景 navmesh 卡住 | 单进程 `-n 1`，或跳过该场景 |
| 推送/存储大文件失败 | Git LFS；bundle 分批上传 |
| stage3/4 权重未入库 | 从 `/mnt/nso_data/nso_runs/models/` 手动备份 |

---

## 十、快速命令

```bash
# 模块验证
python scripts/eval_nso_paper.py

# 论文模式训练
bash scripts/run_nso_paper_train.sh

# 论文模式评估
python main.py --paper_mode --eval 1 --load_global <ckpt> ...

# 消融
python scripts/eval_nso_paper.py --ablation all
```
