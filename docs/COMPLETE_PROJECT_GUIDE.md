# NSO 完整项目指南

> **作者：** 李兆宇（项目开创者）  
> **论文：** [`Semantic_Enhanced_Active_SLAM_Paper.tex`](../Semantic_Enhanced_Active_SLAM_Paper.tex)  
> **最后更新：** 2025-12

本文档从设计与实现角度，说明 NSO 系统的技术原理、代码结构与使用方法。若你只想快速跑通，请先看 [README](../README.md) 与 [使用说明](./INSTRUCTIONS.md)。

---

## 目录

1. [研究动机与定位](#1-研究动机与定位)
2. [系统架构](#2-系统架构)
3. [四项核心创新](#3-四项核心创新)
4. [代码与论文对照](#4-代码与论文对照)
5. [奖励函数](#5-奖励函数)
6. [安装与环境](#6-安装与环境)
7. [训练与评估](#7-训练与评估)
8. [关键参数](#8-关键参数)
9. [与 ANS 的关系](#9-与-ans-的关系)
10. [已知局限与未来工作](#10-已知局限与未来工作)

---

## 1. 研究动机与定位

### 1.1 我要解决什么问题

主动覆盖式建图的目标是让智能体在未知环境中**尽可能完整地探索并构建一致地图**，而非到达某个特定物体（ObjectNav）。我在 ANS 基础上继续这一方向，但聚焦**大尺度多房间室内场景**。

实践中我总结出三类瓶颈：

| 瓶颈 | 表现 | 我的应对 |
|------|------|---------|
| 目标不可达 | 全局目标落在墙后，FMM 报错、探索震荡 | RPN-UQ 具身可达性 + 不确定性掩码 |
| 语义封闭 | YOLO 固定类别，新场景泛化差 | OV-SDF 开放词汇 CLIP 密度场 |
| 无拓扑认知 | 长走廊反复回溯，多房间效率低 | STGHP 在线拓扑图层次规划 |

### 1.2 优化目标

将探索建模为 POMDP，优化全图**覆盖率**与**地图一致性**（而非 ObjectNav 的物体到达率）：

```
J = E[ Σ γ^t R_total(S_t, g_t) ]
```

`R_total` 由 IGCR、语义密度、结构感知、前沿引导与内在惩罚组成（论文式 10–14）。

### 1.3 与相关工作的区别

- **ANS**：仅几何占据图，无语义与拓扑；
- **SemExp / PONI**：面向 ObjectNav，封闭集语义，无拓扑图；
- **OVRL**：预训练表征强，但未针对覆盖探索做拓扑与可达性建模；
- **NSO（本项目）**：开放词汇语义 + 在线拓扑图 + 不确定性感知 RPN + 信息增益奖励，面向覆盖探索。

---

## 2. 系统架构

```
┌─────────────────────────────────────────────────────────────────┐
│                        RGB-D 观测                                │
└────────────┬───────────────────────────────┬────────────────────┘
             │                               │
             ▼                               ▼
    ┌─────────────────┐           ┌──────────────────────┐
    │  Neural SLAM    │           │ CLIP + GroundingDINO │
    │  占据 / 位姿     │           │ OV-SDF → M_sem       │
    └────────┬────────┘           └──────────┬───────────┘
             │                               │
             └──────────────┬────────────────┘
                            ▼
                 ┌─────────────────────┐
                 │ 多通道地图 M         │
                 │ obs/exp/traj/reach/sem│
                 └──────────┬──────────┘
                            │
         ┌──────────────────┼──────────────────┐
         ▼                  ▼                  ▼
  ┌─────────────┐  ┌──────────────┐  ┌───────────────┐
  │ STGHP 拓扑图 │  │ 全局策略 PPO  │  │ 回环后端       │
  │ G=(V,E)     │  │ Actor/Critic  │  │ NetVLAD+PGO   │
  └──────┬──────┘  │ RPN-UQ       │  └───────────────┘
         │           └──────┬───────┘
         └────────┬─────────┘
                  ▼
         ┌─────────────────┐
         │ 层次规划器       │
         │ 拓扑层 → FMM层  │
         └────────┬────────┘
                  ▼
         ┌─────────────────┐
         │ Local Policy     │
         │ ResNet-18+LSTM  │
         └─────────────────┘
```

**设计原则：** 我在 ANS「全局长期目标 + 局部执行」骨架上，把语义、拓扑、可达性分别注入**感知层、规划层、采样层**，形成三层协同而非简单奖励叠加。

---

## 3. 四项核心创新

### 3.1 OV-SDF（开放词汇语义密度场）

**论文 §4.2 | 代码：** `nso/clip_semantic_map.py`

语义密度定义为 CLIP 嵌入空间的余弦相似度核：

```
S_{i,j}(q) = Σ cos(φ_v(I[b_k]), φ_t(q)) · conf_k
```

- 检测前端：GroundingDINO（降级：YOLOv8 + CLIP）
- 默认双查询：`indoor furniture and appliances` (0.7) + `doorway and passage` (0.3)
- 优势：修改自然语言 `q` 即可零样本切换探索偏好，无需重训检测器

### 3.2 STGHP（语义拓扑图层次规划）

**论文 §4.3 | 代码：** `nso/topo_graph.py`

- **节点 V：** 已探索连通分量（房间），属性：面积 A、前沿长度 F、语义密度 S̄
- **有向边 E：** 门框/狭窄通道，属性：位置、通过置信度、未探索邻域
- **拓扑层（每 100 步）：** 式 (5) 选择目标房间
- **几何层（每 25 步）：** 目标房间内 FMM 前沿探索

将多房间全局规划复杂度从 O(HW) 降至 O(|V|)。

### 3.3 RPN-UQ（不确定性感知具身可达性预测）

**论文 §4.4 | 代码：** `nso/reachability_uq.py`

- MC-Dropout（T_MC=10）输出 μ_reach 与 σ²_reach
- 掩码调制（式 6）：

```
P_final(g) ∝ π_act · μ^α · exp(-β·σ²)
```

- 标签：FMM 几何可达 ∪ 25 步具身回溯（50 cm 邻域）
- 损失：ECE-aware BCE（λ_ece=0.1）

### 3.4 IGCR（信息增益覆盖奖励）

**论文 §4.5 | 代码：** `utils/reward.py`

```
R_ig = Σ_{(i,j)∈Δexp} H(M_obs(i,j))
```

对门后等高不确定区域给予额外奖励，替代简单面积增量。

**完整奖励：**

```
R_total = R_ig + λ_sem·R_sem + λ_struct·R_struct + λ_front·R_front + R_intrinsic
```

默认：λ_sem=λ_struct=0.12，λ_front=0.15。

---

## 4. 代码与论文对照

| 论文概念 | 代码路径 | 启用参数 |
|---------|---------|---------|
| OV-SDF | `nso/clip_semantic_map.py` | `--use_open_vocab_semantic` |
| STGHP | `nso/topo_graph.py` | `--use_topo_graph` |
| RPN-UQ | `nso/reachability_uq.py` | `--use_rpn_uq` |
| IGCR + 融合奖励 | `utils/reward.py` | `--use_igcr` + `--paper_rewards` |
| 统一管理器 | `nso/components.py` | `main.py` 中 `NSO_Components` |
| 论文指标 | `utils/paper_eval.py` | 训练/评估自动记录 |
| 消融/验证 | `scripts/eval_nso_paper.py` | — |
| 回环后端 | `loop/` | `--use_loop_detection` |

一键启用全部创新：`--paper_mode`（见 `arguments.py`）。

---

## 5. 奖励函数

融合奖励由 `NSO_RewardComputer`（`utils/reward.py`）计算，各分项可独立消融：

| 分项 | 含义 | 关键机制 |
|------|------|---------|
| R_ig | 信息增益覆盖 | 新增格点占据熵 |
| R_sem | 开放词汇语义密度 | fresh mask 归一化 |
| R_struct | 结构感知 | 门框/窄道/开阔区加权，与 STGHP 共享门框检测 |
| R_front | 前沿引导 | 门框邻域 Room Boost 1.5× |
| R_intrinsic | 重复惩罚 | 已探索格点作目标时扣分 |

历史文档 [SEMANTIC_REWARD_EXPLANATION.md](./SEMANTIC_REWARD_EXPLANATION.md) 描述的是早期 YOLOv8 固定类别方案；当前论文主线为 OV-SDF。

---

## 6. 安装与环境

### 6.1 推荐配置

- Ubuntu 20.04+，Python 3.8/3.9
- NVIDIA GPU ≥ 8 GB（论文实验：RTX 3090 24 GB）
- Conda 环境：`nso`（Habitat 1.x）或 `nso_h2`（Habitat 2.x）

### 6.2 依赖安装

```bash
pip install -r requirements.txt
pip install git+https://github.com/openai/CLIP.git    # OV-SDF
# 可选：pip install groundingdino-py                  # 最优检测前端
```

### 6.3 数据与场景

- Gibson / MP3D 场景网格需单独下载，仓库内 `data/scene_datasets/` 多为**软链接**指向云盘
- PointNav 配置：`data/datasets/pointnav/`
- 详见 [场景与数据集](./SCENE_SELECTION_AND_DATASETS.md)

### 6.4 预训练权重

`pretrained_models/model_best.{global,local,slam}` 来自 ANS 官方，作为阶段 1 起点。

---

## 7. 训练与评估

完整分阶段流程见 [训练与实验方案](./TRAINING_AND_EVALUATION_PLAN.md)。

**快速训练：**

```bash
python main.py --paper_mode --eval 0 --exp_name paper_nso
```

**论文评估协议：**

- 20 场景（Gibson 10 + MP3D 10），50 回合/场景
- 单回合 1000 步，全局策略每 25 步更新
- 指标：覆盖率、探索面积、漂移 RMSE、无效目标、具身成功率、回环次数

**论文报告结果（20 场景均值）：**

| 方法 | 覆盖率 | 漂移 | 无效目标 |
|------|--------|------|---------|
| ANS | 74.2±3.1% | 48.7 cm | 43 |
| PONI | 76.1±3.6% | 47.2 cm | 31 |
| NSO | **92.4±1.8%** | **11.9 cm** | **2** |

---

## 8. 关键参数

| 参数 | 默认值 | 含义 |
|------|--------|------|
| `--paper_mode` | off | 一键启用四项创新 |
| `num_local_steps` | 25 | 局部执行 / 全局决策周期 |
| `topo_update_period` | 100 | STGHP 拓扑层更新周期 |
| `rpn_mc_samples` | 10 | RPN-UQ MC-Dropout 次数 |
| `reachability_mask_alpha` | 2.0 | μ 调制强度 α |
| `reachability_mask_beta` | 1.0 | σ² 惩罚强度 β |
| `map_resolution` | 5 cm | 栅格分辨率 |
| `global_lr` | 2.5e-5 | 全局 PPO 学习率 |

完整参数列表：`python main.py --help` 或 `arguments.py`。

---

## 9. 与 ANS 的关系

本项目**不是** ANS 的简单 fork，而是在其上的系统性扩展：

| 层次 | ANS | NSO |
|------|-----|-----|
| 感知 | 几何占据图 | + OV-SDF 开放词汇语义密度 |
| 规划 | 全局 CNN 直接采样 | + STGHP 拓扑图层次规划 |
| 采样 | 无可达性约束 | + RPN-UQ 不确定性感知掩码 |
| 奖励 | 面积增量 | + IGCR 信息增益 + 结构/前沿 |
| 后端 | 无 | 回环检测（NetVLAD + CLIP 指纹） |

基础模块（Neural SLAM、Local Policy、PPO 框架）仍沿用 ANS 实现（`model.py`、`algo/ppo.py`）。

---

## 10. 已知局限与未来工作

（与论文 §7 一致）

1. CLIP + GroundingDINO 推理约 80 ms/帧（RTX 3090），嵌入式需量化蒸馏；
2. 拓扑图依赖门框检测，弱纹理场景可能漏建边；
3. 实验主要在 Habitat 仿真，真实机器人尚未系统验证；
4. RPN-UQ 训练早期（<50k 步）不确定性可能偏高。

**我计划中的后续方向：** 真实机器人部署、CLIP 在线轻量化、多智能体协同拓扑图构建。

---

## 附录：目录结构

```
NSO/
├── Semantic_Enhanced_Active_SLAM_Paper.tex
├── main.py / model.py / arguments.py
├── nso/                    # 四项创新
├── utils/reward.py         # IGCR + 融合奖励
├── utils/paper_eval.py     # 论文指标
├── semantic/               # 历史语义模块（YOLOv8，消融用）
├── loop/                   # 回环后端
├── env/                    # Habitat 环境封装
├── scripts/                # 训练/评估脚本
├── pretrained_models/      # ANS 预训练
├── trained_models/         # 自训练 checkpoint
└── docs/                   # 文档
```

---

**文档维护：** 李兆宇  
**基础参考：** [Active Neural SLAM](https://github.com/devendrachaplot/Neural-SLAM) (Chaplot et al., ICLR 2020)
