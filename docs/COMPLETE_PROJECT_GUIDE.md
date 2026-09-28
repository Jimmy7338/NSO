# NSO 完整项目指南

> **作者：** 李兆宇（项目开创者）
> **当前研究入口：** [Goal](research/GOAL_PROMPT.md) · [V46 论文稿](thesis/NSO_Manuscript_V46_20260922.tex) · [严格审查](thesis/V46_STRICT_REVIEW_20260923.md)
> **证据说明更新：** 2026-09-23

**本文保留早期 ANS 神经网络路径的设计和操作说明，不是当前 CPU 原型的实现报告，也不是已完成的神经系统实验结果。** 下文“论文”章节号指[历史稿](../Semantic_Enhanced_Active_SLAM_Paper.tex)，不能直接对应 V46。命令、模型及数据依赖须按实际环境核查，`--paper_mode` 启用配置不表示四模块已验证有效。

当前研究使用 CPU 虚拟传感和 TSDF 重建；已记录的 S/G 联合指标相对差为 6.1638%，限于两个已见、相近模板布局的受控比较。未验证原神经 ANS、自然开放词汇前端、四模块各自收益或独立场景泛化。当前实现与结果入口见 [README](../README.md) 和 [证据复核](thesis/V46_EVIDENCE_REVIEW_20260922.md)。

---

## 目录

1. [研究动机与定位](#1-研究动机与定位)
2. [系统架构](#2-系统架构)
3. [四模块历史设计](#3-四模块历史设计)
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

## 3. 四模块历史设计

以下为原神经路径的设计说明，其算法有效性与性能尚不能由当前 CPU 机制实验推出。

### 3.1 OV-SDF（开放词汇语义密度场）

**论文 §4.2 | 代码：** `nso/clip_semantic_map.py`

语义密度定义为 CLIP 嵌入空间的余弦相似度核：

```
S_{i,j}(q) = Σ cos(φ_v(I[b_k]), φ_t(q)) · conf_k
```

- 检测前端：GroundingDINO（降级：YOLOv8 + CLIP）
- 默认双查询：`indoor furniture and appliances` (0.7) + `doorway and passage` (0.3)
- 设计目标：通过修改自然语言 `q` 切换探索偏好；本项目尚无充分实验验证其零样本泛化收益

### 3.2 STGHP（语义拓扑图层次规划）

**论文 §4.3 | 代码：** `nso/topo_graph.py`

- **节点 V：** 已探索连通分量（房间），属性：面积 A、前沿长度 F、语义密度 S̄
- **有向边 E：** 门框/狭窄通道，属性：位置、通过置信度、未探索邻域
- **拓扑层（每 100 步）：** 式 (5) 选择目标房间
- **几何层（每 25 步）：** 目标房间内 FMM 前沿探索

拓扑层在房间图上选择目标，局部层仍需处理栅格路径；这里不据此宣称完整规划复杂度或实测耗时已降低。

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

历史配置开关：`--paper_mode`（见 `arguments.py`）。配置存在不等于训练、消融或效果验证已经完成。

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

历史文档 [SEMANTIC_REWARD_EXPLANATION.md](./SEMANTIC_REWARD_EXPLANATION.md) 描述的是早期 YOLOv8 固定类别方案；本节为后续 OV-SDF 神经路径设计。当前 CPU 研究目标以 [Goal](research/GOAL_PROMPT.md) 为准。

---

## 6. 安装与环境

### 6.1 历史神经路径拟用配置（非实测运行环境）

- Ubuntu 20.04+，Python 3.8/3.9
- NVIDIA GPU：显存需求须针对模型和批量实测；此前列出的 RTX 3090 不是已核实论文实验的硬件记录
- Conda 环境：`nso`（Habitat 1.x）或 `nso_h2`（Habitat 2.x）

### 6.2 依赖安装

```bash
pip install -r requirements.txt
pip install git+https://github.com/openai/CLIP.git    # OV-SDF
# 可选：pip install groundingdino-py                  # 候选检测前端
```

### 6.3 数据与场景

- Gibson / MP3D 场景网格需单独下载，仓库内 `data/scene_datasets/` 多为**软链接**指向云盘
- PointNav 配置：`data/datasets/pointnav/`
- 详见 [场景与数据集](./SCENE_SELECTION_AND_DATASETS.md)

### 6.4 预训练权重

`pretrained_models/model_best.{global,local,slam}` 是拟用的 ANS 官方预训练权重路径。运行前需确认真实权重已取得、不是 Git LFS 指针，并核对兼容性；文件名存在不证明训练完成。

---

## 7. 训练与评估

历史分阶段设想见 [训练与实验方案](./TRAINING_AND_EVALUATION_PLAN.md)。下列命令尚不是当前 CPU 结果的复现入口。

**快速训练：**

```bash
python main.py --paper_mode --eval 0 --exp_name paper_nso
```

**历史拟议评估配置（无已核实的完整执行结果）：**

- 20 场景（Gibson 10 + MP3D 10），50 回合/场景
- 单回合 1000 步，全局策略每 25 步更新
- 指标：覆盖率、探索面积、漂移 RMSE、无效目标、具身成功率、回环次数

**结果状态：未验证。** 此处原有 ANS、PONI、NSO 的覆盖率、漂移和无效目标表格没有可核实的实测依据，已删除，不能作为论文结果或性能目标引用。上述 20 场景、50 回合仅为历史拟议配置；真实 CPU 实验采用不同任务和指标，见 [当前结果](../README.md#当前可复核的受控结果)。

---

## 8. 关键参数

| 参数 | 默认值 | 含义 |
|------|--------|------|
| `--paper_mode` | off | 启用历史四模块配置 |
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

下表描述原神经路径拟增加的功能；它不表示已完成与 ANS 的公平性能比较：

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

1. 尚无可核实的本项目 CLIP + GroundingDINO 推理耗时，原定量耗时表述已删除。
2. 原神经路径和 Habitat 主实验尚未完成有效性验证；不能将 CPU 原型结果转给这些实现。
3. CPU 原型使用受控类别提示、已知导航约束和真实位姿，任务范围不等于全未知环境主动 SLAM；自然设备与实车效果待验证。
4. 当前实验未证明学习型 RPN-UQ、开放词汇泛化或四模块各自具有独立性能收益。

下一步执行 [当前 Goal](research/GOAL_PROMPT.md) 中有停止条件的语义决策机制验证；实车部署暂不作为前置任务。

---

## 附录：目录结构

```
NSO/
├── Semantic_Enhanced_Active_SLAM_Paper.tex
├── main.py / model.py / arguments.py
├── nso/                    # 历史神经模块及后续 CPU 原型
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
