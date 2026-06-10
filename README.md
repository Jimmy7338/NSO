# NSO：面向大尺度未知场景的开放词汇语义增强层次化主动覆盖探索

**作者 / 项目开创者：** 李兆宇  
**论文：** [`Semantic_Enhanced_Active_SLAM_Paper.tex`](Semantic_Enhanced_Active_SLAM_Paper.tex)

---

## 项目简介

NSO（*Neural Semantic-Structure Active Exploration with Open-vocabulary*）是我在 Active Neural SLAM（ANS, ICLR 2020）双层解耦架构上，面向**大尺度室内主动覆盖建图**独立设计与实现的研究系统。

我在实践中发现，当场景从单房间扩展到多房间、长走廊时，纯几何前沿法与封闭集语义方法都会遇到三类耦合瓶颈：

1. **全局目标不可达**——长期目标频繁落在墙体后方，引发规划震荡；
2. **语义泛化不足**——固定类别检测器无法适应新物体与新任务；
3. **缺乏拓扑认知**——智能体在多房间结构中反复回溯，探索效率随尺度衰退。

针对这些问题，我提出并实现了四项可独立消融的核心机制：

| 模块 | 全称 | 作用 |
|------|------|------|
| **OV-SDF** | 开放词汇语义密度场 | CLIP + GroundingDINO 构建语义密度，支持自然语言查询驱动探索 |
| **STGHP** | 语义拓扑图层次规划 | 在线构建房间-门框拓扑图，室级目标 + 局部 FMM 两级解耦 |
| **RPN-UQ** | 不确定性感知具身可达性预测 | MC-Dropout 输出 μ/σ²，风险规避式目标采样 |
| **IGCR** | 信息增益覆盖奖励 | 互信息增益替代简单面积增量，与信息论主动感知对齐 |

回环检测（NetVLAD + CLIP 语义指纹）作为**后端系统组件**集成，用于抑制长时里程计漂移，不作为核心创新贡献。

---

## 系统架构

```
RGB-D 观测
    ├── Neural SLAM ──────────► 占据地图 / 位姿
    ├── CLIP + GroundingDINO ─► OV-SDF 语义密度场 M_sem
    │
    ├── STGHP 拓扑图构建 ─────► 房间节点 + 门框边 G=(V,E)
    │
    ├── 全局策略 (PPO)
    │     ├── Actor / Critic
    │     └── RPN-UQ (μ_reach, σ²_reach)
    │
    ├── 层次规划器 ───────────► 拓扑层目标 → FMM 几何层路径
    │
    └── 局部策略 (ResNet-18 + LSTM) ─► 离散动作
              │
              ▼
         IGCR + 语义/结构/前沿融合奖励
```

详细设计见论文第 4 节；代码入口见 [`nso/`](nso/) 包与 [`main.py`](main.py)。

---

## 快速开始

### 环境

```bash
conda activate nso          # 或 nso_h2（Habitat 2.x）
cd NSO
pip install -r requirements.txt
# CLIP（OV-SDF 必需）
pip install git+https://github.com/openai/CLIP.git
```

### 论文完整配置（一键启用四项创新）

```bash
python main.py --paper_mode --eval 0
# 或
bash scripts/run_nso_paper_train.sh
```

`--paper_mode` 自动启用：OV-SDF、STGHP、RPN-UQ、IGCR、融合奖励与回环后端。

### 模块验证（无需 Habitat）

```bash
python scripts/eval_nso_paper.py
```

### 评估

```bash
python main.py --paper_mode --eval 1 \
  --load_global <checkpoint>/model_best.global \
  --load_slam pretrained_models/model_best.slam \
  --load_local pretrained_models/model_best.local
```

---

## 代码结构

```
NSO/
├── Semantic_Enhanced_Active_SLAM_Paper.tex   # 论文原文
├── main.py / arguments.py / model.py       # 训练主入口（基于 ANS 扩展）
├── nso/                                    # 四项核心创新实现
│   ├── clip_semantic_map.py                  # OV-SDF
│   ├── topo_graph.py                         # STGHP
│   ├── reachability_uq.py                    # RPN-UQ
│   └── components.py                         # 统一管理器
├── utils/reward.py                           # IGCR + 融合奖励
├── scripts/eval_nso_paper.py                 # 评估与消融脚本
├── pretrained_models/                        # ANS 官方预训练权重
├── trained_models/                           # 自训练各阶段 checkpoint
└── docs/                                     # 项目文档
```

---

## 实验结果（论文表 2，Gibson+MP3D 20 场景均值）

| 方法 | 覆盖率 (%) | 漂移 (cm) | 无效目标 |
|------|-----------|----------|---------|
| ANS | 74.2±3.1 | 48.7±6.2 | 43±11 |
| PONI | 76.1±3.6 | 47.2±6.0 | 31±8 |
| **NSO (Ours)** | **92.4±1.8** | **11.9±1.7** | **2±1** |

典型场景 Cantwell（161.2 m²）：覆盖率 92.4%，无效目标 2 次。

---

## 文档索引

| 文档 | 说明 |
|------|------|
| [完整项目指南](docs/COMPLETE_PROJECT_GUIDE.md) | 技术原理、四项创新、代码对照 |
| [使用说明](docs/INSTRUCTIONS.md) | 训练、评估、场景指定 |
| [训练与实验方案](docs/TRAINING_AND_EVALUATION_PLAN.md) | 分阶段训练与论文复现 |
| [对比实验指南](docs/COMPARISON_EXPERIMENT_GUIDE.md) | 基线对比与消融协议 |
| [自训练权重说明](trained_models/README.md) | 各阶段 checkpoint |
| [场景与数据集](docs/SCENE_SELECTION_AND_DATASETS.md) | 论文 20 场景与数据配置 |
| [故障排除](docs/TROUBLESHOOTING.md) | 常见问题（含 NSO 专项） |
| [GitHub 上传](GITHUB_UPLOAD_GUIDE.md) | LFS 与大文件推送 |
| [Habitat 2 迁移](docs/HABITAT2_MIGRATION.md) | Habitat 2.x 可选升级 |

**归档（早期 YOLO / SSC 实验）：** [SSC 集成](docs/SSC_INTEGRATION_GUIDE.md)、[语义训练](docs/TRAINING_WITH_SEMANTIC.md)、[语义检测优化](docs/SEMANTIC_DETECTION_OPTIMIZATION.md)

---

## 引用

若使用本项目，请引用：

```bibtex
@article{zhaoyu2025nso,
  title={NSO: 面向大尺度未知场景的开放词汇语义增强层次化主动覆盖探索},
  author={李兆宇},
  year={2025}
}
```

## 致谢

本项目基于 [Active Neural SLAM](https://github.com/devendrachaplot/Neural-SLAM)（Chaplot et al., ICLR 2020）开源实现扩展，使用 [Habitat](https://github.com/facebookresearch/habitat-lab) 仿真平台。

**License:** MIT（见 [LICENSE](LICENSE)）
