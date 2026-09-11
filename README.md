# NSO：面向大尺度未知场景的开放词汇语义增强层次化主动覆盖探索

> **后续 Goal 入口（2026-09-11）：** 用户已将研究主线明确为“面向工业设施、覆盖约束下的语义引导主动三维建图”。请先阅读 [关键目标导向文档](docs/research/RESEARCH_GOAL_AND_STORY.md) 和 [新对话 Goal 文本](docs/research/GOAL_PROMPT.md)。V10.3 的反馈版本遗漏已由 V10.3.1 修复并重跑；机制闭环通过，但完整/关闭反馈仍为 8/8 相同轨迹，性能增量未成立。联合指标增幅不能解释为二维覆盖或精度增幅。

当前进展：[V10四模块真实接口与冷启动问题](docs/research/FOUR_MODULE_CPU_CLOSED_LOOP_V10_IMPLEMENTATION.md)。语义实包干预34项通过，类别交换改变目标；32动作三维轨迹完整回放通过，但全为原地旋转，尚未验证平移探索。[V9.1有限域语义结果与F1上限](docs/research/COMPETITION_V9_1_RESULT_AND_F1_CEILING.md)保留，完整方案优势仍未证明。

> **最新研究状态（2026-09-11）：** 保留 ANS 双层及 OV-SDF、STGHP、RPN-UQ、IGCR 四模块。[双对象24分支实验](eval_results/competition_v8_1_analysis_20260911/REPORT.md)已完成752动作及独立回放，全部路线返程、零碰撞。原评分的语义面积率比内部几何／对象性对照高10.55%，但终点F1低0.00291，且传感器排他假设失败，原进阶结论保持未通过。[统一总收益评分的单项事后诊断](eval_results/competition_v8_1_objective_diagnosis_20260911/REPORT.md)在相同42动作下得到面积+34.78%、F1+0.00979；这是已看数据的开发诊断，需在新冻结上下文确认。当前四历史二维覆盖起点已为1，未证明完整四模块或主流方法优势。[下一阶段设计](docs/research/SEMANTIC_BUDGET_ALIGNMENT_V9_DESIGN.md)与[论文研究材料](docs/research/README.md)已更新；原V7/V8失败及下列历史阶段范围均保留。

> **当前语义方案重设计：** 用户明确要求保留可验证的语义创新。已统一 CPU 三维后端、分开相机与雷达覆盖，并实现语义条件的隐藏表面规划、实测几何修正及自遮挡检查。最新完成 64 回合匹配对照与标签干预：出现语义增量，但覆盖代价和新增模块净收益尚未通过判据，仍未证明完整方案稳定优势。见 [开发依据、实现与证据边界](docs/SEMANTIC_JOINT_V3_DEVELOPMENT.md)。

> **最新机制诊断：** 已完成 224 个受控补看分支和 32 个后端尺度诊断。补看有条件地改善补全或局部精度，但旧角度奖励不能预测精度收益；三维后端尺度也需优化。见 [实验结论与下一版实施决定](docs/INSPECTION_MECHANISM_AND_QUALITY_V2.md)。尚未将这些结果当作语义优势或实车效果证明。

> **三维虚拟实验新进展：** 已运行 CPU 深度＋单线雷达→二维／TSDF 三维建图→覆盖与质量规划→独立评价的闭环，并准备 ROS 1 / ZED / 镭神雷达数据录制与同后端回放。见 [虚拟实验与小车前置工作](docs/VIRTUAL_3D_AND_ROS1_PREPARATION.md)、[32 回合三维小试](eval_results/virtual3d_pilot_v1_20260910/results.md) 和 [可旋转三维模型](eval_results/virtual3d_pilot_v1_20260910/viewer.html)。这是新双目标的流程验证；未宣称原系统、语义优势或实车导航已经通过验证。

> **2026-09-10 审计状态：** 下方 92.4% 等主实验数字尚无已核实的实测证据，应视为待实测占位值。历史 paper_fast 存在评估时 RPN 更新及面积混入奖励的问题；当前本地模型为 LFS 指针。请先阅读 [资产与评估审计](docs/ASSET_AND_EVALUATION_AUDIT.md)。

> **当前工作优先级：先验证方案，暂停扩展训练和正向论文结论。** 原语义／结构组合未通过效果检验。已完成 [根因复核、近期论文依据与重设计方案](docs/PLANNER_REDESIGN_AND_FEASIBILITY.md)，以及 [144 回合开发消融与全部失败记录](eval_results/cpu_coverage_redesign_review_20260910/results.md)。修订原型尚未稳定超过几何对照，未替换默认方法；下方四模块架构属于待验证设计，不能当作已完成有效性验证的系统。

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

## CPU 二维实验入口

无 GPU 可先运行已接通的二维闭环平台和两条几何基线，见 [CPU 平台与实测说明](docs/CPU_GRID_BASELINES.md)。该平台无需 Habitat 或模型权重；开发小试不等同于下文神经系统的论文主实验。

现已接入可见合成语义、区域拓扑和 CPU 候选可达性网络，完成 46 回合开发采集与消融，见 [机制实现、真实结果与复现](docs/CPU_MECHANISM_INTEGRATION.md)。新增模块会影响决策，但当前平均覆盖率未超过几何对照；这些结果不能作为下文完整神经系统的有效性证据。

后续已完成 [冻结协议与 176 回合正式二维比较](docs/CPU_FROZEN_EXPERIMENT.md)，并整理 [论文实验章节](docs/THESIS_CPU_EXPERIMENTS.md)。正式测试使用 11 张与已有数据内容去重的地图、2 个起点、8 种方法；当前结果仍不支持语义结构组合优于几何对照，RPN 也未显示超越软动作预算规则的明确收益。

针对项目适用场景，另新增 [大规模功能语义场景设计与 84 回合实验](docs/TARGETED_SCENE_STUDY.md)：办公走廊、仓储区和连通展厅，48/64 米、10–14 个房间，含正确、置换、统一强度和误标线索对照。64 米组组合 AUC 出现约 5.3% 的平均正向信号，但区间仍包含零且末端覆盖未提高；48 米组及总体退化同时保留。

```bash
.venv-cpu/bin/python scripts/eval_cpu.py --config configs/cpu/pilot.json --output eval_results/my_cpu_run
```

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

## 当前已验证实验结果

V10.3 使用四组程序化工业资产条件及左右互换，共八个配对历史；所有方法共享150动作基础覆盖，再执行48动作自适应巡检。主指标为“新增唯一表面积 × 最终F1@5cm”。

| 方法 | 联合指标均值 | 新增表面积 (m²) | F1增量 |
|------|-----------:|-----------------:|-------:|
| **S：正确类别条件** | **9.079661** | **9.124547** | **0.033539** |
| G：几何完成 | 7.037373 | 7.091615 | 见完整结果 |
| N：无类别 | 7.690027 | 7.741954 | 0.031061 |
| X：反转类别关联 | 4.820972 | 4.916976 | 0.019396 |

V10.3.1 中 S 相对 G/N 的检查联合指标均值提高29.02%/18.07%；40个分支全部零碰撞并回到共享锚点，独立物理与指标重放40/40通过。S 与关闭反馈仍完全相同，所以当前 IGCR 标量反馈没有性能增量。该结论限于人工标记工业资产开发域；该百分比不是二维覆盖或三维精度提升，也不代表Gibson/MP3D、自然开放词汇或TARE/FUEL等原作者系统结果。

---

## 文档索引

| 文档 | 说明 |
|------|------|
| [V10.3语义覆盖结果与理论](docs/research/V10_3_SEMANTIC_COVERAGE_RESULT_AND_THEORY.md) | 当前四模块CPU闭环、联合指标、因果消融与论文边界 |
| [V10.3.1反馈修订与指标契约](docs/research/V10_3_1_FEEDBACK_CORRECTION_AND_METRIC_CONTRACT.md) | 反馈遗漏修复、新旧对照、局部质量与下一阶段约束 |
| [网络训练审计与优化路线](docs/research/NETWORK_TRAINING_AUDIT_AND_OPTIMIZATION_PLAN.md) | V10未使用神经权重的边界、LFS资产状态、学习型语义收益头与逐模块训练顺序 |
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
