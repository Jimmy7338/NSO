# NSO 训练与实验检验方案

> 配套论文：《NSO：面向大尺度未知场景的开放词汇语义增强层次化主动覆盖探索》
> 对应代码：`nso/` 包（OV-SDF / STGHP / RPN-UQ / IGCR）+ `main.py --paper_mode`
> 硬件基准：单卡 RTX 3090（24 GB），Habitat-Lab + Gibson/MP3D

---

## 一、总体路线图

```
阶段0 环境与数据自检（0.5 天）
  ↓
阶段1 基础模块训练：Neural SLAM + 局部策略（约 2~3 天 GPU）
  ↓
阶段2 全局策略训练：融合奖励 PPO（约 3~5 天 GPU）
  ↓
阶段3 RPN-UQ 在线自监督训练（与阶段2 并行/继训，约 1~2 天）
  ↓
阶段4 主实验评估：20 场景 × 50 回合（约 2~3 天）
  ↓
阶段5 消融实验 + 基线对比（约 3~4 天）
  ↓
阶段6 结果汇总 → 回填论文表格
```

总计约 2~3 周（单卡）。各阶段产出 checkpoint 均存于 `$NSO_RUN_ROOT/models/<exp_name>/`。

---

## 二、阶段 0：环境与数据自检

### 0.1 模块单元验证（无需 Habitat）

```bash
conda activate nso
cd ~/NSO
python scripts/eval_nso_paper.py        # 5 项模块验证应全部 ✓
```

验证项：OV-SDF（CLIP 语义密度场）、STGHP（拓扑图）、RPN-UQ（MC-Dropout）、融合奖励、NSO_Components 集成。

### 0.2 数据集检查

```bash
python verify_mp3d_setup.py             # MP3D 场景完整性
ls data/scene_datasets/gibson | wc -l   # Gibson 场景数
```

实验所需场景（论文 §5.1）：
- **Gibson（10 个）**：Cantwell、Adrian、Denmark、Eastville、Edgemere、Elmira、Eudora、Greigsville、Pablo、Sands
- **MP3D（10 个）**：从 val split 选取可行进面积 > 80 m² 的场景

### 0.3 冒烟测试

```bash
bash scripts/train_stage0_smoke.sh      # 短回合跑通完整流水线
```

通过标准：单回合无崩溃、语义密度图非全零、拓扑图节点数 ≥ 1、RPN loss 正常下降。

---

## 三、阶段 1：基础模块训练（SLAM + 局部策略）

已有预训练权重 `pretrained_models/model_best.{slam,local,global}` 可直接复用（来自 ANS）。
**若复用预训练，本阶段可跳过**；若需从头训练：

```bash
bash scripts/train_stage1_slam_local.sh
# 等价命令：
python main.py \
  --train_slam 1 --train_local 1 --train_global 0 \
  --num_episodes 10000 --max_episode_length 1000 \
  --exp_name stage1_slam_local
```

| 项目 | 配置 |
|---|---|
| 训练步数 | ≥ 10M 帧（ANS 论文标准） |
| 监控指标 | SLAM 占据预测 loss、位姿估计误差、局部策略模仿 loss |
| 收敛标准 | 占据 loss < 0.1，位姿误差 < 5 cm/步 |
| checkpoint | `model_best.slam`、`model_best.local` |

---

## 四、阶段 2：全局策略训练（融合奖励 PPO）

NSO 的核心训练阶段。加载阶段 1 权重，冻结 SLAM 与局部策略，仅训练全局策略：

```bash
bash scripts/run_nso_paper_train.sh
# 核心参数（--paper_mode 自动启用）：
#   OV-SDF:  --use_open_vocab_semantic --clip_model ViT-B/32
#   STGHP:   --use_topo_graph --topo_update_period 100
#   RPN-UQ:  --use_rpn_uq --rpn_mc_samples 10 --rpn_dropout 0.1
#   IGCR:    --use_igcr
#   奖励系数: λ_sem=0.12, λ_struct=0.12, λ_front=0.15
```

### 4.1 训练超参数（论文表 1）

| 参数 | 值 |
|---|---|
| `global_lr` | 2.5e-5 |
| PPO clip / epoch / mini-batch | 0.2 / 4 / auto |
| γ（折扣因子） | 0.99 |
| `num_local_steps`（全局决策周期） | 25 |
| `max_episode_length` | 1000 |
| 训练总回合 | ≥ 30k（约 750k 全局决策步） |

### 4.2 训练监控（TensorBoard）

```bash
tensorboard --logdir $NSO_RUN_ROOT/models/paper_h2/
```

| 指标 | 期望趋势 | 异常处理 |
|---|---|---|
| `g_episode_rewards` | 持续上升后平台 | 长期不升 → 检查奖励分项 breakdown |
| 奖励分解 `r_ig / r_sem / r_struct / r_front` | r_ig 主导，其余非零 | r_sem≡0 → 检查 CLIP/检测器 |
| `reach_losses`（RPN） | 下降至 < 0.3 | 不降 → 检查具身标签生成 |
| 覆盖率（训练时） | 400 步快照 > 80% | 停滞 → 调大 λ_front 或检查拓扑图 |
| 无效目标频次 | 随训练下降至个位数 | 不降 → 提高 α 或检查 UQ 掩码 |

### 4.3 中断恢复

```bash
python main.py --paper_mode \
  --load_global $RUN_ROOT/models/paper_h2/model_best.global \
  --load_slam pretrained_models/model_best.slam \
  --load_local pretrained_models/model_best.local ...
```

---

## 五、阶段 3：RPN-UQ 在线自监督继训

RPN-UQ 在阶段 2 已随主训练在线更新（`RPNTrainer`，每 25 步一次梯度更新）。
本阶段做**校准检验与必要的精调**：

### 5.1 校准检验

收集 ≥ 5000 条（预测概率, 实际可达）样本对，绘制可靠性曲线（reliability diagram）：

- **ECE < 0.05**：校准良好，进入阶段 4
- **ECE ≥ 0.05**：用收集数据离线精调 RPN 头（冻结其余参数），`--rpn_lambda_ece 0.2` 加强校准正则

### 5.2 MC 采样次数权衡

| T_MC | 推理耗时（80×80, CPU） | σ² 稳定性 |
|---|---|---|
| 5 | ~15 ms | 偏低 |
| **10（默认）** | ~30 ms | 良好 |
| 20 | ~60 ms | 边际增益小 |

---

## 六、阶段 4：主实验评估

### 6.1 评估协议（论文 §5.1）

- 20 个场景（Gibson 10 + MP3D 10），每场景 **50 回合**，固定随机种子序列 `seed = 1..50`
- 单回合上限 1000 步；评估时冻结所有网络（`--eval 1`）
- RPN-UQ 推理保持 MC-Dropout 开启（这是方法的一部分，非训练泄漏）

```bash
python main.py --paper_mode --eval 1 \
  --load_global <best_ckpt> --load_slam ... --load_local ... \
  --num_episodes 50 --max_episode_length 1000 \
  --exp_name eval_full_nso
```

### 6.2 评估指标（论文 §5.2）

| 指标 | 定义 | 采集来源 |
|---|---|---|
| 覆盖率 Exploration Ratio | 已探索面积 / GT 可行进面积 | `PaperMetricsTracker` |
| 探索面积（m²） | 已探索格点 × 分辨率² | 同上 |
| 轨迹漂移 RMSE（cm） | 估计位姿 vs GT 位姿 | 同上 |
| 无效目标频次 | FMM 判定不可达的全局目标计数 | 同上 |
| 具身目标成功率 | 25 步内抵达目标 50 cm 邻域的比例 | RPN 标签流 |
| 回环触发次数 | PGO 触发计数 | LoopDetector |

### 6.3 统计要求

- 每项指标报告 **均值 ± 标准差**（50 回合）
- NSO vs 最强基线做配对 t 检验，显著性水平 p < 0.05
- 结果 JSON 存于 `eval_results/`，由 `scripts/eval_nso_paper.py` 汇总为表格

---

## 七、阶段 5：消融实验与基线对比

### 7.1 消融矩阵（论文表 4，配置已写入 `eval_nso_paper.py` 的 `ABLATION_CONFIGS`）

| 配置 | OV-SDF | STGHP | RPN-UQ | IGCR | 命令标签 |
|---|---|---|---|---|---|
| 纯几何 | × | × | × | × | `geometry_only` |
| +固定语义 | fixed (YOLOv8) | × | × | × | `fixed_semantic` |
| +OV-SDF | ✓ | × | × | × | `ov_sem` |
| +STGHP | ✓ | ✓ | × | × | `ov_sem_topo` |
| +RPN-UQ | ✓ | ✓ | ✓ | × | `ov_sem_topo_rpn` |
| 完整 NSO | ✓ | ✓ | ✓ | ✓ | `full_nso` |

执行：场景 Cantwell，**400 步快照** + 完整 1000 步两组，各 20 回合。

```bash
python scripts/eval_nso_paper.py --ablation all --num_episodes 20 --scene Cantwell
```

注意事项：
- 消融各配置需要**分别训练**对应的全局策略（奖励结构不同），共 6 个 checkpoint；
  为节省算力，可从完整 NSO checkpoint 出发做 5k 回合的适配性继训（fine-tune）
- `fixed_semantic` 与 `ov_sem` 的对比是论文核心论点之一（开放词汇增益），需额外做
  **零样本迁移测试**：在训练时未见过的 MP3D 场景上评估两者差距

### 7.2 基线对比（论文表 2/3）

| 基线 | 来源 | 适配说明 |
|---|---|---|
| Frontier | 经典几何法 | 项目内置（`scripts/run_comparison.sh`） |
| ANS | ICLR 2020 官方权重 | `pretrained_models/` 直接评估 |
| SemExp | CoRL 2022 | 检测头换为本项目 YOLOv8，目标函数改为覆盖率 |
| PONI | 2022 | 势函数模块按原文复现，前端共享本项目 SLAM |
| OVRL | CVPR 2022 | 加载官方预训练视觉编码器，其余沿用 ANS 框架 |

所有基线使用与 NSO **相同的评估协议**（同场景、同种子、同步数上限）。

### 7.3 计算开销分析（论文 §7 局限部分）

记录各模块单帧推理耗时（RTX 3090）：

```
CLIP ViT-B/32 编码      : ~__ ms（实测填入）
GroundingDINO/YOLO 检测 : ~__ ms
RPN-UQ (T_MC=10)        : ~__ ms
拓扑图更新（每100步）   : ~__ ms
```

---

## 八、阶段 6：结果汇总与论文回填

1. `eval_results/*.json` → `scripts/eval_nso_paper.py` 自动输出论文格式表格
2. 回填以下论文表格（当前为预期值占位，**必须以实测替换**）：
   - 表 2：主实验 20 场景平均
   - 表 3：Cantwell 典型场景
   - 表 4：消融实验
3. 导出可视化素材：
   - 拓扑图叠加探索轨迹图（`nso.topo_graph.draw_on_map`）
   - 语义密度热力图（OV-SDF 归一化输出）
   - 可靠性曲线（RPN-UQ 校准）
   - 覆盖率-步数曲线（各方法对比）
4. 检查实验数字与论文叙述一致（增益百分点、显著性结论）

---

## 九、风险与应对

| 风险 | 概率 | 应对 |
|---|---|---|
| CLIP 推理拖慢训练吞吐 | 高 | `--semantic_interval 5` 降频检测；或预提取文本嵌入缓存 |
| RPN 早期标签噪声导致掩码误杀 | 中 | 前 50k 步将 α 线性升温（0→2.0）；β 同理 |
| 拓扑图在开放大厅场景节点过少 | 中 | `min_room_area_cells` 调低；门框检测失效时退化为纯几何前沿 |
| 消融需训 6 个模型算力不足 | 高 | 从 full_nso 继训 5k 回合替代从头训练；论文注明该协议 |
| 基线复现结果与原文偏差 | 中 | 报告自测数字并标注复现协议；引用原文数字作参考列 |
| Habitat 长跑显存泄漏 | 低 | 每 5k 回合自动保存并重启进程（`run_stage_in_tmux.sh` 已支持） |

---

## 十、快速命令索引

```bash
# 模块验证（不需要 Habitat）
python scripts/eval_nso_paper.py

# 完整论文模式训练
bash scripts/run_nso_paper_train.sh

# 主实验评估
python main.py --paper_mode --eval 1 --load_global <ckpt> ...

# 消融实验（全部配置）
python scripts/eval_nso_paper.py --ablation all --num_episodes 20

# 监控
tensorboard --logdir $NSO_RUN_ROOT/models/paper_h2/
```
