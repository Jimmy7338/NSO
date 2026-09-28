# 资产审计与评估修复交付记录

日期：2026-09-10。对应计划第 1 周的资产审计、冻结评估和指标修正。

## 已完成

- 建立并实际运行可重复的本地资产审计工具，检查文件摘要、LFS 指针、失效链接及压缩 episode JSON。
- 修复 `--eval 1` 与 `--paper_mode` 的交互：最终参数统一关闭 Global、Local、SLAM、RPN 和语义训练；评估不创建这些训练优化器。
- 冻结运行中的网络参数，记录评估前后的参数和缓冲区摘要；变化时运行失败，不标为冻结评估成功。保留 MC Dropout 的随机推理能力。
- 从实际已观测格点计算面积和累计覆盖率，停止用融合奖励反推面积。
- 每个环境动作只更新一次指标；修复同步环境及 Habitat 2 适配器的自动重置数据覆盖。
- 按全局目标尝试计数，局部重复规划不重复增加尝试数；在局部动作预算内曾到达 50 cm 邻域记成功。
- 新增逐回合 JSONL、运行配置、代码/模型摘要、采样步数、场景/episode 标识与独立运行目录。
- 汇总工具区分旧日志和 schema v2：保留零值，缺失值标 `null`；旧奖励累计值不再标作平方米。

此次没有训练策略、恢复大模型权重、修改论文主表数字或启动 Habitat 三维仿真。

## 本地资产结果

机器可读完整清单：[asset_inventory.json](../audit_results/2026-09-10/asset_inventory.json)。此清单只反映本次工作环境，未检查远端云盘、原 GPU 服务器或 LFS 服务的可访问性。

| 项目 | 结果 |
|---|---|
| 检查的本地资产条目 | 190 |
| 可解析的压缩 episode JSON | 153 |
| 预训练/自训练模型与 YOLO 文件 | 10 个，全部仍为 LFS 指针 |
| LFS 指针总数 | 11 个，另一个为数据集 ZIP；声明的总内容大小约 821 MB |
| 失效场景链接 | 3 个 |
| 本地三维场景 `.glb/.ply/.navmesh` | 0 个 |
| `.npz/.npy/.h5` 地图或回放缓存 | 未发现 |
| 其余普通文件 | 23 个；已计算摘要，未据此宣称可加载模型/可信 pickle |

失效链接为 `data/scene_datasets/17DRP5sb8fy`、`gibson`、`habitat-test-scenes`，均指向当前不存在的 `/mnt/nso_data/scene_datasets/…`。

本次已建立 `.venv` 并安装 **PyTorch 2.6.0+cpu** 用于回归检查。原环境中的 NumPy/SciPy/Matplotlib 通过 system-site-packages 使用；未安装 Habitat、OpenCV 或视觉模型依赖。该环境用于本阶段 CPU 验证，不是原三维训练环境的复刻。

## 历史评估审计

新报告：[paper_fast_audit.json](../audit_results/2026-09-10/paper_fast_audit.json)。历史 `eval_results/paper_fast/` 内原始日志、矩阵和旧 JSON 均未覆盖。

| 项目 | 审计结论 |
|---|---|
| 实际配置 | 4 进程 × 每进程 20 回合 × 500 步 |
| 原始矩阵 | 80 行 × 20 个采样点 |
| 终点覆盖率 | 79.5459%，总体标准差 25.7392 个百分点；保留为历史运行记录 |
| 是否冻结评估 | 否；日志为 `train_goal_reachability=True` |
| 旧“面积”均值 12122.9875 | 实际为被缩放的融合奖励累计量，不能恢复为真实 m² |
| 轨迹漂移 | 未有可用于 ATE 的成对估计/真值轨迹，记缺失 |
| 旧在线 coverage | 是覆盖增量，不是语义覆盖率；保留全部 1050 个快照后均值约 2.4794%，不用于论文总覆盖率 |
| 旧无效目标均值 0.8419 | 重复计数器快照的逐日志均值，不是逐回合无效目标数 |

这些修复不会使历史结果自动变成有效正式实验，需要恢复资产后重新进行冻结评估。

## 新指标口径

- `explored_area_m2 = observed_free_cells × (map_resolution_cm / 100)²`。只计算观测与环境 GT 可探索掩码的交集；例如 100 个 5 cm 格点为 0.25 m²。
- `coverage_ratio = observed_free_cells / explorable_free_cells`，独立于 `exp_reward`；`coverage_delta` 是单步增量。原 `exp_ratio` 保留为旧训练路径的周期增量别名，汇总不再使用它推断总覆盖率。
- 当前分母明确标作 **`aligned_gt_explorable_map`**：它是原 Habitat 环境对齐/裁剪后的 GT 地图，尚不是新建的起点连通分量或 navmesh 精确物理面积。跨系统比较必须对齐这一口径；下一阶段二维平台另行定义起点可达分母。
- `unreachable_goal_count` 统计同一全局目标在其预算内曾被现有规划器报告不可达/需回退的次数，每目标至多一次；不是用真值重算的全场景几何不可达率。
- 具身成功率按已完成目标尝试计算；新目标的地图坐标包含局部窗口原点，位置真值只用于评价。没有完成目标时成功率为 `null`。
- `trajectory_drift_rmse_cm` 暂为 `null`。已有单步里程计增量误差分别写入 `odometry_step_translation_rmse_cm` 与 `odometry_step_rotation_rmse_deg`；它们不冒充轨迹 ATE。
- 语义奖励只在实际发放周期采样，不重复统计用于界面显示的旧值。回环后端的晚到事件单独记数，不增加动作数。
- episode 结束先保存终止信息，再清空统计器；下一回合 reset 信息只用于下一回合。遇到不符合当前固定回合长度协议的提前/缺失终止，评估明确失败，不继续拼接出错误矩阵。

## 新输出与使用

新三维评估的输出位置为：

```text
{dump_location}/dump/{exp_name}/eval_runs/{run_id}/
  run_metadata.json
  episodes.jsonl
  explored_area.txt
  explored_ratio.txt
```

多 GPU 时沿用项目现有的 `{exp_name}_gpu{gpu_id}` 外层目录。每次评估使用唯一 `run_id`，避免不同运行混写。`run_metadata.json` 初始为 `running`，模型状态核对后才标为 `complete`；汇总还会核对回合数量、长度和模型摘要。完成标识不等于算法正确或感知模块已完整接入。

从项目根目录执行 CPU 检查：

```bash
.venv/bin/python -m unittest discover -s tests/cpu -v
```

重新审计时使用新的输出文件名，避免覆盖之前的资产/结果快照：

```bash
.venv/bin/python scripts/audit_assets.py --output audit_results/next/asset_inventory.json
.venv/bin/python scripts/summarize_eval_results.py \
  --log eval_results/paper_fast/train.log \
  --dump eval_results/paper_fast \
  --tag paper_fast_audit \
  --output audit_results/next/paper_fast_audit.json
```

未来三维评估完成后，把终端打印的完整运行目录传入 `--dump`，无需再解析文本日志：

```bash
.venv/bin/python scripts/summarize_eval_results.py \
  --dump /path/to/eval_runs/RUN_ID \
  --tag frozen_eval \
  --output /path/to/new_summary.json
```

这是新输出协议；依赖旧目录和旧 JSON 字段的外部脚本需要相应调整。汇总工具遇到已有输出路径会拒绝覆盖。

## 验证与剩余限制

**16 项 CPU 回归检查全部通过**，修改的 Python 文件语法检查通过。CPU 回归覆盖物理单位、奖励隔离、零值/缺失值、单步计数、目标预算、模型参数及 BatchNorm 缓冲区冻结、MC Dropout、两条自动重置路径、真实历史矩阵解析，以及从主程序提取的实际逐回合记录逻辑。后者在不加载 Habitat 的条件下验证末步、非整周期采样和跨回合重置。

完整 Habitat 端到端评估仍未执行：本地缺少仿真器、可用场景和真实权重。旧 Habitat 1 的 vendored API 目录也未恢复，其多进程行为尚不能实测。网络冻结和指标修复已经落地，但不能据此声明三维链路全部通过。

下一步优先搭建 CPU 二维闭环和几何基线。只有后续明确需要三维验证时，才恢复所需少量权重和场景；无需现在下载完整数据集或重新训练全部网络。
