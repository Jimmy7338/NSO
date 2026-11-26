# 从零自主探索的四栋留出建筑对照

四栋封存建筑各运行扩池中性 G、扩池类别 S、原 S、严格最近可达前沿。每轮以未知地图和该建筑共同首帧启动，预算为 512 个计费动作。相机、激光、圆形足迹、逐动作可行性检查、重规划和返航合同沿用原协议。仿真向两主臂提供当前可见设施的理想标签。仅 S 在局部表面收益中使用该类别条件先验，两主臂均用中性规则选择全局目标。

| 建筑 | 房间连接 | 真值可达自由面积 |
|---|---|---:|
| `ordinary_chain_13` | 3 道连通门的四房链式 | 123.60 m² |
| `ordinary_cycle_14` | 4 道连通门的四房环式 | 144.44 m² |
| `ordinary_chain_15` | 3 道连通门的四房链式 | 165.08 m² |
| `ordinary_cycle_16` | 4 道连通门的四房环式 | 187.60 m² |

可达自由面积由独立评价器的初始位姿连通参考格计数乘以 $0.20^2\,\mathrm{m}^2$ 得到。[建筑与评价背景](measurement/layouts.csv)

## 结果与验证记录

- [冻结协议](PROTOCOL.json)、[运行与首帧哈希](runner/holdout_freeze.json)、[独立评价器哈希](measurement/EVALUATOR_FREEZE.json)、[两处已见状态的回归](diagnostics/regression_verification.json)与[总核验](completion_audit.json)。
- [逐建筑主表](results/terminal_J_table_zh.md)、[逐臂终点结果](measurement/results/episodes.csv)、[三组逐建筑配对差](measurement/results/contrasts.csv)和[预注册工程门槛核验](measurement/results/engineering_gate.json)。终点 $J=C_{\rm free}F_{1,5\rm cm}$ 在各策略实际停止时计算。配对差为百分点。建筑是独立单位。
- [等动作检查点](measurement/results/checkpoints.csv)、[逐动作正确自由空间覆盖](measurement/results/coverage_curve.csv)和[首次实际动作、位姿及传感差](measurement/results/pair_causal_audit.json)。早停后的登记检查点保留为未取得。
- [选路、传感与计算开销对照](results/route_mechanism_zh.md)汇总两主臂的候选量、规划耗时和设施净匹配面积。规划耗时记录于共享仿真工作站。
- [终点配对图](results/terminal_J_by_building.pdf) · [可编辑图](results/terminal_J_by_building.svg) · [中英文稿件段落](results/manuscript_results_zh_en.md)。右图虚线标出主对照 +0.10 个百分点的逐建筑门槛。

## 复核与复现

[逐臂证据索引](EPISODE_INDEX.json)记录所有压缩档的哈希。压缩档包含首次观测的指纹、计费动作、RGB、深度与激光指纹、候选池及评分、导航保护记录、地图占据快照和各检查点回执。逐动作覆盖记录另存为 CSV 和无损压缩的 JSON。原生 PLY 保存在索引所列的实验工作站目录，回执引用文件名、字节数和 SHA-256。输入与评价器哈希清单分别追踪继承代码、初始传感数据及参考几何。仿真部分可用和本轮一致的 Python 3.12 运行时及登记输入复算。

1. 按两份冻结清单核对前序目录和场景资产。`runner/regression.py` 复核两处已见状态的选择和第一个实际动作分歧。
2. `runner/supervise.py --phase all` 按主对照、两基线的次序运行。需要恢复同一配置时使用该脚本的 `--resume`，续用原冻结清单。
3. `measurement/evaluate.py --phase all` 独立评价完整矩阵。`report.py` 从终点记录生成配对图和双语正文。

前序四备选规则及两个选定开发状态的结果继续归档在各自的目录。[原留出设计](https://github.com/Jimmy7338/NSO/tree/02e9c5dcc0da7df61f23d8683fec2dd92c145c1e/experiments/offline_20261121_same_goal_route_choice) · [两个开发状态](https://github.com/Jimmy7338/NSO/tree/53ef1f0b292e5f358e05c27aee56d19e4052b22c/experiments/offline_20261121_same_goal_two_witness)
