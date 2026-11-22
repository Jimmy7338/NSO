# 192 步未知探索的几何权重与类别先验对照

本实验以同目标可见性规划为基础，沿用中性全局目标、完整局部候选池、原生观测、地图融合、逐动作保护和返航。局部评分为 $A_p + u\,0.20^2(U-U_0) - c(K-K_0)$。其中 $A_p$ 是预计表面积，$U$ 是预测可见未知栅格数，$K$ 是去程动作数，下标 0 表示原生路线。$u$ 与 $c$ 均非负。零附加权重退回原局部排序。G 和 G* 的局部面积传递函数只计算中性预计面积。继承的实例关联与全局诊断仍在共同管线中，局部排序对类别先验的隔离由独立回归记录核验。

两个开发布局来自既有实验。每栋以实际冷启动、192 个计费动作预算运行九个中性权重组合和一次原 S。路线权重在 S 的相同已测状态上按精确指令和位姿序列拟合。地图权重依据各组在两栋开发楼上的独立原生终点地图选出。两组权重封存后，链式 17、环式 18、链式 19、环式 20 各自使用共同首帧执行登记策略。 [预注册](PROTOCOL.json) · [旧版 Q1 与 west/96 复核](decision/historical_regression.json) · [零权重与交换先验回归](runner/local_parity/receipt.json)

| 策略 | 局部表面积先验 | 附加权重 |
|---|---|---|
| G | 中性 | 零 |
| G_route | 中性 | 路线复现拟合 |
| G_perf | 中性 | 地图质量拟合 |
| S | 正确可见类别 | 零 |
| X | 对调开放／封闭类别先验 | 零 |
| S_perf | 正确可见类别 | 与 G_perf 相同 |

同一布局中完全相同的策略输入可复用一次完整执行，以 `alias.json` 绑定实际来源。真实执行后的轨迹或终点相等照实报告。建筑是独立比较单位。运行中的中止、早停、失败与未取得的等步数检查点留在各自的记录里。

## 复跑与复算

以下命令适用于中控原目录结构、同一仿真环境和已封存场景。跨机器复跑需按 `dev_freeze.json` 和 `holdout_freeze.json` 的绝对路径还原依赖，其中包括上一轮的 `same_goal_route_choice_20261121`、`same_goal_cold_holdout_20261121` 和原生仿真扩展。固定配置中断时使用对应 `run` 命令的 `--resume`。首次启动的 20 个进程在仿真前因哈希输入的路径类型错误退出，故障记录见 `execution/preflight_attempt_001`。修复为哈希调用前转换路径类型。

```sh
NSO_PY=/home/ouc/nso_robot/envs/science312/bin/python
"$NSO_PY" study.py seal development
"$NSO_PY" study.py run development
"$NSO_PY" measurement/evaluate.py --split development --arms S G G_u000_c005 G_u000_c020 G_u005_c000 G_u005_c005 G_u005_c020 G_u025_c000 G_u025_c005 G_u025_c020 --only-terminal --require-finished
"$NSO_PY" decision/fit_routes.py fit
"$NSO_PY" study.py fit
"$NSO_PY" study.py seal holdout
"$NSO_PY" study.py run holdout
"$NSO_PY" measurement/evaluate.py --split holdout --arms G G_route G_perf S X S_perf --require-finished
"$NSO_PY" decision/fit_routes.py evaluate --frozen-weights fit_weights.json
"$NSO_PY" report.py
```

地图用冻结真值可达栅格和可评表面独立复算。逐动作、位姿、RGB／深度／激光摘要和终点网格回执分属各臂。规划耗时是计费步骤的实测规划回调总耗时，任务耗时另列为仿真墙钟时间。等步数图使用登记的 0、48、96、144、192 步，未到达的时刻保留为空。归档中的 `EVIDENCE_INDEX.json` 指向首帧、评价真值、运行日志和逐轮回执。原生实测网格留在中控，索引记录原路径和 SHA-256，用于核对独立复算的输入。

[逐状态路线选择](decision/heldout_states.csv) · [逐栋终点与耗时](measurement/results/holdout/episodes.csv) · [实际首次分歧](decision/heldout_actual_first_differences.csv) · [逐栋配对差值](results/paired_buildings.csv) · [等步数图](results/equal_budget_paired_quality.pdf) · [论文结果段](results/manuscript_zh.md)
