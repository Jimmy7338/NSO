# 同状态、同目标的单次选路干预

两个分叉来自已经出现策略分歧的仿真病例。链式 17 在累计第 105 步测试开放货架，环式 20 从首帧测试封闭柜体。每个分叉先复原原 S 的公共历史，并在两臂的融合网格、观测、候选池和返航点一致后，指定一次类别路线或同代价几何路线。其余选路使用相同的 G_perf 权重，继续采集和融合各自的 RGB-D 与激光观测。在链式 17，几何对照是指定的等代价路线，在完整 G_perf 池中排第 12。环式 20 的几何对照排第 1。

[冻结方案](PROTOCOL.json) · [运行和评价文件封存](RUN_FREEZE.json) · [候选池和历史状态](source/frozen_anchors.json) · [设施身份绑定](source/evaluator_bindings.json) · [面积评价规则](evaluation/MEASUREMENT_PROTOCOL.json)

## 复跑

复跑使用中控上的 Python 3.12 科学环境及源研究的目录结构。`RUN_FREEZE.json` 的绝对路径和 SHA-256 确定了 native actor、原 S 日志、两案首帧和评价源。源研究在提交 `8071a50fc7d04980d2eb390f633be7cae950706e` 的 `experiments/offline_20261122_weighted_geometric_control`。已封存的运行目录包含原始网格。GitHub 中的网格索引给出各文件的完整 SHA-256。

在不含旧运行输出的复跑目录中执行下列命令。已有 `RUN_FREEZE.json` 时直接执行 `study.py run`。两臂需要同时启动，分叉前检查对方的实际复原结果。`replay-missing` 只补采有真实观测但缺快照的共同步数。它从同一执行历史恢复，保留原来的运行目录和终点。

```sh
NSO_PY=/home/ouc/nso_robot/envs/science312/bin/python
"$NSO_PY" evaluation/evaluate.py --check-inputs
"$NSO_PY" study.py seal
"$NSO_PY" study.py run
"$NSO_PY" study.py replay-missing
"$NSO_PY" evaluation/evaluate.py --run-root runner/runs --replay-root runner/replays
"$NSO_PY" figures.py
"$NSO_PY" summarize.py
```

## 结果

- [两案主结果](evaluation/results/paired_primary.csv) 与 [逐臂测量时刻和设施面积](evaluation/results/snapshots.csv)
- [实际路线叠图](results/executed_routes.pdf) 与 [共同及独有的新增表面](results/measured_facility_surfaces.pdf)
- [论文结果段](results/manuscript_zh.md) 与 [运行及网格索引](EVIDENCE_INDEX.json)

面积通过冻结可评参考采样和融合三角网格测量，匹配容差为 5 cm。共同与独有面积基于同一个已恢复表面掩码，从原始分叉时刻扣除已有匹配点。路线事件与完整闭环终止分别计量，192 步预算包含公共前缀和返航。
