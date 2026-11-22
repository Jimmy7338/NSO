# 链式 17 的 G_perf 首选路线对照

本研究从已出现选路分歧的链式 17 中追加一例事后机制比较。在原 S 的第 105 个计费动作后复原融合地图和状态。新闭环让 G_perf 按完整候选池正常选择首选路线。已封存的两臂分别执行类别路线和名义等代价的几何竞争路线，本轮直接复用其回执与深度融合网格。原冷启动 G_perf 在第 44 步已与 S 分歧。

| 路线 | 同一状态下 G_perf 名次 | 名义去程（步） | 数据 |
|---|---:|---:|---|
| 类别路线 `via_32_37_0` | 13/34 | 67 | 已封存 |
| 等代价几何对照 `via_33_29_3` | 12/34 | 67 | 已封存 |
| 几何首选 `native` | 1/34 | 63 | 本轮新增 |

局部几何权重分别为未知可见面积 0.25、额外动作代价 0.20。保留原目标、碰撞与返航保护，计费上限含公共前缀和返航，总计 192 步。开放货架 `open31` 的可评表面、面积权重和 5 cm 匹配容差来自原封存评价。每对共同计费时刻取实际到达或初始路线中止步数的较大值。

[试验协议](PROTOCOL.json) · [封存哈希](RUN_FREEZE.json) · [量测口径](evaluation/MEASUREMENT_PROTOCOL.json) · [上一轮源数据与回执](https://github.com/Jimmy7338/NSO/tree/874bff785debcad6982059530cdeda892a26e7ba/experiments/offline_20261122_route_intervention)

## 复算与复跑

Python 3.12 环境安装在中控。`RUN_FREEZE.json` 固定复跑所需的绝对路径和哈希。已有 `runner/runs/paired_chain_17/native/manifest.json` 时可直接核对回执与复算，复跑命令应在完整还原的空白输出目录执行。`capture-clocks` 仅在已执行的共同步数没有原生快照时做确定性重放。所有报告均从实际网格、观测和动作回执生成。

```sh
NSO_PY=/home/ouc/nso_robot/envs/science312/bin/python
"$NSO_PY" evaluation/evaluate.py --check-inputs
"$NSO_PY" study.py seal
"$NSO_PY" study.py run
"$NSO_PY" study.py capture-clocks
"$NSO_PY" evaluation/evaluate.py --freeze RUN_FREEZE.json --run runner/runs/paired_chain_17/native --output evaluation/results
"$NSO_PY" summarize.py
```

## 交付

- [三臂结果表](evaluation/results/three_arms.csv) · [评价明细](evaluation/results/measured.json) · [表面掩码](evaluation/results/surface_masks.npz)
- [运行回执与原始网格索引](EVIDENCE_INDEX.json) · [论文结果段](results/manuscript_zh.md)
