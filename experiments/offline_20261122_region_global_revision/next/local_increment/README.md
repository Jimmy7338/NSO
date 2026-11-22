# 同目标的受支持隐面增量

两个预登记仿真场景分别提供无遮挡货架和带侧向遮挡板的货架。场景使用不同房间配置、摆位与种子，保留两个既有设施的源模型。它们是各自独立的机制案例，用于检验受支持隐面信号。两个布局同时改变了其他几何条件。

直接巡检 D 使用当前已测实例的表面欠观察量。D+ 复用 D 和原生候选池，只额外计入同实例的类别增量。隐藏样本须在 0.25 m 内找到确实测得的同实例深度表面，距离大于 0.10 m，所属实例有至少四个测量体素，已见类别投票一致率不低于 75%。类别条件后验与中性后验采用同一粒子基。隐面权重取两者对该样本的正向面积差。原有测量网格和候选结构的遮挡检验之后，两个固定中间射线点还须得到历史 RGB-D 的自由空间证据。预测不写入重建。

观测决策状态由原生策略实现。各布局在符合规则的同一首帧产生候选。在打开分支实测结果之前，把 D、D+、S 各自的首选与原生路线去重并封存。完整原生候选和评分留在 `ROUTE_FREEZE.json`。每条封存路线以原生足迹、同一目标姿态、预留同一返航路径执行一次。共同评价时钟取封存子集最短计划去程步数，实际提前停止按缺测。另保留各自实际到达或取消时刻。停止发生在去程结果出炉时，所以这里只能说明去程机制。

输入和后处理分开封存。在线评分只读模拟传感包、实例关联、历史深度和融合网格，源模型负责独立评价。类别标签是仿真提供的当前可见实例标签。候选门槛用于本次机制对照。比较没有更新全局策略、启动实车或检查完整闭环覆盖。

初次六个见证进程在发出动作之前停止。D 审计里的类别字典在 JSON 保存时将整数键转成字符串，导致活对象与重读对象的相等断言失败。修复采用同一 JSON 表示比较，并重新生成两个候选表，逐字段验证科学内容与原封存完全相同。旧运行、源码和封存均保留在 `technical_preflight_001/`。两次冻结间的校验见 `technical_recovery.json`。

## 复跑

在保留相同依赖路径的新目录生成输出。

```sh
PY=/home/ouc/nso_robot/envs/science312/bin/python
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1
for layout in increment_open increment_screen; do
  "$PY" scenes/prepare.py "$layout"
done
"$PY" seal.py inputs
for layout in increment_open increment_screen; do
  "$PY" run.py "$layout" probe "probes/$layout"
done
"$PY" seal.py routes
"$PY" study.py
"$PY" measure.py
```

- `results/summary.json` 和 `results/routes.csv` 提供逐路线结果、已恢复与遗失的表面面积和选择后悔值。
- `results/*_masks.npz` 保存同一参考采样顺序和恢复掩码。
- `runs/…/launch.json`、`exit.json` 与 `run/manifest.json` 绑定执行、姿态和逐帧数据。面积使用真实融合的 TSDF 原生网格与独立参考匹配。
