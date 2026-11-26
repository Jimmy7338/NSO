# A 阶段归档复用核验

原四楼每楼的 D0G0 和 D1G1 对应 `region_S` 与 `combined_S`。复用前检查原实验固定版本、任务下发与完成回执、动作预算、冷启动位姿、首帧摘要、局部 S 策略以及原始日志哈希。以 2026 年 11 月 24 日已核对的归档提交 `74ecf883503d5a117d1e566417aff0213f76472a` 为查阅入口，实际运行源码基点为 `f482722f6cbfc3135358080b08f27274f54cb535`。

从每一步已观测栅格、实际位姿和执行事件恢复区域状态。重算区域图、入口候选、访问和拒绝记录、原区域分数、完整修订重排以及最终目标。重排共用当时已通过底层可行性校验的原生候选和其去程观测组件。独立复加了雷达累计覆盖、相机累计覆盖与已测表面质量。此次复用检查走遍原轨迹的全部 145 次全局选择，重算 527 条区域打分并核对 811 条原生收益记录。它校验冻结实现的区域层重放和原生收益算术。底层传感渲染和原生收益预测使用已归档的实际记录。

11 月 22 日的固定状态对照有六条分支，分别在长走廊、环路和普通楼层的相同前缀实施一次独立的 D 或 G 更改。三个分叉点的共同前缀、候选和旧选点均通过当时的逐项核验。那些实验验证被控状态下的选点变化，采用不同楼层和单次干预，列为历史机制证据。 [11 月 22 日固定状态结果](https://github.com/Jimmy7338/NSO/blob/74ecf883503d5a117d1e566417aff0213f76472a/experiments/offline_20261122_region_global_revision/next/global_attribution/results/RESULTS.md)

从本目录复核并重新生成表图。

```sh
nso_python=/home/ouc/nso_robot/envs/science312/bin/python
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1
"$nso_python" replay_rankings.py | cmp - reuse_receipt.json
"$nso_python" summarize_factor.py
```

后一个命令读取 A 阶段共享评估器的冻结输出，核对文件摘要及 96 步条件效应与动作数删失，再生成摘要表和交互图。
