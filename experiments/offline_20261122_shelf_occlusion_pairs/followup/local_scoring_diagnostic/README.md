# 固定候选池的局部评分诊断

[六个场景、十二个摆位的结果](RESULTS.md) · [逐摆位评分与动作](formal/placements.csv) · [封存汇总](formal/summary.json)

该脚本读取正式封存的候选、轨迹和测量结果。逐摆位核验候选文件的原始哈希和规范化候选池哈希，复算四臂的同池评分，记录 D+ 的非零支持次数与最大隐面估计、D 首选相对于最接近备选的差值、加入隐面后该首选的差值、S 与 G 互相替换路线时两种评分的差值，以及真实动作首次分歧。`actor_*` 字段记录决策模型的估计或评分。`measured_*` 字段来自实际执行与独立网格评价。

```sh
NSO_PY=/home/ouc/nso_robot/envs/science312/bin/python
"$NSO_PY" followup/local_scoring_diagnostic/diagnose.py \
  --results results/formal \
  --out followup/local_scoring_diagnostic/formal
```

候选评分的排序变化与实际路线分歧分别计数。若 D+ 提供非零隐面估计但仍选择 D 的首选，去重后的同一轨迹只提供相同执行结果。未执行的备选路线没有独立实测收益。
