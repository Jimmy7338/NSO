# 西侧遮挡场景的独立闭环验证

单次冷启动仿真使用版本 `west_directional_endpoint_G_sim_v1`。双方的起始 RGB-D、激光和位姿数据一致。该版本采用固定回放已核验的障碍端点方向落格，并沿用原 G 评分、执行保护和含返航的 192 动作预算。 [落格候选和两条固定回放](../west_projection_shadow/RESULTS.md) · [闭环源码封存](attempt_001/SOURCE_FREEZE.json)

| 回合 | 实付动作 | 行驶距离 | 返航 | 仿真碰撞 |
| --- | ---: | ---: | --- | ---: |
| 原版 G | 61 | 10.6 m | 起点在观测图中断连 | 0 |
| 方向落格 G | 192 | 32.8 m | 返回起始位置与朝向 `[39, 5, 0]` | 0 |

新轨迹在第 25 个动作首次与原版分歧。原版在第 24 个动作后因下一步足迹在观测图中未确认安全而重规划，同一首选目标在新回合第 33 个动作后触发同类保护。新回合随后继续探索并完成返航。此次单场景、单策略执行支持仿真中的技术恢复。真实传感器条件与不同场景的适用性仍待验证。 [实际动作与终点回执](attempt_001/receipt.json) · [原生执行清单](attempt_001/native/manifest.json)

独立复跑需要原工作目录和新的输出目录。

```sh
NSO_PY=/home/ouc/nso_robot/envs/science312/bin/python
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1
"$NSO_PY" -X faulthandler closure.py --out REPRODUCTION_OUTPUT
```
