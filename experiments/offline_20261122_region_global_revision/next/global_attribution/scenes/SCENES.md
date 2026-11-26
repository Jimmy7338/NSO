# 全局决策归因的两栋新楼层

两栋楼层均使用原生 RGB-D 与平面激光、0.20 m 网格、半径 0.20 m 的圆形足迹和已用设施源网格。场景几何、起点、设施摆位和各自随机种子在策略运行前登记。正式上限为 192 个计费动作，含转向、返航和恢复初始朝向。

| 楼层 | 拓扑 | 房区与门 | 正确自由区分母（m²） | 首帧未知可达格 | 最远单区去返（步） | 全区去返几何见证（步） |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| attr26_corridor_crosspass | 水平贯通长走廊，南北错位房间，北侧横向连通 | 6／6 | 60.64 | 832 | 102 | 152 |
| attr26_loop_bypass | 外围闭环、中部互通、南侧双通道 | 7／9 | 63.76 | 986 | 110 | 136 |

走廊为 11.6 m × 8.0 m，层高和传感配置沿用原生仿真。初始朝向为东，冷启动激光可区分三道近门。沿走廊中线分别检验五道侧门，均报告实际穿门激光与双门垛回波。环路为 11.4 m × 8.4 m，外圈和南侧旁通分别允许先探索不同区域。扫掠验证和预算见证覆盖每一道门，两栋最小门宽均为 1.4 m。

预算见证要求进入每个房区边界至少 0.4 m 后回起点，只验证几何可行性。预检根据室内占据真值搜索证书。真实房区划分、来源网格、路径证书和固定表面样点供渲染或独立评价，在线输入仍为原生传感包、融合图和动作反馈。占据和可达二值图在旋转、镜像下与 43 份既有归档及两场景彼此均不同。两个种子与 35 份历史冷启动记录均不同。预检时两个场景的策略选择调用数均为 0。

复跑预检采用新目录，已有场景可直接执行最后一条命令校验。

```sh
cd /home/ouc/nso_robot/analyses/region_global_revision_20261122/next/global_attribution/scenes
NSO_PY=/home/ouc/nso_robot/envs/science312/bin/python
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1
for layout in attr26_corridor_crosspass attr26_loop_bypass; do
  "$NSO_PY" prepare_scenes.py "$layout"
done
"$NSO_PY" geometry_audit.py
"$NSO_PY" seal_scenes.py
"$NSO_PY" seal_scenes.py --verify
```

[可见输入与模拟器缓存封存](actor_freeze_holdout.json) · [独立评价封存](evaluator_freeze_holdout.json) · [几何与预算核验](formal_geometry_audit.json)
