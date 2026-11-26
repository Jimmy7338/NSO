# 四个留出楼层

四个楼层用新房间矩形和门位置构成。设施采用原有 ReplicaCAD 资产和本轮登记的摆位。各楼层的种子、房间坐标、门、冷启动位姿和接纳条件在策略运行前登记。旋转与镜像下的占据/可达图与既有的 59 份网格归档及四楼层彼此均不同。四个种子与 51 份带种子的历史冷启动记录均不同。 [布局登记](LAYOUT_REGISTRATION.json) · [几何与采样审核](formal_geometry_audit.json)

| 楼层 | 房间／门／独立环 | 总面积（m²） | 可达自由区（m²） | 起始朝向 | 首帧未知可达格 | 全房区去返见证（动作） |
| --- | ---: | ---: | ---: | --- | ---: | ---: |
| 纵向长廊 `ind26_vertical_spine` | 7／6／0 | 113.68 | 76.28 | 北 | 1284 | 174 |
| 折弯走廊 `ind26_bent_corridor` | 7／6／0 | 98.28 | 62.36 | 东 | 881 | 192 |
| 错位外环带支房 `ind26_offset_ring` | 7／7／1 | 97.20 | 65.88 | 北 | 953 | 140 |
| 双环连廊 `ind26_double_loop` | 7／8／2 | 102.08 | 68.64 | 西 | 1040 | 146 |

原生圆形足迹半径为 0.20 m，网格 0.20 m。所有登记的门均具有可双向通过的扫掠边，最佳穿门中心线超出足迹半径的余量至少 0.40 m。去返见证从门网格和原生碰撞几何中搜索，进入各房区至少 0.40 m，计入所有转向并回到原朝向。它衡量几何上的房间可达性。覆盖指标由实际感知与融合轨迹另行测量。

所有楼层的原生参数与两栋已知归因案例逐字段相同。RGB-D 为 120×90、视场 90°、相机高 1.20 m、深度上限 8 m。平面激光高 0.25 m，每帧 180 束。采用仿真准确位姿、零额外深度噪声、0.06 m 体素和 0.12 m TSDF 截断。上表四栋楼首帧有效深度像素分别为 8261、8709、8682、9227，满帧均为 10800 像素。每栋楼的近中远深度与激光分位数、传感器内外参、连通图和各门的扫掠余量见几何审核。参考三角网格、固定可见源样点及去返见证位于独立评价文件。运行策略接收原生传感包、已获取的地图和动作反馈，碰撞缓存仅由模拟器使用。

折弯走廊初次几何登记的起点为 `(1.3, 1.3) m`，全房区最短去返需要 210 步，超过 192 步预算。该候选版本按注册条件排除。正式冻结前以相同房区、门、设施与种子重新登记起点 `(4.3, 4.1) m`，朝东。第二次预检通过 192 步的几何去返见证。两版占据、可达、足迹和扫掠数组以及源网格完全相同。此调整前后均无策略结果。 [旧版排除记录](preregistration_v1/formal_geometry_audit.json) · [起点修正凭据](preregistration_v1/ANCHOR_CORRECTION.json)

## 复跑几何预检

在新目录保留同样的源码路径层级和原始 NSO 仿真依赖。现有冻结目录可直接执行最后一条命令。

```sh
cd /home/ouc/nso_robot/analyses/region_independent_floors_20261123/scenes
NSO_PY=/home/ouc/nso_robot/envs/science312/bin/python
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1
"$NSO_PY" register.py
for layout in ind26_vertical_spine ind26_bent_corridor ind26_offset_ring ind26_double_loop; do
  "$NSO_PY" prepare_scenes.py "$layout"
done
"$NSO_PY" geometry_audit.py
"$NSO_PY" seal_scenes.py
"$NSO_PY" seal_scenes.py --verify
```

[可见输入与模拟器缓存](actor_freeze_holdout.json) · [离线评价输入](evaluator_freeze_holdout.json)
