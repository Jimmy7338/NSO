# 留出楼层和首帧复跑

四个楼层的拓扑、门位、设施摆位和种子先在 `scene.py` 固定。`paired_cycle_18` 是本研究开发／回归楼层。本目录链接其既有首帧和评价参考，`actor_freeze_development.json` 明确登记它在本研究中的角色。

| 楼层 | 布局 | 区域数／门数 | 可达面积（m²） | 最远区域单独往返所需动作 | 固定内部点全区访问并返航的见证（动作） |
|---|---|---:|---:|---:|---:|
| `floor26_ordinary` | 错位四房 | 4／3 | 50.56 | 88 | 144 |
| `floor26_branch` | 五岔中央枢纽 | 6／5 | 60.56 | 76 | 170 |
| `floor26_corridor` | 双侧长走廊与一条侧向连接 | 7／7 | 61.88 | 158 | 238 |
| `floor26_loop` | 六区域环路 | 6／6 | 68.60 | 120 | 160 |

取样分辨率为 0.20 m，保留半径 0.20 m 的圆形扫掠足迹、相机与平面激光的 8 m 量程。各真门在源网格上都有实际可通行的扫掠边，且与注册的区域邻接一一对应。几何往返动作数包含转向及回到起始朝向。238 步的走廊见证使用提前指定的内部点，所有区域分别可在 192 步内往返。楼层探索的成功条件仍由策略实际采集、建图和返航结果决定。

首帧准备和几何预检不调用策略。模拟器以合成建筑和未缩放的原始设施网格渲染 RGB-D、平面激光和当前可见实例。在线方法接收传感包、估计地图及已执行状态，真实房间邻接、真值栅格、源网格和参考采样用于离线评价。演员输入封存和真值封存分列。新四栋楼层的占据和可达掩码与此前归档及彼此比较，旋转或镜像后均没有完全重复。

在本目录恢复尚未生成的输出时执行下列命令。已经生成的首帧、真值和预检由哈希复核。

```sh
NSO_PY=/home/ouc/nso_robot/envs/science312/bin/python
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
"$NSO_PY" prepare_scenes.py paired_cycle_18
for layout in floor26_ordinary floor26_branch floor26_corridor floor26_loop; do
  "$NSO_PY" prepare_scenes.py "$layout"
done
"$NSO_PY" geometry_audit.py
"$NSO_PY" seal_scenes.py development
"$NSO_PY" seal_scenes.py holdout
"$NSO_PY" seal_scenes.py development --verify
"$NSO_PY" seal_scenes.py holdout --verify
```

逐门的名义净宽、扫掠状态、每区可达栅格数和初始未知栅格数在 `formal_geometry_audit.json`。参考面的采样、可见筛选、面积权重和距离容差记录在 `reference_protocol.json` 及各 `evaluator_references/<layout>/RECEIPT.json`。已登记的 192 步仍用于冻结矩阵，几何预检没有读取新的轨迹或地图结果。
