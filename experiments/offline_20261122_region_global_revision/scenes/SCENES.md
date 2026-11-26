# 独立配对布局

两栋楼层用原有 RGB-D、平面激光、0.20 m 网格、半径 0.20 m 的圆形扫掠模型及原始设施源网格。新布置使用独立墙体、门位、设施位置和种子。单轮上限为 192 个计费动作，含转向、返航和恢复起始朝向。

| 布局 | 房区／门口 | 可达面积（m²） | 首帧未知可达栅格 | 最远单区访问并返航（动作） | 全区进入再返航见证（动作） |
|---|---:|---:|---:|---:|---:|
| `rev26_corridor_receiving` | 6／5 | 67.68 | 1,167 | 110 | 172 |
| `rev26_loop_courtyard` | 7／7 | 64.56 | 860 | 108 | 160 |

走廊由一条长 12.4 m 的直通区域和两侧五间错位房室组成。首帧原生激光对两道近门同时给出两侧门垛反射和穿过开口的射线。其余近观测位置在预先规定的走廊中线，五道门均通过同一观测核验。环路为六区域外围循环加一间内侧展廊，展廊只有一条入口。

见证只用于几何验收。全区见证以未知场景背后的图为输入，要求机器人中心到每个区域边界至少 0.4 m，且最后返回起点原朝向。固定房间中心全部访问的更严格见证在走廊和环路分别为 236 和 192 步。预检没有移动模拟机器人或调用探索策略。新楼层的占据和可达二值图，与 39 份历史归档及彼此相比，旋转或镜像后均没有重复。两个布局的种子与 31 份历史首帧记录核对过。

`scene.py` 提供原有运行器接口 `SPECS`、`LAYOUTS`、`make_world` 和 `build_scene`。在线策略仍接收传感包、融合地图和动作反馈。碰撞缓存由模拟环境使用。源网格、真实邻接、路径见证和标准表面样本仅供几何预检和独立评价。可见首帧、模拟端碰撞缓存与评价真值分别登记在下列文件。

- `actor_freeze_holdout.json` 保存首帧、种子和渲染器源文件哈希，模拟端碰撞缓存采用独立字段。
- `evaluator_freeze_holdout.json` 保存独立真值、可见参考的采样结果和几何预检哈希。
- `formal_geometry_audit.json` 保存逐门扫掠状态、原生激光可见性证明、未知栅格和几何路径见证。

在新的无运行结果目录复建场景时，沿用同一台中控与下面的命令。已有文件使用最后一条命令核验。

```sh
NSO_PY=/home/ouc/nso_robot/envs/science312/bin/python
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1
for layout in rev26_corridor_receiving rev26_loop_courtyard; do
  "$NSO_PY" prepare_scenes.py "$layout"
done
"$NSO_PY" geometry_audit.py
"$NSO_PY" seal_scenes.py
"$NSO_PY" seal_scenes.py --verify
```
