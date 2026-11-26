# 外部探索规划对照

复用 [2026-11-23 四楼实验](../region_independent_floors_20261123/README.md) 的四个场景、首帧、准确仿真位姿、平面激光、RGB-D、TSDF、动作计费与返航核验。本轮增加外部规划器，比较其轨迹取得的实测地图。四栋楼均在上轮运行前冻结，本轮为同场地追加对照。

## Explore Lite

使用 [官方 Noetic 源码](https://github.com/hrnr/m-explore/tree/712bdd41027b645c9c876a4c0071478f090825ea/explore) 的 `FrontierSearch`。保留原始未知格聚类、距离与簇长排序及 `explore_costmap.launch` 的参数。输入为共同执行器的实测占据栅格与足迹膨胀。共同执行器选取紧邻该簇、离其质心最近的可达已知格，采用继承的四朝向未知可见性规则，跳过当前同朝向的零动作位姿。若一个簇无可派发的位姿，沿官方排序继续下一个。执行与账本来自现有前沿执行器。仿真实验只比较官方选点核心与共用执行的组合。计时包含接口开销。

## 复跑

在本目录预先下载并在 `work/explore/source` 校验 `EXPLORE_PROTOCOL.json` 中的上游提交，用 ROS Noetic 的 `costmap_2d` 编译。可执行文件写在本实验目录。

```sh
source /opt/ros/noetic/setup.bash
export PKG_CONFIG_PATH=/opt/ros/noetic/lib/pkgconfig
git -C work/explore/source rev-parse HEAD
g++ -std=c++14 -O2 -Wall -Wextra -Iwork/explore/source/explore/include \
    $(pkg-config --cflags costmap_2d) explore_lite_search.cpp \
    work/explore/source/explore/src/frontier_search.cpp \
    -o work/explore/frontier_search $(pkg-config --libs costmap_2d)
NSO_PY=/home/ouc/nso_robot/envs/science312/bin/python
"$NSO_PY" smoke_explore.py
"$NSO_PY" run_matrix.py
```

保留已有 `execution/explore` 后再复跑。编译器或链接库版本变化可能产生新的可执行文件哈希，跨机器复跑应核对相同源文件、上游参数和场景后记录新的二进制冻结信息。

两次前置运行分别因 Python 模块解析顺序和审计时钟字段名失败。均未执行任何计费动作，日志保存在 `execution/explore_pre_actor_attempt_001` 与 `execution/explore_pre_action_attempt_002`。这两次修复只涉及接口和记录。

## ARiADNE

[官方 ROS 规划器](https://github.com/marmotlab/ARiADNE-ROS-Planner/tree/773ebcf60334b2a16a865f1207938f53fee92031) 的公开检查点、规划图和贪心航点推理在中控 CPU 上运行。节点间距 2 m，栅格分辨率和传感范围分别按原实验设为 0.20 m 和 8 m。策略输入仅为已观测栅格与共同足迹。航点由共同执行器沿最短可达栅格路径执行，采用末段行驶朝向。无效航点按原生停止方向进入共同返航逻辑，记录原因。比较对象是官方策略与共享仿真执行的组合。 [冻结协议](work/ariadne/PROTOCOL.json) · [复跑](work/ariadne/README.md)

## TARE 接口验证

[TARE 官方源码](https://github.com/caochao39/tare_planner) 在中控 ARM64 上通过编译，在五组归档 RGB-D 与平面激光包输入后输出了新航点。通信使用独立本机 ROS master。原版每五帧更新一次关键位姿，按 360°×24° 三维激光覆盖建模，四楼目前采用 90° RGB-D 与平面激光。该记录只验证输入、原生规划与输出链路。 [构建记录](work/tare/PROBE.md) · [验证回执](work/tare/smoke_receipt.json)

## 评价

主要报告首次达到 95% 正确可达自由区覆盖的动作数、同一动作时钟下的覆盖率 C、3D 表面 P／R／F1@5 cm 及联合指标 J=C×F1。第 96 步为共同检查点，另保存第 48、144、192 步和各自终点。预算内未达保留终点覆盖，提前停止造成的缺测留空。保留实际返航、碰撞、停止与耗时。

Explore Lite 的四轮结果可从 [逐楼 A95、耗时与终点](results/explore/episodes.csv) 和 [逐楼逐检查点配对](results/explore/pairs.csv) 复核。相对该基线，修订 NSO 的第 96 步 J 在直线和折线走廊提高，在错位环道和双环下降，四楼平均差为 +2.08 个百分点。错位环道的 Explore Lite 在第 142 步提前返航，其第 144、192 步保留缺测。

ARiADNE 的四轮结果见 [逐楼 A95、耗时与终点](results/ariadne/episodes.csv) 和 [逐楼逐检查点配对](results/ariadne/pairs.csv)。NSO 第 96 步 J 在两楼提高、两楼下降，平均差为 +5.42 个百分点。双方均达 95% 的三楼中，NSO 提前 9、63 和 52 步。ARiADNE 的直线和折线走廊分别在 152 和 132 步返航，这两楼的终点 J 来自不同动作数。

| 楼层 | 95% 覆盖步数 NSO／Explore Lite／ARiADNE | 第 96 步 J（%）NSO／Explore Lite／ARiADNE |
| --- | --- | --- |
| 直线主走廊 | 55／101／64 | 70.33／59.28／72.31 |
| 折线走廊 | 三法均未达，末值 91.79%／93.07%／88.45% | 67.88／59.10／70.27 |
| 错位环道 | 91／76／154 | 81.86／84.06／64.44 |
| 双环 | 127／121／179 | 71.26／80.59／62.63 |

两种外部选点策略各完成并返航 4/4，碰撞 0 次。比较以楼层为单位。高覆盖速度与早期重建联合质量的排序均随策略和布局变化。共享准确位姿和 TSDF 将比较范围限定为探索决策对所采集观测的影响。

使用 `evaluate_explore.py` 复算测量。默认复算 Explore Lite，以下命令复算 ARiADNE。选项 `--output` 指定空路径。配对 CSV 的 `explore_*` 字段统一表示本次传入的外部策略。

```sh
"$NSO_PY" evaluate_explore.py
"$NSO_PY" evaluate_explore.py --arm ARiADNE_policy --protocol work/ariadne/PROTOCOL.json \
    --receipt ariadne_receipt.json --runs work/ariadne/execution/attempt_001 \
    --output results/ariadne
```

归档的 `native_evidence.zip` 保留实际动作、逐动作占据历史、规划输出、执行回执和检查点哈希。原生融合网格位于检查点表列出的中控路径。
