# TARE 四楼探索对照

本实验延用已登记的四个虚拟楼层和每轮 192 个计费动作，与 [修订 NSO](../offline_20261123_region_independent_floors/README.md) 配对。各楼层共用起始位姿、初始观测、仿真渲染器、准确位姿、占据建图、TSDF、动作保护和返航规则。各策略从自身实际执行的轨迹继续采集观测。完整场景网格只参与仿真与独立评价。

## 共享传感器与 TARE 适配

- RGB-D 输入为 120×90、水平视场 90°、轴向深度上限 8 m、相机高 1.20 m。平面激光为 180 束、量程 8 m、安装高 0.25 m。每个计费动作和第 0 帧各发一份实际观测。有效深度和量程内激光命中用各自的真实外参转换成地图点云。量程端点不伪装成障碍，不补造侧后方三维点，也不提供仿真真值边界或地形话题。
- TARE 使用官方 `44500592` ARM64 构建和室内配置，仅关闭供外部控制器使用的航点外推。内部 360°×24° 覆盖模型、7.5 m 规划范围、视点碰撞余量和随机采样均沿用原实现。原生点云更新沿用单一参考原点，适配里程以实际水平位置和航向、相机高度为参考。归档范围为**共享 RGB-D＋平面激光输入下的 TARE 规划器**。 [官方实现](https://github.com/caochao39/tare_planner) · [室内配置](https://github.com/caochao39/tare_planner/blob/44500592b86138257273e0cab264e6a847ccefc7/src/tare_planner/config/indoor.yaml)
- TARE 每五个有效扫描触发一次关键位姿。首帧记第 0 步，之后最多四次预算内右转完成首组扫描。每个新确认航点刷新共同执行器的路径。航点可达时走到所在栅格，航点超出可达观测区时仅向其推进至连续已观察前缀。无可派发目标时最多补一次五步原生更新机会，然后按共同规则返航。原生结束信号直接请求共同返航。动作和原生时间单列，软件原有壁钟计时器和随机采样保留。

## 复跑及评价

先按 [已验证的 ARM64 编译过程](../offline_20261124_external_exploration_baselines/work/tare/PROBE.md) 恢复官方源和 native 二进制，再确认 TARE 与四楼冻结文件的哈希。下列命令从本机本实验目录运行。其他主机需要记录二进制重建的哈希，并复用相同已冻结源码和配置。每轮保留实际取得的原生采样结果。

```sh
nso_python=/home/ouc/nso_robot/envs/science312/bin/python
"$nso_python" run_matrix.py --workers 2 --output execution/reproduction
"$nso_python" ../external_exploration_baselines_20261124/evaluate_explore.py \
  --protocol TARE_PROTOCOL.json --arm TARE_sensor_transfer --receipt tare_receipt.json \
  --runs execution/reproduction --output results/reproduction
"$nso_python" verify_tare_run.py --runs execution/reproduction --results results/reproduction --replay-packets
```

联合指标 J 为正确可达自由区覆盖率 C 与全场景表面 F1@5 cm 的乘积。沿用首次达到 95% 覆盖的计费动作数、第 96 步共同检查点以及第 48、144、192 步和真实终点。未达到预算检查点时保留缺测。路径索引和逐轮回执指向中控的原生网格，评价文件同时记录实际覆盖、精确率、召回率、F1、终止和返航。

四轮的 772 个入站包均按已执行动作独立重渲染，核对点云输入所用深度、外参、激光量测和命中计数。桥接等待包含在墙钟和传感更新阶段。逐动作决策时间只计共享执行器生成动作的时间，适用于核查本次运行流程。

[主表与终点对照](RESULTS.md) · [逐楼检查点](results/formal/pairs.csv) · [逐动作真实覆盖](results/formal/coverage.csv) · [逐帧与执行核验](results/formal/transport_audit.json) · [传感包独立复算](results/formal/packet_replay.json)
