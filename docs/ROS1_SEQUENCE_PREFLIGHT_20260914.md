# ROS 1 录制数据的离线检查与坐标约束

> 2026-09-14 补充：新增来源元数据交叉校验，13 项正反例及 5 条模拟记录的独立 CLI 检查通过。详见 [新报告](ROS1_SOURCE_METADATA_CROSSCHECK_20260914.md)。旧数组检查器仍只检查数组；内部元数据一致不等于实际标定或实车就绪。下文未完成描述保留为该阶段历史状态。

已新增 `scripts/audit_ros1_rgbd_sequence.py`，读取现有录制器输出的 `frames/*.npz` 和 `scans/*.npz`，不连接 ROS、不发布控制指令。已有 61 帧虚拟序列通过检查；5 项正反例测试覆盖正常数据、时间错位、缺帧、非法雷达变换、重复时间戳和非有限时间戳。该结果验证存储数据的部分契约，尚不能证明实车接入成功。

检查入口：

```bash
.venv-3d/bin/python scripts/audit_ros1_rgbd_sequence.py \
  --source /absolute/path/to/recorded_session \
  --output /absolute/path/to/new_audit.json \
  --max-skew-s 0.05 --max-depth-m 4
```

距离和允许时间偏差须与本次录制配置一致，不能为通过检查事后放宽。输出记录输入哈希、逐帧失败原因及未验证项；发现问题返回非零状态。无三维参考真值时不生成“真实重建精度”结论。

## 已检查与未覆盖

工具检查深度／RGB／语义数组对齐和标定形式、深度单位范围、正深度支持、帧时间递增、深度与雷达的时间差、雷达范围和刚体变换，以及丢失或多出的雷达文件。输出固定区分 `status` 与 `robot_interface_ready`，前者通过不代表后者成立。

旧录制格式没有保存原始 RGB／语义消息时间戳、CameraInfo 来源时间与 frame_id、LaserScan 每束时间增量，无法从已有 NPZ 补回这些证据。回校录制时必须同时保留原始 bag，包含相机、深度、CameraInfo、雷达、IMU、GPS、位姿和 TF。当前工具明确报告这些缺口，不以深度／雷达同步代替所有传感器同步。

## 新录制器的来源元数据补充

`scripts/record_ros1_rgbd.py` 现在为新录制生成 `metadata/<frame_index>.json`，保留各输入消息的原始秒／纳秒、序号、frame_id、图像编码、所用 CameraInfo 的 K/P/R/D 和来源时间、TF 查询坐标系及查询时间，以及雷达束数、角度、范围、逐束时间增量和扫描周期。每份元数据还记录对应 RGB-D／雷达 NPZ 的 SHA256；消息时间采用原始整数字段保存，避免只剩浮点秒后无法核验纳秒差值。

录制前额外核对 RGB／深度／雷达／可选语义图的最大时间跨度，而不仅逐一比较它们与深度的差值。4 项模拟消息检查通过，覆盖大时间戳下的 1 纳秒差值保留、允许零时间戳的静态标定、两端消息时间跨度超限、过期语义图及非有限雷达时间增量。证据见 [测试结果](../audit_results/ros1_source_metadata_20260914/result.json)与[原始测试输出](../audit_results/ros1_source_metadata_20260914/tests.txt)。未进行实际 ROS 收发测试。

CameraInfo 仍采用最近收到的一份，元数据明确标注它并未与图像按时间同步；记录雷达逐束时间也不等于完成运动补偿。该补充不能追溯修复旧数据，不能替代原始 bag。现有数组检查器报告发现的元数据数量，但 `source_metadata_validated=false`，尚不对这些新字段与外部 TF／标定的真实性提供认证。

## 坐标系及时间语义

ROS 的机体常用约定为 x 向前、y 向左、z 向上；相机光学系为 x 向右、y 向下、z 向前。ENU 是地理坐标表示的约定，不能默认实验室现有室内地图的 x 轴已经指向东。因此已修正 `robot_ros1.yaml` 中无条件使用 ENU 的描述，要求记录实际局部朝向；只有完成地理参考核验才标记为 ENU。[^1]

`odom` 强调局部连续性但可累积漂移，`map` 的定位估计可以发生离散跳变。使用历史时间戳查询 TF 是必要条件，但仍不能自动解决地图跳变导致的既有 TSDF 表面错位；原始 `map→odom` 记录应保留，重新优化位姿后重新融合的实车流程尚待实现。[^2]

LaserScan 的头时间戳表示第一束测量时间，`time_increment` 描述束间时间。当前简化 PlanarScan 使用一个变换表示整帧扫描，没有实现运动畸变补偿；运动中采集不能按“扫描中所有束在同一时刻”声称高精度对齐。原始 bag 必须保存这些字段，后续再接入逐束时间变换或已验证的去畸变点云。[^3]

ZED ROS 1 官方文档给出默认 32 位浮点米深度及可配置的 16 位毫米模式。当前录制器按编码显式转换，但实车 topic、图像对齐关系、光学 TF 和驱动版本仍需要现场核验，不能仅凭示例名称认定已匹配。[^4]

## 来源与验收边界

[^1]: ROS REP 103，*Standard Units of Measure and Coordinate Conventions*，2010，后续更新 2014。[官方源文件](https://raw.githubusercontent.com/ros-infrastructure/rep/master/rep-0103.rst)。
[^2]: ROS REP 105，*Coordinate Frames for Mobile Platforms*，2010。[官方源文件](https://raw.githubusercontent.com/ros-infrastructure/rep/master/rep-0105.rst)。
[^3]: ROS `sensor_msgs/LaserScan.msg`，Noetic 分支。[官方消息定义](https://raw.githubusercontent.com/ros/common_msgs/noetic-devel/sensor_msgs/msg/LaserScan.msg)。
[^4]: StereoLabs，*Getting Started with ROS and ZED*。[官方 ROS 1 文档](https://docs.stereolabs.com/docs/integrations/ros)。文献核对日期 2026-09-14。

本次结果见 `audit_results/ros1_sequence_preflight_20260914/virtual_sequence.json`。尚未连接实验室小车，没有验证外参真值、位姿漂移、导航动作服务器、自动驾驶或真实语义增益。CPU 仿真、接口准备与实车效能继续分别验收。
