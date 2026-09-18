# ROS1 来源元数据与保存数据的交叉校验

> 后续进展：独立 recording/3 记录器与对应检查器已实现并完成模拟回调测试，见 [V3 说明](ROS1_RECORDING_V3_20260914.md)。本文对 recording/2 的缺口仍成立，旧数据不能追溯补齐。

2026-09-14。已实现 `scripts/audit_ros1_source_metadata.py`，完成 13 项正反例测试，并由独立命令行进程检查 5 条模拟消息生成的存储记录。**没有连接小车，没有使用真实 rosbag，不构成实车接入、标定精度或导航效果验证。**

## 使用入口

对 `record_ros1_rgbd.py` 生成的 `ros1_rgbd_recording/2` 会话运行：

```bash
.venv-3d/bin/python scripts/audit_ros1_source_metadata.py \
  --source /absolute/path/to/recorded_session \
  --output /absolute/path/to/new_source_metadata_audit.json
```

工具从会话 `sequence.json` 读取录制时的同步阈值、深度范围、世界/光学坐标系和语义订阅配置，不提供为通过检查临时放宽这些配置的参数。源文件只读，输出文件须为新路径；失败返回非零状态，不连接 ROS、不发布运动命令。

旧版没有来源元数据的序列不能通过本工具，仍可由已有数组检查器检查有限的数组契约。不得为旧序列人工补造来源时间戳或标定信息。

## 检查内容

- 会话帧数与连续编号、深度/扫描/元数据文件一一对应；缺失和多余文件均报告。
- 原始整数秒、纳秒及 `timestamp_ns` 一致；RGB、深度、雷达和可选语义来源时间递增；跨度与相对偏移重新计算，静态 CameraInfo 时间不参加同步门。
- 按整数纳秒执行同步阈值。保存的浮点秒必须等于记录器的 `secs + nsecs/1e9` 转换；旧数组检查仅加入相应浮点表示误差余量，不能替代整数门。测试覆盖大时间戳下恰好 50 ms 通过、超限 1 ns 仍失败的情况。
- 两份 NPZ 的 SHA 与元数据一致，深度编码声明有效；未配置语义源时不能出现非零语义数据。
- CameraInfo 分辨率、K/P/R/D 的尺寸与有限性、非零标定、旋转形式，以及所存内参与记录器的 P/K 选择一致；CameraInfo 光学坐标系须与相机 TF 查询源一致。
- TF 查询目标、源坐标系和查询时间须对应录制配置与各自消息时间；相机和雷达不会被强行解释为同一时刻。
- 雷达束数、起始/终止角、角增量、量程与保存数组一致；采集时间增量为有限非负数，不能在当前记录器中宣称已完成去畸变。

继续调用旧数组检查器验证深度/RGB/语义形状、有效深度、雷达范围及刚体矩阵。校验结果保留原始整数同步门、浮点表示余量、输入哈希、逐帧错误和未验证项，`robot_interface_ready` 固定为 false。

## 已取得的证据

[检查记录](../audit_results/ros1_metadata_crosscheck_20260914/tests.json)及同目录 `tests.txt`：13 项通过，包括时间边界、时间字段冲突、文件篡改、相机内参错配、未标定相机、错误的相机/雷达坐标系、雷达角度不一致、非有限采集时长、缺失元数据、NPZ/来源时间不一致、虚假去畸变声明。

[独立 CLI 结果](../audit_results/ros1_metadata_crosscheck_20260914/fixture_audit.json)：5 条明确标为测试夹具的模拟记录通过。夹具调用实际 `sensor_metadata` 辅助函数并写入契约 NPZ，但没有调用 ROS 订阅、cv_bridge 或 TF 查询；不能称作 5 帧实车数据或真实驱动回放。源码快照、命令、输入和输出哈希一并保留。

本阶段仅新增脚本、测试和文档，没有修改当前正在执行的 V15 仿真、规划、重建或评估冻结源码。

## 仍需处理的真实接口问题

内部一致不等于物理真实：元数据和数据文件同时写错时，哈希仍可能一致。实际 TF 响应、RGB/深度配准、外参真值、位姿漂移、`map→odom` 跳变与重新融合仍须原始 bag 或现场验证。

`recording/2` 未保存 CameraInfo 的 binning 和 ROI，尚不能证明裁剪/缩放后的投影正确；ROS 官方消息定义明确这两项会影响输出图像几何。[CameraInfo 定义](https://raw.githubusercontent.com/ros/common_msgs/noetic-devel/sensor_msgs/msg/CameraInfo.msg)

本轮还核对到现有录制器会把有限且高于 `range_max` 的雷达值截到最大量程。ROS 消息约定超范围值应丢弃，而当前已保存数据没有原始 ranges，无法倒推出哪些端点来自这种截断。这一行为需要在后续独立版本录制器中修正，并保留原始扫描与转换统计，不能靠本次元数据一致性检查认证其测距正确。扫描头时间对应第一束，当前单个刚体矩阵也不等于逐束运动补偿。[LaserScan 定义](https://raw.githubusercontent.com/ros/common_msgs/noetic-devel/sensor_msgs/msg/LaserScan.msg)

浮点秒一致性使用 ROS `genpy` 的实际整数转浮点实现；原始整数纳秒仍是同步判断依据。[ROS 时间实现](https://raw.githubusercontent.com/ros/genpy/noetic-devel/src/genpy/rostime.py)

下一步宜单独版本化录制器：保存 binning/ROI、原始扫描、明确无返回与非法值转换，记录实际 TF 响应来源；随后以带有已知标定与运动的回放做检查。实际机器人是否准备好，应在完成真实录制、TF/标定、控制接口和现场安全测试后另行验收。
