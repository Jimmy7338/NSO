# ROS1 真实通信测试夹具与当前环境状态

2026-09-14。**真实 ROS 通信测试未执行。** 已准备 `scripts/ros1_live_recording_v3_probe.py` 并运行依赖探测；当前主机没有 roscore，使用的 Python 缺少 rospy、tf2_ros、message_filters、cv_bridge、sensor_msgs 和 geometry_msgs。Docker 服务可响应，但当前镜像与容器均为零。

以前的 TARE Noetic 容器构建和节点检查是有留存证据的历史结果，不代表其运行环境仍在本机。没有把本次环境缺失写成历史构建失败，也没有在存储紧张时重新下载完整 ROS 镜像。

## 已准备的真实通信夹具

在已加载 ROS1 环境的机器上运行：

```bash
python3 /absolute/path/to/NSO/scripts/ros1_live_recording_v3_probe.py \
  --output /absolute/path/to/new_live_ros_probe
```

夹具设计为启动独立本机 ROS master，使用真实 CameraInfo、Image、LaserScan 和静态 TF 消息，启动实际 V3 记录器并检查订阅建立，发送十组合成传感数据，再运行来源/原始数组检查。输出目录保存环境、子进程日志、会话、元数据、实际消息主题清单和结果。它不发布运动指令，不使用实验室机器人或真实驱动数据。

只有依赖完整后才启动 master。当前实际运行在依赖门返回非零状态，没有启动 ROS master、记录器或传感发布器；后续消息通信、TF 缓冲和 cv_bridge 路径仍未跑过。源码通过 Python 3.8 语法解析，这也不能替代在 Noetic Python 环境实际导入与运行。

即使未来夹具通过，能够支持的结论也仅是合成数据经过实际 ROS 通信/TF/cv_bridge/记录器链路。它不证明 ZED/镭神驱动正确、外参真实、自然语义有效或机器人自主运动成功。

[本次环境结果](../audit_results/ros1_live_v3_runtime_probe_20260914/result.json)与同目录源码快照保留了 `not_executed_missing_runtime` 状态。此前 [V3 模拟回调验证](ROS1_RECORDING_V3_20260914.md)的正证据继续成立，但不能替换此未执行项。

## 为主实验回收空间

对明确已完成的 V13、V13.2、V15 smoke 及当前 V15 已独立重放的分支，先只读检查 260 份至少 1 MB 的 NPZ，找到 28 组精确重复内容。操作前检查了打开的可写文件句柄、常规文件类型、设备、权限、所有者与文件签名；逐字节核对后，用硬链接合并重复存储，最后逐个复查原路径 SHA。

共替换 49 个路径，末链接被移除的文件分配块合计 80,429,056 字节，约 76.70 MiB。该数为去重文件的占用统计，不是同时运行其他任务时磁盘空闲量差值。所有文件路径和原始内容保留；没有删除冻结实验、原始传感数据或其他项目临时文件，没有修改正在写入的分支。

[去重证据](../audit_results/v15_completed_mesh_dedup_20260914/result.json)记录了原/新 inode、文件路径、SHA 和空间核算。当前候选实验源码哈希再次检查通过，继续使用会话 68829；以后接续仍须轮询该实际进程，不根据旧状态文件或短暂无输出重启任务。
