# ROS1 记录器 V3：保留原始值并复核转换

> 当前真实通信环境探测：主机没有 ROS1 运行依赖，也没有缓存镜像/容器；真实消息测试未执行。已准备独立本机测试夹具，见 [环境与夹具状态](ROS1_LIVE_PROBE_RUNTIME_STATUS_20260914.md)。

2026-09-14。新增独立记录器 `scripts/record_ros1_rgbd_v3.py` 和检查器 `scripts/audit_ros1_recording_v3.py`，旧 recording/2 文件、记录器和检查结果保持不变。**9 项测试通过，实际 V3 回调在模拟 ROS 模块下完成一条记录，独立命令行检查通过；没有连接真实 ROS 或小车。**

## 修正与新增记录

旧记录器把有限且高于雷达最大量程的值截到最大量程，混淆了无效输入与可用观测。V3 将有限超范围值、低于最小量程或非正数、NaN、负无穷转换为无效零值。正无穷默认同样为无效；仅在核实驱动把它定义为无返回后，才能显式选择 `clear_to_max`。两种模式均不把有限超范围值截成有效值。ROS 的 LaserScan 消息定义要求丢弃超范围值。[官方定义](https://raw.githubusercontent.com/ros/common_msgs/noetic-devel/sensor_msgs/msg/LaserScan.msg)

每条记录新增 `raw/<编号>.npz`，保留 cv_bridge passthrough 深度数组及原始 float32 雷达 ranges/intensities，包括无穷和 NaN。元数据记录转换分类数量、正无穷策略以及三份 NPZ 的 SHA。这里没有保存原始图像消息的全部字段、压缩图像字节或完整 TF 链，仍需要原始 bag。

CameraInfo 现在额外保存 binning、ROI、实际使用的内参，以及从校正后相机到来源光学坐标系的旋转。已经校正的左相机图像支持先减去 ROI 偏移、再按 binning 缩放焦距和主点；保存图像尺寸须与裁剪/合并后的尺寸一致。ROS image_geometry 使用相同的偏移和缩放顺序。[官方实现](https://raw.githubusercontent.com/ros-perception/vision_opencv/noetic/image_geometry/src/pinhole_camera_model.cpp)

CameraInfo 的 R 将来源相机坐标变换到校正后的相机坐标，因此反投影射线返回来源坐标使用 R 的转置，再与来源光学 TF 复合。这是依据消息坐标定义作出的矩阵推导，必须在实际驱动上核验来源 TF 是否采用该坐标基。[CameraInfo 定义](https://raw.githubusercontent.com/ros/common_msgs/noetic-devel/sensor_msgs/msg/CameraInfo.msg)

当前输入契约为配准、校正后的左相机深度/RGB：要求有效 K、P 和 R；不能丢弃非零双目投影平移来冒充单相机内参。需要再次矫正的 ROI（`do_rectify=true`）会被拒绝，本工具不执行图像去畸变或重采样，不能称为任意 CameraInfo 配置均支持。

TF 查询分别使用相机和雷达自身时间。记录实际 TransformStamped 返回的目标/源坐标系、整数时间、平移、四元数和矩阵，而不仅是查询声明。传感时间为零会请求“最新 TF”，与历史时刻查询不同，因此该记录器拒绝零时间的传感帧。静态 CameraInfo 的零时间仍允许。

每条完整记录写入后更新会话清单，清单通过临时文件替换发布；回调与关闭保存串行化。文件组不具有跨文件事务保证，断电或磁盘写满后的残留/缺失文件仍应由检查器报告，不能仅凭目录存在断言录制成功。

## 回校使用

在已经配置 ROS1 的小车终端，用能够导入 rospy、tf2_ros、message_filters、cv_bridge、sensor_msgs 和 NumPy 的 Python 运行：

```bash
python3 /absolute/path/to/NSO/scripts/record_ros1_rgbd_v3.py \
  _output:=/absolute/path/to/new_recording \
  _world_frame:=map \
  _camera_optical_frame:=VERIFIED_LEFT_OPTICAL_FRAME
```

将光学坐标系和相机/雷达话题替换为小车已验证的实际配置；默认话题沿用 ROS1 ZED 左图、配准深度和 `/scan` 示例。本工具不发布控制命令。仍须同时录制包含相关传感器、标定、位姿、`/tf` 和 `/tf_static` 的原始 bag。

在 CPU 工作区对新目录运行：

```bash
.venv-3d/bin/python scripts/audit_ros1_recording_v3.py \
  --source /absolute/path/to/new_recording \
  --output /absolute/path/to/new_audit.json
```

检查器在 recording/2 检查基础上新增：从原始数组重新计算深度和雷达转换、核对转换分类统计、重算裁剪/合并内参、从保存的四元数重算 TF 矩阵，并检查相机 TF 与 R 转置的复合。只检查保存数据及其内部关系，不认证物理标定；`robot_interface_ready=false` 保持不变。它不接受 recording/2 目录，旧目录使用对应的旧检查器，不能补造来源字段。

## 验证证据与边界

[测试及源码记录](../audit_results/ros1_recording_v3_20260914/manifest.json)包含 9 项测试：有限超量程值、显式正无穷策略、深度编码与数据类型、裁剪/合并投影、校正坐标基、无法直接使用的 ROI/双目平移、TF 返回坐标系错配，以及模拟 ROS 下实际 `main()`/同步回调/关闭保存流程。

[独立检查结果](../audit_results/ros1_recording_v3_20260914/offline_audit.json)来自该回调生成的数据。正例包含非单位校正旋转、ROI、binning、原始无穷/NaN 和毫米深度；反例把无效雷达值改为 4 米、或从所存位姿中移除校正旋转，即使重新填写 NPZ 哈希仍会被拒绝。最初一项测试把 float32 的 0.2 与 float64 字面值逐位比较失败，已将预期数组改为契约声明的 float32；没有放宽转换算法或验收容差。

尚未验证真实 message_filters 配对、cv_bridge 兼容性、TF 缓冲历史、ZED 具体驱动的 R/深度语义、外参真实性、雷达逐束去畸变或实际导航。V3 保存逐束时长但仍使用单个雷达变换，不宣称已完成运动补偿。原始数据与实际 TF 返回记录让后续检查可追溯，不能替代这些物理验证。

V15 仿真候选实验继续使用原冻结源码，本阶段没有修改其规划、传感或评价实现。
