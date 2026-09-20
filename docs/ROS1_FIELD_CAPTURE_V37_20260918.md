# V37 回校静态采集清单与 ROS1 接口核验

2026-09-18。当前交付的是可填写的采集配置和操作清单，尚未收到实验室原始 bag 或真实设备图像，也未在小车运行。沿用 [V3 记录器](ROS1_RECORDING_V3_20260914.md)、`scripts/audit_ros1_recording_v3.py` 和已有合成通信夹具；不修改旧记录器、冻结实验或已有检查结果。

用户已确认 ROS1、ZED 深度和位姿可发布、镭神单线雷达、IMU 与 GPS；具体相机/雷达型号、话题、坐标系、标定、车载计算平台仍待实机确认。V35/V36 的精确模拟位姿和人工类别结果不能验证这些实机条件。本阶段只采集静止观测，不发布 `/cmd_vel` 或其他运动控制命令，不追加虚拟闭环任务，累计仍为 35/36。

## 本机能做与不能做的部分

对当前 `.venv-3d/bin/python` 的模块发现和命令查找结果为：NumPy 可用；`rospy`、`tf2_ros`、`message_filters`、`cv_bridge`、`sensor_msgs`、`geometry_msgs`、`rosbag` 均不可发现；`roscore`、`rosbag`、`rostopic`、`rosparam`、`rosrun` 不在 PATH 中。仅做发现检查，没有导入 ROS 库、启动 master、连接小车或下载镜像。

这是本机运行环境缺失，真实通信测试状态为“未执行”，不是新的通信失败。此前 V3 的 9 项转换/模拟回调测试继续有效，但不等于实机验证。已有 `scripts/ros1_live_recording_v3_probe.py` 可在已配置 ROS1 的独立测试终端运行；它会启动本机独立 master 并发送合成消息，不能把它当作只读依赖探测器，也不要在当前采集流程中顺手运行。

## 先填写配置和核验记录

[配置示例](../configs/virtual3d/ros1_field_capture_v37.example.json)是交接清单，**V3 不直接读取这份 JSON**。其中所有话题字符串均为例子；`VERIFY_*`、`/ABSOLUTE/*` 必须替换，`verification` 的每一项须有实际证据后才能改成 `true`。在 `verification_evidence` 保存检查命令输出或记录文件的路径/哈希、核验时间与人员；这些人工标记自身不能证明硬件正确。

2026-09-20 接口验收补充：新增离线预检器从 V3 源码提取真实参数集合，核对填写项后仅打印命令，不连接 ROS。用未填写示例运行下面命令时，预期退出码为 2 且不输出采集命令；完成实机核验后，对另存的配置运行同一检查。输出目录的父目录须已存在，目标须为新路径；应在实际录制机器上核对。

```bash
.venv-3d/bin/python -B scripts/preflight_ros1_field_capture_v37.py \
  --config configs/virtual3d/ros1_field_capture_v37.example.json --print-commands
```

新增 `scripts/preflight_ros1_field_capture_v37.py` 只做离线配置校验，从现有 V3 源码 AST 提取实际私有参数集合。例子默认返回非零状态 2，未确认的字段不输出采集命令。填写完后，可在采集主机运行以下命令；它只打印经过 shell 引用的两个命令，不执行它们、不导入 ROS、不启动 master，也不创建录制目录。所有核验标记均须为 `true`，且每项存在证据引用；这仍只是人工核验声明完整性，不认证证据内容或硬件。

```bash
python3 /ABSOLUTE/NSO/scripts/preflight_ros1_field_capture_v37.py \
  --config /ABSOLUTE/FILLED_CAPTURE_CONFIG.json --print-commands
```

预检要求输出父目录在当前主机存在、目标新目录/原始 bag 前缀未使用；空间量仍须人工测量。打印 V3 命令时省略空 `semantic_topic`，沿用其空默认值，避免 ROS 命令行把空 YAML 值解析为 null。其余参数名均来自 V3；位姿、IMU、GPS 只进入 bag 命令。需要非标准 TF 话题重映射时应另行核验，本导出器固定保留 `/tf` 与 `/tf_static`。

| 采集内容 | 示例/待确认值 | V3 是否直接读取 |
| --- | --- | --- |
| 校正后的左 RGB | `/zed/zed_node/left/image_rect_color` | 是，Image |
| 与左 RGB 对齐的轴向深度 | `/zed/zed_node/depth/depth_registered` | 是，Image |
| 对应深度/左相机的 CameraInfo | `/zed/zed_node/depth/camera_info` | 是，接收最近一条，不参与时间同步 |
| 镭神单线扫描 | `/scan` | 是，LaserScan |
| ZED/小车位姿 | `/VERIFY_POSE_TOPIC`，实际类型待查 | 否；原始 bag 必须保留 |
| IMU | `/VERIFY_IMU_TOPIC` | 否；原始 bag 保留 |
| GPS | `/VERIFY_GPS_TOPIC` | 否；原始 bag 保留，室内不用于定位 |
| 动态/静态变换 | `/tf`、`/tf_static`，也须确认 | V3 通过 tf2 查询；原始 bag 保留整条链 |
| 类别图 | 本阶段空字符串，未提供 | 可选 mono8 外部标签，当前关闭 |

V3 依赖相机和雷达在各自时间戳上的 TF；只有位姿话题而没有可查询 TF 不足以运行。相机和雷达分别查变换，不能拿一条相机位姿替代雷达外参。

下表为从 V3 源码核对的完整私有参数名。ROS 命令行使用 `_名称:=值`；不要把输出 `sequence.json` 中的 `optical_frame`、`info_topic`、`slop_s` 误写成参数名。

| 参数名 | 现有默认值/规则 |
| --- | --- |
| `output` | 必填，绝对新目录；已存在目录会拒绝 |
| `world_frame` | `map`，应按实际定位链核验 |
| `camera_optical_frame` | 空；本次要求显式填写已核验左光学坐标系 |
| `depth_topic` | `/zed/zed_node/depth/depth_registered` |
| `rgb_topic` | `/zed/zed_node/left/image_rect_color` |
| `camera_info_topic` | `/zed/zed_node/depth/camera_info` |
| `scan_topic` | `/scan` |
| `semantic_topic` | 空；有外部标签时会额外加入同步队列 |
| `max_depth_m` | `4.0`，保存有效深度上限；不是相机量程认证 |
| `sync_slop_s` | `0.05`，RGB/深度/雷达近似同步窗口；须检查实测分布 |
| `positive_inf_policy` | `invalid`；只有核验驱动把正无穷定义为无返回后才可选 `clear_to_max` |
| `min_interval_s` | `0.2`，限流间隔；不保证每秒一定保存 5 帧 |

## 实机只读预检与采集模板

在实验室已加载正确 ROS1 环境的终端，先保存 `rostopic list`、相关 `rostopic info`、`rostopic type` 和各消息的 `header` 样本。对 RGB、深度、CameraInfo、LaserScan、位姿、IMU、GPS 均核对类型、非零传感时间戳、frame_id、频率；GPS 室内无有效定位也保留原始状态。确认所用 Python 实际能导入记录器依赖，不能只凭模块文件存在判定兼容。

核对 `world → 左相机来源光学坐标系` 与 `world → 激光坐标系` 的 TF 链和实际外参。在传感时间戳查询 TF；一次 `tf_echo` 最新值可用于查看链，但不能验证历史查询、相机/扫描同步或物理标定。记录 `map → odom` 的跳变；发生回环/重定位跳变时先标记和隔离片段，V3 不会自动重新融合旧点云。

以下是**替换全部占位值之后才可使用**的原始 bag 模板。在终端 A 先录 bag，在终端 B 启动 V3。先确认目标盘空间，短录一段后用 `rosbag info` 检查话题和实际增长率，再决定正式片段时长；分卷不是空间配额，不会自动保证磁盘不满。

```bash
rosbag record --split --size=1024 -O /ABSOLUTE/NEW_RAW_BAG_PREFIX \
  /VERIFIED_RGB_TOPIC /VERIFIED_REGISTERED_DEPTH_TOPIC \
  /VERIFIED_CAMERA_INFO_TOPIC /VERIFIED_SCAN_TOPIC \
  /VERIFIED_POSE_TOPIC /VERIFIED_IMU_TOPIC /VERIFIED_GPS_TOPIC \
  /tf /tf_static
```

```bash
python3 /ABSOLUTE/NSO/scripts/record_ros1_rgbd_v3.py \
  _output:=/ABSOLUTE/NEW_RECORDING_DIRECTORY \
  _world_frame:=VERIFIED_WORLD_FRAME \
  _camera_optical_frame:=VERIFIED_LEFT_OPTICAL_FRAME \
  _depth_topic:=/VERIFIED_REGISTERED_DEPTH_TOPIC \
  _rgb_topic:=/VERIFIED_RGB_TOPIC \
  _camera_info_topic:=/VERIFIED_CAMERA_INFO_TOPIC \
  _scan_topic:=/VERIFIED_SCAN_TOPIC \
  _max_depth_m:=4.0 _sync_slop_s:=0.05 \
  _positive_inf_policy:=invalid _min_interval_s:=0.2
```

`semantic_topic` 未传递，使用空默认值，不伪造自然语义标签。bag 命令显式列话题，不执行 `rosbag record -a`。两个终端均以 Ctrl-C 正常停止并检查清单、文件组、bag 可读性；V3 只有组文件写完后才更新清单，断电/写满的残缺须由离线检查报告。保存启动命令、配置、驱动版本、时间同步设置、机器 ROS 环境和 bag SHA256。若需 bag 回放，另开隔离 ROS 环境，保留原始时间/TF，不在现有机器人 master 上混入回放数据。

## 数据入库前必须看原始值

1. **深度**：从 bag 核对消息 encoding、dtype、图像尺寸及已知距离平面。V3 仅支持 `16UC1` 的 uint16 毫米轴向深度与 `32FC1` 的 float32 米轴向深度；不能按值大小猜单位，也不能把沿射线距离当轴向 Z。0、非有限、负值和大于 `max_depth_m` 的值保存为无效零；原始数组另存 `raw/`。驱动约定与此不同应先明确转换方案，不修改字段冒充兼容。
2. **配准与标定**：核对 RGB/深度是否为同一校正光学相机的同尺寸对齐图像。CameraInfo 的 K/P/R、ROI、binning 和来源光学 frame 要匹配；V3 不执行重投影、去畸变或图像重采样，不接受 `ROI.do_rectify=true` 或非零双目投影平移。反投影后使用 `TF × R转置`，必须确认驱动 TF 基与 R 的定义，不能仅凭内部矩阵自洽证明外参正确。CameraInfo 若改变，要分段并核对最近接收关联，不能声称已时间同步。
3. **激光**：从原始 ranges 统计有限有效值、低于最小量程/非正值、有限超上限、NaN、正/负无穷。默认全部无效类别均不当自由空间；有限超上限不能截成有效最大距离。只有有驱动证据时才启用正无穷清空策略。保留角度、`time_increment`、`scan_time`；V3 使用一帧扫描的单个 TF，未做逐束运动补偿，因此第一阶段只用静态停稳片段。
4. **TF 与时间**：核对实际返回的 target/source、查询传感时刻、TF 历史可用性、四元数及相机/雷达各自时间。源传感时间为零会请求最新 TF，V3 明确拒绝；静态 CameraInfo 的零时戳可以接受。累计丢帧、拒绝、相对时差都应报告，不能只看录到了多少帧。
5. **实物对齐**：对同一固定平面和设备，从两个停稳视点检查点云的距离/尺度、重复边缘、重影和法向。保留全部失败视点，先定位内参、R/TF、时间与位姿误差。此处是物理核验，现有 V3 离线检查器并不自动计算或认证这些误差。

先运行已有检查器，不重新实现原始数组转换：

```bash
/ABSOLUTE/NSO/.venv-3d/bin/python -B \
  /ABSOLUTE/NSO/scripts/audit_ros1_recording_v3.py \
  --source /ABSOLUTE/NEW_RECORDING_DIRECTORY \
  --output /ABSOLUTE/NEW_RECORDING_AUDIT.json
```

检查通过仅证明保存数据、来源元数据和转换的一致性；`robot_interface_ready=false` 继续保留，直到真实通信、驱动语义、外参、时钟、静态多视图和后续机器人接口分别验证。

## 静态设备多视图与语义先验标注

首批目标为至少两类、每类五个独立实物实例、每实例两个不同停稳视点；这只是可行性规模，不构成充分统计功效。设备应固定不动，小车的重新摆位由实验室现有安全操作流程负责，当前项目不发运动命令。记录可见正面身份线索、侧面/背面的外露结构与遮挡，包含类别先验不成立的型号、未知设备和识别失败样本；不在拍摄时通过人工颜色标签制造类别关联。

每个样本至少记：`physical_device_instance_id`、`category_id` 与定义版本、`model_id`（已知才填）、采集许可/来源、bag SHA256、图像话题、整数时间戳、V3 帧索引、视点 ID、可见区域/遮挡、设备局部正面定义、观察侧的外部结构事实与证据。铭牌/目录匹配应独立标成“设备身份先验”；人工标注的复杂侧/目标侧属于评估真值，不允许作为运行时输入。设备位置或文件名不得编码类别给前端。

在训练/阈值校准前按**实物实例**固定分组：同一设备跨日期、bag、近邻帧和视点全放同一组；每类五实例可预定三实例 train/calibration、两实例 holdout，并保存分组哈希。若目标是跨型号迁移，另须整型号留出，不能把同型号的新实例称新型号泛化。holdout 只做一次冻结前端评估，不能选出有利帧后再评分。任何类不足目标实例数时照实报告规模，不能用连拍增加“独立实例”。

先验关系与识别准确率分开记录：前端识别正确但类别—结构侧关系失效，属于先验反例；前端认错则属于感知误差。两者都要进入混淆矩阵、拒识率及关系命中统计。真实图像和依赖未具备前，这些指标保持未测，不把 V35/V36 颜色标记或 mock ROS 图像写进自然图像测试集。

本清单完成后仍需连续可达空间、局部控制、回环处理和安全返航适配，才能讨论在线实车；整数安全图的 CPU 控制器不能直接作为小车导航节点。
