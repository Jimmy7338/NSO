# 三维虚拟实验与 ROS 1 小车前置工作

日期：2026-09-10。接受新的双目标：有限预算下的二维覆盖范围与三维重建质量。保留此前的真实动作成本、安全视点、多步视点顺序与收益去重；原有静态语义加分仅作为对照。

**最新阶段已完成：** [受控补看机制与第二版实施决定](INSPECTION_MECHANISM_AND_QUALITY_V2.md)。224 个分支表明补全与精度收益需要分开预测；角度代理的精度排序相关接近零，后端尺度也是显著因素。下一步先独立固定后端尺度，再校准质量预测，不直接扩大旧奖励的闭环实验。

## 1. 实验室设备与接入决定

用户确认：ROS 1、ZED 双目相机、镭神单线雷达、IMU、GPS；ZED 已能发布深度和位姿。车载中控的准确型号尚未核对，不据口述推断 GPU 型号。**以现成可用的 ZED 深度链路为实车输入，不要求在当前无 GPU 服务器重建 ZED SDK 环境。**

分工：平面雷达负责二维自由／障碍观测，ZED 深度补充雷达扫描平面之外的障碍并融合三维表面，现有定位链路提供位姿。IMU／GPS 的原始数据随 bag 保留；本阶段没有实现新的惯性／GPS 融合，室内不默认采用 GPS 定位。

ZED 官方 ROS 1 文档列出注册深度、相机标定和位姿话题，深度默认为 32 位浮点米，也支持 16 位毫米配置。准确话题与光学坐标系必须以实车为准。[ZED 节点文档](https://docs.stereolabs.com/docs/integrations/ros/zed-node)

当前维护目标是兼容实验室现有 ROS 1。官方 ROS 1 wrapper 已停止维护，因此回校后记录现有 ROS、ZED SDK、wrapper 与系统版本，并保留可用环境；本次不强制升级或迁移 ROS 2。[官方维护状态](https://github.com/stereolabs/zed-ros-wrapper)

## 2. 已实现的 CPU 三维闭环

```text
三维网格世界（仅模拟器和评价器可见）
  → CPU 深度射线投射＋平面雷达
  → 同步传感器帧＋位姿接口
  → 二维占据地图＋Open3D TSDF＋已观测表面的视角记录
  → 安全候选视点＋含转向的最短路＋两个视点的边际收益
  → 移动／转向
  → 新的传感器观测与三维重建
  → 独立评价二维覆盖、三维准确率／完整率／F1及乘积
```

采用 Open3D 0.19.0 CPU wheel，独立环境 `.venv-3d`，未改变旧 `.venv-cpu` 或其冻结依赖。无需显示器、OpenGL 窗口、CUDA 或下载大型场景资产。射线投射与 TSDF 都在 CPU 执行。[CPU 安装说明](https://www.open3d.org/docs/release/getting_started.html)、[射线投射接口](https://www.open3d.org/docs/release/python_api/open3d.t.geometry.RaycastingScene.html)、[TSDF 融合](https://www.open3d.org/docs/release/tutorial/pipelines/rgbd_integration.html)

当前可生成 8×6 m 的分房间／杂物场景，有平面柜体、多层架体、立柱和遮挡。参数：机器人半径 0.2 m、二维栅格 0.2 m、相机高度 0.8 m、96×72 深度图、水平 90°、深度 4 m、TSDF 体素 6 cm、平面雷达 180 束覆盖 360°。深度噪声标准差 1 cm、随机无效像素率 1%；这些是虚拟实验参数，**不是已标定的 ZED 或镭神参数**。

这是三维几何和传感器闭环，不是照片级渲染，也不是双目图像匹配仿真。传感器输出相当于“ZED 已计算完成的深度”；当前采用完美模拟里程计，没有模拟轮胎打滑、定位漂移、IMU 或 GPS。位姿已以可替换接口输入。每个原子动作计 1 秒虚拟时间，在动作终点采样；实际小车的速度、转向时长与连续采样还需标定。当前二维控制器仍为离散平面运动，不应直接转换为实车速度指令。

## 3. 联合目标及对照

新任务允许覆盖与重建质量之间的可解释权衡。旧二维覆盖实验的负结果仍保留，不按新指标重新包装；旧“必须提高纯二维 AUC”的关口不再单独决定双目标方案的成败。

`J(t) = C2D(t) × F1_3D(t)`，同时独立报告 C2D、三维准确率 P、完整率 R、平均／95% 表面误差、F1、复杂物体与简单物体表面完整率。阈值为 5 cm 和 10 cm；不能只报告乘积，也不能把完整率提升称作表面误差下降。

完整率分母由固定可达位置格点、传感器范围和高度筛出的可观测真实表面定义，与策略无关，包含未访问区域。它是确定格点近似的可观测集，不宣称遍历了全部连续位姿。真实表面按三角形面积采样，重建表面同样按面积采样；距离查询直接针对网格。准确率查询真实网格；完整率查询重建网格。最终乘积精确由对应标量计算；联合 AUC 是每 20 步网格检查点的梯形近似，**不是逐帧精确 AUC**。

| 对照 | 目的 |
|---|---|
| coverage | 只最大化预测二维覆盖，三维后端照常运行 |
| geometric_quality | 在覆盖基础上寻找观测方向不足的表面，无语义权重 |
| semantic_static | 持续偏好指定语义类别，检验“高语义多看” |
| semantic_quality | 语义只放大尚有新视角收益的表面，收益饱和后不重复奖励 |

所有对照共享传感器、二维映射、TSDF、动作成本和两视点规划实现。质量方法使用同一套补看候选。位姿＋朝向最短路计算完整移动／转向成本；两个目标的覆盖并集去重，表面补看收益在一次短路线内保守地最多计算一次。长距离转移仍保留，不把短规划时域当作全局不可达条件。

目前质量代理是“同一已观测表面的新方位角与方向数量不足”，不是 TSDF 真实误差，也不是已校准的不确定性后验。语义是传感器可见的合成类别，不是运行 ZED 目标检测或开放词汇模型。当前对照中的 `shuffled` 具体是交换简单／复杂两种类别，不是每帧随机噪声；扰动不改变几何、深度、雷达或真实评价分类。RGB 是类别伪彩色，标签交换也改变此显示颜色；本协议只比较几何重建，不评价真实纹理或颜色。无语义的两种对照在标签交换前后应产生相同轨迹与几何质量。

原计划的完整全局区域访问路线尚未接入三维闭环；当前只结合安全视点、朝向最短路和短期两视点顺序。后续应在本平台上实现并单独消融全局区域规划，不能把本次实现描述为完整 TARE／FALCON 复现。

## 4. 已交付入口与复现

初次配置：

```bash
python3 -m venv .venv-3d
.venv-3d/bin/python -m pip install --no-cache-dir -r requirements-3d.lock.txt
```

短流程：

```bash
OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1 .venv-3d/bin/python scripts/eval_virtual3d.py \
  --config configs/virtual3d/smoke.json --output eval_results/my_virtual3d_smoke
```

多场景试验与指标复核：

```bash
OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1 .venv-3d/bin/python scripts/eval_virtual3d.py \
  --config configs/virtual3d/pilot_v1.json --output eval_results/my_virtual3d_pilot
OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1 .venv-3d/bin/python scripts/analyze_virtual3d.py \
  eval_results/my_virtual3d_pilot
```

输出目录必须不存在，防止覆盖证据。每回合包含深度／RGB／标签／标定／位姿帧、雷达帧、二维地图观测、动作、候选分数、TSDF 网格和逐检查点指标。运行目录还保存源码快照、配置与依赖哈希。

已有结果：[32 回合小试报告](../eval_results/virtual3d_pilot_v1_20260910/results.md)、[可旋转三维模型](../eval_results/virtual3d_pilot_v1_20260910/viewer.html)、[审计](../eval_results/virtual3d_pilot_v1_20260910/verification.json)。包含 4 个几何实例、每个正确／交换标签条件、4 种方法；只能支持工程流程与初步机制观察，不能作为大尺度泛化或实车效果证明。

正确标签的 4 个几何实例上，语义质量方法的联合 AUC 均值为 0.7600，纯覆盖 0.7231，几何质量方法 0.7414；最终联合分数分别为 0.9457、0.9435、0.9250。语义质量方法的平均表面误差为 7.75 mm，纯覆盖为 7.36 mm，**未证明表面精度改善**。标签交换后语义质量方法联合 AUC 降至 0.7367，最终联合分数降至 0.8998。结论是存在早期重建完整性的信号、对标签有依赖且场景间不稳定；不是语义普遍有效的证据。

32 回合保存 3780 帧，222 个三维网格检查点的指标已独立复算；零碰撞，执行循环合计约 147 秒，峰值进程内存约 304 MiB。CPU 可承担当前前置实验，尚未测量大场景、多线程负载下或实车上的实时性能。

## 5. 回校前可复用的数据格式

每一帧 NPZ 包含：

- `timestamp_s`：采集时刻（秒）。
- `depth_m`：光学轴方向深度、米；无效值为零，不把欧氏距离直接当成深度。
- `color_rgb`：与深度对齐的 RGB。
- `intrinsic`：对应当前分辨率、校正后的 3×3 内参。
- `world_from_camera`：相机光学系到固定世界系的 4×4 变换；光学系为 x 右、y 下、z 前。
- `semantic`：可选、与深度对齐的类别图；无感知模块时为零。

雷达独立记录量程、角度、最大量程、自己的采集时刻与 `world_from_laser`。禁止使用“当前最新 TF”替代历史采集时刻的 TF。模拟器与 ROS 录制器都输出此格式；同一离线工具融合为 PLY 网格。

```bash
OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1 .venv-3d/bin/python scripts/replay_rgbd_sequence.py \
  --source eval_results/virtual3d_smoke_20260910/rooms_seed71_aligned_semantic_quality \
  --output eval_results/my_replayed_mesh --config configs/virtual3d/smoke.json
```

本地已验证 61 帧录制回放与在线重建的顶点、三角形一致。真实序列回放不需要三维真值；但没有独立真值时不能输出“真实三维精度”结论。小车地图可能有任意全局坐标，TSDF 使用完整米制位姿；回放器内的固定二维栅格不用于实车导航。

## 6. 回校后的最短验证顺序

### A. 先录制，复用现有驱动

先确认真实 topic、时间戳、深度单位、相机内参、`map/odom → base_link → camera_optical_frame` 与雷达外参。配置清单见 `configs/virtual3d/robot_ros1.yaml`。

在小车已经配置好 ROS 的终端运行录制器（使用 ROS 的系统 Python，不使用服务器 Python 3.12 环境）：

```bash
python3 scripts/record_ros1_rgbd.py \
  _output:=/absolute/path/to/new_session \
  _world_frame:=map \
  _camera_optical_frame:=ACTUAL_VERIFIED_OPTICAL_FRAME \
  _depth_topic:=/zed/zed_node/depth/depth_registered \
  _rgb_topic:=/zed/zed_node/left/image_rect_color \
  _camera_info_topic:=/zed/zed_node/depth/camera_info \
  _scan_topic:=/scan
```

这些是示例 topic；`ACTUAL_VERIFIED_OPTICAL_FRAME` 必须替换为实车真实 TF，不能照抄。录制器同步 RGB、深度与雷达，按各自时间戳查 TF，检查深度编码及标定尺寸；没有正确标定时拒绝帧并给出原因。可选 `~semantic_topic` 输入与深度对齐的 mono8 标签图。默认没有语义识别器，不凭颜色给真实物体自动赋类。

同时保存原始 bag：RGB／深度／CameraInfo／雷达／IMU／GPS／TF／现有位姿，topic 以实车为准。现成 ZED 位姿可作第一阶段输入，但其漂移与地图闭环跳变需要另行检验。录制器只采集数据，不发布运动指令。**当前服务器没有 ROS 1 环境，录制器完成语法与共享数据格式验证，尚未在 ROS 或实车运行。**

### B. 同一后端重建真实数据

在小房间手动走一条短路线，保存数据、离线融合 PLY，检查尺度、墙体重影、坐标方向和遮挡表面。站定重复拍摄、改变视角绕物体拍摄分别记录，用独立参考或已知尺寸物体验证真实收益。普通 bag 只能证明给定轨迹的数据重建，不能证明任意主动规划轨迹有效；不能在固定录制轨迹上“模拟走到”没有录制的视点。

### C. 再接导航与自动目标

确认现有定位和导航接口后，以目标位姿接入 `move_base`，复用实车局部避障与执行状态。当前离散栅格动作不直接下发 `cmd_vel`。这一接口尚未实现与验证，本次不宣称已达到即插即用自动巡航。先显示目标、人工执行采集，再在现场确认底盘约束后开放自动目标。

### D. 最后做实车对照

同一区域、相同时间预算、相同后端，比较纯覆盖、几何质量补看、语义质量补看；单独保留静态语义对照。选择独立参考扫描、量测目标或质量足够的独立高密度采集作参考，说明参考精度与配准方式。不能把被测方法融合的地图同时当作真值。重建模型对齐协议统一，并报告位姿／地图误差；只报 ICP 对齐后的表面距离可能掩盖定位漂移。

## 7. 后续工作关口

1. **本次：工程闭环**——深度、雷达、映射、轨迹、网格和指标连通；回放一致；输入与真值隔离。
2. **机制诊断已完成，质量模型待实现**——已比较角度代理与实际误差／完整率收益，见最新报告；先固定后端尺度，再加入距离、三维遮挡、视角、深度置信度与位姿不确定性，替换单纯视角数量。
3. **全局层：区域路线**——在多房间与更大场景上验证预算路线与局部补看的协同。固定各方法预算，报告覆盖／质量折中和失败家族。
4. **泛化**——独立场景种子、几何复杂度与语义类别解耦、错误／缺失标签、深度与位姿噪声、体素和阈值敏感性。先冻结新协议，再做独立验证。
5. **实车**——先录制回放，再闭环导航，最后做重复配对实验。新目标的有效性仍需实验，而非因为更改指标就自动成立。
