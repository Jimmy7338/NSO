# 固定官方 TARE 的实际构建探测与后续执行方案

本轮获得了**官方 TSP 路线求解组件的原生编译和运行证据**，尚未获得完整 TARE 节点、CMU 场景演示或公平基线结果。当前 Docker 服务可用，但无缓存镜像；宿主没有 ROS 1。完整规划器的实际 CMake 配置在缺少 catkin 时失败。考虑本轮新增磁盘最多 1 GiB，没有拉取 ROS 镜像或安装超出预算的依赖。

所有命令、日志、版本、测试程序和摘要保存在 [`audit_results/tare_native_build_probe_20260910`](../../audit_results/tare_native_build_probe_20260910)。主要文件是 `command_ledger.json`、`probe_summary.json`、`planner_configure.log`、`tsp_build.log`、`tsp_execution.jsonl`。

## 已核验的当前状态

| 项目 | 本次实际检查结果 |
|---|---|
| 操作系统 | Ubuntu 24.04.2 LTS，amd64 |
| Docker | client 和 daemon 均为 29.1.3，服务可响应；0 容器、0 镜像 |
| CPU / 内存 | 4 CPU，约 7.75 GiB 内存；本轮编译仅用 1 个工作进程 |
| ROS 1 / catkin | `/opt/ros`、`roscore`、`catkin_make` 不存在；当前 Noble 软件源没有所查 ROS 1 软件包候选 |
| 编译工具 | GNU C++ 13.3.0，CMake 版本原样保存在 `cmake.log` |
| TARE 固定源码 | `caochao39/tare_planner`，commit `44500592b86138257273e0cab264e6a847ccefc7`；2846 个文件哈希与下载时清单一致 |
| CMU 固定源码 | `HongbiaoZ/autonomous_exploration_development_environment`，commit `bf0cba71365271ebff09831a05afd78578150300`；124 个文件哈希一致 |
| 官方场景资产 | 本地仅有下载脚本，没有 garage 等场景 mesh |

“只有 Docker client”已不是当前事实。本次没有安装系统软件、没有下载镜像层，也没有改变任何官方算法文件。

## 真正执行过的构建

完整规划器使用其原始 CMakeLists：

```bash
cmake -S third_party/official_baselines/tare_official/src/tare_planner \
  -B audit_results/tare_native_build_probe_20260910/planner_configure
```

实际返回 1：`find_package(catkin REQUIRED ...)` 无法找到 catkin 配置。并未越过 ROS/PCL 依赖阶段，不能据此记录“完整 TARE 编译成功”。

可独立运行的部分是原 `src/tsp_solver/tsp_solver.cpp` 和其原头文件。其依赖是源码已带的 OR-Tools 9.8.3296；`ldd` 验证共享库依赖均能在当前主机解析。探测工程只新增测试入口和外围 CMake，直接编译原文件并链接原随附库，没有复制改写求解器：

```bash
cmake -S audit_results/tare_native_build_probe_20260910/tsp_component \
  -B audit_results/tare_native_build_probe_20260910/tsp_build
cmake --build audit_results/tare_native_build_probe_20260910/tsp_build --parallel 1
audit_results/tare_native_build_probe_20260910/tsp_build/tare_official_tsp_probe
```

4 个测试均返回成功：单节点、双节点闭环、正方形闭环、带 dummy 节点的线形开放路径。测试独立检查输出节点排列及路径代价与 `getPathLength()` 一致，后两例分别为 4 米和 3 米。编译仅出现 bundled OR-Tools 中 `std::iterator` 的弃用告警；未改变原头文件消除告警。

这证明该官方组件和已打包的 C++ OR-Tools 在当前 CPU/ABI 下可以工作。它没有运行分层覆盖规划、传感器订阅、机器人导航或实际探索场景，也不是论文对照结果。

## 磁盘预算与为什么没有直接安装完整环境

| 内容 | 测量或预算 | 证据性质 |
|---|---:|---|
| 本次探测新增文件 | 约 1 MiB | 实际生成量，无新依赖 |
| `ros:noetic-ros-core-focal` amd64 镜像层 | 409,194,642 字节，约 390.24 MiB 压缩量 | 只读取远端 manifest，没有拉层；解包体积尚未测量 |
| 宿主 `libpcl-dev` + Eigen + glog | 283 个变更包，禁用 recommends/suggests，下载约 250.92 MiB，安装增量 1,181,589,504 字节，约 1.10 GiB | 当前 Noble 的只读 APT 求解；**不是 Focal 的精确预算** |
| Noetic/Focal 完整 planner 构建 | 暂按新增 3–5 GiB、开工前至少 6 GiB 空闲保留 | 保守工程预算，Focal 依赖尚未实际安装测量 |
| Gazebo + CMU 官方场景完整演示 | 暂按新增 6–10 GiB、建议至少 10 GiB 空闲保留 | 还含 Gazebo、渲染依赖、场景展开和编译缓存；尚未实测 |

远端镜像已固定到 linux/amd64 manifest：`sha256:4c1435fd85be3edde3820f0d132ab30a6b030209dce4fa3ac428a2aa5a763caf`。压缩大小不能当成 Docker 最终占用；没有现成镜像可复用，也无法证明完整 Noetic 依赖能落在本轮 1 GiB 增量内，因此未盲目拉取。上述未来预算必须在 Focal 容器的 APT 模拟和实际解包后再次核算，不能把估计数字写成已测量结果。

官方原生路线使用 Ubuntu 18.04/Melodic 或 20.04/Noetic，CMU 环境还需另下载约 500 MB 的场景。当前主机上的 ROS 2 指引不会自动补齐本次固定 ROS 1 源码的依赖。[TARE 官方说明](https://www.cmu-exploration.com/tare-planner)、[CMU 环境官方说明](https://www.cmu-exploration.com/)。

## 可执行的最小完整 planner 下一步

已准备但**没有执行** `Dockerfile.noetic-planner` 与 `noetic_node_smoke.sh`。它们只构建 TARE，暂不安装 Gazebo 和 CMU 场景：

1. 当单独存储预算允许时，先在 Docker 数据盘核对空闲量及共享层；本地已有镜像可以减少增量，但必须实际核算。
2. 使用已固定的官方 Noetic/Focal amd64 基础镜像。Dockerfile 安装 TARE package.xml / CMake 所需 ROS 消息、roscpp/rospy、pcl_ros、Eigen、glog 和编译器；在层内清除本次 APT 索引和下载包。
3. 仅将已验证的官方 `src` 放入容器，运行 `catkin_init_workspace` 恢复下载时省略的外部工作空间链接，再执行原 `catkin_make -j1 -l1`。不修改 TARE 原 CMakeLists 或算法文件。

在仓库根目录，后续构建命令为：

```bash
docker build --platform linux/amd64 \
  -f audit_results/tare_native_build_probe_20260910/Dockerfile.noetic-planner \
  -t nso-tare-noetic-native:44500592 \
  third_party/official_baselines/tare_official
```

构建上下文限制在已固定的 TARE 源码目录，避免发送整个项目及实验数据。镜像尚未构建，APT 可用性和依赖解算应以实际构建日志为准，不预先承诺成功。

构建通过后，可在无外网、1 CPU 容器中执行已准备的原节点存活检查：

```bash
docker run --rm --network none --cpus 1 --memory 3g \
  --mount type=bind,src=/root/NSO/audit_results/tare_native_build_probe_20260910/noetic_node_smoke.sh,dst=/probe.sh,readonly \
  nso-tare-noetic-native:44500592 bash /probe.sh
```

该脚本使用原 `explore_garage.launch` 的 `rviz:=false` 参数，并检查 `/sensor_coverage_planner/tare_planner_node` 响应。它没有输入传感器数据，只能算原节点启动检查，不能称为官方探索演示。脚本已通过 shell 语法检查，尚未在 ROS 容器运行。

## 官方 CPU/headless 演示的后续边界

CMU 固定源码的 VLP-16 宏默认 `gpu:=false`，使用 Gazebo CPU `ray` 插件，所以三维雷达仿真本身不要求 NVIDIA。完整 launch 仍有 Gazebo 相机渲染，而且 `system_garage.launch` 在 `gazebo_gui=false` 时依然无条件启动 RViz；“关闭 Gazebo 窗口”不等于完全无显示依赖。

后续可以在 Noetic 容器增加 Gazebo ROS、OpenCV、libusb、xacro、ROS 控制及必要的 Qt/渲染依赖，使用 Mesa 软件渲染和 Xvfb 保持原 launch 内容运行。启动形式可用 `xvfb-run -a env LIBGL_ALWAYS_SOFTWARE=1 roslaunch vehicle_simulator system_garage.launch gazebo_gui:=false`，再启动 TARE 的 `rviz:=false` 配置。此路径目前是待执行方案，软件渲染性能未验证，不保证实时性。

官方场景下载脚本会写固定 `/tmp/gcokie`；本轮没有运行它，也没有清理任何他人的临时文件。以后下载应在隔离容器自己的临时目录和独立场景存储中完成，确认空间后再展开模型。原生演示只有在观测话题、状态估计、terrain、规划路径和机器人运动共同产生后才算通过，不能用节点存活代替。

## 原生官方验证与本项目窄视场适配必须分别记录

原 TARE 的 `LidarModel` 是 360° 水平、24° 垂直，garage 配置量程 12 米；CMU 使用 VLP-16。小车的 ZED 深度视场和单线雷达不是该传感器模型的直接等价替换。

适配阶段需要将实际 ZED 深度点云转换为观测得到的 `/registered_scan`，将位姿同步到 `/state_estimation_at_scan`，并从观测生成 `/terrain_map` / `/terrain_map_ext`。随后检查相机方向、遮挡、候选 yaw 和视场条件下的信息增益；仅改 ROS 话题名不能解决原 360° 传感器假设。任何这种修改都应在独立适配版本留补丁，并共享本项目的深度、位姿、导航预算和 GT 评价。

因此要分别保留三项状态：当前通过的原官方 TSP 组件；尚未通过的完整官方 planner / CMU 原生演示；尚未实现的地面窄视场公平对照。当前 CPU3D 内部 `geometry` / `coverage` 对照不能因本次组件构建而被改名成官方 TARE。
