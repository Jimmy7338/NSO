# V39：完整 TARE 运行时恢复与传感桥接预检

日期：2026-09-20。范围：CPU、ROS 1 Noetic、固定官方完整规划器；本记录没有新增世界、传感渲染、TSDF 重建或 Q 评估，也不构成效果对比。

## 实际完成情况

已恢复并运行固定官方 TARE 完整节点，原生局部覆盖规划、全局图与 TSP 均来自原 ELF。没有以历史独立 TSP probe 代替完整规划器。零传感启动、ROS 话题注册、依赖解析、Python 消息导入，以及五个既有付费观测文件的消息传递均已实际通过。

- 官方源：`third_party/official_baselines/tare_official`，提交 `44500592b86138257273e0cab264e6a847ccefc7`；仓库 <https://github.com/caochao39/tare_planner>。
- 完整 ELF：`audit_results/tare_native_node_attempt_20260911/tare_planner_node`，1,133,016 B，SHA-256 `59fd9b818f4cc6a2edc3e5bf4a8641e2dad92b19ba1d824f1427624aecfd6f03`。
- 原 ELF 已有 2026-09-11 的 Focal/Noetic、官方源码未改动、单进程 catkin 构建记录。本次复用该二进制，未重新编译或修改原算法。
- 新桥接：`scripts/tare_ros_bridge_v39.py`，预检结束 SHA-256 `7b0ff56613efd3f1b2bee9385c9e90d3fa7d2ef7543d98ad881057f449908e51`。
- 完整收据：`audit_results/v39_tare_runtime_preflight_20260920/verification.json`；逐包 URL、版本与 SHA 在同目录 `installed_packages.json`。

## 环境、空间与下载核验

宿主为 Ubuntu 24.04 amd64，无现成 `/opt/ros`。Docker 29.1.3 daemon 可访问，但 images、containers、volumes 和 build cache 均为空；本次没有拉镜像或创建容器。`/tmp` 与仓库共用根分区，不能缓解持久盘紧张；`/boot` 不作为数据目录使用。

专属运行时位于 `/dev/shm/nso_v39_tare`，与其他任务的 tmpfs 文件隔离。从官方包中仅提取 57 个包，不运行 maintainer scripts、不改宿主 dpkg、HOME 或系统 Python：压缩下载合计 29,738,296 B，包声明安装量 185,470,976 B；含包索引、日志与 ELF 的专属文件总量为 255,476,045 B。下载的 `.deb` 校验并提取后逐个删除。已有 OR-Tools 库直接引用仓库原文件，不复制其约 101 MB 库目录。

主要运行组件为 Python 3.8.2、Noetic roscpp/rospy/rosmaster 1.17.4、PCL 1.10、Boost 1.71，以及消息定义与必要小型依赖。最终 `ldd_current.txt` 没有 `not found`，`python_import_check.txt` 确认 rospy、rosmaster、rosgraph、NumPy、YAML、PointCloud2、Odometry 导入成功。初期导入缺少 catkin 与 libffi7，补齐后通过；没有把初次失败当作最终成功。

包元数据采用官方签名链核验：Ubuntu Focal Release 用宿主 Ubuntu archive keyring 验签；ROS Release 用官方 `rosdistro/ros.key` 验签，再验证 Packages 索引 SHA-256、每个 `.deb` 的 SHA-256 与字节数。ROS HTTPS 在当前环境出现证书主机名不匹配，因此采用官方 HTTP 地址传输**已签名、逐级校验的数据**，未关闭 TLS 校验。签名输出、索引哈希、Release 和签名保存在收据目录。

环境恢复及桥接检查要求 MemAvailable ≥ 2 GiB、tmpfs 空余 ≥ 512 MiB；安装脚本还限制专属任务文件 ≤ 1.5 GiB。根分区另保留 ≥ 64 MiB。预检中父任务迁移了其自身可再生缓存，其他并行任务也有磁盘写入；本记录不声称它们属于 TARE 安装量。实际文件系统、内存、Docker 快照见 `host_snapshot.json`。本子任务未删除其他资料。

## 启动及接口

在仓库根目录运行以下命令；运行时 RAM 目录需要当前任务已授权的环境配置权限。

```bash
python3 -B scripts/tare_ros_bridge_v39.py --planning-wait-s 5
```

默认服务自行启动 `http://127.0.0.1:11339` 的独立 ROS master，端口占用即退出，拒绝接管其他 master。它依据 `environment.json` 重启到专属 Focal Python 3.8，加载官方 `garage.yaml` 的全部私有参数，启动 `/sensor_coverage_planner/tare_planner_node`。`--overrides-json /absolute/parameters.json` 可加载冻结的平坦参数字典；服务输出完整参数及其 SHA，参数适配必须在正式协议中冻结。

调用方必须等标准输出首行 `status=ready` 且 `native_initial_timer_observed=true`。该握手等待原生 1 Hz 定时器执行无里程计初始化分支，不提供额外观测。随后逐行写 NDJSON：

```json
{"op":"boundary","xy":[[0,0],[10,0],[10,10],[0,10]]}
{"op":"observe","packet_path":"/absolute/paid/000.npz","action_id":0}
{"op":"observe","packet_path":"/absolute/paid/001.npz","action_id":1}
{"op":"close"}
```

边界例子仅说明协议，正式运行应使用任务公开边界。`observe` 只接受上述三个键，动作号从 0 严格连续。每个付费 packet 仅发布一次 registered scan；不会将一帧复制五次来强制规划。原生 `RegisteredScanCallback` 每五个非空 scan 形成 keypose 并触发规划；在此之前 waypoint 返回 null。

输入只读取 depth、内参、相机位姿、单线雷达有效命中与标定，使用真实深度反投影和坐标变换。桥接不读取 RGB 类别、语义字段、packet metadata、假设、场景标识或 GT。单线雷达最大量程的无命中值不会被当成障碍点。

| 方向 | 话题 | 类型 |
|---|---|---|
| 输入 | `/registered_scan` | `sensor_msgs/PointCloud2` |
| 输入 | `/state_estimation_at_scan` | `nav_msgs/Odometry` |
| 输入 | `/terrain_map`, `/terrain_map_ext` | `sensor_msgs/PointCloud2` |
| 输入 | `/start_exploration` | `std_msgs/Bool` |
| 输入 | `/navigation_boundary` | `geometry_msgs/PolygonStamped` |
| 输出 | `/way_point` | `geometry_msgs/PointStamped` |
| 输出 | `/sensor_coverage_planner/global_path`, `local_path`, `exploration_path` | `nav_msgs/Path` |
| 输出 | `/sensor_coverage_planner/exploration_finish` | `std_msgs/Bool` |

waypoint 的 `frame_id=map`，x/y 是传感 packet 的全局世界坐标；z 是原生规划器输出，不保证恒等于相机高度。调用方负责在**公开、观测得到的**导航图上投影目标与执行动作。桥接没有替代局部/全局规划器，也没有自己的探索后备策略。`fresh_waypoint` 表示此次收到新目标；相邻原生规划周期之间允许保持已确认目标。超时、空目标和提前结束需要如实记录，不能冒称正常探索。

地形适配使用实测 XYZ，并以 `abs(z-public_ground_z)` 作为 intensity，公共平地高度默认 0。它不是完整 CMU terrain-analysis 实现，也没有为未见区域生成平面或点云，应在比较限制中披露。

## 实际验证与启动竞态修复

无传感 smoke 实际退出 0，确认完整原生节点存活并注册主要输入输出；当时发布观测数为 0。

随后只读取 `audit_results/v36_online_confirmation_20260918/case00/packets/000.npz` 至 `004.npz`，共五个已有文件，各自 SHA 保存在收据。前四帧没有行动目标，第五帧原生节点生成目标。初次测试虽退出 0，但发现 `exploration_finished=true`：源码显示 `start_time_` 仅在首次尚无 odometry 的定时器分支赋值，原桥接过早 ready 允许立即发送 odometry，可能令时间原点保持 0。

修复只涉及桥接启动握手与剔除未观测初始目标，未改 TARE 算法。修复后的同五帧检查退出 0、总耗时 4.674 s；动作 4 获得 fresh waypoint `(-5.15685424949238, -5.15685424949238, 0)`、frame `map`，`exploration_finished=false`。前四帧均无目标。原 garage 的 8 m lookahead / waypoint 延伸使输出可能超出这个小场景，下一步需要公开几何约束下的参数适配和动作投影。

首次和修复后的收据均保留，分别是 `saved_five_observations_receipt.json` 与 `saved_five_initialized_receipt.json`。这些检查是**保存数据的传输/节点集成测试**，不是在线导航，不增加旧 V35/V36 实验配额，也不是优越性证据。Python 日志配置搜索提示仍存在，原生 stdout/stderr 独立保存；该提示没有被隐藏。

## 无编译适配与后续构建边界

可由 ROS 参数调整的是量程、采样网格、碰撞尺度、传感高度、lookahead 等。当前公开协议已有相机量程 4 m、单线雷达量程 8 m、相机高度 0.9 m、导航网格 0.2 m，可据此制定一次物理尺度适配；不能根据 h0/h1 或 Q 结果搜索参数。

无法通过参数变成 ZED 视场模型：`include/lidar_model/lidar_model.h` 中水平 360°、垂直 24°、角度量化及数组尺寸为编译期常量；`viewpoint_manager.cpp` 还硬编码 `tan(pi/15)` 的垂直视场。位置候选没有对应的相机朝向计划，仅修改常量也不能得到正确的朝向约束。正式相机模型需要源码修改、重编与单独验证。当前完整节点只能称为保留原 360° 模型的传感/场景迁移基线。

搜寻未发现历史完整 core 的 `.a` / `.o`；仅存在独立 TSP probe 的目标文件。因此“只重编一个受影响 object，再链接原完整 core”目前不可执行：原 ELF 不能作为静态对象库使用。完整目标由约 18 个源文件组成，若继续源码适配，需要重编这些核心目标。

宿主已有 g++ / CMake。根据签名包索引，PCL-dev 本体 11,047 KiB、Eigen 6,993 KiB、Boost headers 134,951 KiB、FLANN-dev 11,908 KiB；另需小型 ROS PCL/tf/visualization 消息头文件。直接按所用源码编译、复用已有运行库，预计可绕过完整 `find_package(PCL)` 拉入 VTK/Qt/OpenNI 的大依赖树；这尚未实际编译，不能标为成功。若必须精确复现 GCC 9，g++/gcc/cpp/libgcc-dev/libstdc++-dev 五包额外约 115,502 KiB，仍需少量编译器运行依赖。后续应先冻结相机朝向适配定义，再恢复所需头文件与单线程构建，不在正式对比期间修改算法。

## 恢复路径

可执行恢复入口为 `scripts/restore_tare_runtime_v39.py`。只依赖宿主 Python 标准库、gpg/gpgv、dpkg-deb 与 ldd，重启后即使研究虚拟环境的 RAM 依赖已消失也可运行。安装顺序、57 个包的精确版本/URL/SHA 与 ROS 官方 key 的 SHA 均固定在脚本及收据中。它重新验证官方签名 Release 与索引，再逐项匹配冻结 recipe；仓库不再提供该精确版本时明确停止，不默默换版本。随后自动提取包、清除下载临时包、恢复固定 ELF/环境/目录，检查 ldd、Python 导入和零传感完整节点启动。

```bash
python3 -B scripts/restore_tare_runtime_v39.py
python3 -B scripts/restore_tare_runtime_v39.py --verify-only
```

第一行完整恢复，第二行仅离线核对现有环境。已有相同包收据时默认跳过重复解包，若本地内容损坏可加 `--force-reextract`。检查端口已占用时不接管已有 master，可用 `--port` 指定另一个仅用于 smoke 的端口。正式 adapter 使用 11339。库搜索路径引用仓库已存在的官方 OR-Tools；首次探索式恢复脚本 `runtime_restore_used.py` 仍保留作历史收据。

该脚本已经在**此前不存在的** `/dev/shm/nso_v39_tare_restore_check` 从空状态执行，使用独立端口 11349：

```bash
python3 -B scripts/restore_tare_runtime_v39.py --runtime /dev/shm/nso_v39_tare_restore_check --port 11349
```

完整下载/签名/哈希检查、57 包提取、依赖/导入和原生零传感启动均通过，退出 0，总耗时 82.299 s。新脚本只保留选中包元数据，无需保存原探索式安装的 73 MB 全包字典；验证目录文件量为 183,164,552 B。检查结束保存 `clean_restore/` 收据后仅删除该新验证副本，释放 RAM；主运行时未改动。恢复脚本 SHA-256 为 `1d1de827277bb09b04113c969d8eda1a3f03f7a759fc08348563a8a9dc6b520f`。该运行时可再生，重启主机后 tmpfs 会消失；小型持久化证据不依赖其存活。

下一步在线 L2 对比必须使用新的独立协议、执行配额、冻结参数和传感预算；已完成的本预检不授权额外重跑旧冻结实验。
