# 官方 TARE 完整规划器构建与节点接口验证

固定的官方 TARE 源码（`caochao39/tare_planner`，commit `44500592b86138257273e0cab264e6a847ccefc7`）已在 `linux/amd64`、Ubuntu 20.04、ROS Noetic 的 CPU 容器中完成原生构建。未修改官方算法源码；`catkin_make -j1 -l1` 返回 0，并生成 `tare_planner_node`。保存的二进制 SHA-256 为 `59fd9b818f4cc6a2edc3e5bf4a8641e2dad92b19ba1d824f1427624aecfd6f03`。

第一次分层 Docker 镜像构建在安装依赖层完成后因磁盘空间不足而无法提交，且尚未进入规划器编译。清理该失败构建后，改为一次性容器、只读挂载官方源码、把日志写回宿主，避免复制镜像层；第二次构建成功。这个过程证明先前宿主机 CMake 失败的直接原因是缺少 catkin/ROS 依赖，而非固定源码不可编译。

节点测试也保留了三个阶段。第一版夹具因在加载 ROS 环境前启用未定义变量检查而在启动前失败；修正外围夹具后，节点存活测试通过；再增加 `/registered_scan` 接口注册门后，最终测试返回 0。节点实际注册了 `/registered_scan`、`/state_estimation_at_scan`、`/terrain_map`、`/terrain_map_ext` 等输入，并发布 `/way_point`、全局/局部/探索路径、运行时间及探索结束状态。

本结果只证明官方规划器在固定 ROS 1 CPU 环境中可编译、可启动且接口完成注册。测试没有输入传感器数据，没有完成导航或探索回合，也没有建立与 NSO 的同场景、同传感器、同时间/路程预算比较，因此不能支持“NSO 战胜 TARE”。下一步需要实现从共享虚拟传感器记录到 TARE 输入话题的适配器，再以相同轨迹执行器和 GT 评价器运行成对实验。

结构化结论见 `verification.json`；完整构建、节点启动和接口列表分别见 `dependency_and_build.log`、`node_smoke.log`。首次夹具失败和只检查存活的中间结果也原样保留。
