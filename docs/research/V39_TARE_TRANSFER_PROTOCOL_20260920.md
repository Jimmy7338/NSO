# V39 L2 原始 TARE 节点的地面 RGB-D 迁移协议

本协议在 L2 主实验前冻结。输出独立于 V36 和 V39 L1；仅比较同一公共输入、预算和测量链下的迁移结果，不把原 TARE 的宽视场内部传感模型称为公平的窄视场排名。本轮不根据 F1、隐藏构型、语义后验或主实验效果调参。

## 冻结对象与运行单元

原始仓库提交 `44500592b86138257273e0cab264e6a847ccefc7`，完整原生 `tare_planner_node` SHA256 `59fd9b818f4cc6a2edc3e5bf4a8641e2dad92b19ba1d824f1427624aecfd6f03`。`scripts/tare_ros_bridge_v39.py` 启动原节点，通过 ROS 1 发布当前付费观测并读取实际 `/way_point`；不使用自编覆盖策略替代节点输出。启动必须确认原生 1 Hz 初始化计时器已运行，初始默认 waypoint 不作为探索目标。独占 ROS master 端口 11339，逐例退出，无并行 ROS master 复用。

`TareExperimentV39` 继承已冻结 `run_external_routes_v39.ExperimentV39`。P00/P01 各 h0/h1，共 4 个主实验；最多 4 个独立新进程的固定轨迹重建复核。每例 42 个付费动作上限，前 18 动作为共同前缀，初始观测另记为 0。允许返航后提前停止，不人为用满预算。主实验失败计为已消耗开始次数，保存原因，不为提高性能重跑。

单例：观测保存包→原生 ROS 发布一次→实际原生 waypoint→公共安全图上的 BFS 执行→下一次真实观测。禁止重复发布同一包推动 native keypose；禁止从未来帧、隐藏网格、真实目标表面、F1 或语义类别生成探索目标。共用控制器的后验只用于被动收据，不参与 TARE 选择。

## 公共尺度的一次性参数适配

覆盖 `configs/virtual3d/v39_tare_transfer_overrides_20260920.json` 所列项，其他 `garage.yaml` 项保留：3D 有效量程 4 m；40×40 视点阵列的 x/y 间距 0.2 m（公共栅格）；相机离公共地面 0.9 m；碰撞水平边界 0.2 m（公共机器人半径）；lookahead 与 waypoint 两种延伸均 1 m（单次平移动作）。surface cloud leaf 0.04 m 对应公共 TSDF 体素；keypose/collision leaf 0.1 m 为半机器人半径；rolling occupancy xyz 体素 0.2 m 对应公共二维格；coverage dilation 0.2 m 是一格的覆盖邻域平滑，**不是碰撞半径**。

原始 min-add-point、frontier cluster、coverage 状态切换等数量阈值不变。但降采样密度变化会改变这些数量阈值的有效面积意义，因此仍存在迁移差异；不得声称与原论文完全等价。原生硬编码水平 360°、垂直 24°（并存在 `tan(pi/15)` 比率实现）保留，和真实窄 RGB-D 观测不同；原生可对未真正看见的方向高估覆盖。

## 数据接口及执行约束

bridge 仅反投影已付费 depth，以已知 K/T 和里程计位姿投至 map；合并当前实际单线雷达命中。RGB 与 semantic 不输入原生决策。terrain intensity 使用距**公共平地 z=0** 的残差，原生 `intensity>.5` 判非地面；这是平地 residual 适配，不是完整 terrainAnalysis，也不读隐藏地形。位姿为仿真精确里程计，未评价真实 SLAM 漂移鲁棒性。

waypoint 必须 map 坐标系且 xyz 有限；新无效目标清空旧目标并记录拒绝。无新消息可保留最后有效原生目标。只用 x/y 执行；z 值记录而不强制等于相机高度。减去公共 translation 后投至最近可达公共节点，先按欧氏距离、再按 BFS 动作代价、再按固定节点序打破平局。最大投影距离为 1 m 动作网格单元的外接圆半径 `sqrt(0.5)` m；超出即拒绝，绝不代替生成探索目标。公共图路径是执行适配，不是另一个全局信息增益算法。

原生每 5 个非空实际 scan 产生 keypose。到达已有目标或尚无目标时，若距离下个 keypose 周期不超过 4 个实际观测、且右转后仍可精确返航，则最多执行 4 个**真实付费原地右转**等待；每步传感、建图、计预算。下个规划周期仍无可执行目标，或返航预算不足，则安全返航。这些固定等待方向归属 adapter cadence，不声称由 TARE 选出，不复制旧 scan。原生主动 `exploration_finished` 直接返航；adapter 拒绝/无进展/预算返航与原生完成分别报告。

## 测量、保存及复核

沿用 L1 的实际噪声 RGB-D、CPU TSDF、预测 ROI 提取、J5、F1@5 cm、覆盖面积分母与前缀/终点测量；离线 GT 仅在预测封存后加载评分。每例保存传感包、trace、原生 ready/参数/ack、投影与拒绝原因、付费等待次数、控制器与地图/mesh 摘要。native 退出并收集日志后再封存，不能在 seal 后写入 case。

原生 ROS 计时可能非确定。独立进程复核按主实验**已记录动作**重新传感并重建 TSDF、比较每帧/地图/mesh/指标；不再启动原生策略。报告 `policy_reexecuted=false`，只证明固定轨迹的感知、融合和评分复现，不能叫完整策略重复实验。4 例仅为两个已见布局的描述性迁移证据，不做大规模泛化推断。

主实验前先使用现存 P00 case00 的 19 个 prefix 包验证真实节点接口，不创建 World、不增加传感查询、不重建 TSDF、不计算质量。该 smoke 的观察收据和源 SHA 独立保存；它只检验接口，不能用其输出宣称性能。冻结同时记录 bridge、adapter、覆盖参数、原始 garage、预检 `installed_packages.json`/`environment.json`/`verification.json` 与共同输入 SHA，以定位 Noetic/PCL/原生可执行依赖。
