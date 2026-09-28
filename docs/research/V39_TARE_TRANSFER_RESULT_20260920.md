# V39 原始 TARE 节点迁移结果

已封存 4 个预注册条件，完成 4 例、失败 0 例；完成例均通过独立进程的固定主轨迹感知、TSDF 和指标复核。合计 4 次实际主实验、5 次固定轨迹复核尝试，其中原比较器序列化失败 1 次、成功 4 次。不是全部首次通过。失败保留 failed，未知质量为空，不代填 0；均值仅对完成例计算。

原 case00 主轨迹逐文件 SHA 复制到 r1，计为同一次主实验；没有重跑主策略。原失败重放的 39 帧传感与前缀/终点地图、mesh 比较已通过，但在质量评价前因 cue 字典整数键与 JSON 字符串键的直接比较失败。更正只使用无容差的规范 JSON 比较，保留全部物理字段；额外一次复核明示计入成本，原失败目录与封存仍在。

## 逐条件结果

| 条件 | 状态 | C_map | F1@5cm | J5 | 动作 | 返回 | native目标/adapter等待/adapter返航 |
|---|---|---:|---:|---:|---:|---|---|
| P00/h0 | complete | 0.807823 | 0.630463 | 0.509303 | 38 | 是 | 7/4/9 |
| P00/h1 | complete | 0.807823 | 0.668723 | 0.540210 | 38 | 是 | 7/4/9 |
| P01/h0 | complete | 0.821856 | 0.643906 | 0.529198 | 42 | 是 | 8/7/9 |
| P01/h1 | complete | 0.821856 | 0.712029 | 0.585186 | 42 | 是 | 8/7/9 |

动作归因不含共同 18 动作前缀；每例满足前缀 + 原生目标执行 + adapter 付费等待 + adapter 安全返航 = 实际动作。原生目标动作是到实际 native waypoint 的公共图 BFS 执行，不等同于原生控制器直接发出的离散动作。终止决定不计付费动作。

## 原生输出与接口保护

| 条件 | fresh/确认目标ID | fresh（前缀后） | 规划周期 | 投影拒绝决定 | 返航预算拒绝决定 | 原生finished | 提前停止 |
|---|---:|---:|---:|---:|---:|---|---|
| P00/h0 | 7/7 | 4 | 7 | 0 | 10 | False | True |
| P00/h1 | 7/7 | 4 | 7 | 0 | 10 | False | True |
| P01/h0 | 8/8 | 5 | 8 | 0 | 0 | True | False |
| P01/h1 | 8/8 | 5 | 8 | 0 | 0 | True | False |

fresh 是保存回执的实际新路点事件；规划周期按非空 scan 每 5 次累计。投影/预算拒绝按决定次数统计（含终止决定），不是独立目标数。native finished 与 adapter 提前终止分列，不能将安全返航归因于 TARE 主动完成。

- P00/h0 终止：`native_goal_not_return_affordable`；最后实际动作的选择原因 `native_goal_not_return_affordable`；控制器状态 `returned_no_affordable_action`；非法新路点 0 个，碰撞 0 次。
- P00/h1 终止：`native_goal_not_return_affordable`；最后实际动作的选择原因 `native_goal_not_return_affordable`；控制器状态 `returned_no_affordable_action`；非法新路点 0 个，碰撞 0 次。
- P01/h0 终止：`budget_end_no_policy_selection`；最后实际动作的选择原因 `native_finished_return`；控制器状态 `sensor_or_budget_end_returned`；非法新路点 0 个，碰撞 0 次。
- P01/h1 终止：`budget_end_no_policy_selection`；最后实际动作的选择原因 `native_finished_return`；控制器状态 `sensor_or_budget_end_returned`；非法新路点 0 个，碰撞 0 次。

## 可支持的结论与限制

该结果证明固定原始 TARE 原生节点已在同一虚拟感知—建图—评分链中实际运行，并保留真实路点、执行及返航证据。仅有两个已见布局、每个两构型的迁移测试，不能据此证明跨场景泛化，也不与 L1 的 SWAP-I/VISTA-I 拼接为完整 TARE 公平排名。

原生 360°×24°内部可见模型与真实窄视场 RGB-D 不同；surface/keypose/collision/occupancy 尺度及覆盖膨胀采用公开物理尺寸的一次性适配，其余点数阈值保留但有效面积意义随点密度改变。terrain 为公共平地 residual 近似；精确仿真位姿不涉及真实 SLAM 漂移。BFS 路点投影、每次等待最多 4 个真实付费右转与精确返航守卫属于 adapter，其影响不能全归因于原生 TARE 算法。

**重放范围：policy_reexecuted=false。** 原生 ROS 调度可能非确定，复核只按主实验记录动作重新获得传感、TSDF 和评分，不再次运行原生策略。汇总器核验主/重放 seal、冻结源码归档、共同输入 SHA、每个存储包的逻辑 SHA、地图摘要、mesh 结构与绑定到主结果的重放收据；不新建 World，不重新融合或评分，也不把复核当独立统计样本。fresh mesh 的逐数组比较由封存 runner 执行，汇总器只审其收据并读取主实验保存数组。

来源：`audit_results/v39_tare_adapter_r1_20260920`；逐步动作/停止原因和 mesh 审计详见 `manifest.json`。
