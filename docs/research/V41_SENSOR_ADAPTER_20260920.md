# V41 R0/R1：连续姿态运动与 CPU 传感适配

本实现位于新文件 `env/development_sensor_v41.py`，不修改 P0/P1 封存源码、配置或资产。本轮只运行小型解析三角形、噪声、碰撞和计费 fixture；没有创建实际 DEV World，没有生成开发场景传感轨迹或新研究样本。

## 稳定 API

```python
camera_transform_xyyaw(pose_xyyaw_rad, height_m=.9) -> T_world_camera
propose_motion(pose, action, translation_step_m=.25, rotation_step_deg=30.) -> next_pose
swept_circle_collision(start_xy, end_xy, obstacle_polygons, radius_m=.2) -> bool
paid_motion_transition(pose, action, *, paid_step, max_actions, start_pose,
                       obstacle_polygons, radius_m=.2, ...) -> receipt
return_pose_matches(pose, start_pose, *, position_tolerance_m=1e-8,
                    yaw_tolerance_rad=1e-8) -> bool
first_hit_triangles(origins, unit_directions, vertices, triangles,
                    *, max_distance_m, ...) -> radial_distance, triangle_index
render_rgbd_arrays(vertices, triangles, *, intrinsic, world_from_camera,
                   width, height, ...) -> rgb, axial_depth
render_planar_scan(vertices, triangles, pose, *, maximum_range_m=8., ...) -> PlanarScan
storage_report_v41(persistent_output_root, expected_batch_peak_bytes) -> report
create_development_sensor(asset_dir, public_spec, *, episode_id, noise_seed,
                          persistent_output_root, expected_batch_peak_bytes, ...) -> DevelopmentSensorV41
```

`first_hit_triangles` 及 decal helpers 是渲染器私有工具；三角形 ID、实际实例 ID 和真实标记面片不能进入 planner。`render_rgbd_arrays` 只返回 RGB 与深度。实际 adapter 返回 `SensorStepV41(rgbd, scan, receipt)`，其中 `rgbd` 严格为已有 `PaidRGBDObservationV40` 六字段：frame_id、paid_step、rgb、depth_m、intrinsic、world_from_camera；没有 owner、隐藏类别、实例真值、真实构型或 GT metadata。

## 姿态与付费动作

连续位姿是 `(x_m, y_m, yaw_rad)`，yaw=0 面向世界 +X，正方向逆时针。forward 沿当前朝向前进 .25 m；turn_left=+30°，turn_right=-30°；observe 原地采集。适配不依赖旧 1 m 网格或 90° 方位。

构造器不自动观测。调用 `initial_observation()` 明确领取一次公共起始帧：paid_step=0、动作成本 0、`initial_frames=1`，RGB-D 与扫描计数单独记录。所有方法享有相同初始 grant；不能重复调用它获得免费补看。

此后每个 forward/turn/observe 都消耗一个动作预算，并输出一组 RGB-D 和单线扫描。碰撞的 forward 尝试也扣 1，保持原姿态并记录 collision；不能免费重试。动作名非法或预算耗尽时拒绝执行。close 只关闭适配器，不生成观测、不自动返航、不重置或传送姿态。

返航要求 XY 与 yaw 都回到声明起点容差内；只回到起点位置而朝向不同不计完整返回。没有隐式免费返航，规划器/执行器必须为实际移动和转向预留预算。扫描时间戳使用 paid_step 对应的名义秒刻度，不是墙钟实测运行时间，不能拿它比较实时性能。

## 相机与雷达

相机 optical-z 向前、x 向右、y 向下；高度按公共 motion 参数。当前 DEV factory 固定使用 P0 的 96×72 图像、K=`[[48,0,47.5],[0,48,35.5],[0,0,1]]`、**轴向深度 .1—4 m**、.25 m/30° 动作与径向 8 m 雷达；不满足则拒绝，不能把旧配置默默混用。只支持精确模拟里程计；有噪位姿尚未实现，应明确报错。

相机先计算全三角网格第一命中，之后按 optical-z 裁剪。角落像素可以拥有大于 4 m 的径向射线长度而轴向深度仍小于 4 m；不能用径向 4 m 截断它。小于 .1 m 的近物体仍遮挡后面物体，不能跳过近物体去看后方。未命中或无效深度为 0。

标记只覆盖实际第一命中点落在正确实例的已有实体标牌面片内、且从正面观看的像素。被遮挡、面外、悬空或错误 owner 的标记不会着色。中性材质与小型人工 RGB decal 分开；不把整设施涂成类别颜色，也不输出仿真分割标签。这是受控人工标记，不是自然语义网络。

深度噪声为 `z_noisy=z_clean*(1+sigma*N(0,1))`，每帧从独立传感 seed、paid_step 和固定流标识派生 RNG，既不消耗场景几何 RNG，也不改变全局 NumPy RNG。固定 seed/step 可复现；改变 seed 或 step 改变噪声。噪声后超出有效区间的像素置 0，不恢复原本无效深度。RGB 保留该实际第一命中的颜色，不因深度噪声伪造类别。

单线雷达默认为 .3 m 高度、360 条射线，角域 `[-pi,pi)`、激光 x 前/y 左/z 上；量程是实际径向距离，未命中返回 8 m。本版没有激光噪声，不应声称已匹配实验室实物噪声。

## 任意方向线段碰撞

圆足迹沿整条连续 XY 线段扫掠，使用线段—凸多边形的最小距离与端点包含关系；不是仅检查终点，也不限轴对齐移动。相切按碰撞处理。

实际 P1 设施使用保守 AABB 足迹，背景盒排除 `zmax<=0` 的地面再投影为障碍。这会保守禁止一些货架下方或凹腔穿越，应作为模型限制披露；它不提供 GT 免费给 planner。碰撞器的真实足迹仅位于私有 renderer/executor，规划器使用另行声明的公开导航输入。

## 硬资源门与本轮状态

`DevelopmentSensorV41.__init__` 的第一步执行：

```text
required_free_bytes = max(10 GiB, 2*expected_batch_peak_bytes + 2 GiB)
```

检查目标持久输出路径所在文件系统的实际空闲空间；tmpfs、ramfs、devtmpfs 及未知文件系统不通过。不存在的输出目录只检查其最近存在父路径，不创建目录。factory 与直接构造都走同一道门，没有 skip、test 或 RAM 绕过参数。

门失败时抛出 `ResourceGateBlocked`，状态为 **blocked_before_world_creation**，同时保留 free/required/filesystem/reason 报告。失败发生在读取实际 DEV mesh、marker、实例清单、构造 World 或采集帧之前。已有实现工作与解析测试可以继续，因此不把整个项目任务标记为 blocked。

本轮开始时主机约 0.55 GiB 可用，低于 10 GiB 最低门槛，实际 DEV 运行不准入；实时字节数以 `storage_report_v41` 保存的报告为准。资源门是前置检查，不是磁盘配额或未来空闲空间保证；后续 runner 仍需按批次核实持久空间并保存失败记录。正常计算中的数组内存并不取消持久证据容量要求。

## 验证与尚未完成项

```bash
PYTHONDONTWRITEBYTECODE=1 python3 -m unittest discover -s tests -p test_development_sensor_v41.py -v
```

测试仅使用显式解析平面/三角形与纯运动函数；实际构造器测试使用远超任何磁盘容量的声明峰值，验证在真实资源检查中先失败，并监视资产加载函数不得被调用。它没有把资源门 mock 成通过，也没有实例化实际 DEV World。

覆盖 optical-z/径向距离区别、近遮挡、first-hit、物理 decal、防 owner 泄漏、噪声种子隔离、雷达方向、连续斜向碰撞、转向/观察/撞击计费、完整返航姿态、预算耗尽与两种构造入口的硬 gate。当前不宣称 DEV 运行器闭环通过、真实噪声匹配、SLAM 漂移得到验证或任何方法优势。
