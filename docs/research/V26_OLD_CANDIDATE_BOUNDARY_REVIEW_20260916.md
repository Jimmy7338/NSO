# V26 旧候选与观测边界只读审查

2026-09-16。仅阅读源码；未运行传感、运动、TSDF、候选实验或指标评估，未修改任何旧源码。本文件是新版本设计建议，不是已完成修复或效能证明。

## 结论

必须区分三件事：V20 的覆盖候选、V16.3 的几何设施候选、V18/V19/V21 的设施质量评分。前两者的发现过程未用类别或人工标记位置作为筛选条件；第三者明确用人工标记识别任务相关设施。因此不能把旧 G 整体称为“不含人工标记的几何对照”，也不能反过来指认其几何候选全是标记泄漏。V25 的 GT 固定采集目录另属实验准备，不能与这些在线候选混为一谈。

## 实际调用链与证据

1. `facility_runtime_v21.py:76` 调用 `hierarchical_options_v16_3.generate_options(AxisHistoryViewV16(...))`，与 `ObservedRouteSpaceV20.coverage_candidates()` 合并；资产角色在 N/G 池中共同保留。它不读取 V25 `route_catalog`、`service_regions`、真值 owner 或隐藏附件。随后仅按当前观测安全连通分区过滤。
2. `facility_candidates_v20.py:31` 的路径空间读取 `mapper.belief`、公共地图尺度、机器人半径、当前位置、已声明返航锚点和剩余动作数。`coverage_candidates()` 由未知格、可达已知自由格边缘、可见未知格数量及空间分散规则产生候选；没有标签、marker 或设施定位输入。
3. 几何资产路径为 `hierarchical_options_v16_3.py:61 _geometry_assets` → `observed_asset_axes_v16.py:47 measured_assets_v16` → `competition_candidates_v8_1.py:18 measured_assets` → `response_features_v7.py:55 observed_patches`。最后一层先从 `quality_evidence.point` 投影为二维占据格，闭运算、连通聚类，再按支持点和几何跨度筛选；在第 96 行以后才读取 label 2/3，附加 `marked_points/class_vote`。没有用标签决定点属于哪个聚类或接受哪个聚类。
4. `_geometry_assets()` 只保留 group、观测点与边界、轴、宽深高及支持量，去掉上述类别/marker字段。V16 轴修正从实测法向及有深度一致支持的历史相机定向，再吸附到四个主轴；它替换 V8.1 最早相机定向，不读取真实物体朝向。`AxisHistoryViewV16` 提供完整已付费观测历史，不是假设新视图。
5. **真正人工标记入口**：`facility_runtime_v18.py:24` 的 `target = marked_points > 0`，以及 `facility_runtime_v19.py:36` 的 `marked` 索引。V19 方向项和 precision 项只对 marked 资产求和，再除公共任务数量；V21 G 继续使用这两个项。去掉标记但保留相同深度，代码上仍可产生同样的几何候选，但这些设施质量奖励会被去掉。交换 2/3 不会改变 marked 门，所以只做类别交换测试不足以发现这种 objectness 输入。`FacilityRuntimeV18.install_pilot_option()` 第 60 行也用 marked 筛出 A/B 目标；V21 自身禁用该旧干预入口，但未来不可直接拿它充当纯几何发现器。
6. `cpu_four_modules_v16.update_semantic()` 还把标签 2/3 的位置写入 density；所审 V21 候选/拓扑路径使用 belief 连通域，并未以 density 生成目标。不能仅凭变量名将其描述为候选泄漏。

## “几何候选”仍包含的假设和限制

- `observed_patches` 硬性要求 0.2 m 栅格，0.15 m 质量体素，至少 8 点、水平跨度不超过 2.2 m、高度跨度至少 0.25 m；3×3 二维闭运算可把邻近设施/隔板合并，导致较大聚类被过滤。它不是已验证的真实实例分割，group 也不是稳定物体 ID。
- `quality_evidence()` 只取高度 0.12–1.8 m，最多均匀抽样 10000 点；方向/范围来自这些已见点，5%/95% 分位边界不是完整外形或背面真值。`QualityMapperV2` 的点选择、位置、法向和抽样不按类别筛选；label 另行记录。
- V16.3 的 entry/deep 是沿观测轴偏移 0.4/0.8 m 的通用后侧假设；corner 是可见面的边缘接近位姿；aperture 是观测边界上的 5×5 假设平面。其 FOV 支持不证明隐藏附件可见。V17 已知 TSDF 遮挡可否定一部分支持，但未知射线仍不是可见性证书。这些可以是公开的几何先验，不能称“检测到了真实隐藏检修点”。
- 名为 `SemanticHistoryMapperV3` 的父类只保存 keyframes，不会自动实例化同文件的 `ObjectCompletionModel`。不能因为 import 名称就断言该 mapper 做了类别形状补全。
- `ObservedRuntimeMapperV10` 保护当前格的既有占据、短雷达命中和深度障碍，避免被旧 mapper 的原点清空覆盖。其导航和 TSDF 几何不按标签更新。不过 `frame.semantic` 作为传入数组被保存，协议只校验格式，并不认证标签来源；驱动仍须证明它来自真实已见 RGB 或显式识别器。当前人工色标解码不能称开放词汇识别。

## 可保留的 observed-only 路径与返航组件

`ObservedRouteSpaceV20.route/refresh`、`route_coverage_v2.orientation_graph` 和 `ObservedExecutionGuard` 可以独立于资产/类别使用。未知和占据都挡机器人，地图外边界也挡；圆形足迹膨胀包含栅格面积余量；图开放每个安全格的全部左右转和朝向前进，动作单位成本为 1。路由约束出程加返航不超过剩余预算，并恢复锚点朝向；不把预定返航当成已获得观察。

只生成一次静态路线还不够：`cpu_four_modules_v10.py:397 assess_action` 逐动作检查最新足迹和**动作后**剩余预算内返航；`plan_return()` 使用当前 belief。当前格/锚点未知或不安全、连通断裂、剩余预算不足都必须失败，不可清空这些格“救活”路径。`RecoveryRuntimeV14` 可以有限重规划，仍经过同一守卫。这保证的是当前离散观测地图内的规划约束，不保证真实定位误差、漏障碍或后续新障碍下绝对安全。

另有三个复用限制：覆盖预测仍用 `max_depth_m` 作为 360° 雷达距离，不能直接适配独立远距雷达；`route_mask` 对出程未知格并集去重，但未知障碍仍可能让预测过高；V16.3 原覆盖内部代理把 radar/camera 求和可能重复，不过 V21 已替换这些覆盖候选并重新计算 V20 并集。几何采集/安全路径不依赖 fixed-ROI 进度账本；该账本只是覆盖时间预留代理，不能当评价 C 或可通行真值。

## 下一版最小纠正建议（本次未实施）

1. 让 G 的发现/评分只消费几何视图：belief、已付深度/位姿、实测点/法向/历史方向和观测不确定性。移除 G 的 `marked_points` 任务筛选，或另列“具有共同人工目标标识”的 objectness 对照；不能只删 `class_vote` 后仍称完全无标记。保留从几何缺口、遮挡边界、稀疏观察及前沿自行提出补看目标的能力。
2. S 可以用实际识别到的类别/区域先验追加或重排候选，但目标位置必须来自已见检测/深度关联，未见部分明确为带不确定性的假设，不得输入 owner、任务评价窗、隐藏附件或 GT 服务视点。主端到端比较允许 G/S 自己发现候选；“同合法候选池只比排序”另作机制消融，不能替代候选发现价值。
3. 保留上述导航/返航组件和共同传感、几何后端、预算契约。将观测簇与语义注释拆成独立数据结构，来源审计至少区分 `category_used`、`marker_objectness_used`、`observed_geometry_used` 和 `truth_used`。下一版的小型反例应包括：标签交换、全部标记消失而几何不变、可见无标记对象、遮挡墙合并/超跨度、无安全返航。前两者要分别检验，且不可用删除 G 发现机会来保证 S 赢。

本审查仅支持定位上述输入边界及复用范围，不改变已有负结果，不增加新的物理任务，不宣称修正后语义必有增益。
