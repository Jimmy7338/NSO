# V41 实测支持关联合同（R0/R1 接线）

新增实现 `nso/observed_instances_v41.py`；测试 `tests/test_observed_instances_v41.py`。V40/P0 及 V35/V39 封存源码不变。本轮只有解析 RGB-D 合同测试，不创建 World、执行正式策略或宣称多设施语义增益。

## 输入与稳定 API

`ObservedInstancesV41(palette=..., structure_names=..., class_structure_prior=..., mode='S'|'G', geometry_prior=None, ...)` 复用 V40 配置校验与严格 `PaidRGBDObservationV40` 六字段包：`frame_id, paid_step, rgb, depth_m, intrinsic, world_from_camera`。不接受 GT owner、真实设施清单/边界框、真位姿/尺寸、未来帧或预制实例mask。类别 RGB 与结构先验表独立配置；支持不少于2种结构、一个类别多种结构和多个类别共享结构。

`observe(observation)` 要求原始 ID 未使用、paid_step 从0连续；原frame_id重传和非连续step拒绝。包 SHA 使用 `observation.sha256()`，另对实际 RGB/depth/K/T 内容计算不含 ID/step 的 digest。新frame_id与连续step下，即使像素和K/T完全相同也接受，并标记 `duplicate_measurement=true, no_new_support=true`；静止无噪声观测或全缺测帧可能合法重复，数据层不能仅靠像素相同判定动作非法。重复测量不创建实例、不扩展支持、不增加类别证据或几何反馈资格；当前关联仍绑定新包SHA，不能把旧源帧换别名取得新分数。router/sensor必须保证每个新step之前有计费动作并记录原始包来源，该账本不能单独证明外部观测是否付费。near/far 默认0.1/4 m，**均按 optical axial z**，不沿用 P0 的径向裁剪。

结果 `accepted[]` 每项提供 `instance_id, frame_id, paid_step, observation_sha256, pixel_indices, points_world_m, support_sha256, marker_pixel_indices, novel_support_voxels, geometry_feedback_eligible, association`。像素为当前有效深度区域的行优先 flat indices，递增且与世界点一一对应。`marker_pixel_indices` 仅为当前确实可见的已声明色标，允许空。`marker_anchor_world_m/marker_observation_sha256/marker_frame_id/marker_paid_step` 始终追溯首次已付marker，不假设其是物体中心或真位姿。接口不输出伪造的GT法向；下游需自行从实际marker像素拟合平面并记录误差。

公开 `support_digest(points_world_m)`：转换为 float64 连续数组[N,3]，以 `dtype.str:shape:` 头加 C-order bytes 计算 SHA256。下游残差caller可用实际包K/T与pixel_indices重算点和hash，拒绝篡改mask或点。`snapshot().instances[]` 输出 `anchor_world_m, support_points_world_m, support_sha256, structure_names, geometry_prior, active_structure_prior, geometry_log_scores, structure_probabilities` 等；`geometry_snapshot()` 是明确的无语义条件视图，供同包 S/G 对照。

## 从色标发现到无色标侧面

1. 仅用有效深度反投影到世界坐标。对相邻像素做4连通分组，只有实测3D距离≤0.15 m才相连；不是按类别分组。原始连通区域任一世界轴跨度>2.5 m即拒绝，避免一次接受大面积房间背景。
2. 首次发现必须有唯一人工marker连通片（默认≥4像素）和至少16个有效深度像素。锚点是marker实测点中位数，初始支持只收距锚点≤0.6 m的当前点。多个近接bootstrap区域整批拒绝，没有按遍历顺序选一个赢家。
3. 后续区域不要求当前marker可见。与每实例已有实测支持计算欧氏最近点距离；≤0.12 m的重叠点至少4个且占区域≥10%，才可关联。多实例均达到最少重叠点时拒绝；多个区域抢同一实例也整批拒绝，并暂时撤回语义条件。下一次唯一清晰关联可恢复。
4. 只收距离此前支持≤0.4 m的当前点，限制单帧扩展；同帧其他区域刚加入的点不能作为本帧的新关联证据。完全无重叠的背面不会被色标身份或oracle补绑定；需要中间真实视角形成连续实测支持链。靠近既有支持却重叠不足的marker也拒绝，避免创建便利的重复身份。

这些是预设保守几何门限，尚未经过自然设备数据标定。连通性和近邻重叠不等于可靠物体分割：接触物体/地面、相邻设施、深度噪声和极大设施可能误分或被拒绝；本接口不声称自然语义实例分割已完成。完全断开的新面暂不支持，需要后续显式观测与更可靠实例关联，不能从GT补齐。

## 支持去重与真实反馈入口

支持使用0.05 m世界体素，每体素保留首个实测点；每实例最多4096点。新点须距已有支持>0.05 m，至少4个新体素且占当前点≥5%，并实际加入支持，才取得一次新证据资格。每实例最多32个证据帧。新视域原地转向可以获得资格；仅改变朝向而观察同一支持、重复颜色、微小位姿抖动均不会无限增强信念。这比将每个朝向当独立样本更保守，不估计相同面的连续精度改善。

`apply_geometry_feedback(instance_id, *, frame_id, observation_sha256, log_likelihoods)` 只接受**刚observe的当前已付帧**、完全匹配的包SHA和唯一关联实例，每帧/实例最多应用一次，且必须有新支持资格。未来帧、早期帧、无关联实例拒绝；重复支持/已应用返回 `applied=false`。每帧相对log似然截断为[-6,0]，累计相对分数截断为[-24,0]。账本不能证明任意数值score来自真实残差，因此必须接独立残差caller：读取当前实际RGB-D、重算支持hash、使用公开原型假设，输出来源收据后再apply。不能把单元测试手填score称为真实闭环。

几何分数与语义先验分开：`posterior ∝ active_prior × exp(geometry_scores)`。S只在同一实例唯一类别得到至少2次新支持帧且当前关联无歧义时使用类别先验；G始终使用共同几何先验。类别冲突只影响本实例且永久退回几何先验，本版本不声称已实现冲突撤销。所有概率都是未经校准的机制分数。

## 本轮可验证范围

解析测试覆盖：两个设施与独立类别、S/G同几何快照、palette改名不影响关联、无marker增量区域桥接、盒体正面→侧面→背面的真实射线深度链及跳过中间视角拒绝、多实例遮挡歧义/单实例区域分裂拒绝、原地转向新支持、原frame_id重传/未来帧/旧反馈拒绝、新付费同内容包和连续全缺测帧接受但不增强、同支持微小扰动不重复增强、SHA与像素反投影对应、轴向4 m边界、空帧/未知类别、局部类别冲突、支持/证据/分数上限、同帧bootstrap双向拒绝及大背景拒绝。几何盒体仅在测试生成解析深度，真实边界/owner从未传给账本。

上述测试证明接口合同与有限关联机制，不能代替 noisy RGB-D 适配、自然实例识别、真实残差可辨识性或主策略效果实验。后续须先接真实残差caller和共享在线几何，再决定是否启动有界开发试点。
