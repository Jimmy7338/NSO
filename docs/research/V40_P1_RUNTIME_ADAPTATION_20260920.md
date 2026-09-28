# V40 P1：多设施运行链接入审计与下一轮实现合同

日期：2026-09-20。状态：只读集成审计，**没有实现或运行新的多设施闭环**。本次只新增此文档，未修改源代码、构造 World、调用传感/规划/TSDF/评价器，未读取正式测试私有种子。P1 同期开发几何、独立表面评价由各自实现任务负责；下面将已存在接口、同期拟定接口和仍缺失部件分开。

结论：保存/封存框架与观测 TSDF 可复用，但不能靠把 V39 的 parent 参数换成多设施场景完成接入。必须新增运动与传感适配、动作 0 控制器、实例几何关联和真实残差调用方；去掉旧全局双模板/全节点预先可见性表和单 ROI 依赖。保持 ANS 全局—局部组织和 OV-SDF、STGHP、RPN-UQ、IGCR 四接口，模块内部使用可审计 CPU 实现。

## 1. 已有能力与尚未完成的链条

| 接入环节 | 已存在且可核对 | 仍缺失 |
|---|---|---|
| 场景信息隔离 | `nso/scene_contract_v40.py` 的 `validate_scene_spec()`、`public_planner_spec()` 严格白名单 | P0 仅预留场景与种子承诺，public graph 仍 pending；并未实现可行路径或未知地图探索 |
| 实例发现与信念 | `PaidRGBDObservationV40`、`InstanceBeliefV40.observe()/snapshot()/geometry_snapshot()` | 实际多设施轨迹中的稳定关联；无色块/背面观察的几何关联；实例姿态/尺度估计 |
| 几何反馈 | `InstanceBeliefV40.apply_geometry_feedback(instance_id, frame_id, log_likelihoods)` 接受有界分数并去重 | 分数的真实观测计算器。它不证明调用方没有用 GT 或未来帧；P0 保存包兼容检查实际调用次数为 0 |
| 真实重建 | `ObservedRuntimeMapperV10.update(frame, scan)`、`mesh()` | 新坐标/动作合同适配，模型资源门；不能用预测形状更新 TSDF |
| 自主规划 | V35 `accept()` 已要求先收动作 0 的观测，空 prefix 在 controller 本体可以通过 | 工厂、模型、观测类型、候选/收益与返航仍绑定旧任务；目前不存在多设施 action-0 planner |
| 全体实例评价 | P0 规定所有任务实例的路线无关可达外表面，含水平/垂直面、漏检实例 | 同期 P1 评价器负责具体面积采样和归属定义；旧 V34 垂直面单设施评分不能直接充当新指标 |

P0 汇总 `audit_results/v40_p0_20260920/result.json` 报告：172 个保存包、344 次实例账本 observe，实际新场景几何/World/轨迹/规划/TSDF/评价均为 0。不能从这些合同通过推出多设施闭环有效。

## 2. 旧运行器中必须拆除的具体依赖

| 文件与函数 | 旧依赖及影响 | V40 最小接法 |
|---|---|---|
| [run_external_routes_v39.py](/root/NSO/scripts/run_external_routes_v39.py:28)：常量；`ExperimentV39.prepare()` | BUDGET=42、PREFIX=18；父为 P00/P01，隐藏构型 0/1；加载 V33/V34 全节点信息包和旧实验输入 | 新建独立 V40 runner/config/输出；开发场景 ID 来自开发清单，预算从 public.task 读取，prefix 必须空；不扩展旧冻结 cohort |
| 同文件 `create_controller()` | `load_public_models_v35()`、`PublicTemplatesV35.from_saved()`、预置 informative_nodes | 新工厂只接经过白名单提取的公共配置/导航先验、开发原型先验和实际观测账本；无实际构型、真实实例列表、GT ROI、未来可见性表 |
| 同文件 `run_case()` | `InformationPixelWorldV36(parent,hypothesis,episode)`；`ObservationV35`；按 18/final 抽 mesh；评价 `world.instance_mesh(0)` | 实际场景由隔离的开发 World 适配器持有；控制器接新数据类。checkpoint 改为预声明的实际动作时刻，保存完整观测 mesh 后独立评分 |
| 同文件 `analyze()` | 16 例固定顺序、共同前缀逐帧完全一致、固定两父均值；路程按 forward 次数当米 | 新统计用 parent/seed/method/phase；仅初始相同状态需同包合同，轨迹分歧后各自采集；路程为实际位移或 forward×0.25 m，不再把动作数当米 |
| [run_online_routes_v36.py](/root/NSO/scripts/run_online_routes_v36.py:497)：`observed_pose()` | 世界 XY 减固定 shift 后必须为整数米；heading 直接取四朝向 | 新位姿来自实际 T 或基座外参，保留米制连续 XY 与 30° yaw；不能将 .25 m 位移四舍五入成整米 |
| [cpu_four_modules_v35.py](/root/NSO/nso/cpu_four_modules_v35.py:30)：`ObservationV35.__post_init__()` | pose 三分量必须整数，heading<4；没有 K/T | 新控制观测包含 `PaidRGBDObservationV40` 及独立 paid scan/odometry/action/map 摘要；不要伪造 V35 pose 来绕过校验 |
| 同文件 `CPUFourModuleControllerV35.__init__()/accept()/compute_reward()` | 恰好两个模型、两个 mask、全局 posterior；每步按已知 `model.observed_masks[node]` 增量累计 | 按观测实例维护局部结构支持/方向账本；几何部分由实际帧产生。公开原型只用于预测，不用本次场景精确遮挡表更新“已见”状态 |
| 同文件 `select_target()/assess_action()` | `len(prefix)` 内直接执行前缀；全局直接选一个 next pose，局部仅检查公有边与 BFS 返航 hop | 保留 accept→全局选择→局部验证→实际执行的时序；全局选发现/设施/观察方向子目标，局部将其变成 0.25 m/30° 原语，并按原语完整返航成本守卫 |
| [online_planner_v35.py](/root/NSO/nso/online_planner_v35.py:156)：`load_public_models_v35()` | 限 P00/P01，强制两个模板和 18 动作回 anchor | 新 planner 不调用此 loader；空 prefix 单改一处不足以解除工厂与模型依赖 |
| 同文件 `ForecastBeliefPlannerV35.select()` | 两个 potential mask、单个 probability0、预知 informative_nodes；可能 150 万 memo 状态/20 s | 使用有限数量的观测实例、候选视点、短时域/有限路线组合；新信息预测来自开发原型与当前观测，不来自全节点真实候选深度 |
| [observation_belief_v35.py](/root/NSO/nso/observation_belief_v35.py:122)：`PublicTemplatesV35.from_saved()/geometry_evidence()/template_information()` | 两套全场景、全节点 clean depth/scan，旧固定双构型完整知识；全局类别 log-odds | 新实例残差模块使用开发原型 + 已观测局部位姿/尺度 + 当前实际帧；不得用新场景 GT 渲染全部候选模板后当公开先验 |
| [information_pixel_v34.py](/root/NSO/env/information_pixel_v34.py:131)：World 构造、`_rgb()`、`instance_mesh()` | 固定 V33 SHA；所有任务盒并入 `_reference_boxes`；色码 `2+hypothesis`；仅 owner=0 | 新开发 World 从 P1 私有几何构建真实第一命中传感。类别与结构独立字段，RGB 只在可见物理 marker 上着色；控制器不能接 primitive/owner/实例真值 |
| [surface_measurement_v34.py](/root/NSO/nso/surface_measurement_v34.py:90)：`extract_observed_asset_mesh()` / `SurfaceMeasurementV34` | 单公共 envelope 裁剪，地面剔除，召回仅完整垂直面，ROI 外精确率不惩罚 | 新整幅预测先封存；P1 evaluator 在独立进程按新参考面及错误预测规则评分。不得遍历 GT AABB 在主链裁出“已发现实例” |
| [observed_geometry_v39.py](/root/NSO/nso/observed_geometry_v39.py:47)：`ObservedGeometryV39.observe()`；`external_planners_v39.py` | 类型锁定 ObservationV35；可按单 public_bounds 筛点；class 登记和支持收益为全局 | 可复用已见 surfel/方向/边界算法思想，需新观测类型和实例台账。几何边界含 FOV/无效深度，不自动等于真实结构孔洞 |

## 3. 运动、坐标与传感适配是第一个运行门

P0 当前声明与 V39 实际不一致，不能静默沿用旧值：

| 项目 | P0 public_defaults | V39/V34 绑定 | 下一轮要求 |
|---|---|---|---|
| 前进/转向 | 0.25 m / 30° | `InformationPixelWorldV34.step()` 为 1 m / 90° | 实际几何姿态按新原语推进，转向也收费；12 个 yaw 档位与完整返回朝向 |
| 导航与地图分辨率 | 公共粗图 1 m；未规定它等于观测栅格 | 观测图 0.2 m，原机器人恰好处于格心 | 将粗拓扑、连续机器人位姿、观测栅格、TSDF 体素分开。保留 0.2 m 观测栅格、0.04 m TSDF 可作为开发 runtime 明示设置，不能把图分辨率直接给 mapper |
| 相机裁剪 | optical-z 轴向 0.1–4 m | `validate_sensor_packet_v34()` 拒绝≤0.15 m，另强制欧氏范围≤4 m | V40 renderer、packet validator、实例筛选、候选可见性及评价均使用相机光轴 z 的近/远裁剪；不再混用轴向 near 与径向 far，不偷换回旧 .15 m |
| 深度噪声 | relative std=0.01 | V39 为立体视差 iid_025px，V36 固定 seed | 新配对噪声模型/随机源按开发配置实现并记录；不用不同参数名称包装旧噪声 |
| 完整校准 | K、宽高、相机高公开；激光最大范围8m | 固定激光180线、0.25m高、360°；半径0.2m | P0 未包含的 laser rays/height、robot footprint、动作时间需在新 runtime 合同显式补充并统一给所有方法 |

[sensor_contract_v34.py](/root/NSO/nso/sensor_contract_v34.py:21) 还强制 `heading in range(4)`，并通过 `env.virtual3d.camera_pose()` 从格心反算 T；[cpu_sensor_contract_v10.py](/root/NSO/nso/cpu_sensor_contract_v10.py:54) 的 mapper 检查要求 origin=(0,0)、resolution=.2。不能因为底层 `RGBDFrame.validate()` 接受任意合法 T，就声称整个 packet/runner 已支持连续位姿。

P1 集成统一的未来相机深度约定为 **optical z 轴向近/远截断**：相机坐标中的 `z` 用于 `near_m=.1`、`far_m=4.` 及 RGB-D/TSDF 深度值，另检查投影是否在图像范围；斜射线的欧氏距离可以大于4 m。遮挡判断使用射线到目标的实际欧氏长度不构成额外相机远裁剪。若 renderer 使用单位方向射线，其返回欧氏距离必须转为 optical z；若射线 optical z 分量为1，参数通常已是轴向深度。两种射线参数化不得混用。相对深度噪声作用于轴向深度，近/远边界的浮点容差与 TSDF `depth_trunc` 行为需用同一组解析例核对。激光8 m范围仍是雷达射线距离，不随相机约定改名。

当前 `InstanceBeliefV40.observe()` 的 `norm(points)<=maximum_range_m`、`ObservedGeometryV39.observe()` 的欧氏4 m筛选仍属于旧实现；未来 V40 runtime 必须显式适配为轴向相机域，不能与已改为轴向的 `CandidateViewV40` 评价混用。`QualityMapperV2.update()` 还有 `d>.15` 的质量证据近深度筛选，不能在声称支持 .1 m 时漏掉这一处。世界 z 的地面/机器人高度过滤是另一种空间语义，不能把所有 .15 常数一律改成相机近裁剪。上述改动应新版本落盘并保留旧 V39/P0 兼容结果，不回写旧冻结数据。

推荐新 `SensorPacketV40`/validator（建议接口名，尚未实现）：复用 `RGBDFrame`、`PlanarScan` 的数组格式和 packet IO/hash 思路，保存连续世界位姿与 yaw，并分别验证基座—相机—激光固定外参、尺度、step/action 对应、射线范围和裁剪。控制器另接去除 scene/episode metadata 的严格数据类。地图 cell 由 T 投影得到，不要求 T 等于 cell center；位姿不能从栅格中心反推。

`swept_clear_v34()` 在斜向运动时会报错，只支持轴对齐线段；新 30° 原语需要圆形足迹对一般线段/实体的扫掠碰撞。私有 World 可用真实几何判定**已提交动作**是否碰撞，控制器只能用共同公共先验及已测几何进行安全决策，不可在候选循环调用真值碰撞或 packet_at()。第一版也可显式限定前进方向为公共粗边的轴向、转向分三次 30°执行；这必须作为所有方法相同的局部执行限制，而不是宣称已支持任意 12 向移动。

30°朝向与 1 m 粗边需要展开为实际原语成本：例如轴向 1 m 直线是四个前进动作，90°原地转向是三个动作。返航 Dijkstra/BFS 必须使用展开后的有向原语成本及终点朝向，不能把每条粗边当一个付费动作。精确返回的 XY/yaw 容差也须预先声明，不能在末尾自动瞬移到 anchor。

## 4. P1 几何与评价的边界交接

同期几何 owner 提供的接口方案为 `build_development_geometry(scene_spec, protocol) -> DevelopmentGeometryV40`，新文件拟为 `nso/development_geometry_v40.py`。几何数组为 vertices、triangles、triangle_instance_id（背景=-1）和中性 triangle_rgb；private_instances 含真实类别/结构/姿态/AABB/局部实体，marker_patches 含物理标记平面、法向、RGB。拟输出：

- `renderer_private/geometry.npz`、`renderer_private/markers.json`：只供传感模拟器使用；first-hit 结果允许决定物理标记是否可见，triangle owner 不进入在线帧。
- `evaluation_private/instances.json`：只供离线评价与谱系审计，不为 planner 创建初始实例。
- `public_workspace.json`：仅场地 bounds、公开 start pose、全局 palette；不含实例目录/ROI。public graph 本轮几何任务不负责生成，仍需独立实现和校验。

以上是同期实现的交接口径，不等于本审计已运行或验证其传感/控制能力。不得将这些附加字段随意塞入 P0 public 字典绕开严格 schema；使用显式新版 runtime 公共配置将 `public_planner_spec()` 的拷贝与逐字段校验的 workspace/palette/graph 组合。P0 `assert_formal_test_ready()` 有意对正式测试始终拒绝，不能通过改 status 字符串解锁；P1 只用 development 资产。

同期 evaluator owner 拟接口：`CandidateViewV40(...)`、`freeze_reference_v40(vertices, triangles, triangle_instance_id, candidate_views, ...) -> ReferenceSurfaceV40`、`evaluate_surface_v40(reference, pred_vertices, pred_triangles, C_map=..., ...)`。它不要求预测 owner ID，由离线空间距离分配实例；参考使用预声明可达视点下的外表面，路线无关，有限视点/面积采样的近似必须披露。水平面、垂直面、漏检实例均按新协议处理；不能拿 V34 单实例结果直接拼平均。

冻结候选视点集用于**离线参考面定义**，不是免费提供给 planner 的 GT 可见面表。整幅观测预测需在评价前封存，不先用真值实例盒“美化”裁剪。正确背景、目标误差、错误背景及域外错误预测的精确率处理以 P1 evaluator 的最终审计合同为准；runner 只传封存完整预测，不另设更有利 ROI。当前这两个新接口需在代码落盘后再核对具体模块名与参数，不应将本段当作已经通过的集成测试。

## 5. 按四接口接入真实反馈：必须补上的实例支持

建议新 `CPUFourModuleControllerV40(public_context, prior_bank, mode, limits)`，不继承 V35 的两个 model/mask 语义。模块调用日志沿用四个名称并保留实际输入/输出证据，但不把日志名字当成四个已训练神经模块的有效性证明。

| 接口 | 实际状态和动作 | 下一轮最小实现 |
|---|---|---|
| OV-SDF | 只融合真实 RGB-D/scan，维护已测表面/方向与观测实例；类别只改变结构先验 | driver 更新观测 mapper；`InstanceBeliefV40.observe()` 接真实 K/T 和 RGB-D；另有所有方法共享的几何发现/关联层 |
| STGHP | 全局选择发现设施、已有设施的观察方向或返航子目标 | 无实例时依据公共粗导航/已测前沿探索；发现实例后按有界候选的预测支持增量、信息价值与实际动作成本分配预算 |
| RPN-UQ | 把全局目标展开为实际原语，检查观测障碍/返回预算；执行第一动作 | 所有方法保留同一安全守卫。若只有可行性和剩余预算判断，就明确 learned uncertainty=false；独立风险评分以后实现再消融 |
| IGCR | 同一付费帧对各已关联实例的结构残差，更新后重新排序 | 新建可追溯 residual caller，输入实际像素/点云支持、观测 K/T、开发原型及拟合位姿；输出每结构 log score、使用像素/点数、失配/遮挡原因和源 hash |

### 当前实例账本的三个阻塞点

1. `InstanceBeliefV40.observe()` 仅从精确色块连通域检测实例。G/S 的实例关联不使用类别值，几何合同可以相同，但这仍是共同可见标记前端，**不是无色块几何目标发现**。未识别设施的几何补看能力须由共享几何台账提供，不能把 G 限制为只走前沿来制造语义优势。
2. `apply_geometry_feedback()` 要求当前指定 frame 在 `observe()` 中已无歧义关联到该实例。观察背面/凹面时正面的 marker 常不可见；现接口将无法给该帧反馈。因此需要从已测几何支持关联到已有实例的路径，并与新帧/像素 mask/packet SHA 绑定。不能传 GT owner 或真实 AABB，不能借用旧 marker frame_id 包装新帧分数；关联不明确时拒绝更新并留下原因。
3. `_view()` 仅按相机中心平移距离去重，默认最小 .25 m、最多16 view；原地 30°转向一直落在旧 view，且不同的新表面可能得不到反馈。新设计应依据观测支持、视向和位置去重，保留相关观测证据上限，而非按次数固定累计置信。先冻结去重规则，再比较 S/G；当前 .25 m 阈值边界还需注意浮点运动误差。

`PaidRGBDObservationV40` 已复制数组为只读并检查严格白名单；`observe()` 强制从 0 开始逐步递增、frame_id 不重复。应保留这些检查。`apply_geometry_feedback()` 目前仅证明 frame/instance 存在且分数有限，无法证明分数来源；真实 caller 及封存日志才补足这一点。

### 残差 caller 的可实现最小版本

开发原型库由通用结构构造函数/开发资产总结产生，所有 G/S 共用；其 nominal 尺度、姿态支持和类别条件表独立冻结。不能拿本次 private_instances 的真实 pose/尺寸复制一个模板，再说类别预测对了。已观测 marker 中心只是表面锚点，不自动等于设施中心或朝向；使用共同几何拟合得到尺度/朝向候选，拟合不确定则保留多个姿态或拒绝过早强更新。

对已无歧义关联的当前真实像素/点集，计算各开发原型在**当前观测位姿**的几何残差，按传感噪声和模型偏差尺度归一、鲁棒截断，保留每结构使用样本数与 loss。遮挡只由当前已见几何解释；当前缺测不直接当成“真实无表面”。全帧激光常含其他设施，未经实例几何关联不得把其全部 loss 归给某一实例。可先只做可验证的实例 RGB-D 反馈，scan 保留建图/安全用途，不能假装已完成对象级激光融合。

预测候选视点时允许射线查询**开发先验原型或已观测地图**，禁止查询实际 World 或其 ray scene。候选分数继续标为规划代理，实例总数未知时也不能用离线全任务实例数量做在线质量归一。完整评价 J、F1、真值表面积和可达参考 mask 都不参与规划。

## 6. 建议的新运行器控制时序

以下为建议伪代码，不是已存在 API：

```text
prepare_development_run:
    校验 development spec/资产 SHA/public runtime 合同/本批剩余配额
    新输出目录写 started（计一次实际开始），锁定代码、配置与种子
    driver 私有：创建 DevelopmentWorld；公开：创建 planner/controller
    创建共享实际 mapper；controller 无 World/GT/实际实例目录引用

for paid_step = 0 ... public.max_actions:
    packet = world.observe_initial() if paid_step == 0 else world.execute(pending_action)
    校验一次实际动作只产生一次当前帧；保存 packet；检查碰撞与动作时序
    mapper.update(actual RGBD, actual scan)                         # OV-SDF几何
    paid = PaidRGBDObservationV40(opaque_id, paid_step, RGBD, K, T)
    association = shared_observed_instance_support.observe(paid)    # S/G共同
    instance_belief.observe(paid)                                   # OV-SDF类别
    residuals = paid_geometry_residual(paid, association, prior_bank)
    instance_belief.apply_geometry_feedback(... actual current frame ...) # IGCR
    global_target = STGHP.select(observed geometry, beliefs, remaining budget)
    pending_action = RPN.validate_and_expand(global_target, return_reserve)
    保存四接口调用、全候选摘要、理由、剩余预算、计时与源 hash
    若到预声明 checkpoint：保存完整 observed map/mesh（不评分）
    若已返航并停止、预算耗尽、碰撞、超时或协议错误：停止并保存状态

freeze_prediction:
    封存实际包、动作、控制器日志、完整预测及源码/配置 SHA
evaluate_separately:
    读取离线固定参考，评价所有任务实例与全预测；不回流 controller
replay_fresh_process:
    同一开发输入/种子重新执行策略，比较动作/观测/地图/结果
    若只重放固定动作，明确 policy_reexecuted=false
```

接入时保持并细化 V39 的观测时序：原 runner 已先 `accept()`、再 `after_observation()`、最后 `next_action()`，原 `accept()` 内已执行全局双模板 IGCR。新实例残差依赖共享几何关联，应把该关联/残差边界显式移入新 controller 的接收流程，确保全部当前帧更新都在本步全局选点前完成，且同一帧只处理一次；不能先 `next_action()` 再补写反馈。mapper 在前，controller 的证据摘要包括实际 mapper 版本/hash；不因保存 checkpoint 再融合一遍同帧。

停止要区分：正常完成/已返航但低覆盖、预算不足未返回、碰撞、算法资源超限、完整性错误。返回但资格未满足是有意义的低分结果，不强制重跑。候选都不可行时优先按已验证返航路线执行；若返回路线也被当前证据否定，应报告失败/安全停止，不能增加免费动作。新动作图与预算变化后，旧 `all prefix packets equal` 检查改为相同状态/动作合同检查；S/G 轨迹分歧之后地图无需相同。

## 7. 下一轮最小开发 smoke 顺序

本轮仍只做静态资产/解析评价检查。下面 R2–R4 物理 smoke 只有在运行器实现、来源封存、小批预算登记，且持久存储达到既定 `max(10 GiB, 2×预计本批峰值 + 2 GiB)` 门之后才能执行；否则须由用户明确修订协议。本审计不降低该门。它们全部属于 development，不解封测试种子，不占旧 V35/V36/V39 配额。

| 门 | 检查 | 新 World/轨迹及通过标准 |
|---|---|---|
| R0 静态接线 | 验证 public graph/start/palette、开发原型来源、几何私有字段不进入 controller、评价定义绑定 | 0；graph pending 或原型姿态支持不明确就不创建 World |
| R1 解析观测合同 | 0.25 m 位移、30°转向的 T/K/scan 合同；两实例正反类别、同类不同实例、色块不可见几何反馈、重复/未来帧拒绝 | 0 World；用解析观测 fixture 测接口，不把手工 likelihood 当成真实反馈有效性证据 |
| R2 真实传感/运动小例 | 一个最小开发场景，初始帧和有限往返/转向动作；first-hit marker 遮挡、范围/近裁剪、实际碰撞与动作计数 | 计入 1 条开发诊断轨迹；诊断动作可预声明，但不称自主语义效果。最多约12动作，debug截断与真实任务终止分开 |
| R3 单设施动作0闭环 | 同一开发场景 S/G 各一次，初始观测后由策略选择目标，真实融合，首次反馈有可追溯像素来源 | 2 条开发主轨迹 + 各自独立回放；不得使用18步模板前缀，动作/碰撞/归航/资源完整，不设“必须S赢”通过门 |
| R4 多设施最小分配 | 一个预声明相反类别/不同朝向的开发场景，S/G 各一次；至少两实例由实际观测发现 | 2 条开发主轨迹 + 回放；记录跨设施预算分配和反馈归属，未发现实例仍留在离线分母。若未出现有效决策先查日志，不追加有利场景 |

建议第一批物理上限为 **5 次新主开始**（1 诊断 + 4 自主）与单列最多 5 次回放，纳入 V40 开发≤48总账。任何调试失败也占一次开始；资源门之前拒绝创建不占物理开始。若只需要单帧 World 检查却另外构造 World，必须另计开始或替代 R2，不能称为免费第零次。R3/R4 的每例上限读取已有 public.task（当前标准160），诊断早停不用于性能结论。超时/文件上限触发后保留部分包与错误，不自动换配置重跑。

R4 通过只表示初步多设施机制可运行；仍不能证明独立场景优势、自然识别、完整神经 ANS 或真实 SLAM 精度。后续才在冻结开发策略后进入正式矩阵。

## 8. CPU、存储与保存策略

只读探测时持久空间约 **574 MiB**，`/proc/meminfo` 的 MemAvailable 约 **5.85 GiB**；没有取得可用 cgroup 上限路径，因此这不是资源保证。当前空间尚不满足**任何新物理实验批次**的既定存储门 `max(10 GiB, 2×预计本批峰值 + 2 GiB)`，包括开发诊断与 R2–R4 smoke，不仅是正式测试。不可删除唯一原始证据为新批次腾空间。

- CPU 初始并发=1，`OMP_NUM_THREADS=OPENBLAS_NUM_THREADS=MKL_NUM_THREADS=1`，使用现有 `.venv-3d`。库与缓存部分位于宿主 RAM，进程启动前确认可访问；本轮不安装环境。实际 World/ray scene、mapper 和 evaluator 分段记录墙钟/峰值 RSS，避免同时持有两个大场景。
- 新候选数量建议初始 cap=32、每实例候选≤8、局部滚动深度≤3、每次全局组合状态≤50,000；规划 watchdog 初始2秒、每例600秒、RSS软门2 GiB。这些是待开发试点冻结的工程初值，不是已测性能或算法优越性阈值。超限执行有日志的共同 fallback，不能静默增加预算。
- 0.2 m观测图下 12/20/30 m 方形场地分别约3,600/10,000/22,500格，二维数组很小；内存增长主要来自稀疏TSDF、面片/历史帧和几何候选，不能仅用二维格数估计成本。160动作的161张原生 RGB-D 光栅未压缩约7.8 MB，加 scan/K/T/元数据；实际压缩包及网格需用 R2/R3 实测，不据此承诺每例仅8 MB。
- 第一批建议每例持久输出 cap=64 MiB、单文件 cap=32 MiB、整批主/回放/源/失败 cap=384 MiB，并另估64 MiB系统余量及至少64 MiB编码/暂存空间。这是产物和局部峰值的预算初值，**不是物理启动许可**；当前574 MiB不能据此启动 R2–R4，仍须先满足上面的10 GiB最低门或用户明确修订。批量可在运行前按实测成本收缩，但不能借较小cap绕开既定存储协议。不得沿用V39总128MiB去覆盖未来520条。
- 主物理包流式写入，原始完整 mesh至少保存终点；初次smoke可只存初始/终点或极少预声明checkpoint，不需要每步抽三角网格。回放优先与主数据逐包/逐状态比对并保存hash/差异，不重复保存整套成功RGB-D。失败与不一致数据必须保留。二次封存与源zip同样计入输出。
- 静态 P1 几何、解析评价与小型展示属于此前单独声明的小工具例外，按各自产物上限运行；该例外不延伸到创建 World、付费传感或运行物理轨迹的开发 smoke。本文件不改变任何新物理批次的存储门。

## 9. 实现完成时必须能回答的核查项

1. 第一张真实帧到来前，controller 的实例集合是否为空？已知的是场地/粗拓扑/类别—结构先验，而非本场景实际实例及其结构。
2. 0.25 m/30°是否真的执行和收费？返航是否包含全部转向/位移，是否绕开旧整数米/四朝向转换？
3. 对同一实际帧，G/S 的几何发现、实例关联、候选、安全与残差是否一致，类别只改变其条件先验/相应决策？轨迹分歧后的观测差异是否按真实路径保留？
4. 从看不到 marker 的新侧面取得的几何是否能合法关联、纠错？每个反馈的原始packet SHA、有效像素/点、原型来源与loss是否可复算？
5. 候选循环是否完全没有实际World/未来packet/真实表面积/GT遮挡访问？没有未知类别时是否仍有几何发现和补看路径？
6. TSDF 是否只融合每个实际帧一次？评价器是否在完整预测封存后读取私有参考？漏发现设施、错误域外面与水平外表面是否按新协议进入结果？
7. 每次开始/失败/回放、时间/RSS/字节数是否登记，低分任务是否保留，尚未通过的能力是否明确标为未验证？

## 10. 本审计核对的关键源指纹

以下为本次读取时的 SHA256；它们是审计定位，不替代下一轮实际运行前的完整源码封存。同期新几何/评价文件尚在实现，不在此处冒填 SHA。

| 文件 | SHA256 |
|---|---|
| scripts/run_external_routes_v39.py | `bac0d6b228dfe8ab81996f3b72b505d8be69f2f35366569e359cc404a43e24df` |
| scripts/run_online_routes_v36.py | `281d73c3d1bc05e89754f1c7386a336d497110067793b6dfee4ace0c42422b11` |
| env/information_pixel_v34.py | `b43d53b83f0cc4da3d99906fbc8b12bc18ecc3d9c550f40c45fd2554708629c4` |
| nso/sensor_contract_v34.py | `ddc64c80467807ac10960dac7ad0b2dabec1abfdaf981f231a59745d8c2405f9` |
| nso/cpu_four_modules_v35.py | `e2924616b24f5ef66f52dcf10e49000c9ec9060e9fcb90a8e4e517bf13d9476a` |
| nso/online_planner_v35.py | `d689cf16745375ca668bdde604836fb0fbb7382565675e510e34ece18411b5d6` |
| nso/observation_belief_v35.py | `09a6a4d9d0ac392b179a0da3cdd6a88fcb1c05e5bc3d20d393337d3a20551653` |
| nso/instance_belief_v40.py | `249f9aa6ec14d9aefcf0dbfcc42641d4dd1cd46ae516d4f798e0d2f22063872d` |
| nso/scene_contract_v40.py | `b17cd88125b61db6c85fe8361589157b0e4b4150b86ecf231edead3004a15e47` |
| configs/virtual3d/v40_scene_protocol_20260920.json | `abb23ef5eb45f46006dddc736ff4e6e9203cc499d7d936657e63f1ba450dda98` |
| nso/observed_runtime_mapper_v10.py | `c27a977a1b7ae417748a786a118287b282771b630ac88b56c0de7e11f5b00a3b` |
| nso/mapping3d.py | `275144d68c1bc6e8876ac622629f19916d09ca07651681dd8cf2b11dbe64b38f` |
