# V24 双对象公共前缀外形表示检查：执行前协议

日期2026-09-15。仅处理已封存的4条V24.1公共前缀，固定一版、一次4例×2个 ObservedShapeBackendV24，不新增传感、TSDF融合、策略动作或训练。P01此前缀C<0.8事实保留。旧145份冻结源码、后端、世界和参数不变。

## 输入与时序

输入根 audit_results/facility_choice_v24_paid_prefix_20260915 必须complete、4个不同PID验证通过；先核其1003产物及145源SHA。每例读所有0..234或0..254包，逐包核packetSHA。两个后端均从frame0起接收全部付费depth/K/T，公开重力上方向为(0,0,1)。只在完整前缀终点各做一次snapshot，合计8次；不选择有利帧或阶段。

实例关联完全复用冻结prefix的二值RGB MarkerTracks：两类实际颜色并集，连通分量至少16有效深度像素，0.75 m关联。每个观测轨迹第一次形成合格分量时，在该分量的真实深度点中取距三维中位数最近的一点作为首次seed；平局按原像素行优先序。不使用中位数虚构一个测量点，不读类别、隐藏中心、尺寸或GT窗口。两个实例槽按观测轨迹首次出现顺序建立；漏检槽保持无seed，额外轨迹记录且关联门失败，不按Q选择、手工合并或替换。后续每步均仅给几何，不再次给seed。完整重建的MarkerTracks输出需与原prefix trace一致。

后端输入限制为depth/intrinsic/world_from_camera/observation_id/首次实际seed。原final_mesh.npz先读三数组验SHA，几何作为共同raw_mesh供两个snapshot使用；不重新积分TSDF。所有地面、实测支持、外壳先验和拒绝均由冻结后端产生。

## 固定输出与评价关联

固定输出 raw、measured、inferred 和 completed=measured+inferred 四种表示，不按Q选择表示。raw只引用原文件与数组SHA；每实例保存measured/inferred顶点三角形压缩文件，completed保存固定组成和SHA以避免重复；全局表示按观测槽0、1顺序相加，推断单独保留。大点云只保存数量/SHA，原depth始终保留可重算；地面/边缘/自由射线/拒绝审计完整保存。

没有推断三角形时completed应与measured几何及评价完全相同，必须显式记录。推断获准时，其新增顶/背面属于规则封闭外壳先验贡献，不得写成未测表面的实测精度改善。允许测量与推断重叠，按冻结后端原样保留，不删除不利几何。

全部处理结束后，单独评价侧才构造V24.1 ReferenceWorld，其step/sense/scan方法均抛异常，不生成新包。使用冻结V23三投影指标和阈值：boundary spacing 0.01 m，F1@0.02/0.05/0.10 m，严格完成0.05 m/IoU0.9。所有任务对象等权、漏建为0；原C、实际返航/碰撞/付费长度随结果报告，但这是prefix终点表示，不能称正式选择任务成功。

评价关联在运行前固定为两层：全局联合mesh按GT固定不交叠窗口归属，是空间几何分数；另以每backend首次实际seed是否唯一位于一个GT窗口建立评价审计映射。零匹配保持unassigned，多匹配保持ambiguous；多backend对应同一GT记重复，未有唯一backend对应的GT记missing。绝不通过最近真值中心、最优匹配或人工修正。该映射不会回传后端，也不修剪保存的后端输出。

seed关联门仅表示每个GT恰好一个唯一seed关联backend、无额外/漏检/重复/ambiguous且完整marker配对。另设最小实例分离门：每槽须有非空实测支持和实测网格，两个backend不能连到同一非空组件。两门共同通过才称当前观测实例关联/最低分离条件可用，仍不证明完整实例身份或外形正确。即使联合mesh经GT窗口得到高Q，关联门失败也不能称实例建档可用。保存每实例固定映射、归属到其他窗口的几何评价和全局分数，保留未归属输出；空间分数不能证明实例身份正确。

同父两排列的seed、观测帧数、点集合SHA、measured/inferred/completed网格SHA、地面和拒绝审计必须相同；GT形状随排列变动，评价数值不要求相同。若配对失败保留首差及false门，不修改后端重跑。

## 容量、执行和失败

脚本 scripts/replay_facility_choice_shape_v24.py；输出 audit_results/facility_choice_v24_shape_prefix_20260915。prepare先只读核来源，再在新目录冻结新脚本/协议小源码包及旧源/证据SHA。旧根、重复prepare/run入口在可写try外拒绝。一次run处理4例，按例释放大数组；预计1960次backend.observe、8个终点snapshot，CPU约3–10分钟、峰值RAM约0.5–1.5 GiB，实际用量记录，估计不作保证。

所有新产物写前在内存序列化/压缩，检查新目录总容量8 MiB硬上限、至少32 MiB可用空间；日常写入另保留64 KiB失败元数据余量。raw、完整点云和completed副本不重复保存。超限保留失败，不以降采样、删除独特几何、改参数或改评价来凑容量。所有文件封存SHA，失败receipt与manifest保留，不自动重启。

本轮是已保存前缀的共同表示开发检查，不是独立新环境、自然语义、选择收益、自主运行或四模块优势验证。只读审查通过后按已有授权执行一次；不围绕任何Q结果继续调后端。
