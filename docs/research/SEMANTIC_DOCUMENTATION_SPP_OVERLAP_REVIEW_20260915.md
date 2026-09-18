# SPP 直接邻近工作补充核查

核查日期：2026-09-15。**SPP 已经实现“语义预测未见结构—提前规划观察—用实际观测纠正预测”。这一宽泛主张与我们的候选叙述直接重叠，不能再作为新创新。** 尚可检验的差别是：相同已观测几何下，类别是否提供对定向观察后外形质量净改善的额外预测能力，并在真实预算竞争中有用。本次没有证明这个更窄主张新颖，更没有证明我们已实现它。

本补充只核查 Mihir Dharmadhikari、Kostas Alexis 的 *Semantics-aware Predictive Inspection Path Planning*，不修改[前三篇报告](/root/NSO/docs/research/SEMANTIC_DOCUMENTATION_RELATED_WORK_GAP_20260915.md)。来源链是[作者研究页面](https://www.autonomousrobotslab.com/autonomous-navigation-and-exploration.html)→[作者论文 arXiv:2506.06560v1](https://arxiv.org/html/2506.06560v1)→[作者代码](https://github.com/ntnu-arl/predictive_planning_ros)。已读正文方法、公式、实验及结论；代码按提交 `91c3128c78417d7f6b59c5311c1755dd6b03388a` 核查预测、调度及配置，并检查其指定 `gbplanner_ros/predictive_planning` 分支当前提交 `55d75669d4dae026ee08cbf09002c263a82a8463` 的收益/巡检实现。仅静态阅读小型源码，未安装、编译、运行或下载数据集；不等同完整复现或对最终出版版逐字校勘。

**预测内容与公式。** 语义场景图把类别、空间关系和包围盒结合起来，改进 SUBDUE 寻找重复子图，再从入口类节点向未知区域外推。PP-AE 的论文式(8)为

\[
\Upsilon_{AE}(v)=(\alpha+\delta)\Upsilon_S(v)+(1-\alpha)\Upsilon_{VE}(v),
\]

其中语义项统计预测目标区域的可见未知体素，\(\alpha\) 表示预测与检测目标的匹配比例。它已经是预测驱动的观察收益，并非只有“目标在哪里”。[论文 §IV、V-E](https://arxiv.org/html/2506.06560v1#S5.SS5)

以下判断进一步来自实际作者代码，避免只按论文标题判断：

|核查点|可复核实现|对我们的含义|
|---|---|---|
|类别是否影响未见结构预测|`extendGraphToStructs` 从 `L_MANHOLE` 延展、对齐已有子结构，按节点标签与空间邻近检查匹配。|不能说 SPP 只有无语义几何复制；类别参与结构匹配。|
|是否迁移形状和观察位置|OI 取首个工位实测 longitudinal 的中心与两个端点，平移至预测工位，再生成视点与访问顺序。|“类别/结构先验帮助决定从哪边看”也不足以区分我们的方案。|
|是否随观测修改|AE 更新 `kSem=overlap`；零匹配时衰减先验。OI 检查对应目标及已看标记，缺失则探索，超次数跳过；目标列表耗尽可触发返航。|“反馈纠错、允许反悔、避免反复看”都已有实现基础。|

第一行依据[预测源码，1632 行起](https://github.com/ntnu-arl/predictive_planning_ros/blob/91c3128c78417d7f6b59c5311c1755dd6b03388a/predictive_planning/src/predictive_planning.cpp#L1632)；后两行依据[调度与反馈，105 行起](https://github.com/ntnu-arl/predictive_planning_ros/blob/91c3128c78417d7f6b59c5311c1755dd6b03388a/predictive_planning/src/planner_manager.cpp#L105)、同文件 275–435、588–637 行。这里的反馈核验目标存在/已看状态；在已核查代码中，没有发现以每次观察后的实例外形误差下降为监督标签的收益学习头。

依赖代码也给出明确边界：预测标注从 `L_LONG` 的位姿、长度和端点构造，语义收益采用射线返回体素集合；`isSeen` 查询已看标记。`getBlindTSPOrder` 使用欧氏距离，或开启选项时用平移/转向时间的最大值排序，实际去目标另行规划可行路径。故“考虑路线/转向成本”也不能单独作为差异；但排序代理不等于逐条完整路径的严格预算证明。[GBPlanner：收益 3016、3305 行；已看 4874 行；排序 5150 行；预测标注 8860 行](https://github.com/ntnu-arl/gbplanner_ros/blob/55d75669d4dae026ee08cbf09002c263a82a8463/gbplanner/src/rrg.cpp#L3016)

**实验及任务口径。** 仿真包括压载舱与工厂，各方法各环境 5 次；另有缺失目标消融和两船共 6 次实机。压载舱 PP-AE/PP-OI/Baseline 每舱时间为 48.97/40.33/65.58 s，语义表面覆盖为 98.96/99.93/99.87%。该覆盖按相机实际射线命中的目标网格面比例计算；预测模式时间只统计启用预测的舱室。第一船完整任务时间为 279/258/318 s。不能把局部时间改善直接当完整任务改善。[论文 §VI、VII，表2](https://arxiv.org/html/2506.06560v1#S6.SS1)

从上述目标与实现推断：这是“以更少时间完成指定目标表面巡检”，评价没有按设施先分别算外形误差再等权平均，也没有本项目的 \(J=C_{2D}Q_{shape}\)、\(C_{2D}\ge0.8\) 联合资格口径。它有真实执行和返航，不能描述成忽略预算的离线视点挑选。**实例等权、外轮廓精度、覆盖门槛是我们任务定义的区别；这些区别本身不能证明算法创新。** 表面看见率也不能直接推导出有噪声融合后的尺寸、占地和边界精度；反过来，我们不能因换成有利指标而抹去 SPP 已有的预测和反馈机制。

**可复现性边界。** 作者 README 提供 ROS Noetic/catkin 及 AE、OI、非预测 Baseline 入口，包含初始化运动和任务结束返航；这是比另外两套 GPU 语义高斯系统更接近当前 ROS 1 接口的参考。但依赖含 Voxblox、工位场景图、manhole 检测、SUBDUE、特定 GBPlanner/控制器分支和 Gazebo/LFS 模型。这里只确认有公开实现，未验证本机 CPU 构建或传感器替换可用。[作者安装与运行说明](https://github.com/ntnu-arl/predictive_planning_ros/blob/91c3128c78417d7f6b59c5311c1755dd6b03388a/README.md)、[依赖清单](https://github.com/ntnu-arl/predictive_planning_ros/blob/91c3128c78417d7f6b59c5311c1755dd6b03388a/vcstool/packages.repos)

公开 demo 也不能未经核对就代表论文配置：AE/OI YAML 的 `use_pose_cost=False`、`overlap_thr=0.8`、`init_sem_weight=0.0`；源代码确实将 pose 开关传入 SUBDUE。收益实现另带最小探索权重，OI 采用首个工位端点平移。应区分论文算法、作者当前示例和我们未来的机制适配器，不用其中一个名称替另一个背书；也不能据这些差异否定论文实验。[AE 配置](https://github.com/ntnu-arl/predictive_planning_ros/blob/91c3128c78417d7f6b59c5311c1755dd6b03388a/predictive_planning/config/bwt1/ae/predictive_planner_config.yaml)、[参数传递](https://github.com/ntnu-arl/predictive_planning_ros/blob/91c3128c78417d7f6b59c5311c1755dd6b03388a/predictive_planning/src/predictive_planning.cpp#L102)

对[毕业恢复计划](/root/NSO/docs/research/GRADUATION_RECOVERY_PLAN_20260915.md)的具体约束如下，均是我们提出的待验要求：

1. 若最终方案只是“类别/重复工位→预测目标包围盒→可见覆盖收益→匹配反馈”，科学主张已与 SPP 重叠。增加 ANS 四模块名称、网络实现或 \(C\times Q\) 不足以排除重复。应明确承认工程适配，不能包装成原创语义机制。
2. 候选差别必须落到 \(\hat g(h,c,a)=\mathbb E[\Delta Q_{shape}\mid h,c,a]\)：预测的是实际付费观察段后的外形质量净变化，包含方向、遮挡、噪声及当前已看状态，而非只预测目标位置、体素数或固定类别优先级。\(Q\) 真值仅离线用于标签和评价；在线反馈只能来自已获观测。即使实现此头，也需进一步证明它相对简单先验有必要，不能因本篇没有该头就称“首次”。
3. 最小强对照除共同几何 G、静态类别规则外，应有**结构模板/预测可见收益加实测纠错的机制基线**。模板仅由合法历史或共同训练集获得，不能读未见真值；与收益头共用候选、实例接口、短探查、改选、返航守卫及真实动作成本。若两目标入口没有足够重复结构，应如实记录原 SPP 无预测的冷启动，不能专门利用这一点安排弱基线。适配器须明确不是运行作者完整 SPP。
4. 可证伪差别是：在位置/可见区域预测质量相当时，类别条件收益头仍能更准确排序有噪声的外形改善，并在留出形状与布置、类别置换、等容量无类别头及反馈消融中产生预算内净收益。若模板基线已解释全部收益，或几何短探查后类别优势消失，应收回该创新主张；不能再次靠禁止改选或只报较高 \(Q\) 忽略覆盖失败。

本次排除了“宽泛语义预测巡检”作为新主张的空间；没有发现足以断言“类别条件的边际外形质量预测”与本篇完全相同的实现。留下的是一个更严格、也更难的待验差别，而不是已经成立的新颖性结论。本轮到此停止检索，不新增实验或扩展其他论文。
