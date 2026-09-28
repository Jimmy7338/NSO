# 场景级方法与实验：小论文／大论文并稿片段

准备日期：2026-09-28。数据核对时点：2026-09-28 12:29 UTC（北京时间20:29）。本文件是独立插入稿，未修改英文正文、毕业方法章或毕业实验章。未运行新World、TSDF或评价，也未重新绘图。

当前英文正文为[VIRTUAL_PAPER_EN_20260928.md](/root/NSO/docs/thesis/VIRTUAL_PAPER_EN_20260928.md)，其方法与主要数字仍属于双构型局部观察实验。毕业正文对应[方法章](/root/NSO/docs/thesis/GRADUATION_METHOD_CHAPTER_20260928.md)与[实验章](/root/NSO/docs/thesis/GRADUATION_SIMULATION_CHAPTER_20260928.md)。新多实例实现已有[完整方法附录](/root/NSO/docs/thesis/ARTICLE_MULTI_INSTANCE_METHOD_APPENDIX_20260928.md)；本文件提供适合正文的短版本。

## 1. 两种论文的放置方式

| 内容 | 小论文 | 毕业论文 |
|---|---|---|
| 已验证的类别方向选择与几何纠错 | 保持主方法、主结果及预算边界，不改写原实验定义 | 作为局部机制验证，保留全部配对、消融和负结果 |
| 四构型、多实例及共同运动预算 | 在方法末尾增加“场景级扩展”，必要公式移入附录 | 在现有方法章第7节之后新增独立一节，明确与双构型控制器的区别 |
| 地面背景导致的对象支持连通及修正 | 一段方法实现、一段固定路径机制结果 | 展开关联、类别资格、反馈和共享信息到达的过程分析 |
| AISLE四方法同轨迹、同网格 | 作为开发条件与适用边界；不替代局部正结果，也不声称共享有效 | 给出完整数值、真实场景网格和全部实例，不把持平结果省略 |
| 后续三场景共同前端对照 | 等全部预声明结果可比后，决定主文或补充材料位置 | 单列场景级实验，逐布局报告效果、成本与失败，不与旧指标混算 |
| 数值退化面与共同评价 | 实验实现中用一短段说明，保留原失败率 | 在平台实现与实验结果中区分运动完成、原端到端状态和补充质量 |

**主线建议。** 小论文先讲清“类别知识如何改变有代价的观察选择，以及实测几何如何限制错误先验”；新场景级内容补充其实现条件与扩展边界。毕业论文进一步解释从单设施到多设施的状态、关联、共享和预算耦合。当前不宜把原先6.1638%的局部收益改写成多实例共享方法的成绩，也不宜把地面前端本身命名为语义创新。

如果后续共同前端内的S/B闭环对照显示稳定、可解释的新增收益，再将多实例方法提升为小论文主方法。届时应同步调整问题定义、方法公式和主表，不能只替换摘要中的方案名称。

## 2. 可直接用于英文稿的方法文字

建议作为现有Section 4之后的独立扩展小节。此处G、B、S和NBV仅指下述场景级实现；它们与前文局部控制器的同名标签不混用。

### Scene-level extension with observed instances

The scene-level extension considers active documentation after an industrial layout change, with the environment static throughout each mission. Four facilities from two repeated categories compete for a shared motion budget. Each observed instance maintains four possible structures: planar, recessed, louvered, and open-frame. Instances are established from acquired RGB-D data; their ground-truth poses and configurations are not supplied to the controller. A conservative navigation graph, calibrated camera geometry, and exact simulated poses define the current operating conditions. Synthetic category markers isolate the effect of category information on planning. This experiment therefore examines observation allocation under a navigation prior, rather than localization accuracy or moving-obstacle avoidance.

The four module interfaces are retained. OV-SDF manages observed instance support, category qualification, and structure beliefs; IGCR accumulates bounded evidence from measured depth; STGHP selects affordable observation targets; and RPN-UQ checks primitive execution and the cost of returning to the initial pose. Templates inform prediction, while occupancy and TSDF integrate the complete measured packets only. Compared with the local two-configuration implementation, this controller couples several instances through one feasible route and replans after an executed observation macro.

For instance \(i\), let \(L_i(h)\) denote accumulated geometric evidence, \(q_0(h)\) a uniform structure prior, and \(q_1(h\mid c_i)\) the declared category prior. The posterior has the form

\[
b_i(h)\propto\bigl[(1-\rho_i)q_0(h)+\rho_iq_1(h\mid c_i)\bigr]\exp L_i(h).
\]

G uses the geometry prior, B fixes the mixture weight at \(\rho_i=1/2\), and S updates this weight using qualified same-category peers. Own-instance evidence is applied once, with leave-one-out sharing and replacement of cumulative peer messages. The weight expresses relative support for two prior models; it is not a calibrated recognition confidence. NBV uses the same geometric evidence as G but compares direct views without diagnostic lookahead. G, B, and S may all pay for an additional diagnostic observation.

For a candidate pose \(v\), the direct utility combines an unknown-space proxy \(D(v)\) and expected new nominal surface area \(A_{ih}(v)\):

\[
U_{\mathrm{dir}}(v)=
\frac{D(v)+\sum_i\sum_h b_i(h)A_{ih}(v)}
{d(s_t,v)+1+d(v,s_0)}.
\]

The denominator counts translation, rotation, explicit observation, and return actions. Diagnostic alternatives consider one paid observation followed by one affordable final view; instance utilities are summed before selecting the common final pose. The planner executes only the first macro and then updates its decision from actual observations. These nominal utilities guide scheduling and are distinct from measured coverage and surface F1. Earlier information can affect the executed route only if it changes an actionable candidate ranking before the return constraint dominates.

### Geometry-supported instance association

Connected depth support can grow through the floor and link nearby facilities, making previously observed instance identities ambiguous. Such ambiguity also prevents category qualification and the transfer of useful geometric evidence. We therefore use a common geometric frontend that excludes a measured floor band from object-association candidates. The band is estimated from the current paid depth image and calibrated mounting height, and is accepted only when spatial support, height consistency, and noise checks pass. Otherwise, association uses the original input. The rule does not read category values or private facility geometry, and it is identical for G, B, S, and NBV. The complete RGB-D image remains available to occupancy mapping and TSDF fusion. This frontend assumes a horizontal static floor and known camera geometry; near-ground contact features and nonplanar terrain remain limitations.

## 3. 可直接用于英文稿的结果与讨论文字

建议增加在现有Section 5.6之后，标题为“Scene-level development and information timing”。以下文字只引用已核验的AISLE端点和固定路径回放，不预报新闭环结果。

### Scene-level development and information timing

In the original AISLE development condition, G, B, S, and NBV produce exactly identical executed actions, full-pose sequences, vertex arrays, and triangle indices. Each run uses 160 paid actions, travels 24.0 m, and includes 54 rotations and 10 explicit observations. Their common navigable coverage is 0.999289, macro surface F1 is 0.857363, and joint score is 0.856753. The equality is informative: additional belief mechanisms do not necessarily change an executable decision. In particular, the first informative same-category peer appears at action 136, after the original controller enters its return phase at action 126.

To isolate the association bottleneck, we replayed the same 161 saved observations with the original frontend and the measured-floor revision. The original control reproduced the stored evidence exactly. With the revised frontend, the first informative peer and the first S/B posterior difference both move from action 136 to action 82; uncertain instance-frame records decrease from 278 to 3. However, the number of applied feedback updates decreases from 21 to 18, and the second cabinet's first own-instance feedback moves from action 85 to action 87. The evidence therefore concerns earlier availability of shareable information, not a general increase in feedback. Because the input route is fixed, these results do not establish a new trajectory, a better structure estimate, or a reconstruction gain. A subsequent closed-loop comparison must determine whether the earlier information changes affordable actions and measured quality.

### Full-scene reconstruction and common evaluation

Scene-level evaluation separates observed-free navigable coverage from the mean surface F1 of all four declared facilities. Undiscovered facilities remain in the denominator. The joint score is \(J_{\mathrm{nav}}=C_{\mathrm{nav}}(\tfrac14\sum_i F_{1,i})\), and its components are reported separately. Precision and recall use the complete prediction and a fixed observable reference; display crops do not enter scoring. The common-view AISLE meshes show that observation of most navigable space can coexist with incomplete equipment surfaces. For example, the second facility has recall 0.557492, whereas the first reaches 1.000000 under the observable reference. Ground-truth visualizations include additional unobservable surfaces and are not themselves the scoring domain.

A numerical compatibility issue occurred after the CELL geometric run had completed its motion: the original evaluator rejected one finite triangle of area \(4.24\times10^{-13}\,\mathrm{m}^2\). The common evaluation version removes faces of area at most \(5\times10^{-13}\,\mathrm{m}^2\) from every prediction under the same rule, without consulting semantics or cropping a region of interest. Original data and failure status are retained. Results from this supplement are reported separately from original end-to-end completion. Where no face is removed, a previous score is reused only after confirming identical prediction, reference, parameters, coverage, and qualification. Changed predictions require a derived measurement because removing a face may also change the evaluator's subsequent sampling sequence.

### Discussion bridge

The local experiments establish a conditional benefit of category information for selecting an observation direction. The scene-level study adds a stricter requirement: the relevant instance must remain geometrically identifiable, and its information must arrive early enough to change an affordable shared route. The AISLE ties and fixed-path frontend diagnosis identify these requirements without demonstrating an additional cross-instance planning benefit. The two studies consequently answer different questions and use different evaluation contracts; their percentage gains should not be pooled.

## 4. 毕业方法章可直接插入的中文段落

建议置于当前第7节双构型实例之后，标题为“从局部观察到多实例场景规划”。当前章前文的双构型公式和参数继续描述原实验，不以以下文字追溯替换。

### 从局部观察到多实例场景规划

局部实验考察类别线索能否帮助机器人选择设施的有效观察方向。扩展到完整工作区后，机器人还需要在多个设施之间分配同一份运动预算，并处理实例关联是否稳定、某个设施的信息是否能够帮助另一设施等问题。本节以布局调整后的工业建档为应用背景，单次任务内环境保持静止。每个场景包含四个设施，分属两个重复类别；每个在线实例维护平面、凹入、百叶和开放框架四种结构假设。实例由已经取得的观测建立，实际设施位置与结构不提供给规划器。保守导航图、准确模拟位姿和人工类别标记构成当前实验条件。

系统保留四模块及决策—执行层次。OV-SDF组织实测实例支持、类别资格与结构信念；IGCR依据当前深度残差更新几何证据；STGHP比较到达、观察和返航共同约束下的候选；RPN-UQ检查实际原子动作和剩余返航代价。地图与TSDF只融合实际获得的完整观测，名义模板不用于补齐未观测表面。与前文直接输出下一原子子目标的双构型控制器不同，多实例实现先选择一个可执行的观察宏动作，再根据实际执行情况更新或取消该宏动作。

G与NBV不以类别值改变结构先验，B将几何先验与类别先验按固定比例混合，S额外用合格同类实例的几何证据调整混合权重。该共享采用留一方式，使当前实例自身证据只加入一次。G、B、S均可根据几何信息决定是否付费补看；NBV使用共同的几何前端和直接观察评分，但没有诊断前瞻。因而S/B对照检验跨实例共享，B/G对照检验类别先验，G/NBV对照检验诊断规划，不通过剥夺几何方法的补看能力构造优势。

共享信息必须经过正确的实例关联才能用于规划。原深度连通支持可能沿地面扩展，造成相邻设施支持重叠，进而使已观察实例暂时失去类别与反馈资格。共同前端修正仅从对象关联候选中排除经过当前深度和已知安装高度核验的水平地面带；支撑不足、高度不一致或噪声过大时回退原输入。该规则对所有方法相同，不读取类别或设备真值。完整深度继续进入占据地图和TSDF，因此它修正的是信息归属，不是删除地面以提高重建分数。该实现依赖水平静态地面及准确标定，对接近地面的设备边缘仍有局限。

候选收益由未知区域面积代理与各实例的后验加权名义新增表面构成，并除以前往、付费观察和完整返航的总代价。不同实例的收益先相加，再选择共同的后续观察姿态，避免为各个物体分别选择实际不能同时执行的路线。规划收益是用于选向的代理，不是直接读取实测F1。即使语义改变了结构信念，若候选新增表面差异太小、原分数间隔过大或已进入返航阶段，最终动作仍可能不变。

## 5. 毕业实验章可直接插入的中文段落

建议放在当前第8节预算／几何扩展之后、新的章节小结之前，作为单独的“场景级开发验证”。旧局部结果与新结果分表，不合并均值。

### 场景级开发验证与信息到达时机

首先在通道型工业工作区中比较G、B、S与NBV。四种方法均使用160个付费动作，平移24米、转向54次、额外观察10次，共取得161帧RGB-D与扫描；各任务无碰撞并返回起始位置及朝向。保存的动作、完整姿态序列和终点网格数组完全相同，故覆盖、表面F1及联合分数也相同。该结果表明，增加类别与共享信念并不必然改变实际路线，本场景不能作为多实例语义优势的证据。

对保存过程的分析发现，首个合格且具有信息的同类实例在第136步才进入共享，而原策略已在第126步开始返航。为检验实例支持连通是否推迟了信息可用时间，固定原161帧观测，分别回放原前端与地面关联修正。旧前端回放与原记录完全一致。修正后，首个有效几何反馈由第85步提前到第8步，首个可共享同类信息及S/B后验差异由第136步提前到第82步，关联不确定的“实例×帧”记录由278降至3。不过，实际应用反馈总数由21降至18，第二个cabinet的首次自身反馈由85延至87。改善主要体现为部分有用信息更早到达，而不是所有对象或所有反馈都改善。

该回放仍使用原轨迹，既没有生成新候选动作，也没有形成新的终点重建。它支持前端连通问题对信息资格和时序的影响，但不能据此断言路径更短、最终精度更高或S优于B。新的闭环对照需要让G、B、S、NBV共同使用修正前端，在原声明的通道、工作单元和环路三个开发布局中分别比较动作与质量；目前插入的段落不预写该对照的结果。

### 实际三维网格与设备完整性

图X在统一世界坐标、视角和尺度下显示AISLE真值、G和S的实际终点TSDF，并叠加保存的XY路线。每份预测的205,566个三角面全部参与显示，没有抽稀、补面或模板表面叠加。网格颜色表示面中心高度及几何光照，不表示重建误差。G与S的几何和路线完全相同，这一可视化用于说明完整重建流程及当前限制。

可导航区域覆盖达到0.999289，但全设施宏平均表面F1为0.857363，说明空间观察充分并不代表所有设施表面完整。四个设备的召回分别为1.000000、0.557492、0.887814和0.831784，第二个设备仍存在明显缺失。GT图包含任务中不可观测的顶面等表面，而分数只针对预先固定的可观测参考，二者不能直接按视觉缺口逐一对应。设备局部放大只裁剪显示范围，评价始终使用完整预测，全部四个设施始终保留在宏平均分母中。

### 评价兼容性与失败记录

CELL的G任务在160个动作后完成无碰撞返航，原评价器随后因一个极小三角面拒绝评分，原端到端记录因此保留为失败。统一补充评价采用所有方法共同的面积门限，仅剔除不大于\(5\times10^{-13}\,\mathrm{m}^2\)的数值退化面，并保持其余顶点、三角面及参考几何不变。运动完成、原端到端成功和新版本质量可用性分别报告；补充评分不覆盖原失败。由于删除面可能改变后续表面采样位置，存在删除的记录使用独立测量版本，不能仅凭删除面积很小就宣称分数完全不变。本插入稿不引用尚未核验的补充评分。

## 6. 可直接使用的表格

**表X 原AISLE开发条件的完整四方法对照。** 这是场景级开发结果，不是原双构型确认实验。P/R/F1为固定四实例的宏平均，J为覆盖与宏平均F1之积。

| 方法 | \(C_{\rm nav}\) | P | R | F1 | \(J_{\rm nav}\) | 路程/m | 转向 | 额外观察 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| NBV | 0.999289 | 0.921930 | 0.819272 | 0.857363 | 0.856753 | 24.0 | 54 | 10 |
| G | 0.999289 | 0.921930 | 0.819272 | 0.857363 | 0.856753 | 24.0 | 54 | 10 |
| B | 0.999289 | 0.921930 | 0.819272 | 0.857363 | 0.856753 | 24.0 | 54 | 10 |
| S | 0.999289 | 0.921930 | 0.819272 | 0.857363 | 0.856753 | 24.0 | 54 | 10 |

**表Y 条件于同一条AISLE路径的前端回放。** 时间单位为原路径已付费步数；第126步是原策略的返航分界，不是新策略的返航时间。“实例×帧”不是独立场景数或GT匹配错误率。

| 项目 | 原前端 | 共同地面修正 |
|---|---:|---:|
| 首次可靠平面 | 11 | 8 |
| 首次实际几何反馈 | 85 | 8 |
| 首个有信息同类实例／首次S−B后验差异 | 136 | 82 |
| 关联不确定实例×帧 | 278 | 3 |
| 全路径实际反馈次数 | 21 | 18 |
| 原返航前实际反馈次数 | 15 | 15 |
| 第二个cabinet首次自身反馈 | 85 | 87 |

## 7. 图与双语图注

### 可直接放入正文的单张场景图

![同尺度AISLE实际场景网格与路线](/root/NSO/docs/thesis/figures/article_scene_meshes_20260928/aisle_gbs_v2/aisle_gt_g_s_preview.png)

**图X／Figure X。** AISLE完整真值和实际保存的G、S终点网格，共享正交视角、世界范围及物理尺度。每份预测的全部205,566个三角面用于显示。XY路线投影到地面并覆盖遮挡以便阅读，菱形表示起点与返航点。颜色表示三角面中心高度与几何光照，不是误差。G/S顶点、三角面及路线完全相同。GT包含不可观测表面，而指标使用固定可观测参考。[PDF](/root/NSO/docs/thesis/figures/article_scene_meshes_20260928/aisle_gbs_v2/aisle_gt_g_s_preview.pdf)。

*Full ground truth and saved G/S endpoint meshes in AISLE under a common orthographic view, world extent, and metric scale. All 205,566 predicted triangles are rendered. Executed XY routes are projected onto the floor and overlaid for visibility; diamonds indicate start and return. Colors encode triangle-centroid height with geometric lighting, not error. G and S have identical vertices, triangle indices, and routes. GT includes unobservable surfaces, whereas the scores use the fixed observable reference.*

### 当前其他图的使用限制

[三场景×五列总览](/root/NSO/docs/thesis/figures/article_scene_meshes_20260928/aisle_gbs_v2/scene_meshes_all_declared.png)和[全部设备局部图](/root/NSO/docs/thesis/figures/article_scene_meshes_20260928/aisle_gbs_v2/art1_aisle_dev_b160_n92801_all_instance_closeups.png)生成于更早的明确快照，当时只有AISLE的G/B/S三条已审查，NBV及其他槽位标为未启动。这些状态是历史快照，不代表本文件核对时的最新进度。不能把它们作为“当前12条完整比较图”直接插入。当前正文优先使用上面的GT/G/S图；后续完整批次再用同一脚本和新快照生成最终矩阵。

局部图的通用英文图注可保留：*All four declared facilities are displayed at a common camera angle and physical scale. Each offline GT box is expanded by 0.18 m, and only original triangles wholly within the display box are shown. Precision, recall, and F1 come from the full-scene evaluation against the fixed observable reference; local crops do not enter scoring. Missing measurements are labeled according to the specified snapshot.*

## 8. 并稿时必须同步处理的三处衔接

1. 英文Section 3和毕业方法章第1节仍描述一个二值构型、已给设施坐标系与双模板。新增多实例段落必须标为独立扩展；不要把四构型公式直接接成同一个状态定义。
2. 旧局部评价使用公共设施ROI及“已知单元”覆盖，新场景级使用完整预测、四实例可观测参考和“观测为空闲”的可导航覆盖。两种J不可直接平均，也不能把旧ROI图替换成全场景图后沿用同一句指标定义。
3. 原摘要和结论的6.1638%、2.6241%、42步正收益以及54步反转继续对应原证据。新前端回放只增加“信息可用时间”的结果；共享规划优势需由共同前端内S/B的新闭环结果支持，不能从回放后验变化推断。

## 9. 段落数据来源

用于写作复核的来源集中列在这里，投稿正文无需反复出现文件名、SHA和内部版本号。

- 四方法AISLE数值与资格：[G review](/root/NSO/audit_results/article_stage_20260928/episode_reviews_v1/dev_AISLE_G_b160_n92801/review.json)、[B review](/root/NSO/audit_results/article_stage_20260928/episode_reviews_v1/dev_AISLE_B_b160_n92801/review.json)、[S review](/root/NSO/audit_results/article_stage_20260928/episode_reviews_v1/dev_AISLE_S_b160_n92801/review.json)、[NBV review](/root/NSO/audit_results/article_stage_20260928/episode_reviews_v1/dev_AISLE_NBV_b160_n92801/review.json)。四份review的文件manifest均重新验证；完整动作／姿态列以及预测vertices／triangles数组逐一核对相同。
- 关联回放：[比较摘要](/root/NSO/audit_results/article_stage_20260928/analysis_v1/ground_v2_attempt03/summary.json)、[完整解释](/root/NSO/docs/research/ARTICLE_GROUND_ASSOCIATION_REPLAY_20260928.md)。比较输出manifest复核通过；旧控制回放3,979,754个比较项完全相同。
- 实际全场景图：[图源说明](/root/NSO/docs/research/ARTICLE_SCENE_MESH_FIGURES_20260928.md)、[正式图包manifest](/root/NSO/docs/thesis/figures/article_scene_meshes_20260928/aisle_gbs_v2/manifest.json)。
- 共同数值处理及原失败区别：[独立设计说明](/root/NSO/docs/research/ARTICLE_COMMON_NUMERIC_EVALUATION_DESIGN_20260928.md)。
- 后续在线消融的范围：[固定设计](/root/NSO/docs/research/ARTICLE_GROUND_ABLATION_PREDECLARATION_20260928.md)。本插入稿不引用其尚未完成或未复核的结果。
