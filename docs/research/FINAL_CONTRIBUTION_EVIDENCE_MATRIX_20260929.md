# 最终贡献—相关方法—实现—证据矩阵

日期与原论文访问日期：**2026-09-29**。对应最终目标 P1，供小论文与毕业论文共用；不改实验协议、冻结源码或既有结果。以下定位依据已读原论文与本项目实际实现，是有范围的比较判断，不是对全部文献的首创性证明。

## 1. 锁定论文要解决的问题

**在设施构型尚有几何歧义、观察预算有限时，利用可被实测纠正的类别先验选择观察方向，并检验该选择对实际重建表面的价值。**

工业设施需要重新建档是应用背景；单次任务内环境静止。论文的主要研究对象是预算受限的主动观测分配。主线保留 OV-SDF、STGHP、RPN-UQ、IGCR 四接口与规划/执行层次：类别形成构型信念，规划比较带诊断预期的后续动作，执行检查完整姿态返航预算，实测几何再改变信念。TSDF 只融合取得的深度，公开模板用于决策预测。

最相关方法已经研究了语义设施检查、重复结构预测、任务相关选视以及语义—几何联合探索。本文应突出**类别信息在共同主动几何规划器中的具体决策作用及其条件**，不能将“使用语义”“可以补看”“贝叶斯更新”“覆盖乘质量”或四模块名称本身列为首创新算法。

## 2. 三条可辩护贡献

|编号与贡献|具体实现或分析|已有证据与剩余工作|贡献性质及放置|
|---|---|---|---|
|C1：类别条件的预算受限观察方法|公开双构型预测、类别先验与实测几何权重共同进入有限预算规划；每次前瞻最多包含一次诊断事件；只执行下一原子动作，并检查返航余量。G/S共享规划器、动作能力和重建后端。|V35/V36给出类别输入、同一非语义前缀、首次动作分歧及实测终点；V36为2胜2平、均值J5相对提高6.1638%、平均覆盖相同。最后8条新布局G/S验证尚未执行。|**任务化方法与系统组合贡献。** 引言第一点、方法主线。不是新的通用POMDP求解器，也不是四项独立网络创新。|
|C2：可解释的类别选向与几何纠错机制|简化二构型决策分析说明类别置信、诊断代价与方向效用的关系；鲁棒实测残差能够推翻有限错误先验。保存状态中的后验干预连接实测更新与下一动作。|V35第28步纠正、29步改选；两次首次纠正状态干预保持其他规划输入不变。X/Xnf开发均值提高2.6241%，其含义为完整纠错策略。|**机制分析与验证贡献。** 方法分析和结果机制段。标准决策/Bayes原则作为基础；不把完整策略差值归给单次更新。|
|C3：从决策到实测表面的条件性证据|共同预算下分列覆盖、P/R/F1、J与动作代价，展示完整实际路径、同尺度网格、五个固定重建时点；预算与同族几何扩展检验作用范围。|8条确认路线的40个检查点；128次扩展保留96合格与32低预算失败，B42原布局/变体增益为6.2505%/4.2289%，B54原布局差为−0.6995%。36次独立场景级扩展保留共享持平。|**实验与应用证据贡献。** 实验和讨论。过程测量不增加独立场景数；共享不是已证明的新优势。|

这三点构成“方法—作用机制—实测证据”的论文组织。C2、C3不另设算法发明，也不依靠将失败或持平删除来成立。新布局验证补的是 C1/C3 的空间迁移证据，不能预先写入正结果。

## 3. 最近相关方法与本文的具体差异

本表比较各原论文的方法定义，不排列性能高低。每行的“本文差异”来自本文代码与原论文之间的对照；没有运行的原系统不作为实测基线。

|原方法与已核验技术|本文具体差异|本文实现入口|对应实验或证据|
|---|---|---|---|
|[SWAP，ICRA 2023，§IV-A–D](https://arxiv.org/html/2303.07236v1)：组织体积探索、语义网格孔洞补看和按分辨率/角度要求检查表面；原方法无需给定目标形状，实机使用AprilTag提供语义对象识别。|本文在给定安全图和公开构型族中，研究**类别改变构型概率后如何提前选择方向**。G仍可主动诊断和补看，问题不是证明只有语义方法会补看。|`nso/observation_belief_v35.py`、`nso/online_planner_v35.py`；适配比较在`nso/external_planners_v39.py`。|V36同规划器G/S是语义增量的主要证据。SWAP-I为共同CPU机制比较，未复现原孔洞覆盖完整规划器，不能写成优于原SWAP。|
|[SPP，T-FR 2025，§4–5](https://arxiv.org/html/2506.06560v1)：从语义场景图发现精确或不精确的重复模式，预测未知区域中的结构位置，以两种预测检查策略提高巡检效率。|本文主线使用**给定类别—构型关系和当前实例的可纠正信念**，决策尺度是有限预算内的观察方向；没有实现SPP式未知空间图扩展。多实例共享属于另一个实现，结果须单列。|`nso/observation_belief_v35.py`；扩展对照见`nso/controller_article_v1.py`及其Ground/Exposure版本。|局部V35/V36与128扩展支持限定类别机制。36条场景级任务中9组S/B动作、质量相同，不能作为预测共享优于SPP的证据。|
|[VISTA，作者论文§I、III](https://arxiv.org/html/2507.01125v1)：将查询语义相关性与历史视角多样性结合，使用在线语义3DGS与滚动轨迹选择服务目标搜索。|本文的类别变量主要表示**潜在几何构型**，不是查询相关性权重；观察的价值通过构型信念和诊断后续选择进入预算规划。实测表面由共同TSDF评价。|`nso/online_planner_v35.py`、`nso/cpu_four_modules_v35.py`；VISTA-I在`nso/external_planners_v39.py`。|V36完整G/S链和V35错误先验链。VISTA-I仅适配已测方向及相关性机制；其分数不能代替原VISTA任务成功率或完整系统比较。|
|[GenNBV，CVPR 2024，§3.1–3.2](https://arxiv.org/html/2402.16174v2)：以强化学习学习5维视点策略，融合几何、图像语义和动作表示；其中语义表示由灰度图像序列网络提取。|本文使用**显式类别—构型先验与可读的诊断/返航约束**，在地面离散动作中分析类别条件化的作用，不训练跨对象策略。不能把GenNBV的图像表示等同本文人工类别概率表。|`nso/observation_belief_v35.py`、`nso/online_planner_v35.py`、`nso/cpu_four_modules_v35.py`。|预算和同族变体结果解释本文受控适用范围；没有GenNBV完整复现实验，也不主张其跨数据集能力或飞行自由度。|
|[ActiveSGM，NeurIPS 2025，§3.1–3.2](https://arxiv.org/html/2506.00225v2)：在语义高斯地图中使用稀疏类别分布，结合渲染轮廓缺失、语义熵及距离项选择视角；语义损失只更新语义属性。|本文的隐变量为**构型假设**，用公开预测与当前实测的差异形成残差证据，并显式考虑一次诊断后的选择。这个区别不能写成“已有方法没有语义不确定性或几何反馈”。|`nso/observation_belief_v35.py`、`nso/online_planner_v35.py`；实测评价`nso/surface_measurement_v34.py`。|首次类别分歧、首次纠正干预和重建检查点支撑本方法机制；没有ActiveSGM原系统指标排名。|

SWAP 和 SPP 是工业设施规划最直接的任务相关工作；VISTA 与 ActiveSGM 说明语义和几何共同引导视角并非空白；GenNBV说明图像信息辅助重建选视同样已有研究。本文的可辩护位置是上述具体条件下的显式先验决策组合与机制验证。CPU可运行和无需训练是实现特点，不单独当作首次技术创新。

## 4. 可直接并入正文的正向表述

### 中文引言贡献段落

本文面向工业设施的预算受限主动建档，研究几何尚有歧义时类别信息如何影响观察资源分配。主要贡献如下。第一，构建类别条件的主动观测方法，将设施构型先验、诊断前瞻、返航约束与实测表面融合连接为规划—执行闭环；几何对照保留相同的诊断、补看和重建能力，以检验类别信息的增量作用。第二，给出类别置信、诊断代价与观察方向效用之间的简化关系，并通过保存决策和同状态后验干预，解释类别提前选向及错误先验被几何证据纠正的执行机制。第三，以配对路径、实际TSDF网格、重建过程和预算/几何扩展建立从决策变化到实测质量的证据链。在原有两布局、双构型确认中，联合质量均值提高6.1638%，且平均二维覆盖保持相同；预算扩展同时揭示收益消失和反转的条件。

方法假设段保留一句：本文采用公开构型族、已给定安全图、准确模拟位姿和受控类别提示，类别—构型关系作为给定知识，验证范围为这些条件下的观察分配。

纠错结果段保留一句：2.6241%的错误先验对照收益属于包含实测更新和未来诊断预期的完整纠错策略；同状态干预单独支持后验更新对下一动作的影响。

多实例讨论段保留一句：场景级S/B检验额外同类共享，与局部S/G的类别可用性问题不同；当前完整九组S/B配对均未产生动作或质量差异。

### English contribution paragraph

This study addresses budget-constrained active documentation of industrial facilities, focusing on how category information allocates observations while geometry remains ambiguous. We make three contributions. First, we develop a category-conditioned observation method that connects configuration priors, diagnostic lookahead, return constraints, and measured surface reconstruction in a planning–execution loop. The geometric control retains the same diagnosis, revisiting, and reconstruction capabilities, enabling a matched test of the additional category information. Second, we analyze the relationship between category confidence, diagnostic cost, and directional utility, and use recorded decisions and same-state posterior interventions to explain early direction selection and the correction of misleading priors. Third, we connect these decisions to measured reconstruction through paired trajectories, actual TSDF meshes, fixed reconstruction checkpoints, and budget and geometry variations. In the original confirmation on two layouts with two configurations each, the mean joint quality improves by 6.1638% at identical mean planar coverage. The budget extension also identifies conditions in which this benefit disappears or reverses.

Corresponding method-scope sentence: The experiments assume a public configuration family, a supplied safe graph, exact simulated poses, and controlled category cues; the category–configuration relation is provided knowledge.

Corresponding correction-results sentence: The 2.6241% incorrect-prior improvement evaluates the complete correction policy, which includes measured updating and anticipated diagnosis; same-state interventions separately establish an effect of the posterior update on the next action.

Corresponding scene-extension sentence: Scene-level S/B tests additional sharing across instances, whereas local S/G tests category availability; all nine complete scene-level S/B pairs currently have identical actions and reconstruction scores.

### 可替换相关工作结尾的英文定位段

Semantic inspection connects target identity with surface observation requirements, as in [SWAP](https://arxiv.org/html/2303.07236v1), and [SPP](https://arxiv.org/html/2506.06560v1) exploits repeated semantic arrangements to predict inspection targets. [VISTA](https://arxiv.org/html/2507.01125v1) emphasizes query relevance and view diversity, while [ActiveSGM](https://arxiv.org/html/2506.00225v2) combines geometric coverage and semantic uncertainty. Our study focuses on a complementary question: when does a category-conditioned configuration prior change the choice between return-feasible observation routes, and how does measured geometry revise that choice? We examine this question with matched active controls and reconstruction from acquired depth. The contribution is the task-specific method and its decision-to-reconstruction evidence; Bayesian updating and semantic revisiting provide established foundations.

上述贡献段可替换当前引言末段；具体相关工作保留各原文引用。限制分别进入方法假设、相应实验图注和讨论，不在每段重复。最后8条补证的结果另填实验段；未运行前不加入摘要或贡献段。

## 5. 额外混淆检查决定：0条新增定向科学实验

**保持8条新布局G/S主验证，不额外预定标签打乱或单独纠错闭环。** 本决定在主验证成绩出现前做出，不以本轮结果好坏触发追加矩阵。

理由是目前三条贡献已对应到恰当的已有对照：

- “类别使用是否改变决策”由V36同一非语义前缀、同一规划器、G/S动作19分歧与真实终点支持。V35已有去类别前缀检查，移除类别后S与G的状态和首次决策一致。
- “类别与构型的关联是否重要”已有V35正确类别S与交换类别X：同一P00双构型、相同几何反馈能力下，原开发均值分别为0.7712658288和0.6722544223。它是既有开发关联扰动证据，不是自然识别可靠度验证或新的确认样本。继续新增随机标签任务不会使当前主张自动成为更原创的算法。
- “实际后验更新是否影响动作”已有两次预定首次纠正状态干预；“完整纠错策略是否改善终点”已有X/Xnf。本文明确不主张单独实测更新的独立终点收益，因此无需为该未提出的主张增加一个策略分支。

唯一待补的核心证据是新局部布局下的空间迁移。主验证仍严格执行最终目标中的2布局×2构型×G/S、B=42、冻结原控制器/指标/类别映射以及全部结果报告。两个布局是新增布局单位，双构型与噪声不变成额外布局。正、零或负结果都可完成检验；零/负时收缩外推，不重选场景或修改先验追正差。

最多8次预留不因此变成必做工作量，若有必要仅按总目标处理已记录工程失败。所有试跑、失败和重试计入16次总上限。本文件不授权重启跨设施共享48条矩阵，不新增自然识别、实车或完整SOTA复现要求。

## 6. 对应实现和内部证据入口

|用途|可复核入口|
|---|---|
|OV-SDF/IGCR：受控类别、重复抑制、鲁棒深度与扫描残差|[observation_belief_v35.py](/root/NSO/nso/observation_belief_v35.py:65)，`ObservationBeliefV35.update`|
|STGHP：两模板、一次诊断前瞻和剩余预算选择|[online_planner_v35.py](/root/NSO/nso/online_planner_v35.py:49)，`ForecastBeliefPlannerV35.select`|
|层次执行、公共前缀与RPN-UQ返航检查|[cpu_four_modules_v35.py](/root/NSO/nso/cpu_four_modules_v35.py:277)，`select_target`、`assess_action`|
|真实表面指标、公共ROI和空预测规则|[surface_measurement_v34.py](/root/NSO/nso/surface_measurement_v34.py:82)，`extract_observed_asset_mesh`、`SurfaceMeasurementV34.evaluate`|
|局部确认和原始四配对|[V36结果](/root/NSO/docs/research/V36_ONLINE_CONFIRMATION_RESULT_20260918.md)、[独立复核](/root/NSO/docs/research/V36_ONLINE_CONFIRMATION_INDEPENDENT_REVIEW_20260918.md)|
|类别关联交换、首次纠正和同状态干预|[V35结果](/root/NSO/docs/research/V35_ONLINE_SEMANTIC_RESULT_20260918.md)、[保存链条复核结果](/root/NSO/audit_results/v35_semantic_chain_review_20260918/result.json)|
|预算/几何边界与完整失败分母|[128次扩展结果](/root/NSO/docs/research/THESIS_EXPANSION_RESULT_20260928.md)、[完整配对CSV](/root/NSO/audit_results/thesis_expansion_20260928/analysis_review/complete_paired_metrics.csv)|
|实测过程曲线与同尺度网格|[重建检查点报告](/root/NSO/docs/research/ARTICLE_RECONSTRUCTION_CHECKPOINTS_20260928.md)、[40检查点CSV](/root/NSO/audit_results/article_stage_20260928/reconstruction_checkpoints/checkpoint_metrics.csv)|
|共同CPU机制比较范围|[V39结果](/root/NSO/docs/research/V39_EXTERNAL_CPU_RESULTS_20260920.md)、[适配代码](/root/NSO/nso/external_planners_v39.py)|
|跨设施扩展的实际结果边界|[36条最终报告](/root/NSO/docs/research/ARTICLE_EXPOSURE_ABLATION_RESULT_20260928.md)、[完整36行数据](/root/NSO/audit_results/article_stage_20260928/analysis_v1/article36_final_summary_v1/rows36.csv)|

局部J5与场景级J_nav的覆盖定义、预测评价范围和策略S含义不同，保留分节报告。方法代码可追踪并不等于性能贡献已单独消融；对照表只将已经观察到的效应写成结论。

## 7. 原论文核验记录

全部链接于 **2026-09-29** 实际访问；方法判断依据以下作者全文，而非搜索摘要、第三方评论或二手排行榜。

1. Dharmadhikari M., Alexis K. *Semantics-aware Exploration and Inspection Path Planning*. ICRA 2023。已读[作者全文v1](https://arxiv.org/html/2303.07236v1) §IV-A–D、§V-B；[作者摘要与会议说明](https://arxiv.org/abs/2303.07236)。
2. Dharmadhikari M., Alexis K. *Semantics-aware Predictive Inspection Path Planning*. IEEE Transactions on Field Robotics, 2025。已读[作者全文v1](https://arxiv.org/html/2506.06560v1) §4、§5、§6.B；[作者页确认T-FR接收](https://arxiv.org/abs/2506.06560)，[实验室出版目录](https://www.autonomousrobotslab.com/publications.html)提供DOI。DOI页面本次不可访问，未假称读取出版社全文。
3. Nagami K. et al. *VISTA: Open-Vocabulary, Task-Relevant Robot Exploration with Online Semantic Gaussian Splatting*. 本次技术对照使用[作者全文v1](https://arxiv.org/html/2507.01125v1)，[作者版本记录](https://arxiv.org/abs/2507.01125)仅列2025-07-01 v1。现稿RA-L 2026书目信息可在期刊适配时沿用已核验记录；本次DOI页面不可访问，不将作者v1与出版社最终版逐字一致作为事实。
4. Chen X., Li Q., Wang T., Xue T., Pang J. *GenNBV: Generalizable Next-Best-View Policy for Active 3D Reconstruction*. CVPR 2024。已读[作者全文v2](https://arxiv.org/html/2402.16174v2) §3与[作者项目页](https://gennbv.tech/)，后者明确会议。CVF落地页本次返回403，使用作者全文完成方法核验。
5. Chen L., Zhan H., Yin H., Xu Y., Mordohai P. *Understanding while Exploring: Semantics-driven Active Mapping*. NeurIPS 2025。已读[作者全文v2](https://arxiv.org/html/2506.00225v2) §3.1–3.2及[会议稿](https://openreview.net/pdf?id=RkHUDvy9QR)；v2日期由[作者页](https://arxiv.org/abs/2506.00225)确认。

没有从这五篇的对比推出“本文首次提出正确先验优于未知先验”或“尚无其他工作使用可纠正构型先验”。P1完成的是具体贡献与最近相关实现的清晰定位；编辑与同行对新颖性和刊物匹配的判断属于投稿流程。
