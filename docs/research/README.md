# NSO 论文研究材料

> **后续执行入口（2026-09-11）：** [关键目标导向与论文主线](RESEARCH_GOAL_AND_STORY.md)、[V11 网络训练与数据交接](V11_NETWORK_TRAINING_AND_DATA_ADEQUACY_HANDOFF.md)、[可复制 Goal 文本](GOAL_PROMPT.md)。V10.3 的反馈版本遗漏、指标解释和局部质量检查已由 [V10.3.1 修订](V10_3_1_FEEDBACK_CORRECTION_AND_METRIC_CONTRACT.md)完成；当前进入可校正语义收益模型的正式训练与复核。以下旧阶段进展不代表整体验收已通过。

当前进展：[V10.3语义覆盖结果、理论与论文边界](V10_3_SEMANTIC_COVERAGE_RESULT_AND_THEORY.md)。保留四模块与ANS层次，修复后40个紧凑闭环分支全部安全；特殊人工资产开发域中，S相对G/N的检查联合指标均值分别提高29.02%/18.07%，错误语义X在8/8配对历史退化。S与关闭反馈仍8/8相同，H3未成立；人工标记、非未知起点、非外部主流实现等限制同步保留。

最新进展：固定双对象 V8.1 的24条路线、752动作全部完成并通过[独立回放](COMPETITION_V8_1_INDEPENDENT_REPLAY_20260911.md)。[原协议结果](../../eval_results/competition_v8_1_analysis_20260911/REPORT.md)显示语义面积率较内部G/O高10.55%，但终点F1低0.00291、8条deep跨对象排他条件全部失败，因此原进阶判定不变。[单项事后目标诊断](../../eval_results/competition_v8_1_objective_diagnosis_20260911/REPORT.md)统一比较总潜在收益后，全部方法选择42动作路线，S新增面积高34.78%、F1高0.00979；它是开发依据，不能计作独立验证。见[预算一致的下一阶段设计](SEMANTIC_BUDGET_ALIGNMENT_V9_DESIGN.md)。

[首次V8构造](COMPETITION_V8_CONSTRUCTION_AUDIT_20260911.md)的目标坐标门槛失败已保留；[V8.1修订](SEMANTIC_COMPETITION_V8_1_REVISION.md)只统一实测点云／占据格导航边界，复用原始前缀，评分先验未改。10条旧路线守卫副作用已全部[执行并回放](EXECUTION_GUARD_V8_CHANGED_ROUTE_REPORT_20260911.md)，9返程1停机、零碰撞，不能当作原任务全部完成。


本目录围绕“保留 ANS 双层架构与四模块、保留可验证语义创新”的约束维护。研究主线为覆盖约束下的语义条件有效观测规划。当前完成了文献研究、理论推导、原架构最小接线及 CPU V4 开发对照；尚未证明完整系统优于主流方法。

| 材料 | 用途 |
|---|---|
| [V11 网络训练与数据交接](V11_NETWORK_TRAINING_AND_DATA_ADEQUACY_HANDOFF.md) | 说明 V10 未训练原神经栈、V7 主任务信息价值为零、V9 可作开发训练及下一阶段唯一执行顺序 |
| [网络训练审计与优化路线](NETWORK_TRAINING_AUDIT_AND_OPTIMIZATION_PLAN.md) | 区分 V10 CPU 机制、历史轻量拟合、不可用 LFS 权重，并定义学习型语义收益头与逐模块优化顺序 |
| [V10四模块实现与失败边界](FOUR_MODULE_CPU_CLOSED_LOOP_V10_IMPLEMENTATION.md) | 真实传感接口、预算和反馈、语义干预、原地旋转问题与下一轮修正 |
| [综合方案与验证设计](NSO_SEMANTIC_COVERAGE_REDESIGN.md) | 任务定位、根因、四模块改造、实验与实车前置工作 |
| [原始文献研究](SEMANTIC_COVERAGE_LITERATURE.md) | 16 篇论文的研究内容、最接近前作、限制和官方基线准备 |
| [来源清单](source_inventory.json) | 阅读深度、发表状态、仓库固定版本与可引用主张 |
| [BibTeX](semantic_coverage_references.bib) | 已核对的近期论文引用，不补造未核实出版信息 |
| [数学理论与边界](COVERAGE_SEMANTIC_THEORY.md) | 条件命题、证明、反例、与实现的区别 |
| [数学数值检查](semantic_theory_numerical_checks.json) | 固定代理、延迟界、风险界和重复收益算例 |
| [架构与基线审计](ARCHITECTURE_INVARIANTS_AND_BASELINE_AUDIT.md) | 真实执行链、最小接线、CPU/原 ANS/实车的证据边界 |
| [旧参考面审计](LEGACY_SURFACE_REFERENCE_AUDIT.md) | 组合盒网格与唯一外表面的权重差异，不将面积差冒充指标偏差 |
| [反事实视角价值协议](COUNTERFACTUAL_VIEW_UTILITY_PROTOCOL.md) | 共同历史、候选、真实动作成本及新观测价值；先验证语义机制 |
| [候选充分性审计](CANDIDATE_SUFFICIENCY_AUDIT_20260911.md) | 定位对象身份、朝向压缩与晚期零收益通道 |
| [直接观测响应理论](SEMANTIC_VIEW_UTILITY_V6_THEORY.md) | 条件响应、训练／预测分离、信息价值及其适用条件 |
| [响应探针独立审阅](SEMANTIC_VIEW_RESPONSE_V6_REVIEW.md) | 独立SVD重拟合、标签隔离、外推和未过关原因 |
| [有限验证协议](SEMANTIC_V6_NEXT_VALIDATION_PROTOCOL.md) | 配对场景、有限训练／校准、强对照与存储规则；现已执行T，首次C前停止 |
| [V7实施与证据](SEMANTIC_RESPONSE_V7_IMPLEMENTATION_AND_EVIDENCE.md) | 逐点方向修正、配对前缀、96条训练分支、零类别决策价值及在线执行缺口 |
| [V7执行审查](RESPONSE_V7_EXECUTION_STATIC_REVIEW_20260911.md) | 真值隔离、动作计费、唯一外表面及封存边界 |
| [V7拟合审查](RESPONSE_V7_FIT_STATIC_REVIEW_20260911.md) | T内分组拟合、强对照和全C预测先封存门禁 |
| [网格归档与恢复审查](RESPONSE_V7_MESH_COMPACTION_REVIEW_20260911.md) | 保留原始观测，先回放后归档，以及逐文件精确恢复证据 |
| [V8执行保护回归](EXECUTION_GUARD_V8_FAILURE_REGRESSION_20260911.md) | 4条已知碰撞修复并实际返回；另6条成功路线受影响，尚需验证副作用 |

正式论文使用顺序：先固定问题和适用域，再对应已实现方法；实验数字从通过复核的目录引用，独立确认与开发实验分开；最后按证据写结论。局部理论性质不等于整体最优，合成可见语义不等于开放词汇识别，规则 CPU 原型不等于完整神经系统。

原论文中缺少记录的成绩与不准确引用需要替换；历史草稿只保留用于追溯，不作为实验依据。当前有效材料以此目录、源码和带配置/源码快照的实验记录共同为准。

补充材料：[设施场景规范](INSPECTION_SCENE_SPEC.md)、[论文主张证据账本](THESIS_CLAIM_EVIDENCE_LEDGER.md)、[官方源码及哈希](official_baseline_source_manifest.json)。新场景已支持直接从可见 RGB 铭牌读取语义的 CPU 通路；这属于人工标识设施条件，尚非自然图像开放词汇验证。

最新实验证据：[72回合V4开发与匹配消融](../../eval_results/semantic_joint_v4_development_review_20260910/results.md)。覆盖保护改善了覆盖代价，但隐藏表面模块仍负贡献，去反馈轨迹不变；独立确认尚未启动。[可编辑研究报告](NSO_SEMANTIC_COVERAGE_REDESIGN.docx)。

评价复核现已覆盖[64条轨迹、929个检查点](../../eval_results/legacy_surface_posthoc_auc_20260910/INTERPRETATION.md)：唯一外表面口径下 V4 相对纯覆盖的联合 AUC 约高4.61%，但隐藏模型净贡献仍负、反馈仍无差异。[终点口径复核](../../eval_results/legacy_surface_posthoc_endpoint_20260910/results.md)同时保留，不能用AUC替代终点结论。

[同历史协议](COUNTERFACTUAL_VIEW_UTILITY_PROTOCOL.md)首批已执行[72条真实分支](../../eval_results/counterfactual_view_development_first_20260910/analysis/results.md)并完整回放通过：6个可用历史均有标记，但语义改变选择0次，另2个预定历史因前缀提前结束不可用。先根据[评分诊断](COUNTERFACTUAL_SCORE_DIAGNOSIS.md)和[V5内部修正](SEMANTIC_COMPLETION_V5_DESIGN.md)解决机制问题；753/754及独立测试尚未启动。

[两模型×两评分目标重评分](../../eval_results/counterfactual_view_v5_rescore_first_20260910/results.md)已完成：共同预算AUC改善部分均值，但细类别仍没有选择增益，去隐藏模型仍更好。20项相关模型/评分检查通过，原6历史分数精确复现、997个源输入哈希复核；技术验证与效能门槛分别记录。

2026-09-11新增证据：[固定时序审计](../../eval_results/counterfactual_prefix_timing_first_20260911/TIMING_AUDIT.md)的64个可用历史发现2次G/S分歧；[全部分歧路线执行](../../eval_results/counterfactual_timing_divergences_first_20260911/DIVERGENCE_EXECUTION_REPORT.md)完成4条路线、100动作，均回放通过。局部语义选择有真实收益，但较明显一次N与S同选，完整净优势仍未成立。

[新增16条路线的完整联合池比较](../../eval_results/counterfactual_union_pool_v6_20260911/results.md)将每历史原12条与新增最多4条共同评分，原rate面积率提高约25.32%，六评分器仍同选；共同预算AUC下去隐藏模型仍更好。新增660动作均回放通过，不能把候选生成的共同收益当语义收益。

[V6响应探针](../../eval_results/semantic_view_response_v6_review_20260911/REVIEW.md)的语义版本未超过O/X/N的面积率，[冻结模型迁移](../../eval_results/semantic_view_response_v6_candidate_transfer_20260911/REVIEW.md)在新增候选后仍未改选。两项独立数值审阅通过，仅证明计算与隔离约束，不能证明模型有效。753/754和独立种子继续保留。

V7最新结论：[96条训练分支](../../eval_results/response_v7_training_20260911/ACQUISITION_REPORT.md)已全部留档与回放，4条碰撞失败保留。16个历史的语义／几何／对象性等9通道仍全部同选，8对世界的当前面积率理想类别信息价值均为零。[独立SVD复核](../../eval_results/response_v7_T_fit_independent_review_20260911/FIT_SVD_REVIEW.md)确认5模型、120个训练内折和144条选择一致，但这不证明语义有效。首次C前已停止，先修最新地图执行门禁并诊断任务的决策差异；四模块与ANS层次保留。
