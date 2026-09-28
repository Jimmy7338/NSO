# 小论文贡献定位与英文结果/讨论草稿

2026-09-28；独立交付，未改英文稿或毕业论文正文。范围固定为既有局部实验、初始场景级 12 条、Ground 12 条，以及已复核 Exposure AISLE/CELL。此稿不纳入尚未完整交付的 Exposure LOOP，也不将部分结果合成完整新矩阵均值。

## 中文定位建议

小论文仍以“预算受限设施建档中的类别先验与实测几何反馈”为主线：可验证语义贡献落在几何尚有歧义时提前选择观察方向，以及错误先验下的反馈纠正。四模块是完整实现接口，不能写成四个各自获益的新算法。同类共享保留为场景级扩展和负面发现，暂不列为已证实创新。新增大场景用于说明适用条件、共同前端和预测效用的局限；不要拿 CELL 的 G/NBV 优势替代语义优势。以模拟研究投稿，实车仅作可选补充；是否接收取决于贡献清晰度与期刊匹配，不以必须实车或必须共享有效为前提。

## English manuscript-ready draft

### Recommended title and contribution positioning

**Category Priors and Geometric Feedback for Budget-Constrained Active Observation of Facilities**

This study addresses facility documentation after industrial layout reconfiguration, with a static environment during each mapping episode. Its question is whether category knowledge can help allocate a limited observation budget when current geometry cannot distinguish useful viewing directions. The contribution is an executable semantic–geometric observation policy and a controlled account of its operating conditions. It is not a new semantic recognition network, a general solution to unknown-environment SLAM, or an independently validated improvement from every architectural module.

Three contributions should organize the paper. First, a hierarchical observation and execution loop connects category-conditioned configuration beliefs, measured geometric correction, paid diagnostic observations, and return-budget checks. Public templates guide decisions, whereas only acquired depth contributes to the reconstructed surface. Second, paired geometric controls isolate category conditioning and whole-policy correction in local configuration-ambiguity tasks, retaining ties, budget failures, and reversals. Third, a separate multi-instance extension connects saved beliefs and candidate values to executed paths and measured surface quality, revealing when additional perception or predictive planning changes behavior without guaranteeing better reconstruction. Leave-one-out reliability sharing remains an evaluated extension, not a demonstrated performance contribution.

### Local category evidence

The strongest positive semantic evidence comes from the local paired experiment. Across two previously seen parent layouts and two configurations per layout, category conditioning raises the equal-parent mean joint score from 0.702576 to 0.745881, a 6.1638% increase. The four paired conditions contain two wins and two ties, with identical mean coverage. Because the geometric control uses the same templates, planner, measured feedback, and reconstruction backend, the difference concerns the additional category cue rather than permission to revisit or diagnose. These tasks establish a conditional benefit for choosing observation directions before geometry resolves the configuration; they do not establish a general benefit across industrial environments.

The separate incorrect-category experiment yields a 2.6241% whole-policy correction benefit, including measured correction and anticipation of future correction. It does not isolate those two mechanisms individually. A fixed 128-trial extension retains 96 qualified completions and 32 low-budget planning failures. At budget 42, category gains are 6.25% on original layouts and 4.23% on same-family variants; at budget 54, the original-layout difference becomes −0.70%. The budget dependence is therefore part of the result. The executed finite-horizon policy and uncalibrated exposure model do not inherit a guarantee that additional observations improve measured quality.

### Scene-level behavior and a category counterexample

The initial three-layout, four-method extension produces 11 original pipeline completions and one evaluation failure after valid CELL/G motion. A common numerical surface treatment preserves this original failure while providing a separately derived score. Under the common measurement, shared reliability S and fixed category prior B tie in all three layouts. A subsequent 12-episode common ground-association revision improves LOOP but reduces AISLE and CELL scores. Earlier instance availability is thus an enabling condition for planning, not evidence that the resulting decisions are beneficial. These versions should remain separate experimental batches rather than be pooled as interchangeable repetitions.

The exposure-deduplicated AISLE runs provide a direct category counterexample. B and S reach J = 0.734222, compared with 0.782938 for G and NBV. In the B/G pair, physical observations are identical through frame 31 and measured geometric evidence agrees between controls over that prefix. Category-conditioned beliefs nevertheless select different targets at decision 27, followed by opposite turns at paid action 32. The semantic cue is operationally active here, but its resulting trajectory has lower endpoint quality. This example distinguishes a verified semantic influence from a favorable semantic effect. It also shows why changing a posterior, selecting a different view, and improving reconstruction must be examined separately.

### Diagnostic planning changes actions, but predicted information is not measured information

In the reviewed Exposure CELL pair, G and myopic NBV share identical physical observations through frame 19. At decision 17, their 32 direct candidates have identical scores. G additionally considers seven paid diagnostic candidates and selects a diagnostic score of 0.082740, above the best direct score of 0.078458. The selected option's anticipated information increment is 0.006106. Both alternatives satisfy the actual budget and execution guards. The first action divergence occurs at paid action 20: G observes in place, while NBV turns. Their committed macros complete at frames 20 and 29, respectively, and both episodes return safely.

However, G's diagnostic observation applies no fresh geometric feedback, leaves its uniform structure posterior unchanged, and adds no known two-dimensional cells at that frame. The final joint scores are 0.786500 for G and 0.515457 for NBV, after 24 m and 20 m of travel. These endpoints support a useful planning-component case under one development condition, and the trace establishes that diagnostic lookahead changes an executed macro. They do not demonstrate realization of the predicted information value or isolate the effect of that single observation from subsequent replanning. Both controls are geometry-only, so this comparison cannot serve as evidence for semantic sharing.

### Interpretation, limitations, and publication scope

No additional S/B quality benefit has been established: the original and ground-aware versions tie on all three layouts, and the reviewed Exposure AISLE and CELL endpoints also tie. The unfinished Exposure LOOP batch is outside this discussion. Shared evidence can arrive too late, alter weights without changing feasible choices, or interact with an inaccurate observation-value proxy. These are distinct explanations; none should be inferred solely from an endpoint tie. A bounded sensitivity analysis supports only fixed direct-view candidate comparisons, not arbitrary diagnostic replanning or reconstruction guarantees.

The supported conclusion is that category priors can improve early observation allocation under identifiable configuration ambiguity, while their benefit depends on geometry, budget, and the agreement between predicted and measured value. Public templates, synthetic category cues, exact poses, and a supplied safe graph delimit the present simulation evidence. Local region-based evaluation and scene-level full-prediction evaluation must remain distinguished. Paired scene results should include all declared methods and signed outcomes, with layouts rather than frames treated as descriptive units. Physical deployment may supplement this simulation study; it is not needed to restate the narrower contribution supported by the existing evidence.

## 整合提示

- 主摘要与贡献仍以局部类别增益、完整反馈策略和可执行闭环为中心；勿新增“共享泛化已证实”“复杂场景一致优越”或“所有模块均有效”。
- 既有局部四条件图保留；场景级图使用全部布局固定 G 的共同模块比较和完整四方法补充图。AISLE B/G 负例与 CELL G/NBV 机制链可作为不同作用来源的两个案例。
- Exposure LOOP 交付后，再依据完整新矩阵更新末段与表格；当前不能据 AISLE/CELL 推断它，也不计算缺布局的总体均值。
- SWAP-I/VISTA-I 继续标为共同条件下的机制适配，不能改写成原完整系统的复现或普遍 SOTA 优势。实车章节可以后补，不阻塞以模拟证据为主的小论文。

## 本次实际查阅的证据

1. `docs/thesis/VIRTUAL_PAPER_EN_20260928.md`：局部确认、错误类别与固定 128 条扩展的已整合数值；本稿不改该文件。
2. `audit_results/article_stage_20260928/analysis_v1/ground12_complete/`：完整原始/Ground 配对；manifest SHA `f6166a2859d8090f88b193929dc8b9ce4ab488d4d63478a5b60c1a02e1a6f47f`，9 个输出文件逐一核对。
3. `audit_results/article_stage_20260928/analysis_v1/exposure_AISLE_mechanism_final/`：类别作用的同实际输入前缀；manifest SHA `e796ba7cd3fcfec66b34af252b6e56ec5102a68e9088d27d33907dddd9dfb9e6`，6 个输出文件逐一核对。
4. `audit_results/article_stage_20260928/analysis_v1/exposure_cell_g_nbv_lookahead/`：已完成付费宏动作及未兑现信息收益；manifest SHA `90f9c629f1c25fd2eedcfb42bf13696852f241a827f480e5d51b9cb502705f39`，9 个输出文件逐一核对。
5. `audit_results/article_stage_20260928/episode_reviews_exposure_v3/exposure_{AISLE,CELL}_{G,B,S,NBV}_b160_n92801/`：8 条原流程合格与端点评价；各复核 manifest 中输出 SHA 已核对。这里只据 CELL B/S 已复核同分报告无质量收益，未自行宣称其全部原始路径相同。

未读取 Exposure LOOP 结果；未创建新的矩阵快照、实验、评价或图。
