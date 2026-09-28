# 英文稿与毕业章节主张审查

2026-09-28，只读审稿；未改正文、PDF、实验数据或冻结源码。依据当前英文稿、毕业方法/实验章、完整原12条开发记录、统一数值补测、语义模型范围审查及已审查保存候选rho分析。Ground/V3完整表待主线程合并，本文不把仍在执行的结果预写为成功或失败。

**判断：现有文章的核心仍可成立为“受控局部双构型任务中，类别输入与整体纠错政策具有条件作用”。它不是“共享可靠性模块已经证明有效”的文章。正文后半部大体保持了这一区分，但摘要/结论、符号复用和新共享理论的两处措辞容易让读者把旧正结果误接到新系统。建议优先处理以下六项。**

## 1. 摘要与结论补齐新场景的结论边界

位置：`VIRTUAL_PAPER_EN_20260928.md:7`、`:407`。摘要只汇总局部6.1638%、纠错2.6241%和128项扩展；文章现在又包含场景级共享实现与12条开发结果，若摘要/结论完全不提，会造成“全文最新实现也获验证”的印象。现有数值本身无需撤回。

建议在摘要末句前添加：

> A separate four-facility development study yielded identical outcomes for the shared-reliability and fixed category-prior controls across three layouts; the local category benefit therefore does not establish an additional benefit from cross-instance sharing.

结论可接：

> The demonstrated benefit is restricted to the local category-conditioning and correction experiments. The scene-level implementation establishes an executable testbed, while an additional shared-reliability benefit remains unestablished.

中文对应：已有证据支持的是局部任务下的类别条件与整体纠错作用；场景级实现提供了可复核的执行流程，但目前尚未证实共享可靠性相对固定类别先验的额外收益。

毕业实验章`:327`已明确“三布局S/B持平、正结论仍来自局部实验”，可以保留这条边界；不能在后续整合时被更宽的“验证整体架构优越性”替换。

## 2. 同一个S符号正在表示两种不同干预

位置：英文`:233`、`:247`、`:259`、`:377`；毕业方法章`:241`、`:291–300`；实验章`:251`。

旧局部S是“开启类别先验”的策略，旧G/S检验类别可用性；新场景B已经开启固定类别先验，新S再加同类可靠性共享，新S/B才检验共享。仅靠后文一句“S/B tests sharing”不足以避免把旧6.16%称为共享模块结果。两版结构维数、先验强度、规划器和指标也不同，不能把旧S视为新B的相同算法。

建议在首个新场景段落或一个短对照表注明：

> The policy labels are local to each implementation: S denotes category conditioning in the binary experiments, whereas scene-level B already includes category conditioning and scene-level S additionally shares class-model evidence. These comparisons answer different questions and their effect sizes are not pooled.

若不改已封存图中的G/S标注，正文和表题可使用“local S”和“scene-level S”明确指代。

## 3. rho界约束模型权重，不直接限制完整后验集中程度

位置：英文`:239`；毕业方法章`:339–351`。

英文“a weak category prior cannot become an arbitrarily concentrated configuration belief”容易把对\(\rho\)的界写成对最终构型后验的界。自身几何证据仍能使后验高度集中。后面的“a candidate-pair contrast reaches its extrema at the interval endpoints”也应明确为**每实例贡献**；多实例不同线性分式相加未必是一个导数恒号的线性分式。中文已经明确逐实例端点再求保守和，较准确。

建议替换英文对应两句：

> A single peer therefore bounds the model-mixing weight, not the concentration produced by the target's own geometric evidence. For fixed own evidence, each instance's contribution to a fixed candidate-pair score difference is linear fractional in its mixing weight and attains its extrema at the interval endpoints. Summing these extrema yields a conservative envelope; the per-instance endpoints need not be jointly realizable.

不要把active-prior TV上界直接代替posterior TV上界。本次保存分析中LOOP端点posterior TV约0.019880，已高于对应active-prior界0.018750，说明这一区分确实影响数值。解析界仅适用于当前至多一个合格peer的状态；物理每类别两个设备不能在关联出错时自动保证在线peer数也至多一个。

## 4. 两段理论中的rho含义不同；理想诊断例子不是共享增益证明

位置：英文`:200–221`与`:227–239`；毕业方法章`:156–177`与`:284–291`。

前者\(\rho\)是简化二元观测模型中类别线索的校准对称准确率；后者\(\rho_i\)是均匀先验与固定类别表之间的广义贝叶斯模型权重。它们不能相互代入。建议将前者改记为\(r_c\)，并加一句：

> This two-route calculation concerns the value of an available category cue; it is not a performance guarantee for the scene-level shared-model weight.

条件独立、二元对称通道、诊断后两路线仍可行等假设已经列出；诊断闭式表达式与数值算例没有发现算术错误。返航不变量也已限制为固定图、准确代价和成功执行，并区分“存在返航路径”和“实际返航成功”，这些限定应保留。

## 5. 不应继续把共享零收益仅解释为信息到达晚

位置：英文`:387–395`、毕业实验章`:301–315`；新增方法章`:339–351`。

固定路径地面修正确实把首次有信息peer从136提前到82，但这不是共同闭环收益。更关键的新证据是：已审查V1与Ground AISLE B/G共215个direct池重建成功；有类别的B/S共138池，在当前资格和假想一名合格peer两种可达范围下均能排除直接top-1变化。即便把资格时机乐观放宽，当前固定候选的one-peer信号仍不足越过分数间隔。这个结论不能扩到未分析Ground/V3轨迹、诊断分支或未来候选。

建议在固定路径分析后加一句简短结果，而非只写“需要更早信息”：

> Saved-candidate analysis also found that the one-peer reliability range could not change direct-view ranking in the reviewed category-conditioned states, even when peer availability was relaxed. Association timing is therefore only one bottleneck; actionable belief strength and the accuracy of predicted observation value must also be examined.

另一项重要设计边界来自 `ARTICLE_SEMANTIC_MODEL_SCOPE_20260928.md`：各类别构型由固定0.25/0.75分位确定，DEV/T0/T1同族的类别—构型多重集合相同；它们不是从共享潜在可靠性域独立采样。主48如执行，可检验固定设备组合下的空间布局外推，不能表述为学习或适应新类别关系分布。建议在新场景设置加一句此事实，避免把空间外推称作语义分布泛化。

## 6. 局部/原多实例/Ground/Exposure版本与图的归属要显式列出

位置：英文`:66–79`、`:223–239`、`:363–395`；毕业方法章`:278–337`；实验章`:40–42`、`:247–327`。

系统图与摘要说“直接选择下一原子动作”，这是局部二元实现；新场景先选择宏动作再由局部执行器分步执行。方法章已加入Ground与Exposure，而当前场景结果表仍是原ArticleV1。若统一称“本系统”，读者可能以为新曝光规则产生了表6中的旧轨迹。

建议给图2/毕业方法图1加“local binary implementation”范围；新章节前加一张四行版本表，列出唯一共同前端变化、对应已执行批次、评价版本，声明历史结果不追溯采用新控制器。表6继续明确“original scene-level development”。主线程稍后再按完整槽位追加Ground与Exposure表，不替换原表。

## 可保持的现有表述

当前正文已经正确区分局部\(J_5\)与完整网格\(J_{\rm nav}\)、原CELL/G评价失败与共同版本补测、整体纠错与单反馈消融、SWAP-I/VISTA-I机制适配与完整外部算法排名。128次试验、40检查点及相同轨迹也没有被写成独立布局样本。图中真实网格与示意图/生成式平台图来源已有明确标注。这些边界不需要因本次共享零结果重写或削弱。

审查结束时正文SHA：英文 `f529f6e6db351ede7d946d54a488b2a023b5c47a36f36c2540f27a6cc9c0b374`；方法章 `0217a8f1e907e7dfb9b4f02b75bdf9e5c123a17bb0c55732b55a00a0fbc84038`；实验章 `bb0c0a1902c781f22f355778dfcd945413f2055802a882cb280af9c24ab6a544`。正文仍在主线程编辑，行号仅对应本次快照。
