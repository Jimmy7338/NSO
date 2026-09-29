# 中文摘要

## 摘要

工业生产中的设备、工位与物料布局调整，会带来重新获取环境几何的需求。受遮挡和相机视场限制，较高的二维覆盖率并不意味着设施背面、开口和凹入表面已被充分重建。本文研究静态任务内的预算受限主动观测，考察类别知识何时能够帮助选择有效方向，以及实测几何如何纠正错误先验。

本文通过OV-SDF、STGHP、RPN-UQ、IGCR四个接口连接观测、构型信念、诊断规划和返航检查，构建无独立显卡条件下的分层决策与执行框架。局部双构型实现选择下一原子动作，多实例扩展选择付费观察宏动作并逐步执行。公开模板只用于预测，实际三维地图仅融合已经取得的深度；规划代理与终点评价分别计算。

局部配对确认在两个已见父布局、每布局两种构型上取得两胜两平，相对共同主动几何对照，覆盖率与设施表面F1@5 cm乘积的平均值提高6.16%，平均覆盖率相同。错误类别实验中，包含实测更新与未来纠错预期的整体政策提高平均联合指标2.62%。另在两种预先冻结的新背景布局中执行八次任务，四组配对一胜三平，平均联合指标提高2.98%；该结果检验同设施族的背景布局变化。另一个固定128项扩展保留96项合格结果与32项低预算规划失败；42步原布局及同族变体的类别收益分别为6.25%和4.23%，54步原布局则为−0.70%。

独立场景级研究在三个开发布局上完成原开发、共同地面关联消融及历史曝光消融各12次，共36次尝试。原流程的一项评价失败与共同数值补测分别保留，其余35项合格。共同组件的效果随布局和方法改变，三个版本中的共享可靠性S与固定类别先验B均无行动或质量差异。动作级复核另表明，类别可以改变所选目标却降低端点质量，几何诊断规划可以改变路径却未在该诊断帧获得预期信息。结果支持局部类别利用与纠错的条件作用，未建立额外共享收益。本研究验证受控输入与准确位姿下的主动观测和实测建档流程，不宣称SLAM定位精度提升或普遍的跨实例语义优势。

**关键词：**工业设施；主动建图；类别先验；几何反馈；预算约束规划

## Abstract

Changes to industrial equipment and workstation layouts create a need to reacquire environmental geometry. Occlusion and a limited camera field of view mean that high two-dimensional coverage does not ensure complete reconstruction of facility surfaces. This thesis studies budget-constrained active observation in static mapping episodes, asking when category knowledge can guide useful viewing directions and how measured geometry can correct misleading priors.

A CPU framework connects observations, configuration beliefs, diagnostic planning, and return checks through four module interfaces. The local binary implementation selects atomic actions; a separate multi-instance extension selects paid observation macros for primitive execution. Public templates support prediction, while three-dimensional maps contain only fused acquired depth. Planning proxies remain separate from measured endpoint quality.

Local confirmation uses two previously seen layouts with two configurations each. Compared with a shared active geometric control, category conditioning achieves two wins and two ties, improving the mean coverage–surface-F1 product by 6.16% at identical mean coverage. A complete correction policy improves the mean joint score by 2.62% under incorrect category inputs. Eight further tasks in two preregistered background layouts give one win and three ties, with a 2.98% mean joint gain under the frozen policy and the same facility family. A fixed 128-trial extension retains 96 qualified completions and 32 low-budget planning failures. At budget 42, category gains are 6.25% on original layouts and 4.23% on same-family variants; the original-layout difference becomes −0.70% at budget 54.

A separate scene-level study completes 36 attempts across three development layouts and three controller versions: original, ground-aware, and exposure-discounted. One original evaluation failure and its numerical supplement remain distinguished; the other 35 attempts qualify. Component effects depend on layout and policy, while shared reliability S and fixed category prior B have identical actions and quality within every version. Saved decision sequences further show that an active category cue can lower quality, and diagnostic planning can change a path without obtaining the anticipated feedback at the diagnostic frame. The evidence supports conditional local category and correction benefits, but no additional sharing benefit. The thesis validates controlled observation and reconstruction with exact poses, not improved SLAM localization accuracy.

**Keywords:** industrial facilities; active mapping; category priors; geometric feedback; budget-constrained planning
