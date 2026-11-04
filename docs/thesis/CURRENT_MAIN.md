# 当前小论文主稿

2026-11-04 更新实车证据及未知探索对照后的研究修订稿为当前入口，未投稿。后续修改继续使用本地论文写作 skills。

- [当前 PDF（28页）](submission_20261104_joint_exploration/main.pdf)
- [可编辑主文](ARTICLE_SUBMISSION_EN_SKILL_REVISED_20261029.md)
- [LaTeX 源码](submission_20261104_joint_exploration/main.tex)
- [当前包与构建验证](submission_20261104_joint_exploration/validation.json)
- [毕设未知探索增补章节（6页）](unknown_exploration_revision_20261104/chapter.pdf)
- [GitHub 本轮成果草稿 PR](https://github.com/Jimmy7338/NSO/pull/1)
- [Skill审查与修改说明](../research/skill_review_20261029/REVIEW_SUMMARY.txt)
- [24项自审与主张—证据映射](../research/skill_review_20261029/FINAL_REVIEW.json)
- [检查记录](../research/skill_review_20261029/VALIDATION.json)
- [补实验方案](../research/REVISION_AND_EXPERIMENT_PLAN_20261029.txt)

本轮保留原受控机制实验和表 7 的实车完整回合及续接失败。第 6.2 节新增 20 条确定性仿真：原始 v10.3.1 的 8 条 G/S、部署策略核心的 8 条 G/S，以及同候选生成器的 4 条覆盖评分 C 消融。表 8 保留全部数值；图 10 和图 11 显示平局、原版负结果、部署变体正结果及过程中的领先方变化。共 11 图、8 表；本地两遍编译为 28 页，无未定义引用或溢出版面，新增表图已检查渲染结果。

当前 PDF 为期刊中立审阅稿。原 V35 受控双模板机制、v10.3.1 仿真和实车 exploration_joint_v1 变体是不同的证据范围，不能合并宣称方法等价。新矩阵使用同一 Q0 的两个相关布置，物理适配器关闭；C 不是标准 frontier 基线。语义带来的实机收益、跨场景覆盖效率和有独立参照的真实重建精度仍待验证。

毕设主线仍是“基于语义增强与结构感知的未知大场景主动覆盖探索方法”；受控物体机制实验不能替代未知探索评价。新增章节为可阅读工作稿，全文整合及学校格式尚未完成。此前的 25 页实车修订稿、2026-10-29 和 2026-09-29 稿件及原始记录均保留。
