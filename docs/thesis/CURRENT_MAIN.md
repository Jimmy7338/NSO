# 当前小论文与毕设入口

2026-11-04：当前文件为研究审阅稿，未投稿。后续修订继续使用本地论文写作 skills。

- [小论文 PDF（30页）](submission_20261104_frontier_comparison/main.pdf)
- [小论文可编辑主文](ARTICLE_SUBMISSION_EN_SKILL_REVISED_20261029.md)
- [LaTeX 与构建文件](submission_20261104_frontier_comparison/README.md)
- [本轮验证](submission_20261104_frontier_comparison/validation.json)与[主张—证据审查](submission_20261104_frontier_comparison/review.json)
- [毕设全文阅读稿（74页）](graduation_integrated_20261104/thesis.pdf)
- [毕设可编辑全文](graduation_integrated_20261104/thesis.md)与[构建验证](graduation_integrated_20261104/validation.json)
- [成果草稿 PR](https://github.com/Jimmy7338/NSO/pull/1)

小论文保留受控机制实验、实车完整回合与续接失败。表8包含原始v10.3.1、联合部署策略核心和覆盖评分消融的20条确定性仿真；表9新增4条本地frontier矩阵轨迹，图12比较覆盖过程。frontier矩阵在west/48短试跑后冻结，矩阵中的west/48精确复现该试跑，不计额外独立样本。全部24条矩阵轨迹仍属于一个Q0场景上下文的两个相关布置。

frontier更早达到95%覆盖；在96动作预算下，联合策略更早达到99%。唯一出现联合G/S表面差异的条件，在差异出现时已覆盖99.55%；这支持特定条件下后期表面观察的差异，不能证明语义提高了早期未知覆盖效率。frontier采用首次拒绝后永久返航的适配契约，其决策处理不同于NSO，且不是完整ROS explore_lite系统。

毕设已整合为六章全文，主线是“基于语义增强与结构感知的未知大场景主动覆盖探索方法”。181个公式均经本地渲染；学校模板尚未提供，因此当前是阅读工作稿。原V35机制、v10.3.1和exploration_joint_v1保持版本边界。实机语义收益、跨独立场景覆盖效率及具有独立参照的真实重建精度仍待验证。

此前28页探索修订稿、25页实车修订稿、6页探索增补章节及原始历史文件均保留。本轮新增内容已经检查数据、链接、编译与关键页面；这不代替作者全文审阅或期刊/学校格式验收。
