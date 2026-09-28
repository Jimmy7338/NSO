# 投稿稿实质内容审阅

审阅对象：`docs/thesis/ARTICLE_SUBMISSION_EN_20260928.md`。只读审阅；未改投稿稿、执行源码、实验数据或图。

**结论：未发现需要阻断本轮定稿的实质性错误，无需为本次审阅补跑实验或增加泛化限制。**

核对结果：

- 第 3–4 节局部指标与代码一致：覆盖计入非 unknown 的可达安全格；预测表面按共同模板包围盒加 0.20 m 裁剪并去地面；精确率对完整参考表面、召回率对固定外部竖直表面计算。模板曝光明确作为规划代理，未当成真实重建质量。实测残差、累计权重裁剪、一次未来诊断和完整姿态返航条件与旧 CPU 实现一致。
- 第 5.6 节已明确区分局部 S/G 的类别可用性对照与多实例 S/B 的额外共享对照；新指标为实际自由格覆盖乘四实例 F1 的宏平均，完整预测网格不使用局部裁剪。原有局部收益没有转述为场景级共享收益。
- 第 5.1、5.3 节将 X/Xnf 解释为同时涉及实测纠错与未来诊断的完整策略比较，没有把 2.6241% 写成单独后验更新消融。第 5.6 节和 Figure 8 将 G/NBV 的终点差归于后续完整路径策略，没有归因于单次诊断的已兑现信息。
- Table 5 的 **36 个 J 数值及 9 个原流程完成计数**已逐项程序核对 `rows36.json`，按六位小数均一致。Original/CELL/G 的 0.765328 保留 †，原流程 3/4；35 次原评价成功与 36 条完成运动的区别正确。三个版本共 9 对 S/B 的动作与质量一致，与完整配对快照一致。
- 已实际打开 Figure 8 图像，并核验图包 6 个输出哈希。AISLE 的目标分歧 27、动作分歧 32、观察完成 41/44；CELL 的选择 17、动作分歧 20、观察完成 20/29，与保存数据一致。终点差 −0.048716/+0.271044 正确；G 在 CELL 第 20 帧没有新反馈或后验更新已明确说明，离线真实几何背景也已标明。未把相同位置的转向误写成额外平移。

本审阅没有新增修改建议；建议维持当前证据范围完成排版收敛。文献的新颖性和目标期刊适配不属于本轮实现一致性审阅，不据此增加新的实验门槛。

审阅输入绑定：

- `docs/thesis/ARTICLE_SUBMISSION_EN_20260928.md`：`85dc7f7f894886d9072905b75341f07b1ed1b62d25a5051c64bd97cb73c9f1b0`。
- `audit_results/article_stage_20260928/analysis_v1/article36_final_summary_v1/manifest.json`：`468f707670baa3711f24f9d80dba254d0ed2076837d9196241e85fddbaf676fa`。
- `audit_results/article_stage_20260928/analysis_v1/exposure_cell_g_nbv_lookahead/manifest.json`：`90f9c629f1c25fd2eedcfb42bf13696852f241a827f480e5d51b9cb502705f39`。
- `docs/thesis/figures/article_mechanism_cases_20260928_final/manifest.json`：`86deb101519a8d19d51f3e006394782968ea2ae1bea14b67b29dd221e864f11e`。
- `nso/observation_belief_v35.py`：`09a6a4d9d0ac392b179a0da3cdd6a88fcb1c05e5bc3d20d393337d3a20551653`。
- `nso/online_planner_v35.py`：`d689cf16745375ca668bdde604836fb0fbb7382565675e510e34ece18411b5d6`。
- `nso/surface_measurement_v34.py`：`5d0c6b74ffdff611af4377d060a9341442c2f239ce1e9f8c7ba26a7773a6a25f`。
- `docs/thesis/ARTICLE_MULTI_INSTANCE_METHOD_APPENDIX_20260928.md`：`3292350e759df9ea6ff820f0cd30437ec3c137fc40ce2c0b66c5b01d14afb4fc`。
