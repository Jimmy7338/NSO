# CELL 保存轨迹中的 G / NBV 前瞻规划机制

这是在端点差异已知后开展的两条已复核轨迹诊断，不是完整矩阵分析或独立测试集证据。两条运行均保持 ExposureV3 的 65 文件冻结源码、共同地面/曝光前端、传感预算和统一评价；保存配置仅 article_method 不同。G/NBV 都是几何方法，全程没有启用类别或同类共享，因此本报告不满足 S/B 语义机制门。

实际 RGB-D 与扫描数组在帧 0–19 完全相同，仅忽略运行命名的 frame_id。第 17 步，两者共有的 32 个直接观察候选及分数完全相同，后验也相同；G 另有 7 个付费诊断候选。G 选中 n_003_001:3 的 diagnose_then_observe，分数 0.082740260；NBV 选中 n_004_001:2 的 direct，分数 0.078458418。所选诊断的预测信息增量为 0.006106456，将其分数从 0.076633804 提高到高于最佳直接候选。两者均有 143 个剩余付费动作，所选首宏动作加完整返航成本分别为 28、42，预算和足迹守卫均通过。

第 17 步取消的是两者共同的旧初始化承诺，原因是已经取得平面；随后提交的宏动作 2 没有取消。两者先实际前进两步，第 20 次付费动作首次分歧：G 原地 observe，NBV turn_right。G 的宏动作 2 在帧 20 随额外付费观测完成，共 3 动作；NBV 的宏动作 2 在帧 29 完成，共 12 动作。保存逐步动作、目标、完整预算/守卫和完成记录见 macro_actions.csv 与 report.json。

这次 G 诊断观测 **没有应用新的几何反馈**，原因记录为 repeated_support_or_feedback_or_history_cap；其前后结构后验仍相同，均为均匀分布，新增已知二维格为 0。后续第 20 步实际重规划选中 n_003_002:1。故现有证据证明了“前瞻候选改变被选宏动作并改变真实路径”，没有证明本次预测的信息收益被真实反馈兑现，更不能将该次观察直接归因为最终三维质量的改善。完整深度仍按原实现融合；未测中途 F1。

|方法|付费动作|实际路程/m|转向|额外observe|C_nav|F1|J_nav|
|---|---:|---:|---:|---:|---:|---:|---:|
|G|160|24.0|52|12|0.943259|0.833812|0.786500|
|NBV|158|20.0|66|12|0.796641|0.647038|0.515457|

两条轨迹均无碰撞、完成全位姿返航并通过独立复核。G−NBV 的端点 J 差为 +0.271044，描述这一个开发场景下全部后续路径共同作用的结果。未计算 p 值，也未运行新的 World、控制器、TSDF 或质量评价。NBV 配置中继承的 future_information=true 不能用来判断实际分支：保存选择收据明确 myopic_nbv=true 且诊断候选数为 0；冻结 controller_article_v1.py 的实际分支也区分了两方法。

## English insertion

In the CELL development scene, the saved G and myopic NBV runs share exactly the same physical observations through paid frame 19. At decision 17, their 32 direct candidates have identical scores, but G selects an additional paid diagnostic macro. Its predicted score (0.082740) exceeds the best direct score (0.078458), leading to an observed action divergence at paid action 20: G observes in place while NBV turns right. Both macros complete and both episodes return safely. However, the first diagnostic observation triggers no fresh geometric feedback and leaves the uniform structure posterior unchanged. This trace therefore establishes a lookahead-to-action link, not realization of the predicted information gain. The final joint scores are 0.786500 and 0.515457; this single-scene difference includes all subsequent route changes and is not evidence for semantic sharing or an isolated effect of that observation.
